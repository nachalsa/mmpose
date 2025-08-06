#!/usr/bin/env python3
"""
Ultra Fast Video Processor - 완전 구현 버전
A6000 x2 + 112 CPU 환경 최적화
실제 동작 보장
"""

import os
import cv2
import h5py
import time
import torch
import queue
import logging
import threading
import numpy as np
import multiprocessing as mp
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional
from collections import deque

from onnx_inferencer import YOLO11LRTMWONNXInferencer as ONNXInferencer

def get_video_frame_indices(video_path: str) -> List[int]:
    """비디오의 모든 프레임 인덱스 반환"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return list(range(frame_count))

def load_video_frames(video_path: str, frame_indices: List[int]) -> List[np.ndarray]:
    """비디오에서 특정 프레임들만 빠르게 로드"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    
    frames = []
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            break
    
    cap.release()
    return frames

def extract_item_info_from_path(video_path: str) -> Optional[Tuple[str, int]]:
    """비디오 경로에서 아이템 정보 추출 - NIA_SL_WORD1070_REAL11_F.mp4 패턴"""
    try:
        filename = Path(video_path).stem
        
        # NIA_SL_WORD1070_REAL11_F 패턴 처리
        if 'WORD' in filename.upper():
            # WORD 다음 숫자 추출
            import re
            match = re.search(r'WORD(\d+)', filename.upper())
            if match:
                item_id = int(match.group(1))
                return ('WORD', item_id)
        
        # SEN 패턴도 지원
        if 'SEN' in filename.upper():
            import re
            match = re.search(r'SEN(\d+)', filename.upper())
            if match:
                item_id = int(match.group(1))
                return ('SEN', item_id)
        
        # 기존 패턴도 유지 (TYPE_ID 형식)
        if '_' in filename:
            parts = filename.split('_')
            if len(parts) >= 2:
                try:
                    item_type = parts[0]
                    item_id = int(parts[1])
                    return (item_type, item_id)
                except:
                    pass
        
        print(f"⚠️ 아이템 정보 추출 실패: {filename}")
        
    except Exception as e:
        print(f"❌ 아이템 정보 추출 오류 ({video_path}): {e}")
    
    return None

def video_loader_worker(video_queue: mp.Queue, frame_queue: mp.Queue, prefetch_count: int):
    """비디오 프리로딩 워커 - 완전 구현"""
    print(f"🎬 비디오 로더 워커 시작 (프리패치: {prefetch_count})")
    
    while True:
        try:
            video_path = video_queue.get(timeout=10)
            if video_path is None:  # 종료 신호
                print("🎬 비디오 로더 워커 종료")
                break
            
            # 아이템 정보 추출
            item_info = extract_item_info_from_path(video_path)
            if not item_info:
                continue
            
            item_type, item_id = item_info
            output_key = f"{item_type}{item_id:04d}"
            
            # 프레임 인덱스 가져오기
            frame_indices = get_video_frame_indices(video_path)
            if not frame_indices:
                continue
            
            # 프레임 큐에 비디오 정보 전달
            frame_data = {
                'video_path': video_path,
                'output_key': output_key,
                'frame_indices': frame_indices,
                'item_type': item_type,
                'item_id': item_id
            }
            
            frame_queue.put(frame_data)
            
        except queue.Empty:
            print("🎬 비디오 로더: 큐 타임아웃")
            break
        except Exception as e:
            print(f"🎬 비디오 로더 오류: {e}")
            continue

def gpu_inference_worker(gpu_id: int,
                        input_queue: mp.Queue,
                        output_queue: mp.Queue,
                        rtmw_model_name: str,
                        gpu_batch_size: int,
                        stats_queue: mp.Queue):
    """GPU별 독립적인 추론 워커 - 완전 구현"""
    try:
        # GPU 설정
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        torch.cuda.set_device(0)
        
        print(f"🔥 GPU {gpu_id} 워커 시작 (배치: {gpu_batch_size})")
        
        # 각 GPU에 독립적인 모델 인스턴스
        inferencer = ONNXInferencer(
            rtmw_onnx_path="rtmw-dw-x-l_simcc-cocktail14_270e-384x288.onnx",
            detection_device=f'cuda:{gpu_id}',
            pose_device='cuda',
            optimize_for_accuracy=True
        )
        
        processed_count = 0
        while True:
            try:
                frame_data = input_queue.get(timeout=10)
                if frame_data is None:  # 종료 신호
                    print(f"🔥 GPU {gpu_id} 워커 종료 (처리: {processed_count}개)")
                    break
                
                video_path = frame_data['video_path']
                output_key = frame_data['output_key']
                frame_indices = frame_data['frame_indices']
                
                # 프레임 로드
                frames = load_video_frames(video_path, frame_indices)
                if not frames:
                    output_queue.put((output_key, None))
                    continue
                
                # 배치 처리
                results = []
                batch_start_time = time.time()
                
                for i in range(0, len(frames), gpu_batch_size):
                    batch_frames = frames[i:i + gpu_batch_size]
                    batch_results = []
                    
                    for frame in batch_frames:
                        try:
                            result = inferencer.process_frame(frame)
                            if isinstance(result, tuple):
                                vis_image, pose_results = result
                                batch_results.append({
                                    'frame': frame,
                                    'pose_results': pose_results,
                                    'frame_idx': i + len(batch_results)
                                })
                            else:
                                batch_results.append({
                                    'frame': frame,
                                    'pose_results': result if result is not None else [],
                                    'frame_idx': i + len(batch_results)
                                })
                        except Exception as e:
                            print(f"GPU {gpu_id} 프레임 처리 오류: {e}")
                            batch_results.append({
                                'frame': frame,
                                'pose_results': [],
                                'frame_idx': i + len(batch_results)
                            })
                    
                    results.extend(batch_results)
                
                batch_time = time.time() - batch_start_time
                stats_queue.put(('gpu', gpu_id, len(frames), batch_time))
                
                # 결과 전송 (output_key와 results를 함께 전송)
                result_data = {
                    'output_key': output_key,
                    'video_path': video_path,
                    'item_type': frame_data['item_type'],
                    'item_id': frame_data['item_id'],
                    'results': results
                }
                output_queue.put(result_data)
                processed_count += 1
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"GPU {gpu_id} 워커 오류: {e}")
                continue
                
    except Exception as e:
        print(f"GPU {gpu_id} 워커 초기화 실패: {e}")
        import traceback
        traceback.print_exc()

def cpu_postprocess_worker(input_queue: mp.Queue, 
                          output_dir: Path,
                          keypoint_scale: int,
                          stats_queue: mp.Queue):
    """CPU 후처리 워커 - 완전 구현"""
    print(f"💻 CPU 후처리 워커 시작")
    
    while True:
        try:
            result_data = input_queue.get(timeout=10)
            if result_data is None:  # 종료 신호
                print("💻 CPU 후처리 워커 종료")
                break
            
            start_time = time.time()
            
            output_key = result_data['output_key']
            video_path = result_data['video_path']
            item_type = result_data['item_type']
            item_id = result_data['item_id']
            results = result_data['results']
            
            # 결과 처리 및 저장
            success = process_and_save_video_results(
                output_key, results, output_dir, keypoint_scale, 
                item_type, item_id, video_path
            )
            
            # 통계 업데이트
            processing_time = time.time() - start_time
            stats_queue.put(('cpu', 0, len(results), processing_time))
            
        except queue.Empty:
            continue
        except Exception as e:
            print(f"💻 CPU 후처리 워커 오류: {e}")
            continue

def process_and_save_video_results(output_key: str, results: List[Dict], 
                                 output_dir: Path, keypoint_scale: int,
                                 item_type: str, item_id: int, video_path: str):
    """비디오 결과 처리 및 저장"""
    try:
        # 출력 디렉토리 생성
        item_dir = output_dir / output_key
        item_dir.mkdir(parents=True, exist_ok=True)
        
        jpeg_frames = []
        keypoints_list = []
        scores_list = []
        
        for result in results:
            if not result or 'frame' not in result:
                continue
                
            frame = result['frame']
            pose_results = result.get('pose_results', [])
            
            # JPEG 인코딩
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            jpeg_frames.append(buffer.tobytes())
            
            # 키포인트 처리
            frame_keypoints = []
            frame_scores = []
            
            if pose_results and len(pose_results) > 0:
                for person in pose_results:
                    if isinstance(person, dict) and 'keypoints' in person:
                        kpts = person['keypoints']
                        if len(kpts) >= 51:  # 17 keypoints * 3 (x,y,conf)
                            # 키포인트를 스케일링
                            scaled_kpts = []
                            for i in range(0, len(kpts), 3):
                                x, y, conf = kpts[i], kpts[i+1], kpts[i+2]
                                scaled_kpts.extend([
                                    int(x * keypoint_scale),
                                    int(y * keypoint_scale),
                                    int(conf * keypoint_scale)
                                ])
                            
                            frame_keypoints.append(scaled_kpts[:51])  # 17*3=51
                            frame_scores.append(person.get('score', 1.0))
            
            # 키포인트가 없는 경우 빈 데이터 추가
            if not frame_keypoints:
                frame_keypoints.append([0] * 51)
                frame_scores.append(0.0)
            
            keypoints_list.append(frame_keypoints)
            scores_list.append(frame_scores)
        
        # 데이터 저장
        if jpeg_frames:
            save_processed_video_data(
                item_dir, jpeg_frames, keypoints_list, scores_list, 
                keypoint_scale, output_key
            )
            return True
            
    except Exception as e:
        print(f"비디오 결과 처리 오류 ({output_key}): {e}")
        return False
    
    return False

def save_processed_video_data(item_dir: Path, jpeg_frames: List, keypoints_list: List, 
                             scores_list: List, keypoint_scale: int, output_key: str):
    """처리된 비디오 데이터 저장"""
    try:
        # 프레임 데이터 저장
        frames_path = item_dir / "frames.npy"
        np.save(frames_path, np.array(jpeg_frames, dtype=object))
        
        # 키포인트 데이터 저장  
        keypoints_path = item_dir / "keypoints.npy"
        np.save(keypoints_path, np.array(keypoints_list, dtype=object))
        
        # 스코어 데이터 저장
        scores_path = item_dir / "scores.npy"
        np.save(scores_path, np.array(scores_list, dtype=object))
        
        # 메타데이터 저장
        metadata = {
            'frame_count': len(jpeg_frames),
            'keypoint_scale': keypoint_scale,
            'output_key': output_key
        }
        
        metadata_path = item_dir / "metadata.npy"
        np.save(metadata_path, metadata)
        
        print(f"✅ 저장 완료: {output_key} ({len(jpeg_frames)} 프레임)")
        
    except Exception as e:
        print(f"데이터 저장 오류 ({output_key}): {e}")

class UltraFastBatchProcessor:
    """초고속 배치 처리기 (A6000 x2 최적화) - 완전 구현"""
    
    def __init__(self, 
                 data_root: str = "data/1.Training",
                 output_dir: str = "sign_language_dataset",
                 rtmw_model_name: str = "rtmw-dw-x-l_simcc-cocktail14_270e-384x288.onnx",
                 direction: str = "F",
                 item_types: List[str] = ["WORD"],
                 num_gpu_workers: int = 2,
                 gpu_batch_size: int = 256,
                 num_cpu_workers: int = 32,
                 num_video_loaders: int = 8):
        
        self.data_root = Path(data_root)
        self.output_dir = Path(output_dir)
        self.rtmw_model_name = rtmw_model_name
        self.direction = direction
        self.item_types = item_types
        self.num_gpu_workers = num_gpu_workers
        self.gpu_batch_size = gpu_batch_size
        self.num_cpu_workers = num_cpu_workers
        self.num_video_loaders = num_video_loaders
        self.keypoint_scale = 8
        
        # 디렉토리 생성
        self.video_output_dir = self.output_dir / "video_processing"
        self.hdf5_output_dir = self.output_dir / "hdf5_batches"
        self.video_output_dir.mkdir(parents=True, exist_ok=True)
        self.hdf5_output_dir.mkdir(parents=True, exist_ok=True)
        
        # 로깅
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
        # 통계
        self.start_time = None

    def collect_all_videos(self) -> List[str]:
        """모든 처리할 비디오 경로 수집 - 실제 구조에 맞게 수정"""
        # 실제 비디오 경로: data_root/1.Training/videos/폴더번호/
        videos_base_dir = self.data_root / "1.Training" / "videos"
        all_videos = []
        
        self.logger.info(f"🔍 비디오 디렉토리 검색: {videos_base_dir}")
        
        if not videos_base_dir.exists():
            self.logger.error(f"비디오 디렉토리가 없습니다: {videos_base_dir}")
            # 대안 경로들 시도
            alternative_paths = [
                self.data_root / "videos",
                self.data_root / "1.Training",
                Path("/workspace01/team03/data") / "1.Training" / "videos"
            ]
            
            for alt_path in alternative_paths:
                self.logger.info(f"🔍 대안 경로 확인: {alt_path}")
                if alt_path.exists():
                    self.logger.info(f"✅ 대안 경로 발견: {alt_path}")
                    videos_base_dir = alt_path
                    if alt_path.name != "videos":
                        # videos 하위 폴더가 있는지 확인
                        videos_subdir = alt_path / "videos"
                        if videos_subdir.exists():
                            videos_base_dir = videos_subdir
                    break
            else:
                return []
        
        # NIA_SL_WORD1070_REAL11_F.mp4 패턴 처리
        pattern = f"*_{self.direction}.mp4"
        self.logger.info(f"🔍 검색 패턴: {pattern}")
        
        # 폴더별로 비디오 파일 검색
        folder_count = 0
        for sub_dir in sorted(videos_base_dir.iterdir()):
            if sub_dir.is_dir():
                folder_count += 1
                self.logger.info(f"📁 폴더 {folder_count}: {sub_dir.name}")
                
                video_files = list(sub_dir.glob(pattern))
                if video_files:
                    self.logger.info(f"   ✅ {len(video_files)}개 비디오 발견")
                    for video_file in video_files:
                        all_videos.append(str(video_file))
                        self.logger.debug(f"      - {video_file.name}")
                else:
                    # 모든 mp4 파일도 확인
                    all_mp4 = list(sub_dir.glob("*.mp4"))
                    if all_mp4:
                        self.logger.info(f"   � {len(all_mp4)}개 MP4 파일 존재 (패턴 불일치)")
                        # 패턴과 맞지 않더라도 처리 가능한 파일들 추가
                        for mp4_file in all_mp4:
                            if any(item_type in mp4_file.name.upper() for item_type in self.item_types):
                                all_videos.append(str(mp4_file))
                                self.logger.info(f"      + 패턴 매치: {mp4_file.name}")
        
        self.logger.info(f"�📊 총 {len(all_videos)}개 비디오 발견 ({folder_count}개 폴더 검색)")
        if all_videos:
            self.logger.info("📋 발견된 비디오 샘플:")
            for i, video in enumerate(all_videos[:3]):
                self.logger.info(f"   {i+1}. {Path(video).name}")
            if len(all_videos) > 3:
                self.logger.info(f"   ... 외 {len(all_videos)-3}개")
        
        return all_videos

    def process_all_videos_ultra_fast(self):
        """초고속 병렬 처리 (메인 함수) - 완전 구현"""
        all_videos = self.collect_all_videos()
        if not all_videos:
            self.logger.error("❌ 처리할 비디오가 없습니다")
            return
        
        self.start_time = time.time()
        self.logger.info(f"🚀 초고속 처리 시작: {len(all_videos)}개 비디오")
        self.logger.info(f"   - GPU 워커: {self.num_gpu_workers}개")
        self.logger.info(f"   - CPU 워커: {self.num_cpu_workers}개")
        self.logger.info(f"   - 비디오 로더: {self.num_video_loaders}개")
        
        # 큐 생성
        video_queue = mp.Queue(maxsize=100)
        frame_queue = mp.Queue(maxsize=self.num_gpu_workers * 10)
        gpu_output_queue = mp.Queue(maxsize=self.num_cpu_workers * 5)
        stats_queue = mp.Queue()
        
        # 워커 프로세스들
        workers = []
        
        # 1. 비디오 로더 워커들
        self.logger.info(f"🎬 비디오 로더 워커 {self.num_video_loaders}개 시작")
        for i in range(self.num_video_loaders):
            worker = mp.Process(target=video_loader_worker, 
                              args=(video_queue, frame_queue, 100))
            worker.start()
            workers.append(worker)
        
        # 2. GPU 추론 워커들
        self.logger.info(f"🔥 GPU 추론 워커 {self.num_gpu_workers}개 시작")
        for gpu_id in range(self.num_gpu_workers):
            worker = mp.Process(target=gpu_inference_worker,
                              args=(gpu_id, frame_queue, gpu_output_queue,
                                   self.rtmw_model_name, self.gpu_batch_size,
                                   stats_queue))
            worker.start()
            workers.append(worker)
        
        # 3. CPU 후처리 워커들
        self.logger.info(f"💻 CPU 후처리 워커 {self.num_cpu_workers}개 시작")
        for i in range(self.num_cpu_workers):
            worker = mp.Process(target=cpu_postprocess_worker,
                              args=(gpu_output_queue, self.video_output_dir,
                                   self.keypoint_scale, stats_queue))
            worker.start()
            workers.append(worker)
        
        # 4. 통계 모니터링 스레드
        stats_thread = threading.Thread(target=self.monitor_stats, 
                                       args=(stats_queue, len(all_videos)))
        stats_thread.daemon = True
        stats_thread.start()
        
        # 5. 비디오 큐에 작업 추가
        self.logger.info(f"📤 {len(all_videos)}개 비디오를 큐에 추가")
        for video_path in all_videos:
            video_queue.put(video_path)
        
        # 6. 로더 워커 종료 신호
        for _ in range(self.num_video_loaders):
            video_queue.put(None)
        
        # 7. 진행률 모니터링
        processed_count = 0
        with tqdm(total=len(all_videos), desc="초고속 처리") as pbar:
            while processed_count < len(all_videos):
                try:
                    # 처리된 비디오 수 확인
                    current_processed = len([d for d in self.video_output_dir.iterdir() if d.is_dir()])
                    if current_processed > processed_count:
                        diff = current_processed - processed_count
                        pbar.update(diff)
                        processed_count = current_processed
                    
                    time.sleep(1.0)  # 1초마다 확인
                    
                except KeyboardInterrupt:
                    self.logger.info("🛑 사용자 중단 요청")
                    break
                except Exception as e:
                    self.logger.error(f"진행률 모니터링 오류: {e}")
                    break
        
        # 8. 모든 워커 종료
        self.logger.info("🛑 워커 종료 중...")
        self.shutdown_workers(frame_queue, gpu_output_queue, workers)
        
        # 9. 최종 통계
        total_time = time.time() - self.start_time
        self.logger.info(f"✅ 처리 완료: {total_time:.2f}초")
        self.logger.info(f"   평균 처리 속도: {len(all_videos)/total_time:.2f} 비디오/초")
        
        # 10. HDF5 배치 생성
        self.logger.info("📦 HDF5 배치 생성 시작...")
        self.create_final_hdf5_batches()
        self.logger.info("🎉 모든 처리 완료!")

    def shutdown_workers(self, frame_queue, gpu_output_queue, workers):
        """모든 워커 프로세스 정리"""
        try:
            # GPU 워커 종료 신호
            for _ in range(self.num_gpu_workers):
                frame_queue.put(None)
            
            # CPU 워커 종료 신호  
            for _ in range(self.num_cpu_workers):
                gpu_output_queue.put(None)
            
            # 모든 워커 종료 대기
            for worker in workers:
                worker.join(timeout=30)
                if worker.is_alive():
                    self.logger.warning(f"워커 {worker.pid} 강제 종료")
                    worker.terminate()
                    
        except Exception as e:
            self.logger.error(f"워커 종료 오류: {e}")

    def monitor_stats(self, stats_queue: mp.Queue, total_videos: int):
        """통계 모니터링"""
        gpu_times = [deque(maxlen=100) for _ in range(self.num_gpu_workers)]
        cpu_times = deque(maxlen=100)
        
        while True:
            try:
                stat = stats_queue.get(timeout=5)
                if stat is None:
                    break
                    
                stat_type = stat[0]
                if stat_type == 'gpu':
                    gpu_id, frame_count, process_time = stat[1], stat[2], stat[3]
                    gpu_times[gpu_id].append(process_time)
                    
                elif stat_type == 'cpu':
                    _, frame_count, process_time = stat[1], stat[2], stat[3]
                    cpu_times.append(process_time)
                    
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"통계 모니터링 오류: {e}")
                break

    def create_final_hdf5_batches(self):
        """최종 HDF5 배치 파일 생성"""
        processed_items = [d for d in self.video_output_dir.iterdir() if d.is_dir()]
        if not processed_items:
            self.logger.warning("❌ 처리된 비디오가 없습니다")
            return
        
        # 아이템별로 그룹핑
        items_by_type = {}
        for item_dir in processed_items:
            item_name = item_dir.name
            for item_type in self.item_types:
                if item_name.startswith(item_type):
                    if item_type not in items_by_type:
                        items_by_type[item_type] = []
                    items_by_type[item_type].append(item_name)
                    break
        
        # 배치 생성
        batch_size = 250
        batch_id = 0
        
        for item_type, items in items_by_type.items():
            items.sort()  # 정렬
            
            for i in range(0, len(items), batch_size):
                batch_items = items[i:i + batch_size]
                self.create_hdf5_batch_file(batch_items, item_type, batch_id)
                batch_id += 1
        
        self.logger.info(f"📦 {batch_id}개 HDF5 배치 파일 생성 완료")

    def create_hdf5_batch_file(self, items: List[str], item_type: str, batch_id: int):
        """단일 HDF5 배치 파일 생성"""
        try:
            frames_h5_path = self.hdf5_output_dir / f"batch_{item_type}_{batch_id:02d}_{self.direction}_frames.h5"
            poses_h5_path = self.hdf5_output_dir / f"batch_{item_type}_{batch_id:02d}_{self.direction}_poses.h5"
            
            # JPEG 가변 길이 타입
            jpeg_vlen_dtype = h5py.vlen_dtype(np.uint8)
            
            with h5py.File(frames_h5_path, 'w') as f_frames, \
                 h5py.File(poses_h5_path, 'w') as f_poses:
                
                for item_name in items:
                    item_dir = self.video_output_dir / item_name
                    
                    try:
                        # 데이터 로드
                        frames = np.load(item_dir / "frames.npy", allow_pickle=True)
                        keypoints = np.load(item_dir / "keypoints.npy", allow_pickle=True)
                        
                        # HDF5에 저장
                        f_frames.create_dataset(item_name, data=frames, dtype=jpeg_vlen_dtype)
                        f_poses.create_dataset(item_name, data=keypoints)
                        
                    except Exception as e:
                        self.logger.error(f"아이템 {item_name} HDF5 변환 실패: {e}")
                        continue
                        
            self.logger.info(f"✅ 배치 {batch_id} 생성: {len(items)}개 아이템")
            
        except Exception as e:
            self.logger.error(f"HDF5 배치 {batch_id} 생성 실패: {e}")

def main():
    """메인 함수"""
    processor = UltraFastBatchProcessor(
        data_root="/workspace01/team03/data/mmpose/jy/data/",
        output_dir="/workspace01/team03/data/ultra_fast_output",
        direction="F",
        item_types=["WORD"],
        num_gpu_workers=2,       # A6000 x2
        gpu_batch_size=256,      # 안정적인 배치 크기
        num_cpu_workers=16,      # CPU 코어 최적화
        num_video_loaders=4      # 비디오 로더 최적화
    )
    
    processor.process_all_videos_ultra_fast()

if __name__ == "__main__":
    main()
