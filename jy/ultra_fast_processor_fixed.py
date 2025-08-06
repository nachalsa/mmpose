#!/usr/bin/env python3
"""
울트라 고성능 비디오 처리기 (수정된 버전)
- A6000 x2 + 112 CPU 코어 환경 최적화
- 멀티 GPU 병렬 처리 + 파이프라인 최적화
"""

import os
import cv2
import json
import h5py
import numpy as np
import time
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from tqdm import tqdm
from datetime import datetime
import multiprocessing as mp
import queue
import threading
import torch
from collections import deque

# 기존 임포트
from config import MODELS_DIR, YOLO_MODEL_CONFIG, RTMW_ONNX_MODEL_OPTIONS
from onnx_inferencer import YOLO11LRTMWONNXInferencer

def get_model_path(model_name: str) -> str:
    """모델 경로 반환"""
    models_dir = Path(MODELS_DIR)
    possible_paths = [
        models_dir / model_name,
        Path("models") / model_name,
        Path(model_name)
    ]
    
    for path in possible_paths:
        if path.exists():
            return str(path)
    
    return str(possible_paths[0])

def extract_item_info_from_path(video_path: str) -> Optional[Tuple[str, int]]:
    """비디오 경로에서 아이템 정보 추출"""
    filename = Path(video_path).stem
    
    for item_type in ['WORD', 'SEN']:
        pattern = rf'_{item_type}(\d{{4}})_'
        match = re.search(pattern, filename)
        if match:
            return item_type, int(match.group(1))
    return None

def get_video_frame_indices(video_path: str) -> List[int]:
    """비디오의 모든 프레임 인덱스 반환"""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []
        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        return list(range(frame_count))
    except:
        return []

def load_video_frames(video_path: str, frame_indices: List[int]) -> List[np.ndarray]:
    """비디오에서 특정 프레임들만 빠르게 로드"""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []
        
        frames = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if not frame_indices:
            frame_indices = list(range(total_frames))
        
        frame_indices = sorted([i for i in frame_indices if 0 <= i < total_frames])
        
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
        
        cap.release()
        return frames
        
    except Exception as e:
        print(f"비디오 로드 오류 {video_path}: {e}")
        return []

def gpu_inference_worker(gpu_id: int,
                        input_queue: mp.Queue,
                        output_queue: mp.Queue,
                        rtmw_model_name: str,
                        gpu_batch_size: int,
                        stats_queue: mp.Queue):
    """GPU별 독립적인 추론 워커"""
    
    try:
        # GPU 설정
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        torch.cuda.set_device(0)
        
        print(f"🔥 GPU {gpu_id} 워커 시작 (배치: {gpu_batch_size})")
        
        # 각 GPU에 독립적인 모델 인스턴스
        inferencer = YOLO11LRTMWONNXInferencer(
            rtmw_onnx_path=get_model_path(rtmw_model_name),
            detection_device=f'cuda:{gpu_id}',
            pose_device='cuda',
            optimize_for_accuracy=True
        )
        
        # GPU 워밍업
        dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        for _ in range(3):
            try:
                inferencer.process_frame(dummy_frame)
            except:
                pass
        
        print(f"✅ GPU {gpu_id} 워커 초기화 완료")
        
        # 배치 처리 메인 루프
        while True:
            try:
                task = input_queue.get(timeout=5)
                if task is None:  # 종료 신호
                    break
                
                video_path, frame_indices, output_key = task
                
                # 비디오 프레임 로드
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
                            batch_results.append(result)
                        except Exception as e:
                            print(f"GPU {gpu_id} 처리 오류: {e}")
                            batch_results.append(None)
                    
                    results.extend(batch_results)
                
                batch_time = time.time() - batch_start_time
                stats_queue.put(('gpu', gpu_id, len(frames), batch_time))
                
                # 결과 전송
                output_queue.put((output_key, results))
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"GPU {gpu_id} 워커 오류: {e}")
                continue
        
        print(f"🔥 GPU {gpu_id} 워커 종료")
                
    except Exception as e:
        print(f"GPU {gpu_id} 워커 초기화 실패: {e}")
        import traceback
        traceback.print_exc()

def cpu_postprocess_worker(input_queue: mp.Queue, 
                          output_dir: Path,
                          keypoint_scale: int,
                          stats_queue: mp.Queue):
    """CPU 후처리 워커"""
    
    while True:
        try:
            task = input_queue.get(timeout=5)
            if task is None:  # 종료 신호
                break
                
            output_key, results = task
            if results is None:
                continue
                
            start_time = time.time()
            
            # 후처리 수행
            process_video_results(output_key, results, output_dir, keypoint_scale)
            
            process_time = time.time() - start_time
            stats_queue.put(('cpu', 0, len(results), process_time))
            
        except queue.Empty:
            continue
        except Exception as e:
            print(f"CPU 후처리 워커 오류: {e}")
            continue
    
    print("🔧 CPU 후처리 워커 종료")

def process_video_results(output_key: str, results: List[Dict], 
                         output_dir: Path, keypoint_scale: int):
    """단일 비디오의 결과 처리 및 저장"""
    try:
        # 출력 디렉토리 생성
        item_dir = output_dir / output_key
        item_dir.mkdir(parents=True, exist_ok=True)
        
        # 결과 분리
        jpeg_frames = []
        keypoints_list = []
        scores_list = []
        
        for i, result in enumerate(results):
            if result is None:
                continue
                
            if isinstance(result, tuple) and len(result) == 2:
                vis_image, pose_results = result
                frame = vis_image
                
                if pose_results and len(pose_results) > 0:
                    keypoints = pose_results[0].get('keypoints', [])
                    scores = pose_results[0].get('keypoint_scores', [])
                else:
                    keypoints = []
                    scores = []
            else:
                frame = result.get('frame', np.zeros((480, 640, 3), dtype=np.uint8))
                keypoints = result.get('keypoints', [])
                scores = result.get('scores', [])
            
            # JPEG 인코딩
            try:
                success, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                if success:
                    jpeg_frames.append(buffer.tobytes())
                else:
                    empty_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    success, buffer = cv2.imencode('.jpg', empty_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    jpeg_frames.append(buffer.tobytes())
            except:
                empty_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                success, buffer = cv2.imencode('.jpg', empty_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                jpeg_frames.append(buffer.tobytes())
            
            # 키포인트 처리
            if keypoints:
                keypoints_array = np.array(keypoints, dtype=np.float32)
                if keypoints_array.ndim == 1:
                    keypoints_array = keypoints_array.reshape(-1, 2)
                keypoints_list.append(keypoints_array * keypoint_scale)
            else:
                keypoints_list.append(np.zeros((133, 2), dtype=np.float32))
            
            # 점수 처리
            if scores:
                scores_list.append(np.array(scores, dtype=np.float32))
            else:
                scores_list.append(np.zeros(133, dtype=np.float32))
        
        # 저장
        if jpeg_frames:
            save_processed_video(item_dir, jpeg_frames, keypoints_list, 
                               scores_list, keypoint_scale, output_key)
        
    except Exception as e:
        print(f"비디오 결과 처리 오류 {output_key}: {e}")

def save_processed_video(item_dir: Path, jpeg_frames: List, keypoints_list: List, 
                        scores_list: List, keypoint_scale: int, output_key: str):
    """처리된 비디오 데이터 저장"""
    try:
        # 메타데이터
        metadata = {
            'total_frames': len(jpeg_frames),
            'keypoint_count': len(keypoints_list[0]) if keypoints_list else 0,
            'keypoint_scale': keypoint_scale,
            'processed_at': datetime.now().isoformat(),
            'output_key': output_key
        }
        
        # 저장
        with open(item_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # JPEG 프레임 저장
        np.save(item_dir / "frames.npy", jpeg_frames)
        
        # 키포인트 저장
        if keypoints_list:
            np.save(item_dir / "keypoints.npy", keypoints_list)
        
        # 점수 저장
        if scores_list:
            np.save(item_dir / "scores.npy", scores_list)
            
    except Exception as e:
        print(f"저장 오류 {output_key}: {e}")

def video_loader_worker(video_queue: mp.Queue, frame_queue: mp.Queue, 
                       prefetch_count: int):
    """비디오 프리로딩 워커"""
    
    while True:
        try:
            item = video_queue.get(timeout=5)
            if item is None:
                break
            
            video_path = item
            
            # 아이템 정보 추출
            item_info = extract_item_info_from_path(video_path)
            if not item_info:
                continue
                
            item_type, item_id = item_info
            output_key = f"{item_type}{item_id:04d}"
            
            # 비디오 프레임 인덱스 로드
            frame_indices = get_video_frame_indices(video_path)
            if not frame_indices:
                continue
            
            # GPU 큐에 전송
            frame_queue.put((video_path, frame_indices, output_key))
                
        except queue.Empty:
            continue
        except Exception as e:
            print(f"비디오 로더 오류: {e}")
            
    print("📹 비디오 로더 워커 종료")

class UltraFastBatchProcessor:
    """초고속 배치 처리기 (A6000 x2 최적화)"""
    
    def __init__(self, 
                 data_root: str = "data/1.Training",
                 output_dir: str = "sign_language_dataset",
                 rtmw_model_name: str = "rtmw-dw-x-l_simcc-cocktail14_270e-384x288.onnx",
                 direction: str = "F",
                 item_types: List[str] = ["WORD"],
                 num_gpu_workers: int = 2,
                 gpu_batch_size: int = 512,
                 num_cpu_workers: int = 32,
                 num_video_loaders: int = 8):
        
        self.data_root = Path(data_root).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.rtmw_model_name = rtmw_model_name
        self.direction = direction.upper()
        self.item_types = [t.upper() for t in item_types]
        self.keypoint_scale = 8
        
        # 성능 설정
        self.num_gpu_workers = num_gpu_workers
        self.gpu_batch_size = gpu_batch_size
        self.num_cpu_workers = num_cpu_workers
        self.num_video_loaders = num_video_loaders
        
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
        self.processed_videos = 0
        self.processed_frames = 0
        
        self.logger.info("🚀 울트라 고속 배치 처리기 초기화")
        self.logger.info(f"   - GPU 워커: {num_gpu_workers}개")
        self.logger.info(f"   - GPU 배치: {gpu_batch_size}")
        self.logger.info(f"   - CPU 워커: {num_cpu_workers}개")
        self.logger.info(f"   - 비디오 로더: {num_video_loaders}개")

    def collect_all_videos(self) -> List[str]:
        """모든 처리할 비디오 경로 수집"""
        videos_base_dir = self.data_root / "videos"
        all_videos = []
        
        if not videos_base_dir.exists():
            self.logger.error(f"❌ videos 폴더 없음: {videos_base_dir}")
            return all_videos
        
        pattern = f"*_{self.direction}.mp4"
        for sub_dir in videos_base_dir.iterdir():
            if not sub_dir.is_dir():
                continue
            
            for video_file in sub_dir.glob(pattern):
                # 아이템 타입 확인
                filename = video_file.stem
                for item_type in self.item_types:
                    if f"_{item_type}" in filename:
                        all_videos.append(str(video_file))
                        break
        
        self.logger.info(f"📊 총 {len(all_videos)}개 비디오 발견")
        return all_videos

    def process_all_videos_ultra_fast(self):
        """초고속 병렬 처리 (메인 함수)"""
        all_videos = self.collect_all_videos()
        if not all_videos:
            self.logger.error("❌ 처리할 비디오가 없습니다")
            return
        
        self.start_time = time.time()
        self.logger.info(f"🚀 초고속 처리 시작: {len(all_videos)}개 비디오")
        
        # 큐 생성
        video_queue = mp.Queue(maxsize=100)
        frame_queue = mp.Queue(maxsize=self.num_gpu_workers * 10)
        gpu_output_queue = mp.Queue(maxsize=self.num_cpu_workers * 5)
        stats_queue = mp.Queue()
        
        # 워커 프로세스들
        workers = []
        
        # 1. 비디오 로더 워커들
        for i in range(self.num_video_loaders):
            worker = mp.Process(target=video_loader_worker, 
                              args=(video_queue, frame_queue, 100))
            worker.start()
            workers.append(worker)
        
        # 2. GPU 추론 워커들
        for gpu_id in range(self.num_gpu_workers):
            worker = mp.Process(target=gpu_inference_worker,
                              args=(gpu_id, frame_queue, gpu_output_queue,
                                   self.rtmw_model_name, self.gpu_batch_size,
                                   stats_queue))
            worker.start()
            workers.append(worker)
        
        # 3. CPU 후처리 워커들
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
                    current_count = len(list(self.video_output_dir.iterdir()))
                    if current_count > processed_count:
                        pbar.update(current_count - processed_count)
                        processed_count = current_count
                    
                    time.sleep(1)
                    
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
                continue

    def create_final_hdf5_batches(self):
        """최종 HDF5 배치 파일 생성"""
        processed_items = list(self.video_output_dir.iterdir())
        if not processed_items:
            self.logger.warning("❌ 처리된 비디오가 없습니다")
            return
        
        # 아이템별로 그룹핑
        items_by_type = {}
        for item_dir in processed_items:
            if not item_dir.is_dir():
                continue
                
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
                
                # 메타데이터
                f_frames.attrs['batch_id'] = batch_id
                f_frames.attrs['item_type'] = item_type
                f_frames.attrs['direction'] = self.direction
                f_frames.attrs['total_items'] = len(items)
                f_frames.attrs['created_at'] = datetime.now().isoformat()
                
                f_poses.attrs['batch_id'] = batch_id
                f_poses.attrs['item_type'] = item_type
                f_poses.attrs['direction'] = self.direction
                f_poses.attrs['total_items'] = len(items)
                f_poses.attrs['created_at'] = datetime.now().isoformat()
                
                for item_name in items:
                    item_dir = self.video_output_dir / item_name
                    
                    if not item_dir.exists():
                        continue
                    
                    try:
                        # 프레임 데이터 로드
                        frames_path = item_dir / "frames.npy"
                        if frames_path.exists():
                            frames_data = np.load(frames_path, allow_pickle=True)
                            f_frames.create_dataset(item_name, data=frames_data, 
                                                  dtype=jpeg_vlen_dtype, compression='lzf')
                        
                        # 포즈 데이터 로드
                        keypoints_path = item_dir / "keypoints.npy"
                        scores_path = item_dir / "scores.npy"
                        
                        if keypoints_path.exists() and scores_path.exists():
                            keypoints = np.load(keypoints_path, allow_pickle=True)
                            scores = np.load(scores_path, allow_pickle=True)
                            
                            pose_group = f_poses.create_group(item_name)
                            pose_group.create_dataset('keypoints', data=keypoints, 
                                                    compression='lzf')
                            pose_group.create_dataset('scores', data=scores, 
                                                    compression='lzf')
                                                    
                    except Exception as e:
                        self.logger.error(f"아이템 {item_name} 처리 오류: {e}")
                        continue
                        
            self.logger.info(f"✅ 배치 {batch_id} 생성: {len(items)}개 아이템")
            
        except Exception as e:
            self.logger.error(f"HDF5 배치 {batch_id} 생성 실패: {e}")

def main():
    """메인 함수"""
    mp.set_start_method('spawn', force=True)
    
    print("🚀 울트라 고속 배치 처리기 (A6000 x2 최적화)")
    print("="*80)
    
    # 사용자 입력
    print("\n처리할 아이템 타입:")
    print("1. WORD만 처리")
    print("2. SEN만 처리") 
    print("3. WORD + SEN 모두 처리 (기본값)")
    
    type_choice = input("선택 (1-3, 기본값: 3): ").strip()
    item_types = {'1': ['WORD'], '2': ['SEN']}.get(type_choice, ['WORD', 'SEN'])
    print(f"✅ 선택: {', '.join(item_types)}")
    
    # 방향 선택
    direction_choice = input("\n방향 (F/U/L/R/D, 기본값: F): ").strip().upper()
    direction = direction_choice if direction_choice in ['F','U','L','R','D'] else 'F'
    print(f"✅ 방향: {direction}")
    
    # 성능 설정
    print(f"\n🔧 성능 설정:")
    gpu_workers = int(input("GPU 워커 수 (기본값: 2): ") or "2")
    gpu_batch = int(input("GPU 배치 크기 (기본값: 256): ") or "256")
    cpu_workers = int(input("CPU 워커 수 (기본값: 32): ") or "32")
    video_loaders = int(input("비디오 로더 수 (기본값: 8): ") or "8")
    
    print(f"✅ 설정: GPU워커={gpu_workers}, 배치={gpu_batch}, CPU워커={cpu_workers}")
    
    # 처리기 초기화 및 실행
    processor = UltraFastBatchProcessor(
        item_types=item_types,
        direction=direction,
        num_gpu_workers=gpu_workers,
        gpu_batch_size=gpu_batch,
        num_cpu_workers=cpu_workers,
        num_video_loaders=video_loaders
    )
    
    # 실행
    processor.process_all_videos_ultra_fast()
    print("🎉 모든 처리 완료!")

if __name__ == "__main__":
    main()
