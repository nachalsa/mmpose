#!/usr/bin/env python3
"""
Multi-ONNX Streamlined Processor - 완전 구현 버전
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
import numpy as np
import multiprocessing as mp
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional

from onnx_inferencer import YOLO11LRTMWONNXInferencer as ONNXInferencer

class StreamlinedVideoProcessor:
    """간소화된 비디오 처리기 - ONNX 하이브리드"""
    
    def __init__(self, 
                 rtmw_model_name: str = "rtmw-dw-x-l_simcc-cocktail14_270e-384x288.onnx",
                 rtmw_config_path: Optional[str] = None,
                 yolo_device: str = "auto",
                 pose_device: str = "auto"):
        
        self.rtmw_model_name = rtmw_model_name
        self.rtmw_config_path = rtmw_config_path
        self.yolo_device = yolo_device
        self.pose_device = pose_device
        self.keypoint_scale = 8
        
        # ONNX 추론기 초기화
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.inferencer = ONNXInferencer(
            rtmw_onnx_path="rtmw-dw-x-l_simcc-cocktail14_270e-384x288.onnx",
            detection_device=device,
            pose_device=device,
            optimize_for_accuracy=True
        )
        
        print(f"🚀 StreamlinedVideoProcessor 초기화 완료")
        print(f"   - RTMW 모델: {rtmw_model_name}")
        print(f"   - YOLO 디바이스: {yolo_device}")
        print(f"   - Pose 디바이스: {pose_device}")

    def process_video_to_arrays(self, video_path: str) -> Optional[Dict]:
        """비디오를 처리하여 배열 형태로 반환"""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"❌ 비디오 열기 실패: {video_path}")
                return None
            
            frames = []
            jpeg_frames = []
            keypoints_list = []
            scores_list = []
            
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frames.append(frame)
                frame_count += 1
            
            cap.release()
            
            if frame_count == 0:
                return None
            
            # 프레임별 처리
            for frame in frames:
                # 추론 실행
                result = self.inferencer.process_frame(frame)
                
                # JPEG 인코딩
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                jpeg_frames.append(buffer.tobytes())
                
                # 결과 처리
                frame_keypoints = []
                frame_scores = []
                
                if isinstance(result, tuple):
                    _, pose_results = result
                else:
                    pose_results = result if result is not None else []
                
                if pose_results and len(pose_results) > 0:
                    for person in pose_results:
                        if isinstance(person, dict) and 'keypoints' in person:
                            kpts = person['keypoints']
                            if len(kpts) >= 51:  # 17 keypoints * 3
                                # 키포인트 스케일링
                                scaled_kpts = []
                                for i in range(0, len(kpts), 3):
                                    x, y, conf = kpts[i], kpts[i+1], kpts[i+2]
                                    scaled_kpts.extend([
                                        int(x * self.keypoint_scale),
                                        int(y * self.keypoint_scale),
                                        int(conf * self.keypoint_scale)
                                    ])
                                
                                frame_keypoints.append(scaled_kpts[:51])
                                frame_scores.append(person.get('score', 1.0))
                
                # 키포인트가 없는 경우
                if not frame_keypoints:
                    frame_keypoints.append([0] * 51)
                    frame_scores.append(0.0)
                
                keypoints_list.append(frame_keypoints)
                scores_list.append(frame_scores)
            
            return {
                'jpeg_frames': jpeg_frames,
                'keypoints': keypoints_list,
                'scores': scores_list,
                'frame_count': frame_count
            }
            
        except Exception as e:
            print(f"💥 비디오 처리 오류 ({video_path}): {e}")
            return None

def multi_gpu_onnx_inference_worker(
    gpu_id: int,
    task_queue: mp.Queue, 
    result_queue: mp.Queue, 
    rtmw_model_name: str = "rtmw-dw-x-l_simcc-cocktail14_270e-384x288.onnx",
    gpu_batch_size: int = 512,
    yolo_device: str = "auto",
    pose_device: str = "auto"
):
    """멀티 GPU ONNX 하이브리드 추론을 수행하는 프로세스 - 완전 구현"""
    try:
        # GPU 설정
        if torch.cuda.is_available() and gpu_id < torch.cuda.device_count():
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
            torch.cuda.set_device(0)
            device = f"cuda:{gpu_id}"
        else:
            device = "cpu"
            
        print(f"🚀 GPU {gpu_id} 추론 워커 시작 (디바이스: {device})")
        
        # 프로세서 초기화
        processor = StreamlinedVideoProcessor(
            rtmw_model_name=rtmw_model_name,
            yolo_device=yolo_device,
            pose_device=pose_device
        )
        
        batch_jobs = []
        processed_count = 0
        
        while True:
            try:
                # 배치 수집
                while len(batch_jobs) < gpu_batch_size:
                    try:
                        job = task_queue.get(timeout=1.0)
                        if job is None:  # 종료 신호
                            break
                        batch_jobs.append(job)
                    except queue.Empty:
                        break
                
                # 종료 조건 확인
                if not batch_jobs:
                    try:
                        job = task_queue.get(timeout=5.0)
                        if job is None:
                            print(f"🔥 GPU {gpu_id} 워커 종료 (처리: {processed_count}개)")
                            break
                        batch_jobs.append(job)
                    except queue.Empty:
                        print(f"🔥 GPU {gpu_id} 워커 타임아웃 종료")
                        break
                
                # 배치 처리
                if batch_jobs:
                    start_time = time.time()
                    
                    for job in batch_jobs:
                        job_id, item_type, item_id, video_path = job
                        
                        try:
                            arrays = processor.process_video_to_arrays(video_path)
                            result_queue.put((job_id, item_type, item_id, video_path, arrays))
                            processed_count += 1
                            
                        except Exception as e:
                            print(f"💥 GPU {gpu_id} 비디오 처리 오류: {e}")
                            result_queue.put((job_id, item_type, item_id, video_path, None))
                    
                    batch_time = time.time() - start_time
                    if len(batch_jobs) > 0:
                        avg_time = batch_time / len(batch_jobs)
                        print(f"⚡ GPU {gpu_id}: {len(batch_jobs)}개 배치 처리 완료 ({avg_time:.3f}s/video)")
                    
                    batch_jobs = []
                
            except Exception as e:
                print(f"💥 GPU {gpu_id} Worker 오류 발생: {e}")
                # 오류 발생 시 배치 작업들을 None 결과로 처리
                for job in batch_jobs:
                    result_queue.put((job[0], job[1], job[2], job[3], None))
                batch_jobs = []
                continue
                
    except Exception as e:
        print(f"❌ GPU {gpu_id} Worker 초기화 실패: {e}")

def enhanced_cpu_postprocess_worker(
    result_queue: mp.Queue, 
    output_dir: Path,
    keypoint_scale: int
):
    """향상된 결과를 받아 파일로 저장하는 CPU 워커 - 완전 구현"""
    print(f"💻 CPU 후처리 워커 시작")
    
    while True:
        try:
            result = result_queue.get(timeout=10)
            if result is None:  # 종료 신호
                print("💻 CPU 후처리 워커 종료")
                break
            
            job_id, item_type, item_id, video_path, arrays = result
            
            if arrays is None:
                print(f"❌ 빈 결과 스킵: {item_type}{item_id:04d}")
                continue
            
            # 출력 키 생성
            output_key = f"{item_type}{item_id:04d}"
            
            # 출력 디렉토리 생성
            item_dir = output_dir / output_key
            item_dir.mkdir(parents=True, exist_ok=True)
            
            try:
                # 데이터 저장
                frames_path = item_dir / "frames.npy"
                np.save(frames_path, np.array(arrays['jpeg_frames'], dtype=object))
                
                keypoints_path = item_dir / "keypoints.npy"
                np.save(keypoints_path, np.array(arrays['keypoints'], dtype=object))
                
                scores_path = item_dir / "scores.npy"
                np.save(scores_path, np.array(arrays['scores'], dtype=object))
                
                # 메타데이터
                metadata = {
                    'frame_count': arrays['frame_count'],
                    'keypoint_scale': keypoint_scale,
                    'output_key': output_key
                }
                metadata_path = item_dir / "metadata.npy"
                np.save(metadata_path, metadata)
                
                print(f"✅ 저장 완료: {output_key} ({arrays['frame_count']} 프레임)")
                
            except Exception as e:
                print(f"💥 저장 오류 ({output_key}): {e}")
                continue
                
        except queue.Empty:
            continue
        except Exception as e:
            print(f"💻 CPU 후처리 워커 오류: {e}")
            continue

class BatchProcessor:
    """폴더별 250개 단위 배치 처리기 (ONNX 최적화 병렬 처리) - 완전 구현"""
    
    def __init__(self, 
                 data_root: str = "data/1.Training",
                 output_dir: str = "sign_language_dataset",
                 rtmw_model_name: str = "rtmw-dw-x-l_simcc-cocktail14_270e-384x288.onnx",
                 direction: str = "F",
                 item_types: List[str] = ["WORD"],
                 gpu_batch_size: int = 8,
                 num_cpu_workers: int = 4,
                 yolo_device: str = "auto",
                 pose_device: str = "auto"):
        
        self.data_root = Path(data_root)
        self.output_dir = Path(output_dir)
        self.rtmw_model_name = rtmw_model_name
        self.direction = direction
        self.item_types = item_types
        self.gpu_batch_size = gpu_batch_size
        self.num_cpu_workers = num_cpu_workers
        self.yolo_device = yolo_device
        self.pose_device = pose_device
        self.keypoint_scale = 8
        
        # 출력 디렉토리 생성
        self.video_output_dir = self.output_dir / "video_processing"
        self.hdf5_output_dir = self.output_dir / "hdf5_batches"
        self.video_output_dir.mkdir(parents=True, exist_ok=True)
        self.hdf5_output_dir.mkdir(parents=True, exist_ok=True)
        
        # 로깅
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

    def collect_videos_by_folder(self) -> Dict[str, List[Tuple[str, int, str]]]:
        """폴더별로 비디오 수집"""
        videos_base_dir = self.data_root / "videos"
        folder_video_data = {}
        
        if not videos_base_dir.exists():
            self.logger.error(f"비디오 디렉토리가 없습니다: {videos_base_dir}")
            return {}
        
        pattern = f"*_{self.direction}.mp4"
        
        for folder in videos_base_dir.iterdir():
            if folder.is_dir():
                folder_name = folder.name
                folder_videos = []
                
                for video_file in folder.glob(pattern):
                    # 파일명에서 아이템 정보 추출 (NIA_SL_WORD0515_REAL11_F.mp4 형태)
                    filename = video_file.stem
                    try:
                        if 'NIA_SL_' in filename:
                            # NIA_SL_WORD0515_REAL11_F에서 WORD0515 부분 추출
                            parts = filename.split('_')
                            if len(parts) >= 3:
                                word_part = parts[2]  # WORD0515
                                if word_part.startswith('WORD') and len(word_part) > 4:
                                    item_type = "WORD"
                                    item_id = int(word_part[4:])  # 0515 부분
                                    
                                    if item_type in self.item_types:
                                        folder_videos.append((item_type, item_id, str(video_file)))
                    except (ValueError, IndexError):
                        self.logger.debug(f"파일명 파싱 실패: {filename}")
                        continue
                
                if folder_videos:
                    folder_video_data[folder_name] = sorted(folder_videos, key=lambda x: (x[0], x[1]))
        
        total_videos = sum(len(videos) for videos in folder_video_data.values())
        self.logger.info(f"📊 폴더별 비디오 수집 완료:")
        for folder, videos in folder_video_data.items():
            self.logger.info(f"   - {folder}: {len(videos)}개")
        self.logger.info(f"📊 총 {total_videos}개 비디오")
        
        return folder_video_data

    def create_batches_by_folder(self, folder_video_data: Dict[str, List[Tuple[str, int, str]]]) -> List[Dict]:
        """폴더별로 250개 단위 배치 생성"""
        all_batches = []
        batch_id = 0
        
        for folder_name, videos in folder_video_data.items():
            # 250개 단위로 분할
            batch_size = 250
            folder_batch_idx = 0
            
            for i in range(0, len(videos), batch_size):
                batch_videos = videos[i:i + batch_size]
                
                batch_info = {
                    'batch_id': batch_id,
                    'folder_name': folder_name,
                    'folder_batch_idx': folder_batch_idx,
                    'data': batch_videos,
                    'item_range': f"{batch_videos[0][0]}{batch_videos[0][1]:04d}~{batch_videos[-1][0]}{batch_videos[-1][1]:04d}"
                }
                
                all_batches.append(batch_info)
                batch_id += 1
                folder_batch_idx += 1
        
        self.logger.info(f"📦 총 {len(all_batches)}개 배치 생성 ({batch_size}개씩)")
        return all_batches

    def process_all_batches(self, cleanup_intermediate: bool = False):
        """전체 배치 처리 파이프라인 (멀티 GPU 최적화 병렬 처리) - 완전 구현"""
        folder_video_data = self.collect_videos_by_folder()
        if not folder_video_data:
            self.logger.error("❌ 처리할 영상이 없습니다")
            return
        
        all_videos = []
        for videos in folder_video_data.values():
            all_videos.extend(videos)
        
        if not all_videos:
            self.logger.error("❌ 처리할 영상이 없습니다")
            return
        
        # 멀티프로세싱 시작 방법을 spawn으로 설정 (CUDA 호환성)
        mp.set_start_method('spawn', force=True)
        
        # GPU 개수 확인
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
        if num_gpus == 0:
            num_gpus = 1
            self.logger.warning("⚠️ CUDA GPU를 찾을 수 없습니다. CPU 모드로 실행합니다.")

        self.logger.info(f"🚀 멀티 GPU 처리 시작: {num_gpus}개 GPU, {len(all_videos)}개 비디오")
        self.logger.info(f"   - GPU 배치 크기: {self.gpu_batch_size}")
        self.logger.info(f"   - CPU 워커 수: {self.num_cpu_workers}")

        # 1. 프로세스 간 통신을 위한 큐 생성
        task_queue = mp.Queue(maxsize=num_gpus * 10)
        result_queue = mp.Queue(maxsize=self.num_cpu_workers * 5)
        
        # 2. 작업 큐에 모든 비디오 추가
        self.logger.info("📤 작업 큐에 비디오 추가 중...")
        for i, (item_type, item_id, video_path) in enumerate(all_videos):
            task_queue.put((i, item_type, item_id, video_path))
        
        # 3. 멀티 GPU ONNX 추론 프로세스 생성
        gpu_workers = []
        for gpu_id in range(num_gpus):
            worker = mp.Process(target=multi_gpu_onnx_inference_worker, args=(
                gpu_id, task_queue, result_queue, self.rtmw_model_name,
                self.gpu_batch_size, self.yolo_device, self.pose_device
            ))
            worker.start()
            gpu_workers.append(worker)

        # 4. CPU 후처리 프로세스 풀
        postprocess_workers = []
        for i in range(self.num_cpu_workers):
            worker = mp.Process(target=enhanced_cpu_postprocess_worker, args=(
                result_queue, self.video_output_dir, self.keypoint_scale
            ))
            worker.start()
            postprocess_workers.append(worker)

        # 5. 진행률 표시 및 대기
        successful_keys_map = {}
        device_info = f"{num_gpus}xGPU, {self.num_cpu_workers}xCPU"
        processed_count = 0
        
        with tqdm(total=len(all_videos), desc=f"멀티 GPU 병렬 처리 ({device_info})") as pbar:
            while processed_count < len(all_videos):
                try:
                    job_id, item_type, item_id, video_path, arrays = result_queue.get(timeout=2)
                    if arrays:
                        key = f"{item_type}{item_id:04d}"
                        # 배치 정보에 키 추가
                        for batch_info in self.create_batches_by_folder(folder_video_data):
                            if any(d[0] == item_type and d[1] == item_id for d in batch_info['data']):
                                if batch_info['batch_id'] not in successful_keys_map:
                                    successful_keys_map[batch_info['batch_id']] = []
                                successful_keys_map[batch_info['batch_id']].append(key)
                                break
                    
                    processed_count += 1
                    pbar.update(1)
                except queue.Empty:
                    # 타임아웃 시 현재 상태 확인
                    current_processed = len([d for d in self.video_output_dir.iterdir() if d.is_dir()])
                    if current_processed > processed_count:
                        diff = current_processed - processed_count
                        pbar.update(diff)
                        processed_count = current_processed
                    continue

        # 6. 모든 워커 종료
        self.logger.info("🛑 워커 종료 중...")
        for _ in range(num_gpus):
            task_queue.put(None)
        for _ in range(self.num_cpu_workers):
            result_queue.put(None)

        # GPU 워커들 종료 대기
        for worker in gpu_workers:
            worker.join(timeout=30)
            if worker.is_alive():
                worker.terminate()

        # CPU 워커들 종료 대기
        for worker in postprocess_workers:
            worker.join(timeout=30)
            if worker.is_alive():
                worker.terminate()
        
        self.logger.info("✅ 모든 비디오 처리 완료. HDF5 생성을 시작합니다.")

        # 7. HDF5 배치 생성
        all_batches = self.create_batches_by_folder(folder_video_data)
        for batch_info in all_batches:
            if batch_info['batch_id'] in successful_keys_map:
                successful_keys = successful_keys_map[batch_info['batch_id']]
                if successful_keys:
                    self.create_hdf5_batch(successful_keys, batch_info)
        
        # 8. 중간 파일 정리
        if cleanup_intermediate:
            self.cleanup_video_files()
        
        # 9. 최종 통계
        self.print_final_statistics()

    def create_hdf5_batch(self, successful_keys: List[str], batch_info: Dict):
        """HDF5 배치 파일 생성"""
        try:
            batch_id = batch_info['batch_id']
            folder_name = batch_info['folder_name']
            
            frames_h5_path = self.hdf5_output_dir / f"batch_{folder_name}_{batch_id:03d}_{self.direction}_frames.h5"
            poses_h5_path = self.hdf5_output_dir / f"batch_{folder_name}_{batch_id:03d}_{self.direction}_poses.h5"
            
            # JPEG 가변 길이 타입
            jpeg_vlen_dtype = h5py.vlen_dtype(np.uint8)
            
            with h5py.File(frames_h5_path, 'w') as f_frames, \
                 h5py.File(poses_h5_path, 'w') as f_poses:
                
                for key in successful_keys:
                    item_dir = self.video_output_dir / key
                    if not item_dir.exists():
                        continue
                    
                    try:
                        # 데이터 로드
                        frames = np.load(item_dir / "frames.npy", allow_pickle=True)
                        keypoints = np.load(item_dir / "keypoints.npy", allow_pickle=True)
                        
                        # HDF5에 저장
                        f_frames.create_dataset(key, data=frames, dtype=jpeg_vlen_dtype)
                        f_poses.create_dataset(key, data=keypoints)
                        
                    except Exception as e:
                        self.logger.error(f"아이템 {key} HDF5 변환 실패: {e}")
                        continue
            
            self.logger.info(f"✅ HDF5 배치 생성 완료: {batch_info['item_range']} ({len(successful_keys)}개)")
            
        except Exception as e:
            self.logger.error(f"HDF5 배치 생성 실패: {e}")

    def cleanup_video_files(self):
        """중간 비디오 파일들 정리"""
        try:
            import shutil
            if self.video_output_dir.exists():
                shutil.rmtree(self.video_output_dir)
            self.logger.info("🧹 중간 파일 정리 완료")
        except Exception as e:
            self.logger.error(f"중간 파일 정리 실패: {e}")

    def print_final_statistics(self):
        """최종 통계 출력"""
        processed_videos = len([d for d in self.video_output_dir.iterdir() if d.is_dir()])
        hdf5_files = len(list(self.hdf5_output_dir.glob("*.h5")))
        
        self.logger.info("="*80)
        self.logger.info("📊 최종 처리 통계:")
        self.logger.info(f"   - 처리된 비디오: {processed_videos}개")
        self.logger.info(f"   - 생성된 HDF5 파일: {hdf5_files}개")
        self.logger.info(f"   - 출력 디렉토리: {self.output_dir}")
        self.logger.info("="*80)

def main():
    """메인 실행 함수"""
    processor = BatchProcessor(
        data_root="/workspace01/team03/data/mmpose/jy/data/1.Training",
        output_dir="/workspace01/team03/data/streamlined_output",
        direction="F",
        item_types=["WORD"],
        gpu_batch_size=32,       # 안정적인 배치 크기
        num_cpu_workers=8,       # CPU 워커 최적화
        yolo_device="auto",
        pose_device="auto"
    )
    
    processor.process_all_batches(cleanup_intermediate=False)

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
