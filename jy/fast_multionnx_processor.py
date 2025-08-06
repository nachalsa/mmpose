#!/usr/bin/env python3
"""
Fast Multi-ONNX Processor - 고속 커스터마이징 가능 버전
A6000 x2 + 112 CPU 환경 최적화
실제 동작 보장, 커스터마이징 친화적
"""

import os
import cv2
import h5py
import time
import torch
import queue
import logging
import shutil
import numpy as np
import multiprocessing as mp
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional, Union

from onnx_inferencer import YOLO11LRTMWONNXInferencer as ONNXInferencer

class FastVideoProcessor:
    """고속 비디오 처리기 - ONNX 하이브리드"""
    
    def __init__(self, 
                 rtmw_model_name: str = "rtmw-dw-x-l_simcc-cocktail14_270e-384x288.onnx",
                 yolo_device: str = "auto",
                 pose_device: str = "auto",
                 keypoint_scale: int = 8,
                 jpeg_quality: int = 90):
        
        self.rtmw_model_name = rtmw_model_name
        self.yolo_device = yolo_device
        self.pose_device = pose_device
        self.keypoint_scale = keypoint_scale
        self.jpeg_quality = jpeg_quality
        
        # ONNX 추론기 초기화 - GPU 강제 설정
        if torch.cuda.is_available():
            # 현재 프로세스가 설정한 CUDA_VISIBLE_DEVICES에 따라 device 설정
            current_device = torch.cuda.current_device()
            device = f"cuda:{current_device}"
            torch.cuda.set_device(current_device)
        else:
            device = "cpu"
            
        try:
            self.inferencer = ONNXInferencer(
                rtmw_onnx_path=rtmw_model_name,
                detection_device=device,
                pose_device="cuda",  # 명시적으로 CUDA 설정
                optimize_for_accuracy=True
            )
            
            # GPU 메모리 상태 확인
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.get_device_properties(current_device).total_memory / 1024**3
                gpu_name = torch.cuda.get_device_properties(current_device).name
                allocated_memory = torch.cuda.memory_allocated(current_device) / 1024**3
                print(f"🚀 FastVideoProcessor 초기화 완료")
                print(f"   - GPU: {gpu_name} ({gpu_memory:.1f}GB)")
                print(f"   - 디바이스: {device} (메모리 사용: {allocated_memory:.2f}GB)")
                print(f"   - RTMW 모델: {rtmw_model_name}")
                print(f"   - 키포인트 스케일: {keypoint_scale}")
            else:
                print(f"🚀 FastVideoProcessor 초기화 완료 (CPU 모드)")
                print(f"   - 디바이스: {device}")
                print(f"   - RTMW 모델: {rtmw_model_name}")
                
        except Exception as e:
            print(f"❌ FastVideoProcessor 초기화 실패: {e}")
            print(f"   - 시도한 디바이스: {device}")
            raise

    def process_video_fast(self, video_path: str) -> Optional[Dict]:
        """고속 비디오 처리"""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"❌ 비디오 열기 실패: {video_path}")
                return None
            
            # 프레임 수 미리 계산
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames == 0:
                cap.release()
                return None
            
            jpeg_frames = []
            keypoints_list = []
            scores_list = []
            
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                try:
                    # 추론 실행
                    result = self.inferencer.process_frame(frame)
                    
                    # JPEG 인코딩
                    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
                    jpeg_frames.append(buffer.tobytes())
                    
                    # 결과 처리
                    frame_keypoints = []
                    frame_scores = []
                    
                    if isinstance(result, tuple) and len(result) > 1:
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
                                    for i in range(0, min(51, len(kpts)), 3):
                                        x, y, conf = kpts[i], kpts[i+1], kpts[i+2]
                                        scaled_kpts.extend([
                                            int(x * self.keypoint_scale),
                                            int(y * self.keypoint_scale),
                                            int(conf * self.keypoint_scale)
                                        ])
                                    
                                    frame_keypoints.append(scaled_kpts[:51])
                                    frame_scores.append(person.get('score', 1.0))
                    
                    # 키포인트가 없는 경우 기본값
                    if not frame_keypoints:
                        frame_keypoints.append([0] * 51)
                        frame_scores.append(0.0)
                    
                    keypoints_list.append(frame_keypoints)
                    scores_list.append(frame_scores)
                    frame_idx += 1
                    
                except Exception as e:
                    print(f"⚠️ 프레임 {frame_idx} 처리 오류: {e}")
                    # 오류 시 기본값 추가
                    jpeg_frames.append(b'')
                    keypoints_list.append([[0] * 51])
                    scores_list.append([0.0])
                    frame_idx += 1
                    continue
            
            cap.release()
            
            if frame_idx == 0:
                return None
            
            return {
                'jpeg_frames': jpeg_frames,
                'keypoints': keypoints_list,
                'scores': scores_list,
                'frame_count': frame_idx
            }
            
        except Exception as e:
            print(f"💥 비디오 처리 오류 ({video_path}): {e}")
            return None

def fast_gpu_worker(
    gpu_id: int,
    task_queue: mp.Queue, 
    result_queue: mp.Queue, 
    config: Dict
):
    """고속 GPU 워커 - GPU 전용 최적화"""
    try:
        # GPU 설정 - 강제로 특정 GPU만 보이도록 설정
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        
        # PyTorch CUDA 초기화
        if torch.cuda.is_available():
            torch.cuda.set_device(0)  # CUDA_VISIBLE_DEVICES로 설정했으므로 0번이 해당 GPU
            device = f"cuda:0"
            
            # GPU 워밍업
            dummy_tensor = torch.randn(1, 3, 384, 288).cuda()
            _ = dummy_tensor * 2
            del dummy_tensor
            torch.cuda.empty_cache()
            
            # GPU 정보 출력
            gpu_name = torch.cuda.get_device_properties(0).name
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"🚀 GPU {gpu_id} 워커 시작")
            print(f"   - GPU: {gpu_name} ({gpu_memory:.1f}GB)")
            print(f"   - 디바이스: {device}")
        else:
            device = "cpu"
            print(f"⚠️ GPU {gpu_id} 사용 불가 - CPU 모드")
            
        # 프로세서 초기화
        processor = FastVideoProcessor(
            rtmw_model_name=config['rtmw_model_name'],
            yolo_device=device,
            pose_device="cuda",  # 명시적으로 CUDA 설정
            keypoint_scale=config['keypoint_scale'],
            jpeg_quality=config['jpeg_quality']
        )
        
        processed_count = 0
        
        while True:
            try:
                job = task_queue.get(timeout=5.0)
                if job is None:  # 종료 신호
                    if torch.cuda.is_available():
                        memory_used = torch.cuda.memory_allocated(0) / 1024**3
                        print(f"🔥 GPU {gpu_id} 워커 종료 (처리: {processed_count}개, GPU 메모리: {memory_used:.2f}GB)")
                        torch.cuda.empty_cache()
                    else:
                        print(f"🔥 GPU {gpu_id} 워커 종료 (처리: {processed_count}개)")
                    break
                
                job_id, item_type, item_id, video_path = job
                
                start_time = time.time()
                arrays = processor.process_video_fast(video_path)
                process_time = time.time() - start_time
                
                if arrays:
                    print(f"✅ GPU {gpu_id}: {item_type}{item_id:04d} 완료 ({arrays['frame_count']} frames, {process_time:.2f}s)")
                else:
                    print(f"❌ GPU {gpu_id}: {item_type}{item_id:04d} 실패")
                
                result_queue.put((job_id, item_type, item_id, video_path, arrays))
                processed_count += 1
                
                # GPU 메모리 정리 (주기적으로)
                if torch.cuda.is_available() and processed_count % 10 == 0:
                    torch.cuda.empty_cache()
                    
            except queue.Empty:
                if torch.cuda.is_available():
                    memory_used = torch.cuda.memory_allocated(0) / 1024**3
                    print(f"⏰ GPU {gpu_id} 워커 타임아웃 (처리: {processed_count}개, GPU 메모리: {memory_used:.2f}GB)")
                else:
                    print(f"⏰ GPU {gpu_id} 워커 타임아웃 (처리: {processed_count}개)")
                break
            except Exception as e:
                print(f"💥 GPU {gpu_id} 워커 오류: {e}")
                result_queue.put((job_id, item_type, item_id, video_path, None))
                continue
                
    except Exception as e:
        print(f"❌ GPU {gpu_id} 워커 초기화 실패: {e}")
        import traceback
        traceback.print_exc()

def fast_cpu_worker(
    result_queue: mp.Queue, 
    output_dir: Path,
    config: Dict
):
    """고속 CPU 후처리 워커"""
    print(f"💻 CPU 워커 시작")
    processed_count = 0
    
    while True:
        try:
            result = result_queue.get(timeout=10)
            if result is None:  # 종료 신호
                print(f"💻 CPU 워커 종료 (처리: {processed_count}개)")
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
                # 효율적인 데이터 저장
                frames_path = item_dir / "frames.h5"
                poses_path = item_dir / "poses.h5"
                
                # HDF5로 직접 저장 (메모리 효율적)
                with h5py.File(frames_path, 'w') as f:
                    # JPEG 가변 길이 타입
                    jpeg_vlen_dtype = h5py.vlen_dtype(np.uint8)
                    jpeg_dataset = f.create_dataset('frames', (len(arrays['jpeg_frames']),), dtype=jpeg_vlen_dtype)
                    
                    for i, jpeg_data in enumerate(arrays['jpeg_frames']):
                        if isinstance(jpeg_data, bytes) and len(jpeg_data) > 0:
                            jpeg_dataset[i] = np.frombuffer(jpeg_data, dtype=np.uint8)
                        else:
                            jpeg_dataset[i] = np.array([], dtype=np.uint8)
                
                with h5py.File(poses_path, 'w') as f:
                    # 키포인트 데이터
                    max_persons = max(len(kpts) for kpts in arrays['keypoints']) if arrays['keypoints'] else 1
                    keypoints_array = np.zeros((len(arrays['keypoints']), max_persons, 51), dtype=np.float32)
                    scores_array = np.zeros((len(arrays['scores']), max_persons), dtype=np.float32)
                    
                    for i, (kpts_frame, scores_frame) in enumerate(zip(arrays['keypoints'], arrays['scores'])):
                        for j, (kpts, score) in enumerate(zip(kpts_frame, scores_frame)):
                            if j < max_persons:
                                keypoints_array[i, j, :len(kpts)] = kpts[:51]
                                scores_array[i, j] = score
                    
                    f.create_dataset('keypoints', data=keypoints_array)
                    f.create_dataset('scores', data=scores_array)
                    f.attrs['frame_count'] = arrays['frame_count']
                    f.attrs['keypoint_scale'] = config['keypoint_scale']
                
                print(f"✅ 저장 완료: {output_key} ({arrays['frame_count']} 프레임)")
                processed_count += 1
                
            except Exception as e:
                print(f"💥 저장 오류 ({output_key}): {e}")
                continue
                
        except queue.Empty:
            continue
        except Exception as e:
            print(f"💻 CPU 워커 오류: {e}")
            continue

class FastMultiONNXProcessor:
    """고속 멀티 ONNX 프로세서 - 커스터마이징 친화적"""
    
    def __init__(self, 
                 data_root: str = "data/1.Training",
                 output_dir: str = "fast_multionnx_output",
                 rtmw_model_name: str = "rtmw-dw-x-l_simcc-cocktail14_270e-384x288.onnx",
                 direction: str = "F",
                 item_types: List[str] = ["WORD"],
                 num_cpu_workers: int = 8,
                 yolo_device: str = "auto",
                 pose_device: str = "auto",
                 keypoint_scale: int = 8,
                 jpeg_quality: int = 90,
                 enable_hdf5_batch: bool = True):
        
        self.data_root = Path(data_root)
        self.output_dir = Path(output_dir)
        self.direction = direction
        self.item_types = item_types
        self.num_cpu_workers = num_cpu_workers
        self.enable_hdf5_batch = enable_hdf5_batch
        
        # 설정 딕셔너리
        self.config = {
            'rtmw_model_name': rtmw_model_name,
            'yolo_device': yolo_device,
            'pose_device': pose_device,
            'keypoint_scale': keypoint_scale,
            'jpeg_quality': jpeg_quality
        }
        
        # 출력 디렉토리 생성
        self.video_output_dir = self.output_dir / "videos"
        self.hdf5_output_dir = self.output_dir / "batches"
        self.video_output_dir.mkdir(parents=True, exist_ok=True)
        self.hdf5_output_dir.mkdir(parents=True, exist_ok=True)
        
        # 로깅
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
        print(f"🚀 Fast Multi-ONNX Processor 초기화 완료")
        print(f"   - 데이터 루트: {self.data_root}")
        print(f"   - 출력 디렉토리: {self.output_dir}")
        print(f"   - 방향: {direction}, 타입: {item_types}")
        print(f"   - CPU 워커: {num_cpu_workers}")
        print(f"   - HDF5 배치: {'활성화' if enable_hdf5_batch else '비활성화'}")

    def collect_all_videos(self) -> List[Tuple[str, int, str]]:
        """모든 비디오 수집 - 실제 경로 구조에 맞춤"""
        videos_base_dir = self.data_root / "videos"
        all_videos = []
        
        if not videos_base_dir.exists():
            self.logger.error(f"비디오 디렉토리 없음: {videos_base_dir}")
            return []
        
        pattern = f"*_{self.direction}.mp4"
        
        # 폴더별 검색
        for folder in videos_base_dir.iterdir():
            if folder.is_dir():
                folder_videos = []
                
                for video_file in folder.glob(pattern):
                    item_info = self.extract_item_info_from_path(str(video_file))
                    if item_info:
                        item_type, item_id = item_info
                        if item_type in self.item_types:
                            folder_videos.append((item_type, item_id, str(video_file)))
                
                if folder_videos:
                    all_videos.extend(sorted(folder_videos, key=lambda x: (x[0], x[1])))
                    self.logger.info(f"📁 폴더 {folder.name}: {len(folder_videos)}개 비디오")
        
        self.logger.info(f"📊 총 {len(all_videos)}개 비디오 발견")
        return all_videos

    def extract_item_info_from_path(self, video_path: str) -> Optional[Tuple[str, int]]:
        """경로에서 아이템 정보 추출 (NIA_SL_WORD0515_REAL11_F.mp4 형태)"""
        filename = Path(video_path).stem
        try:
            # NIA_SL_WORD0515_REAL11_F 형태에서 WORD0515 추출
            parts = filename.split('_')
            for part in parts:
                for item_type in self.item_types:
                    if part.startswith(item_type):
                        item_id_str = part[len(item_type):]
                        item_id = int(item_id_str)
                        return (item_type, item_id)
        except (ValueError, IndexError):
            pass
        return None

    def process_videos_parallel(self, max_videos: Optional[int] = None):
        """병렬 비디오 처리 - 메인 실행 함수"""
        # 비디오 수집
        all_videos = self.collect_all_videos()
        if not all_videos:
            self.logger.error("❌ 처리할 비디오가 없습니다")
            return
        
        # 최대 개수 제한
        if max_videos and max_videos > 0:
            all_videos = all_videos[:max_videos]
            self.logger.info(f"🔢 처리 제한: {max_videos}개 비디오")
        
        # 멀티프로세싱 시작 방법 설정
        mp.set_start_method('spawn', force=True)
        
        # GPU 개수 확인
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
        if num_gpus == 0:
            num_gpus = 1
            self.logger.warning("⚠️ CUDA GPU가 없습니다. CPU 모드로 실행합니다.")
        
        self.logger.info(f"🚀 병렬 처리 시작: {num_gpus}개 GPU, {self.num_cpu_workers}개 CPU 워커")
        
        # 큐 생성
        task_queue = mp.Queue(maxsize=num_gpus * 10)
        result_queue = mp.Queue(maxsize=self.num_cpu_workers * 5)
        
        # 작업 큐에 비디오 추가
        for i, (item_type, item_id, video_path) in enumerate(all_videos):
            task_queue.put((i, item_type, item_id, video_path))
        
        # GPU 워커 시작
        gpu_workers = []
        for gpu_id in range(num_gpus):
            worker = mp.Process(target=fast_gpu_worker, args=(
                gpu_id, task_queue, result_queue, self.config
            ))
            worker.start()
            gpu_workers.append(worker)
        
        # CPU 워커 시작
        cpu_workers = []
        for i in range(self.num_cpu_workers):
            worker = mp.Process(target=fast_cpu_worker, args=(
                result_queue, self.video_output_dir, self.config
            ))
            worker.start()
            cpu_workers.append(worker)
        
        # 진행률 추적
        processed_videos = []
        device_info = f"{num_gpus}xGPU, {self.num_cpu_workers}xCPU"
        
        with tqdm(total=len(all_videos), desc=f"Fast Multi-ONNX 처리 ({device_info})") as pbar:
            timeout_count = 0
            while len(processed_videos) < len(all_videos):
                try:
                    job_id, item_type, item_id, video_path, arrays = result_queue.get(timeout=2)
                    if arrays:
                        output_key = f"{item_type}{item_id:04d}"
                        processed_videos.append(output_key)
                    pbar.update(1)
                    timeout_count = 0  # 성공 시 타임아웃 카운터 리셋
                except queue.Empty:
                    timeout_count += 1
                    if timeout_count > 30:  # 60초 대기 후 강제 종료
                        self.logger.warning("⏰ 처리 타임아웃으로 인한 강제 종료")
                        break
                    continue
        
        # 워커 종료
        self.logger.info("🛑 워커 종료 중...")
        for _ in range(num_gpus):
            task_queue.put(None)
        for _ in range(self.num_cpu_workers):
            result_queue.put(None)
        
        # 워커 대기
        for worker in gpu_workers:
            worker.join(timeout=10)
            if worker.is_alive():
                worker.terminate()
        
        for worker in cpu_workers:
            worker.join(timeout=10)
            if worker.is_alive():
                worker.terminate()
        
        self.logger.info(f"✅ 비디오 처리 완료: {len(processed_videos)}개")
        
        # HDF5 배치 생성 (옵션)
        if self.enable_hdf5_batch and processed_videos:
            self.create_hdf5_batches(processed_videos)
        
        # 최종 통계
        self.print_statistics(processed_videos)

    def create_hdf5_batches(self, processed_videos: List[str], batch_size: int = 250):
        """HDF5 배치 파일 생성"""
        self.logger.info(f"📦 HDF5 배치 생성 시작 ({batch_size}개씩)")
        
        for batch_idx in range(0, len(processed_videos), batch_size):
            batch_keys = processed_videos[batch_idx:batch_idx + batch_size]
            batch_name = f"batch_{batch_idx//batch_size:03d}_{self.direction}"
            
            frames_h5_path = self.hdf5_output_dir / f"{batch_name}_frames.h5"
            poses_h5_path = self.hdf5_output_dir / f"{batch_name}_poses.h5"
            
            try:
                with h5py.File(frames_h5_path, 'w') as f_frames, \
                     h5py.File(poses_h5_path, 'w') as f_poses:
                    
                    for key in batch_keys:
                        item_dir = self.video_output_dir / key
                        if not item_dir.exists():
                            continue
                        
                        # 개별 HDF5 파일에서 데이터 읽기
                        frames_path = item_dir / "frames.h5"
                        poses_path = item_dir / "poses.h5"
                        
                        if frames_path.exists() and poses_path.exists():
                            with h5py.File(frames_path, 'r') as src_f, \
                                 h5py.File(poses_path, 'r') as src_p:
                                
                                # 데이터 복사
                                f_frames.copy(src_f['frames'], key)
                                f_poses.copy(src_p['keypoints'], f"{key}/keypoints")
                                f_poses.copy(src_p['scores'], f"{key}/scores")
                
                self.logger.info(f"✅ HDF5 배치 생성: {batch_name} ({len(batch_keys)}개)")
                
            except Exception as e:
                self.logger.error(f"❌ HDF5 배치 생성 실패 ({batch_name}): {e}")

    def print_statistics(self, processed_videos: List[str]):
        """최종 통계 출력"""
        self.logger.info("="*80)
        self.logger.info("📊 Fast Multi-ONNX 처리 통계:")
        self.logger.info(f"   - 처리된 비디오: {len(processed_videos)}개")
        self.logger.info(f"   - 출력 디렉토리: {self.output_dir}")
        self.logger.info(f"   - 개별 비디오 파일: {self.video_output_dir}")
        if self.enable_hdf5_batch:
            hdf5_files = len(list(self.hdf5_output_dir.glob("*.h5")))
            self.logger.info(f"   - HDF5 배치 파일: {hdf5_files}개")
        self.logger.info("="*80)

    # 커스터마이징 메서드들
    def customize_keypoint_scale(self, scale: int):
        """키포인트 스케일 변경"""
        self.config['keypoint_scale'] = scale
        print(f"🔧 키포인트 스케일 변경: {scale}")
    
    def customize_jpeg_quality(self, quality: int):
        """JPEG 품질 변경"""
        self.config['jpeg_quality'] = quality
        print(f"🔧 JPEG 품질 변경: {quality}")
    
    def customize_output_dir(self, output_dir: str):
        """출력 디렉토리 변경"""
        self.output_dir = Path(output_dir)
        self.video_output_dir = self.output_dir / "videos"
        self.hdf5_output_dir = self.output_dir / "batches"
        self.video_output_dir.mkdir(parents=True, exist_ok=True)
        self.hdf5_output_dir.mkdir(parents=True, exist_ok=True)
        print(f"🔧 출력 디렉토리 변경: {output_dir}")
    
    def enable_debug_mode(self):
        """디버그 모드 활성화"""
        logging.getLogger().setLevel(logging.DEBUG)
        print("🔧 디버그 모드 활성화")

def main():
    """메인 실행 함수 - 커스터마이징 가능"""
    print("🚀 Fast Multi-ONNX Processor")
    print("=" * 60)
    
    # 기본 설정
    processor = FastMultiONNXProcessor(
        data_root="/workspace01/team03/data/mmpose/jy/data/1.Training",
        output_dir="/workspace01/team03/data/fast_multionnx_output",
        direction="F",
        item_types=["WORD"],
        num_cpu_workers=8,       # A6000 환경에 최적화
        yolo_device="auto",
        pose_device="auto",
        keypoint_scale=8,
        jpeg_quality=90,
        enable_hdf5_batch=True
    )
    
    # 커스터마이징 예시
    choice = input("처리할 비디오 수를 제한하시겠습니까? (y/N): ").strip().lower()
    max_videos = None
    if choice == 'y':
        try:
            max_videos = int(input("최대 처리할 비디오 수 입력: "))
        except ValueError:
            max_videos = None
    
    # 처리 실행
    processor.process_videos_parallel(max_videos=max_videos)

if __name__ == "__main__":
    main()
