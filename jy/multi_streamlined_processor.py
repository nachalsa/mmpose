#!/usr/bin/env python3
"""
스트림라인 비디오 처리기 - HDF5 배치 처리용 (WORD + SEN 지원)
GPU/CPU 병렬 처리를 통해 성능 극대화
"""

import os
import cv2
import json
import h5py
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import logging
from tqdm import tqdm
import time
import shutil
import urllib.request
from datetime import datetime
import re
import multiprocessing as mp
import queue # 작업 큐
# import gc # 가비지 컬렉터 임포트

# 설정 및 MMPose 관련 임포트
from config import MODELS_DIR, YOLO_MODEL_CONFIG, RTMW_MODEL_OPTIONS
from yolo11l_xpu_hybrid_inferencer import YOLO11LXPUHybridInferencer

# RTMW 전처리 함수들 (video_processor_yolo11l.py에서 가져옴)
def bbox_xyxy2cs(bbox: np.ndarray, padding: float = 1.10) -> Tuple[np.ndarray, np.ndarray]:
    """바운딩박스를 center, scale로 변환 (패딩 1.10으로 수정)"""
    dim = bbox.ndim
    if dim == 1:
        bbox = bbox[None, :]
    
    scale = (bbox[..., 2:] - bbox[..., :2]) * padding
    center = (bbox[..., 2:] + bbox[..., :2]) * 0.5
    
    if dim == 1:
        center = center[0]
        scale = scale[0]
    
    return center, scale

def _rotate_point(pt: np.ndarray, angle_rad: float) -> np.ndarray:
    """점을 회전"""
    cos_val = np.cos(angle_rad)
    sin_val = np.sin(angle_rad)
    return np.array([pt[0] * cos_val - pt[1] * sin_val,
                     pt[0] * sin_val + pt[1] * cos_val])

def _get_3rd_point(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """세 번째 점을 계산 (직교점)"""
    direction = a - b
    return b + np.array([-direction[1], direction[0]])

def get_warp_matrix(center: np.ndarray, scale: np.ndarray, rot: float, 
                   output_size: Tuple[int, int]) -> np.ndarray:
    """아핀 변환 매트릭스 계산"""
    src_w, src_h = scale[:2]
    dst_w, dst_h = output_size[:2]
    
    rot_rad = np.deg2rad(rot)
    src_dir = _rotate_point(np.array([src_w * -0.5, 0.]), rot_rad)
    dst_dir = np.array([dst_w * -0.5, 0.])
    
    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center
    src[1, :] = center + src_dir
    
    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir
    
    # aspect ratio 고정
    src[2, :] = _get_3rd_point(src[0, :], src[1, :])
    dst[2, :] = _get_3rd_point(dst[0, :], dst[1, :])
    
    warp_mat = cv2.getAffineTransform(src, dst)
    return warp_mat

def fix_aspect_ratio(bbox_scale: np.ndarray, aspect_ratio: float) -> np.ndarray:
    """bbox를 고정 종횡비로 조정"""
    w, h = bbox_scale[0], bbox_scale[1]
    if w > h * aspect_ratio:
        new_h = w / aspect_ratio
        bbox_scale = np.array([w, new_h])
    else:
        new_w = h * aspect_ratio
        bbox_scale = np.array([new_w, h])
    return bbox_scale

class StreamlinedVideoProcessor:
    """HDF5용 간소화된 비디오 처리기 (WORD + SEN 지원) - 클래스는 그대로 유지"""
    
    def __init__(self, 
                 rtmw_config_path: str = "configs/wholebody_2d_keypoint/rtmpose/cocktail14/rtmw-x_8xb320-270e_cocktail14-384x288.py",
                 rtmw_model_name: str = "rtmw-x",
                 yolo_device: str = "cpu",  # CPU로 변경
                 pose_device: str = "xpu"): # Pose는 GPU 유지
        
        self.logger = logging.getLogger(__name__)
        self.keypoint_scale = 8
        self.yolo_device = yolo_device
        self.pose_device = pose_device
        
        try:
            base_dir = Path(__file__).parent.parent
        except NameError:
            base_dir = Path.cwd().parent
        rtmw_config_path = str(base_dir / rtmw_config_path)
        
        yolo_model_path = self._ensure_yolo_model()
        rtmw_model_path = self._ensure_rtmw_model(rtmw_model_name)
        
        self.inferencer = YOLO11LXPUHybridInferencer(
            rtmw_config=rtmw_config_path,
            rtmw_checkpoint=rtmw_model_path,
            detection_device=yolo_device,  # CPU 사용
            pose_device=pose_device,       # GPU 유지
            optimize_for_accuracy=True
        )
        
        device_info = f"YOLO: {yolo_device.upper()}, Pose: {pose_device.upper()}"
        self.logger.info(f"✅ 스트림라인 비디오 처리기 초기화 완료 ({device_info})")

    def _ensure_yolo_model(self) -> str:
        """YOLO 모델 파일 확인 및 다운로드"""
        yolo_config = YOLO_MODEL_CONFIG
        model_path = Path(MODELS_DIR) / yolo_config["filename"]
        
        if model_path.exists():
            self.logger.info(f"✅ 기존 YOLO 모델 발견: {model_path}")
            return str(model_path)
        
        self.logger.info(f"📥 YOLO 모델 다운로드 시작: {yolo_config['filename']}")
        
        # models 디렉토리 생성
        Path(MODELS_DIR).mkdir(parents=True, exist_ok=True)
        
        try:
            from ultralytics import YOLO
            temp_model = YOLO(yolo_config["filename"])
            cache_dir = Path.home() / '.cache' / 'ultralytics'
            downloaded_model = None
            
            for weights_dir in [cache_dir, cache_dir / 'weights']:
                if weights_dir.exists():
                    for model_file in weights_dir.glob(yolo_config["filename"]):
                        downloaded_model = model_file
                        break
                if downloaded_model:
                    break
            
            if downloaded_model and downloaded_model.exists():
                shutil.copy2(downloaded_model, model_path)
                self.logger.info(f"✅ YOLO 모델 복사 완료: {model_path}")
                self.logger.info(f"   파일 크기: {model_path.stat().st_size / (1024*1024):.1f} MB")
                return str(model_path)
            else:
                self.logger.warning(f"⚠️ YOLO 모델 다운로드 위치를 찾을 수 없음")
                return yolo_config["filename"]
            
        except Exception as e:
            self.logger.warning(f"⚠️ YOLO 모델 다운로드 실패: {e}")
            self.logger.info("   ultralytics가 자동으로 다운로드할 예정")
            return yolo_config["filename"]

    def _ensure_rtmw_model(self, model_name: str = "rtmw-x") -> str:
        """RTMW 모델 파일 확인 및 다운로드"""
        rtmw_config = None
        for config in RTMW_MODEL_OPTIONS:
            if model_name in config["filename"]:
                rtmw_config = config
                break
        
        if not rtmw_config:
            self.logger.error(f"❌ 알 수 없는 RTMW 모델명: {model_name}")
            rtmw_config = RTMW_MODEL_OPTIONS[0]
        
        model_path = Path(rtmw_config["path"])
        
        if model_path.exists():
            self.logger.info(f"✅ 기존 RTMW 모델 발견: {model_path}")
            return str(model_path)
        
        self.logger.info(f"📥 RTMW 모델 다운로드 시작: {rtmw_config['filename']}")
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            download_url = rtmw_config["url"]
            if not download_url:
                self.logger.error(f"❌ 다운로드 URL이 없음: {rtmw_config['filename']}")
                return str(model_path)
            
            self.logger.info(f"🔄 다운로드 중: {download_url}")
            
            def download_progress_hook(block_num, block_size, total_size):
                if total_size > 0:
                    percent = min(100, (block_num * block_size * 100) // total_size)
                    if block_num % 50 == 0:
                        self.logger.info(f"   다운로드 진행률: {percent}%")
            
            urllib.request.urlretrieve(download_url, model_path, download_progress_hook)
            
            if model_path.exists() and model_path.stat().st_size > 1024 * 1024:
                self.logger.info(f"✅ RTMW 모델 다운로드 완료: {model_path}")
                self.logger.info(f"   파일 크기: {model_path.stat().st_size / (1024*1024):.1f} MB")
                return str(model_path)
            else:
                self.logger.error(f"❌ 다운로드된 파일이 유효하지 않음: {model_path}")
                if model_path.exists():
                    model_path.unlink()
                return str(model_path)
            
        except Exception as e:
            self.logger.error(f"❌ RTMW 모델 다운로드 실패: {e}")
            if model_path.exists():
                model_path.unlink()
            return str(model_path)

    def _crop_person_image_rtmw(self, image: np.ndarray, bbox: List[float]) -> Optional[np.ndarray]:
        """RTMW 방식으로 사람 이미지 크롭"""
        try:
            input_width, input_height = 288, 384
            bbox_array = np.array(bbox, dtype=np.float32)
            center, scale = bbox_xyxy2cs(bbox_array)
            aspect_ratio = input_width / input_height
            scale = fix_aspect_ratio(scale, aspect_ratio)
            warp_mat = get_warp_matrix(
                center=center,
                scale=scale,
                rot=0.0,
                output_size=(input_width, input_height)
            )
            cropped_image = cv2.warpAffine(
                image, 
                warp_mat, 
                (input_width, input_height), 
                flags=cv2.INTER_LINEAR
            )
            h, w = cropped_image.shape[:2]
            if h == input_height and w == input_width:
                return cropped_image
            else:
                self.logger.warning(f"⚠️ 크기 오류: {h}x{w}, 예상: {input_height}x{input_width}")
                return cropped_image
                
        except Exception as e:
            self.logger.warning(f"⚠️ RTMW 전처리 실패: {e}")
            return None

    def process_video_to_arrays(self, video_path: str) -> Optional[Dict[str, Union[List[np.ndarray], np.ndarray, int]]]:
        """비디오를 처리하여 포즈 정보와 'JPEG 인코딩된 프레임'을 반환"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self.logger.error(f"❌ 비디오 열기 실패: {video_path}")
            return None
        
        try:
            all_jpeg_frames, all_keypoints, all_scores = [], [], []
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                try:
                    vis_image, results = self.inferencer.process_frame(frame)
                    if not results or len(results) == 0:
                        frame_idx += 1
                        continue
                    
                    _, _, bbox = results[0]
                    crop_image = self._crop_person_image_rtmw(frame, bbox)
                    if crop_image is None:
                        frame_idx += 1
                        continue
                    
                    keypoints, scores = self.inferencer.estimate_pose_on_crop(crop_image)
                    ret_jpg, encoded_jpg = cv2.imencode('.jpg', crop_image, [cv2.IMWRITE_JPEG_QUALITY, 90])
                    if not ret_jpg:
                        self.logger.warning(f"프레임 {frame_idx} JPEG 인코딩 실패: {video_path}")
                        frame_idx += 1
                        continue

                    all_jpeg_frames.append(encoded_jpg)
                    all_keypoints.append(keypoints)
                    all_scores.append(scores)
                    
                except Exception as e:
                    self.logger.warning(f"프레임 {frame_idx} 처리 실패: {e}")
                    frame_idx += 1
                    continue
                
                frame_idx += 1
            
            if not all_jpeg_frames:
                self.logger.warning(f"⚠️ 유효한 프레임이 없습니다: {video_path}")
                return None
            
            return {
                'jpeg_frames': all_jpeg_frames,
                'keypoints': np.stack(all_keypoints),
                'scores': np.stack(all_scores),
                'frame_count': len(all_jpeg_frames)
            }
            
        except Exception as e:
            self.logger.error(f"❌ 비디오 처리 실패: {video_path}, 오류: {e}")
            return None
        finally:
            if cap:
                cap.release()

    def process_video_to_arrays_batched(self, video_path: str, frame_batch_size: int = 16) -> Optional[Dict[str, Union[List[np.ndarray], np.ndarray, int]]]:
        """GPU 배치 처리를 위한 개선된 비디오 처리"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self.logger.error(f"❌ 비디오 열기 실패: {video_path}")
            return None
        
        try:
            all_jpeg_frames, all_keypoints, all_scores = [], [], []
            frame_batch = []
            bbox_batch = []
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    # 마지막 배치 처리
                    if frame_batch:
                        self._process_frame_batch(frame_batch, bbox_batch, all_jpeg_frames, all_keypoints, all_scores)
                    break
                
                # YOLO 검출 (개별 처리)
                vis_image, results = self.inferencer.process_frame(frame)
                if not results or len(results) == 0:
                    continue
                
                _, _, bbox = results[0]
                crop_image = self._crop_person_image_rtmw(frame, bbox)
                if crop_image is None:
                    continue
                
                frame_batch.append(crop_image)
                bbox_batch.append(bbox)
                
                # 배치가 차면 처리
                if len(frame_batch) >= frame_batch_size:
                    self._process_frame_batch(frame_batch, bbox_batch, all_jpeg_frames, all_keypoints, all_scores)
                    frame_batch = []
                    bbox_batch = []
            
            if not all_jpeg_frames:
                return None
            
            return {
                'jpeg_frames': all_jpeg_frames,
                'keypoints': np.stack(all_keypoints),
                'scores': np.stack(all_scores),
                'frame_count': len(all_jpeg_frames)
            }
            
        except Exception as e:
            self.logger.error(f"❌ 비디오 처리 실패: {video_path}, 오류: {e}")
            return None
        finally:
            if cap:
                cap.release()

    def _process_frame_batch(self, frame_batch: List[np.ndarray], bbox_batch: List, 
                        all_jpeg_frames: List, all_keypoints: List, all_scores: List):
        """프레임 배치를 GPU에서 한번에 처리"""
        try:
            # 배치 포즈 추정 (GPU 병렬 처리)
            batch_keypoints, batch_scores = self.inferencer.estimate_pose_batch(frame_batch)
            
            # 결과 저장
            for i, (crop_image, keypoints, scores) in enumerate(zip(frame_batch, batch_keypoints, batch_scores)):
                ret_jpg, encoded_jpg = cv2.imencode('.jpg', crop_image, [cv2.IMWRITE_JPEG_QUALITY, 90])
                if ret_jpg:
                    all_jpeg_frames.append(encoded_jpg)
                    all_keypoints.append(keypoints)
                    all_scores.append(scores)
                
        except Exception as e:
            self.logger.warning(f"배치 처리 실패: {e}")
            # 폴백: 개별 처리
            for crop_image in frame_batch:
                try:
                    keypoints, scores = self.inferencer.estimate_pose_on_crop(crop_image)
                    ret_jpg, encoded_jpg = cv2.imencode('.jpg', crop_image, [cv2.IMWRITE_JPEG_QUALITY, 90])
                    if ret_jpg:
                        all_jpeg_frames.append(encoded_jpg)
                        all_keypoints.append(keypoints)
                        all_scores.append(scores)
                except:
                    continue

    def process_video(self, item_type: str, item_id: int, video_path: str, output_dir: Path) -> bool:
        """
        WORD/SEN ID 기반으로 비디오 처리하고 저장
        crop_images를 직접 반환하지 않고, JPEG으로 인코딩하여 파일로 저장
        
        Args:
            item_type: "WORD" 또는 "SEN"
            item_id: WORD/SEN 번호 (예: 1)
            video_path: 비디오 파일 경로
            output_dir: 출력 디렉토리
            
        Returns:
            bool: 성공 여부
        """
        try:
            self.logger.info(f"🎬 처리 중: {item_type}{item_id:04d} - {Path(video_path).name}")
            start_time = time.time()
            
            arrays = self.process_video_to_arrays(video_path)
            if arrays is None:
                return False
            
            processing_time = time.time() - start_time
            
            item_dir = output_dir / f"{item_type}{item_id:04d}"
            item_dir.mkdir(parents=True, exist_ok=True)
            
            jpeg_frames_dict = {f'frame_{i}': frame for i, frame in enumerate(arrays['jpeg_frames'])}
            np.savez_compressed(item_dir / "crop_images_jpeg.npz", **jpeg_frames_dict)

            keypoints_scaled = np.round(arrays['keypoints'] * self.keypoint_scale).astype(np.int32)
            np.save(item_dir / "keypoints_scaled.npy", keypoints_scaled)
            np.save(item_dir / "scores.npy", arrays['scores'])
            
            metadata = {
                'item_type': item_type,
                'item_id': item_id,
                'video_path': str(video_path),
                'video_filename': Path(video_path).name,
                'frame_count': arrays['frame_count'],
                'processing_time': processing_time,
                'shape_info': {
                    'frame_count': arrays['frame_count'],
                    'keypoints_scaled': list(keypoints_scaled.shape),
                    'scores': list(arrays['scores'].shape)
                },
                'keypoint_scale': self.keypoint_scale,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            with open(item_dir / "metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.logger.info(f"✅ {item_type}{item_id:04d} 완료: {arrays['frame_count']}프레임, {processing_time:.2f}초")
            return True

        except Exception as e:
            self.logger.error(f"❌ {item_type}{item_id:04d} 처리 실패: {e}")
            return False


# +++ 신규: GPU 추론을 위한 워커 함수 +++
def gpu_inference_worker(
    task_queue: mp.Queue, 
    result_queue: mp.Queue, 
    rtmw_model_name: str, 
    rtmw_config_path: str,
    gpu_batch_size: int = 8,
    yolo_device: str = "cpu",  # YOLO CPU 사용
    pose_device: str = "xpu"   # Pose GPU 유지
):
    """GPU/CPU 하이브리드 추론을 수행하는 프로세스"""
    print(f"🚀 하이브리드 Inference Worker 시작 (YOLO: {yolo_device.upper()}, Pose: {pose_device.upper()}, 배치: {gpu_batch_size})...")
    
    # 이 프로세스 내에서 자체적으로 처리기 초기화
    processor = StreamlinedVideoProcessor(
        rtmw_model_name=rtmw_model_name,
        rtmw_config_path=rtmw_config_path,
        yolo_device=yolo_device,
        pose_device=pose_device
    )
    
    batch_jobs = []
    
    while True:
        try:
            # 배치 크기만큼 작업 수집
            for _ in range(gpu_batch_size):
                try:
                    job = task_queue.get_nowait()
                    if job is None:  # 종료 신호
                        # 남은 배치 처리
                        if batch_jobs:
                            process_batch(processor, batch_jobs, result_queue)
                        return
                    batch_jobs.append(job)
                except queue.Empty:
                    break
            
            # 수집된 작업들을 배치로 처리
            if batch_jobs:
                process_batch(processor, batch_jobs, result_queue)
                batch_jobs = []
            
            if not batch_jobs:  # 더 이상 작업이 없으면 잠시 대기
                time.sleep(0.1)
                
        except Exception as e:
            print(f"💥 하이브리드 Worker 오류 발생: {e}")
            continue
            
    print("👋 하이브리드 Inference Worker 종료.")

def process_batch(processor, batch_jobs, result_queue):
    """배치 작업 처리"""
    for job in batch_jobs:
        try:
            job_id, item_type, item_id, video_path = job
            arrays = processor.process_video_to_arrays(video_path)
            result_queue.put((job_id, item_type, item_id, video_path, arrays))
        except Exception as e:
            print(f"💥 배치 처리 오류: {e}")
            result_queue.put((job[0], job[1], job[2], job[3], None))



# +++ 신규: CPU 후처리를 위한 워커 함수 +++
def cpu_postprocess_worker(
    result_queue: mp.Queue, 
    output_dir: Path,
    keypoint_scale: int
):
    """결과를 받아 파일로 저장하는 CPU 워커"""
    while True:
        try:
            result = result_queue.get(timeout=5)
            if result is None:
                break
                
            job_id, item_type, item_id, video_path, arrays = result
            
            if arrays is None:
                print(f"⚠️ {item_type}{item_id:04d} (영상: {Path(video_path).name}) 처리 실패. 건너뜁니다.")
                continue

            start_time = time.time()
            item_dir = output_dir / f"{item_type}{item_id:04d}"
            item_dir.mkdir(parents=True, exist_ok=True)
            
            # npz 파일 저장
            jpeg_frames_dict = {f'frame_{i}': frame for i, frame in enumerate(arrays['jpeg_frames'])}
            np.savez_compressed(item_dir / "crop_images_jpeg.npz", **jpeg_frames_dict)

            # npy 파일 저장
            keypoints_scaled = np.round(arrays['keypoints'] * keypoint_scale).astype(np.int32)
            np.save(item_dir / "keypoints_scaled.npy", keypoints_scaled)
            np.save(item_dir / "scores.npy", arrays['scores'])
            
            # 메타데이터 저장
            metadata = {
                'item_type': item_type,
                'item_id': item_id,
                'video_path': str(video_path),
                'video_filename': Path(video_path).name,
                'frame_count': arrays['frame_count'],
                'shape_info': {
                    'frame_count': arrays['frame_count'],
                    'keypoints_scaled': list(keypoints_scaled.shape),
                    'scores': list(arrays['scores'].shape)
                },
                'keypoint_scale': keypoint_scale,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            with open(item_dir / "metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # print(f"💾 {item_type}{item_id:04d} 저장 완료 ({time.time() - start_time:.2f}초)")

        except queue.Empty:
            print("CPU Post-Processor: 결과 큐가 비었습니다. 종료합니다.")
            break
        except Exception as e:
            print(f"💥 CPU 후처리 오류: {e}")

class BatchProcessor:
    """폴더별 250개 단위 배치 처리기 (GPU 최적화 병렬 처리)"""
    
    def __init__(self, 
                 data_root: str = "data/1.Training",
                 output_dir: str = "sign_language_dataset",
                 batch_size: int = 250,
                 rtmw_model_name: str = "rtmw-x",
                 rtmw_config_path: str = "configs/wholebody_2d_keypoint/rtmpose/cocktail14/rtmw-x_8xb320-270e_cocktail14-384x288.py", 
                 direction: str = "F",
                 item_types: List[str] = ["WORD"],
                 num_cpu_workers: int = 4,
                 gpu_batch_size: int = 8,
                 yolo_device: str = "cpu",    # YOLO CPU 사용
                 pose_device: str = "xpu"):   # Pose GPU 유지
        
        self.data_root = Path(data_root).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.batch_size = batch_size
        self.rtmw_model_name = rtmw_model_name
        self.rtmw_config_path = rtmw_config_path
        self.direction = direction.upper()
        self.item_types = [t.upper() for t in item_types]
        self.keypoint_scale = 8
        
        # 디바이스 설정 추가
        self.yolo_device = yolo_device
        self.pose_device = pose_device
        
        # GPU 최적화 설정
        self.num_cpu_workers = min(num_cpu_workers, os.cpu_count())
        self.gpu_batch_size = gpu_batch_size
        
        # 유효성 검사
        if self.direction not in {'F', 'U', 'L', 'R', 'D'}:
            raise ValueError(f"Invalid direction: {direction}")
        if set(self.item_types) - {'WORD', 'SEN'}:
            raise ValueError(f"Invalid item types: {set(self.item_types) - {'WORD', 'SEN'}}")

        # 로깅 설정
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                            handlers=[logging.FileHandler('batch_processing.log'), logging.StreamHandler()])
        self.logger = logging.getLogger(__name__)
        
        # 출력 디렉토리
        self.video_output_dir = self.output_dir / "video_processing"
        self.hdf5_output_dir = self.output_dir / "hdf5_batches"
        self.video_output_dir.mkdir(parents=True, exist_ok=True)
        self.hdf5_output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("✅ GPU 최적화 배치 처리기 초기화 완료")
        self.logger.info(f"   - YOLO 디바이스: {yolo_device.upper()}")
        self.logger.info(f"   - Pose 디바이스: {pose_device.upper()}")
        self.logger.info(f"   - CPU 워커 수: {self.num_cpu_workers}")
        self.logger.info(f"   - GPU 배치 크기: {self.gpu_batch_size}")

    # ... (extract_item_info, collect_videos_by_folder, create_batches_by_folder 메서드는 변경 없음) ...
    def extract_item_info(self, video_path: Path) -> Optional[Tuple[str, int]]:
        filename = video_path.stem
        for item_type in self.item_types:
            pattern = rf'_{item_type}(\d{{4}})_'
            match = re.search(pattern, filename)
            if match:
                return item_type, int(match.group(1))
        return None

    def collect_videos_by_folder(self) -> Dict[str, List[Tuple[str, int, str]]]:
        videos_base_dir = self.data_root / "videos"
        folder_video_data = {}
        if not videos_base_dir.exists():
            self.logger.error(f"❌ videos 폴더 없음: {videos_base_dir}")
            return folder_video_data
        
        for sub_dir in videos_base_dir.iterdir():
            if not sub_dir.is_dir():
                continue
                
            folder_name = sub_dir.name
            video_data = []
            pattern = f"*_{self.direction}.mp4"
            for video_file in sub_dir.glob(pattern):
                item_info = self.extract_item_info(video_file)
                if item_info:
                    video_data.append((*item_info, str(video_file)))
            
            if video_data:
                video_data.sort(key=lambda x: (x[0], x[1]))
                folder_video_data[folder_name] = video_data
        
        return folder_video_data

    def create_batches_by_folder(self, folder_video_data: Dict[str, List[Tuple[str, int, str]]]) -> List[Dict]:
        all_batches = []
        batch_counter = 0
        for folder_name, video_data in folder_video_data.items():
            for i in range(0, len(video_data), self.batch_size):
                batch_data = video_data[i:i + self.batch_size]
                all_batches.append({
                    'batch_id': batch_counter, 
                    'folder_name': folder_name,
                    'folder_batch_idx': i // self.batch_size,
                    'data': batch_data,
                    'item_range': f"{batch_data[0][0]}{batch_data[0][1]:04d}~{batch_data[-1][0]}{batch_data[-1][1]:04d}"
                })
                batch_counter += 1
        self.logger.info(f"📦 전체 {len(all_batches)}개 배치 생성 완료")
        return all_batches

    # +++ 핵심 로직 변경: process_all_batches +++
    def process_all_batches(self, cleanup_intermediate: bool = False):
        """전체 배치 처리 파이프라인 (GPU 최적화 병렬 처리)"""
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

        # 1. 프로세스 간 통신을 위한 큐 생성
        task_queue = mp.Queue()
        result_queue = mp.Queue()
        
        # 2. 작업 큐에 모든 비디오 추가
        for i, (item_type, item_id, video_path) in enumerate(all_videos):
            task_queue.put((i, item_type, item_id, video_path))
        
        # 3. GPU/CPU 하이브리드 추론 프로세스 생성
        gpu_worker = mp.Process(target=gpu_inference_worker, args=(
            task_queue, result_queue, self.rtmw_model_name, self.rtmw_config_path, 
            self.gpu_batch_size, self.yolo_device, self.pose_device  # 디바이스 설정 전달
        ))
        gpu_worker.start()

        # CPU 후처리 프로세스 풀
        postprocess_pool = mp.Pool(self.num_cpu_workers, cpu_postprocess_worker, (
            result_queue, self.video_output_dir, self.keypoint_scale
        ))

        # 4. 진행률 표시 및 대기
        successful_keys_map = {}
        device_info = f"YOLO:{self.yolo_device.upper()}, Pose:{self.pose_device.upper()}"
        with tqdm(total=len(all_videos), desc=f"하이브리드 병렬 처리 ({device_info})") as pbar:
            for _ in range(len(all_videos)):
                job_id, item_type, item_id, video_path, arrays = result_queue.get()
                if arrays:
                    key = f"{item_type}{item_id:04d}"
                    for batch_info in self.create_batches_by_folder(folder_video_data):
                        if any(d[0] == item_type and d[1] == item_id for d in batch_info['data']):
                            if batch_info['batch_id'] not in successful_keys_map:
                                successful_keys_map[batch_info['batch_id']] = []
                            successful_keys_map[batch_info['batch_id']].append(key)
                            break
                pbar.update(1)

        # 5. 모든 워커 종료
        task_queue.put(None)
        for _ in range(self.num_cpu_workers):
             result_queue.put(None)

        gpu_worker.join()
        postprocess_pool.close()
        postprocess_pool.join()
        
        self.logger.info("✅ 모든 비디오 처리 완료. HDF5 생성을 시작합니다.")

        # 6. HDF5 배치 생성
        all_batches = self.create_batches_by_folder(folder_video_data)
        for batch_info in tqdm(all_batches, desc="HDF5 배치 생성"):
            batch_id = batch_info['batch_id']
            successful_keys = successful_keys_map.get(batch_id, [])
            if successful_keys:
                self.create_hdf5_batch(successful_keys, batch_info)
                if cleanup_intermediate:
                    self.cleanup_video_files(successful_keys, batch_info)

        self.logger.info("🎉 전체 폴더별 배치 처리 완료!")
        self.print_final_statistics(all_batches)

    # ... (create_hdf5_batch, cleanup_video_files, print_final_statistics, process_test_batch 메서드는 변경 없음) ...
    def create_hdf5_batch(self, successful_keys: List[str], batch_info: Dict):
        try:
            batch_id = batch_info['batch_id']
            folder_name = batch_info['folder_name']
            folder_batch_idx = batch_info['folder_batch_idx']
            
            self.logger.info(f"📦 배치 {batch_id} [{folder_name}] HDF5 생성 시작")
            
            types_str = "_".join(self.item_types)
            frames_h5_path = self.hdf5_output_dir / f"batch_{types_str}_{folder_name}_{folder_batch_idx:02d}_{self.direction}_frames.h5"
            poses_h5_path = self.hdf5_output_dir / f"batch_{types_str}_{folder_name}_{folder_batch_idx:02d}_{self.direction}_poses.h5"
            
            jpeg_vlen_dtype = h5py.vlen_dtype(np.uint8)

            with h5py.File(frames_h5_path, 'w') as f_frames, \
                 h5py.File(poses_h5_path, 'w') as f_poses:
                
                batch_metadata = {
                    'folder_name': folder_name,
                    'folder_batch_idx': folder_batch_idx,
                    'item_range': batch_info['item_range'],
                    'item_types': self.item_types,
                    'direction': self.direction,
                    'video_count': len(successful_keys),
                    'creation_time': str(datetime.now())
                }
                f_frames.attrs.update(batch_metadata)
                f_poses.attrs.update(batch_metadata)
                
                keys_to_process = sorted(successful_keys)

                for key in tqdm(keys_to_process, desc=f"HDF5 배치 {batch_id} [{folder_name}]", leave=False):
                    match = re.match(r"([A-Z]+)(\d+)", key)
                    if not match: continue
                    item_type, item_id = match.group(1), int(match.group(2))
                    item_dir = self.video_output_dir / key
                    
                    try:
                        with np.load(item_dir / "crop_images_jpeg.npz") as npz_file:
                            frame_keys_in_npz = sorted(npz_file.files, key=lambda k: int(k.split('_')[1]))
                            jpeg_frames = [npz_file[k] for k in frame_keys_in_npz]
                    except FileNotFoundError:
                        self.logger.warning(f"⚠️ {key}의 npz 파일을 찾을 수 없어 HDF5 생성에서 건너뜁니다.")
                        continue
                    
                    keypoints_scaled = np.load(item_dir / "keypoints_scaled.npy")
                    scores = np.load(item_dir / "scores.npy")
                    with open(item_dir / "metadata.json", 'r') as f: metadata = json.load(f)
                    
                    video_group = f"video_{item_type.lower()}{item_id:04d}"
                    
                    frame_group = f_frames.create_group(video_group)
                    frame_group.create_dataset("frames_jpeg", data=jpeg_frames, dtype=jpeg_vlen_dtype)
                    f_frames.create_dataset(f"{video_group}/metadata", data=json.dumps(metadata))
                    
                    pose_group = f_poses.create_group(video_group)
                    pose_group.create_dataset("keypoints_scaled", data=keypoints_scaled, compression='lzf')
                    pose_group.create_dataset("scores", data=scores, compression='lzf')
            
            self.logger.info(f"✅ 배치 {batch_id} [{folder_name}] HDF5 생성 완료")
            
        except Exception as e:
            self.logger.error(f"❌ 배치 {batch_info['batch_id']} HDF5 생성 실패: {e}", exc_info=True)

    def cleanup_video_files(self, keys: List[str], batch_info: Dict):
        for key in keys:
            item_dir = self.video_output_dir / key
            if item_dir.exists():
                shutil.rmtree(item_dir)
        self.logger.info(f"🧹 배치 {batch_info['batch_id']} 중간 파일 {len(keys)}개 정리 완료")

    def print_final_statistics(self, all_batches: List[Dict]):
        folder_stats = {}
        for batch_info in all_batches:
            folder_name = batch_info['folder_name']
            if folder_name not in folder_stats:
                folder_stats[folder_name] = {'batches': 0, 'videos': 0, 'types': set()}
            
            folder_stats[folder_name]['batches'] += 1
            folder_stats[folder_name]['videos'] += len(batch_info['data'])
            for item_type, _, _ in batch_info['data']:
                folder_stats[folder_name]['types'].add(item_type)
        
        self.logger.info("\n📊 최종 처리 통계:")
        self.logger.info("=" * 60)
        
        total_batches = 0
        total_videos = 0
        
        for folder_name, stats in folder_stats.items():
            types = ", ".join(sorted(list(stats['types'])))
            self.logger.info(f"📁 {folder_name}: {stats['batches']}개 배치, {stats['videos']}개 영상 ({types})")
            total_batches += stats['batches']
            total_videos += stats['videos']
        self.logger.info("=" * 60)
        self.logger.info(f"🎯 전체: {total_batches}개 배치, {total_videos}개 영상")

    def process_test_batch(self, test_count: int = 5):
        """테스트용 소규모 배치 처리 (병렬 아님) - 간단한 확인용"""
        self.logger.info("🧪 테스트 모드는 병렬 처리 없이 순차적으로 실행됩니다.")
        
        # 모델 초기화 (테스트는 단일 프로세스에서 진행)
        processor = StreamlinedVideoProcessor(
            rtmw_model_name=self.rtmw_model_name,
            rtmw_config_path=self.rtmw_config_path
        )
        
        folder_video_data = self.collect_videos_by_folder()
        if not folder_video_data:
            self.logger.error("❌ 처리할 영상이 없습니다")
            return
            
        test_folder = list(folder_video_data.keys())[0]
        video_data = folder_video_data[test_folder][:test_count]
        
        self.logger.info(f"🧪 테스트 배치 처리 시작 [{test_folder}] ({len(video_data)}개)")
        
        batch_info = {
            'batch_id': 999, 'folder_name': test_folder, 'folder_batch_idx': 0,
            'data': video_data,
            'item_range': f"{video_data[0][0]}{video_data[0][1]:04d}~{video_data[-1][0]}{video_data[-1][1]:04d}"
        }
        
        successful_keys = []
        for item_type, item_id, video_path in tqdm(video_data, desc="테스트 처리"):
            # 기존의 process_video 사용
            success = processor.process_video(item_type, item_id, video_path, self.video_output_dir)
            if success:
                successful_keys.append(f"{item_type}{item_id:04d}")
        
        if successful_keys:
            self.create_hdf5_batch(successful_keys, batch_info)
            self.logger.info("✅ 테스트 배치 HDF5 생성 완료.")
        
def main():
    """메인 실행 함수 (GPU 최적화 병렬 처리)"""
    mp.set_start_method('spawn', force=True)

    print("🚀 GPU 최적화 스트림라인 배치 처리기")
    print("=" * 60)
    
    print("\n처리할 아이템 타입을 선택하세요:")
    print("1. WORD만 처리\n2. SEN만 처리\n3. WORD + SEN 모두 처리 (기본값)")
    type_choice = input("타입 선택 (1-3, 기본값: 3): ").strip()
    item_types = {'1': ['WORD'], '2': ['SEN'], '3': ['WORD', 'SEN'], '': ['WORD', 'SEN']}.get(type_choice, ['WORD', 'SEN'])
    print(f"✅ 선택된 타입: {', '.join(item_types)}")

    print("\n사용할 RTMW 모델을 선택하세요:")
    print("1. RTMW-x (최고 성능, 기본값)")
    print("2. RTMW-l (균형)")
    
    model_choice = input("모델 선택 (1-2, 기본값: 1): ").strip()
    
    rtmw_model_map = {
        '1': 'rtmw-x',
        '2': 'rtmw-dw-x-l',
        '': 'rtmw-x'  # 기본값
    }
    
    rtmw_model_name = rtmw_model_map.get(model_choice, 'rtmw-x')
    print(f"✅ 선택된 모델: {rtmw_model_name}")

    print("\n처리할 방향을 선택하세요:")
    print("1. F (Front, 정면) - 기본값")
    print("2. U (Up, 위)")
    print("3. L (Left, 왼쪽)")
    print("4. R (Right, 오른쪽)")
    print("5. D (Down, 아래)")
    
    direction_choice = input("방향 선택 (1-5, 기본값: 1): ").strip()
    
    direction_map = {
        '1': 'F',
        '2': 'U',
        '3': 'L',
        '4': 'R',
        '5': 'D',
        '': 'F'  # 기본값
    }
    
    direction = direction_map.get(direction_choice, 'F')
    print(f"✅ 선택된 방향: {direction}")
    
    # 디바이스 설정 추가
    print("\n🔧 디바이스 설정:")
    print("1. GPU 전체 사용 (YOLO: GPU, Pose: GPU) - 최고 성능")
    print("2. 하이브리드 (YOLO: CPU, Pose: GPU) - GPU 메모리 절약, 권장")
    print("3. CPU 전체 사용 (YOLO: CPU, Pose: CPU) - GPU 없을 때")
    
    device_choice = input("디바이스 선택 (1-3, 기본값: 1): ").strip()
    device_map = {
        '1': ('xpu', 'xpu'),
        '2': ('cpu', 'xpu'),   # 기본값: YOLO CPU, Pose GPU
        '3': ('cpu', 'cpu'),
        '': ('xpu', 'xpu')     # 기본값
    }
    yolo_device, pose_device = device_map.get(device_choice, ('cpu', 'xpu'))
    print(f"✅ 디바이스 설정: YOLO={yolo_device.upper()}, Pose={pose_device.upper()}")
    
    # CPU 워커 수 자동 조정
    if yolo_device == 'cpu':
        default_cpu_workers = min(8, os.cpu_count())  # YOLO CPU 사용시 더 많은 워커
        print(f"💡 YOLO CPU 사용: CPU 워커 수를 {default_cpu_workers}개로 권장")
    else:
        default_cpu_workers = max(1, os.cpu_count() // 2)
    
    cpu_workers_input = input(f"\n사용할 CPU 워커 수를 입력하세요 (기본값: {default_cpu_workers}): ").strip()
    try:
        num_cpu_workers = int(cpu_workers_input) if cpu_workers_input else default_cpu_workers
    except ValueError:
        num_cpu_workers = default_cpu_workers
    print(f"✅ CPU 워커 수: {num_cpu_workers}")
    
    # GPU 배치 크기 자동 조정
    if yolo_device == 'cpu':
        default_gpu_batch = 16  # YOLO CPU 사용시 더 큰 배치
        print(f"💡 YOLO CPU 사용: GPU 배치 크기를 {default_gpu_batch}로 권장")
    else:
        default_gpu_batch = 8
        
    gpu_batch_input = input(f"\nGPU 배치 크기를 입력하세요 (기본값: {default_gpu_batch}): ").strip()
    try:
        gpu_batch_size = int(gpu_batch_input) if gpu_batch_input else default_gpu_batch
    except ValueError:
        gpu_batch_size = default_gpu_batch
    print(f"✅ GPU 배치 크기: {gpu_batch_size}")

    # 배치 처리기 초기화
    print(f"\n📥 하이브리드 모델 초기화 중... (YOLO: {yolo_device.upper()}, Pose: {pose_device.upper()})")
    try:
        batch_processor = BatchProcessor(
            rtmw_model_name=rtmw_model_name, 
            direction=direction,
            item_types=item_types,
            num_cpu_workers=num_cpu_workers,
            gpu_batch_size=gpu_batch_size,
            yolo_device=yolo_device,  # 디바이스 설정 추가
            pose_device=pose_device
        )
        print("✅ 하이브리드 초기화 완료!")
    except Exception as e:
        print(f"❌ 초기화 실패: {e}")
        return
    
    choice = '3'

    while True:
        print("\n처리 모드를 선택하세요(기본값: 3):")
        print("1. 테스트 처리 (5개 영상, 순차 실행)")
        print("2. 전체 병렬 처리 (권장)")
        print("3. 전체 병렬 처리 + 중간파일 정리")
        print("4. 영상 목록만 확인")
        print("5. 모델 정보 확인")
        print("0. 종료")
        
        user_input = input(f"선택 (0-5, Enter 입력 시 '{choice}' 실행): ").strip()
        
        # 4. 사용자가 입력한 경우에만 choice 변수를 업데이트
        if user_input:
            choice = user_input
        
        if choice == '1':
            batch_processor.process_test_batch(test_count=5)
        elif choice == '2':
            batch_processor.process_all_batches(cleanup_intermediate=False)
        elif choice == '3':
            batch_processor.process_all_batches(cleanup_intermediate=True)
        elif choice == '4':
            folder_video_data = batch_processor.collect_videos_by_folder()
            total_videos = sum(len(videos) for videos in folder_video_data.values())
            
            type_counts = {}
            for videos in folder_video_data.values():
                for item_type, _, _ in videos:
                    type_counts[item_type] = type_counts.get(item_type, 0) + 1
            type_stats = ", ".join([f"{t}:{c}개" for t, c in type_counts.items()])
            print(f"\n📊 총 {total_videos}개 {direction} 방향 영상 발견 ({type_stats}):")
            
            for folder_name, video_data in list(folder_video_data.items())[:3]:
                folder_type_counts = {}
                for item_type, _, _ in video_data:
                    folder_type_counts[item_type] = folder_type_counts.get(item_type, 0) + 1
                
                folder_type_stats = ", ".join([f"{t}:{c}" for t, c in folder_type_counts.items()])
                print(f"  📁 {folder_name}: {len(video_data)}개 ({folder_type_stats})")
                
                for i, (item_type, item_id, video_path) in enumerate(video_data[:5]):
                    print(f"    {i+1}. {item_type}{item_id:04d} - {Path(video_path).name}")
                if len(video_data) > 5:
                    print(f"    ... 외 {len(video_data) - 5}개")
            
            if len(folder_video_data) > 3:
                remaining_folders = len(folder_video_data) - 3
                remaining_videos = sum(len(videos) for videos in list(folder_video_data.values())[3:])
                print(f"  ... 외 {remaining_folders}개 폴더 ({remaining_videos}개 영상)")
        elif choice == '5':
            print(f"\n📋 현재 모델 정보:")
            print(f"  - 처리 타입: {', '.join(item_types)}")
            print(f"  - RTMW 모델: {rtmw_model_name}")
            print(f"  - YOLO 모델: {YOLO_MODEL_CONFIG['filename']}")
            print(f"  - 처리 방향: {direction}")
            print(f"  - 모델 디렉토리: {MODELS_DIR}")
            
            # 모델 파일 존재 확인
            yolo_path = Path(MODELS_DIR) / YOLO_MODEL_CONFIG["filename"]
            print(f"  - YOLO 파일 존재: {'✅' if yolo_path.exists() else '❌'}")
            
            for config in RTMW_MODEL_OPTIONS:
                if rtmw_model_name in config["filename"]:
                    rtmw_path = Path(config["path"])
                    print(f"  - RTMW 파일 존재: {'✅' if rtmw_path.exists() else '❌'}")
                    if rtmw_path.exists():
                        size_mb = rtmw_path.stat().st_size / (1024*1024)
                        print(f"    크기: {size_mb:.1f} MB")
                    break
        elif choice == '0':
            print("👋 종료합니다.")
            break
        else:
            print("❌ 잘못된 선택입니다.")


if __name__ == "__main__":
    main()