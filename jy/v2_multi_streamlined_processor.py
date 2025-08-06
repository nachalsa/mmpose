#!/usr/bin/env python3
"""
최적화된 비디오 처리기 - CPU YOLO + GPU 배치 RTMW 지원
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
import queue
import threading

# 설정 및 MMPose 관련 임포트
from config import MODELS_DIR, YOLO_MODEL_CONFIG, RTMW_MODEL_OPTIONS

# RTMW 전처리 함수들 (video_processor_yolo11l.py에서 가져옴)
def bbox_xyxy2cs(bbox: np.ndarray, padding: float = 1.10) -> Tuple[np.ndarray, np.ndarray]:
    """바운딩박스를 center, scale로 변환"""
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

class OptimizedInferencer:
    """CPU YOLO + GPU 배치 RTMW 추론기"""
    
    def __init__(self, 
                 rtmw_config_path: str,
                 rtmw_model_path: str,
                 yolo_device: str = 'cpu',  # 'cpu' or 'xpu'
                 pose_device: str = 'xpu',
                 batch_size: int = 8):
        
        self.logger = logging.getLogger(__name__)
        self.yolo_device = yolo_device
        self.pose_device = pose_device
        self.batch_size = batch_size
        
        # YOLO 모델 초기화
        from ultralytics import YOLO
        yolo_model_path = self._ensure_yolo_model()
        self.yolo_model = YOLO(yolo_model_path)
        
        self.logger.info(f"✅ YOLO 로딩 완료 (디바이스: {yolo_device})")
        
        # RTMW 모델 초기화 (GPU에서 배치 처리)
        self._init_rtmw_model(rtmw_config_path, rtmw_model_path)
        
        self.logger.info(f"✅ RTMW 로딩 완료 (디바이스: {pose_device}, 배치크기: {batch_size})")

    def _ensure_yolo_model(self) -> str:
        """YOLO 모델 파일 확인 및 다운로드"""
        yolo_config = YOLO_MODEL_CONFIG
        model_path = Path(MODELS_DIR) / yolo_config["filename"]
        
        if model_path.exists():
            return str(model_path)
        
        self.logger.info(f"📥 YOLO 모델 다운로드: {yolo_config['filename']}")
        Path(MODELS_DIR).mkdir(parents=True, exist_ok=True)
        
        try:
            from ultralytics import YOLO
            temp_model = YOLO(yolo_config["filename"])
            cache_dir = Path.home() / '.cache' / 'ultralytics'
            
            for weights_dir in [cache_dir, cache_dir / 'weights']:
                if weights_dir.exists():
                    for model_file in weights_dir.glob(yolo_config["filename"]):
                        shutil.copy2(model_file, model_path)
                        self.logger.info(f"✅ YOLO 모델 복사 완료: {model_path}")
                        return str(model_path)
            
            return yolo_config["filename"]
        except Exception as e:
            self.logger.warning(f"⚠️ YOLO 모델 다운로드 실패: {e}")
            return yolo_config["filename"]

    def _init_rtmw_model(self, config_path: str, model_path: str):
        """RTMW 모델 초기화"""
        try:
            from mmpose.apis import MMPoseInferencer
            # MMPose 버전별 호환성 처리
            try:
                # 최신 버전 (pose2d 파라미터 사용)
                self.rtmw_inferencer = MMPoseInferencer(
                    pose2d=config_path,
                    pose2d_weights=model_path,
                    device=self.pose_device
                )
            except TypeError:
                try:
                    # 이전 버전 (model 파라미터 사용)
                    self.rtmw_inferencer = MMPoseInferencer(
                        model=config_path,
                        weights=model_path,
                        device=self.pose_device
                    )
                except TypeError:
                    # 가장 기본적인 초기화 방식
                    self.rtmw_inferencer = MMPoseInferencer(
                        config_path, 
                        model_path, 
                        device=self.pose_device
                    )
        except ImportError:
            self.logger.error("❌ mmpose를 설치해주세요: pip install mmpose")
            raise
        except Exception as e:
            self.logger.error(f"❌ RTMW 모델 초기화 실패: {e}")
            self.logger.info("💡 MMPose 버전을 확인해주세요: pip show mmpose")
            raise

    def detect_persons_cpu(self, frames: List[np.ndarray]) -> List[List[List[float]]]:
        """CPU에서 YOLO 객체 탐지 (여러 프레임 처리)"""
        all_bboxes = []
        
        for frame in frames:
            try:
                # CPU에서 YOLO 실행
                results = self.yolo_model(frame, device=self.yolo_device, verbose=False)
                
                person_bboxes = []
                for result in results:
                    if hasattr(result, 'boxes') and result.boxes is not None:
                        boxes = result.boxes
                        for i, cls in enumerate(boxes.cls):
                            if int(cls) == 0:  # person class
                                bbox = boxes.xyxy[i].cpu().numpy().tolist()
                                conf = float(boxes.conf[i])
                                if conf > 0.5:  # 신뢰도 임계값
                                    person_bboxes.append(bbox)
                
                # 가장 큰 박스만 선택 (면적 기준)
                if person_bboxes:
                    areas = [(box[2] - box[0]) * (box[3] - box[1]) for box in person_bboxes]
                    max_idx = np.argmax(areas)
                    all_bboxes.append([person_bboxes[max_idx]])
                else:
                    all_bboxes.append([])
                    
            except Exception as e:
                self.logger.warning(f"YOLO 탐지 실패: {e}")
                all_bboxes.append([])
        
        return all_bboxes

    def crop_batch_images(self, frames: List[np.ndarray], bboxes_list: List[List[List[float]]]) -> Tuple[List[np.ndarray], List[bool]]:
        """배치로 이미지 크롭"""
        cropped_images = []
        valid_mask = []
        
        for frame, bboxes in zip(frames, bboxes_list):
            if bboxes:
                bbox = bboxes[0]  # 첫 번째 (가장 큰) 박스
                cropped = self._crop_person_image_rtmw(frame, bbox)
                if cropped is not None:
                    cropped_images.append(cropped)
                    valid_mask.append(True)
                else:
                    valid_mask.append(False)
            else:
                valid_mask.append(False)
        
        return cropped_images, valid_mask

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
                image, warp_mat, (input_width, input_height),
                flags=cv2.INTER_LINEAR
            )
            return cropped_image
            
        except Exception as e:
            self.logger.warning(f"⚠️ RTMW 전처리 실패: {e}")
            return None

    def estimate_pose_batch(self, cropped_images: List[np.ndarray]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """배치로 포즈 추정"""
        if not cropped_images:
            return [], []
        
        try:
            # MMPose 배치 추론
            batch_results = self.rtmw_inferencer(cropped_images, return_vis=False)
            
            keypoints_list = []
            scores_list = []
            
            for result in batch_results:
                if 'predictions' in result and len(result['predictions']) > 0:
                    pred = result['predictions'][0]  # 첫 번째 사람
                    keypoints = pred['keypoints']  # shape: (133, 2)
                    keypoint_scores = pred.get('keypoint_scores', np.ones(len(keypoints)))
                    
                    keypoints_list.append(keypoints)
                    scores_list.append(keypoint_scores)
                else:
                    # 실패한 경우 더미 데이터
                    keypoints_list.append(np.zeros((133, 2)))
                    scores_list.append(np.zeros(133))
            
            return keypoints_list, scores_list
            
        except Exception as e:
            self.logger.error(f"배치 포즈 추정 실패: {e}")
            # 실패 시 더미 데이터 반환
            dummy_keypoints = [np.zeros((133, 2)) for _ in cropped_images]
            dummy_scores = [np.zeros(133) for _ in cropped_images]
            return dummy_keypoints, dummy_scores

class OptimizedVideoProcessor:
    """최적화된 비디오 처리기 (CPU YOLO + 배치 RTMW)"""
    
    def __init__(self,
                 rtmw_config_path: str = "configs/wholebody_2d_keypoint/rtmpose/cocktail14/rtmw-x_8xb320-270e_cocktail14-384x288.py",
                 rtmw_model_name: str = "rtmw-x",
                 yolo_device: str = 'cpu',
                 batch_size: int = 8):
        
        self.logger = logging.getLogger(__name__)
        self.keypoint_scale = 8
        self.batch_size = batch_size
        
        # 경로 설정
        try:
            base_dir = Path(__file__).parent.parent
        except NameError:
            base_dir = Path.cwd().parent
        rtmw_config_path = str(base_dir / rtmw_config_path)
        
        # 모델 다운로드 확인
        rtmw_model_path = self._ensure_rtmw_model(rtmw_model_name)
        
        # 추론기 초기화
        self.inferencer = OptimizedInferencer(
            rtmw_config_path=rtmw_config_path,
            rtmw_model_path=rtmw_model_path,
            yolo_device=yolo_device,
            pose_device='xpu',
            batch_size=batch_size
        )
        
        self.logger.info(f"✅ 최적화된 비디오 처리기 초기화 완료 (YOLO: {yolo_device}, 배치: {batch_size})")

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
        
        # 다운로드 로직 (기존과 동일)
        self.logger.info(f"📥 RTMW 모델 다운로드: {rtmw_config['filename']}")
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            download_url = rtmw_config["url"]
            if download_url:
                urllib.request.urlretrieve(download_url, model_path)
                if model_path.exists() and model_path.stat().st_size > 1024 * 1024:
                    self.logger.info(f"✅ RTMW 모델 다운로드 완료: {model_path}")
                    return str(model_path)
            
        except Exception as e:
            self.logger.error(f"❌ RTMW 모델 다운로드 실패: {e}")
        
        return str(model_path)

    def process_video_to_arrays(self, video_path: str) -> Optional[Dict[str, Union[List[np.ndarray], np.ndarray, int]]]:
        """비디오를 배치 단위로 처리"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self.logger.error(f"❌ 비디오 열기 실패: {video_path}")
            return None
        
        try:
            all_jpeg_frames, all_keypoints, all_scores = [], [], []
            frame_buffer = []
            
            # 프레임 배치 단위로 처리
            while True:
                ret, frame = cap.read()
                if not ret:
                    # 마지막 배치 처리
                    if frame_buffer:
                        self._process_frame_batch(frame_buffer, all_jpeg_frames, all_keypoints, all_scores)
                    break
                
                frame_buffer.append(frame)
                
                # 배치 크기에 도달하면 처리
                if len(frame_buffer) >= self.batch_size:
                    self._process_frame_batch(frame_buffer, all_jpeg_frames, all_keypoints, all_scores)
                    frame_buffer = []
            
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
            cap.release()

    def _process_frame_batch(self, frames: List[np.ndarray], 
                           all_jpeg_frames: List[np.ndarray],
                           all_keypoints: List[np.ndarray], 
                           all_scores: List[np.ndarray]):
        """프레임 배치 처리"""
        try:
            # 1. CPU YOLO 탐지 (배치)
            bboxes_list = self.inferencer.detect_persons_cpu(frames)
            
            # 2. 이미지 크롭 (배치)
            cropped_images, valid_mask = self.inferencer.crop_batch_images(frames, bboxes_list)
            
            if not cropped_images:
                return
            
            # 3. GPU RTMW 포즈 추정 (배치)
            keypoints_list, scores_list = self.inferencer.estimate_pose_batch(cropped_images)
            
            # 4. 결과 저장
            crop_idx = 0
            for i, (frame, is_valid) in enumerate(zip(frames, valid_mask)):
                if is_valid and crop_idx < len(cropped_images):
                    # JPEG 인코딩
                    ret_jpg, encoded_jpg = cv2.imencode('.jpg', cropped_images[crop_idx], 
                                                       [cv2.IMWRITE_JPEG_QUALITY, 90])
                    if ret_jpg:
                        all_jpeg_frames.append(encoded_jpg)
                        all_keypoints.append(keypoints_list[crop_idx])
                        all_scores.append(scores_list[crop_idx])
                    
                    crop_idx += 1
                    
        except Exception as e:
            self.logger.warning(f"배치 처리 실패: {e}")

    def process_video(self, item_type: str, item_id: int, video_path: str, output_dir: Path) -> bool:
        """비디오 처리 및 저장"""
        try:
            self.logger.info(f"🎬 처리 중: {item_type}{item_id:04d} - {Path(video_path).name}")
            start_time = time.time()
            
            arrays = self.process_video_to_arrays(video_path)
            if arrays is None:
                return False
            
            processing_time = time.time() - start_time
            
            # 결과 저장
            item_dir = output_dir / f"{item_type}{item_id:04d}"
            item_dir.mkdir(parents=True, exist_ok=True)
            
            # JPEG 프레임 저장
            jpeg_frames_dict = {f'frame_{i}': frame for i, frame in enumerate(arrays['jpeg_frames'])}
            np.savez_compressed(item_dir / "crop_images_jpeg.npz", **jpeg_frames_dict)

            # 키포인트 저장
            keypoints_scaled = np.round(arrays['keypoints'] * self.keypoint_scale).astype(np.int32)
            np.save(item_dir / "keypoints_scaled.npy", keypoints_scaled)
            np.save(item_dir / "scores.npy", arrays['scores'])
            
            # 메타데이터 저장
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

# 병렬 처리를 위한 워커 함수들
def optimized_inference_worker(
    task_queue: mp.Queue, 
    result_queue: mp.Queue, 
    rtmw_model_name: str, 
    rtmw_config_path: str,
    yolo_device: str,
    batch_size: int
):
    """최적화된 추론 워커 (CPU YOLO + GPU 배치 RTMW)"""
    print(f"🚀 최적화된 Inference Worker 시작... (YOLO: {yolo_device}, 배치: {batch_size})")
    
    processor = OptimizedVideoProcessor(
        rtmw_model_name=rtmw_model_name,
        rtmw_config_path=rtmw_config_path,
        yolo_device=yolo_device,
        batch_size=batch_size
    )
    
    while True:
        try:
            job = task_queue.get(timeout=5)
            if job is None:
                break
            
            job_id, item_type, item_id, video_path = job
            arrays = processor.process_video_to_arrays(video_path)
            result_queue.put((job_id, item_type, item_id, video_path, arrays))
                
        except queue.Empty:
            print("최적화된 Worker: 작업 큐가 비었습니다. 종료합니다.")
            break
        except Exception as e:
            print(f"💥 최적화된 Worker 오류: {e}")
            continue
            
    print("👋 최적화된 Inference Worker 종료.")

def cpu_postprocess_worker(result_queue: mp.Queue, output_dir: Path, keypoint_scale: int):
    """결과 후처리 워커 (기존과 동일)"""
    while True:
        try:
            result = result_queue.get(timeout=5)
            if result is None:
                break
                
            job_id, item_type, item_id, video_path, arrays = result
            
            if arrays is None:
                print(f"⚠️ {item_type}{item_id:04d} 처리 실패")
                continue

            item_dir = output_dir / f"{item_type}{item_id:04d}"
            item_dir.mkdir(parents=True, exist_ok=True)
            
            # 파일 저장
            jpeg_frames_dict = {f'frame_{i}': frame for i, frame in enumerate(arrays['jpeg_frames'])}
            np.savez_compressed(item_dir / "crop_images_jpeg.npz", **jpeg_frames_dict)

            keypoints_scaled = np.round(arrays['keypoints'] * keypoint_scale).astype(np.int32)
            np.save(item_dir / "keypoints_scaled.npy", keypoints_scaled)
            np.save(item_dir / "scores.npy", arrays['scores'])
            
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

        except queue.Empty:
            print("CPU Post-Processor 종료")
            break
        except Exception as e:
            print(f"💥 CPU 후처리 오류: {e}")

class OptimizedBatchProcessor:
    """최적화된 배치 처리기 (CPU YOLO + GPU 배치 RTMW)"""
    
    def __init__(self, 
                 data_root: str = "data/1.Training",
                 output_dir: str = "sign_language_dataset",
                 batch_size: int = 250,
                 rtmw_model_name: str = "rtmw-x",
                 rtmw_config_path: str = "configs/wholebody_2d_keypoint/rtmpose/cocktail14/rtmw-x_8xb320-270e_cocktail14-384x288.py", 
                 direction: str = "F",
                 item_types: List[str] = ["WORD"],
                 num_cpu_workers: int = 4,
                 yolo_device: str = 'cpu',  # 새로 추가
                 inference_batch_size: int = 8):  # 새로 추가
        
        self.data_root = Path(data_root).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.batch_size = batch_size
        self.rtmw_model_name = rtmw_model_name
        self.rtmw_config_path = rtmw_config_path
        self.direction = direction.upper()
        self.item_types = [t.upper() for t in item_types]
        self.keypoint_scale = 8
        self.num_cpu_workers = min(num_cpu_workers, os.cpu_count())
        self.yolo_device = yolo_device  # CPU/GPU 선택
        self.inference_batch_size = inference_batch_size  # 배치 크기
        
        # 검증
        if self.direction not in {'F', 'U', 'L', 'R', 'D'}:
            raise ValueError(f"Invalid direction: {direction}")
        if set(self.item_types) - {'WORD', 'SEN'}:
            raise ValueError(f"Invalid item types: {set(self.item_types) - {'WORD', 'SEN'}}")

        # 로깅 설정
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                            handlers=[logging.FileHandler('optimized_batch_processing.log'), logging.StreamHandler()])
        self.logger = logging.getLogger(__name__)
        
        # 출력 디렉토리
        self.video_output_dir = self.output_dir / "video_processing"
        self.hdf5_output_dir = self.output_dir / "hdf5_batches"
        self.video_output_dir.mkdir(parents=True, exist_ok=True)
        self.hdf5_output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("✅ 최적화된 배치 처리기 초기화 완료")
        self.logger.info(f"   - YOLO 디바이스: {yolo_device}")
        self.logger.info(f"   - 추론 배치 크기: {inference_batch_size}")
        self.logger.info(f"   - CPU 워커 수: {self.num_cpu_workers}")

    def extract_item_info(self, video_path: Path) -> Optional[Tuple[str, int]]:
        """비디오 파일에서 아이템 정보 추출"""
        filename = video_path.stem
        for item_type in self.item_types:
            pattern = rf'_{item_type}(\d{{4}})_'
            match = re.search(pattern, filename)
            if match:
                return item_type, int(match.group(1))
        return None

    def collect_videos_by_folder(self) -> Dict[str, List[Tuple[str, int, str]]]:
        """폴더별로 비디오 수집"""
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
        """폴더별로 배치 생성"""
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

    def process_all_batches(self, cleanup_intermediate: bool = False):
        """전체 배치 처리 (최적화된 파이프라인)"""
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

        self.logger.info(f"🚀 최적화된 파이프라인 시작 (YOLO: {self.yolo_device}, 배치: {self.inference_batch_size})")

        # 프로세스 간 통신 큐
        task_queue = mp.Queue()
        result_queue = mp.Queue()
        
        # 작업 큐에 모든 비디오 추가
        for i, (item_type, item_id, video_path) in enumerate(all_videos):
            task_queue.put((i, item_type, item_id, video_path))
        
        # 최적화된 추론 워커 시작 (CPU YOLO + GPU 배치 RTMW)
        inference_worker = mp.Process(target=optimized_inference_worker, args=(
            task_queue, result_queue, 
            self.rtmw_model_name, self.rtmw_config_path,
            self.yolo_device, self.inference_batch_size
        ))
        inference_worker.start()

        # CPU 후처리 워커 풀
        postprocess_pool = mp.Pool(self.num_cpu_workers, cpu_postprocess_worker, (
            result_queue, self.video_output_dir, self.keypoint_scale
        ))

        # 진행률 표시 및 결과 수집
        successful_keys_map = {}
        with tqdm(total=len(all_videos), desc="최적화된 비디오 처리") as pbar:
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

        # 워커 정리
        task_queue.put(None)
        for _ in range(self.num_cpu_workers):
             result_queue.put(None)

        inference_worker.join()
        postprocess_pool.close()
        postprocess_pool.join()
        
        self.logger.info("✅ 최적화된 비디오 처리 완료. HDF5 생성을 시작합니다.")

        # HDF5 배치 생성
        all_batches = self.create_batches_by_folder(folder_video_data)
        for batch_info in tqdm(all_batches, desc="HDF5 배치 생성"):
            batch_id = batch_info['batch_id']
            successful_keys = successful_keys_map.get(batch_id, [])
            if successful_keys:
                self.create_hdf5_batch(successful_keys, batch_info)
                if cleanup_intermediate:
                    self.cleanup_video_files(successful_keys, batch_info)

        self.logger.info("🎉 최적화된 전체 배치 처리 완료!")
        self.print_final_statistics(all_batches)

    def create_hdf5_batch(self, successful_keys: List[str], batch_info: Dict):
        """HDF5 배치 생성 (기존과 동일)"""
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
                
                # HDF5 메타데이터 (기존과 동일한 구조 유지)
                batch_metadata = {
                    'folder_name': folder_name,
                    'folder_batch_idx': folder_batch_idx,
                    'item_range': batch_info['item_range'],
                    'item_types': self.item_types,
                    'direction': self.direction,
                    'video_count': len(successful_keys),
                    'creation_time': str(datetime.now())
                    # 최적화 관련 정보는 제거 (기존과 동일한 구조 유지)
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
        """중간 파일 정리"""
        for key in keys:
            item_dir = self.video_output_dir / key
            if item_dir.exists():
                shutil.rmtree(item_dir)
        self.logger.info(f"🧹 배치 {batch_info['batch_id']} 중간 파일 {len(keys)}개 정리 완료")

    def print_final_statistics(self, all_batches: List[Dict]):
        """최종 통계 출력"""
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
        self.logger.info(f"🔧 설정: YOLO({self.yolo_device}), 배치크기({self.inference_batch_size})")

    def process_test_batch(self, test_count: int = 5):
        """테스트 배치 처리"""
        self.logger.info("🧪 최적화된 테스트 모드 실행")
        
        processor = OptimizedVideoProcessor(
            rtmw_model_name=self.rtmw_model_name,
            rtmw_config_path=self.rtmw_config_path,
            yolo_device=self.yolo_device,
            batch_size=self.inference_batch_size
        )
        
        folder_video_data = self.collect_videos_by_folder()
        if not folder_video_data:
            self.logger.error("❌ 처리할 영상이 없습니다")
            return
            
        test_folder = list(folder_video_data.keys())[0]
        video_data = folder_video_data[test_folder][:test_count]
        
        self.logger.info(f"🧪 테스트 배치 처리 [{test_folder}] ({len(video_data)}개)")
        
        batch_info = {
            'batch_id': 999, 'folder_name': test_folder, 'folder_batch_idx': 0,
            'data': video_data,
            'item_range': f"{video_data[0][0]}{video_data[0][1]:04d}~{video_data[-1][0]}{video_data[-1][1]:04d}"
        }
        
        successful_keys = []
        for item_type, item_id, video_path in tqdm(video_data, desc="최적화 테스트"):
            success = processor.process_video(item_type, item_id, video_path, self.video_output_dir)
            if success:
                successful_keys.append(f"{item_type}{item_id:04d}")
        
        if successful_keys:
            self.create_hdf5_batch(successful_keys, batch_info)
            self.logger.info("✅ 최적화된 테스트 배치 완료")

def main():
    """메인 실행 함수 (최적화된 버전)"""
    mp.set_start_method('spawn', force=True)

    print("🚀 최적화된 스트림라인 배치 처리기 (CPU YOLO + GPU 배치 RTMW)")
    print("=" * 70)
    
    # 아이템 타입 선택
    print("\n처리할 아이템 타입을 선택하세요:")
    print("1. WORD만 처리\n2. SEN만 처리\n3. WORD + SEN 모두 처리 (기본값)")
    type_choice = input("타입 선택 (1-3, 기본값: 3): ").strip()
    item_types = {'1': ['WORD'], '2': ['SEN'], '3': ['WORD', 'SEN'], '': ['WORD', 'SEN']}.get(type_choice, ['WORD', 'SEN'])
    print(f"✅ 선택된 타입: {', '.join(item_types)}")

    # RTMW 모델 선택
    print("\n사용할 RTMW 모델을 선택하세요:")
    print("1. RTMW-x (최고 성능, 기본값)")
    print("2. RTMW-l (균형)")
    
    model_choice = input("모델 선택 (1-2, 기본값: 1): ").strip()
    rtmw_model_map = {'1': 'rtmw-x', '2': 'rtmw-dw-x-l', '': 'rtmw-x'}
    rtmw_model_name = rtmw_model_map.get(model_choice, 'rtmw-x')
    print(f"✅ 선택된 모델: {rtmw_model_name}")

    # 방향 선택
    print("\n처리할 방향을 선택하세요:")
    print("1. F (Front, 정면) - 기본값")
    print("2. U (Up, 위)\n3. L (Left, 왼쪽)\n4. R (Right, 오른쪽)\n5. D (Down, 아래)")
    
    direction_choice = input("방향 선택 (1-5, 기본값: 1): ").strip()
    direction_map = {'1': 'F', '2': 'U', '3': 'L', '4': 'R', '5': 'D', '': 'F'}
    direction = direction_map.get(direction_choice, 'F')
    print(f"✅ 선택된 방향: {direction}")
    
    # YOLO 디바이스 선택 (새로 추가!)
    print("\n🔥 YOLO 실행 디바이스를 선택하세요:")
    print("1. CPU (안정적, 메모리 절약)")
    print("2. GPU (빠름, 메모리 많이 사용) - 기본값")
    
    yolo_device_choice = input("YOLO 디바이스 (1-2, 기본값: 2): ").strip()
    yolo_device_map = {'1': 'cpu', '2': 'xpu', '': 'xpu'}
    yolo_device = yolo_device_map.get(yolo_device_choice, 'xpu')
    print(f"✅ YOLO 디바이스: {yolo_device}")
    
    # 추론 배치 크기 선택 (새로 추가!)
    print("\n⚡ 추론 배치 크기를 선택하세요:")
    print("1. 작음 (4) - 메모리 부족 시")
    print("2. 중간 (8) - 기본값")  
    print("3. 큼 (16) - 고성능 GPU")
    print("4. 매우 큼 (32) - 최고 성능")
    
    batch_choice = input("배치 크기 (1-4, 기본값: 2): ").strip()
    batch_map = {'1': 4, '2': 8, '3': 16, '4': 32, '': 8}
    inference_batch_size = batch_map.get(batch_choice, 8)
    print(f"✅ 추론 배치 크기: {inference_batch_size}")
    
    # CPU 워커 수
    default_cpu_workers = max(1, os.cpu_count() // 2)
    cpu_workers_input = input(f"\n사용할 CPU 워커 수 (기본값: {default_cpu_workers}): ").strip()
    try:
        num_cpu_workers = int(cpu_workers_input) if cpu_workers_input else default_cpu_workers
    except ValueError:
        num_cpu_workers = default_cpu_workers
    print(f"✅ CPU 워커 수: {num_cpu_workers}")

    # 최적화된 배치 처리기 초기화
    print("\n📥 최적화된 모델 초기화 중...")
    try:
        batch_processor = OptimizedBatchProcessor(
            rtmw_model_name=rtmw_model_name, 
            direction=direction,
            item_types=item_types,
            num_cpu_workers=num_cpu_workers,
            yolo_device=yolo_device,
            inference_batch_size=inference_batch_size
        )
        print("✅ 최적화된 초기화 완료!")
    except Exception as e:
        print(f"❌ 초기화 실패: {e}")
        return

    choice = '3'  # 기본값

    while True:
        print("\n🚀 최적화된 처리 모드를 선택하세요:")
        print("1. 테스트 처리 (5개 영상)")
        print("2. 전체 최적화 처리")
        print("3. 전체 최적화 처리 + 중간파일 정리 (기본값)")
        print("4. 영상 목록만 확인")
        print("5. 최적화 설정 확인")
        print("0. 종료")
        
        user_input = input(f"선택 (0-5, Enter 시 '{choice}' 실행): ").strip()
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
            print(f"\n📊 총 {total_videos}개 {direction} 방향 영상 ({type_stats}):")
            
            for folder_name, video_data in list(folder_video_data.items())[:3]:
                print(f"  📁 {folder_name}: {len(video_data)}개")
                for i, (item_type, item_id, video_path) in enumerate(video_data[:3]):
                    print(f"    {i+1}. {item_type}{item_id:04d} - {Path(video_path).name}")
                if len(video_data) > 3:
                    print(f"    ... 외 {len(video_data) - 3}개")
            
        elif choice == '5':
            print(f"\n📋 현재 최적화 설정:")
            print(f"  🎯 처리 타입: {', '.join(item_types)}")
            print(f"  🤖 RTMW 모델: {rtmw_model_name}")
            print(f"  🔍 YOLO 디바이스: {yolo_device}")
            print(f"  ⚡ 추론 배치 크기: {inference_batch_size}")
            print(f"  🖥️ CPU 워커 수: {num_cpu_workers}")
            print(f"  📍 처리 방향: {direction}")
            
            # 성능 예측 정보
            if yolo_device == 'cpu':
                print(f"\n💡 성능 예측:")
                print(f"  - CPU YOLO: 안정적이지만 느림")
                print(f"  - GPU 배치 RTMW: 높은 처리량")
                print(f"  - 메모리 사용량: 중간 수준")
            else:
                print(f"\n💡 성능 예측:")
                print(f"  - GPU YOLO: 빠르지만 메모리 사용")
                print(f"  - GPU 배치 RTMW: 최고 성능")
                print(f"  - 메모리 사용량: 높음")
                
        elif choice == '0':
            print("👋 최적화된 처리기를 종료합니다.")
            break
        else:
            print("❌ 잘못된 선택입니다.")

if __name__ == "__main__":
    main()