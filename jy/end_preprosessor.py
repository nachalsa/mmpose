#!/usr/bin/env python3
"""
개선된 배치 Fast Multi-ONNX 처리기 - GPU 병렬처리 + 프레임 배치 로드
WORD/SEN 네이밍 규칙 + 패딩 크롭 + 133개 키포인트 (얼굴+손 포함)
GPU 병렬처리와 180 프레임 배치로 미리 로드 기능 추가
"""

import os
import warnings
import cv2
import h5py
import time
import json
import torch
import torch.multiprocessing as mp
import queue
import logging
import threading
import traceback
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional, Union
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
from dataclasses import dataclass
from collections import deque
from datetime import datetime
import psutil
import GPUtil
import argparse
import sys
import re

# PyTorch 멀티프로세싱 설정
mp.set_start_method('spawn', force=True)
torch.multiprocessing.set_sharing_strategy('file_system')

# ONNX Runtime 경고 억제
os.environ.setdefault('ORT_LOGGING_LEVEL', '3')

from onnx_inferencer import YOLO11LRTMWONNXInferencer as ONNXInferencer

# ===== RTMW 전처리 함수들 (두 번째 코드에서 가져옴) =====

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

def crop_person_image_rtmw(image: np.ndarray, bbox: List[float]) -> Optional[np.ndarray]:
    """RTMW 방식으로 사람 이미지 크롭 (패딩 + aspect ratio 고정)"""
    try:
        # RTMW 설정: width=288, height=384
        input_width, input_height = 288, 384
        
        # 1. bbox를 center, scale로 변환
        bbox_array = np.array(bbox, dtype=np.float32)
        center, scale = bbox_xyxy2cs(bbox_array)
        
        # 2. aspect ratio 고정 (width/height = 288/384 = 0.75)
        aspect_ratio = input_width / input_height  # 0.75
        scale = fix_aspect_ratio(scale, aspect_ratio)
        
        # 3. 아핀 변환 매트릭스 계산
        warp_mat = get_warp_matrix(
            center=center,
            scale=scale,
            rot=0.0,  # 회전 없음
            output_size=(input_width, input_height)
        )
        
        # 4. 아핀 변환 적용
        cropped_image = cv2.warpAffine(
            image, 
            warp_mat, 
            (input_width, input_height), 
            flags=cv2.INTER_LINEAR
        )
        
        # 5. 크기 검증
        h, w = cropped_image.shape[:2]
        if h == input_height and w == input_width:
            return cropped_image
        else:
            print(f"⚠️ 크기 오류: {h}x{w}, 예상: {input_height}x{input_width}")
            return cropped_image
            
    except Exception as e:
        print(f"⚠️ RTMW 전처리 실패: {e}")
        return None

# ===== 아이템 정보 추출 함수 =====

def extract_item_info_from_path(video_path: str) -> Optional[Tuple[str, int]]:
    """
    비디오 파일명에서 아이템 정보 추출 (WORD 또는 SEN)
    
    Args:
        video_path: 비디오 파일 경로 (예: NIA_SL_WORD0002_REAL05_F.mp4)
        
    Returns:
        Optional[Tuple[str, int]]: (item_type, item_id) 또는 None
        예: ("WORD", 2), ("SEN", 1234)
    """
    filename = Path(video_path).stem
    
    # 패턴 매칭: NIA_SL_WORD0002_REAL05_F 또는 NIA_SL_SEN1234_REAL09_F
    for item_type in ['WORD', 'SEN']:
        pattern = rf'_{item_type}(\d{{4}})_'
        match = re.search(pattern, filename)
        if match:
            item_id = int(match.group(1))
            return item_type, item_id
    
    return None

def calculate_batch_info(item_type: str, item_id: int) -> Tuple[int, int]:
    """
    아이템 타입과 ID로부터 배치 번호와 배치 내 인덱스 계산
    
    Args:
        item_type: "WORD" 또는 "SEN"
        item_id: 아이템 ID (예: 1, 2, 3000)
    
    Returns:
        Tuple[int, int]: (batch_number, batch_index)
    """
    batch_size = 250
    
    if item_type == "WORD":
        # WORD: 0001~3000 -> 배치 00~11 (12개 배치)
        if not (1 <= item_id <= 3000):
            raise ValueError(f"WORD ID 범위 오류: {item_id} (1~3000 범위)")
        batch_number = (item_id - 1) // batch_size
        batch_index = (item_id - 1) % batch_size
    elif item_type == "SEN":
        # SEN: 0001~2000 -> 배치 00~07 (8개 배치)
        if not (1 <= item_id <= 2000):
            raise ValueError(f"SEN ID 범위 오류: {item_id} (1~2000 범위)")
        batch_number = (item_id - 1) // batch_size
        batch_index = (item_id - 1) % batch_size
    else:
        raise ValueError(f"지원되지 않는 아이템 타입: {item_type}")
    
    return batch_number, batch_index

def get_batch_filename(item_type: str, batch_number: int, data_type: str) -> str:
    """
    배치 파일명 생성
    
    Args:
        item_type: "WORD" 또는 "SEN"
        batch_number: 배치 번호 (0~11 for WORD, 0~7 for SEN)
        data_type: "frames" 또는 "poses"
    
    Returns:
        str: 파일명 (예: batch_WORD_00_F_frames.h5)
    """
    return f"batch_{item_type}_{batch_number:02d}_F_{data_type}.h5"

# ===== 개선된 배치 처리기 클래스들 =====

@dataclass
class BatchMetrics:
    """배치 처리 메트릭스"""
    frames_processed: int = 0
    processing_time: float = 0.0
    gpu_memory_used: float = 0.0
    throughput_fps: float = 0.0
    batch_id: int = 0
    gpu_id: int = 0

class SmartVRAMBuffer:
    """스마트 VRAM 버퍼링 시스템"""
    
    def __init__(self, gpu_id: int, batch_size: int = 256, max_vram_usage: float = 0.85):
        self.gpu_id = gpu_id
        self.batch_size = batch_size
        self.max_vram_usage = max_vram_usage
        self.device = f"cuda:{gpu_id}"
        
        # GPU 메모리 정보
        torch.cuda.set_device(gpu_id)
        gpu_props = torch.cuda.get_device_properties(gpu_id)
        self.total_vram = gpu_props.total_memory / (1024**3)
        self.max_vram = self.total_vram * max_vram_usage
        
        # 프레임 배치 버퍼 (VRAM에 미리 로드)
        single_frame_size = 384 * 288 * 3 * 4  # float32
        self.batch_memory_mb = (single_frame_size * batch_size) / (1024 * 1024)
        
        print(f"🚀 SmartVRAMBuffer GPU {gpu_id} 초기화")
        print(f"   - 총 VRAM: {self.total_vram:.1f}GB")
        print(f"   - 최대 사용: {self.max_vram:.1f}GB")
        print(f"   - 배치 메모리: {self.batch_memory_mb:.1f}MB")

# ===== 개선된 클래스 정의 =====

@dataclass
class FrameBatch:
    """프레임 배치 데이터 구조"""
    frames: List[np.ndarray]
    indices: List[int]
    metadata: Dict
    
class VideoFrameLoader:
    """비디오 프레임 배치 로더 (180 프레임 배치) - 최적화"""
    def __init__(self, batch_size: int = 180):
        self.batch_size = batch_size
        
    def load_video_frames(self, video_path: str, max_frames: int = None) -> List[np.ndarray]:
        """비디오 프레임 전체 로드 - 200프레임 이하 최적화"""
        frames = []
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return frames
            
        # 비디오 정보 확인
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # 200프레임 이하 비디오를 위한 최적화
        if total_frames <= 200:
            print(f"📹 짧은 비디오 감지 ({total_frames}프레임) - 최적화 모드")
        
        # 최대 프레임 제한 (메모리 보호)
        if max_frames is None:
            max_frames = 500  # 기본 최대값
        
        frame_count = 0
        while frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
            frame_count += 1
                
        cap.release()
        
        if frame_count >= max_frames:
            print(f"⚠️ 프레임 제한 도달: {max_frames}개로 제한됨")
        
        return frames
    
    def create_frame_batches(self, frames: List[np.ndarray]) -> List[FrameBatch]:
        """프레임을 180개씩 배치로 분할 - 200프레임 이하 최적화"""
        batches = []
        total_frames = len(frames)
        
        # 200프레임 이하인 경우 배치 크기 조정
        if total_frames <= 200:
            # 작은 비디오는 단일 배치로 처리
            batch_size = min(self.batch_size, total_frames)
            print(f"📦 짧은 비디오 배치 최적화: {batch_size}개씩 분할")
        else:
            batch_size = self.batch_size
        
        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i:i + batch_size]
            batch_indices = list(range(i, i + len(batch_frames)))
            
            batch = FrameBatch(
                frames=batch_frames,
                indices=batch_indices,
                metadata={
                    'batch_start': i,
                    'batch_size': len(batch_frames),
                    'total_frames': len(frames),
                    'is_short_video': total_frames <= 200,
                    'optimized_batch': total_frames <= 200 and len(batch_frames) == total_frames
                }
            )
            batches.append(batch)
            
        return batches

class GPUBatchProcessor:
    """GPU 병렬 배치 처리기"""
    def __init__(self, gpu_ids: List[int], batch_size: int = 64):
        self.gpu_ids = gpu_ids
        self.batch_size = batch_size
        self.num_gpus = len(gpu_ids)
        self.device_pools = {gpu_id: [] for gpu_id in gpu_ids}
        
        print(f"🖥️ GPU 병렬 처리기 초기화:")
        print(f"   - 사용 GPU: {gpu_ids}")
        print(f"   - GPU 개수: {self.num_gpus}")
        print(f"   - 배치 크기: {batch_size}")
    
    def distribute_batches(self, frame_batches: List[FrameBatch]) -> Dict[int, List[FrameBatch]]:
        """배치를 GPU별로 분배"""
        gpu_batches = {gpu_id: [] for gpu_id in self.gpu_ids}
        
        for i, batch in enumerate(frame_batches):
            gpu_id = self.gpu_ids[i % self.num_gpus]
            gpu_batches[gpu_id].append(batch)
        
        return gpu_batches
    
    def process_batch_on_gpu(self, gpu_id: int, frame_batch: FrameBatch, inferencer) -> List[Dict]:
        """특정 GPU에서 배치 처리 - 프레임 GPU 미리 로드 최적화"""
        try:
            torch.cuda.set_device(gpu_id)
            results = []
            
            # 프레임 GPU 미리 로드 (선택적)
            gpu_frames = []
            if hasattr(inferencer, 'preload_frames_to_gpu') and getattr(inferencer, 'preload_frames_to_gpu', True):
                try:
                    # 프레임들을 GPU 메모리에 미리 로드
                    for frame in frame_batch.frames:
                        # OpenCV BGR을 RGB로 변환하고 normalize
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frame_tensor = torch.from_numpy(frame_rgb).float().cuda(gpu_id) / 255.0
                        gpu_frames.append(frame_tensor)
                except Exception as e:
                    print(f"⚠️ GPU {gpu_id} 프레임 미리 로드 실패, CPU 모드로 전환: {e}")
                    gpu_frames = []
            
            for frame_idx, frame in enumerate(frame_batch.frames):
                # GPU에서 미리 로드된 프레임 사용 또는 CPU 프레임 사용
                if gpu_frames and frame_idx < len(gpu_frames):
                    # GPU 프레임을 다시 CPU로 변환 (추론기 호환성)
                    gpu_frame = gpu_frames[frame_idx]
                    processed_frame = (gpu_frame.cpu().numpy() * 255.0).astype(np.uint8)
                    processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR)
                else:
                    processed_frame = frame
                
                # YOLO 검출
                person_boxes = inferencer.detect_persons_high_accuracy(processed_frame)
                
                if not person_boxes:
                    # 기본값
                    result = {
                        'frame_idx': frame_batch.indices[frame_idx],
                        'keypoints': [[0.0] * 266],
                        'scores': [[0.0] * 133],
                        'bbox': None,
                        'crop_image': cv2.resize(processed_frame, (288, 384))
                    }
                    results.append(result)
                    continue
                
                # RTMW 크롭
                bbox = person_boxes[0]
                crop_image = crop_person_image_rtmw(processed_frame, bbox)
                
                if crop_image is None:
                    crop_image = cv2.resize(processed_frame, (288, 384))
                    keypoints = [[0.0] * 266]
                    scores = [[0.0] * 133]
                else:
                    # 포즈 추정
                    keypoints, scores = inferencer.estimate_pose_on_crop(crop_image)
                    
                    # 키포인트 형식 변환
                    if isinstance(keypoints, np.ndarray):
                        if keypoints.ndim == 2:
                            kpts_flat = []
                            for joint_idx in range(min(133, keypoints.shape[0])):
                                joint = keypoints[joint_idx]
                                kpts_flat.extend([float(joint[0]), float(joint[1])])
                            while len(kpts_flat) < 266:
                                kpts_flat.extend([0.0, 0.0])
                            keypoints = [kpts_flat]
                        else:
                            kpts_flat = keypoints.flatten()[:266].tolist()
                            while len(kpts_flat) < 266:
                                kpts_flat.append(0.0)
                            keypoints = [kpts_flat]
                    else:
                        keypoints = [[0.0] * 266]
                    
                    # 스코어 변환
                    if isinstance(scores, np.ndarray):
                        score_list = scores.flatten()[:133].tolist()
                        while len(score_list) < 133:
                            score_list.append(0.0)
                        scores = [score_list]
                    else:
                        scores = [[0.0] * 133]
                
                result = {
                    'frame_idx': frame_batch.indices[frame_idx],
                    'keypoints': keypoints,
                    'scores': scores,
                    'bbox': bbox,
                    'crop_image': crop_image
                }
                results.append(result)
            
            # GPU 메모리 정리
            if gpu_frames:
                del gpu_frames
                torch.cuda.empty_cache()
            
            return results
            
        except Exception as e:
            print(f"⚠️ GPU {gpu_id} 배치 처리 실패: {e}")
            # GPU 메모리 정리
            if 'gpu_frames' in locals():
                del gpu_frames
            torch.cuda.empty_cache()
            return []

class PerformanceMonitor:
    """실시간 성능 모니터링 시스템"""
    
    def __init__(self):
        self.metrics_history = deque(maxlen=100)
        self.start_time = time.time()
        self.frame_count = 0
        self.batch_count = 0
        
        try:
            self.gpus = GPUtil.getGPUs()
        except:
            self.gpus = []

class ImprovedBatchVideoProcessor:
    """개선된 배치 비디오 처리기 - GPU 병렬처리 + 프레임 배치 로드"""
    
    def __init__(self, 
                 rtmw_model_name: str = "rtmw-dw-x-l_simcc-cocktail14_270e-384x288.onnx",
                 gpu_ids: List[int] = [0, 1],
                 batch_size: int = 256,
                 frame_batch_size: int = 180,
                 keypoint_scale: int = 8,
                 jpeg_quality: int = 90,
                 max_vram_usage: float = 0.85,
                 preload_frames_to_gpu: bool = True):
        
        self.rtmw_model_name = rtmw_model_name
        self.gpu_ids = gpu_ids
        self.batch_size = batch_size
        self.frame_batch_size = frame_batch_size
        self.keypoint_scale = keypoint_scale
        self.jpeg_quality = jpeg_quality
        self.max_vram_usage = max_vram_usage
        self.preload_frames_to_gpu = preload_frames_to_gpu
        
        # 병렬처리 구성요소 초기화
        self.frame_loader = VideoFrameLoader(batch_size=frame_batch_size)
        self.gpu_processor = GPUBatchProcessor(gpu_ids=gpu_ids, batch_size=batch_size)
        self.monitor = PerformanceMonitor()
        
        # GPU 메모리 풀 초기화 (각 GPU별로)
        self.gpu_memory_pools = {}
        for gpu_id in gpu_ids:
            self.gpu_memory_pools[gpu_id] = SmartVRAMBuffer(gpu_id, batch_size, max_vram_usage)
        
        # 각 GPU별로 추론기 초기화
        self.inferencers = {}
        for gpu_id in gpu_ids:
            torch.cuda.set_device(gpu_id)
            self.inferencers[gpu_id] = ONNXInferencer(
                rtmw_onnx_path=rtmw_model_name,
                detection_device=f"cuda:{gpu_id}",
                pose_device=f"cuda:{gpu_id}",
                optimize_for_accuracy=True
            )
        
        try:
            print(f"🚀 ImprovedBatchVideoProcessor 초기화 완료")
            print(f"   - 사용 GPU: {gpu_ids}")
            print(f"   - GPU 메모리 풀: 활성화")
            print(f"   - 배치 크기: {batch_size}")
            print(f"   - 프레임 배치: {frame_batch_size} (200프레임 이하 비디오 최적화)")
            print(f"   - 프레임 GPU 로드: {'활성화' if preload_frames_to_gpu else '비활성화'}")
            print(f"   - RTMW 모델: {rtmw_model_name}")
            print(f"   - 크롭 방식: RTMW (288x384, 패딩, 133 키포인트)")
            
            # GPU 워밍업
            self._warmup_gpus()
            
        except Exception as e:
            print(f"❌ ImprovedBatchVideoProcessor 초기화 실패: {e}")
            raise
    
    def _warmup_gpus(self):
        """모든 GPU 워밍업"""
        print(f"🔥 GPU 워밍업 시작 (GPU {self.gpu_ids}, 배치 {self.batch_size})")
        start_time = time.time()
        
        try:
            for gpu_id in self.gpu_ids:
                torch.cuda.set_device(gpu_id)
                
                # 더미 배치 텐서 생성
                dummy_batch = torch.randn(self.batch_size, 3, 384, 288).cuda()
                
                # 몇 번 연산 수행하여 GPU 활성화
                for _ in range(3):
                    _ = dummy_batch * 2.0
                    _ = torch.nn.functional.interpolate(dummy_batch, size=(288, 384))
                
                # 메모리 정리
                del dummy_batch
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            
            warmup_time = time.time() - start_time
            
            print(f"✅ 모든 GPU 워밍업 완료: {warmup_time:.2f}초")
            
            # 각 GPU VRAM 사용량 표시
            for gpu_id in self.gpu_ids:
                torch.cuda.set_device(gpu_id)
                vram_used = torch.cuda.memory_allocated(gpu_id) / 1024**3
                print(f"   - GPU {gpu_id}: VRAM {vram_used:.2f}GB")
                
        except Exception as e:
            print(f"⚠️ GPU 워밍업 실패: {e}")

    def process_video_rtmw_crop(self, video_path: str, progress_callback=None) -> Optional[Dict]:
        """GPU 병렬처리와 프레임 배치 로드를 사용한 비디오 처리"""
        start_total_time = time.time()
        
        try:
            # 파일 존재 여부 및 크기 확인
            if not os.path.exists(video_path):
                print(f"❌ 비디오 파일이 존재하지 않음: {video_path}")
                return None
            
            # 아이템 정보 추출
            item_info = extract_item_info_from_path(video_path)
            if item_info is None:
                print(f"❌ 아이템 정보 추출 실패: {video_path}")
                return None
            
            item_type, item_id = item_info
            print(f"📋 GPU 병렬 처리 중: {item_type}{item_id:04d} - {Path(video_path).name}")
            
            # 1. 프레임 배치 로드 (전체 프레임을 메모리에 로드)
            print(f"📥 프레임 로드 시작 (200프레임 이하 최적화)...")
            frames_load_start = time.time()
            frames = self.frame_loader.load_video_frames(video_path, max_frames=500)
            
            if not frames:
                print(f"❌ 유효한 프레임 없음: {video_path}")
                return None
            
            frames_load_time = time.time() - frames_load_start
            actual_frame_count = len(frames)
            
            # 프레임 수에 따른 처리 방식 안내
            if actual_frame_count <= 200:
                print(f"✅ 짧은 비디오 로드 완료: {actual_frame_count}개 프레임 ({frames_load_time:.2f}초)")
                print(f"   → 200프레임 이하 최적화 모드 활성화")
            else:
                print(f"✅ 프레임 로드 완료: {actual_frame_count}개 ({frames_load_time:.2f}초)")
            
            # 2. 프레임을 300개씩 배치로 분할 (200프레임 이하는 단일 배치)
            print(f"🔄 프레임 배치 생성...")
            frame_batches = self.frame_loader.create_frame_batches(frames)
            print(f"✅ 배치 생성 완료: {len(frame_batches)}개 배치")
            
            # 3. GPU별로 배치 분배
            gpu_batches = self.gpu_processor.distribute_batches(frame_batches)
            
            print(f"🖥️ GPU별 배치 분배:")
            for gpu_id, batches in gpu_batches.items():
                print(f"   - GPU {gpu_id}: {len(batches)}개 배치")
            
            # 4. GPU 병렬 처리 (ThreadPoolExecutor 사용)
            all_results = []
            processing_start = time.time()
            
            with ThreadPoolExecutor(max_workers=len(self.gpu_ids)) as executor:
                # 각 GPU별로 태스크 제출
                future_to_gpu = {}
                
                for gpu_id, batches in gpu_batches.items():
                    if batches:  # 배치가 있는 경우에만
                        for batch in batches:
                            future = executor.submit(
                                self.gpu_processor.process_batch_on_gpu,
                                gpu_id, batch, self.inferencers[gpu_id]
                            )
                            future_to_gpu[future] = (gpu_id, batch)
                
                # 결과 수집
                completed_batches = 0
                total_batches = len(future_to_gpu)
                
                for future in as_completed(future_to_gpu):
                    gpu_id, batch = future_to_gpu[future]
                    try:
                        batch_results = future.result(timeout=300)  # 5분 타임아웃
                        all_results.extend(batch_results)
                        completed_batches += 1
                        
                        # 진행률 업데이트
                        if progress_callback:
                            progress = 0.3 + (completed_batches / total_batches) * 0.7
                            progress_callback(min(progress, 1.0))
                        
                        print(f"✅ GPU {gpu_id} 배치 완료: {len(batch_results)}개 프레임 ({completed_batches}/{total_batches})")
                        
                    except Exception as e:
                        print(f"⚠️ GPU {gpu_id} 배치 처리 실패: {e}")
                        continue
            
            processing_time = time.time() - processing_start
            print(f"🚀 GPU 병렬 처리 완료: {processing_time:.2f}초")
            
            # 5. 결과 정렬 (프레임 인덱스 순서로)
            all_results.sort(key=lambda x: x['frame_idx'])
            
            # 6. 최종 데이터 구성
            jpeg_frames = []
            keypoints_list = []
            scores_list = []
            crop_images = []
            
            for result in all_results:
                # JPEG 인코딩
                _, buffer = cv2.imencode('.jpg', result['crop_image'], 
                                       [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
                jpeg_frames.append(buffer)
                
                # 데이터 추가
                keypoints_list.append(result['keypoints'])
                scores_list.append(result['scores'])
                crop_images.append(result['crop_image'])
            
            total_processing_time = time.time() - start_total_time
            
            # 배치 및 인덱스 정보 계산
            batch_number, batch_index = calculate_batch_info(item_type, item_id)
            
            # 최종 결과 구성
            result = {
                'item_type': item_type,
                'item_id': item_id,
                'batch_number': batch_number,
                'batch_index': batch_index,
                'total_frames': actual_frame_count,
                'processed_frames': len(all_results),
                'jpeg_frames': jpeg_frames,
                'crop_images': crop_images,
                'keypoints': keypoints_list,
                'scores': scores_list,
                'processing_time': total_processing_time,
                'fps': len(all_results) / max(total_processing_time, 0.001),
                'gpu_processing_time': processing_time,
                'frame_load_time': frames_load_time,
                'frame_batches_count': len(frame_batches),
                'video_info': {
                    'original_fps': 30.0,  # 기본값
                    'resolution': f"{frames[0].shape[1]}x{frames[0].shape[0]}" if frames else "unknown",
                    'duration': actual_frame_count / 30.0,
                    'crop_size': "288x384"
                }
            }
            
            print(f"🎉 {item_type}{item_id:04d} GPU 병렬 처리 완료:")
            print(f"   - 배치: {batch_number:02d}, 인덱스: {batch_index}")
            print(f"   - 전체 시간: {total_processing_time:.2f}초")
            print(f"   - 프레임 로드: {frames_load_time:.2f}초")
            print(f"   - GPU 처리: {processing_time:.2f}초")
            print(f"   - 처리 속도: {result['fps']:.1f} FPS")
            print(f"   - 프레임 배치: {len(frame_batches)}개 ({self.frame_batch_size}개씩)")
            print(f"   - 사용 GPU: {self.gpu_ids}")
            
            return result
            
        except Exception as e:
            print(f"💥 GPU 병렬 비디오 처리 오류 ({video_path}): {e}")
            traceback.print_exc()
            return None

def save_to_rtmw_hdf5_format(result_data: Dict, output_path: str):
    """RTMW 크롭 방식 HDF5 저장 (프레임과 포즈 분리)"""
    try:
        item_type = result_data['item_type']
        item_id = result_data['item_id']
        batch_number = result_data['batch_number']
        
        output_path_obj = Path(output_path)
        
        # 프레임과 포즈 파일 분리
        frames_filename = get_batch_filename(item_type, batch_number, 'frames')
        poses_filename = get_batch_filename(item_type, batch_number, 'poses')
        
        frames_h5_path = output_path_obj.parent / frames_filename
        poses_h5_path = output_path_obj.parent / poses_filename
        
        # JPEG 가변 길이 타입
        jpeg_vlen_dtype = h5py.vlen_dtype(np.uint8)
        
        print(f"💾 RTMW HDF5 저장: {item_type}{item_id:04d} -> 배치 {batch_number:02d}")
        
        # 두 파일을 동시에 열어서 데이터 추가
        with h5py.File(frames_h5_path, 'a') as f_frames, \
             h5py.File(poses_h5_path, 'a') as f_poses:
            
            # 배치 메타데이터 (없는 경우에만 추가)
            if not f_frames.attrs:
                batch_metadata = {
                    'batch_type': item_type,
                    'batch_number': batch_number,
                    'item_range': f"{item_type}_{batch_number*250+1:04d}~{item_type}_{min((batch_number+1)*250, 3000 if item_type=='WORD' else 2000):04d}",
                    'direction': 'F',
                    'crop_method': 'RTMW',
                    'crop_size': '288x384',
                    'keypoint_count': 133,
                    'keypoint_format': 'RTMW_133_with_face_hands',
                    'creation_time': str(datetime.now())
                }
                f_frames.attrs.update(batch_metadata)
                f_poses.attrs.update(batch_metadata)
            
            # 비디오 그룹 이름
            video_group_name = f"video_{item_type.lower()}{item_id:04d}"
            
            # === 프레임 파일 저장 ===
            if video_group_name not in f_frames:
                frame_group = f_frames.create_group(video_group_name)
                
                # JPEG 프레임 데이터 저장
                jpeg_frames = result_data.get('jpeg_frames', [])
                if jpeg_frames:
                    frame_group.create_dataset("frames_jpeg", data=jpeg_frames, dtype=jpeg_vlen_dtype)
                
                # 메타데이터 저장
                metadata = {
                    'item_type': item_type,
                    'item_id': item_id,
                    'batch_number': batch_number,
                    'batch_index': result_data['batch_index'],
                    'frame_count': result_data['total_frames'],
                    'processing_time': result_data.get('processing_time', 0.0),
                    'crop_method': 'RTMW',
                    'crop_size': '288x384',
                    'keypoint_scale': 8,
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                }
                f_frames.create_dataset(f"{video_group_name}/metadata", data=json.dumps(metadata))
            
            # === 포즈 파일 저장 ===
            if video_group_name not in f_poses:
                pose_group = f_poses.create_group(video_group_name)
                
                keypoints = result_data.get('keypoints', [])
                scores = result_data.get('scores', [])
                
                if keypoints and scores:
                    num_frames = len(keypoints)
                    
                    # 133개 키포인트 배열 초기화 (RTMW 크롭 좌표계)
                    keypoints_array = np.zeros((num_frames, 133, 2), dtype=np.float32)
                    scores_array = np.zeros((num_frames, 133), dtype=np.float32)
                    
                    for frame_idx, (frame_kpts, frame_scores) in enumerate(zip(keypoints, scores)):
                        # 첫 번째 사람만 사용
                        if isinstance(frame_kpts, list) and len(frame_kpts) > 0:
                            person_kpts = frame_kpts[0]
                            person_scores = frame_scores[0] if len(frame_scores) > 0 else []
                            
                            # 키포인트 변환 (x,y,x,y... -> [[x,y], [x,y], ...])
                            if isinstance(person_kpts, list) and len(person_kpts) >= 266:  # 133*2
                                for joint_idx in range(133):
                                    x_idx = joint_idx * 2
                                    y_idx = joint_idx * 2 + 1
                                    if y_idx < len(person_kpts):
                                        keypoints_array[frame_idx, joint_idx, 0] = person_kpts[x_idx]
                                        keypoints_array[frame_idx, joint_idx, 1] = person_kpts[y_idx]
                            
                            # 스코어 변환 (133개 전체)
                            if isinstance(person_scores, list) and len(person_scores) >= 133:
                                scores_array[frame_idx, :] = person_scores[:133]
                    
                    # 키포인트 8배 스케일링 후 int32로 저장
                    keypoints_scaled = np.round(keypoints_array * 8).astype(np.int32)
                    
                    pose_group.create_dataset("keypoints_scaled", data=keypoints_scaled, compression='lzf')
                    pose_group.create_dataset("scores", data=scores_array, compression='lzf')
        
        print(f"✅ RTMW HDF5 저장 완료: {frames_filename}, {poses_filename}")
        
    except Exception as e:
        print(f"💥 RTMW HDF5 저장 실패: {e}")
        traceback.print_exc()
        raise

# ===== 메인 처리 함수들 =====

def process_videos_with_rtmw_crop(
    video_paths: List[str],
    output_dir: str,
    config: Dict,
    progress_callback=None
) -> Dict:
    """GPU 병렬처리와 프레임 배치 로드를 사용한 비디오들 처리"""
    
    start_time = time.time()
    total_videos = len(video_paths)
    
    print(f"🚀 GPU 병렬 + 프레임 배치 비디오 처리 시작")
    print(f"   - 총 비디오: {total_videos}개")
    print(f"   - 출력 디렉토리: {output_dir}")
    print(f"   - 사용 GPU: {config['gpu_ids']}")
    print(f"   - 프레임 배치: {config['frame_batch_size']}개씩")
    print(f"   - 크롭 방식: RTMW (288x384, 패딩, 133 키포인트)")
    
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # GPU 병렬 배치 처리기 초기화
    processor = ImprovedBatchVideoProcessor(
        rtmw_model_name=config['rtmw_model_name'],
        gpu_ids=config['gpu_ids'],
        batch_size=config['batch_size'],
        frame_batch_size=config['frame_batch_size'],
        keypoint_scale=config.get('keypoint_scale', 8),
        jpeg_quality=config.get('jpeg_quality', 90),
        max_vram_usage=config.get('max_vram_usage', 0.85)
    )
    
    # 결과 수집
    results = {}
    successful_count = 0
    failed_count = 0
    
    for video_idx, video_path in enumerate(video_paths):
        print(f"\n📹 GPU 병렬 처리 중 ({video_idx+1}/{total_videos}): {Path(video_path).name}")
        
        def video_progress_callback(progress):
            overall_progress = (video_idx + progress) / total_videos
            if progress_callback:
                progress_callback(overall_progress)
        
        # GPU 병렬 + 프레임 배치 처리
        result = processor.process_video_rtmw_crop(video_path, video_progress_callback)
        
        if result:
            # HDF5 저장
            try:
                save_to_rtmw_hdf5_format(result, output_dir)
                successful_count += 1
                
                # GPU별 처리 시간 정보 출력
                if 'gpu_processing_time' in result:
                    print(f"   ⚡ GPU 처리 시간: {result['gpu_processing_time']:.2f}초")
                    print(f"   📦 프레임 배치: {result.get('frame_batches_count', 0)}개")
                
            except Exception as e:
                print(f"❌ HDF5 저장 실패: {e}")
                failed_count += 1
        else:
            failed_count += 1
        
        results[video_idx] = {
            'video_path': video_path,
            'result': result,
            'status': 'completed' if result else 'failed'
        }
    
    total_time = time.time() - start_time
    
    # 최종 결과 정리
    summary = {
        'total_videos': total_videos,
        'completed': successful_count,
        'failed': failed_count,
        'total_time': total_time,
        'average_time_per_video': total_time / max(total_videos, 1),
        'results': results,
        'processing_method': 'GPU_Parallel_Frame_Batch',
        'gpu_count': len(config['gpu_ids']),
        'frame_batch_size': config['frame_batch_size']
    }
    
    print(f"\n🏁 GPU 병렬 + 프레임 배치 처리 완료:")
    print(f"   - 성공: {successful_count}개")
    print(f"   - 실패: {failed_count}개")
    print(f"   - 총 시간: {total_time:.2f}초")
    print(f"   - 평균 시간: {summary['average_time_per_video']:.2f}초/비디오")
    print(f"   - 사용 GPU: {len(config['gpu_ids'])}개")
    print(f"   - 프레임 배치: {config['frame_batch_size']}개씩")
    
    return summary

def find_all_videos_for_processing(data_root: str = None) -> List[str]:
    """전체 처리용 비디오 파일 찾기"""
    if data_root is None:
        # 가능한 경로들 시도
        possible_paths = [
            "data/1.Training/videos",
            "/workspace01/team03/data/mmpose/jy/data/1.Training/videos",
            "../data/1.Training/videos",
            "./data/1.Training/videos"
        ]
        
        data_root = None
        for path in possible_paths:
            if os.path.exists(path):
                data_root = path
                break
    
    if not data_root or not os.path.exists(data_root):
        print(f"❌ 비디오 디렉토리를 찾을 수 없습니다.")
        print(f"   시도된 경로들:")
        for path in possible_paths:
            print(f"   - {path}")
        return []
    
    print(f"🔍 비디오 파일 탐색: {data_root}")
    
    all_videos = []
    video_stats = {'WORD': 0, 'SEN': 0, 'OTHER': 0}
    
    # 모든 하위 폴더에서 _F.mp4 파일 찾기
    for root, dirs, files in os.walk(data_root):
        for file in files:
            if file.endswith('_F.mp4'):
                video_path = os.path.join(root, file)
                
                # 아이템 정보 확인
                item_info = extract_item_info_from_path(video_path)
                if item_info:
                    item_type, item_id = item_info
                    all_videos.append(video_path)
                    video_stats[item_type] += 1
                else:
                    video_stats['OTHER'] += 1
    
    # 정렬 (WORD 먼저, 그 다음 SEN, 각각 ID 순)
    def sort_key(video_path):
        item_info = extract_item_info_from_path(video_path)
        if item_info:
            item_type, item_id = item_info
            return (item_type, item_id)
        return ('ZZZ', 9999)  # 알 수 없는 파일은 마지막에
    
    all_videos.sort(key=sort_key)
    
    print(f"✅ 발견된 비디오: {len(all_videos)}개")
    print(f"   - WORD: {video_stats['WORD']}개")
    print(f"   - SEN: {video_stats['SEN']}개") 
    print(f"   - 기타: {video_stats['OTHER']}개")
    
    return all_videos

def process_all_videos_production(all_videos: List[str], output_dir: str, config: Dict) -> Dict:
    """전체 비디오 Production 처리"""
    
    start_time = time.time()
    total_videos = len(all_videos)
    
    print(f"\n🚀 전체 비디오 처리 시작")
    print("=" * 80)
    print(f"📁 출력 디렉토리: {output_dir}")
    print(f"📹 총 비디오 수: {total_videos}개")
    print(f"⚙️ 배치 크기: {config['batch_size']}")
    print(f"🖥️ 최대 VRAM: {config['max_vram_usage']*100}%")
    print(f"📊 예상 배치 수:")
    
    # 배치 예상 계산
    batch_estimates = {}
    for video_path in all_videos:
        item_info = extract_item_info_from_path(video_path)
        if item_info:
            item_type, item_id = item_info
            batch_number, _ = calculate_batch_info(item_type, item_id)
            batch_key = f"{item_type}_{batch_number:02d}"
            batch_estimates[batch_key] = batch_estimates.get(batch_key, 0) + 1
    
    for batch_key, count in sorted(batch_estimates.items()):
        print(f"   - {batch_key}: {count}개 비디오")
    
    print(f"   - 총 예상 배치: {len(batch_estimates)}개")
    
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 처리 실행
    try:
        summary = process_videos_with_rtmw_crop(
            video_paths=all_videos,
            output_dir=output_dir,
            config=config,
            progress_callback=None  # 내장된 ProgressDisplay 사용
        )
        
        total_time = time.time() - start_time
        
        # 최종 통계
        print(f"\n🎉 전체 처리 완료!")
        print("=" * 80)
        print(f"✅ 처리 결과:")
        print(f"   - 성공: {summary['completed']}개")
        print(f"   - 실패: {summary['failed']}개")
        print(f"   - 성공률: {summary['completed']/total_videos*100:.1f}%")
        print(f"   - 총 시간: {total_time:.2f}초 ({total_time/3600:.1f}시간)")
        print(f"   - 평균 시간: {total_time/total_videos:.2f}초/비디오")
        
        # 생성된 HDF5 파일 확인
        h5_files = [f for f in os.listdir(output_dir) if f.endswith('.h5')]
        frames_files = [f for f in h5_files if '_frames.h5' in f]
        poses_files = [f for f in h5_files if '_poses.h5' in f]
        
        print(f"\n💾 생성된 HDF5 파일:")
        print(f"   - 프레임 파일: {len(frames_files)}개")
        print(f"   - 포즈 파일: {len(poses_files)}개")
        
        # 파일 크기 정보
        total_size = 0
        for f in h5_files:
            file_path = os.path.join(output_dir, f)
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path)
                total_size += file_size
        
        print(f"   - 총 용량: {total_size / (1024*1024*1024):.2f}GB")
        
        # 샘플 파일들 표시
        print(f"\n📄 생성된 파일 샘플:")
        for f in sorted(h5_files)[:10]:
            file_path = os.path.join(output_dir, f)
            file_size = os.path.getsize(file_path) / (1024*1024)
            print(f"   - {f} ({file_size:.1f}MB)")
        
        if len(h5_files) > 10:
            print(f"   ... 및 {len(h5_files) - 10}개 추가 파일")
        
        # 배치별 통계
        print(f"\n📊 배치별 결과:")
        batch_results = {}
        for task_id, result_item in summary['results'].items():
            if result_item['status'] == 'completed' and result_item['result']:
                result_data = result_item['result']
                item_type = result_data['item_type']
                batch_number = result_data['batch_number']
                batch_key = f"{item_type}_{batch_number:02d}"
                batch_results[batch_key] = batch_results.get(batch_key, 0) + 1
        
        for batch_key, count in sorted(batch_results.items()):
            print(f"   - {batch_key}: {count}개 완료")
        
        return summary
        
    except Exception as e:
        print(f"💥 전체 처리 실패: {e}")
        traceback.print_exc()
        return {'status': 'failed', 'error': str(e)}

def main():
    """메인 실행 함수 - GPU 병렬처리 + 프레임 배치 로드"""
    print("🚀 개선된 배치 처리기 - GPU 병렬처리 + 프레임 배치 로드")
    print("=" * 80)
    
    # 기본 설정
    config = {
        'rtmw_model_name': 'rtmw-dw-x-l_simcc-cocktail14_270e-384x288.onnx',
        'gpu_ids': [0, 1],  # GPU 병렬처리
        'batch_size': 128,
        'frame_batch_size': 180,  # 프레임 배치 크기
        'keypoint_scale': 8,
        'jpeg_quality': 90,
        'max_vram_usage': 0.75
    }
    
    print(f"⚙️ 처리 설정:")
    print(f"   - RTMW 모델: {config['rtmw_model_name']}")
    print(f"   - 사용 GPU: {config['gpu_ids']}")
    print(f"   - 배치 크기: {config['batch_size']}")
    print(f"   - 프레임 배치: {config['frame_batch_size']}개씩 (200프레임 이하 비디오 최적화)")
    print(f"   - 키포인트 스케일: {config['keypoint_scale']}")
    print(f"   - JPEG 품질: {config['jpeg_quality']}%")
    print(f"   - 최대 VRAM 사용: {config['max_vram_usage']*100}%")
    print(f"   - 크롭 방식: RTMW (288x384, 133 키포인트)")
    print(f"   - GPU 병렬처리: 활성화")
    print(f"   - 프레임 미리로드: GPU 메모리 최적화")
    
    # GPU 가용성 확인
    available_gpus = []
    if torch.cuda.is_available():
        for gpu_id in config['gpu_ids']:
            if gpu_id < torch.cuda.device_count():
                available_gpus.append(gpu_id)
                print(f"✅ GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
            else:
                print(f"⚠️ GPU {gpu_id}: 사용 불가")
    
    if not available_gpus:
        print("❌ 사용 가능한 GPU가 없습니다.")
        return 1
    
    config['gpu_ids'] = available_gpus
    
    # 출력 디렉토리 설정
    output_dir = "data/prepro"
    print(f"📁 출력 디렉토리: {output_dir}")
    
    # 전체 비디오 찾기
    all_videos = find_all_videos_for_processing()
    
    if not all_videos:
        print("❌ 처리할 비디오가 없습니다.")
        return 1
    
    print(f"\n📋 처리 대상 비디오 샘플 (처음 10개):")
    for i, video_path in enumerate(all_videos[:10]):
        item_info = extract_item_info_from_path(video_path)
        if item_info:
            item_type, item_id = item_info
            batch_number, batch_index = calculate_batch_info(item_type, item_id)
            print(f"   {i+1:2d}. {item_type}{item_id:04d} → 배치 {batch_number:02d} (인덱스 {batch_index})")
    
    if len(all_videos) > 10:
        print(f"   ... 및 {len(all_videos) - 10}개 추가 비디오")
    
    # 사용자 확인
    print(f"\n❓ {len(all_videos)}개 비디오를 GPU 병렬처리로 {output_dir}에 처리하시겠습니까?")
    print(f"   예상 처리 시간: {len(all_videos) * 2 / 3600:.1f}시간 (GPU 병렬처리, 비디오당 2초 기준)")
    print(f"   프레임 배치: 각 비디오를 {config['frame_batch_size']}개씩 배치로 분할")
    print(f"   GPU 병렬화: {len(available_gpus)}개 GPU 동시 사용")
    
    try:
        choice = input("계속하려면 Enter, 중단하려면 Ctrl+C: ")
    except KeyboardInterrupt:
        print("⏹️ 사용자가 처리를 중단했습니다.")
        return 0
    
    # 전체 처리 실행
    try:
        summary = process_all_videos_production(all_videos, output_dir, config)
        
        if summary.get('status') != 'failed':
            print(f"🎉 GPU 병렬 비디오 처리가 성공적으로 완료되었습니다!")
            print(f"📁 결과 확인: {output_dir}")
            return 0
        else:
            print(f"❌ 처리 중 오류가 발생했습니다.")
            return 1
            
    except KeyboardInterrupt:
        print(f"⏹️ 사용자가 처리를 중단했습니다.")
        return 0
    except Exception as e:
        print(f"💥 처리 실패: {e}")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    main()