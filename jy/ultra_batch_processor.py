#!/usr/bin/env python3
"""
Ultra Batch Processor - A6000 x2 GPU 완전 활용 (256 배치)
Phase 1: 배치 처리 아키텍처 구축 ⚡
Phase 2: 성능 모니터링 시스템 📊  
Phase 3: 고속 파이프라인 구현 🚀
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
from typing import List, Dict, Tuple, Optional, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from collections import deque
import psutil
import GPUtil

from onnx_inferencer import YOLO11LRTMWONNXInferencer as ONNXInferencer

# ===== Phase 1: 배치 처리 아키텍처 구축 ⚡ =====

@dataclass
class BatchMetrics:
    """배치 처리 메트릭스"""
    frames_processed: int = 0
    processing_time: float = 0.0
    gpu_memory_used: float = 0.0
    throughput_fps: float = 0.0
    batch_id: int = 0
    gpu_id: int = 0

@dataclass
class VideoData:
    """비디오 데이터 구조체"""
    video_path: str
    frames: List[np.ndarray]
    bboxes: List[List[float]]
    total_frames: int
    video_info: Dict

# RTMW 전처리 함수들 (streamlined_processor.py와 동일)
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

class SmartVRAMBuffer:
    """스마트 VRAM 버퍼링 시스템 - Phase 1"""
    
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
        
        # 버퍼 초기화
        self.frame_buffer = torch.zeros((batch_size, 3, 384, 288), 
                                      dtype=torch.float32, device=self.device)
        self.buffer_ready = False
        
        print(f"🚀 SmartVRAMBuffer GPU {gpu_id} 초기화")
        print(f"   - 총 VRAM: {self.total_vram:.1f}GB")
        print(f"   - 최대 사용: {self.max_vram:.1f}GB")
        print(f"   - 배치 메모리: {self.batch_memory_mb:.1f}MB")
        
    def preload_batch(self, frames: List[np.ndarray]) -> torch.Tensor:
        """프레임 배치를 VRAM에 미리 로드"""
        batch_size = min(len(frames), self.batch_size)
        
        with torch.cuda.device(self.device):
            batch_tensor = torch.zeros((batch_size, 3, 384, 288), 
                                     dtype=torch.float32, device=self.device)
            
            for i, frame in enumerate(frames[:batch_size]):
                if frame is not None:
                    # OpenCV (H,W,C) -> PyTorch (C,H,W) 변환
                    frame_tensor = torch.from_numpy(frame.transpose(2, 0, 1)).float()
                    batch_tensor[i] = frame_tensor.to(self.device) / 255.0
            
            return batch_tensor
    
    def get_memory_usage(self) -> Dict[str, float]:
        """현재 GPU 메모리 사용량 반환"""
        allocated = torch.cuda.memory_allocated(self.device) / (1024**3)
        reserved = torch.cuda.memory_reserved(self.device) / (1024**3)
        return {
            'allocated': allocated,
            'reserved': reserved,
            'usage_percent': (allocated / self.total_vram) * 100
        }

class PerformanceMonitor:
    """실시간 성능 모니터링 시스템 - Phase 2"""
    
    def __init__(self):
        self.metrics_history = deque(maxlen=100)  # 최근 100개 배치 기록
        self.start_time = time.time()
        self.frame_count = 0
        self.batch_count = 0
        
        # GPU 모니터링 초기화
        self.gpus = GPUtil.getGPUs()
        
    def log_batch_metrics(self, metrics: BatchMetrics):
        """배치 메트릭스 기록"""
        self.metrics_history.append(metrics)
        self.frame_count += metrics.frames_processed
        self.batch_count += 1
        
    def get_realtime_stats(self) -> Dict:
        """실시간 통계 반환"""
        if not self.metrics_history:
            return {}
        
        recent_metrics = list(self.metrics_history)[-10:]  # 최근 10개 배치
        
        avg_fps = np.mean([m.throughput_fps for m in recent_metrics])
        avg_gpu_usage = np.mean([m.gpu_memory_used for m in recent_metrics])
        
        total_time = time.time() - self.start_time
        overall_fps = self.frame_count / max(total_time, 0.001)
        
        # GPU 상태 조회
        gpu_stats = []
        for gpu in GPUtil.getGPUs():
            gpu_stats.append({
                'id': gpu.id,
                'name': gpu.name,
                'memory_used': gpu.memoryUsed,
                'memory_total': gpu.memoryTotal,
                'memory_percent': gpu.memoryUtil * 100,
                'gpu_util': gpu.load * 100
            })
        
        return {
            'batch_count': self.batch_count,
            'total_frames': self.frame_count,
            'avg_fps_recent': avg_fps,
            'overall_fps': overall_fps,
            'avg_gpu_memory': avg_gpu_usage,
            'total_processing_time': total_time,
            'gpu_stats': gpu_stats,
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent
        }
    
    def print_progress(self):
        """진행 상황 출력"""
        stats = self.get_realtime_stats()
        if stats:
            print(f"\n📊 실시간 성능 통계:")
            print(f"   - 처리된 배치: {stats['batch_count']}")
            print(f"   - 총 프레임: {stats['total_frames']}")
            print(f"   - 전체 FPS: {stats['overall_fps']:.1f}")
            print(f"   - 최근 평균 FPS: {stats['avg_fps_recent']:.1f}")
            print(f"   - CPU 사용률: {stats['cpu_percent']:.1f}%")
            print(f"   - 메모리 사용률: {stats['memory_percent']:.1f}%")
            
            for gpu_stat in stats['gpu_stats']:
                print(f"   - GPU {gpu_stat['id']} ({gpu_stat['name']}): "
                      f"VRAM {gpu_stat['memory_percent']:.1f}%, "
                      f"사용률 {gpu_stat['gpu_util']:.1f}%")

class AsyncDataLoader:
    """비동기 데이터 로딩 시스템 - Phase 3"""
    
    def __init__(self, max_queue_size: int = 4):
        self.max_queue_size = max_queue_size
        self.data_queue = queue.Queue(maxsize=max_queue_size)
        self.loading_thread = None
        self.is_loading = False
        
    def start_loading(self, video_paths: List[str]):
        """비동기 데이터 로딩 시작"""
        self.is_loading = True
        self.loading_thread = threading.Thread(
            target=self._load_videos_async,
            args=(video_paths,),
            daemon=True
        )
        self.loading_thread.start()
        
    def _load_videos_async(self, video_paths: List[str]):
        """비동기로 비디오 로딩"""
        for video_path in video_paths:
            if not self.is_loading:
                break
                
            try:
                video_data = self._load_single_video(video_path)
                if video_data:
                    self.data_queue.put(video_data, timeout=10)
            except queue.Full:
                print(f"⚠️ 큐가 가득참, 비디오 스킵: {video_path}")
                continue
            except Exception as e:
                print(f"❌ 비디오 로딩 실패: {video_path} - {e}")
                continue
    
    def _load_single_video(self, video_path: str) -> Optional[VideoData]:
        """단일 비디오 로딩"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
            
        frames = []
        video_info = {
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        }
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        
        cap.release()
        
        return VideoData(
            video_path=video_path,
            frames=frames,
            bboxes=[],  # 나중에 채움
            total_frames=len(frames),
            video_info=video_info
        )
    
    def get_next_video(self, timeout: float = 5.0) -> Optional[VideoData]:
        """다음 비디오 데이터 가져오기"""
        try:
            return self.data_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def stop_loading(self):
        """로딩 중지"""
        self.is_loading = False
        if self.loading_thread and self.loading_thread.is_alive():
            self.loading_thread.join(timeout=2)

class UltraBatchProcessor:
    """Ultra 배치 처리기 - 모든 Phase 통합"""
    
    def __init__(self,
                 rtmw_model_name: str = "rtmw-dw-x-l_simcc-cocktail14_270e-384x288.onnx",
                 batch_size: int = 256,
                 gpu_ids: List[int] = [0, 1],  # A6000 x2
                 keypoint_scale: int = 8,
                 jpeg_quality: int = 90,
                 max_vram_usage: float = 0.85):
        
        self.rtmw_model_name = rtmw_model_name
        self.batch_size = batch_size
        self.gpu_ids = gpu_ids
        self.keypoint_scale = keypoint_scale
        self.jpeg_quality = jpeg_quality
        self.max_vram_usage = max_vram_usage
        
        # Phase 2: 성능 모니터링 시스템 초기화
        self.monitor = PerformanceMonitor()
        
        # Phase 1: GPU별 VRAM 버퍼 초기화
        self.gpu_buffers = {}
        self.gpu_processors = {}
        
        for gpu_id in self.gpu_ids:
            self.gpu_buffers[gpu_id] = SmartVRAMBuffer(
                gpu_id=gpu_id, 
                batch_size=batch_size,
                max_vram_usage=max_vram_usage
            )
            
            # GPU별 ONNX 추론기 초기화
            torch.cuda.set_device(gpu_id)
            self.gpu_processors[gpu_id] = ONNXInferencer(
                rtmw_onnx_path=rtmw_model_name,
                detection_device=f"cuda:{gpu_id}",
                pose_device=f"cuda:{gpu_id}",
                optimize_for_accuracy=True
            )
        
        # Phase 3: 비동기 데이터 로더 초기화
        self.data_loader = AsyncDataLoader(max_queue_size=8)
        
        print(f"🚀 UltraBatchProcessor 초기화 완료")
        print(f"   - GPU 개수: {len(self.gpu_ids)}")
        print(f"   - 배치 크기: {batch_size}")
        print(f"   - RTMW 모델: {rtmw_model_name}")
        
    def _crop_person_image_rtmw(self, image: np.ndarray, bbox: List[float]) -> Optional[np.ndarray]:
        """RTMW 방식으로 사람 이미지 크롭 (streamlined_processor.py와 동일)"""
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
            
            return cropped_image
                
        except Exception as e:
            print(f"⚠️ RTMW 전처리 실패: {e}")
            return None
    
    def _process_video_batch_gpu(self, gpu_id: int, video_batch: List[VideoData]) -> List[Dict]:
        """GPU별 비디오 배치 처리"""
        results = []
        gpu_buffer = self.gpu_buffers[gpu_id]
        processor = self.gpu_processors[gpu_id]
        
        batch_start_time = time.time()
        total_frames_processed = 0
        
        print(f"🔄 GPU {gpu_id}: {len(video_batch)}개 비디오 배치 처리 시작")
        
        for video_data in video_batch:
            try:
                # 1. YOLO 검출로 바운딩박스 추출
                bboxes = []
                cropped_frames = []
                
                for frame in video_data.frames:
                    # YOLO 추론
                    vis_image, detections = processor.process_frame(frame)
                    if detections and len(detections) > 0:
                        _, _, bbox = detections[0]  # 첫 번째 사람만 사용
                        bboxes.append(bbox)
                        
                        # RTMW 방식 크롭
                        cropped = self._crop_person_image_rtmw(frame, bbox)
                        if cropped is not None:
                            cropped_frames.append(cropped)
                        else:
                            cropped_frames.append(None)
                    else:
                        bboxes.append(None)
                        cropped_frames.append(None)
                
                # 2. 유효한 프레임들을 배치로 처리
                valid_frames = [f for f in cropped_frames if f is not None]
                if not valid_frames:
                    print(f"⚠️ GPU {gpu_id}: 유효한 프레임 없음 - {video_data.video_path}")
                    continue
                
                # 3. 배치 단위로 포즈 추정
                all_keypoints = []
                all_scores = []
                all_jpeg_frames = []
                
                for i in range(0, len(valid_frames), self.batch_size):
                    batch_frames = valid_frames[i:i + self.batch_size]
                    
                    # VRAM에 배치 로드
                    batch_tensor = gpu_buffer.preload_batch(batch_frames)
                    
                    # 배치 포즈 추정
                    batch_keypoints = []
                    batch_scores = []
                    
                    for frame in batch_frames:
                        keypoints, scores = processor.estimate_pose_on_crop(frame)
                        batch_keypoints.append(keypoints)
                        batch_scores.append(scores)
                        
                        # JPEG 인코딩
                        encode_params = [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
                        success, buffer = cv2.imencode('.jpg', frame, encode_params)
                        if success:
                            all_jpeg_frames.append(buffer)
                    
                    all_keypoints.extend(batch_keypoints)
                    all_scores.extend(batch_scores)
                    total_frames_processed += len(batch_frames)
                
                # 4. 결과 구성
                video_result = {
                    'video_path': video_data.video_path,
                    'jpeg_frames': all_jpeg_frames,
                    'keypoints': np.array(all_keypoints),
                    'scores': np.array(all_scores),
                    'frame_count': len(all_jpeg_frames),
                    'video_info': video_data.video_info
                }
                results.append(video_result)
                
            except Exception as e:
                print(f"❌ GPU {gpu_id} 비디오 처리 실패: {video_data.video_path} - {e}")
                continue
        
        # 배치 메트릭스 기록
        batch_time = time.time() - batch_start_time
        gpu_memory = gpu_buffer.get_memory_usage()
        
        metrics = BatchMetrics(
            frames_processed=total_frames_processed,
            processing_time=batch_time,
            gpu_memory_used=gpu_memory['allocated'],
            throughput_fps=total_frames_processed / max(batch_time, 0.001),
            batch_id=len(self.monitor.metrics_history),
            gpu_id=gpu_id
        )
        self.monitor.log_batch_metrics(metrics)
        
        print(f"✅ GPU {gpu_id}: {len(results)}개 비디오 완료 "
              f"({total_frames_processed}프레임, {metrics.throughput_fps:.1f} FPS)")
        
        return results
    
    def process_videos_ultra_batch(self, video_paths: List[str]) -> List[Dict]:
        """Ultra 배치 비디오 처리 - 모든 Phase 통합 실행"""
        print(f"🚀 Ultra Batch 처리 시작: {len(video_paths)}개 비디오")
        
        # Phase 3: 비동기 데이터 로딩 시작
        self.data_loader.start_loading(video_paths)
        
        all_results = []
        processed_videos = 0
        
        # GPU별 처리를 위한 큐
        gpu_queues = {gpu_id: [] for gpu_id in self.gpu_ids}
        
        try:
            while processed_videos < len(video_paths):
                # Phase 3: 다음 배치 비디오들 로드
                video_batch = []
                for _ in range(len(self.gpu_ids) * 2):  # GPU 개수의 2배씩 배치
                    video_data = self.data_loader.get_next_video(timeout=2.0)
                    if video_data is None:
                        break
                    video_batch.append(video_data)
                
                if not video_batch:
                    break
                
                # GPU별로 비디오 분배
                for i, video_data in enumerate(video_batch):
                    gpu_id = self.gpu_ids[i % len(self.gpu_ids)]
                    gpu_queues[gpu_id].append(video_data)
                
                # 병렬 GPU 처리
                futures = []
                with ThreadPoolExecutor(max_workers=len(self.gpu_ids)) as executor:
                    for gpu_id in self.gpu_ids:
                        if gpu_queues[gpu_id]:
                            future = executor.submit(
                                self._process_video_batch_gpu, 
                                gpu_id, 
                                gpu_queues[gpu_id]
                            )
                            futures.append(future)
                            gpu_queues[gpu_id] = []  # 큐 비우기
                    
                    # 결과 수집
                    for future in as_completed(futures):
                        try:
                            batch_results = future.result(timeout=300)
                            all_results.extend(batch_results)
                            processed_videos += len(batch_results)
                        except Exception as e:
                            print(f"❌ 배치 처리 실패: {e}")
                
                # Phase 2: 실시간 성능 모니터링
                self.monitor.print_progress()
                
        finally:
            # Phase 3: 비동기 로딩 정리
            self.data_loader.stop_loading()
        
        # 최종 통계
        final_stats = self.monitor.get_realtime_stats()
        print(f"\n🎉 Ultra Batch 처리 완료!")
        print(f"   - 총 처리 비디오: {len(all_results)}")
        print(f"   - 총 처리 프레임: {final_stats.get('total_frames', 0)}")
        print(f"   - 전체 처리 시간: {final_stats.get('total_processing_time', 0):.2f}초")
        print(f"   - 전체 평균 FPS: {final_stats.get('overall_fps', 0):.1f}")
        
        return all_results

# 편의 함수들
def create_ultra_processor(gpu_ids: List[int] = [0, 1], batch_size: int = 256) -> UltraBatchProcessor:
    """Ultra Batch Processor 생성"""
    return UltraBatchProcessor(
        rtmw_model_name="rtmw-dw-x-l_simcc-cocktail14_270e-384x288.onnx",
        batch_size=batch_size,
        gpu_ids=gpu_ids,
        keypoint_scale=8,
        jpeg_quality=90,
        max_vram_usage=0.85
    )

def process_video_directory(video_dir: str, gpu_ids: List[int] = [0, 1], 
                          pattern: str = "*.mp4") -> List[Dict]:
    """비디오 디렉토리 처리"""
    video_paths = list(Path(video_dir).glob(pattern))
    video_paths = [str(p) for p in video_paths]
    
    processor = create_ultra_processor(gpu_ids=gpu_ids)
    return processor.process_videos_ultra_batch(video_paths)

if __name__ == "__main__":
    # 테스트 실행
    video_dir = "/workspace01/team03/data/mmpose/jy/data/1.Training/videos/05"
    results = process_video_directory(video_dir, gpu_ids=[0, 1])
    print(f"처리 완료: {len(results)}개 비디오")
