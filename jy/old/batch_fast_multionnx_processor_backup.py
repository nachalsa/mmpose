#!/usr/bin/env python3
"""
Batch Fast Multi-ONNX Processor - GPU 배치 256 최적화 버전
A6000 x2 GPU 완전 활용, VRAM 미리 로딩, 병렬 처리
Production Ready Version - 실제 운영 환경용
"""

import os
import cv2
import h5py
import time
import torch
import queue
import logging
import shutil
import traceback
import numpy as np
import multiprocessing as mp
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional, Union
from concurrent.futures import ThreadPoolExecutor

from onnx_inferencer import YOLO11LRTMWONNXInferencer as ONNXInferencer

class BatchFastVideoProcessor:
    """배치 고속 비디오 처리기 - GPU 배치 256 최적화"""
    
    def __init__(self, 
                 rtmw_model_name: str = "rtmw-dw-x-l_simcc-cocktail14_270e-384x288.onnx",
                 yolo_device: str = "auto",
                 pose_device: str = "auto",
                 keypoint_scale: int = 8,
                 jpeg_quality: int = 90,
                 batch_size: int = 256,
                 gpu_warmup: bool = True,
                 max_vram_usage: float = 0.85):
        
        self.rtmw_model_name = rtmw_model_name
        self.yolo_device = yolo_device
        self.pose_device = pose_device
        self.keypoint_scale = keypoint_scale
        self.jpeg_quality = jpeg_quality
        self.batch_size = batch_size
        self.gpu_warmup = gpu_warmup
        self.max_vram_usage = max_vram_usage
        
        # GPU 설정 및 VRAM 정보
        if torch.cuda.is_available():
            current_device = torch.cuda.current_device()
            device = f"cuda:{current_device}"
            torch.cuda.set_device(current_device)
            
            # GPU 메모리 정보
            gpu_props = torch.cuda.get_device_properties(current_device)
            total_vram = gpu_props.total_memory / 1024**3
            max_vram = total_vram * max_vram_usage
            
            # 배치 메모리 계산 (384x288x3 float32)
            single_frame_mb = (384 * 288 * 3 * 4) / (1024 * 1024)
            batch_memory_mb = single_frame_mb * batch_size
            
            print(f"🚀 BatchVideoProcessor GPU {current_device} 초기화")
            print(f"   - GPU: {gpu_props.name}")
            print(f"   - 총 VRAM: {total_vram:.1f}GB")
            print(f"   - 최대 사용: {max_vram:.1f}GB ({max_vram_usage*100:.0f}%)")
            print(f"   - 배치 크기: {batch_size} 프레임")
            print(f"   - 배치 메모리: {batch_memory_mb:.1f}MB")
            
        else:
            device = "cpu"
            print(f"⚠️ CPU 모드로 실행")
            
        try:
            self.inferencer = ONNXInferencer(
                rtmw_onnx_path=rtmw_model_name,
                detection_device=device,
                pose_device="cuda",  # 명시적으로 CUDA 설정
                optimize_for_accuracy=True
            )
            
            # GPU 정보 및 배치 설정 출력
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.get_device_properties(current_device).total_memory / 1024**3
                gpu_name = torch.cuda.get_device_properties(current_device).name
                allocated_memory = torch.cuda.memory_allocated(current_device) / 1024**3
                print(f"🚀 BatchFastVideoProcessor 초기화 완료")
                print(f"   - GPU: {gpu_name} ({gpu_memory:.1f}GB)")
                print(f"   - 디바이스: {device} (메모리 사용: {allocated_memory:.2f}GB)")
                print(f"   - 배치 크기: {batch_size}")
                print(f"   - RTMW 모델: {rtmw_model_name}")
            
            # GPU 워밍업
            if self.gpu_warmup and torch.cuda.is_available():
                self._warmup_gpu()
                
        except Exception as e:
            print(f"❌ BatchFastVideoProcessor 초기화 실패: {e}")
            raise
    
    def _warmup_gpu(self):
        """GPU 워밍업 - 배치 처리 최적화"""
        print(f"🔥 GPU 워밍업 시작 (배치 {self.batch_size})")
        start_time = time.time()
        
        try:
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
            vram_used = torch.cuda.memory_allocated() / 1024**3
            
            print(f"✅ GPU 워밍업 완료: {warmup_time:.2f}초")
            print(f"   - VRAM 사용: {vram_used:.2f}GB")
            print(f"   - 배치 텐서: {self.batch_size} x 3 x 384 x 288")
            
        except Exception as e:
            print(f"⚠️ GPU 워밍업 실패: {e}")

    def process_video_batch_optimized(self, video_path: str, progress_callback=None) -> Optional[Dict]:
        """배치 최적화된 비디오 처리 - Production Ready"""
        start_total_time = time.time()
        
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"❌ 비디오 열기 실패: {video_path}")
                return None
            
            # 전체 프레임 수 계산
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames == 0:
                cap.release()
                return None
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            print(f"📹 비디오 정보: {total_frames}프레임, {fps:.2f}FPS, {width}x{height}")
            
            # 메모리 효율적 프레임 로딩
            frames = []
            frame_count = 0
            
            # 프레임 로딩 with 진행률
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
                frame_count += 1
                
                # 진행률 콜백
                if progress_callback and frame_count % 50 == 0:
                    progress = min(frame_count / total_frames * 0.3, 0.3)  # 로딩 30%
                    progress_callback(progress)
            
            cap.release()
            actual_frame_count = len(frames)
            
            if actual_frame_count == 0:
                print(f"❌ 유효한 프레임 없음: {video_path}")
                return None
            
            print(f"✅ 프레임 로드 완료: {actual_frame_count}개 (배치 {self.batch_size}로 처리)")
            
            # 결과 저장 리스트
            jpeg_frames = []
            keypoints_list = []
            scores_list = []
            
            # 배치별로 처리
            batch_frames = []
            batch_start_time = time.time()
            processed_frames = 0
            
            for frame_idx, frame in enumerate(frames):
                batch_frames.append(frame)
                
                # 배치가 찼거나 마지막 프레임인 경우
                if len(batch_frames) >= self.batch_size or frame_idx == len(frames) - 1:
                    
                    # GPU 배치 처리 실행
                    batch_results = self._process_frame_batch(batch_frames)
                    
                    # 결과 저장 및 JPEG 인코딩
                    for i, (frame, result) in enumerate(zip(batch_frames, batch_results)):
                        # JPEG 인코딩 (품질 설정)
                        encode_params = [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality,
                                       cv2.IMWRITE_JPEG_OPTIMIZE, 1]
                        success, buffer = cv2.imencode('.jpg', frame, encode_params)
                        
                        if success:
                            jpeg_frames.append(buffer.tobytes())
                        else:
                            # 폴백: 기본 인코딩
                            _, buffer = cv2.imencode('.jpg', frame)
                            jpeg_frames.append(buffer.tobytes())
                        
                        # 키포인트 데이터 저장 (스케일링 적용)
                        if result and len(result) == 2:
                            frame_keypoints, frame_scores = result
                            
                            # 키포인트 스케일링
                            if len(frame_keypoints) > 0 and self.keypoint_scale != 1:
                                scaled_keypoints = []
                                for person_kpts in frame_keypoints:
                                    if isinstance(person_kpts, (list, np.ndarray)) and len(person_kpts) > 0:
                                        scaled_kpts = []
                                        for j in range(0, len(person_kpts), 2):
                                            if j + 1 < len(person_kpts):
                                                x = int(person_kpts[j] * self.keypoint_scale)
                                                y = int(person_kpts[j + 1] * self.keypoint_scale)
                                                scaled_kpts.extend([x, y])
                                        scaled_keypoints.append(scaled_kpts)
                                    else:
                                        scaled_keypoints.append(person_kpts)
                                keypoints_list.append(scaled_keypoints)
                            else:
                                keypoints_list.append(frame_keypoints)
                            
                            scores_list.append(frame_scores)
                        else:
                            # 기본값 (검출된 사람 없음)
                            keypoints_list.append([[0] * 34])  # 17개 키포인트 * 2 (x,y)
                            scores_list.append([0.0])
                    
                    processed_frames += len(batch_frames)
                    
                    # 진행률 업데이트
                    if progress_callback:
                        progress = 0.3 + (processed_frames / actual_frame_count) * 0.7  # 30% + 처리 70%
                        progress_callback(min(progress, 1.0))
                    
                    # 배치 초기화
                    batch_frames = []
                    
                    # GPU 메모리 정리 (주기적)
                    if torch.cuda.is_available() and processed_frames % (self.batch_size * 4) == 0:
                        torch.cuda.empty_cache()
            
            total_processing_time = time.time() - batch_start_time
            total_time = time.time() - start_total_time
            
            # 최종 결과 구성
            result = {
                'total_frames': actual_frame_count,
                'processed_frames': processed_frames,
                'jpeg_frames': jpeg_frames,
                'keypoints': keypoints_list,
                'scores': scores_list,
                'processing_time': total_processing_time,
                'total_time': total_time,
                'fps': actual_frame_count / max(total_processing_time, 0.001),
                'video_info': {
                    'original_fps': fps,
                    'resolution': f"{width}x{height}",
                    'duration': actual_frame_count / max(fps, 1.0)
                }
            }
            
            print(f"✅ 비디오 처리 완료:")
            print(f"   - 처리: {actual_frame_count}프레임 ({total_processing_time:.2f}초)")
            print(f"   - 속도: {result['fps']:.1f} FPS")
            print(f"   - 전체: {total_time:.2f}초")
            
            return result
            
        except Exception as e:
            print(f"💥 배치 비디오 처리 오류 ({video_path}): {e}")
            traceback.print_exc()
            return None
    
    def _process_frame_batch(self, frames: List[np.ndarray]) -> List[Optional[Tuple[List[List[int]], List[float]]]]:
        """프레임 배치 처리 - Production Ready GPU 배치 추론"""
        results = []
        batch_start_time = time.time()
        
        try:
            if not frames:
                return []
            
            print(f"🔥 배치 프레임 처리: {len(frames)}개")
            
            # 각 프레임별 결과 초기화
            for _ in range(len(frames)):
                results.append(None)
            
            # GPU 배치 추론을 위한 준비
            all_crops = []  # 모든 크롭 이미지들
            crop_frame_mapping = []  # 각 크롭이 어느 프레임에서 왔는지
            frame_person_counts = []  # 각 프레임에서 검출된 사람 수
            
            # 1단계: 모든 프레임에서 사람 검출 (YOLO)
            detection_start = time.time()
            for frame_idx, frame in enumerate(frames):
                try:
                    # YOLO 사람 검출 (고정확도)
                    person_boxes = self.inferencer.detect_persons_high_accuracy(frame)
                    person_count = len(person_boxes)
                    frame_person_counts.append(person_count)
                    
                    if person_count == 0:
                        continue
                    
                    # 각 사람 영역을 크롭하여 배치에 추가
                    frame_crops = []
                    for person_idx, bbox in enumerate(person_boxes):
                        try:
                            x1, y1, x2, y2 = map(int, bbox[:4])
                            x1, y1 = max(0, x1), max(0, y1)
                            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                            
                            if x2 > x1 + 10 and y2 > y1 + 10:  # 최소 크기 체크
                                crop = frame[y1:y2, x1:x2]
                                
                                # RTMW 전처리 적용
                                processed_crop = self.inferencer._preprocess_image_for_pose(crop)
                                if processed_crop is not None:
                                    all_crops.append(processed_crop)
                                    crop_frame_mapping.append(frame_idx)
                                    frame_crops.append(len(all_crops) - 1)  # crop 인덱스 저장
                                    
                        except Exception as e:
                            print(f"⚠️ 프레임 {frame_idx} 사람 {person_idx} 크롭 실패: {e}")
                            continue
                    
                except Exception as e:
                    print(f"⚠️ 프레임 {frame_idx} YOLO 검출 실패: {e}")
                    frame_person_counts.append(0)
            
            detection_time = time.time() - detection_start
            print(f"   YOLO 검출: {len(all_crops)}개 사람 ({detection_time:.2f}초)")
            
            # 2단계: GPU 배치 포즈 추정 (ONNX)
            pose_start = time.time()
            batch_keypoints = []
            batch_scores = []
            
            if all_crops:
                # 배치 크기 조정 (GPU 메모리에 맞게)
                pose_batch_size = min(64, len(all_crops))  # A6000 기준 최적화
                
                for i in range(0, len(all_crops), pose_batch_size):
                    batch_slice = all_crops[i:i+pose_batch_size]
                    
                    try:
                        # ONNX 배치 추론
                        kpts_batch, scores_batch = self.inferencer.estimate_pose_batch(batch_slice)
                        
                        batch_keypoints.extend(kpts_batch)
                        batch_scores.extend(scores_batch)
                        
                    except Exception as e:
                        print(f"⚠️ 배치 포즈 추정 실패, 개별 처리로 폴백: {e}")
                        
                        # 폴백: 개별 처리
                        for crop in batch_slice:
                            try:
                                kpts, scores = self.inferencer.estimate_pose_on_crop(crop)
                                batch_keypoints.append(kpts)
                                batch_scores.append(scores)
                            except Exception as e2:
                                print(f"⚠️ 개별 포즈 추정도 실패: {e2}")
                                batch_keypoints.append(np.zeros((17, 2)))  # 기본값
                                batch_scores.append(np.zeros(17))
            
            pose_time = time.time() - pose_start
            print(f"   포즈 추정: {len(batch_keypoints)}개 결과 ({pose_time:.2f}초)")
            
            # 3단계: 결과를 프레임별로 재구성
            if batch_keypoints:
                crop_idx = 0
                
                for frame_idx in range(len(frames)):
                    person_count = frame_person_counts[frame_idx]
                    
                    if person_count == 0:
                        results[frame_idx] = ([], [])  # 빈 결과
                        continue
                    
                    frame_keypoints = []
                    frame_scores = []
                    
                    # 해당 프레임의 모든 사람 결과 수집
                    for person_idx in range(person_count):
                        if crop_idx < len(batch_keypoints):
                            kpts = batch_keypoints[crop_idx]
                            scores = batch_scores[crop_idx]
                            
                            # 키포인트 형식 변환 (리스트로)
                            if isinstance(kpts, np.ndarray):
                                if kpts.ndim == 2:  # (17, 2) 형태
                                    kpts_flat = []
                                    for joint in kpts:
                                        kpts_flat.extend([float(joint[0]), float(joint[1])])
                                    frame_keypoints.append(kpts_flat)
                                else:
                                    frame_keypoints.append(kpts.flatten().tolist())
                            else:
                                frame_keypoints.append(kpts)
                            
                            if isinstance(scores, np.ndarray):
                                frame_scores.append(scores.tolist())
                            else:
                                frame_scores.append(scores)
                            
                            crop_idx += 1
                        else:
                            # 데이터 부족 시 기본값
                            frame_keypoints.append([0.0] * 34)  # 17 joints * 2 coords
                            frame_scores.append([0.0] * 17)
                    
                    results[frame_idx] = (frame_keypoints, frame_scores)
            
            # 빈 결과들을 기본값으로 채움
            for i, result in enumerate(results):
                if result is None:
                    results[i] = ([], [])
            
            total_time = time.time() - batch_start_time
            print(f"✅ 배치 처리 완료: {len(frames)}프레임 ({total_time:.2f}초)")
            
            return results
            
        except Exception as e:
            print(f"💥 배치 처리 오류: {e}")
            import traceback
            traceback.print_exc()
            
            # 오류 시 빈 결과 반환
            return [None] * len(frames)
                            batch_keypoints.append(kpts)
                            batch_scores.append(scores)
                            
                    except Exception as e:
                        print(f"⚠️ 배치 포즈 추정 실패, 개별 처리로 폴백: {e}")
                        # 폴백: 개별 처리
                        for crop in batch_slice:
                            try:
                                kpts, scores = self.inferencer.estimate_pose_on_crop(crop)
                                batch_keypoints.append(kpts)
                                batch_scores.append(scores)
                            except Exception as e2:
                                print(f"⚠️ 개별 포즈 추정도 실패: {e2}")
                                batch_keypoints.append(np.zeros((133, 2)))
                                batch_scores.append(np.zeros(133))
            
            # 3단계: 결과를 프레임별로 재구성
            crop_idx = 0
            for frame_idx in range(len(frames)):
                frame_keypoints = []
                frame_scores = []
                
                person_count = frame_person_counts[frame_idx]
                
                for person_idx in range(person_count):
                    if crop_idx < len(batch_keypoints):
                        # 키포인트 스케일링
                        kpts = batch_keypoints[crop_idx]
                        scores = batch_scores[crop_idx]
                        
                        # 133 키포인트를 51로 변환 (COCO 17 keypoints * 3)
                        if len(kpts) >= 17:
                            scaled_kpts = []
                            for i in range(17):
                                if i < len(kpts):
                                    x, y = kpts[i]
                                    # 신뢰도는 scores에서 가져오기
                                    conf = scores[i] if i < len(scores) else 0.0
                                    scaled_kpts.extend([
                                        int(x * self.keypoint_scale),
                                        int(y * self.keypoint_scale),
                                        int(conf * self.keypoint_scale)
                                    ])
                            
                            # 51개로 맞추기
                            while len(scaled_kpts) < 51:
                                scaled_kpts.append(0)
                            
                            frame_keypoints.append(scaled_kpts[:51])
                            frame_scores.append(float(np.mean(scores)))
                        
                        crop_idx += 1
                
                # 키포인트가 없는 경우 기본값
                if not frame_keypoints:
                    frame_keypoints.append([0] * 51)
                    frame_scores.append(0.0)
                
                results.append((frame_keypoints, frame_scores))
        
        except Exception as e:
            print(f"💥 배치 처리 오류: {e}")
            # 완전 폴백: 기존 방식
            for frame in frames:
                result = self._process_single_frame(frame)
                results.append(result)
        
        return results
    
    def _process_single_frame(self, frame: np.ndarray) -> Optional[Tuple[List[List[int]], List[float]]]:
        """단일 프레임 처리"""
        try:
            # 추론 실행
            result = self.inferencer.process_frame(frame)
            
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
            
            return frame_keypoints, frame_scores
            
        except Exception as e:
            print(f"⚠️ 단일 프레임 처리 오류: {e}")
            return [[0] * 51], [0.0]

def batch_gpu_worker(
    gpu_id: int,
    task_queue: mp.Queue, 
    result_queue: mp.Queue, 
    progress_queue: mp.Queue,
    config: Dict
):
    """배치 GPU 워커 - 배치 256 최적화 + 진행률 추적"""
    try:
        # GPU 설정 - 강제로 특정 GPU만 보이도록 설정
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        
        # PyTorch CUDA 초기화
        if torch.cuda.is_available():
            torch.cuda.set_device(0)  # CUDA_VISIBLE_DEVICES로 설정했으므로 0번이 해당 GPU
            device = f"cuda:0"
            
            # GPU 정보 출력
            gpu_props = torch.cuda.get_device_properties(0)
            gpu_name = gpu_props.name
            gpu_memory = gpu_props.total_memory / 1024**3
            print(f"🚀 배치 GPU {gpu_id} 워커 시작")
            print(f"   - GPU: {gpu_name} ({gpu_memory:.1f}GB)")
            print(f"   - 배치 크기: {config['batch_size']}")
            print(f"   - 디바이스: {device}")
            
            # GPU 메모리 예약 및 워밍업 (더 큰 배치)
            print(f"🔥 GPU {gpu_id} 워밍업 중...")
            dummy_tensor = torch.randn(config['batch_size'], 3, 384, 288).cuda()
            # 몇 번의 연산으로 GPU 활성화
            for _ in range(5):
                _ = dummy_tensor * 2.0
                _ = torch.nn.functional.avg_pool2d(dummy_tensor, 2)
            
            del dummy_tensor
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            
            warmup_vram = torch.cuda.memory_allocated(0) / 1024**3
            print(f"✅ GPU {gpu_id} 워밍업 완료 (VRAM: {warmup_vram:.2f}GB)")
            
        else:
            device = "cpu"
            print(f"⚠️ 배치 GPU {gpu_id} 사용 불가 - CPU 모드")
            
        # 배치 최적화 프로세서 초기화
        processor = BatchFastVideoProcessor(
            rtmw_model_name=config['rtmw_model_name'],
            yolo_device=device,
            pose_device="cuda",
            keypoint_scale=config['keypoint_scale'],
            jpeg_quality=config['jpeg_quality'],
            batch_size=config['batch_size'],
            gpu_warmup=False,  # 이미 워밍업했음
            max_vram_usage=0.85
        )
        
        processed_count = 0
        
        while True:
            try:
                job = task_queue.get(timeout=10.0)  # 더 긴 타임아웃
                if job is None:  # 종료 신호
                    if torch.cuda.is_available():
                        final_vram = torch.cuda.memory_allocated(0) / 1024**3
                        print(f"� 배치 GPU {gpu_id} 워커 종료")
                        print(f"   - 처리 완료: {processed_count}개")
                        print(f"   - 최종 VRAM: {final_vram:.2f}GB")
                        torch.cuda.empty_cache()
                    else:
                        print(f"🔥 GPU {gpu_id} 워커 종료 (처리: {processed_count}개)")
                    break
                
                job_id, item_type, item_id, video_path = job
                
                print(f"📹 배치 GPU {gpu_id}: {item_type}{item_id:04d} 처리 시작...")
                
                # 진행률 추적 콜백 함수
                def progress_callback(current_frame, total_frames):
                    try:
                        progress_queue.put((gpu_id, job_id, current_frame, total_frames), timeout=0.1)
                    except queue.Full:
                        pass  # 진행률 큐가 가득 찬 경우 스킵
                
                start_time = time.time()
                arrays = processor.process_video_batch_optimized(video_path, progress_callback)
                process_time = time.time() - start_time
                
                if arrays:
                    avg_fps = arrays.get('avg_fps', 0)
                    batch_count = arrays.get('batch_count', 0)
                    frame_count = arrays.get('frame_count', 0)
                    
                    if torch.cuda.is_available():
                        current_vram = torch.cuda.memory_allocated(0) / 1024**3
                        print(f"✅ 배치 GPU {gpu_id}: {item_type}{item_id:04d} 완료")
                        print(f"   - 프레임: {frame_count}개")
                        print(f"   - 처리 시간: {process_time:.2f}초")
                        print(f"   - 평균 FPS: {avg_fps:.1f}")
                        print(f"   - 배치 수: {batch_count}개")
                        print(f"   - VRAM 사용: {current_vram:.2f}GB")
                    else:
                        print(f"✅ GPU {gpu_id}: {item_type}{item_id:04d} 완료 ({frame_count} frames, {avg_fps:.1f} FPS)")
                else:
                    print(f"❌ 배치 GPU {gpu_id}: {item_type}{item_id:04d} 실패")
                
                result_queue.put((job_id, item_type, item_id, video_path, arrays))
                processed_count += 1
                
                # 주기적 GPU 메모리 정리 (배치 처리 후)
                if torch.cuda.is_available() and processed_count % 3 == 0:
                    torch.cuda.empty_cache()
                    
            except queue.Empty:
                if torch.cuda.is_available():
                    timeout_vram = torch.cuda.memory_allocated(0) / 1024**3
                    print(f"⏰ 배치 GPU {gpu_id} 타임아웃 (처리: {processed_count}개, VRAM: {timeout_vram:.2f}GB)")
                else:
                    print(f"⏰ GPU {gpu_id} 타임아웃 (처리: {processed_count}개)")
                break
            except Exception as e:
                print(f"💥 배치 GPU {gpu_id} 워커 오류: {e}")
                import traceback
                traceback.print_exc()
                result_queue.put((job_id, item_type, item_id, video_path, None))
                continue
                
    except Exception as e:
        print(f"❌ 배치 GPU {gpu_id} 워커 초기화 실패: {e}")
        import traceback
        traceback.print_exc()
        
        while True:
            try:
                job = task_queue.get(timeout=5.0)
                if job is None:  # 종료 신호
                    if torch.cuda.is_available():
                        memory_used = torch.cuda.memory_allocated(0) / 1024**3
                        memory_cached = torch.cuda.memory_reserved(0) / 1024**3
                        print(f"🔥 배치 GPU {gpu_id} 워커 종료")
                        print(f"   - 처리: {processed_count}개")
                        print(f"   - GPU 메모리: {memory_used:.2f}GB 사용, {memory_cached:.2f}GB 캐시")
                        torch.cuda.empty_cache()
                    else:
                        print(f"🔥 배치 GPU {gpu_id} 워커 종료 (처리: {processed_count}개)")
                    break
                
                job_id, item_type, item_id, video_path = job
                
                start_time = time.time()
                arrays = processor.process_video_batch_optimized(video_path)
                process_time = time.time() - start_time
                
                if arrays:
                    fps = arrays['frame_count'] / process_time
                    print(f"✅ 배치 GPU {gpu_id}: {item_type}{item_id:04d} 완료")
                    print(f"   - {arrays['frame_count']} 프레임, {process_time:.2f}s, {fps:.1f} FPS")
                else:
                    print(f"❌ 배치 GPU {gpu_id}: {item_type}{item_id:04d} 실패")
                
                result_queue.put((job_id, item_type, item_id, video_path, arrays))
                processed_count += 1
                
                # GPU 메모리 정리 (주기적으로)
                if torch.cuda.is_available() and processed_count % 5 == 0:
                    torch.cuda.empty_cache()
                    
            except queue.Empty:
                if torch.cuda.is_available():
                    memory_used = torch.cuda.memory_allocated(0) / 1024**3
                    memory_cached = torch.cuda.memory_reserved(0) / 1024**3
                    print(f"⏰ 배치 GPU {gpu_id} 워커 타임아웃")
                    print(f"   - 처리: {processed_count}개")
                    print(f"   - GPU 메모리: {memory_used:.2f}GB 사용, {memory_cached:.2f}GB 캐시")
                else:
                    print(f"⏰ 배치 GPU {gpu_id} 워커 타임아웃 (처리: {processed_count}개)")
                break
            except Exception as e:
                print(f"💥 배치 GPU {gpu_id} 워커 오류: {e}")
                result_queue.put((job_id, item_type, item_id, video_path, None))
                continue
                
    except Exception as e:
        print(f"❌ 배치 GPU {gpu_id} 워커 초기화 실패: {e}")
        import traceback
        traceback.print_exc()

def batch_cpu_worker(
    result_queue: mp.Queue, 
    output_dir: Path,
    config: Dict
):
    """배치 CPU 후처리 워커"""
    print(f"💻 배치 CPU 워커 시작")
    processed_count = 0
    
    while True:
        try:
            result = result_queue.get(timeout=15)  # 배치 처리 고려하여 타임아웃 증가
            if result is None:  # 종료 신호
                print(f"💻 배치 CPU 워커 종료 (처리: {processed_count}개)")
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
                # HDF5로 배치 효율적 저장
                frames_path = item_dir / "frames.h5"
                poses_path = item_dir / "poses.h5"
                
                start_save_time = time.time()
                
                # JPEG 프레임 저장
                with h5py.File(frames_path, 'w') as f:
                    jpeg_vlen_dtype = h5py.vlen_dtype(np.uint8)
                    jpeg_dataset = f.create_dataset('frames', (len(arrays['jpeg_frames']),), dtype=jpeg_vlen_dtype)
                    
                    for i, jpeg_data in enumerate(arrays['jpeg_frames']):
                        if isinstance(jpeg_data, bytes) and len(jpeg_data) > 0:
                            jpeg_dataset[i] = np.frombuffer(jpeg_data, dtype=np.uint8)
                        else:
                            jpeg_dataset[i] = np.array([], dtype=np.uint8)
                
                # 포즈 데이터 저장
                with h5py.File(poses_path, 'w') as f:
                    max_persons = max(len(kpts) for kpts in arrays['keypoints']) if arrays['keypoints'] else 1
                    keypoints_array = np.zeros((len(arrays['keypoints']), max_persons, 51), dtype=np.float32)
                    scores_array = np.zeros((len(arrays['scores']), max_persons), dtype=np.float32)
                    
                    for i, (kpts_frame, scores_frame) in enumerate(zip(arrays['keypoints'], arrays['scores'])):
                        for j, (kpts, score) in enumerate(zip(kpts_frame, scores_frame)):
                            if j < max_persons:
                                keypoints_array[i, j, :len(kpts)] = kpts[:51]
                                scores_array[i, j] = score
                    
                    f.create_dataset('keypoints', data=keypoints_array, compression='gzip', compression_opts=9)
                    f.create_dataset('scores', data=scores_array, compression='gzip', compression_opts=9)
                    f.attrs['frame_count'] = arrays['frame_count']
                    f.attrs['keypoint_scale'] = config['keypoint_scale']
                    f.attrs['batch_size'] = config['batch_size']
                
                save_time = time.time() - start_save_time
                
                print(f"✅ 배치 저장 완료: {output_key}")
                print(f"   - {arrays['frame_count']} 프레임, {save_time:.2f}s")
                processed_count += 1
                
            except Exception as e:
                print(f"💥 배치 저장 오류 ({output_key}): {e}")
                continue
                
        except queue.Empty:
            continue
        except Exception as e:
            print(f"💻 배치 CPU 워커 오류: {e}")
            continue

class BatchFastMultiONNXProcessor:
    """배치 고속 멀티 ONNX 프로세서 - GPU 배치 256 최적화"""
    
    def __init__(self, 
                 data_root: str = "data/1.Training",
                 output_dir: str = "batch_fast_multionnx_output",
                 rtmw_model_name: str = "rtmw-dw-x-l_simcc-cocktail14_270e-384x288.onnx",
                 direction: str = "F",
                 item_types: List[str] = ["WORD"],
                 num_cpu_workers: int = 8,
                 batch_size: int = 256,
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
        self.batch_size = batch_size
        self.enable_hdf5_batch = enable_hdf5_batch
        
        # 배치 설정 딕셔너리
        self.config = {
            'rtmw_model_name': rtmw_model_name,
            'yolo_device': yolo_device,
            'pose_device': pose_device,
            'keypoint_scale': keypoint_scale,
            'jpeg_quality': jpeg_quality,
            'batch_size': batch_size
        }
        
        # 출력 디렉토리 생성
        self.video_output_dir = self.output_dir / "videos"
        self.hdf5_output_dir = self.output_dir / "batches"
        self.video_output_dir.mkdir(parents=True, exist_ok=True)
        self.hdf5_output_dir.mkdir(parents=True, exist_ok=True)
        
        # 로깅
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
        print(f"🚀 Batch Fast Multi-ONNX Processor 초기화 완료")
        print(f"   - 데이터 루트: {self.data_root}")
        print(f"   - 출력 디렉토리: {self.output_dir}")
        print(f"   - 배치 크기: {batch_size} (GPU 최적화)")
        print(f"   - 방향: {direction}, 타입: {item_types}")
        print(f"   - CPU 워커: {num_cpu_workers}")

    def collect_all_videos(self) -> List[Tuple[str, int, str]]:
        """모든 비디오 수집"""
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
        
        self.logger.info(f"📊 총 {len(all_videos)}개 비디오 발견 (배치 {self.batch_size} 최적화)")
        return all_videos

    def extract_item_info_from_path(self, video_path: str) -> Optional[Tuple[str, int]]:
        """경로에서 아이템 정보 추출"""
        filename = Path(video_path).stem
        try:
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
        """병렬 비디오 처리 - 배치 256 최적화"""
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
        
        self.logger.info(f"🚀 배치 병렬 처리 시작:")
        self.logger.info(f"   - {num_gpus}개 GPU (배치 {self.batch_size})")
        self.logger.info(f"   - {self.num_cpu_workers}개 CPU 워커")
        
        # 큐 생성 (배치 처리 + 진행률 추적 최적화)
        task_queue = mp.Queue(maxsize=num_gpus * 3)
        result_queue = mp.Queue(maxsize=self.num_cpu_workers * 2)
        progress_queue = mp.Queue(maxsize=2000)  # 진행률 추적용
        
        # 작업 큐에 비디오 추가
        for i, (item_type, item_id, video_path) in enumerate(all_videos):
            task_queue.put((i, item_type, item_id, video_path))
        
        # 배치 GPU 워커 시작 (진행률 큐 추가)
        gpu_workers = []
        for gpu_id in range(num_gpus):
            worker = mp.Process(target=batch_gpu_worker, args=(
                gpu_id, task_queue, result_queue, progress_queue, self.config
            ))
            worker.start()
            gpu_workers.append(worker)
        
        # 배치 CPU 워커 시작
        cpu_workers = []
        for i in range(self.num_cpu_workers):
            worker = mp.Process(target=batch_cpu_worker, args=(
                result_queue, self.video_output_dir, self.config
            ))
            worker.start()
            cpu_workers.append(worker)
        
        # 고급 진행률 추적 시스템
        processed_videos = self._track_batch_progress(all_videos, result_queue, progress_queue, num_gpus)
        
        # 워커 종료
        self.logger.info("🛑 배치 워커 종료 중...")
        for _ in range(num_gpus):
            task_queue.put(None)
        for _ in range(self.num_cpu_workers):
            result_queue.put(None)
        
        # 워커 대기
        for worker in gpu_workers:
            worker.join(timeout=15)
            if worker.is_alive():
                worker.terminate()
        
        for worker in cpu_workers:
            worker.join(timeout=15)
            if worker.is_alive():
                worker.terminate()
        
        self.logger.info(f"✅ 배치 비디오 처리 완료: {len(processed_videos)}개")
        
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
                        
                        frames_path = item_dir / "frames.h5"
                        poses_path = item_dir / "poses.h5"
                        
                        if frames_path.exists() and poses_path.exists():
                            with h5py.File(frames_path, 'r') as src_f, \
                                 h5py.File(poses_path, 'r') as src_p:
                                
                                f_frames.copy(src_f['frames'], key)
                                f_poses.copy(src_p['keypoints'], f"{key}/keypoints")
                                f_poses.copy(src_p['scores'], f"{key}/scores")
                
                self.logger.info(f"✅ HDF5 배치 생성: {batch_name} ({len(batch_keys)}개)")
                
            except Exception as e:
                self.logger.error(f"❌ HDF5 배치 생성 실패 ({batch_name}): {e}")

    def _track_batch_progress(self, all_videos: List, result_queue: mp.Queue, 
                             progress_queue: mp.Queue, num_gpus: int):
        """고급 배치 진행률 추적 시스템"""
        processed_videos = []
        video_progress = {}  # {job_id: (current_frame, total_frames)}
        gpu_stats = {i: {'videos': 0, 'frames': 0, 'vram': 0} for i in range(num_gpus)}
        
        device_info = f"{num_gpus}xGPU(배치{self.batch_size}), {self.num_cpu_workers}xCPU"
        
        with tqdm(total=len(all_videos), desc=f"🚀 Batch Multi-ONNX ({device_info})") as pbar:
            timeout_count = 0
            start_time = time.time()
            last_update = start_time
            
            while len(processed_videos) < len(all_videos):
                try:
                    # 결과 큐 체크 (높은 우선순위)
                    try:
                        result = result_queue.get(timeout=0.2)
                        job_id, item_type, item_id, video_path, arrays = result
                        
                        if arrays:
                            output_key = f"{item_type}{item_id:04d}"
                            processed_videos.append(output_key)
                            
                            # 통계 업데이트
                            avg_fps = arrays.get('avg_fps', 0)
                            frame_count = arrays.get('frame_count', 0)
                            
                            # 진행률 업데이트
                            pbar.set_postfix({
                                '완료': len(processed_videos),
                                '최근FPS': f"{avg_fps:.1f}",
                                'VRAM': f"{self._get_total_vram_usage():.1f}GB",
                                '처리중': len(video_progress)
                            })
                        
                        pbar.update(1)
                        timeout_count = 0
                        last_update = time.time()
                        
                    except queue.Empty:
                        pass
                    
                    # 진행률 큐 체크 (프레임 단위 진행률)
                    frame_updates = 0
                    try:
                        while frame_updates < 10:  # 한 번에 최대 10개 업데이트 처리
                            progress_update = progress_queue.get(timeout=0.01)
                            gpu_id, job_id, current_frame, total_frames = progress_update
                            video_progress[job_id] = (current_frame, total_frames)
                            frame_updates += 1
                            
                    except queue.Empty:
                        pass
                    
                    # 주기적으로 상세 정보 업데이트 (1초마다)
                    current_time = time.time()
                    if current_time - last_update > 1.0:
                        # 프레임 진행률 계산 (안전한 나눗셈)
                        if video_progress:
                            total_current_frames = sum(p[0] for p in video_progress.values())
                            total_target_frames = sum(p[1] for p in video_progress.values())
                            
                            if total_target_frames > 0:
                                frame_progress = (total_current_frames / total_target_frames) * 100
                            else:
                                frame_progress = 0.0
                            
                            # GPU 활용률 계산
                            gpu_utilization = self._estimate_gpu_utilization()
                            
                            pbar.set_postfix({
                                '완료': len(processed_videos),
                                '진행': f"{frame_progress:.1f}%",
                                'GPU사용': f"{gpu_utilization:.0f}%",
                                'VRAM': f"{self._get_total_vram_usage():.1f}GB"
                            })
                        
                        last_update = current_time
                            total_current = sum(p[0] for p in video_progress.values())
                            total_frames_all = sum(p[1] for p in video_progress.values())
                            
                            if total_frames_all > 0:
                                frame_progress = total_current / total_frames_all * 100
                                
                                # GPU 활용률 추정
                                gpu_util = self._estimate_gpu_utilization()
                                
                                pbar.set_postfix({
                                    '완료': len(processed_videos),
                                    '프레임': f"{frame_progress:.1f}%",
                                    'GPU활용': f"{gpu_util:.0f}%",
                                    'VRAM': f"{self._get_total_vram_usage():.1f}GB"
                                })
                        last_update = current_time
                    
                    timeout_count += 1
                    if timeout_count > 300:  # 30초 타임아웃
                        elapsed = current_time - start_time
                        self.logger.warning(f"⏰ 배치 처리 타임아웃 ({elapsed:.0f}초 경과)")
                        self.logger.warning(f"   현재까지 완료: {len(processed_videos)}/{len(all_videos)}개")
                        self.logger.warning(f"   처리 중인 비디오: {len(video_progress)}개")
                        break
                        
                    time.sleep(0.1)
                    
                except Exception as e:
                    self.logger.error(f"진행률 추적 오류: {e}")
                    break
        
        total_time = time.time() - start_time
        avg_video_per_sec = len(processed_videos) / total_time if total_time > 0 else 0
        
        self.logger.info(f"✅ 배치 처리 완료: {len(processed_videos)}개")
        self.logger.info(f"⏱️ 총 처리 시간: {total_time:.2f}초")
        self.logger.info(f"⚡ 평균 처리 속도: {avg_video_per_sec:.2f} 비디오/초")
        
        return processed_videos

    def _get_total_vram_usage(self) -> float:
        """총 VRAM 사용량 반환 (GB)"""
        if not torch.cuda.is_available():
            return 0.0
        
        total_usage = 0.0
        for i in range(torch.cuda.device_count()):
            try:
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                total_usage += allocated
            except:
                pass
        return total_usage

    def _estimate_gpu_utilization(self) -> float:
        """GPU 사용률 추정 (VRAM 기반)"""
        if not torch.cuda.is_available():
            return 0.0
        
        total_allocated = 0.0
        total_capacity = 0.0
        
        for i in range(torch.cuda.device_count()):
            try:
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                capacity = torch.cuda.get_device_properties(i).total_memory / 1024**3
                total_allocated += allocated
                total_capacity += capacity
            except:
                continue
        
        if total_capacity > 0:
            utilization = (total_allocated / total_capacity) * 100
            return min(utilization * 2, 100)  # VRAM 사용률의 2배로 추정 (실제 연산 고려)
        return 0.0

    def print_statistics(self, processed_videos: List[str]):
        """최종 통계 출력"""
        self.logger.info("="*80)
        self.logger.info("📊 Batch Fast Multi-ONNX 처리 통계:")
        self.logger.info(f"   - 처리된 비디오: {len(processed_videos)}개")
        self.logger.info(f"   - GPU 배치 크기: {self.batch_size} 프레임")
        self.logger.info(f"   - 총 VRAM: {self._get_total_vram_usage():.1f}GB 사용")
        self.logger.info(f"   - 출력 디렉토리: {self.output_dir}")
        self.logger.info(f"   - 개별 비디오 파일: {self.video_output_dir}")
        if self.enable_hdf5_batch:
            hdf5_files = len(list(self.hdf5_output_dir.glob("*.h5")))
            self.logger.info(f"   - HDF5 배치 파일: {hdf5_files}개")
        self.logger.info("="*80)

def main():
    """메인 실행 함수 - 배치 256 최적화"""
    print("🚀 Batch Fast Multi-ONNX Processor")
    print("⚡ A6000 x2 GPU 배치 256 최적화 버전")
    print("=" * 60)
    
    # 배치 최적화 설정
    processor = BatchFastMultiONNXProcessor(
        data_root="/workspace01/team03/data/mmpose/jy/data/1.Training",
        output_dir="/workspace01/team03/data/batch_fast_multionnx_output",
        direction="F",
        item_types=["WORD"],
        num_cpu_workers=8,       # A6000 환경에 최적화
        batch_size=256,          # GPU 배치 크기
        yolo_device="auto",
        pose_device="auto",
        keypoint_scale=8,
        jpeg_quality=90,
        enable_hdf5_batch=True
    )
    
    # 처리할 비디오 수 설정
    choice = input("처리할 비디오 수를 제한하시겠습니까? (y/N): ").strip().lower()
    max_videos = None
    if choice == 'y':
        try:
            max_videos = int(input("최대 처리할 비디오 수 입력: "))
        except ValueError:
            max_videos = None
    
    # 배치 처리 실행
    processor.process_videos_parallel(max_videos=max_videos)

if __name__ == "__main__":
    main()
