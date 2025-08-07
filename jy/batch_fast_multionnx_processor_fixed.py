#!/usr/bin/env python3
"""
Batch Fast Multi-ONNX Processor - GPU 배치 256 최적화 버전 (수정됨)
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
                    results[i] = ([[0.0] * 34], [[0.0] * 17])
            
            total_time = time.time() - batch_start_time
            print(f"✅ 배치 처리 완료: {len(frames)}프레임 ({total_time:.2f}초)")
            
            return results
            
        except Exception as e:
            print(f"💥 배치 처리 오류: {e}")
            traceback.print_exc()
            
            # 오류 시 빈 결과 반환
            return [([[0.0] * 34], [[0.0] * 17])] * len(frames)
    
    def _process_single_frame(self, frame: np.ndarray) -> Optional[Tuple[List[List[int]], List[float]]]:
        """단일 프레임 처리 (폴백 용도)"""
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
                        if len(kpts) >= 34:  # 17 keypoints * 2
                            frame_keypoints.append(kpts[:34])
                            frame_scores.append(person.get('score', 1.0))
            
            # 키포인트가 없는 경우 기본값
            if not frame_keypoints:
                frame_keypoints.append([0.0] * 34)
                frame_scores.append(0.0)
            
            return frame_keypoints, frame_scores
            
        except Exception as e:
            print(f"⚠️ 단일 프레임 처리 오류: {e}")
            return ([[0.0] * 34], [0.0])

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
        torch.cuda.init()
        torch.cuda.set_device(0)  # 여기서는 0이 실제 gpu_id에 해당
        
        print(f"🚀 GPU {gpu_id} 워커 시작")
        print(f"   - CUDA 디바이스: {torch.cuda.current_device()}")
        print(f"   - GPU 이름: {torch.cuda.get_device_name(0)}")
        print(f"   - VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
        
        # 배치 처리기 초기화
        processor = BatchFastVideoProcessor(
            rtmw_model_name=config['rtmw_model_name'],
            batch_size=config['batch_size'],
            keypoint_scale=config.get('keypoint_scale', 8),
            gpu_warmup=config.get('gpu_warmup', True)
        )
        
        total_processed = 0
        worker_start_time = time.time()
        
        while True:
            try:
                # 작업 큐에서 가져오기 (타임아웃 설정)
                try:
                    task = task_queue.get(timeout=5.0)
                    if task is None:  # 종료 신호
                        break
                except queue.Empty:
                    continue
                
                video_path, task_id = task
                
                # 진행률 추적을 위한 콜백
                def progress_callback(progress):
                    try:
                        progress_queue.put({
                            'task_id': task_id,
                            'gpu_id': gpu_id,
                            'progress': progress,
                            'status': 'processing'
                        })
                    except Exception as e:
                        print(f"⚠️ 진행률 업데이트 실패: {e}")
                
                print(f"🔥 GPU {gpu_id} 처리 시작: {Path(video_path).name}")
                start_time = time.time()
                
                # 비디오 처리
                result = processor.process_video_batch_optimized(video_path, progress_callback)
                
                processing_time = time.time() - start_time
                
                if result:
                    fps_achieved = result.get('fps', 0)
                    frame_count = result.get('total_frames', 0)
                    
                    print(f"✅ GPU {gpu_id} 완료: {frame_count}프레임, {fps_achieved:.1f}FPS ({processing_time:.2f}초)")
                    
                    # 결과 큐에 저장
                    result_item = {
                        'task_id': task_id,
                        'gpu_id': gpu_id,
                        'video_path': video_path,
                        'result': result,
                        'processing_time': processing_time,
                        'fps_achieved': fps_achieved,
                        'frame_count': frame_count,
                        'status': 'completed'
                    }
                    
                else:
                    print(f"❌ GPU {gpu_id} 처리 실패: {Path(video_path).name}")
                    result_item = {
                        'task_id': task_id,
                        'gpu_id': gpu_id,
                        'video_path': video_path,
                        'result': None,
                        'processing_time': processing_time,
                        'status': 'failed'
                    }
                
                # 결과 전송
                result_queue.put(result_item)
                total_processed += 1
                
                # 진행률 완료 신호
                try:
                    progress_queue.put({
                        'task_id': task_id,
                        'gpu_id': gpu_id,
                        'progress': 1.0,
                        'status': 'completed'
                    })
                except Exception as e:
                    print(f"⚠️ 완료 신호 전송 실패: {e}")
                
                # 메모리 정리
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                print(f"💥 GPU {gpu_id} 작업 오류: {e}")
                traceback.print_exc()
                
                # 오류 결과 전송
                try:
                    result_queue.put({
                        'task_id': task_id if 'task_id' in locals() else -1,
                        'gpu_id': gpu_id,
                        'video_path': video_path if 'video_path' in locals() else 'unknown',
                        'result': None,
                        'status': 'error',
                        'error': str(e)
                    })
                except:
                    pass
        
        worker_time = time.time() - worker_start_time
        print(f"🏁 GPU {gpu_id} 워커 종료: {total_processed}개 처리 ({worker_time:.2f}초)")
        
    except Exception as e:
        print(f"💥 GPU {gpu_id} 워커 초기화 실패: {e}")
        traceback.print_exc()

def process_videos_dual_gpu_batch(
    video_paths: List[str],
    output_dir: str,
    config: Dict,
    progress_callback=None
) -> Dict:
    """듀얼 GPU 배치 256 비디오 처리 - Production Ready"""
    
    start_time = time.time()
    total_videos = len(video_paths)
    
    print(f"🚀 듀얼 GPU 배치 256 처리 시작")
    print(f"   - 총 비디오: {total_videos}개")
    print(f"   - 출력 디렉토리: {output_dir}")
    print(f"   - 배치 크기: {config['batch_size']}")
    
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 멀티프로세싱 설정
    mp.set_start_method('spawn', force=True)
    
    # 큐 생성
    task_queue = mp.Queue()
    result_queue = mp.Queue()
    progress_queue = mp.Queue()
    
    # 작업 큐에 비디오들 추가
    for idx, video_path in enumerate(video_paths):
        task_queue.put((video_path, idx))
    
    # 종료 신호 추가 (GPU 개수만큼)
    for _ in range(2):
        task_queue.put(None)
    
    # GPU 워커 프로세스 시작
    workers = []
    for gpu_id in range(2):  # A6000 x2
        worker = mp.Process(
            target=batch_gpu_worker,
            args=(gpu_id, task_queue, result_queue, progress_queue, config)
        )
        worker.start()
        workers.append(worker)
    
    # 결과 수집
    results = {}
    completed_count = 0
    failed_count = 0
    
    print(f"⏳ 처리 진행 중...")
    
    # 진행률 추적 스레드
    def progress_tracker():
        gpu_progress = {0: 0.0, 1: 0.0}
        
        while completed_count + failed_count < total_videos:
            try:
                progress_info = progress_queue.get(timeout=1.0)
                gpu_id = progress_info['gpu_id']
                progress = progress_info['progress']
                
                gpu_progress[gpu_id] = progress
                
                # 전체 진행률 계산
                total_progress = (completed_count + failed_count + sum(gpu_progress.values())) / total_videos
                
                if progress_callback:
                    progress_callback(min(total_progress, 1.0))
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"⚠️ 진행률 추적 오류: {e}")
                break
    
    # 진행률 스레드 시작
    from threading import Thread
    progress_thread = Thread(target=progress_tracker, daemon=True)
    progress_thread.start()
    
    # 결과 수집
    while completed_count + failed_count < total_videos:
        try:
            result_item = result_queue.get(timeout=10.0)
            
            task_id = result_item['task_id']
            status = result_item['status']
            
            results[task_id] = result_item
            
            if status == 'completed':
                completed_count += 1
                
                # HDF5 파일로 저장
                video_path = result_item['video_path']
                result_data = result_item['result']
                
                if result_data:
                    output_filename = Path(video_path).stem + '.h5'
                    output_path = os.path.join(output_dir, output_filename)
                    
                    try:
                        save_to_hdf5(result_data, output_path)
                        print(f"💾 저장 완료: {output_filename}")
                    except Exception as e:
                        print(f"⚠️ HDF5 저장 실패 ({output_filename}): {e}")
                
            else:
                failed_count += 1
                print(f"❌ 처리 실패: {result_item.get('video_path', 'unknown')}")
            
            # 진행률 출력
            progress_pct = (completed_count + failed_count) / total_videos * 100
            print(f"📊 진행률: {completed_count + failed_count}/{total_videos} ({progress_pct:.1f}%)")
            
        except queue.Empty:
            print(f"⚠️ 결과 대기 타임아웃 (완료: {completed_count}, 실패: {failed_count})")
            continue
        except Exception as e:
            print(f"💥 결과 수집 오류: {e}")
            break
    
    # 워커 프로세스 종료 대기
    for worker in workers:
        worker.join(timeout=10.0)
        if worker.is_alive():
            print(f"⚠️ 워커 강제 종료")
            worker.terminate()
    
    total_time = time.time() - start_time
    
    # 최종 결과 정리
    summary = {
        'total_videos': total_videos,
        'completed': completed_count,
        'failed': failed_count,
        'total_time': total_time,
        'average_time_per_video': total_time / max(total_videos, 1),
        'results': results
    }
    
    print(f"🏁 듀얼 GPU 배치 처리 완료:")
    print(f"   - 성공: {completed_count}개")
    print(f"   - 실패: {failed_count}개")
    print(f"   - 총 시간: {total_time:.2f}초")
    print(f"   - 평균 시간: {summary['average_time_per_video']:.2f}초/비디오")
    
    return summary

def save_to_hdf5(result_data: Dict, output_path: str):
    """결과 데이터를 HDF5 파일로 저장"""
    try:
        with h5py.File(output_path, 'w') as f:
            # JPEG 프레임 저장
            jpeg_frames = result_data['jpeg_frames']
            if jpeg_frames:
                # 각 JPEG를 개별 데이터셋으로 저장
                frames_group = f.create_group('frames')
                for i, jpeg_data in enumerate(jpeg_frames):
                    frames_group.create_dataset(f'frame_{i:06d}', data=np.frombuffer(jpeg_data, dtype=np.uint8))
            
            # 키포인트 데이터 저장
            keypoints = result_data['keypoints']
            if keypoints:
                # 패딩하여 균일한 크기로 만들기
                max_persons = max(len(frame_kpts) for frame_kpts in keypoints)
                keypoint_dim = 34  # 17 joints * 2 coordinates
                
                padded_keypoints = np.zeros((len(keypoints), max_persons, keypoint_dim))
                
                for frame_idx, frame_kpts in enumerate(keypoints):
                    for person_idx, person_kpts in enumerate(frame_kpts):
                        if person_idx < max_persons and len(person_kpts) >= keypoint_dim:
                            padded_keypoints[frame_idx, person_idx, :] = person_kpts[:keypoint_dim]
                
                f.create_dataset('keypoints', data=padded_keypoints)
            
            # 스코어 데이터 저장
            scores = result_data['scores']
            if scores:
                max_persons = max(len(frame_scores) if isinstance(frame_scores, list) else 1 for frame_scores in scores)
                padded_scores = np.zeros((len(scores), max_persons))
                
                for frame_idx, frame_scores in enumerate(scores):
                    if isinstance(frame_scores, list):
                        for person_idx, score in enumerate(frame_scores):
                            if person_idx < max_persons:
                                padded_scores[frame_idx, person_idx] = score
                    else:
                        padded_scores[frame_idx, 0] = frame_scores
                
                f.create_dataset('scores', data=padded_scores)
            
            # 메타데이터 저장
            metadata = f.create_group('metadata')
            metadata.attrs['total_frames'] = result_data['total_frames']
            metadata.attrs['processing_time'] = result_data['processing_time']
            metadata.attrs['fps'] = result_data['fps']
            
            if 'video_info' in result_data:
                video_info = result_data['video_info']
                metadata.attrs['original_fps'] = video_info.get('original_fps', 0)
                metadata.attrs['resolution'] = video_info.get('resolution', '')
                metadata.attrs['duration'] = video_info.get('duration', 0)
        
        print(f"✅ HDF5 저장 완료: {output_path}")
        
    except Exception as e:
        print(f"💥 HDF5 저장 실패 ({output_path}): {e}")
        raise
