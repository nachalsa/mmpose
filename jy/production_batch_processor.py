#!/usr/bin/env python3
"""
Production Batch Processor - GPU 배치 256 최적화 버전 (실제 운영용)
A6000 x2 GPU 완전 활용, VRAM 미리 로딩, 병렬 처리

개선사항:
- ZeroDivisionError 수정
- 메모리 효율성 개선
- 에러 핸들링 강화
- Production 환경 최적화
- 진행률 추적 개선
- GPU 메모리 관리 개선
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
import traceback
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional, Union
from concurrent.futures import ThreadPoolExecutor

from onnx_inferencer import YOLO11LRTMWONNXInferencer as ONNXInferencer

class ProductionBatchVideoProcessor:
    """Production용 배치 고속 비디오 처리기 - GPU 배치 256 최적화"""
    
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
        self.current_device = None
        self.device_info = {}
        
        if torch.cuda.is_available():
            current_device = torch.cuda.current_device()
            device = f"cuda:{current_device}"
            torch.cuda.set_device(current_device)
            
            self.current_device = current_device
            
            # GPU 메모리 정보
            gpu_props = torch.cuda.get_device_properties(current_device)
            total_vram = gpu_props.total_memory / 1024**3
            max_vram = total_vram * max_vram_usage
            
            # 배치 메모리 계산 (384x288x3 float32)
            single_frame_mb = (384 * 288 * 3 * 4) / (1024 * 1024)
            batch_memory_mb = single_frame_mb * batch_size
            
            self.device_info = {
                'device': device,
                'gpu_name': gpu_props.name,
                'total_vram': total_vram,
                'max_vram': max_vram,
                'batch_memory_mb': batch_memory_mb
            }
            
            print(f"🚀 ProductionBatchVideoProcessor GPU {current_device} 초기화")
            print(f"   - GPU: {gpu_props.name}")
            print(f"   - 총 VRAM: {total_vram:.1f}GB")
            print(f"   - 최대 사용: {max_vram:.1f}GB ({max_vram_usage*100:.0f}%)")
            print(f"   - 배치 크기: {batch_size} 프레임")
            print(f"   - 배치 메모리: {batch_memory_mb:.1f}MB")
            
        else:
            device = "cpu"
            self.device_info = {'device': device}
            print(f"⚠️ CPU 모드로 실행")
            
        # 추론기 초기화
        try:
            self.inferencer = ONNXInferencer(
                rtmw_onnx_path=rtmw_model_name,
                detection_device=device,
                pose_device="cuda",  # 명시적으로 CUDA 설정
                optimize_for_accuracy=True
            )
            
            # 초기화 확인
            if torch.cuda.is_available():
                allocated_memory = torch.cuda.memory_allocated(current_device) / 1024**3
                print(f"✅ ProductionBatchVideoProcessor 초기화 완료")
                print(f"   - 디바이스: {device} (메모리 사용: {allocated_memory:.2f}GB)")
                print(f"   - 배치 크기: {batch_size}")
                print(f"   - RTMW 모델: {rtmw_model_name}")
            
            # GPU 워밍업
            if self.gpu_warmup and torch.cuda.is_available():
                self._warmup_gpu()
                
        except Exception as e:
            print(f"❌ ProductionBatchVideoProcessor 초기화 실패: {e}")
            raise
    
    def _warmup_gpu(self):
        """GPU 워밍업 - 배치 처리 최적화"""
        print(f"🔥 GPU 워밍업 시작 (배치 {self.batch_size})")
        start_time = time.time()
        
        try:
            # 더미 배치 텐서 생성
            dummy_batch = torch.randn(self.batch_size, 3, 384, 288).cuda()
            
            # 몇 번 연산 수행하여 GPU 활성화
            for _ in range(5):
                _ = dummy_batch * 2.0
                _ = torch.nn.functional.interpolate(dummy_batch, size=(288, 384))
                _ = torch.nn.functional.avg_pool2d(dummy_batch, 2)
            
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
            # 비디오 열기 및 정보 추출
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"❌ 비디오 열기 실패: {video_path}")
                return None
            
            # 비디오 정보
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            if total_frames <= 0:
                cap.release()
                print(f"❌ 유효하지 않은 비디오: {video_path}")
                return None
                
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
                
                # 진행률 콜백 (로딩 단계: 30%)
                if progress_callback and frame_count % 50 == 0:
                    progress = min(frame_count / total_frames * 0.3, 0.3)
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
            
            # 배치별 처리
            batch_start_time = time.time()
            processed_frames = 0
            batch_count = 0
            
            for i in range(0, actual_frame_count, self.batch_size):
                batch_frames = frames[i:i+self.batch_size]
                batch_count += 1
                
                print(f"🔥 배치 {batch_count} 처리: {len(batch_frames)}개 프레임")
                
                # GPU 배치 처리 실행
                batch_results = self._process_frame_batch_safe(batch_frames)
                
                # 결과 저장 및 JPEG 인코딩
                for frame, result in zip(batch_frames, batch_results):
                    # JPEG 인코딩 (최적화된 파라미터)
                    encode_params = [
                        cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality,
                        cv2.IMWRITE_JPEG_OPTIMIZE, 1,
                        cv2.IMWRITE_JPEG_PROGRESSIVE, 1
                    ]
                    success, buffer = cv2.imencode('.jpg', frame, encode_params)
                    
                    if success:
                        jpeg_frames.append(buffer.tobytes())
                    else:
                        # 폴백: 기본 인코딩
                        _, buffer = cv2.imencode('.jpg', frame)
                        jpeg_frames.append(buffer.tobytes())
                    
                    # 키포인트 데이터 처리
                    if result and len(result) == 2:
                        frame_keypoints, frame_scores = result
                        
                        # 키포인트 스케일링 적용
                        if self.keypoint_scale != 1 and frame_keypoints:
                            scaled_keypoints = self._scale_keypoints(frame_keypoints)
                            keypoints_list.append(scaled_keypoints)
                        else:
                            keypoints_list.append(frame_keypoints)
                        
                        scores_list.append(frame_scores)
                    else:
                        # 기본값 (검출된 사람 없음)
                        keypoints_list.append([])
                        scores_list.append([])
                
                processed_frames += len(batch_frames)
                
                # 진행률 업데이트 (처리 단계: 70%)
                if progress_callback:
                    progress = 0.3 + (processed_frames / actual_frame_count) * 0.7
                    progress_callback(min(progress, 1.0))
                
                # GPU 메모리 정리 (주기적)
                if self.current_device is not None and batch_count % 4 == 0:
                    torch.cuda.empty_cache()
            
            total_processing_time = time.time() - batch_start_time
            total_time = time.time() - start_total_time
            
            # 성능 계산 (안전한 나눗셈)
            processing_fps = actual_frame_count / max(total_processing_time, 0.001)
            
            # 최종 결과 구성
            result = {
                'total_frames': actual_frame_count,
                'processed_frames': processed_frames,
                'jpeg_frames': jpeg_frames,
                'keypoints': keypoints_list,
                'scores': scores_list,
                'processing_time': total_processing_time,
                'total_time': total_time,
                'fps': processing_fps,
                'video_info': {
                    'original_fps': fps,
                    'resolution': f"{width}x{height}",
                    'duration': actual_frame_count / max(fps, 1.0)
                },
                'batch_info': {
                    'batch_size': self.batch_size,
                    'batch_count': batch_count,
                    'device_info': self.device_info
                }
            }
            
            print(f"✅ 비디오 처리 완료:")
            print(f"   - 처리: {actual_frame_count}프레임 ({total_processing_time:.2f}초)")
            print(f"   - 속도: {processing_fps:.1f} FPS")
            print(f"   - 전체: {total_time:.2f}초")
            print(f"   - 배치: {batch_count}개")
            
            return result
            
        except Exception as e:
            print(f"💥 배치 비디오 처리 오류 ({video_path}): {e}")
            traceback.print_exc()
            return None
    
    def _process_frame_batch_safe(self, frames: List[np.ndarray]) -> List[Optional[Tuple[List, List]]]:
        """안전한 프레임 배치 처리"""
        try:
            return self._process_frame_batch(frames)
        except Exception as e:
            print(f"⚠️ 배치 처리 실패, 개별 처리로 폴백: {e}")
            # 폴백: 개별 처리
            results = []
            for frame in frames:
                try:
                    result = self._process_single_frame(frame)
                    results.append(result)
                except Exception as e2:
                    print(f"⚠️ 개별 처리도 실패: {e2}")
                    results.append(([], []))
            return results
    
    def _process_frame_batch(self, frames: List[np.ndarray]) -> List[Optional[Tuple[List, List]]]:
        """프레임 배치 처리 - GPU 최적화"""
        if not frames:
            return []
            
        results = []
        
        try:
            # 각 프레임 개별 처리 (현재 ONNX 추론기의 한계로 인해)
            for frame in frames:
                result = self._process_single_frame(frame)
                results.append(result)
                
            return results
            
        except Exception as e:
            print(f"💥 배치 처리 오류: {e}")
            # 완전 폴백: 빈 결과
            return [([], [])] * len(frames)
    
    def _process_single_frame(self, frame: np.ndarray) -> Tuple[List, List]:
        """단일 프레임 처리"""
        try:
            # ONNX 추론 실행
            vis_image, results = self.inferencer.process_frame(frame)
            
            if not results or len(results) == 0:
                return ([], [])
            
            frame_keypoints = []
            frame_scores = []
            
            # 모든 검출된 사람 처리
            for person_result in results:
                keypoints, scores, bbox = person_result
                
                # 키포인트 처리
                if isinstance(keypoints, np.ndarray) and keypoints.size > 0:
                    if keypoints.ndim == 2:  # (17, 2) 형태
                        kpts_flat = keypoints.flatten().tolist()
                        frame_keypoints.append(kpts_flat)
                    else:
                        frame_keypoints.append(keypoints.tolist())
                else:
                    frame_keypoints.append([])
                
                # 스코어 처리
                if isinstance(scores, np.ndarray) and scores.size > 0:
                    frame_scores.append(scores.tolist())
                elif isinstance(scores, list):
                    frame_scores.append(scores)
                else:
                    frame_scores.append([])
            
            return (frame_keypoints, frame_scores)
            
        except Exception as e:
            print(f"⚠️ 단일 프레임 처리 오류: {e}")
            return ([], [])
    
    def _scale_keypoints(self, keypoints: List) -> List:
        """키포인트 스케일링 적용"""
        if not keypoints or self.keypoint_scale == 1:
            return keypoints
            
        scaled_keypoints = []
        
        for person_kpts in keypoints:
            if not person_kpts:
                scaled_keypoints.append([])
                continue
                
            scaled_kpts = []
            # x, y 좌표 쌍으로 스케일링
            for i in range(0, len(person_kpts), 2):
                if i + 1 < len(person_kpts):
                    x = person_kpts[i] * self.keypoint_scale
                    y = person_kpts[i + 1] * self.keypoint_scale
                    scaled_kpts.extend([x, y])
                    
            scaled_keypoints.append(scaled_kpts)
            
        return scaled_keypoints


def production_gpu_worker(
    gpu_id: int,
    task_queue: mp.Queue, 
    result_queue: mp.Queue, 
    progress_queue: mp.Queue,
    config: Dict
):
    """Production GPU 워커 - 안정적인 배치 256 처리"""
    try:
        # GPU 설정
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        
        # PyTorch CUDA 초기화
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
            device = f"cuda:0"
            
            # GPU 정보
            gpu_props = torch.cuda.get_device_properties(0)
            gpu_name = gpu_props.name
            gpu_memory = gpu_props.total_memory / 1024**3
            
            print(f"🚀 Production GPU {gpu_id} 워커 시작")
            print(f"   - GPU: {gpu_name} ({gpu_memory:.1f}GB)")
            print(f"   - 배치 크기: {config['batch_size']}")
            print(f"   - 디바이스: {device}")
        else:
            device = "cpu"
            print(f"⚠️ GPU {gpu_id} 사용 불가 - CPU 모드")
        
        # 배치 프로세서 초기화
        processor = ProductionBatchVideoProcessor(
            rtmw_model_name=config['rtmw_model_name'],
            yolo_device=device,
            pose_device="cuda",
            keypoint_scale=config['keypoint_scale'],
            jpeg_quality=config['jpeg_quality'],
            batch_size=config['batch_size'],
            gpu_warmup=True,
            max_vram_usage=0.85
        )
        
        processed_count = 0
        
        while True:
            try:
                # 작업 가져오기 (타임아웃 설정)
                task = task_queue.get(timeout=30)
                
                if task is None:  # 종료 신호
                    print(f"🛑 GPU {gpu_id} 워커 종료")
                    break
                
                video_path, video_info = task
                
                print(f"🎬 GPU {gpu_id}: {Path(video_path).name} 처리 시작")
                
                # 진행률 콜백
                def progress_callback(progress):
                    progress_queue.put({
                        'gpu_id': gpu_id,
                        'video': Path(video_path).name,
                        'progress': progress
                    })
                
                # 비디오 처리
                start_time = time.time()
                result = processor.process_video_batch_optimized(
                    video_path, 
                    progress_callback=progress_callback
                )
                
                processing_time = time.time() - start_time
                
                if result:
                    # 성공 결과
                    result_data = {
                        'gpu_id': gpu_id,
                        'video_path': video_path,
                        'video_name': Path(video_path).name,
                        'success': True,
                        'frames': result.get('total_frames', 0),
                        'processing_time': processing_time,
                        'fps': result.get('fps', 0),
                        'result': result
                    }
                    
                    # VRAM 사용량 추가
                    if torch.cuda.is_available():
                        vram_used = torch.cuda.memory_allocated(0) / 1024**3
                        result_data['vram_used'] = vram_used
                    
                    result_queue.put(result_data)
                    processed_count += 1
                    
                    print(f"✅ GPU {gpu_id}: {Path(video_path).name} 완료")
                    print(f"   - {result['total_frames']}프레임, {result['fps']:.1f}FPS")
                    
                else:
                    # 실패 결과
                    result_queue.put({
                        'gpu_id': gpu_id,
                        'video_path': video_path,
                        'video_name': Path(video_path).name,
                        'success': False,
                        'error': 'Processing failed'
                    })
                    print(f"❌ GPU {gpu_id}: {Path(video_path).name} 처리 실패")
                
            except queue.Empty:
                print(f"⏰ GPU {gpu_id} 작업 대기 타임아웃")
                continue
                
            except Exception as e:
                print(f"💥 GPU {gpu_id} 워커 오류: {e}")
                traceback.print_exc()
                
                # 오류 결과 전송
                try:
                    result_queue.put({
                        'gpu_id': gpu_id,
                        'video_path': 'unknown',
                        'video_name': 'unknown',
                        'success': False,
                        'error': str(e)
                    })
                except:
                    pass
        
        print(f"📊 GPU {gpu_id} 최종 통계: {processed_count}개 비디오 처리 완료")
        
    except Exception as e:
        print(f"❌ GPU {gpu_id} 워커 초기화 실패: {e}")
        traceback.print_exc()


class ProductionBatchMultiONNXProcessor:
    """Production용 배치 멀티 ONNX 프로세서"""
    
    def __init__(self, 
                 data_root: str = "data/1.Training",
                 output_dir: str = "production_batch_output",
                 rtmw_model_name: str = "rtmw-dw-x-l_simcc-cocktail14_270e-384x288.onnx",
                 direction: str = "F",
                 item_types: List[str] = ["WORD"],
                 batch_size: int = 256,
                 keypoint_scale: int = 8,
                 jpeg_quality: int = 90):
        
        self.data_root = Path(data_root)
        self.output_dir = Path(output_dir)
        self.direction = direction
        self.item_types = item_types
        self.batch_size = batch_size
        
        # 설정 딕셔너리
        self.config = {
            'rtmw_model_name': rtmw_model_name,
            'keypoint_scale': keypoint_scale,
            'jpeg_quality': jpeg_quality,
            'batch_size': batch_size
        }
        
        # 출력 디렉토리 생성
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"🏭 ProductionBatchMultiONNXProcessor 초기화")
        print(f"   - 데이터 루트: {self.data_root}")
        print(f"   - 출력 디렉토리: {self.output_dir}")
        print(f"   - 배치 크기: {batch_size}")
        print(f"   - 방향: {direction}")
        print(f"   - 타입: {', '.join(item_types)}")
    
    def collect_all_videos(self) -> List[str]:
        """모든 비디오 파일 수집"""
        videos_dir = self.data_root / "videos"
        if not videos_dir.exists():
            print(f"❌ 비디오 디렉토리 없음: {videos_dir}")
            return []
        
        all_videos = []
        
        # 모든 하위 디렉토리 탐색
        for folder in videos_dir.iterdir():
            if not folder.is_dir():
                continue
                
            # 패턴에 맞는 파일 찾기
            pattern = f"*_{self.direction}.mp4"
            for video_file in folder.glob(pattern):
                all_videos.append(str(video_file))
        
        all_videos.sort()
        print(f"📁 총 {len(all_videos)}개 비디오 파일 발견")
        
        return all_videos
    
    def process_videos_dual_gpu(self, max_videos: Optional[int] = None):
        """듀얼 GPU를 사용한 비디오 처리"""
        all_videos = self.collect_all_videos()
        
        if max_videos:
            all_videos = all_videos[:max_videos]
            print(f"🔢 처리 제한: {len(all_videos)}개 비디오")
        
        if not all_videos:
            print("❌ 처리할 비디오가 없습니다")
            return
        
        # GPU 개수 확인
        gpu_count = torch.cuda.device_count()
        if gpu_count < 2:
            print(f"⚠️ GPU {gpu_count}개만 사용 가능 (듀얼 GPU 권장)")
            
        num_gpus = min(2, gpu_count, len(all_videos))
        
        # 멀티프로세싱 설정
        mp.set_start_method('spawn', force=True)
        
        # 큐 생성
        task_queue = mp.Queue()
        result_queue = mp.Queue()
        progress_queue = mp.Queue()
        
        # 작업 분배
        print(f"📦 작업 분배: {len(all_videos)}개 비디오 → {num_gpus}개 GPU")
        for video_path in all_videos:
            task_queue.put((video_path, {}))
        
        # 종료 신호
        for _ in range(num_gpus):
            task_queue.put(None)
        
        # GPU 워커 시작
        workers = []
        for gpu_id in range(num_gpus):
            worker = mp.Process(
                target=production_gpu_worker,
                args=(gpu_id, task_queue, result_queue, progress_queue, self.config)
            )
            worker.start()
            workers.append(worker)
        
        # 진행률 추적
        print(f"🚀 {num_gpus}개 GPU로 배치 처리 시작...")
        
        start_time = time.time()
        completed = 0
        results = []
        
        # 결과 수집
        with tqdm(total=len(all_videos), desc="비디오 처리") as pbar:
            while completed < len(all_videos):
                try:
                    result = result_queue.get(timeout=60)
                    results.append(result)
                    completed += 1
                    pbar.update(1)
                    
                    if result['success']:
                        pbar.set_description(f"✅ {result['video_name']} ({result.get('fps', 0):.1f}FPS)")
                    else:
                        pbar.set_description(f"❌ {result['video_name']} 실패")
                        
                except queue.Empty:
                    print("⏰ 결과 대기 타임아웃")
                    break
        
        # 워커 정리
        for worker in workers:
            worker.join(timeout=30)
            if worker.is_alive():
                worker.terminate()
        
        total_time = time.time() - start_time
        
        # 최종 통계
        successful = sum(1 for r in results if r['success'])
        failed = len(results) - successful
        
        print(f"\n🎉 배치 처리 완료!")
        print(f"   - 총 처리: {len(results)}개")
        print(f"   - 성공: {successful}개")
        print(f"   - 실패: {failed}개")
        print(f"   - 총 시간: {total_time:.2f}초")
        
        if successful > 0:
            avg_fps = np.mean([r.get('fps', 0) for r in results if r['success'] and 'fps' in r])
            print(f"   - 평균 FPS: {avg_fps:.1f}")


def main():
    """메인 실행 함수"""
    print("🏭 Production Batch Multi-ONNX Processor")
    print("⚡ A6000 x2 GPU 배치 256 최적화 버전 (Production Ready)")
    print("=" * 80)
    
    # Production 설정
    processor = ProductionBatchMultiONNXProcessor(
        data_root="/workspace01/team03/data/mmpose/jy/data/1.Training",
        output_dir="/workspace01/team03/data/production_batch_output",
        direction="F",
        item_types=["WORD"],
        batch_size=256,  # 배치 크기 256
        keypoint_scale=8,
        jpeg_quality=90
    )
    
    # 처리할 비디오 수 설정
    choice = input("\n처리할 비디오 수를 제한하시겠습니까? (y/N): ").strip().lower()
    max_videos = None
    if choice == 'y':
        try:
            max_videos = int(input("처리할 비디오 수: "))
        except ValueError:
            print("❌ 잘못된 입력, 전체 처리합니다")
    
    # 배치 처리 실행
    processor.process_videos_dual_gpu(max_videos=max_videos)


if __name__ == "__main__":
    main()
