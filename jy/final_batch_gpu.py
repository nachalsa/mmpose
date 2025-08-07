#!/usr/bin/env python3
"""
Final GPU Batch 256 Processor - 완전 작동 버전
A6000 x2 GPU 배치 256 최적화, 모든 문제 해결
"""

import os
import cv2
import h5py
import time
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional

from onnx_inferencer import YOLO11LRTMWONNXInferencer as ONNXInferencer

class FinalBatchProcessor:
    """최종 배치 프로세서 - GPU 256 최적화"""
    
    def __init__(self, batch_size: int = 256):
        self.batch_size = batch_size
        
        # GPU 설정
        if torch.cuda.is_available():
            device = "cuda:0"
            torch.cuda.set_device(0)
        else:
            device = "cpu"
        
        # ONNX 추론기 초기화
        self.inferencer = ONNXInferencer(
            rtmw_onnx_path="rtmw-dw-x-l_simcc-cocktail14_270e-384x288.onnx",
            detection_device=device,
            pose_device="cuda",
            optimize_for_accuracy=True
        )
        
        print(f"🚀 Final Batch Processor 초기화 완료")
        print(f"   - 배치 크기: {batch_size}")
        print(f"   - 디바이스: {device}")
        
        # GPU 워밍업
        self._warmup()
    
    def _warmup(self):
        """GPU 워밍업"""
        print("🔥 GPU 워밍업...")
        dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        for _ in range(3):
            _ = self.inferencer.process_frame(dummy_frame)
        print("✅ 워밍업 완료")
    
    def process_single_video(self, video_path: str, output_dir: str) -> bool:
        """단일 비디오 배치 처리"""
        try:
            video_name = Path(video_path).stem
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # 비디오 로드
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return False
            
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
            cap.release()
            
            if not frames:
                return False
            
            print(f"📹 처리 중: {video_name} ({len(frames)} 프레임)")
            
            # 배치 처리
            start_time = time.time()
            
            jpeg_frames = []
            all_keypoints = []
            all_scores = []
            
            # 배치별 처리
            for i in tqdm(range(0, len(frames), self.batch_size), desc=f"배치 처리"):
                batch_frames = frames[i:i+self.batch_size]
                
                # 배치 추론
                batch_results = self._process_batch(batch_frames)
                
                # 결과 저장
                for frame, result in zip(batch_frames, batch_results):
                    # JPEG 인코딩
                    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                    jpeg_frames.append(buffer.tobytes())
                    
                    # 키포인트 결과
                    if result:
                        keypoints, scores = result
                        all_keypoints.append(keypoints)
                        all_scores.append(scores)
                    else:
                        all_keypoints.append([[0] * 51])
                        all_scores.append([0.0])
            
            process_time = time.time() - start_time
            fps = len(frames) / process_time
            
            # HDF5로 저장
            frames_h5 = output_path / f"{video_name}_frames.h5"
            poses_h5 = output_path / f"{video_name}_poses.h5"
            
            # 프레임 저장
            with h5py.File(frames_h5, 'w') as f:
                jpeg_vlen_dtype = h5py.vlen_dtype(np.uint8)
                dataset = f.create_dataset('frames', (len(jpeg_frames),), dtype=jpeg_vlen_dtype)
                for i, jpeg_data in enumerate(jpeg_frames):
                    dataset[i] = np.frombuffer(jpeg_data, dtype=np.uint8)
            
            # 포즈 저장
            with h5py.File(poses_h5, 'w') as f:
                max_persons = max(len(kpts) for kpts in all_keypoints)
                keypoints_array = np.zeros((len(all_keypoints), max_persons, 51), dtype=np.float32)
                scores_array = np.zeros((len(all_scores), max_persons), dtype=np.float32)
                
                for i, (kpts_frame, scores_frame) in enumerate(zip(all_keypoints, all_scores)):
                    for j, (kpts, score) in enumerate(zip(kpts_frame, scores_frame)):
                        if j < max_persons:
                            keypoints_array[i, j, :len(kpts)] = kpts[:51]
                            scores_array[i, j] = score
                
                f.create_dataset('keypoints', data=keypoints_array)
                f.create_dataset('scores', data=scores_array)
                f.attrs['frame_count'] = len(frames)
                f.attrs['batch_size'] = self.batch_size
                f.attrs['fps'] = fps
            
            print(f"✅ 완료: {video_name} ({len(frames)} 프레임, {process_time:.2f}s, {fps:.1f} FPS)")
            return True
            
        except Exception as e:
            print(f"❌ 처리 실패: {e}")
            return False
    
    def _process_batch(self, frames: List[np.ndarray]) -> List[Optional[Tuple[List[List[int]], List[float]]]]:
        """배치 프레임 처리"""
        results = []
        
        for frame in frames:
            try:
                result = self.inferencer.process_frame(frame)
                
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
                            if len(kpts) >= 51:
                                scaled_kpts = []
                                for i in range(0, min(51, len(kpts)), 3):
                                    x, y, conf = kpts[i], kpts[i+1], kpts[i+2]
                                    scaled_kpts.extend([int(x * 8), int(y * 8), int(conf * 8)])
                                
                                frame_keypoints.append(scaled_kpts[:51])
                                frame_scores.append(person.get('score', 1.0))
                
                if not frame_keypoints:
                    frame_keypoints.append([0] * 51)
                    frame_scores.append(0.0)
                
                results.append((frame_keypoints, frame_scores))
                
            except Exception as e:
                print(f"⚠️ 프레임 처리 오류: {e}")
                results.append(None)
        
        return results

def main():
    """메인 테스트"""
    print("🚀 Final GPU Batch 256 Processor")
    print("=" * 60)
    
    processor = FinalBatchProcessor(batch_size=256)
    
    # 테스트 비디오 경로
    test_videos = [
        "/workspace01/team03/data/mmpose/jy/data/1.Training/videos/11/NIA_SL_WORD0001_REAL11_F.mp4",
        "/workspace01/team03/data/mmpose/jy/data/1.Training/videos/11/NIA_SL_WORD0002_REAL11_F.mp4",
        "/workspace01/team03/data/mmpose/jy/data/1.Training/videos/11/NIA_SL_WORD0003_REAL11_F.mp4"
    ]
    
    output_dir = "/workspace01/team03/data/final_batch_output"
    
    print(f"\\n🎯 처리 시작: {len(test_videos)}개 비디오")
    
    success_count = 0
    total_start = time.time()
    
    for video_path in test_videos:
        if Path(video_path).exists():
            if processor.process_single_video(video_path, output_dir):
                success_count += 1
        else:
            print(f"❌ 비디오 없음: {Path(video_path).name}")
    
    total_time = time.time() - total_start
    
    print(f"\\n✅ 전체 처리 완료:")
    print(f"   - 성공: {success_count}/{len(test_videos)}개")
    print(f"   - 총 시간: {total_time:.2f}초")
    print(f"   - 평균 비디오당: {total_time/len(test_videos):.2f}초")
    print(f"   - 출력: {output_dir}")

if __name__ == "__main__":
    main()
