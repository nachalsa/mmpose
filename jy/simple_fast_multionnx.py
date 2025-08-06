#!/usr/bin/env python3
"""
Fast Multi-ONNX Processor - 단순화된 단일 프로세스 버전
커스터마이징 가능하고 빠른 처리 보장
"""

import os
import cv2
import h5py
import time
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional, Union

from onnx_inferencer import YOLO11LRTMWONNXInferencer as ONNXInferencer

class SimpleFastProcessor:
    """단순화된 고속 프로세서 - 멀티프로세싱 없이 안정적 동작"""
    
    def __init__(self, 
                 data_root: str = "data/1.Training",
                 output_dir: str = "simple_fast_output",
                 rtmw_model_name: str = "rtmw-dw-x-l_simcc-cocktail14_270e-384x288.onnx",
                 direction: str = "F",
                 item_types: List[str] = ["WORD"],
                 keypoint_scale: int = 8,
                 jpeg_quality: int = 90,
                 gpu_ids: List[int] = [0, 1]):
        
        self.data_root = Path(data_root)
        self.output_dir = Path(output_dir)
        self.direction = direction
        self.item_types = item_types
        self.keypoint_scale = keypoint_scale
        self.jpeg_quality = jpeg_quality
        self.gpu_ids = gpu_ids
        
        # GPU별 프로세서 초기화
        self.processors = {}
        for gpu_id in gpu_ids:
            if torch.cuda.is_available() and gpu_id < torch.cuda.device_count():
                device = f"cuda:{gpu_id}"
                try:
                    # GPU 설정
                    torch.cuda.set_device(gpu_id)
                    
                    # 프로세서 생성
                    processor = ONNXInferencer(
                        rtmw_onnx_path=rtmw_model_name,
                        detection_device=device,
                        pose_device=device,
                        optimize_for_accuracy=True
                    )
                    self.processors[gpu_id] = processor
                    print(f"✅ GPU {gpu_id} 프로세서 초기화 완료")
                except Exception as e:
                    print(f"⚠️ GPU {gpu_id} 초기화 실패: {e}")
        
        # 출력 디렉토리 생성
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"🚀 Simple Fast Processor 초기화 완료")
        print(f"   - 활성 GPU: {list(self.processors.keys())}")
        print(f"   - 출력 디렉토리: {self.output_dir}")

    def collect_all_videos(self) -> List[Tuple[str, int, str]]:
        """모든 비디오 수집"""
        videos_base_dir = self.data_root / "videos"
        all_videos = []
        
        if not videos_base_dir.exists():
            print(f"❌ 비디오 디렉토리 없음: {videos_base_dir}")
            return []
        
        pattern = f"*_{self.direction}.mp4"
        
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
                    print(f"📁 폴더 {folder.name}: {len(folder_videos)}개 비디오")
        
        print(f"📊 총 {len(all_videos)}개 비디오 발견")
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

    def process_single_video(self, video_path: str, processor, gpu_id: int) -> Optional[Dict]:
        """단일 비디오 처리"""
        try:
            start_time = time.time()
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return None
            
            jpeg_frames = []
            keypoints_list = []
            scores_list = []
            frame_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 추론 실행
                result = processor.process_frame(frame)
                
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
                            if len(kpts) >= 51:
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
                
                if not frame_keypoints:
                    frame_keypoints.append([0] * 51)
                    frame_scores.append(0.0)
                
                keypoints_list.append(frame_keypoints)
                scores_list.append(frame_scores)
                frame_count += 1
            
            cap.release()
            process_time = time.time() - start_time
            
            if frame_count > 0:
                fps = frame_count / process_time
                print(f"✅ GPU {gpu_id}: {Path(video_path).stem} 완료 ({frame_count} frames, {fps:.1f} fps)")
                
                return {
                    'jpeg_frames': jpeg_frames,
                    'keypoints': keypoints_list,
                    'scores': scores_list,
                    'frame_count': frame_count
                }
            
        except Exception as e:
            print(f"💥 GPU {gpu_id} 비디오 처리 오류: {e}")
        
        return None

    def save_video_data(self, output_key: str, arrays: Dict) -> bool:
        """비디오 데이터 저장"""
        try:
            # 출력 디렉토리 생성
            item_dir = self.output_dir / output_key
            item_dir.mkdir(parents=True, exist_ok=True)
            
            # HDF5로 저장
            frames_path = item_dir / "frames.h5"
            poses_path = item_dir / "poses.h5"
            
            with h5py.File(frames_path, 'w') as f:
                jpeg_vlen_dtype = h5py.vlen_dtype(np.uint8)
                jpeg_dataset = f.create_dataset('frames', (len(arrays['jpeg_frames']),), dtype=jpeg_vlen_dtype)
                
                for i, jpeg_data in enumerate(arrays['jpeg_frames']):
                    if isinstance(jpeg_data, bytes) and len(jpeg_data) > 0:
                        jpeg_dataset[i] = np.frombuffer(jpeg_data, dtype=np.uint8)
                    else:
                        jpeg_dataset[i] = np.array([], dtype=np.uint8)
            
            with h5py.File(poses_path, 'w') as f:
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
                f.attrs['keypoint_scale'] = self.keypoint_scale
            
            print(f"💾 저장 완료: {output_key} ({arrays['frame_count']} frames)")
            return True
            
        except Exception as e:
            print(f"💥 저장 오류 ({output_key}): {e}")
            return False

    def process_videos_simple(self, max_videos: Optional[int] = None):
        """단순 순차 처리 - 안정적이고 빠름"""
        all_videos = self.collect_all_videos()
        if not all_videos:
            print("❌ 처리할 비디오가 없습니다")
            return
        
        if max_videos and max_videos > 0:
            all_videos = all_videos[:max_videos]
            print(f"🔢 처리 제한: {max_videos}개 비디오")
        
        if not self.processors:
            print("❌ 사용 가능한 GPU 프로세서가 없습니다")
            return
        
        # GPU를 라운드로빈 방식으로 사용
        gpu_ids = list(self.processors.keys())
        processed_count = 0
        successful_count = 0
        
        print(f"🚀 단순 순차 처리 시작 ({len(gpu_ids)}개 GPU)")
        
        with tqdm(total=len(all_videos), desc="Simple Fast Processing") as pbar:
            for i, (item_type, item_id, video_path) in enumerate(all_videos):
                # GPU 선택 (라운드로빈)
                gpu_id = gpu_ids[i % len(gpu_ids)]
                processor = self.processors[gpu_id]
                
                # 비디오 처리
                output_key = f"{item_type}{item_id:04d}"
                arrays = self.process_single_video(video_path, processor, gpu_id)
                
                if arrays:
                    if self.save_video_data(output_key, arrays):
                        successful_count += 1
                
                processed_count += 1
                pbar.update(1)
        
        print(f"✅ 처리 완료: {successful_count}/{processed_count}개 성공")
        self.print_statistics(successful_count, processed_count)

    def print_statistics(self, successful_count: int, total_count: int):
        """통계 출력"""
        success_rate = (successful_count / total_count * 100) if total_count > 0 else 0
        
        print("="*80)
        print("📊 Simple Fast Processor 통계:")
        print(f"   - 처리된 비디오: {successful_count}/{total_count}개 ({success_rate:.1f}%)")
        print(f"   - 출력 디렉토리: {self.output_dir}")
        print(f"   - 사용된 GPU: {list(self.processors.keys())}")
        print("="*80)

    # 커스터마이징 메서드들
    def customize_keypoint_scale(self, scale: int):
        """키포인트 스케일 변경"""
        self.keypoint_scale = scale
        print(f"🔧 키포인트 스케일 변경: {scale}")
    
    def customize_jpeg_quality(self, quality: int):
        """JPEG 품질 변경"""
        self.jpeg_quality = quality
        print(f"🔧 JPEG 품질 변경: {quality}")
    
    def customize_output_dir(self, output_dir: str):
        """출력 디렉토리 변경"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"🔧 출력 디렉토리 변경: {output_dir}")

def main():
    """메인 실행 함수"""
    print("🚀 Simple Fast Multi-ONNX Processor")
    print("=" * 60)
    
    # GPU 개수 확인
    available_gpus = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            available_gpus.append(i)
    
    if not available_gpus:
        print("⚠️ CUDA GPU가 없습니다")
        available_gpus = [0]  # CPU 모드
    
    print(f"🔍 사용 가능한 GPU: {available_gpus}")
    
    # 프로세서 초기화
    processor = SimpleFastProcessor(
        data_root="/workspace01/team03/data/mmpose/jy/data/1.Training",
        output_dir="/workspace01/team03/data/simple_fast_output",
        direction="F",
        item_types=["WORD"],
        keypoint_scale=8,
        jpeg_quality=90,
        gpu_ids=available_gpus
    )
    
    # 처리할 비디오 수 설정
    choice = input("\n처리할 비디오 수를 제한하시겠습니까? (y/N): ").strip().lower()
    max_videos = None
    if choice == 'y':
        try:
            max_videos = int(input("최대 처리할 비디오 수 입력: "))
        except ValueError:
            max_videos = None
    
    # 처리 실행
    processor.process_videos_simple(max_videos=max_videos)

if __name__ == "__main__":
    main()
