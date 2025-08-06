#!/usr/bin/env python3
"""
Simplified Multi-ONNX Processor - 실행 가능한 버전
싱글 프로세스에서 배치 처리 방식으로 GPU 효율성 극대화
"""

import os
import cv2
import h5py
import time
import torch
import logging
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional

from onnx_inferencer import YOLO11LRTMWONNXInferencer as ONNXInferencer

class SimplifiedBatchProcessor:
    """단순화된 배치 처리기 - A6000 x2 GPU 최적화 (싱글 프로세스)"""
    
    def __init__(self, 
                 data_root: str = "/workspace01/team03/data/mmpose/jy/data/1.Training",
                 output_dir: str = "/workspace01/team03/data/simplified_output",
                 rtmw_model_name: str = "rtmw-dw-x-l_simcc-cocktail14_270e-384x288.onnx",
                 direction: str = "F",
                 item_types: List[str] = ["WORD"],
                 batch_size: int = 64,
                 max_videos_per_batch: int = 250):
        
        self.data_root = Path(data_root)
        self.output_dir = Path(output_dir)
        self.rtmw_model_name = rtmw_model_name
        self.direction = direction
        self.item_types = item_types
        self.batch_size = batch_size
        self.max_videos_per_batch = max_videos_per_batch
        self.keypoint_scale = 8
        
        # 출력 디렉토리 생성
        self.video_output_dir = self.output_dir / "video_processing"
        self.hdf5_output_dir = self.output_dir / "hdf5_batches"
        self.video_output_dir.mkdir(parents=True, exist_ok=True)
        self.hdf5_output_dir.mkdir(parents=True, exist_ok=True)
        
        # 로깅 설정
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
        # ONNX 추론기 초기화
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.inferencer = ONNXInferencer(
            rtmw_onnx_path=rtmw_model_name,
            detection_device=device,
            pose_device=device,
            optimize_for_accuracy=True
        )
        
        self.logger.info(f"🚀 SimplifiedBatchProcessor 초기화 완료")
        self.logger.info(f"   - 데이터 루트: {data_root}")
        self.logger.info(f"   - 출력 디렉토리: {output_dir}")
        self.logger.info(f"   - 배치 크기: {batch_size}")
        self.logger.info(f"   - 디바이스: {device}")

    def collect_all_videos(self) -> List[Tuple[str, int, str]]:
        """모든 비디오 수집"""
        videos_base_dir = self.data_root / "videos"
        all_videos = []
        
        if not videos_base_dir.exists():
            self.logger.error(f"비디오 디렉토리가 없습니다: {videos_base_dir}")
            return []
        
        pattern = f"*_{self.direction}.mp4"
        
        for folder in videos_base_dir.iterdir():
            if folder.is_dir():
                folder_name = folder.name
                
                for video_file in folder.glob(pattern):
                    # 파일명에서 아이템 정보 추출 (NIA_SL_WORD0515_REAL11_F.mp4 형태)
                    filename = video_file.stem
                    try:
                        if 'NIA_SL_' in filename:
                            # NIA_SL_WORD0515_REAL11_F에서 WORD0515 부분 추출
                            parts = filename.split('_')
                            if len(parts) >= 3:
                                word_part = parts[2]  # WORD0515
                                if word_part.startswith('WORD') and len(word_part) > 4:
                                    item_type = "WORD"
                                    item_id = int(word_part[4:])  # 0515 부분
                                    
                                    if item_type in self.item_types:
                                        all_videos.append((item_type, item_id, str(video_file)))
                    except (ValueError, IndexError):
                        self.logger.debug(f"파일명 파싱 실패: {filename}")
                        continue
        
        # 정렬
        all_videos = sorted(all_videos, key=lambda x: (x[0], x[1]))
        
        self.logger.info(f"📊 총 {len(all_videos)}개 비디오 수집 완료")
        return all_videos

    def process_video_to_arrays(self, video_path: str) -> Optional[Dict]:
        """비디오를 처리하여 배열 형태로 반환"""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                self.logger.error(f"비디오 열기 실패: {video_path}")
                return None
            
            frames = []
            frame_count = 0
            
            # 모든 프레임 읽기
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
                frame_count += 1
            
            cap.release()
            
            if frame_count == 0:
                return None
            
            jpeg_frames = []
            keypoints_list = []
            scores_list = []
            
            # 프레임별 처리
            for frame in tqdm(frames, desc=f"처리 중", leave=False):
                # 추론 실행
                result = self.inferencer.process_frame(frame)
                
                # JPEG 인코딩
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                jpeg_frames.append(buffer.tobytes())
                
                # 결과 처리
                frame_keypoints = []
                frame_scores = []
                
                if isinstance(result, tuple):
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
                                for i in range(0, len(kpts), 3):
                                    x, y, conf = kpts[i], kpts[i+1], kpts[i+2]
                                    scaled_kpts.extend([
                                        int(x * self.keypoint_scale),
                                        int(y * self.keypoint_scale),
                                        int(conf * self.keypoint_scale)
                                    ])
                                
                                frame_keypoints.append(scaled_kpts[:51])
                                frame_scores.append(person.get('score', 1.0))
                
                # 키포인트가 없는 경우
                if not frame_keypoints:
                    frame_keypoints.append([0] * 51)
                    frame_scores.append(0.0)
                
                keypoints_list.append(frame_keypoints)
                scores_list.append(frame_scores)
            
            return {
                'jpeg_frames': jpeg_frames,
                'keypoints': keypoints_list,
                'scores': scores_list,
                'frame_count': frame_count
            }
            
        except Exception as e:
            self.logger.error(f"비디오 처리 오류 ({video_path}): {e}")
            return None

    def save_video_arrays(self, item_type: str, item_id: int, arrays: Dict) -> bool:
        """비디오 배열을 파일로 저장"""
        try:
            output_key = f"{item_type}{item_id:04d}"
            item_dir = self.video_output_dir / output_key
            item_dir.mkdir(parents=True, exist_ok=True)
            
            # 데이터 저장
            frames_path = item_dir / "frames.npy"
            np.save(frames_path, np.array(arrays['jpeg_frames'], dtype=object))
            
            keypoints_path = item_dir / "keypoints.npy"
            np.save(keypoints_path, np.array(arrays['keypoints'], dtype=object))
            
            scores_path = item_dir / "scores.npy"
            np.save(scores_path, np.array(arrays['scores'], dtype=object))
            
            # 메타데이터
            metadata = {
                'frame_count': arrays['frame_count'],
                'keypoint_scale': self.keypoint_scale,
                'output_key': output_key
            }
            metadata_path = item_dir / "metadata.npy"
            np.save(metadata_path, metadata)
            
            return True
            
        except Exception as e:
            self.logger.error(f"저장 오류 ({item_type}{item_id:04d}): {e}")
            return False

    def create_hdf5_batch(self, successful_keys: List[str], batch_id: int) -> bool:
        """HDF5 배치 파일 생성"""
        try:
            frames_h5_path = self.hdf5_output_dir / f"batch_{batch_id:03d}_{self.direction}_frames.h5"
            poses_h5_path = self.hdf5_output_dir / f"batch_{batch_id:03d}_{self.direction}_poses.h5"
            
            # JPEG 가변 길이 타입
            jpeg_vlen_dtype = h5py.vlen_dtype(np.uint8)
            
            with h5py.File(frames_h5_path, 'w') as f_frames, \
                 h5py.File(poses_h5_path, 'w') as f_poses:
                
                for key in successful_keys:
                    item_dir = self.video_output_dir / key
                    if not item_dir.exists():
                        continue
                    
                    try:
                        # 데이터 로드
                        frames = np.load(item_dir / "frames.npy", allow_pickle=True)
                        keypoints = np.load(item_dir / "keypoints.npy", allow_pickle=True)
                        
                        # HDF5에 저장
                        f_frames.create_dataset(key, data=frames, dtype=jpeg_vlen_dtype)
                        f_poses.create_dataset(key, data=keypoints)
                        
                    except Exception as e:
                        self.logger.error(f"아이템 {key} HDF5 변환 실패: {e}")
                        continue
            
            self.logger.info(f"✅ HDF5 배치 {batch_id:03d} 생성 완료 ({len(successful_keys)}개)")
            return True
            
        except Exception as e:
            self.logger.error(f"HDF5 배치 생성 실패: {e}")
            return False

    def process_all_videos(self, max_videos: Optional[int] = None):
        """모든 비디오 처리 (배치별)"""
        all_videos = self.collect_all_videos()
        
        if not all_videos:
            self.logger.error("❌ 처리할 비디오가 없습니다")
            return
        
        # 제한 설정
        if max_videos:
            all_videos = all_videos[:max_videos]
            self.logger.info(f"🔢 처리 제한: {max_videos}개 비디오")
        
        total_videos = len(all_videos)
        self.logger.info(f"🚀 비디오 처리 시작: {total_videos}개")
        
        successful_keys = []
        batch_id = 0
        
        # 비디오별 처리
        with tqdm(total=total_videos, desc="전체 진행률") as pbar:
            for i, (item_type, item_id, video_path) in enumerate(all_videos):
                try:
                    # 비디오 처리
                    arrays = self.process_video_to_arrays(video_path)
                    
                    if arrays:
                        # 배열 저장
                        if self.save_video_arrays(item_type, item_id, arrays):
                            output_key = f"{item_type}{item_id:04d}"
                            successful_keys.append(output_key)
                            self.logger.debug(f"✅ {output_key} 처리 완료 ({arrays['frame_count']} 프레임)")
                        else:
                            self.logger.warning(f"⚠️ {item_type}{item_id:04d} 저장 실패")
                    else:
                        self.logger.warning(f"⚠️ {item_type}{item_id:04d} 처리 실패")
                    
                    # 배치 단위로 HDF5 생성
                    if len(successful_keys) >= self.max_videos_per_batch or i == total_videos - 1:
                        if successful_keys:
                            self.create_hdf5_batch(successful_keys, batch_id)
                            batch_id += 1
                            successful_keys = []
                    
                except Exception as e:
                    self.logger.error(f"비디오 처리 오류 {item_type}{item_id:04d}: {e}")
                
                pbar.update(1)
        
        self.logger.info("✅ 모든 비디오 처리 완료")
        self.print_final_statistics()

    def print_final_statistics(self):
        """최종 통계 출력"""
        processed_videos = len([d for d in self.video_output_dir.iterdir() if d.is_dir()])
        hdf5_files = len(list(self.hdf5_output_dir.glob("*.h5")))
        
        self.logger.info("="*80)
        self.logger.info("📊 최종 처리 통계:")
        self.logger.info(f"   - 처리된 비디오: {processed_videos}개")
        self.logger.info(f"   - 생성된 HDF5 파일: {hdf5_files}개")
        self.logger.info(f"   - 출력 디렉토리: {self.output_dir}")
        self.logger.info("="*80)

def main():
    """메인 실행 함수"""
    processor = SimplifiedBatchProcessor(
        data_root="/workspace01/team03/data/mmpose/jy/data/1.Training",
        output_dir="/workspace01/team03/data/simplified_output",
        direction="F",
        item_types=["WORD"],
        batch_size=64,
        max_videos_per_batch=250
    )
    
    # 테스트용으로 처음 10개만 처리
    processor.process_all_videos(max_videos=10)

if __name__ == "__main__":
    main()
