#!/usr/bin/env python3
"""
HDF5 포즈 데이터 시각화 도구
스트림라인 처리기로 생성된 HDF5 파일에서 데이터를 읽어서 포즈 키포인트를 시각화
"""

import os
import cv2
import h5py
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import argparse
from tqdm import tqdm

# RTMW 키포인트 연결 정보 (133개 키포인트)
RTMW_SKELETON = [
    # 얼굴 연결 (68개 랜드마크 기반)
    # 턱선
    [0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], 
    [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16],
    # 오른쪽 눈썹
    [17, 18], [18, 19], [19, 20], [20, 21],
    # 왼쪽 눈썹  
    [22, 23], [23, 24], [24, 25], [25, 26],
    # 코
    [27, 28], [28, 29], [29, 30], [31, 32], [32, 33], [33, 34], [34, 35],
    # 오른쪽 눈
    [36, 37], [37, 38], [38, 39], [39, 40], [40, 41], [41, 36],
    # 왼쪽 눈
    [42, 43], [43, 44], [44, 45], [45, 46], [46, 47], [47, 42],
    # 입술 외곽
    [48, 49], [49, 50], [50, 51], [51, 52], [52, 53], [53, 54], [54, 55], [55, 56], 
    [56, 57], [57, 58], [58, 59], [59, 48],
    # 입술 내곽
    [60, 61], [61, 62], [62, 63], [63, 64], [64, 65], [65, 66], [66, 67], [67, 60],
    
    # 신체 키포인트 (COCO 17개 + 추가)
    # 머리-어깨
    [68, 69], [68, 70],  # nose -> eyes
    [69, 71], [70, 72],  # eyes -> ears
    [68, 73], [73, 74],  # nose -> shoulders
    
    # 팔
    [73, 75], [75, 77], [77, 79],  # 왼팔
    [74, 76], [76, 78], [78, 80],  # 오른팔
    
    # 몸통
    [73, 81], [74, 82],  # 어깨 -> 엉덩이
    [81, 82],  # 엉덩이 연결
    
    # 다리
    [81, 83], [83, 85], [85, 87],  # 왼다리
    [82, 84], [84, 86], [86, 88],  # 오른다리
    
    # 손 키포인트 (왼손 21개 + 오른손 21개)
    # 왼손 (89-109)
    [79, 89],  # 손목 연결
    [89, 90], [90, 91], [91, 92],  # 엄지
    [89, 93], [93, 94], [94, 95], [95, 96],  # 검지
    [89, 97], [97, 98], [98, 99], [99, 100],  # 중지
    [89, 101], [101, 102], [102, 103], [103, 104],  # 약지
    [89, 105], [105, 106], [106, 107], [107, 108],  # 새끼
    
    # 오른손 (110-130)
    [80, 110],  # 손목 연결
    [110, 111], [111, 112], [112, 113],  # 엄지
    [110, 114], [114, 115], [115, 116], [116, 117],  # 검지
    [110, 118], [118, 119], [119, 120], [120, 121],  # 중지
    [110, 122], [122, 123], [123, 124], [124, 125],  # 약지
    [110, 126], [126, 127], [127, 128], [128, 129],  # 새끼
    
    # 발 키포인트 (발가락)
    [87, 131], [88, 132]  # 발가락 연결
]

# 키포인트 색상 정의
POSE_COLORS = {
    'face': (255, 192, 203),      # 핑크
    'body': (0, 255, 0),          # 초록
    'left_hand': (255, 0, 0),     # 빨강
    'right_hand': (0, 0, 255),    # 파랑
    'foot': (255, 255, 0)         # 노랑
}

class HDF5PoseVisualizer:
    """HDF5 포즈 데이터 시각화기"""
    
    def __init__(self, hdf5_frames_path: str, hdf5_poses_path: str):
        """
        Args:
            hdf5_frames_path: 프레임 HDF5 파일 경로
            hdf5_poses_path: 포즈 HDF5 파일 경로
        """
        self.frames_path = Path(hdf5_frames_path)
        self.poses_path = Path(hdf5_poses_path)
        
        if not self.frames_path.exists():
            raise FileNotFoundError(f"프레임 HDF5 파일을 찾을 수 없습니다: {self.frames_path}")
        
        if not self.poses_path.exists():
            raise FileNotFoundError(f"포즈 HDF5 파일을 찾을 수 없습니다: {self.poses_path}")
        
        print(f"✅ HDF5 파일 로드 완료:")
        print(f"   - 프레임: {self.frames_path}")
        print(f"   - 포즈: {self.poses_path}")
        
        self._load_video_list()
    
    def _load_video_list(self):
        """HDF5에서 비디오 목록 로드"""
        with h5py.File(self.frames_path, 'r') as f:
            self.video_ids = list(f.keys())
            self.video_ids.sort()
        
        print(f"📹 총 {len(self.video_ids)}개 비디오 발견")
        for i, video_id in enumerate(self.video_ids[:5]):
            print(f"   {i+1}. {video_id}")
        if len(self.video_ids) > 5:
            print(f"   ... 외 {len(self.video_ids) - 5}개")
    
    def get_video_data(self, video_id: str) -> Dict[str, np.ndarray]:
        """특정 비디오의 데이터 로드"""
        try:
            with h5py.File(self.frames_path, 'r') as f_frames, \
                 h5py.File(self.poses_path, 'r') as f_poses:
                
                # 프레임 데이터 (JPEG 바이트 스트림)
                jpeg_frames_bytes = f_frames[f"{video_id}/frames_jpeg"][:]
                # 각 JPEG 바이트 스트림을 이미지로 디코딩
                frames = [cv2.imdecode(np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR) for jpeg_bytes in jpeg_frames_bytes]
                # 리스트를 NumPy 배열로 변환
                frames = np.array(frames)
                metadata_json = f_frames[f"{video_id}/metadata"][()]
                if isinstance(metadata_json, bytes):
                    metadata_json = metadata_json.decode('utf-8')
                metadata = json.loads(metadata_json)
                
                # 포즈 데이터 - 무조건 스케일링된 데이터만 사용하고 8로 나누기
                keypoints_scaled = f_poses[f"{video_id}/keypoints_scaled"][:]
                scores = f_poses[f"{video_id}/scores"][:]
                
                # 8배 스케일링 해제 (무조건 적용)
                keypoints = keypoints_scaled / 8.0
                
                return {
                    'frames': frames,
                    'keypoints': keypoints,  # 8로 나눈 좌표 (0-36, 0-48 범위)
                    'scores': scores,
                    'metadata': metadata
                }
                
        except Exception as e:
            print(f"❌ 비디오 데이터 로드 실패 ({video_id}): {e}")
            return None
    
    def get_keypoint_color(self, keypoint_idx: int) -> Tuple[int, int, int]:
        """키포인트 인덱스에 따른 색상 반환"""
        if keypoint_idx < 68:  # 얼굴 (0-67)
            return POSE_COLORS['face']
        elif keypoint_idx < 89:  # 신체 (68-88)
            return POSE_COLORS['body']
        elif keypoint_idx < 110:  # 왼손 (89-109)
            return POSE_COLORS['left_hand']
        elif keypoint_idx < 131:  # 오른손 (110-130)
            return POSE_COLORS['right_hand']
        else:  # 발 (131-132)
            return POSE_COLORS['foot']
    
    def draw_pose_on_image(self, image: np.ndarray, keypoints: np.ndarray, scores: np.ndarray, 
                          confidence_threshold: float = 0.3, use_skeleton: bool = True) -> np.ndarray:
        """이미지에 포즈 키포인트 그리기"""
        img = image.copy()
        h, w = img.shape[:2]
        
        # 키포인트는 이미 8로 나누어진 상태로 전달됨 (0-36, 0-48 범위)
        # 288x384 이미지에 맞게 그리기 위해 추가 스케일링 불필요
        
        # 1. 스켈레톤 연결선 그리기
        if use_skeleton:
            for connection in RTMW_SKELETON:
                pt1_idx, pt2_idx = connection
                
                # 범위 체크
                if pt1_idx >= len(keypoints) or pt2_idx >= len(keypoints):
                    continue
                
                # 신뢰도 체크
                if scores[pt1_idx] < confidence_threshold or scores[pt2_idx] < confidence_threshold:
                    continue
                
                pt1 = (int(keypoints[pt1_idx, 0]), int(keypoints[pt1_idx, 1]))
                pt2 = (int(keypoints[pt2_idx, 0]), int(keypoints[pt2_idx, 1]))

                # 좌표 유효성 체크
                if (0 <= pt1[0] < w and 0 <= pt1[1] < h and 
                    0 <= pt2[0] < w and 0 <= pt2[1] < h):
                    
                    color = self.get_keypoint_color(pt1_idx)
                    cv2.line(img, pt1, pt2, color, 1)
        
        # 2. 키포인트 점 그리기
        for i, (kpt, score) in enumerate(zip(keypoints, scores)):
            if score < confidence_threshold:
                continue
            
            x, y = int(kpt[0]), int(kpt[1])
            
            # 좌표 유효성 체크
            if 0 <= x < w and 0 <= y < h:
                color = self.get_keypoint_color(i)
                cv2.circle(img, (x, y), 2, color, -1)
                
                # 높은 신뢰도는 테두리 추가
                if score > 8.0:
                    cv2.circle(img, (x, y), 3, (255, 255, 255), 1)
        
        return img
    
    def visualize_video_frames(self, video_id: str, output_dir: str = "pose_visualization", 
                              max_frames: int = 50, confidence_threshold: float = 0.3,
                              save_images: bool = True, show_skeleton: bool = True):
        """비디오의 모든 프레임에 포즈 시각화"""
        
        # 데이터 로드
        data = self.get_video_data(video_id)
        if data is None:
            return
        
        frames = data['frames']
        keypoints = data['keypoints']  # 이미 8로 나누어진 좌표
        scores = data['scores']
        metadata = data['metadata']
        
        print(f"\n🎬 비디오 시각화: {video_id}")
        print(f"   - 총 프레임: {len(frames)}")
        print(f"   - 키포인트 형태: {keypoints.shape}")
        print(f"   - 신뢰도 임계값: {confidence_threshold}")
        
        # 출력 디렉토리 생성
        output_path = Path(output_dir) / video_id
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 처리할 프레임 수 제한
        num_frames = min(len(frames), max_frames)
        
        for frame_idx in tqdm(range(num_frames), desc="포즈 시각화"):
            # 이미지와 포즈 데이터
            img = frames[frame_idx]
            kpts = keypoints[frame_idx]
            scrs = scores[frame_idx]
            
            # 포즈 그리기
            pose_img = self.draw_pose_on_image(img, kpts, scrs, 
                                             confidence_threshold=confidence_threshold,
                                             use_skeleton=show_skeleton)
            
            # 정보 텍스트 추가
            info_text = f"Frame {frame_idx:03d} | Valid: {np.sum(scrs > confidence_threshold):03d}/133"
            cv2.putText(pose_img, info_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(pose_img, info_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
            
            # 이미지 저장
            if save_images:
                output_file = output_path / f"frame_{frame_idx:03d}_pose.jpg"
                cv2.imwrite(str(output_file), pose_img)
        
        print(f"✅ 시각화 완료: {output_path}")
        
        # 메타데이터 저장
        viz_metadata = {
            'video_id': video_id,
            'original_metadata': metadata,
            'visualization_info': {
                'total_frames': len(frames),
                'visualized_frames': num_frames,
                'confidence_threshold': confidence_threshold,
                'show_skeleton': show_skeleton,
                'keypoint_stats': {
                    'total_keypoints': keypoints.shape[1],
                    'avg_valid_per_frame': float(np.mean(np.sum(scores > confidence_threshold, axis=1))),
                    'min_valid_per_frame': int(np.min(np.sum(scores > confidence_threshold, axis=1))),
                    'max_valid_per_frame': int(np.max(np.sum(scores > confidence_threshold, axis=1)))
                }
            }
        }
        
        with open(output_path / "visualization_metadata.json", 'w') as f:
            json.dump(viz_metadata, f, indent=2)
    
    def create_pose_comparison(self, video_id: str, frame_indices: List[int], 
                              output_file: str = "pose_comparison.jpg"):
        """여러 프레임의 포즈를 한 이미지에 비교"""
        data = self.get_video_data(video_id)
        if data is None:
            return
        
        frames = data['frames']
        keypoints = data['keypoints']  # 이미 8로 나누어진 좌표
        scores = data['scores']
        
        # 서브플롯 설정
        n_frames = len(frame_indices)
        cols = min(4, n_frames)
        rows = (n_frames + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
        if n_frames == 1:
            axes = [axes]
        elif rows == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for i, frame_idx in enumerate(frame_indices):
            if frame_idx >= len(frames):
                continue
            
            img = frames[frame_idx]
            kpts = keypoints[frame_idx]
            scrs = scores[frame_idx]
            
            # 포즈 그리기
            pose_img = self.draw_pose_on_image(img, kpts, scrs)
            
            # BGR to RGB 변환
            pose_img_rgb = cv2.cvtColor(pose_img, cv2.COLOR_BGR2RGB)
            
            axes[i].imshow(pose_img_rgb)
            axes[i].set_title(f"Frame {frame_idx}\nValid: {np.sum(scrs > 0.3)}/133")
            axes[i].axis('off')
        
        # 남은 서브플롯 숨기기
        for i in range(len(frame_indices), len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 비교 이미지 저장: {output_file}")
    
    def analyze_pose_quality(self, video_id: str) -> Dict[str, Any]:
        """포즈 품질 분석"""
        data = self.get_video_data(video_id)
        if data is None:
            return None
        
        keypoints = data['keypoints']
        scores = data['scores'] 
        metadata = data['metadata']
        
        analysis = {
            'video_id': video_id,
            'total_frames': len(keypoints),
            'keypoint_analysis': {}
        }
        
        # 전체 통계
        valid_counts = np.sum(scores > 0.3, axis=1)
        analysis['overall_stats'] = {
            'avg_valid_keypoints': float(np.mean(valid_counts)),
            'min_valid_keypoints': int(np.min(valid_counts)),
            'max_valid_keypoints': int(np.max(valid_counts)),
            'std_valid_keypoints': float(np.std(valid_counts))
        }
        
        # 신체 부위별 분석
        body_parts = {
            'face': (0, 68),
            'body': (68, 89), 
            'left_hand': (89, 110),
            'right_hand': (110, 131),
            'feet': (131, 133)
        }
        
        for part_name, (start_idx, end_idx) in body_parts.items():
            part_scores = scores[:, start_idx:end_idx]
            part_valid = np.sum(part_scores > 0.3, axis=1)
            
            analysis['keypoint_analysis'][part_name] = {
                'total_keypoints': end_idx - start_idx,
                'avg_valid': float(np.mean(part_valid)),
                'detection_rate': float(np.mean(part_valid > 0)),
                'avg_confidence': float(np.mean(part_scores[part_scores > 0.3]))
            }
        
        return analysis


class FlexiblePoseVisualizer:
    """HDF5 및 JPEG 파일을 모두 지원하는 유연한 포즈 시각화기"""
    
    def __init__(self, input_source: str, pose_data_source: str = None):
        """
        Args:
            input_source: HDF5 파일 경로 또는 JPEG 이미지 디렉토리 경로
            pose_data_source: 포즈 데이터 소스 (HDF5 파일 또는 JSON 파일)
        """
        self.input_source = Path(input_source)
        self.pose_data_source = Path(pose_data_source) if pose_data_source else None
        self.mode = self._detect_input_mode()
        
        print(f"🔍 입력 모드: {self.mode}")
        print(f"   - 입력 소스: {self.input_source}")
        if self.pose_data_source:
            print(f"   - 포즈 데이터: {self.pose_data_source}")
        
        if self.mode == "hdf5":
            self.hdf5_visualizer = HDF5PoseVisualizer(str(self.input_source), str(self.pose_data_source))
        elif self.mode == "jpeg":
            self._load_jpeg_data()
    
    def _detect_input_mode(self) -> str:
        """입력 소스 타입 감지"""
        if self.input_source.suffix.lower() == '.h5':
            return "hdf5"
        elif self.input_source.is_dir():
            return "jpeg"
        else:
            raise ValueError(f"지원하지 않는 입력 형식: {self.input_source}")
    
    def _load_jpeg_data(self):
        """JPEG 파일 및 포즈 데이터 로드"""
        self.jpeg_files = sorted(list(self.input_source.glob("*.jpg")) + 
                                list(self.input_source.glob("*.jpeg")))
        
        if not self.jpeg_files:
            raise FileNotFoundError(f"JPEG 파일을 찾을 수 없습니다: {self.input_source}")
        
        print(f"📷 {len(self.jpeg_files)}개 JPEG 파일 발견")
        
        # 포즈 데이터 로드
        if self.pose_data_source and self.pose_data_source.exists():
            if self.pose_data_source.suffix.lower() == '.json':
                with open(self.pose_data_source, 'r') as f:
                    self.pose_data = json.load(f)
            elif self.pose_data_source.suffix.lower() == '.h5':
                # HDF5에서 포즈 데이터만 추출
                with h5py.File(self.pose_data_source, 'r') as f:
                    # 첫 번째 비디오의 포즈 데이터 사용
                    video_ids = list(f.keys())
                    if video_ids:
                        first_video = video_ids[0]
                        self.pose_data = {
                            'keypoints': f[f"{first_video}/keypoints_scaled"][:] / 8.0,  # 스케일 해제
                            'scores': f[f"{first_video}/scores"][:]
                        }
            else:
                print("⚠️ 지원하지 않는 포즈 데이터 형식")
                self.pose_data = None
        else:
            print("⚠️ 포즈 데이터 없음 - 이미지만 표시됩니다")
            self.pose_data = None
    
    def visualize_images(self, output_dir: str = "pose_visualization_jpeg", 
                        confidence_threshold: float = 0.3, show_skeleton: bool = True):
        """JPEG 이미지에 포즈 시각화"""
        if self.mode == "hdf5":
            # HDF5 모드에서는 기존 기능 사용
            video_ids = self.hdf5_visualizer.video_ids
            for video_id in video_ids:
                self.hdf5_visualizer.visualize_video_frames(
                    video_id, output_dir, 
                    confidence_threshold=confidence_threshold,
                    show_skeleton=show_skeleton
                )
            return
        
        # JPEG 모드
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\n🎨 JPEG 이미지 포즈 시각화")
        print(f"   - 총 이미지: {len(self.jpeg_files)}")
        print(f"   - 포즈 데이터: {'있음' if self.pose_data else '없음'}")
        
        for i, img_file in enumerate(tqdm(self.jpeg_files, desc="이미지 처리")):
            # 이미지 로드
            img = cv2.imread(str(img_file))
            if img is None:
                continue
            
            # 포즈 데이터가 있으면 그리기
            if self.pose_data and i < len(self.pose_data.get('keypoints', [])):
                kpts = self.pose_data['keypoints'][i]
                scrs = self.pose_data['scores'][i]
                
                # 포즈 그리기 (HDF5PoseVisualizer의 메서드 재사용)
                pose_img = self._draw_pose_on_image(img, kpts, scrs, 
                                                  confidence_threshold, show_skeleton)
                
                # 정보 텍스트 추가
                valid_count = np.sum(scrs > confidence_threshold)
                info_text = f"Image {i:03d} | Valid: {valid_count:03d}/133"
                cv2.putText(pose_img, info_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(pose_img, info_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
            else:
                pose_img = img
                # 포즈 데이터 없음 텍스트
                info_text = f"Image {i:03d} | No pose data"
                cv2.putText(pose_img, info_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(pose_img, info_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
            
            # 저장
            output_file = output_path / f"{img_file.stem}_pose.jpg"
            cv2.imwrite(str(output_file), pose_img)
        
        print(f"✅ JPEG 시각화 완료: {output_path}")
    
    def _draw_pose_on_image(self, image: np.ndarray, keypoints: np.ndarray, scores: np.ndarray,
                           confidence_threshold: float = 0.3, use_skeleton: bool = True) -> np.ndarray:
        """이미지에 포즈 그리기 (HDF5PoseVisualizer 메서드와 동일)"""
        img = image.copy()
        h, w = img.shape[:2]
        
        # 스켈레톤 연결선 그리기
        if use_skeleton:
            for connection in RTMW_SKELETON:
                pt1_idx, pt2_idx = connection
                
                if pt1_idx >= len(keypoints) or pt2_idx >= len(keypoints):
                    continue
                
                if scores[pt1_idx] < confidence_threshold or scores[pt2_idx] < confidence_threshold:
                    continue
                
                pt1 = (int(keypoints[pt1_idx, 0]), int(keypoints[pt1_idx, 1]))
                pt2 = (int(keypoints[pt2_idx, 0]), int(keypoints[pt2_idx, 1]))
                
                if (0 <= pt1[0] < w and 0 <= pt1[1] < h and 
                    0 <= pt2[0] < w and 0 <= pt2[1] < h):
                    color = self._get_keypoint_color(pt1_idx)
                    cv2.line(img, pt1, pt2, color, 1)
        
        # 키포인트 점 그리기
        for i, (kpt, score) in enumerate(zip(keypoints, scores)):
            if score < confidence_threshold:
                continue
            
            x, y = int(kpt[0]), int(kpt[1])
            
            if 0 <= x < w and 0 <= y < h:
                color = self._get_keypoint_color(i)
                cv2.circle(img, (x, y), 2, color, -1)
                
                if score > 8.0:
                    cv2.circle(img, (x, y), 3, (255, 255, 255), 1)
        
        return img
    
    def _get_keypoint_color(self, keypoint_idx: int) -> Tuple[int, int, int]:
        """키포인트 색상 반환"""
        if keypoint_idx < 68:  # 얼굴
            return POSE_COLORS['face']
        elif keypoint_idx < 89:  # 신체
            return POSE_COLORS['body']
        elif keypoint_idx < 110:  # 왼손
            return POSE_COLORS['left_hand']
        elif keypoint_idx < 131:  # 오른손
            return POSE_COLORS['right_hand']
        else:  # 발
            return POSE_COLORS['foot']
        
        scores = data['scores']
        keypoints = data['keypoints']  # 이미 8로 나누어진 좌표
        
        # 통계 계산
        valid_counts = np.sum(scores > 0.3, axis=1)
        high_conf_counts = np.sum(scores > 0.8, axis=1)
        avg_confidences = np.mean(scores, axis=1)
        
        analysis = {
            'video_id': video_id,
            'total_frames': len(scores),
            'keypoint_stats': {
                'total_keypoints': scores.shape[1],
                'valid_keypoints': {
                    'mean': float(np.mean(valid_counts)),
                    'std': float(np.std(valid_counts)),
                    'min': int(np.min(valid_counts)),
                    'max': int(np.max(valid_counts))
                },
                'high_confidence_keypoints': {
                    'mean': float(np.mean(high_conf_counts)),
                    'std': float(np.std(high_conf_counts)),
                    'min': int(np.min(high_conf_counts)),
                    'max': int(np.max(high_conf_counts))
                },
                'average_confidence': {
                    'mean': float(np.mean(avg_confidences)),
                    'std': float(np.std(avg_confidences)),
                    'min': float(np.min(avg_confidences)),
                    'max': float(np.max(avg_confidences))
                }
            },
            'quality_assessment': {
                'excellent_frames': int(np.sum(valid_counts > 120)),  # 90% 이상
                'good_frames': int(np.sum((valid_counts > 100) & (valid_counts <= 120))),  # 75-90%
                'fair_frames': int(np.sum((valid_counts > 80) & (valid_counts <= 100))),   # 60-75%
                'poor_frames': int(np.sum(valid_counts <= 80))  # 60% 미만
            }
        }
        
        return analysis


def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description="HDF5/JPEG 포즈 데이터 시각화")
    parser.add_argument("--input", type=str, required=True,
                       help="입력 소스 (HDF5 프레임 파일 또는 JPEG 이미지 디렉토리)")
    parser.add_argument("--poses", type=str, 
                       help="포즈 데이터 소스 (HDF5 포즈 파일 또는 JSON 파일)")
    parser.add_argument("--video-id", type=str, default=None,
                       help="시각화할 비디오 ID (HDF5 모드에서만)")
    parser.add_argument("--output", type=str, default="pose_visualization",
                       help="출력 디렉토리")
    parser.add_argument("--max-frames", type=int, default=50,
                       help="최대 처리 프레임 수")
    parser.add_argument("--confidence", type=float, default=0.3,
                       help="신뢰도 임계값")
    parser.add_argument("--no-skeleton", action="store_true",
                       help="스켈레톤 연결선 비활성화")
    parser.add_argument("--comparison", type=str, default=None,
                       help="비교 프레임 인덱스 (예: '0,10,20,30')")
    parser.add_argument("--analysis", action="store_true",
                       help="포즈 품질 분석 수행")
    
    args = parser.parse_args()
    
    try:
        # 유연한 시각화기 초기화
        visualizer = FlexiblePoseVisualizer(args.input, args.poses)
        
        if visualizer.mode == "hdf5":
            # HDF5 모드
            if args.video_id:
                video_ids = [args.video_id]
            else:
                # 사용 가능한 비디오 ID 출력
                print("\n📹 사용 가능한 비디오 ID:")
                for i, vid in enumerate(visualizer.hdf5_visualizer.video_ids[:10]):
                    print(f"   {i+1}. {vid}")
                if len(visualizer.hdf5_visualizer.video_ids) > 10:
                    print(f"   ... 외 {len(visualizer.hdf5_visualizer.video_ids) - 10}개")
                
                # 첫 번째 비디오로 기본 설정
                video_ids = [visualizer.hdf5_visualizer.video_ids[0]]
                print(f"\n🎯 첫 번째 비디오 사용: {video_ids[0]}")
            
            for video_id in video_ids:
                # 기본 시각화
                visualizer.hdf5_visualizer.visualize_video_frames(
                    video_id, args.output, args.max_frames,
                    confidence_threshold=args.confidence,
                    show_skeleton=not args.no_skeleton
                )
                
                # 비교 이미지 생성
                if args.comparison:
                    try:
                        frame_indices = [int(x.strip()) for x in args.comparison.split(',')]
                        output_file = f"{args.output}/comparison_{video_id}.jpg"
                        visualizer.hdf5_visualizer.create_pose_comparison(
                            video_id, frame_indices, output_file
                        )
                    except ValueError as e:
                        print(f"⚠️ 비교 프레임 인덱스 파싱 실패: {e}")
                
                # 품질 분석
                if args.analysis:
                    analysis = visualizer.hdf5_visualizer.analyze_pose_quality(video_id)
                    if analysis:
                        analysis_file = f"{args.output}/analysis_{video_id}.json"
                        with open(analysis_file, 'w') as f:
                            json.dump(analysis, f, indent=2)
                        print(f"📊 분석 결과 저장: {analysis_file}")
                        
                        # 간단한 통계 출력
                        stats = analysis['overall_stats']
                        print(f"\n📈 포즈 품질 통계 ({video_id}):")
                        print(f"   - 평균 유효 키포인트: {stats['avg_valid_keypoints']:.1f}/133")
                        print(f"   - 최소/최대: {stats['min_valid_keypoints']}/{stats['max_valid_keypoints']}")
                        
        else:
            # JPEG 모드
            visualizer.visualize_images(
                args.output,
                confidence_threshold=args.confidence,
                show_skeleton=not args.no_skeleton
            )
        
        print(f"\n🎉 시각화 완료!")
        print(f"   📁 출력 디렉토리: {args.output}")
        
    except Exception as e:
        print(f"❌ 시각화 실패: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
