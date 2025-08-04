#!/usr/bin/env python3
"""
HDF5 배치 데이터 시각화 도구
스트림라인 처리기로 생성된 HDF5 파일에서 이미지와 포즈 데이터를 읽어서 시각화
"""

import h5py
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
import json
from pathlib import Path
import argparse
from typing import List, Tuple, Optional, Dict, Any
import time
import os

# RTMW 키포인트 연결 정보 (133개 키포인트용)
# 얼굴 키포인트 연결
FACE_CONNECTIONS = [
    # 얼굴 윤곽
    (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10),
    (10, 11), (11, 12), (12, 13), (13, 14), (14, 15), (15, 16),
    # 오른쪽 눈썹
    (17, 18), (18, 19), (19, 20), (20, 21),
    # 왼쪽 눈썹
    (22, 23), (23, 24), (24, 25), (25, 26),
    # 코
    (27, 28), (28, 29), (29, 30), (31, 32), (32, 33), (33, 34), (34, 35),
    # 오른쪽 눈
    (36, 37), (37, 38), (38, 39), (39, 40), (40, 41), (41, 36),
    # 왼쪽 눈
    (42, 43), (43, 44), (44, 45), (45, 46), (46, 47), (47, 42),
    # 입술
    (48, 49), (49, 50), (50, 51), (51, 52), (52, 53), (53, 54), (54, 55), (55, 56),
    (56, 57), (57, 58), (58, 59), (59, 48),
    (60, 61), (61, 62), (62, 63), (63, 64), (64, 65), (65, 66), (66, 67), (67, 60)
]

# 몸체 키포인트 연결
BODY_CONNECTIONS = [
    # 몸통
    (68, 69),  # nose -> left_eye
    (68, 70),  # nose -> right_eye
    (69, 71),  # left_eye -> left_ear
    (70, 72),  # right_eye -> right_ear
    (73, 74),  # left_shoulder -> right_shoulder
    (73, 75),  # left_shoulder -> left_elbow
    (74, 76),  # right_shoulder -> right_elbow
    (75, 77),  # left_elbow -> left_wrist
    (76, 78),  # right_elbow -> right_wrist
    (73, 79),  # left_shoulder -> left_hip
    (74, 80),  # right_shoulder -> right_hip
    (79, 80),  # left_hip -> right_hip
    (79, 81),  # left_hip -> left_knee
    (80, 82),  # right_hip -> right_knee
    (81, 83),  # left_knee -> left_ankle
    (82, 84),  # right_knee -> right_ankle
]

# 왼손 키포인트 연결 (85-105)
LEFT_HAND_CONNECTIONS = [
    (85, 86), (86, 87), (87, 88), (88, 89),  # 엄지
    (85, 90), (90, 91), (91, 92), (92, 93),  # 검지
    (85, 94), (94, 95), (95, 96), (96, 97),  # 중지
    (85, 98), (98, 99), (99, 100), (100, 101),  # 약지
    (85, 102), (102, 103), (103, 104), (104, 105)  # 새끼
]

# 오른손 키포인트 연결 (106-126)
RIGHT_HAND_CONNECTIONS = [
    (106, 107), (107, 108), (108, 109), (109, 110),  # 엄지
    (106, 111), (111, 112), (112, 113), (113, 114),  # 검지
    (106, 115), (115, 116), (116, 117), (117, 118),  # 중지
    (106, 119), (119, 120), (120, 121), (121, 122),  # 약지
    (106, 123), (123, 124), (124, 125), (125, 126)  # 새끼
]

# 발 키포인트 연결 (127-132)
FOOT_CONNECTIONS = [
    (127, 128), (129, 130), (131, 132)  # 발가락들
]

# 전체 연결 정보 결합
ALL_CONNECTIONS = FACE_CONNECTIONS + BODY_CONNECTIONS + LEFT_HAND_CONNECTIONS + RIGHT_HAND_CONNECTIONS + FOOT_CONNECTIONS

# 키포인트 부위별 색상 정의
KEYPOINT_COLORS = {
    'face': (255, 200, 200),      # 연한 분홍
    'body': (100, 255, 100),      # 연한 초록
    'left_hand': (255, 255, 100), # 연한 노랑
    'right_hand': (255, 100, 255), # 연한 보라
    'foot': (100, 255, 255)       # 연한 청록
}

def get_keypoint_color(kpt_idx: int) -> Tuple[int, int, int]:
    """키포인트 인덱스에 따른 색상 반환"""
    if kpt_idx < 68:  # 얼굴 (0-67)
        return KEYPOINT_COLORS['face']
    elif kpt_idx < 85:  # 몸체 (68-84)
        return KEYPOINT_COLORS['body']
    elif kpt_idx < 106:  # 왼손 (85-105)
        return KEYPOINT_COLORS['left_hand']
    elif kpt_idx < 127:  # 오른손 (106-126)
        return KEYPOINT_COLORS['right_hand']
    else:  # 발 (127-132)
        return KEYPOINT_COLORS['foot']

class HDF5Visualizer:
    """HDF5 배치 데이터 시각화 클래스"""
    
    def __init__(self, output_dir: str = "h5_visualizations"):
        """
        Args:
            output_dir: 시각화 결과 저장 디렉토리
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # 하위 디렉토리 생성
        (self.output_dir / "single_frames").mkdir(exist_ok=True)
        (self.output_dir / "batch_grids").mkdir(exist_ok=True)
        (self.output_dir / "animations").mkdir(exist_ok=True)
        (self.output_dir / "statistics").mkdir(exist_ok=True)
        
        print(f"📁 시각화 출력 디렉토리: {self.output_dir}")
    
    def load_batch_data(self, frames_h5_path: str, poses_h5_path: str) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """HDF5 배치 파일에서 데이터 로드"""
        try:
            frames_data = None
            poses_data = None
            metadata = {}
            
            # 프레임 데이터 로드
            with h5py.File(frames_h5_path, 'r') as f:
                frames_data = f['frames'][:]
                
                # 메타데이터 로드
                if 'metadata' in f:
                    for key in f['metadata'].keys():
                        value = f['metadata'][key][()]
                        if isinstance(value, bytes):
                            value = value.decode('utf-8')
                        metadata[key] = value
            
            # 포즈 데이터 로드
            with h5py.File(poses_h5_path, 'r') as f:
                poses_data = f['keypoints'][:]
            
            print(f"✅ 배치 데이터 로드 완료:")
            print(f"   - 프레임: {frames_data.shape}")
            print(f"   - 포즈: {poses_data.shape}")
            print(f"   - 메타데이터: {len(metadata)}개 항목")
            
            return frames_data, poses_data, metadata
            
        except Exception as e:
            print(f"❌ 배치 데이터 로드 실패: {e}")
            return None, None, None
    
    def draw_pose_on_image(self, image: np.ndarray, keypoints: np.ndarray, 
                          confidence_threshold: float = 0.3,
                          draw_connections: bool = True,
                          draw_confidence: bool = False) -> np.ndarray:
        """이미지에 포즈 키포인트 그리기"""
        vis_image = image.copy()
        h, w = image.shape[:2]
        
        valid_keypoints = 0
        
        # 키포인트 그리기
        for i, (x, y, score) in enumerate(keypoints):
            if score > confidence_threshold:
                valid_keypoints += 1
                
                # 이미지 범위 내 확인
                if 0 <= x < w and 0 <= y < h:
                    # 키포인트별 색상
                    color = get_keypoint_color(i)
                    
                    # 신뢰도에 따른 크기
                    if score > 0.8:
                        radius = 4
                    elif score > 0.6:
                        radius = 3
                    else:
                        radius = 2
                    
                    # 키포인트 그리기
                    cv2.circle(vis_image, (int(x), int(y)), radius, color, -1)
                    cv2.circle(vis_image, (int(x), int(y)), radius + 1, (0, 0, 0), 1)
                    
                    # 신뢰도 텍스트 (옵션)
                    if draw_confidence and score > 0.7:
                        cv2.putText(vis_image, f'{score:.2f}', 
                                  (int(x) + 5, int(y) - 5),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        # 연결선 그리기
        if draw_connections:
            for connection in ALL_CONNECTIONS:
                pt1_idx, pt2_idx = connection
                
                if (pt1_idx < len(keypoints) and pt2_idx < len(keypoints) and 
                    keypoints[pt1_idx][2] > confidence_threshold and 
                    keypoints[pt2_idx][2] > confidence_threshold):
                    
                    pt1 = (int(keypoints[pt1_idx][0]), int(keypoints[pt1_idx][1]))
                    pt2 = (int(keypoints[pt2_idx][0]), int(keypoints[pt2_idx][1]))
                    
                    # 이미지 범위 내 확인
                    if (0 <= pt1[0] < w and 0 <= pt1[1] < h and 
                        0 <= pt2[0] < w and 0 <= pt2[1] < h):
                        
                        # 연결선 색상 (두 점의 평균 색상)
                        color1 = get_keypoint_color(pt1_idx)
                        color2 = get_keypoint_color(pt2_idx)
                        line_color = tuple((np.array(color1) + np.array(color2)) // 2)
                        
                        cv2.line(vis_image, pt1, pt2, line_color, 1)
        
        # 통계 정보 표시
        cv2.putText(vis_image, f'Valid Points: {valid_keypoints}/133', 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return vis_image
    
    def visualize_single_frame(self, frame_data: np.ndarray, pose_data: np.ndarray,
                              frame_idx: int, save_path: Optional[str] = None) -> str:
        """단일 프레임 시각화"""
        try:
            # 이미지 전처리 (HDF5에서 읽은 데이터는 0-1 범위일 수 있음)
            if frame_data.max() <= 1.0:
                frame_data = (frame_data * 255).astype(np.uint8)
            
            # RGB -> BGR 변환 (OpenCV용)
            if len(frame_data.shape) == 3 and frame_data.shape[2] == 3:
                frame_bgr = cv2.cvtColor(frame_data, cv2.COLOR_RGB2BGR)
            else:
                frame_bgr = frame_data
            
            # 포즈 그리기
            vis_image = self.draw_pose_on_image(frame_bgr, pose_data)
            
            # 저장 경로 결정
            if save_path is None:
                save_path = self.output_dir / "single_frames" / f"frame_{frame_idx:06d}_pose.jpg"
            
            # 저장
            success = cv2.imwrite(str(save_path), vis_image)
            
            if success:
                return str(save_path)
            else:
                print(f"❌ 이미지 저장 실패: {save_path}")
                return None
                
        except Exception as e:
            print(f"❌ 단일 프레임 시각화 실패: {e}")
            return None
    
    def create_batch_grid(self, frames_data: np.ndarray, poses_data: np.ndarray,
                         batch_name: str, grid_size: Tuple[int, int] = (4, 4),
                         frame_indices: Optional[List[int]] = None) -> str:
        """배치 데이터를 그리드로 시각화"""
        try:
            rows, cols = grid_size
            total_samples = min(rows * cols, len(frames_data))
            
            if frame_indices is None:
                # 균등하게 샘플링
                step = max(1, len(frames_data) // total_samples)
                frame_indices = list(range(0, len(frames_data), step))[:total_samples]
            else:
                frame_indices = frame_indices[:total_samples]
            
            # 개별 프레임 크기 확인
            sample_frame = frames_data[0]
            if sample_frame.max() <= 1.0:
                sample_frame = (sample_frame * 255).astype(np.uint8)
            
            frame_h, frame_w = sample_frame.shape[:2]
            
            # 그리드 이미지 생성
            grid_h = frame_h * rows
            grid_w = frame_w * cols
            grid_image = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
            
            print(f"📊 배치 그리드 생성: {total_samples}개 프레임, {rows}x{cols}")
            
            for i, frame_idx in enumerate(frame_indices):
                if i >= total_samples:
                    break
                
                row = i // cols
                col = i % cols
                
                # 프레임 데이터 전처리
                frame_data = frames_data[frame_idx]
                if frame_data.max() <= 1.0:
                    frame_data = (frame_data * 255).astype(np.uint8)
                
                # RGB -> BGR 변환
                if len(frame_data.shape) == 3 and frame_data.shape[2] == 3:
                    frame_bgr = cv2.cvtColor(frame_data, cv2.COLOR_RGB2BGR)
                else:
                    frame_bgr = frame_data
                
                # 포즈 그리기
                vis_frame = self.draw_pose_on_image(frame_bgr, poses_data[frame_idx])
                
                # 프레임 번호 표시
                cv2.putText(vis_frame, f'#{frame_idx}', 
                           (5, frame_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                
                # 그리드에 배치
                start_y = row * frame_h
                end_y = start_y + frame_h
                start_x = col * frame_w
                end_x = start_x + frame_w
                
                grid_image[start_y:end_y, start_x:end_x] = vis_frame
            
            # 그리드 경계선 그리기
            for i in range(1, rows):
                y = i * frame_h
                cv2.line(grid_image, (0, y), (grid_w, y), (255, 255, 255), 2)
            
            for i in range(1, cols):
                x = i * frame_w
                cv2.line(grid_image, (x, 0), (x, grid_h), (255, 255, 255), 2)
            
            # 저장
            grid_path = self.output_dir / "batch_grids" / f"{batch_name}_grid_{rows}x{cols}.jpg"
            success = cv2.imwrite(str(grid_path), grid_image)
            
            if success:
                print(f"✅ 배치 그리드 저장: {grid_path}")
                return str(grid_path)
            else:
                print(f"❌ 배치 그리드 저장 실패: {grid_path}")
                return None
                
        except Exception as e:
            print(f"❌ 배치 그리드 생성 실패: {e}")
            return None
    
    def create_pose_statistics(self, poses_data: np.ndarray, batch_name: str) -> str:
        """포즈 데이터 통계 시각화"""
        try:
            # 신뢰도 통계 계산
            all_scores = poses_data[:, :, 2]  # (frames, keypoints)
            
            # 프레임별 통계
            frame_valid_counts = np.sum(all_scores > 0.3, axis=1)
            frame_high_conf_counts = np.sum(all_scores > 0.8, axis=1)
            frame_avg_scores = np.mean(all_scores, axis=1)
            
            # 키포인트별 통계
            keypoint_avg_scores = np.mean(all_scores, axis=0)
            keypoint_valid_ratios = np.mean(all_scores > 0.3, axis=0)
            
            # 시각화 생성
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle(f'Pose Statistics - {batch_name}', fontsize=16)
            
            # 1. 프레임별 유효 키포인트 수
            ax1 = axes[0, 0]
            ax1.plot(frame_valid_counts, 'b-', alpha=0.7, label='Valid (>0.3)')
            ax1.plot(frame_high_conf_counts, 'r-', alpha=0.7, label='High Conf (>0.8)')
            ax1.set_title('Valid Keypoints per Frame')
            ax1.set_xlabel('Frame Index')
            ax1.set_ylabel('Number of Keypoints')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 2. 프레임별 평균 신뢰도
            ax2 = axes[0, 1]
            ax2.plot(frame_avg_scores, 'g-', alpha=0.7)
            ax2.set_title('Average Confidence per Frame')
            ax2.set_xlabel('Frame Index')
            ax2.set_ylabel('Average Confidence Score')
            ax2.grid(True, alpha=0.3)
            
            # 3. 신뢰도 분포 히스토그램
            ax3 = axes[0, 2]
            valid_scores = all_scores[all_scores > 0]  # 0인 값 제외
            ax3.hist(valid_scores, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
            ax3.set_title('Confidence Score Distribution')
            ax3.set_xlabel('Confidence Score')
            ax3.set_ylabel('Frequency')
            ax3.axvline(np.mean(valid_scores), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(valid_scores):.3f}')
            ax3.legend()
            
            # 4. 키포인트별 평균 신뢰도 (부위별 색상)
            ax4 = axes[1, 0]
            keypoint_colors = [get_keypoint_color(i) for i in range(133)]
            keypoint_colors_normalized = [(r/255, g/255, b/255) for r, g, b in keypoint_colors]
            
            bars = ax4.bar(range(133), keypoint_avg_scores, color=keypoint_colors_normalized, alpha=0.7)
            ax4.set_title('Average Confidence by Keypoint')
            ax4.set_xlabel('Keypoint Index')
            ax4.set_ylabel('Average Confidence')
            ax4.set_xlim(-1, 133)
            
            # 5. 키포인트별 유효 비율
            ax5 = axes[1, 1]
            bars = ax5.bar(range(133), keypoint_valid_ratios, color=keypoint_colors_normalized, alpha=0.7)
            ax5.set_title('Valid Ratio by Keypoint')
            ax5.set_xlabel('Keypoint Index')
            ax5.set_ylabel('Valid Ratio (>0.3)')
            ax5.set_xlim(-1, 133)
            
            # 6. 통계 요약
            ax6 = axes[1, 2]
            ax6.axis('off')
            
            stats_text = f"""
📊 Dataset Statistics:

🎯 Frame-wise Stats:
  • Total Frames: {len(poses_data)}
  • Avg Valid Points: {np.mean(frame_valid_counts):.1f}/133
  • Avg High Conf Points: {np.mean(frame_high_conf_counts):.1f}/133
  • Avg Confidence: {np.mean(frame_avg_scores):.3f}

🔢 Keypoint-wise Stats:
  • Best Keypoint: #{np.argmax(keypoint_avg_scores)} ({np.max(keypoint_avg_scores):.3f})
  • Worst Keypoint: #{np.argmin(keypoint_avg_scores)} ({np.min(keypoint_avg_scores):.3f})
  • Most Reliable: #{np.argmax(keypoint_valid_ratios)} ({np.max(keypoint_valid_ratios):.3f})

📈 Overall Stats:
  • Total Keypoints: {poses_data.size}
  • Valid Keypoints: {np.sum(all_scores > 0.3)}
  • High Conf Keypoints: {np.sum(all_scores > 0.8)}
  • Overall Confidence: {np.mean(valid_scores):.3f}
            """
            
            ax6.text(0.05, 0.95, stats_text, transform=ax6.transAxes, fontsize=11,
                    verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
            
            # 저장
            stats_path = self.output_dir / "statistics" / f"{batch_name}_statistics.png"
            plt.tight_layout()
            plt.savefig(stats_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✅ 통계 시각화 저장: {stats_path}")
            return str(stats_path)
            
        except Exception as e:
            print(f"❌ 통계 시각화 생성 실패: {e}")
            return None
    
    def create_animation(self, frames_data: np.ndarray, poses_data: np.ndarray,
                        batch_name: str, fps: int = 10, max_frames: int = 100) -> str:
        """포즈 애니메이션 생성 (연속 프레임)"""
        try:
            # 프레임 수 제한
            total_frames = min(len(frames_data), max_frames)
            
            # 비디오 라이터 설정
            sample_frame = frames_data[0]
            if sample_frame.max() <= 1.0:
                sample_frame = (sample_frame * 255).astype(np.uint8)
            
            h, w = sample_frame.shape[:2]
            
            # 출력 경로
            animation_path = self.output_dir / "animations" / f"{batch_name}_animation.mp4"
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(str(animation_path), fourcc, fps, (w, h))
            
            print(f"🎬 애니메이션 생성: {total_frames}프레임, {fps}fps")
            
            for i in range(total_frames):
                # 프레임 데이터 전처리
                frame_data = frames_data[i]
                if frame_data.max() <= 1.0:
                    frame_data = (frame_data * 255).astype(np.uint8)
                
                # RGB -> BGR 변환
                if len(frame_data.shape) == 3 and frame_data.shape[2] == 3:
                    frame_bgr = cv2.cvtColor(frame_data, cv2.COLOR_RGB2BGR)
                else:
                    frame_bgr = frame_data
                
                # 포즈 그리기
                vis_frame = self.draw_pose_on_image(frame_bgr, poses_data[i])
                
                # 프레임 정보 표시
                cv2.putText(vis_frame, f'Frame: {i}/{total_frames-1}', 
                           (10, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                # 비디오에 추가
                video_writer.write(vis_frame)
                
                # 진행률 표시
                if i % 10 == 0:
                    progress = (i / total_frames) * 100
                    print(f"   진행률: {progress:.1f}% ({i}/{total_frames})")
            
            video_writer.release()
            
            print(f"✅ 애니메이션 저장: {animation_path}")
            return str(animation_path)
            
        except Exception as e:
            print(f"❌ 애니메이션 생성 실패: {e}")
            return None
    
    def process_batch_files(self, frames_h5_path: str, poses_h5_path: str,
                           batch_name: Optional[str] = None,
                           create_grid: bool = True,
                           create_stats: bool = True,
                           create_anim: bool = True,
                           sample_single_frames: int = 5) -> Dict[str, str]:
        """배치 파일 전체 처리"""
        print(f"\n=== HDF5 배치 시각화 시작 ===")
        print(f"프레임 파일: {os.path.basename(frames_h5_path)}")
        print(f"포즈 파일: {os.path.basename(poses_h5_path)}")
        
        # 배치 이름 자동 생성
        if batch_name is None:
            batch_name = Path(frames_h5_path).stem.replace('_frames', '')
        
        # 데이터 로드
        frames_data, poses_data, metadata = self.load_batch_data(frames_h5_path, poses_h5_path)
        
        if frames_data is None or poses_data is None:
            print("❌ 데이터 로드 실패")
            return {}
        
        results = {}
        
        # 1. 단일 프레임 샘플 저장
        if sample_single_frames > 0:
            print(f"\n📸 단일 프레임 샘플 생성 ({sample_single_frames}개)...")
            step = max(1, len(frames_data) // sample_single_frames)
            
            for i in range(0, len(frames_data), step):
                if len(results) >= sample_single_frames:
                    break
                
                frame_path = self.visualize_single_frame(frames_data[i], poses_data[i], i)
                if frame_path:
                    results[f'single_frame_{i}'] = frame_path
        
        # 2. 배치 그리드 생성
        if create_grid:
            print(f"\n📊 배치 그리드 생성...")
            grid_path = self.create_batch_grid(frames_data, poses_data, batch_name)
            if grid_path:
                results['batch_grid'] = grid_path
        
        # 3. 통계 시각화 생성
        if create_stats:
            print(f"\n📈 통계 시각화 생성...")
            stats_path = self.create_pose_statistics(poses_data, batch_name)
            if stats_path:
                results['statistics'] = stats_path
        
        # 4. 애니메이션 생성
        if create_anim and len(frames_data) > 1:
            print(f"\n🎬 애니메이션 생성...")
            anim_path = self.create_animation(frames_data, poses_data, batch_name)
            if anim_path:
                results['animation'] = anim_path
        
        print(f"\n✅ 배치 시각화 완료: {len(results)}개 결과물")
        for key, path in results.items():
            print(f"   - {key}: {os.path.basename(path)}")
        
        return results

def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description='HDF5 배치 데이터 시각화 도구')
    parser.add_argument('--frames', type=str, required=True, help='프레임 HDF5 파일 경로')
    parser.add_argument('--poses', type=str, required=True, help='포즈 HDF5 파일 경로')
    parser.add_argument('--output', type=str, default='h5_visualizations', help='출력 디렉토리')
    parser.add_argument('--batch-name', type=str, help='배치 이름 (자동 생성 시 생략)')
    parser.add_argument('--no-grid', action='store_true', help='그리드 생성 안함')
    parser.add_argument('--no-stats', action='store_true', help='통계 생성 안함')
    parser.add_argument('--no-anim', action='store_true', help='애니메이션 생성 안함')
    parser.add_argument('--samples', type=int, default=5, help='단일 프레임 샘플 수')
    
    args = parser.parse_args()
    
    # 입력 파일 확인
    if not os.path.exists(args.frames):
        print(f"❌ 프레임 파일 없음: {args.frames}")
        return
    
    if not os.path.exists(args.poses):
        print(f"❌ 포즈 파일 없음: {args.poses}")
        return
    
    # 시각화 도구 생성
    visualizer = HDF5Visualizer(output_dir=args.output)
    
    # 배치 처리 실행
    results = visualizer.process_batch_files(
        frames_h5_path=args.frames,
        poses_h5_path=args.poses,
        batch_name=args.batch_name,
        create_grid=not args.no_grid,
        create_stats=not args.no_stats,
        create_anim=not args.no_anim,
        sample_single_frames=args.samples
    )
    
    if results:
        print(f"\n🎯 시각화 결과:")
        print(f"📁 출력 디렉토리: {visualizer.output_dir}")
        for key, path in results.items():
            print(f"   {key}: {path}")
    else:
        print("❌ 시각화 실패")

if __name__ == "__main__":
    # 직접 실행 시 대화형 모드
    if len(os.sys.argv) == 1:
        print("🎨 HDF5 배치 데이터 시각화 도구")
        print("=" * 50)
        
        # HDF5 파일 검색
        hdf5_dir = Path("mmpose/jy/sign_language_dataset/hdf5_batches")
        if hdf5_dir.exists():
            frame_files = list(hdf5_dir.glob("*_frames.h5"))
            pose_files = list(hdf5_dir.glob("*_poses.h5"))
            
            if frame_files and pose_files:
                print(f"\n📁 사용 가능한 배치 파일:")
                batch_pairs = []
                
                for frame_file in sorted(frame_files):
                    batch_id = frame_file.stem.replace('_frames', '')
                    pose_file = hdf5_dir / f"{batch_id}_poses.h5"
                    
                    if pose_file.exists():
                        batch_pairs.append((str(frame_file), str(pose_file), batch_id))
                        print(f"   {len(batch_pairs)}. {batch_id}")
                
                if batch_pairs:
                    try:
                        choice = int(input(f"\n배치 선택 (1-{len(batch_pairs)}): ")) - 1
                        if 0 <= choice < len(batch_pairs):
                            frames_path, poses_path, batch_name = batch_pairs[choice]
                            
                            # 시각화 도구 생성
                            visualizer = HDF5Visualizer()
                            
                            # 처리 실행
                            results = visualizer.process_batch_files(
                                frames_h5_path=frames_path,
                                poses_h5_path=poses_path,
                                batch_name=batch_name
                            )
                        else:
                            print("❌ 잘못된 선택")
                    except ValueError:
                        print("❌ 숫자를 입력하세요")
                else:
                    print("❌ 매칭되는 배치 파일 쌍이 없습니다")
            else:
                print("❌ HDF5 파일을 찾을 수 없습니다")
        else:
            print("❌ HDF5 배치 디렉토리가 없습니다")
            print("스트림라인 처리기를 먼저 실행하여 HDF5 파일을 생성하세요")
    else:
        main()
