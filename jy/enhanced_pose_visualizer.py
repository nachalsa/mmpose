#!/usr/bin/env python3
"""
Enhanced Pose Visualizer
HDF5 파일, JPEG 이미지, 비디오 파일을 모두 지원하는 통합 포즈 시각화 도구
"""

import os
import cv2
import h5py
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any, Union
import argparse
from tqdm import tqdm
import glob

# 추론기 임포트
try:
    from onnx_inferencer import YOLO11LRTMWONNXInferencer
    INFERENCER_AVAILABLE = True
except ImportError:
    print("⚠️ ONNX 추론기가 없습니다. HDF5 파일 시각화만 가능합니다.")
    INFERENCER_AVAILABLE = False

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

def get_keypoint_color(keypoint_idx: int) -> Tuple[int, int, int]:
    """키포인트 인덱스에 따른 색상 반환"""
    if 0 <= keypoint_idx <= 67:      # 얼굴
        return POSE_COLORS['face']
    elif 68 <= keypoint_idx <= 88:   # 신체
        return POSE_COLORS['body']
    elif 89 <= keypoint_idx <= 109:  # 왼손
        return POSE_COLORS['left_hand']
    elif 110 <= keypoint_idx <= 130: # 오른손
        return POSE_COLORS['right_hand']
    else:                            # 발
        return POSE_COLORS['foot']

class EnhancedPoseVisualizer:
    """Enhanced 포즈 데이터 시각화기 - HDF5, 이미지, 비디오 모두 지원"""
    
    def __init__(self, input_source: Union[str, Tuple[str, str]], use_inferencer: bool = False):
        """
        Args:
            input_source: 입력 소스
                - HDF5 파일인 경우: (frames_h5_path, poses_h5_path) 튜플
                - 이미지/비디오인 경우: 파일 경로 문자열 또는 디렉토리 경로
            use_inferencer: 실시간 포즈 추론 사용 여부
        """
        self.use_inferencer = use_inferencer and INFERENCER_AVAILABLE
        
        # 입력 소스 분류
        if isinstance(input_source, tuple) and len(input_source) == 2:
            # HDF5 파일 모드
            self.mode = "hdf5"
            self.frames_path = Path(input_source[0])
            self.poses_path = Path(input_source[1])
            
            if not self.frames_path.exists():
                raise FileNotFoundError(f"프레임 HDF5 파일을 찾을 수 없습니다: {self.frames_path}")
            if not self.poses_path.exists():
                raise FileNotFoundError(f"포즈 HDF5 파일을 찾을 수 없습니다: {self.poses_path}")
                
            print(f"✅ HDF5 모드: {self.frames_path.name}, {self.poses_path.name}")
            self._load_hdf5_video_list()
            
        else:
            # 이미지/비디오 파일 모드
            input_path = Path(input_source)
            
            if input_path.is_file():
                # 단일 파일
                if input_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    self.mode = "image"
                    self.image_files = [input_path]
                    print(f"✅ 이미지 모드: {input_path.name}")
                elif input_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
                    self.mode = "video"
                    self.video_file = input_path
                    print(f"✅ 비디오 모드: {input_path.name}")
                else:
                    raise ValueError(f"지원되지 않는 파일 형식: {input_path.suffix}")
                    
            elif input_path.is_dir():
                # 이미지 디렉토리
                image_exts = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
                self.image_files = []
                for ext in image_exts:
                    self.image_files.extend(input_path.glob(ext))
                    self.image_files.extend(input_path.glob(ext.upper()))
                
                if not self.image_files:
                    raise ValueError(f"이미지 파일을 찾을 수 없습니다: {input_path}")
                    
                self.mode = "images"
                self.image_files.sort()
                print(f"✅ 이미지 디렉토리 모드: {len(self.image_files)}개 파일")
                
            else:
                raise FileNotFoundError(f"입력 소스를 찾을 수 없습니다: {input_source}")
        
        # 추론기 초기화
        if self.use_inferencer:
            self._init_inferencer()
    
    def _init_inferencer(self):
        """ONNX 추론기 초기화"""
        try:
            # RTMW-DW-X-L 모델 사용 (최고 정확도)
            model_path = "../models/rtmw-dw-x-l_simcc-cocktail14_270e-384x288.onnx"
            self.inferencer = YOLO11LRTMWONNXInferencer(
                rtmw_onnx_path=model_path,
                detection_device="auto",
                pose_device="auto",
                optimize_for_accuracy=True
            )
            print("✅ ONNX 추론기 초기화 완료")
        except Exception as e:
            print(f"❌ 추론기 초기화 실패: {e}")
            self.use_inferencer = False
    
    def _load_hdf5_video_list(self):
        """HDF5에서 비디오 목록 로드"""
        with h5py.File(self.frames_path, 'r') as f:
            self.video_ids = list(f.keys())
            self.video_ids.sort()
        
        print(f"📹 총 {len(self.video_ids)}개 비디오 발견")
        for i, video_id in enumerate(self.video_ids[:5]):
            print(f"   {i+1}. {video_id}")
        if len(self.video_ids) > 5:
            print(f"   ... 외 {len(self.video_ids) - 5}개")
    
    def get_hdf5_video_data(self, video_id: str) -> Dict[str, np.ndarray]:
        """HDF5에서 특정 비디오의 데이터 로드"""
        try:
            with h5py.File(self.frames_path, 'r') as f_frames, \
                 h5py.File(self.poses_path, 'r') as f_poses:
                
                # 프레임 데이터 (JPEG 바이트 스트림 또는 일반 프레임)
                if f"{video_id}/frames_jpeg" in f_frames:
                    # JPEG 바이트 스트림 형태
                    jpeg_frames_bytes = f_frames[f"{video_id}/frames_jpeg"][:]
                    frames = []
                    for jpeg_bytes in jpeg_frames_bytes:
                        if len(jpeg_bytes) > 0:  # 유효한 JPEG 데이터인지 확인
                            frame = cv2.imdecode(np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR)
                            if frame is not None:
                                frames.append(frame)
                elif f"{video_id}/frames" in f_frames:
                    # 일반 프레임 배열 형태
                    frames = f_frames[f"{video_id}/frames"][:]
                    # NumPy 배열을 리스트로 변환
                    if isinstance(frames, np.ndarray):
                        frames = [frames[i] for i in range(len(frames))]
                else:
                    print(f"⚠️ {video_id}: 프레임 데이터를 찾을 수 없습니다")
                    return None
                
                if not frames:
                    print(f"⚠️ {video_id}: 유효한 프레임을 찾을 수 없습니다")
                    return None
                
                frames = np.array(frames)
                
                # 메타데이터 로드
                metadata_json = f_frames[f"{video_id}/metadata"][()]
                if isinstance(metadata_json, bytes):
                    metadata_json = metadata_json.decode('utf-8')
                metadata = json.loads(metadata_json)
                
                # 포즈 데이터 로드
                if f"{video_id}/keypoints_scaled" in f_poses:
                    # 스케일링된 데이터 사용 (8로 나누기)
                    keypoints_scaled = f_poses[f"{video_id}/keypoints_scaled"][:]
                    keypoints = keypoints_scaled / 8.0
                elif f"{video_id}/keypoints" in f_poses:
                    # 일반 키포인트 데이터
                    keypoints = f_poses[f"{video_id}/keypoints"][:]
                else:
                    print(f"⚠️ {video_id}: 키포인트 데이터를 찾을 수 없습니다")
                    return None
                
                if f"{video_id}/scores" in f_poses:
                    scores = f_poses[f"{video_id}/scores"][:]
                else:
                    # 점수가 없으면 모든 키포인트에 기본값 할당
                    scores = np.ones(keypoints.shape[:2])  # (프레임수, 키포인트수) 모양
                
                return {
                    'frames': frames,
                    'keypoints': keypoints,
                    'scores': scores,
                    'metadata': metadata
                }
                
        except Exception as e:
            print(f"❌ {video_id} 데이터 로드 실패: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def infer_pose_from_image(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """이미지에서 포즈 추론"""
        if not self.use_inferencer:
            return np.array([]), np.array([])
        
        try:
            # process_frame 메서드 사용 (올바른 메서드명)
            _, results = self.inferencer.process_frame(image)
            if results and len(results) > 0:
                # 첫 번째 사람의 포즈만 반환
                keypoints = results[0][0]  # 키포인트
                scores = results[0][1]     # 점수
                
                # keypoints가 (133, 2) 형태인지 확인
                if keypoints.shape[1] == 2:
                    return keypoints, scores
                else:
                    print(f"⚠️ 예상치 못한 키포인트 형태: {keypoints.shape}")
                    return np.array([]), np.array([])
            else:
                return np.array([]), np.array([])
        except Exception as e:
            print(f"⚠️ 포즈 추론 실패: {e}")
            return np.array([]), np.array([])
    
    def draw_pose_on_image(self, image: np.ndarray, keypoints: np.ndarray, scores: np.ndarray, 
                          conf_threshold: float = 0.3) -> np.ndarray:
        """이미지에 포즈 키포인트와 스켈레톤 그리기"""
        if len(keypoints) == 0 or len(scores) == 0:
            return image
        
        img_vis = image.copy()
        
        # 스켈레톤 그리기
        for connection in RTMW_SKELETON:
            pt1_idx, pt2_idx = connection
            
            # 인덱스 범위 확인
            if pt1_idx >= len(keypoints) or pt2_idx >= len(keypoints):
                continue
                
            # 신뢰도 확인
            if scores[pt1_idx] < conf_threshold or scores[pt2_idx] < conf_threshold:
                continue
            
            pt1 = tuple(map(int, keypoints[pt1_idx]))
            pt2 = tuple(map(int, keypoints[pt2_idx]))
            
            # 색상 결정 (두 점의 평균 색상) - 정수형 변환
            color1 = get_keypoint_color(pt1_idx)
            color2 = get_keypoint_color(pt2_idx)
            color = tuple(map(int, (np.array(color1) + np.array(color2)) // 2))
            
            # 선 그리기
            cv2.line(img_vis, pt1, pt2, color, 2)
        
        # 키포인트 그리기
        for i, (kp, score) in enumerate(zip(keypoints, scores)):
            if score < conf_threshold:
                continue
                
            center = tuple(map(int, kp))
            color = get_keypoint_color(i)
            
            # 키포인트 원 그리기
            cv2.circle(img_vis, center, 3, color, -1)
            cv2.circle(img_vis, center, 4, (255, 255, 255), 1)  # 흰색 테두리
        
        return img_vis
    
    def visualize_hdf5_video(self, video_id: str, output_dir: str = None, 
                           conf_threshold: float = 0.3, max_frames: int = None):
        """HDF5 비디오 데이터 시각화"""
        if self.mode != "hdf5":
            print("❌ HDF5 모드가 아닙니다")
            return
            
        print(f"🎬 비디오 시각화 시작: {video_id}")
        
        # 데이터 로드
        video_data = self.get_hdf5_video_data(video_id)
        if video_data is None:
            return
        
        frames = video_data['frames']
        keypoints = video_data['keypoints']
        scores = video_data['scores']
        metadata = video_data['metadata']
        
        print(f"   - 프레임 수: {len(frames)}")
        print(f"   - 해상도: {frames[0].shape[:2]}")
        print(f"   - 메타데이터: {metadata}")
        
        # 출력 디렉토리 설정
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
        
        # 프레임별 시각화
        frame_limit = min(len(frames), max_frames or len(frames))
        for i in tqdm(range(frame_limit), desc="프레임 처리"):
            frame = frames[i]
            kp = keypoints[i] if i < len(keypoints) else np.array([])
            sc = scores[i] if i < len(scores) else np.array([])
            
            # 포즈 그리기
            vis_frame = self.draw_pose_on_image(frame, kp, sc, conf_threshold)
            
            # 저장 또는 표시
            if output_dir:
                output_file = output_path / f"{video_id}_frame_{i:04d}.jpg"
                cv2.imwrite(str(output_file), vis_frame)
            else:
                # 창에 표시
                cv2.imshow(f'Pose Visualization - {video_id}', vis_frame)
                key = cv2.waitKey(30) & 0xFF
                if key == ord('q'):
                    break
        
        if not output_dir:
            cv2.destroyAllWindows()
        
        print(f"✅ 시각화 완료: {video_id}")
    
    def visualize_images(self, output_dir: str = None, conf_threshold: float = 0.3):
        """이미지 파일들 시각화"""
        if self.mode not in ["image", "images"]:
            print("❌ 이미지 모드가 아닙니다")
            return
        
        print(f"🖼️ 이미지 시각화 시작: {len(self.image_files)}개 파일")
        
        # 출력 디렉토리 설정
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
        
        for img_file in tqdm(self.image_files, desc="이미지 처리"):
            # 이미지 로드
            image = cv2.imread(str(img_file))
            if image is None:
                print(f"⚠️ 이미지 로드 실패: {img_file}")
                continue
            
            # 포즈 추론 (추론기가 있는 경우)
            if self.use_inferencer:
                keypoints, scores = self.infer_pose_from_image(image)
            else:
                keypoints, scores = np.array([]), np.array([])
            
            # 포즈 그리기
            vis_image = self.draw_pose_on_image(image, keypoints, scores, conf_threshold)
            
            # 저장 또는 표시
            if output_dir:
                output_file = output_path / f"{img_file.stem}_pose.jpg"
                cv2.imwrite(str(output_file), vis_image)
            else:
                # 창에 표시
                cv2.imshow(f'Pose Visualization - {img_file.name}', vis_image)
                key = cv2.waitKey(0) & 0xFF
                if key == ord('q'):
                    break
        
        if not output_dir:
            cv2.destroyAllWindows()
        
        print(f"✅ 이미지 시각화 완료")
    
    def visualize_video(self, output_path: str = None, conf_threshold: float = 0.3):
        """비디오 파일 시각화"""
        if self.mode != "video":
            print("❌ 비디오 모드가 아닙니다")
            return
        
        print(f"🎬 비디오 시각화 시작: {self.video_file.name}")
        
        # 비디오 캡처
        cap = cv2.VideoCapture(str(self.video_file))
        if not cap.isOpened():
            print(f"❌ 비디오 파일을 열 수 없습니다: {self.video_file}")
            return
        
        # 비디오 정보
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"   - FPS: {fps}")
        print(f"   - 해상도: {width}x{height}")
        print(f"   - 총 프레임: {total_frames}")
        
        # 출력 비디오 설정
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        with tqdm(total=total_frames, desc="프레임 처리") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 포즈 추론 (추론기가 있는 경우)
                if self.use_inferencer:
                    keypoints, scores = self.infer_pose_from_image(frame)
                else:
                    keypoints, scores = np.array([]), np.array([])
                
                # 포즈 그리기
                vis_frame = self.draw_pose_on_image(frame, keypoints, scores, conf_threshold)
                
                # 저장 또는 표시
                if output_path:
                    out_writer.write(vis_frame)
                else:
                    cv2.imshow(f'Pose Visualization - {self.video_file.name}', vis_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                
                frame_count += 1
                pbar.update(1)
        
        # 정리
        cap.release()
        if output_path:
            out_writer.release()
        else:
            cv2.destroyAllWindows()
        
        print(f"✅ 비디오 시각화 완료: {frame_count}프레임 처리")

def main():
    parser = argparse.ArgumentParser(description='Enhanced Pose Visualizer')
    parser.add_argument('input', help='입력 소스 (HDF5 파일 경로, 이미지 파일, 비디오 파일, 또는 디렉토리)')
    parser.add_argument('--poses', help='포즈 HDF5 파일 경로 (HDF5 모드용)')
    parser.add_argument('--output', '-o', help='출력 디렉토리 또는 파일 경로')
    parser.add_argument('--conf', type=float, default=0.3, help='신뢰도 임계값 (기본값: 0.3)')
    parser.add_argument('--max-frames', type=int, help='최대 처리 프레임 수')
    parser.add_argument('--video-id', help='HDF5에서 시각화할 비디오 ID')
    parser.add_argument('--use-inferencer', action='store_true', help='실시간 포즈 추론 사용')
    
    args = parser.parse_args()
    
    try:
        # 입력 소스 결정
        if args.poses:
            # HDF5 모드
            input_source = (args.input, args.poses)
            print(f"🗂️ HDF5 모드: {args.input} + {args.poses}")
        else:
            # 파일/디렉토리 모드
            input_source = args.input
            print(f"📁 파일 모드: {args.input}")
        
        # 시각화기 생성
        visualizer = EnhancedPoseVisualizer(input_source, use_inferencer=args.use_inferencer)
        
        # 모드별 시각화 실행
        if visualizer.mode == "hdf5":
            if args.video_id:
                # 특정 비디오 시각화
                visualizer.visualize_hdf5_video(
                    args.video_id, 
                    output_dir=args.output,
                    conf_threshold=args.conf,
                    max_frames=args.max_frames
                )
            else:
                # 모든 비디오 목록 표시
                print(f"\n📹 사용 가능한 비디오 ID:")
                for i, video_id in enumerate(visualizer.video_ids):
                    print(f"   {i+1}. {video_id}")
                print(f"\n사용법: --video-id VIDEO_ID")
                
        elif visualizer.mode in ["image", "images"]:
            # 이미지 시각화
            visualizer.visualize_images(
                output_dir=args.output,
                conf_threshold=args.conf
            )
            
        elif visualizer.mode == "video":
            # 비디오 시각화
            visualizer.visualize_video(
                output_path=args.output,
                conf_threshold=args.conf
            )
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
