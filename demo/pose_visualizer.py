#!/usr/bin/env python3
# Copyright (c) OpenMMLab. All rights reserved.
"""
MMPose 시각화 유틸리티 모듈

포즈 추정 결과를 시각화하는 다양한 함수들을 제공합니다.
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple


class PoseVisualizer:
    """포즈 시각화를 위한 클래스"""
    
    # COCO 17 키포인트 연결 정보
    COCO_17_SKELETON = [
        (0, 1), (0, 2), (1, 3), (2, 4),  # 머리
        (5, 6), (5, 7), (6, 8), (7, 9), (8, 10),  # 팔
        (5, 11), (6, 12), (11, 12),  # 몸통
        (11, 13), (12, 14), (13, 15), (14, 16)  # 다리
    ]
    
    # COCO 17 키포인트 이름
    COCO_17_KEYPOINTS = [
        'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
        'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
    ]
    
    # WholeBody 133 키포인트 스켈레톤 연결 정보
    WHOLEBODY_133_SKELETON = [
        # Body (17 keypoints)
        (0, 1), (0, 2), (1, 3), (2, 4),  # 머리
        (5, 6), (5, 7), (6, 8), (7, 9), (8, 10),  # 팔
        (5, 11), (6, 12), (11, 12),  # 몸통
        (11, 13), (12, 14), (13, 15), (14, 16),  # 다리
        
        # Face (68 keypoints: 17-84)
        # 얼굴 윤곽
        (17, 18), (18, 19), (19, 20), (20, 21), (21, 22), (22, 23), (23, 24), (24, 25), 
        (25, 26), (26, 27), (27, 28), (28, 29), (29, 30), (30, 31), (31, 32), (32, 33),
        # 오른쪽 눈썹
        (34, 35), (35, 36), (36, 37), (37, 38),
        # 왼쪽 눈썹
        (39, 40), (40, 41), (41, 42), (42, 43),
        # 코
        (44, 45), (45, 46), (46, 47), (47, 48), (48, 49), (49, 50), (50, 51), (51, 52),
        # 오른쪽 눈
        (53, 54), (54, 55), (55, 56), (56, 57), (57, 58), (58, 53),
        # 왼쪽 눈
        (59, 60), (60, 61), (61, 62), (62, 63), (63, 64), (64, 59),
        # 입
        (65, 66), (66, 67), (67, 68), (68, 69), (69, 70), (70, 71), (71, 72), (72, 73),
        (73, 74), (74, 75), (75, 76), (76, 77), (77, 78), (78, 79), (79, 80), (80, 81),
        (81, 82), (82, 83), (83, 84), (84, 65),
        
        # Left Hand (21 keypoints: 85-105)
        # 엄지
        (85, 86), (86, 87), (87, 88), (88, 89),
        # 검지
        (85, 90), (90, 91), (91, 92), (92, 93),
        # 중지
        (85, 94), (94, 95), (95, 96), (96, 97),
        # 약지
        (85, 98), (98, 99), (99, 100), (100, 101),
        # 새끼
        (85, 102), (102, 103), (103, 104), (104, 105),
        
        # Right Hand (21 keypoints: 106-126)
        # 엄지
        (106, 107), (107, 108), (108, 109), (109, 110),
        # 검지
        (106, 111), (111, 112), (112, 113), (113, 114),
        # 중지
        (106, 115), (115, 116), (116, 117), (117, 118),
        # 약지
        (106, 119), (119, 120), (120, 121), (121, 122),
        # 새끼
        (106, 123), (123, 124), (124, 125), (125, 126),
        
        # Left Foot (6 keypoints: 127-132)
        (127, 128), (128, 129), (129, 130), (130, 131), (131, 132),
        
        # Right Foot (6 keypoints: 133-138 -> but we only have 133 total)
        # 실제로는 132까지만 있음
        
        # Body와 다른 부위 연결
        (0, 17),   # nose to face_start
        (10, 85),  # left_wrist to left_hand
        (9, 106),  # right_wrist to right_hand
        (16, 127), # left_ankle to left_foot
        (15, 133), # right_ankle to right_foot (if exists)
    ]
    
    # 색상 정의 (부위별)
    COLORS = {
        'keypoint': (0, 255, 0),      # 초록색
        'skeleton': (255, 0, 0),      # 빨간색
        'bbox': (0, 255, 255),        # 노란색
        'text': (255, 255, 255),      # 흰색
        'error': (0, 0, 255),         # 빨간색
        'info': (0, 255, 0),          # 초록색
        
        # WholeBody 부위별 색상
        'body': (0, 255, 0),          # 초록색 - Body
        'face': (255, 0, 255),        # 마젠타 - Face
        'left_hand': (0, 255, 255),   # 시안 - Left Hand
        'right_hand': (255, 255, 0),  # 노란색 - Right Hand
        'left_foot': (255, 128, 0),   # 주황색 - Left Foot
        'right_foot': (128, 0, 255),  # 보라색 - Right Foot
    }
    
    def __init__(self, 
                 keypoint_threshold: float = 0.3,
                 keypoint_radius: int = 4,
                 skeleton_thickness: int = 2,
                 show_keypoint_labels: bool = False,
                 wholebody_mode: bool = True):
        """
        Args:
            keypoint_threshold: 키포인트 표시 임계값
            keypoint_radius: 키포인트 원의 반지름
            skeleton_thickness: 스켈레톤 선의 두께
            show_keypoint_labels: 키포인트 번호 표시 여부
            wholebody_mode: WholeBody 133점 모드 사용 여부
        """
        self.kpt_threshold = keypoint_threshold
        self.kpt_radius = keypoint_radius
        self.skeleton_thickness = skeleton_thickness
        self.show_labels = show_keypoint_labels
        self.wholebody_mode = wholebody_mode
    
    def get_keypoint_color(self, keypoint_idx: int) -> Tuple[int, int, int]:
        """키포인트 인덱스에 따른 색상 반환"""
        if not self.wholebody_mode or keypoint_idx < 17:
            return self.COLORS['body']
        elif keypoint_idx < 85:  # Face (17-84)
            return self.COLORS['face']
        elif keypoint_idx < 106:  # Left Hand (85-105)
            return self.COLORS['left_hand']
        elif keypoint_idx < 127:  # Right Hand (106-126)
            return self.COLORS['right_hand']
        elif keypoint_idx < 133:  # Left Foot (127-132)
            return self.COLORS['left_foot']
        else:  # Right Foot (133+)
            return self.COLORS['right_foot']
    
    def get_skeleton_color(self, link: Tuple[int, int]) -> Tuple[int, int, int]:
        """스켈레톤 연결에 따른 색상 반환"""
        idx1, idx2 = link
        # 두 키포인트 중 더 높은 인덱스의 색상 사용
        max_idx = max(idx1, idx2)
        return self.get_keypoint_color(max_idx)
    
    def draw_keypoints(self, 
                      image: np.ndarray, 
                      keypoints: np.ndarray, 
                      scores: Optional[np.ndarray] = None,
                      color: Tuple[int, int, int] = None) -> np.ndarray:
        """키포인트를 이미지에 그리기
        
        Args:
            image: 입력 이미지
            keypoints: 키포인트 좌표 (N, 2)
            scores: 키포인트 신뢰도 점수 (N,)
            color: 키포인트 색상 (None이면 부위별 색상 사용)
            
        Returns:
            키포인트가 그려진 이미지
        """
        display_img = image.copy()
        
        for i, kpt in enumerate(keypoints):
            score = scores[i] if scores is not None else 1.0
            
            if score > self.kpt_threshold:
                x, y = int(kpt[0]), int(kpt[1])
                if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                    # 색상 결정
                    kpt_color = color if color is not None else self.get_keypoint_color(i)
                    
                    # WholeBody 모드에서는 부위별로 다른 크기 사용
                    radius = self.kpt_radius
                    if self.wholebody_mode:
                        if i < 17:  # Body
                            radius = self.kpt_radius + 1
                        elif i < 85:  # Face
                            radius = max(1, self.kpt_radius - 1)
                        else:  # Hands and Feet
                            radius = max(1, self.kpt_radius - 1)
                    
                    cv2.circle(display_img, (x, y), radius, kpt_color, -1)
                    
                    if self.show_labels:
                        cv2.putText(display_img, str(i), (x+5, y-5), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, 
                                   self.COLORS['text'], 1)
        
        return display_img
    
    def draw_skeleton(self, 
                     image: np.ndarray, 
                     keypoints: np.ndarray, 
                     scores: Optional[np.ndarray] = None,
                     skeleton: List[Tuple[int, int]] = None,
                     color: Tuple[int, int, int] = None) -> np.ndarray:
        """스켈레톤을 이미지에 그리기
        
        Args:
            image: 입력 이미지
            keypoints: 키포인트 좌표 (N, 2)
            scores: 키포인트 신뢰도 점수 (N,)
            skeleton: 연결 정보 리스트
            color: 스켈레톤 색상 (None이면 부위별 색상 사용)
            
        Returns:
            스켈레톤이 그려진 이미지
        """
        display_img = image.copy()
        
        # 스켈레톤 선택
        if skeleton is None:
            skeleton = self.WHOLEBODY_133_SKELETON if self.wholebody_mode else self.COCO_17_SKELETON
        
        for link in skeleton:
            if link[0] < len(keypoints) and link[1] < len(keypoints):
                kpt1, kpt2 = keypoints[link[0]], keypoints[link[1]]
                score1 = scores[link[0]] if scores is not None else 1.0
                score2 = scores[link[1]] if scores is not None else 1.0
                
                if score1 > self.kpt_threshold and score2 > self.kpt_threshold:
                    x1, y1 = int(kpt1[0]), int(kpt1[1])
                    x2, y2 = int(kpt2[0]), int(kpt2[1])
                    
                    # 색상 결정
                    line_color = color if color is not None else self.get_skeleton_color(link)
                    
                    # WholeBody 모드에서는 부위별로 다른 두께 사용
                    thickness = self.skeleton_thickness
                    if self.wholebody_mode:
                        max_idx = max(link[0], link[1])
                        if max_idx < 17:  # Body
                            thickness = self.skeleton_thickness
                        elif max_idx < 85:  # Face
                            thickness = max(1, self.skeleton_thickness - 1)
                        else:  # Hands and Feet
                            thickness = max(1, self.skeleton_thickness - 1)
                    
                    cv2.line(display_img, (x1, y1), (x2, y2), line_color, thickness)
        
        return display_img
    
    def draw_bbox(self, 
                  image: np.ndarray, 
                  bbox: List[float],
                  color: Tuple[int, int, int] = None,
                  thickness: int = 2) -> np.ndarray:
        """바운딩 박스를 이미지에 그리기
        
        Args:
            image: 입력 이미지
            bbox: 바운딩 박스 [x1, y1, x2, y2]
            color: 박스 색상
            thickness: 선 두께
            
        Returns:
            바운딩 박스가 그려진 이미지
        """
        if color is None:
            color = self.COLORS['bbox']
            
        display_img = image.copy()
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(display_img, (x1, y1), (x2, y2), color, thickness)
        
        return display_img
    
    def draw_pose(self, 
                  image: np.ndarray, 
                  predictions: List[Any],
                  draw_bbox: bool = True,
                  draw_keypoints: bool = True,
                  draw_skeleton: bool = True) -> np.ndarray:
        """완전한 포즈를 이미지에 그리기
        
        Args:
            image: 입력 이미지
            predictions: MMPose 예측 결과
            draw_bbox: 바운딩 박스 그리기 여부
            draw_keypoints: 키포인트 그리기 여부
            draw_skeleton: 스켈레톤 그리기 여부
            
        Returns:
            포즈가 그려진 이미지
        """
        display_img = image.copy()
        
        try:
            for pred in predictions:
                if hasattr(pred, 'pred_instances'):
                    instances = pred.pred_instances
                    
                    # 바운딩 박스 그리기
                    if draw_bbox and hasattr(instances, 'bboxes'):
                        for bbox in instances.bboxes:
                            display_img = self.draw_bbox(display_img, bbox)
                    
                    # 키포인트와 스켈레톤 그리기
                    if hasattr(instances, 'keypoints'):
                        keypoints = instances.keypoints
                        scores = getattr(instances, 'keypoint_scores', None)
                        
                        for i, kpts in enumerate(keypoints):
                            kpt_scores = scores[i] if scores is not None else None
                            
                            # 키포인트 개수에 따라 모드 자동 감지
                            if len(kpts) > 17:
                                self.wholebody_mode = True
                            
                            # 스켈레톤 먼저 그리기 (키포인트 아래에 나타나도록)
                            if draw_skeleton:
                                display_img = self.draw_skeleton(
                                    display_img, kpts, kpt_scores)
                            
                            # 키포인트 그리기
                            if draw_keypoints:
                                display_img = self.draw_keypoints(
                                    display_img, kpts, kpt_scores)
                                
        except Exception as e:
            print(f"포즈 그리기 실패: {e}")
            
        return display_img
    
    def add_info_text(self, 
                     image: np.ndarray, 
                     texts: List[str],
                     position: Tuple[int, int] = (10, 30),
                     font_scale: float = 0.6,
                     color: Tuple[int, int, int] = None,
                     thickness: int = 2,
                     line_spacing: int = 30) -> np.ndarray:
        """정보 텍스트를 이미지에 추가
        
        Args:
            image: 입력 이미지
            texts: 표시할 텍스트 리스트
            position: 시작 위치 (x, y)
            font_scale: 폰트 크기
            color: 텍스트 색상
            thickness: 텍스트 두께
            line_spacing: 줄 간격
            
        Returns:
            텍스트가 추가된 이미지
        """
        if color is None:
            color = self.COLORS['info']
            
        display_img = image.copy()
        x, y = position
        
        for i, text in enumerate(texts):
            cv2.putText(display_img, text, (x, y + i * line_spacing),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
        
        return display_img
    
    def add_legend(self, 
                   image: np.ndarray,
                   position: Tuple[int, int] = None) -> np.ndarray:
        """WholeBody 모드에서 색상 범례 추가
        
        Args:
            image: 입력 이미지
            position: 범례 위치 (None이면 오른쪽 상단)
            
        Returns:
            범례가 추가된 이미지
        """
        if not self.wholebody_mode:
            return image
        
        display_img = image.copy()
        
        if position is None:
            x, y = image.shape[1] - 150, 30
        else:
            x, y = position
        
        legend_items = [
            ("Body", self.COLORS['body']),
            ("Face", self.COLORS['face']),
            ("L.Hand", self.COLORS['left_hand']),
            ("R.Hand", self.COLORS['right_hand']),
            ("L.Foot", self.COLORS['left_foot']),
            ("R.Foot", self.COLORS['right_foot'])
        ]
        
        for i, (label, color) in enumerate(legend_items):
            y_pos = y + i * 25
            # 색상 원
            cv2.circle(display_img, (x, y_pos), 6, color, -1)
            # 텍스트
            cv2.putText(display_img, label, (x + 15, y_pos + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.COLORS['text'], 1)
        
        return display_img


def extract_mmpose_results(results: List[Any]) -> Tuple[List[Any], Dict[str, Any]]:
    """MMPose 결과에서 시각화 데이터와 예측 결과 추출
    
    Args:
        results: MMPose inferencer 결과
        
    Returns:
        (predictions, metadata) 튜플
    """
    predictions = []
    metadata = {}
    
    if not results or len(results) == 0:
        return predictions, metadata
    
    result = results[0]
    
    # 다양한 결과 형태 처리
    if isinstance(result, dict):
        # visualization 키가 있는 경우
        if 'visualization' in result:
            metadata['visualization'] = result['visualization']
        
        # predictions 키가 있는 경우
        if 'predictions' in result:
            predictions = result['predictions']
            
    return predictions, metadata


def create_fps_counter():
    """FPS 계산을 위한 카운터 생성"""
    return {
        'count': 0,
        'start_time': None,
        'current_fps': 0.0
    }


def update_fps_counter(fps_counter: Dict[str, Any], update_interval: int = 30) -> float:
    """FPS 카운터 업데이트
    
    Args:
        fps_counter: FPS 카운터 딕셔너리
        update_interval: FPS 업데이트 간격 (프레임 수)
        
    Returns:
        현재 FPS 값
    """
    import time
    
    fps_counter['count'] += 1
    
    if fps_counter['start_time'] is None:
        fps_counter['start_time'] = time.time()
    
    if fps_counter['count'] % update_interval == 0:
        elapsed = time.time() - fps_counter['start_time']
        fps_counter['current_fps'] = update_interval / elapsed
        fps_counter['start_time'] = time.time()
    
    return fps_counter['current_fps']


def handle_mmpose_visualization(image: np.ndarray, 
                               results: List[Any],
                               inferencer: Any,
                               visualizer: PoseVisualizer) -> np.ndarray:
    """MMPose 결과를 처리하여 시각화된 이미지 반환
    
    Args:
        image: 원본 이미지
        results: MMPose 결과
        inferencer: MMPose inferencer 객체
        visualizer: PoseVisualizer 객체
        
    Returns:
        시각화된 이미지
    """
    display_frame = image.copy()
    
    if not results or len(results) == 0:
        return visualizer.add_info_text(display_frame, ["No Pose Detected"], 
                                       color=visualizer.COLORS['error'])
    
    result = results[0]
    
    # 방법 1: visualization 키가 있는 경우
    if isinstance(result, dict) and 'visualization' in result:
        vis_data = result['visualization']
        if isinstance(vis_data, list) and len(vis_data) > 0:
            return vis_data[0]
        elif isinstance(vis_data, np.ndarray):
            return vis_data
    
    # 방법 2: predictions가 있는 경우 수동으로 시각화
    predictions, metadata = extract_mmpose_results(results)
    
    if len(predictions) > 0:
        # MMPose visualizer 사용 시도
        try:
            vis_result = inferencer.visualize(
                [image], 
                predictions,
                return_vis=True,
                show=False,
                draw_bbox=True,
                radius=visualizer.kpt_radius,
                thickness=visualizer.skeleton_thickness,
                kpt_thr=visualizer.kpt_threshold
            )
            if vis_result and len(vis_result) > 0:
                # WholeBody 모드에서 범례 추가
                result_img = vis_result[0]
                if visualizer.wholebody_mode:
                    result_img = visualizer.add_legend(result_img)
                return result_img
        except Exception as vis_e:
            print(f"MMPose 시각화 실패: {vis_e}")
        
        # 커스텀 시각화 사용
        display_frame = visualizer.draw_pose(image, predictions)
        
        # 키포인트 개수 정보 확인
        keypoint_count = 0
        try:
            if len(predictions) > 0 and hasattr(predictions[0], 'pred_instances'):
                instances = predictions[0].pred_instances
                if hasattr(instances, 'keypoints') and len(instances.keypoints) > 0:
                    keypoint_count = len(instances.keypoints[0])
        except:
            pass
        
        # 포즈 개수와 키포인트 개수 정보 추가
        info_texts = [f"Poses: {len(predictions)}"]
        if keypoint_count > 0:
            info_texts.append(f"Keypoints: {keypoint_count}")
            if keypoint_count > 17:
                info_texts.append("Mode: WholeBody")
            else:
                info_texts.append("Mode: Body")
        
        display_frame = visualizer.add_info_text(display_frame, info_texts, 
                                                position=(10, 120))
        
        # WholeBody 모드에서 범례 추가
        if visualizer.wholebody_mode:
            display_frame = visualizer.add_legend(display_frame)
            
    else:
        display_frame = visualizer.add_info_text(display_frame, ["No Pose Detected"], 
                                               position=(10, 120),
                                               color=visualizer.COLORS['error'])
    
    return display_frame