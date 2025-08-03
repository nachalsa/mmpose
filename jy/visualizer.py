#!/usr/bin/env python3
"""
Visualization Module for RTMW-x
RTMW-x WholeBody 포즈 시각화 및 정보 표시
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional


class RTMWVisualizer:
    """RTMW-x WholeBody 포즈 시각화"""
    
    def __init__(self):
        # WholeBody 키포인트 색상 정의
        self.body_color = (0, 0, 255)    # 빨간색 - Body (17개)
        self.face_color = (0, 255, 0)    # 초록색 - Face (68개)  
        self.hands_color = (255, 0, 0)   # 파란색 - Hands (48개)
        self.bbox_color = (0, 255, 255)  # 노란색 - 바운딩 박스
        
        # 키포인트별 색상 리스트 (133개)
        self.keypoint_colors = (
            # Body keypoints (17개) - 빨강 계열
            [(255, 0, 0), (255, 50, 50), (255, 100, 100)] * 6 +
            # Face keypoints (68개) - 초록 계열  
            [(0, 255, 0), (50, 255, 50), (100, 255, 100)] * 23 +
            # Hand keypoints (48개) - 파랑 계열
            [(0, 0, 255), (50, 50, 255), (100, 100, 255)] * 16
        )
        
        # 키포인트 크기
        self.body_radius = 3
        self.face_radius = 2  
        self.hands_radius = 2
        
        # COCO-17 스켈레톤 연결 정보 (Body 키포인트용)
        self.skeleton_links = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # 머리
            (5, 7), (7, 9), (6, 8), (8, 10), # 팔
            (5, 6), (5, 11), (6, 12),        # 몸통
            (11, 12), (11, 13), (13, 15),    # 다리
            (12, 14), (14, 16)               # 다리
        ]
        
    def draw_keypoints(self, frame: np.ndarray, keypoints: np.ndarray, 
                      person_id: int = 0) -> np.ndarray:
        """WholeBody 키포인트 시각화 (133개)
        
        Args:
            frame: 입력 이미지
            keypoints: 키포인트 좌표 (133, 2)
            person_id: 사람 ID
            
        Returns:
            키포인트가 그려진 이미지
        """
        if len(keypoints) < 133:
            print(f"⚠️ 키포인트 수가 부족합니다: {len(keypoints)}/133")
            return frame
            
        # 1. Body 키포인트 (0-16: 17개) - 빨간색
        for kpt_idx in range(min(17, len(keypoints))):
            x, y = int(keypoints[kpt_idx][0]), int(keypoints[kpt_idx][1])
            if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
                cv2.circle(frame, (x, y), self.body_radius, self.body_color, -1)
        
        # 2. Face 키포인트 (17-84: 68개) - 초록색  
        for kpt_idx in range(17, min(85, len(keypoints))):
            x, y = int(keypoints[kpt_idx][0]), int(keypoints[kpt_idx][1])
            if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
                cv2.circle(frame, (x, y), self.face_radius, self.face_color, -1)
        
        # 3. Hands 키포인트 (85-132: 48개) - 파란색
        for kpt_idx in range(85, min(133, len(keypoints))):
            x, y = int(keypoints[kpt_idx][0]), int(keypoints[kpt_idx][1])
            if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
                cv2.circle(frame, (x, y), self.hands_radius, self.hands_color, -1)
        
        return frame
    
    def draw_skeleton(self, frame: np.ndarray, keypoints: np.ndarray) -> np.ndarray:
        """Body 스켈레톤 연결선 그리기 (COCO-17 기준)
        
        Args:
            frame: 입력 이미지
            keypoints: 키포인트 좌표 (133, 2)
            
        Returns:
            스켈레톤이 그려진 이미지
        """
        if len(keypoints) < 17:
            return frame
            
        for link in self.skeleton_links:
            pt1_idx, pt2_idx = link
            if pt1_idx < len(keypoints) and pt2_idx < len(keypoints):
                x1, y1 = int(keypoints[pt1_idx][0]), int(keypoints[pt1_idx][1])
                x2, y2 = int(keypoints[pt2_idx][0]), int(keypoints[pt2_idx][1])
                
                # 좌표가 유효한 범위 내에 있는지 확인
                if (0 <= x1 < frame.shape[1] and 0 <= y1 < frame.shape[0] and
                    0 <= x2 < frame.shape[1] and 0 <= y2 < frame.shape[0]):
                    cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        return frame
    
    def _draw_skeleton(self, frame: np.ndarray, keypoints: np.ndarray) -> np.ndarray:
        """Body 스켈레톤 연결선 그리기 (내부 메서드)"""
        return self.draw_skeleton(frame, keypoints)
    
    def draw_bbox(self, frame: np.ndarray, bbox: Tuple[int, int, int, int], 
                  person_id: int = 0, label: str = None) -> np.ndarray:
        """바운딩 박스 그리기
        
        Args:
            frame: 입력 이미지
            bbox: 바운딩 박스 (x1, y1, x2, y2)
            person_id: 사람 ID
            label: 표시할 라벨
            
        Returns:
            바운딩 박스가 그려진 이미지
        """
        x1, y1, x2, y2 = bbox
        
        # 바운딩 박스 그리기
        cv2.rectangle(frame, (x1, y1), (x2, y2), self.bbox_color, 2)
        
        # 라벨 표시
        if label is None:
            label = f"Person {person_id + 1}"
            
        cv2.putText(frame, label, (x1, y1 - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def draw_info(self, frame: np.ndarray, frame_count: int, 
                  person_count: int, fps: float = 0, 
                  model_name: str = "RTMW-x") -> np.ndarray:
        """프레임 정보 표시
        
        Args:
            frame: 입력 이미지
            frame_count: 프레임 카운트
            person_count: 검출된 사람 수
            fps: FPS
            model_name: 모델 이름
            
        Returns:
            정보가 표시된 이미지
        """
        # 프레임 정보
        cv2.putText(frame, f"Frame: {frame_count}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 검출된 사람 수
        cv2.putText(frame, f"Persons: {person_count}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # FPS 표시
        if fps > 0:
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 모델 이름
        cv2.putText(frame, f"Model: {model_name}", (10, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 조작 안내
        cv2.putText(frame, "ESC: Exit | Red: Body | Green: Face | Blue: Hands", 
                   (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def visualize_pose(self, image: np.ndarray, keypoints: np.ndarray, 
                      bbox: List[float], person_id: int = 0) -> np.ndarray:
        """포즈 시각화"""
        if keypoints is None or len(keypoints) == 0:
            return image
        
        result_image = image.copy()
        
        # 키포인트 형태 확인 및 수정
        if keypoints.shape != (133, 2):
            print(f"⚠️ 예상치 못한 키포인트 형태: {keypoints.shape}")
            if keypoints.shape[1] == 3:  # (x, y, score) 형태인 경우
                keypoints = keypoints[:, :2]  # x, y만 사용
            elif keypoints.shape[0] != 133:
                print(f"❌ 잘못된 키포인트 수: {keypoints.shape[0]}")
                return result_image
        
        # 바운딩박스 그리기
        try:
            x1, y1, x2, y2 = map(int, bbox[:4])  # 처음 4개 값만 사용
            cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 사람 ID 표시
            label = f"Person {person_id + 1}"
            cv2.putText(result_image, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        except Exception as e:
            print(f"⚠️ 바운딩박스 그리기 실패: {e}, bbox: {bbox}")
        
        # 키포인트 그리기 (구분된 색상 사용)
        result_image = self.draw_keypoints(result_image, keypoints, person_id)
        
        # 스켈레톤 그리기 (Body keypoints만)
        result_image = self.draw_skeleton(result_image, keypoints)
        
        return result_image