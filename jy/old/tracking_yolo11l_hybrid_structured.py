#!/usr/bin/env python3
"""
트래킹 기반 YOLO11L + RTMW 하이브리드 수화 인식 시스템
- 포즈 기반 트래킹으로 안정적인 바운딩박스 예측
- 어깨 움직임 보존으로 수화 특징 최적화  
- Intel XPU 가속 지원

Author: JY
Created: 2024
"""

import os
import sys
import time
import cv2
import numpy as np
import torch
from typing import List, Tuple, Dict, Any, Optional
from ultralytics import YOLO
from collections import deque
import math

# MMPose imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from mmpose.apis import init_model, inference_topdown
from mmpose.structures import merge_data_samples


# ============================================================================
# 1. 유틸리티 함수들
# ============================================================================

def check_xpu_availability() -> bool:
    """Intel XPU 사용 가능 여부 확인"""
    try:
        import intel_extension_for_pytorch as ipex
        if hasattr(ipex, 'xpu') and ipex.xpu.is_available():
            device_count = ipex.xpu.device_count()
            print(f"✅ Intel XPU 사용 가능: {device_count}개 디바이스 감지")
            return True
        else:
            print("⚠️ Intel XPU 미사용 가능")
            return False
    except ImportError:
        print("⚠️ Intel Extension for PyTorch 미설치")
        return False


def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """IoU (Intersection over Union) 계산"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    return intersection / (area1 + area2 - intersection)


def smooth_bbox(current_bbox: List[float], previous_bbox: List[float], alpha: float = 0.7) -> List[float]:
    """바운딩박스 스무딩"""
    return [
        alpha * current_bbox[i] + (1 - alpha) * previous_bbox[i]
        for i in range(4)
    ]


# ============================================================================
# 2. 기본 트래커 클래스
# ============================================================================

class PersonTracker:
    """사람 트래킹 클래스 - 포즈 기반 바운딩박스 예측"""
    
    def __init__(self, track_id: int, initial_bbox: List[float], stability_factor: float = 0.3):
        # 기본 설정
        self.track_id = track_id
        self.current_bbox = initial_bbox  # [x1, y1, x2, y2]
        self.stability_factor = stability_factor
        self.is_active = True
        self.missing_frames = 0
        self.max_missing_frames = 30
        
        # 히스토리 관리
        self.bbox_history = deque(maxlen=10)
        self.pose_history = deque(maxlen=5)
        self.bbox_history.append(initial_bbox)
        
        # 바운딩박스 안정화 설정
        self.bbox_stability_threshold = 0.9
        self.min_bbox_change = 5.0
        
        # 수화 특화 키포인트 (COCO-WholeBody)
        self.shoulder_keypoints = [5, 6]    # 어깨
        self.wrist_keypoints = [9, 10]      # 손목
        self.elbow_keypoints = [7, 8]       # 팔꿈치
        
    def _get_keypoint_bbox(self, keypoints: np.ndarray, scores: np.ndarray, 
                          keypoint_indices: List[int], padding: float = 20.0) -> Optional[List[float]]:
        """특정 키포인트들로부터 바운딩박스 계산"""
        valid_points = []
        
        for idx in keypoint_indices:
            if idx < len(keypoints) and scores[idx] > 0.3:
                valid_points.append(keypoints[idx])
        
        if len(valid_points) < 2:
            return None
        
        valid_points = np.array(valid_points)
        x_min, y_min = np.min(valid_points, axis=0)
        x_max, y_max = np.max(valid_points, axis=0)
        
        return [
            max(0, x_min - padding),
            max(0, y_min - padding),
            x_max + padding,
            y_max + padding
        ]
    
    def predict_next_bbox(self, keypoints: np.ndarray, scores: np.ndarray) -> List[float]:
        """포즈 기반 다음 바운딩박스 예측"""
        # 어깨-손목 기반 예측 (1순위)
        shoulder_wrist_indices = self.shoulder_keypoints + self.wrist_keypoints
        predicted_bbox = self._get_keypoint_bbox(keypoints, scores, shoulder_wrist_indices, padding=50.0)
        
        if predicted_bbox is None:
            # 어깨-팔꿈치 기반 예측 (2순위)
            shoulder_elbow_indices = self.shoulder_keypoints + self.elbow_keypoints
            predicted_bbox = self._get_keypoint_bbox(keypoints, scores, shoulder_elbow_indices, padding=40.0)
        
        if predicted_bbox is None:
            # 히스토리 기반 예측 (폴백)
            return self._predict_bbox_from_history()
        
        # 바운딩박스 안정화
        if self.bbox_history:
            last_bbox = self.bbox_history[-1]
            
            # 변화량 계산
            change_ratio = max(
                abs(predicted_bbox[2] - predicted_bbox[0] - (last_bbox[2] - last_bbox[0])) / (last_bbox[2] - last_bbox[0]),
                abs(predicted_bbox[3] - predicted_bbox[1] - (last_bbox[3] - last_bbox[1])) / (last_bbox[3] - last_bbox[1])
            )
            
            # 급격한 변화 억제
            if change_ratio > (1 - self.bbox_stability_threshold):
                predicted_bbox = smooth_bbox(predicted_bbox, last_bbox, alpha=self.stability_factor)
        
        return predicted_bbox
    
    def _predict_bbox_from_history(self) -> List[float]:
        """히스토리 기반 바운딩박스 예측 (폴백)"""
        if len(self.bbox_history) >= 2:
            # 선형 예측
            prev_bbox = self.bbox_history[-2]
            curr_bbox = self.bbox_history[-1]
            
            predicted_bbox = [
                curr_bbox[i] + (curr_bbox[i] - prev_bbox[i]) * 0.5
                for i in range(4)
            ]
            return predicted_bbox
        elif self.bbox_history:
            return self.bbox_history[-1].copy()
        else:
            return self.current_bbox.copy()
    
    def update(self, pose: np.ndarray, scores: np.ndarray, image_shape: Tuple[int, int]) -> List[float]:
        """트래커 업데이트"""
        # 다음 바운딩박스 예측
        predicted_bbox = self.predict_next_bbox(pose, scores)
        
        # 이미지 경계 클리핑
        h, w = image_shape[:2]
        predicted_bbox[0] = max(0, min(predicted_bbox[0], w-1))
        predicted_bbox[1] = max(0, min(predicted_bbox[1], h-1))
        predicted_bbox[2] = max(predicted_bbox[0]+10, min(predicted_bbox[2], w))
        predicted_bbox[3] = max(predicted_bbox[1]+10, min(predicted_bbox[3], h))
        
        # 히스토리 업데이트
        self.current_bbox = predicted_bbox
        self.bbox_history.append(predicted_bbox)
        self.pose_history.append((pose.copy(), scores.copy()))
        
        self.missing_frames = 0
        return predicted_bbox
    
    def mark_missing(self):
        """프레임에서 놓친 경우"""
        self.missing_frames += 1
        if self.missing_frames > self.max_missing_frames:
            self.is_active = False


# ============================================================================
# 3. 수화 특화 어깨 움직임 보존 트래커
# ============================================================================

class SignLanguageShoulderTracker(PersonTracker):
    """수화 인식을 위한 어깨 움직임 보존 트래커"""
    
    def __init__(self, track_id: int, initial_bbox: List[float], stability_factor: float = 0.3):
        super().__init__(track_id, initial_bbox, stability_factor)
        
        # 어깨 움직임 분석
        self.shoulder_history = deque(maxlen=15)  # 어깨 위치 히스토리
        self.shoulder_patterns = {
            'elevation': deque(maxlen=20),     # 어깨 높낮이
            'rotation': deque(maxlen=20),      # 어깨 기울기
            'movement': deque(maxlen=20),      # 어깨 움직임 크기
            'asymmetry': deque(maxlen=20)      # 어깨 비대칭
        }
        
        # 수화 특화 설정
        self.shoulder_movement_threshold = 8.0
        self.adaptive_bbox_policy = "shoulder_aware"  # "fixed" or "shoulder_aware"
        self.movement_amplification = 0.8
        
    def analyze_shoulder_movement(self, keypoints: np.ndarray, scores: np.ndarray) -> Dict[str, Any]:
        """어깨 움직임 분석"""
        shoulder_info = {
            'left_shoulder': None,
            'right_shoulder': None,
            'elevation_change': 0.0,
            'rotation_angle': 0.0,
            'asymmetry_level': 0.0,
            'movement_magnitude': 0.0
        }
        
        # 어깨 키포인트 추출
        left_shoulder_idx, right_shoulder_idx = 5, 6
        
        if (left_shoulder_idx < len(keypoints) and scores[left_shoulder_idx] > 0.3 and
            right_shoulder_idx < len(keypoints) and scores[right_shoulder_idx] > 0.3):
            
            left_shoulder = keypoints[left_shoulder_idx].copy()
            right_shoulder = keypoints[right_shoulder_idx].copy()
            
            shoulder_info['left_shoulder'] = left_shoulder
            shoulder_info['right_shoulder'] = right_shoulder
            
            # 어깨 중심점
            shoulder_center = (left_shoulder + right_shoulder) / 2
            current_shoulders = {
                'left': left_shoulder,
                'right': right_shoulder,
                'center': shoulder_center
            }
            
            # 이전 프레임과 비교
            if self.shoulder_history:
                prev_shoulders = self.shoulder_history[-1]
                
                # 높낮이 변화 (어깨 올리기/내리기)
                prev_center_y = prev_shoulders['center'][1]
                curr_center_y = shoulder_center[1]
                elevation_change = curr_center_y - prev_center_y
                shoulder_info['elevation_change'] = elevation_change
                
                # 회전 각도 (어깨 기울기)
                prev_angle = math.atan2(
                    prev_shoulders['right'][1] - prev_shoulders['left'][1],
                    prev_shoulders['right'][0] - prev_shoulders['left'][0]
                )
                curr_angle = math.atan2(
                    right_shoulder[1] - left_shoulder[1],
                    right_shoulder[0] - left_shoulder[0]
                )
                rotation_angle = math.degrees(curr_angle - prev_angle)
                shoulder_info['rotation_angle'] = rotation_angle
                
                # 비대칭 정도 (한쪽 어깨만 움직임)
                left_movement = np.linalg.norm(left_shoulder - prev_shoulders['left'])
                right_movement = np.linalg.norm(right_shoulder - prev_shoulders['right'])
                asymmetry = abs(left_movement - right_movement)
                shoulder_info['asymmetry_level'] = asymmetry
                
                # 전체 움직임 크기
                movement_magnitude = np.linalg.norm(shoulder_center - prev_shoulders['center'])
                shoulder_info['movement_magnitude'] = movement_magnitude
                
                # 패턴 히스토리 업데이트
                self.shoulder_patterns['elevation'].append(elevation_change)
                self.shoulder_patterns['rotation'].append(rotation_angle)
                self.shoulder_patterns['movement'].append(movement_magnitude)
                self.shoulder_patterns['asymmetry'].append(asymmetry)
            
            # 어깨 히스토리 업데이트
            self.shoulder_history.append(current_shoulders)
        
        return shoulder_info
    
    def predict_next_bbox(self, keypoints: np.ndarray, scores: np.ndarray) -> List[float]:
        """어깨 움직임을 고려한 바운딩박스 예측"""
        # 어깨 움직임 분석
        shoulder_info = self.analyze_shoulder_movement(keypoints, scores)
        
        if shoulder_info['left_shoulder'] is None:
            return self._predict_bbox_from_history()
        
        # 기본 바운딩박스 계산
        base_bbox = super().predict_next_bbox(keypoints, scores)
        
        if self.adaptive_bbox_policy == "fixed":
            return base_bbox
        
        elif self.adaptive_bbox_policy == "shoulder_aware":
            # 어깨 움직임 정도 평가
            movement_magnitude = shoulder_info['movement_magnitude']
            elevation_change = shoulder_info['elevation_change']
            rotation_angle = shoulder_info['rotation_angle']
            
            # 수화 의미있는 움직임 판단
            is_significant_movement = (
                abs(elevation_change) > self.shoulder_movement_threshold or
                abs(rotation_angle) > 5.0 or
                movement_magnitude > 15.0
            )
            
            if is_significant_movement:
                print(f"🤟 ID {self.track_id}: 의미있는 어깨 움직임 감지")
                print(f"   - 높이 변화: {elevation_change:.1f}px")
                print(f"   - 회전 각도: {rotation_angle:.1f}°")
                print(f"   - 움직임 크기: {movement_magnitude:.1f}")
                
                # 바운딩박스 적응적 조정
                x1, y1, x2, y2 = base_bbox
                center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
                width, height = x2 - x1, y2 - y1
                
                # 1. 높이 변화 반영
                if abs(elevation_change) > self.shoulder_movement_threshold:
                    height_adjustment = elevation_change * self.movement_amplification
                    center_y += height_adjustment * 0.3
                    height *= (1 + abs(elevation_change) / 100)
                
                # 2. 회전 반영
                if abs(rotation_angle) > 5.0:
                    rotation_expansion = abs(rotation_angle) / 45.0
                    width *= (1 + rotation_expansion * 0.2)
                    height *= (1 + rotation_expansion * 0.1)
                
                # 3. 비대칭 반영
                asymmetry = shoulder_info['asymmetry_level']
                if asymmetry > 10:
                    width *= (1 + asymmetry / 200)
                
                # 조정된 바운딩박스 계산
                adjusted_bbox = [
                    max(0, center_x - width / 2),
                    max(0, center_y - height / 2),
                    center_x + width / 2,
                    center_y + height / 2
                ]
                
                return adjusted_bbox
            
            return base_bbox
        
        return base_bbox
    
    def get_shoulder_movement_features(self) -> Dict[str, Any]:
        """어깨 움직임 특징 추출 (수화 인식 모델용)"""
        if len(self.shoulder_patterns['elevation']) < 3:
            return {}
        
        features = {}
        
        # 시간적 패턴 분석
        for pattern_name, pattern_data in self.shoulder_patterns.items():
            if len(pattern_data) > 0:
                pattern_array = np.array(pattern_data)
                features[f"{pattern_name}_mean"] = np.mean(pattern_array)
                features[f"{pattern_name}_std"] = np.std(pattern_array)
                features[f"{pattern_name}_trend"] = np.polyfit(range(len(pattern_array)), pattern_array, 1)[0]
                features[f"{pattern_name}_range"] = np.max(pattern_array) - np.min(pattern_array)
        
        # 어깨 움직임 강도 레벨
        recent_movements = list(self.shoulder_patterns['elevation'])[-5:]
        if recent_movements:
            movement_intensity = np.std(recent_movements)
            if movement_intensity > 20:
                features['movement_level'] = 'high'
            elif movement_intensity > 10:
                features['movement_level'] = 'medium'
            else:
                features['movement_level'] = 'low'
        
        return features


# ============================================================================
# 4. 메인 하이브리드 추론기 클래스
# ============================================================================

class TrackingYOLO11LHybridInferencer:
    """트래킹 기반 YOLO11L + RTMW 하이브리드 추론기"""
    
    def __init__(self, 
                 rtmw_config: str, 
                 rtmw_checkpoint: str,
                 detection_device: str = "auto",
                 pose_device: str = "auto",
                 yolo_model_name: str = "yolo11l.pt",
                 use_shoulder_tracking: bool = True):
        
        # 모델 경로 설정
        self.rtmw_config = rtmw_config
        self.rtmw_checkpoint = rtmw_checkpoint
        self.yolo_model_name = yolo_model_name
        
        # XPU 지원 확인
        self.xpu_available = check_xpu_availability()
        
        # 디바이스 설정
        self.detection_device = self._determine_device(detection_device, "검출")
        self.pose_device = self._determine_device(pose_device, "포즈")
        
        # 트래킹 설정
        self.use_shoulder_tracking = use_shoulder_tracking
        self.trackers: Dict[int, PersonTracker] = {}
        self.next_track_id = 0
        self.frame_count = 0
        self.last_detection_frame = -1
        self.redetection_interval = 30  # 30프레임마다 재검출
        
        # 성능 모니터링
        self.inference_times = {
            'detection': [],
            'pose': [],
            'tracking': [],
            'total': []
        }
        
        # 모델 초기화
        self._init_detection_model()
        self._init_pose_model()
        self._setup_sign_language_optimization()
        
        print("🎯 트래킹 기반 YOLO11L + RTMW 하이브리드 시스템 준비 완료")
        print(f"   - 검출 디바이스: {self.detection_device}")
        print(f"   - 포즈 디바이스: {self.pose_device}")
        print(f"   - 어깨 트래킹: {'활성화' if use_shoulder_tracking else '비활성화'}")
    
    def _determine_device(self, device: str, task_name: str) -> str:
        """디바이스 자동 결정"""
        if device == "auto":
            if self.xpu_available:
                return "xpu"
            elif torch.cuda.is_available():
                return "cuda"
            else:
                return "cpu"
        else:
            if device == "xpu" and not self.xpu_available:
                print(f"⚠️ {task_name} XPU 미사용 가능 - CPU로 폴백")
                return "cpu"
            elif device == "cuda" and not torch.cuda.is_available():
                print(f"⚠️ {task_name} CUDA 미사용 가능 - CPU로 폴백")
                return "cpu"
            return device
    
    def _setup_sign_language_optimization(self):
        """수화 인식 최적화 설정"""
        print("🤟 수화 인식 최적화 설정 적용 중...")
        
        # YOLO11L 수화 검출 최적화 파라미터
        self.yolo_conf_thresh = 0.3
        self.yolo_iou_thresh = 0.6
        self.yolo_max_det = 10
        self.yolo_classes = [0]  # 사람 클래스만
        self.detection_img_size = 832
        
        # 트래킹 최적화
        self.tracking_smoothing = 0.8
        self.bbox_stability = 0.9
        self.pose_consistency = 0.85
        
        # 수화 특화 설정
        self.hand_region_expansion = 1.5
        self.upper_body_focus = True
        self.temporal_smoothing = True
        
        print("✅ 수화 인식 최적화 설정 완료")
    
    def _init_detection_model(self):
        """YOLO11L 검출 모델 초기화"""
        print(f"🔧 YOLO11L 검출 모델 로딩 중... (디바이스: {self.detection_device})")
        start_time = time.time()
        
        try:
            model_path = os.path.join("../models", self.yolo_model_name)
            if not os.path.exists(model_path):
                print(f"📥 YOLO11L 모델 다운로드 중: {self.yolo_model_name}")
                
            self.detection_model = YOLO(model_path)
            
            # 디바이스 설정
            if self.detection_device != "cpu":
                try:
                    self.detection_model.to(self.detection_device)
                    print(f"✅ YOLO11L {self.detection_device.upper()} 모드 활성화")
                except Exception as e:
                    print(f"⚠️ YOLO11L {self.detection_device.upper()} 실패, CPU로 폴백: {e}")
                    self.detection_device = "cpu"
                    self.detection_model.to('cpu')
            
            init_time = time.time() - start_time
            print(f"✅ YOLO11L 검출 모델 로딩 완료: {init_time:.2f}초")
            
        except Exception as e:
            print(f"❌ YOLO11L 모델 로딩 실패: {e}")
            self.detection_model = None
    
    def _init_pose_model(self):
        """RTMW 포즈 추정 모델 초기화"""
        print(f"🔧 RTMW 포즈 모델 로딩 중... (디바이스: {self.pose_device})")
        start_time = time.time()
        
        try:
            self.pose_model = init_model(
                config=self.rtmw_config,
                checkpoint=self.rtmw_checkpoint,
                device=self.pose_device
            )
            
            init_time = time.time() - start_time
            print(f"✅ RTMW 포즈 모델 로딩 완료: {init_time:.2f}초")
            
        except Exception as e:
            print(f"❌ {self.pose_device} 포즈 모델 실패: {e}")
            print(f"🔄 CPU로 폴백...")
            self.pose_device = 'cpu'
            
            self.pose_model = init_model(
                config=self.rtmw_config,
                checkpoint=self.rtmw_checkpoint,
                device='cpu'
            )
            
            init_time = time.time() - start_time
            print(f"✅ CPU 포즈 모델 로딩 완료: {init_time:.2f}초")
    
    def detect_persons_full(self, image: np.ndarray) -> List[List[float]]:
        """전체 이미지에서 사람 검출 (YOLO11L)"""
        if self.detection_model is None:
            return self._simple_person_detection(image)
        
        try:
            start_time = time.time()
            
            # YOLO11L 추론
            results = self.detection_model(
                image,
                conf=self.yolo_conf_thresh,
                iou=self.yolo_iou_thresh,
                max_det=self.yolo_max_det,
                classes=self.yolo_classes,
                verbose=False,
                imgsz=self.detection_img_size
            )
            
            detection_time = time.time() - start_time
            self.inference_times['detection'].append(detection_time)
            
            # 사람 검출 결과 추출
            person_boxes = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None and len(boxes) > 0:
                    person_coords = boxes.xyxy
                    person_confs = boxes.conf
                    
                    # 신뢰도 필터링
                    conf_mask = person_confs >= self.yolo_conf_thresh
                    if conf_mask.any():
                        filtered_boxes = person_coords[conf_mask]
                        filtered_confs = person_confs[conf_mask]
                        
                        # numpy 변환
                        if isinstance(filtered_boxes, torch.Tensor):
                            filtered_boxes = filtered_boxes.cpu().numpy()
                            filtered_confs = filtered_confs.cpu().numpy()
                        
                        # 신뢰도순 정렬
                        conf_indices = np.argsort(filtered_confs)[::-1]
                        
                        for idx in conf_indices:
                            bbox = filtered_boxes[idx].tolist()
                            person_boxes.append(bbox)
            
            print(f"🔍 YOLO11L 검출: {len(person_boxes)}명 발견 ({detection_time:.3f}초)")
            return person_boxes
            
        except Exception as e:
            print(f"❌ YOLO11L 검출 실패: {e}")
            return self._simple_person_detection(image)
    
    def _simple_person_detection(self, image: np.ndarray) -> List[List[float]]:
        """간단한 사람 검출 (폴백)"""
        h, w = image.shape[:2]
        center_box = [w*0.25, h*0.1, w*0.75, h*0.9]
        return [center_box]
    
    def estimate_pose(self, image: np.ndarray, bbox: List[float]) -> Tuple[np.ndarray, np.ndarray]:
        """포즈 추정"""
        start_time = time.time()
        
        try:
            # 바운딩박스 전처리
            x1, y1, x2, y2 = map(int, bbox)
            x1, y1 = max(0, x1), max(0, y1)
            x2 = min(image.shape[1], x2)
            y2 = min(image.shape[0], y2)
            
            if x2 <= x1 or y2 <= y1:
                return np.zeros((133, 2)), np.zeros(133)
            
            # RTMW 포즈 추정
            pose_results = inference_topdown(self.pose_model, image, [[x1, y1, x2, y2]])
            
            pose_time = time.time() - start_time
            self.inference_times['pose'].append(pose_time)
            
            if pose_results and len(pose_results) > 0:
                pred_instances = pose_results[0].pred_instances
                keypoints = pred_instances.keypoints[0]
                scores = pred_instances.keypoint_scores[0]
                
                return keypoints.cpu().numpy(), scores.cpu().numpy()
            else:
                return np.zeros((133, 2)), np.zeros(133)
                
        except Exception as e:
            print(f"❌ 포즈 추정 실패: {e}")
            return np.zeros((133, 2)), np.zeros(133)
    
    def update_trackers(self, image: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray, List[float], int]]:
        """트래커 업데이트"""
        start_time = time.time()
        results = []
        
        for track_id, tracker in list(self.trackers.items()):
            if not tracker.is_active:
                del self.trackers[track_id]
                continue
            
            # 현재 바운딩박스에서 포즈 추정
            keypoints, scores = self.estimate_pose(image, tracker.current_bbox)
            
            # 유효한 포즈가 검출된 경우
            if np.any(scores > 0.3):
                # 트래커 업데이트
                updated_bbox = tracker.update(keypoints, scores, image.shape)
                results.append((keypoints, scores, updated_bbox, track_id))
            else:
                # 포즈 검출 실패 - 트래커를 놓침 처리
                tracker.mark_missing()
                if tracker.is_active:
                    # 마지막 알려진 바운딩박스 사용
                    keypoints, scores = np.zeros((133, 2)), np.zeros(133)
                    results.append((keypoints, scores, tracker.current_bbox, track_id))
        
        tracking_time = time.time() - start_time
        self.inference_times['tracking'].append(tracking_time)
        
        return results
    
    def initialize_trackers(self, image: np.ndarray, person_boxes: List[List[float]]) -> List[Tuple[np.ndarray, np.ndarray, List[float], int]]:
        """트래커 초기화"""
        results = []
        
        for bbox in person_boxes:
            # 트래커 생성
            if self.use_shoulder_tracking:
                tracker = SignLanguageShoulderTracker(
                    track_id=self.next_track_id,
                    initial_bbox=bbox,
                    stability_factor=self.tracking_smoothing
                )
            else:
                tracker = PersonTracker(
                    track_id=self.next_track_id,
                    initial_bbox=bbox,
                    stability_factor=self.tracking_smoothing
                )
            
            # 초기 포즈 추정
            keypoints, scores = self.estimate_pose(image, bbox)
            
            # 트래커 등록
            self.trackers[self.next_track_id] = tracker
            results.append((keypoints, scores, bbox, self.next_track_id))
            
            self.next_track_id += 1
        
        return results
    
    def process_frame(self, image: np.ndarray) -> Tuple[np.ndarray, List[Tuple[np.ndarray, np.ndarray, List[float], int]]]:
        """프레임 처리 - 트래킹 기반"""
        start_time = time.time()
        
        # 검출 필요 여부 판단
        need_detection = (
            self.frame_count == 0 or
            len(self.trackers) == 0 or
            (self.frame_count - self.last_detection_frame) >= self.redetection_interval or
            all(not t.is_active for t in self.trackers.values())
        )
        
        if need_detection:
            print(f"🔍 프레임 {self.frame_count}: 전체 검출 수행")
            person_boxes = self.detect_persons_full(image)
            results = self.initialize_trackers(image, person_boxes)
            self.last_detection_frame = self.frame_count
        else:
            results = self.update_trackers(image)
        
        self.frame_count += 1
        
        total_time = time.time() - start_time
        self.inference_times['total'].append(total_time)
        
        # 시각화
        vis_image = self.visualize_tracking_results(image, results)
        
        return vis_image, results
    
    def visualize_tracking_results(self, image: np.ndarray, results: List[Tuple[np.ndarray, np.ndarray, List[float], int]]) -> np.ndarray:
        """트래킹 결과 시각화"""
        vis_image = image.copy()
        
        # 트래커별 색상
        colors = [
            (0, 255, 0),    # 초록
            (255, 0, 0),    # 파랑  
            (0, 0, 255),    # 빨강
            (255, 255, 0),  # 시안
            (255, 0, 255),  # 마젠타
        ]
        
        for i, (keypoints, scores, bbox, track_id) in enumerate(results):
            color = colors[track_id % len(colors)]
            
            # 바운딩박스 그리기
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
            
            # 트래커 ID 표시
            cv2.putText(vis_image, f"ID:{track_id}", (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # 수화 인식 중요 키포인트 강조
            self._draw_sign_language_keypoints(vis_image, keypoints, scores, color)
        
        # 성능 정보 표시
        if self.inference_times['total']:
            fps = 1.0 / self.inference_times['total'][-1]
            cv2.putText(vis_image, f"FPS: {fps:.1f}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 시스템 정보
        cv2.putText(vis_image, "Tracking YOLO11L + RTMW (Sign Language)", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(vis_image, f"Frame: {self.frame_count}", 
                   (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(vis_image, f"Trackers: {len(self.trackers)}", 
                   (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return vis_image
    
    def _draw_sign_language_keypoints(self, image: np.ndarray, keypoints: np.ndarray, scores: np.ndarray, base_color: Tuple[int, int, int]):
        """수화 인식용 키포인트 시각화"""
        # 손 키포인트 (COCO-WholeBody)
        left_hand_indices = list(range(91, 112))   # 왼손
        right_hand_indices = list(range(112, 133)) # 오른손
        face_indices = list(range(17, 91))         # 얼굴
        body_indices = list(range(0, 17))          # 몸통
        
        # 손 키포인트 강조 (더 크게)
        for indices, hand_name in [(left_hand_indices, "L"), (right_hand_indices, "R")]:
            for idx in indices:
                if idx < len(keypoints) and scores[idx] > 0.3:
                    x, y = int(keypoints[idx][0]), int(keypoints[idx][1])
                    if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                        cv2.circle(image, (x, y), 4, base_color, -1)
                        if scores[idx] > 0.8:
                            cv2.circle(image, (x, y), 6, (255, 255, 255), 1)
        
        # 얼굴 키포인트 (중간 크기)
        for idx in face_indices:
            if idx < len(keypoints) and scores[idx] > 0.5:
                x, y = int(keypoints[idx][0]), int(keypoints[idx][1])
                if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                    cv2.circle(image, (x, y), 2, base_color, -1)
        
        # 몸통 키포인트 (기본 크기)
        for idx in body_indices:
            if idx < len(keypoints) and scores[idx] > 0.3:
                x, y = int(keypoints[idx][0]), int(keypoints[idx][1])
                if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                    cv2.circle(image, (x, y), 3, base_color, -1)
    
    def reset_tracking(self):
        """트래킹 리셋"""
        print("🔄 트래킹 리셋")
        self.trackers.clear()
        self.frame_count = 0
        self.last_detection_frame = -1
        self.next_track_id = 0
    
    def get_performance_stats(self) -> Dict[str, float]:
        """성능 통계 반환"""
        stats = {}
        
        for key, times in self.inference_times.items():
            if times:
                stats[f"{key}_avg"] = np.mean(times)
                stats[f"{key}_fps"] = 1.0 / np.mean(times) if np.mean(times) > 0 else 0
        
        return stats


# ============================================================================
# 5. 테스트 및 데모 함수들
# ============================================================================

def test_video_tracking(video_path: str, output_path: str = None):
    """비디오 트래킹 테스트"""
    
    # 하이브리드 추론기 초기화
    inferencer = TrackingYOLO11LHybridInferencer(
        rtmw_config="../configs/wholebody_2d_keypoint/rtmw/coco-wholebody/rtmw-l_8xb64-270e_coco-wholebody-384x288.py",
        rtmw_checkpoint="../models/rtmw-x_simcc-cocktail14_pt-ucoco_270e-384x288-f840f204_20231122.pth",
        detection_device="auto",
        pose_device="auto",
        use_shoulder_tracking=True
    )
    
    print(f"\n=== 트래킹 기반 비디오 테스트: {os.path.basename(video_path)} ===")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ 비디오 파일 열기 실패: {video_path}")
        return
    
    # 비디오 정보
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"📷 비디오 정보: {width}x{height}, {fps}fps, {total_frames}프레임")
    
    # 출력 비디오 설정
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_idx = 0
    start_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 프레임 처리
            vis_frame, results = inferencer.process_frame(frame)
            
            # 결과 출력
            if frame_idx % 30 == 0:  # 30프레임마다 출력
                print(f"🎬 프레임 {frame_idx}/{total_frames} 처리 완료")
                if results:
                    print(f"   - 검출된 사람: {len(results)}명")
                    for keypoints, scores, bbox, track_id in results:
                        valid_keypoints = np.sum(scores > 0.3)
                        print(f"   - ID {track_id}: {valid_keypoints}개 키포인트")
            
            # 출력 비디오 저장
            if output_path:
                out.write(vis_frame)
            
            # 실시간 표시 (옵션)
            cv2.imshow('Tracking YOLO11L + RTMW', vis_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            frame_idx += 1
    
    finally:
        cap.release()
        if output_path:
            out.release()
        cv2.destroyAllWindows()
        
        # 성능 통계
        total_time = time.time() - start_time
        avg_fps = frame_idx / total_time
        
        print(f"\n📊 성능 통계:")
        print(f"   - 총 처리 시간: {total_time:.2f}초")
        print(f"   - 평균 FPS: {avg_fps:.2f}")
        print(f"   - 처리된 프레임: {frame_idx}/{total_frames}")
        
        # 상세 성능 통계
        stats = inferencer.get_performance_stats()
        for key, value in stats.items():
            print(f"   - {key}: {value:.3f}")


def test_webcam_tracking():
    """웹캠 실시간 트래킹 테스트"""
    
    # 하이브리드 추론기 초기화
    inferencer = TrackingYOLO11LHybridInferencer(
        rtmw_config="../configs/wholebody_2d_keypoint/rtmw/coco-wholebody/rtmw-l_8xb64-270e_coco-wholebody-384x288.py",
        rtmw_checkpoint="../models/rtmw-x_simcc-cocktail14_pt-ucoco_270e-384x288-f840f204_20231122.pth",
        detection_device="auto",
        pose_device="auto",
        use_shoulder_tracking=True
    )
    
    print("\n=== 웹캠 실시간 트래킹 테스트 ===")
    print("ESC 키를 눌러 종료")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ 웹캠 연결 실패")
        return
    
    # 웹캠 설정
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 프레임 처리
            vis_frame, results = inferencer.process_frame(frame)
            
            # 실시간 표시
            cv2.imshow('Real-time Tracking YOLO11L + RTMW', vis_frame)
            
            # 키 입력 처리
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC 키
                break
            elif key == ord('r'):  # 'r' 키로 트래킹 리셋
                inferencer.reset_tracking()
                print("🔄 트래킹 리셋됨")
            
            frame_count += 1
            
            # 30프레임마다 성능 출력
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                current_fps = frame_count / elapsed
                print(f"🎥 프레임 {frame_count}: FPS {current_fps:.1f}")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        
        # 최종 성능 통계
        total_time = time.time() - start_time
        avg_fps = frame_count / total_time
        
        print(f"\n📊 웹캠 테스트 결과:")
        print(f"   - 총 처리 시간: {total_time:.2f}초")
        print(f"   - 평균 FPS: {avg_fps:.2f}")
        print(f"   - 처리된 프레임: {frame_count}")


# ============================================================================
# 6. 메인 실행부
# ============================================================================

if __name__ == "__main__":
    print("🤟 트래킹 기반 YOLO11L + RTMW 수화 인식 시스템")
    print("=" * 60)
    
    # 사용법 안내
    print("사용 가능한 테스트:")
    print("1. 비디오 파일 테스트:")
    print("   test_video_tracking('video_path.mp4', 'output_path.mp4')")
    print("2. 웹캠 실시간 테스트:")
    print("   test_webcam_tracking()")
    print()
    
    # 예시 실행
    # test_video_tracking("winter01.jpg", "tracking_output.mp4")
    # test_webcam_tracking()
