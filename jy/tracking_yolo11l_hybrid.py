#!/usr/bin/env python3
"""
트래킹 기반 YOLO11L + RTMW 하이브리드 추론기
수화 인식을 위한 안정적인 포즈 추정 시스템

특징:
- 첫 프레임에서만 YOLO11L 검출
- 이후 프레임에서는 포즈 기반 바운딩 박스 예측
- 안정적인 바운딩 박스 크기 유지
- 수화 동작 인식에 최적화
"""

import os
import torch
import cv2
import numpy as np
import time
from typing import List, Tuple, Optional, Dict, Any
from collections import deque
from mmpose.apis import init_model, inference_topdown

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    print("⚠️ ultralytics 미설치 - pip install ultralytics")
    YOLO_AVAILABLE = False

def check_xpu_availability():
    """XPU 가용성 확인"""
    try:
        if torch.xpu.is_available():
            device_count = torch.xpu.device_count()
            print(f"✅ Intel XPU 사용 가능: {device_count}개 디바이스")
            return True
        else:
            print("⚠️ Intel XPU 사용 불가 - CPU 모드로 실행")
            return False
    except Exception as e:
        print(f"⚠️ XPU 확인 실패: {e} - CPU 모드로 실행")
        return False

class PersonTracker:
    """사람 트래킹 클래스 - 포즈 기반 바운딩박스 예측"""
    
    def __init__(self, track_id: int, initial_bbox: List[float], stability_factor: float = 0.3):
        self.track_id = track_id
        self.current_bbox = initial_bbox  # [x1, y1, x2, y2]
        self.bbox_history = deque(maxlen=10)  # 최근 10프레임 히스토리
        self.pose_history = deque(maxlen=5)   # 최근 5프레임 포즈 히스토리
        self.stability_factor = stability_factor
        self.missing_frames = 0
        self.max_missing_frames = 10
        self.is_active = True
        
        # 바운딩박스 안정화를 위한 파라미터
        self.bbox_smoothing = 0.7  # 이전 프레임 가중치
        self.size_consistency = 0.8  # 크기 일관성 유지
        
        # 수화 인식을 위한 특별 설정
        self.hand_focus_margin = 1.4  # 손 영역 확장 비율
        self.upper_body_focus = True  # 상반신 중심
        
        self.bbox_history.append(initial_bbox)
    
    def predict_next_bbox(self, current_pose: np.ndarray, current_scores: np.ndarray) -> List[float]:
        """포즈 기반 다음 바운딩박스 예측"""
        if current_pose is None or len(current_pose) == 0:
            return self._predict_bbox_from_history()
        
        # 유효한 키포인트 필터링 (신뢰도 > 0.3)
        valid_mask = current_scores > 0.3
        if not np.any(valid_mask):
            return self._predict_bbox_from_history()
        
        valid_keypoints = current_pose[valid_mask]
        
        # 수화 인식을 위한 상반신 중심 바운딩박스
        if self.upper_body_focus:
            # 상반신 키포인트 인덱스 (COCO-WholeBody 기준)
            upper_body_indices = list(range(0, 17)) + list(range(91, 133))  # 몸통 + 손
            face_indices = list(range(17, 91))  # 얼굴
            
            # 상반신 키포인트만 사용
            upper_body_mask = np.zeros(len(current_scores), dtype=bool)
            upper_body_mask[upper_body_indices] = True
            upper_body_mask = upper_body_mask & valid_mask
            
            if np.any(upper_body_mask):
                relevant_keypoints = current_pose[upper_body_mask]
            else:
                relevant_keypoints = valid_keypoints
        else:
            relevant_keypoints = valid_keypoints
        
        # 키포인트 기반 바운딩박스 계산
        x_coords = relevant_keypoints[:, 0]
        y_coords = relevant_keypoints[:, 1]
        
        min_x, max_x = np.min(x_coords), np.max(x_coords)
        min_y, max_y = np.min(y_coords), np.max(y_coords)
        
        # 수화를 위한 손 영역 확장
        width = max_x - min_x
        height = max_y - min_y
        
        # 확장 마진 적용
        margin_x = width * (self.hand_focus_margin - 1) / 2
        margin_y = height * (self.hand_focus_margin - 1) / 2
        
        predicted_bbox = [
            max(0, min_x - margin_x),
            max(0, min_y - margin_y),
            max_x + margin_x,
            max_y + margin_y
        ]
        
        # 이전 프레임과의 부드러운 전환 (안정화)
        if self.bbox_history:
            prev_bbox = self.bbox_history[-1]
            smoothed_bbox = [
                prev_bbox[i] * self.bbox_smoothing + predicted_bbox[i] * (1 - self.bbox_smoothing)
                for i in range(4)
            ]
            
            # 크기 일관성 유지
            prev_width = prev_bbox[2] - prev_bbox[0]
            prev_height = prev_bbox[3] - prev_bbox[1]
            current_width = smoothed_bbox[2] - smoothed_bbox[0]
            current_height = smoothed_bbox[3] - smoothed_bbox[1]
            
            # 크기 변화 제한
            width_ratio = current_width / prev_width if prev_width > 0 else 1.0
            height_ratio = current_height / prev_height if prev_height > 0 else 1.0
            
            if width_ratio > 1.3 or width_ratio < 0.7:  # 30% 이상 변화 제한
                center_x = (smoothed_bbox[0] + smoothed_bbox[2]) / 2
                smoothed_bbox[0] = center_x - prev_width * self.size_consistency / 2
                smoothed_bbox[2] = center_x + prev_width * self.size_consistency / 2
            
            if height_ratio > 1.3 or height_ratio < 0.7:
                center_y = (smoothed_bbox[1] + smoothed_bbox[3]) / 2
                smoothed_bbox[1] = center_y - prev_height * self.size_consistency / 2
                smoothed_bbox[3] = center_y + prev_height * self.size_consistency / 2
            
            predicted_bbox = smoothed_bbox
        
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

class TrackingYOLO11LHybridInferencer:
    """트래킹 기반 YOLO11L + RTMW 하이브리드 추론기"""
    
    def __init__(self, 
                 rtmw_config: str, 
                 rtmw_checkpoint: str,
                 detection_device: str = "auto",
                 pose_device: str = "auto",
                 redetection_interval: int = 30,
                 tracking_stability: float = 0.7):
        """
        Args:
            rtmw_config: RTMW 설정 파일 경로
            rtmw_checkpoint: RTMW 체크포인트 경로
            detection_device: 검출 디바이스
            pose_device: 포즈 추정 디바이스
            redetection_interval: 재검출 간격 (프레임)
            tracking_stability: 트래킹 안정성 (0-1)
        """
        if not YOLO_AVAILABLE:
            raise ImportError("ultralytics가 필요합니다: pip install ultralytics")
            
        self.rtmw_config = rtmw_config
        self.rtmw_checkpoint = rtmw_checkpoint
        self.yolo_model_name = "yolo11l.pt"
        self.redetection_interval = redetection_interval
        self.tracking_stability = tracking_stability
        
        # XPU 가용성 확인
        self.xpu_available = check_xpu_availability()
        
        # 디바이스 결정
        self.detection_device = self._determine_device(detection_device, "검출")
        self.pose_device = self._determine_device(pose_device, "포즈추정")
        
        print(f"🚀 트래킹 기반 YOLO11L + RTMW 하이브리드 추론기 초기화:")
        print(f"   - YOLO 모델: YOLO11L (Large - 고정확도)")
        print(f"   - 검출 디바이스: {self.detection_device}")
        print(f"   - 포즈 추정 디바이스: {self.pose_device}")
        print(f"   - 재검출 간격: {redetection_interval}프레임")
        print(f"   - 트래킹 안정성: {tracking_stability}")
        
        # PyTorch 보안 설정
        self.original_load = torch.load
        torch.load = lambda *args, **kwargs: self.original_load(*args, **kwargs, weights_only=False) if 'weights_only' not in kwargs else self.original_load(*args, **kwargs)
        
        # 모델 초기화
        self._init_detection_model()
        self._init_pose_model()
        
        # 트래킹 상태
        self.trackers: Dict[int, PersonTracker] = {}
        self.next_track_id = 0
        self.frame_count = 0
        self.last_detection_frame = -1
        
        # 성능 통계
        self.inference_times = {
            'detection': [],
            'pose': [],
            'tracking': [],
            'total': []
        }
        
        # 수화 인식 최적화 설정
        self._setup_sign_language_optimization()
        
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
        self.yolo_conf_thresh = 0.3     # 낮은 신뢰도로 놓치지 않게
        self.yolo_iou_thresh = 0.6      # 적당한 IoU
        self.yolo_max_det = 10          # 다중 사람 허용
        self.yolo_classes = [0]         # 사람 클래스만
        self.detection_img_size = 832   # Large 모델에 최적화
        
        # 트래킹 최적화
        self.tracking_smoothing = 0.8   # 높은 안정성
        self.bbox_stability = 0.9       # 바운딩박스 크기 안정성
        self.pose_consistency = 0.85    # 포즈 일관성
        
        # 수화 특화 설정
        self.hand_region_expansion = 1.5  # 손 영역 확장
        self.upper_body_focus = True      # 상반신 중심
        self.temporal_smoothing = True    # 시간적 부드러움
        
        print("✅ 수화 인식 최적화 설정 완료")
    
    def _init_detection_model(self):
        """YOLO11L 검출 모델 초기화"""
        print(f"🔧 YOLO11L 검출 모델 로딩 중... (디바이스: {self.detection_device})")
        start_time = time.time()
        
        try:
            # YOLO11L 모델 로드
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
                        sorted_indices = np.argsort(filtered_confs)[::-1]
                        sorted_boxes = filtered_boxes[sorted_indices]
                        
                        person_boxes.extend(sorted_boxes.tolist())
            
            print(f"🔍 YOLO11L 전체 검출: {len(person_boxes)}명, {detection_time:.3f}초")
            
            return person_boxes if person_boxes else self._simple_person_detection(image)
            
        except Exception as e:
            print(f"❌ YOLO11L 검출 실패: {e}")
            return self._simple_person_detection(image)
    
    def _simple_person_detection(self, image: np.ndarray) -> List[List[float]]:
        """간단한 사람 검출 (폴백)"""
        h, w = image.shape[:2]
        margin_w = int(w * 0.1)
        margin_h = int(h * 0.1)
        bbox = [margin_w, margin_h, w - margin_w, h - margin_h]
        return [bbox]
    
    def estimate_pose(self, image: np.ndarray, bbox: List[float]) -> Tuple[np.ndarray, np.ndarray]:
        """RTMW를 사용한 포즈 추정"""
        try:
            start_time = time.time()
            
            # MMPose 추론
            results = inference_topdown(
                model=self.pose_model,
                img=image,
                bboxes=[bbox],
                bbox_format='xyxy'
            )
            
            pose_time = time.time() - start_time
            self.inference_times['pose'].append(pose_time)
            
            if results and len(results) > 0:
                keypoints = results[0].pred_instances.keypoints[0]
                scores = results[0].pred_instances.keypoint_scores[0]
                
                if isinstance(keypoints, torch.Tensor):
                    keypoints = keypoints.cpu().numpy()
                if isinstance(scores, torch.Tensor):
                    scores = scores.cpu().numpy()
                    
                return keypoints, scores
            else:
                return np.zeros((133, 2)), np.zeros(133)
                
        except Exception as e:
            print(f"❌ 포즈 추정 실패: {e}")
            return np.zeros((133, 2)), np.zeros(133)
    
    def update_trackers(self, image: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray, List[float], int]]:
        """트래커 업데이트 및 포즈 추정"""
        start_time = time.time()
        
        results = []
        active_trackers = list(self.trackers.values())
        
        for tracker in active_trackers:
            if not tracker.is_active:
                continue
            
            # 트래킹 기반 바운딩박스 사용
            bbox = tracker.current_bbox
            
            # 포즈 추정
            keypoints, scores = self.estimate_pose(image, bbox)
            
            # 트래커 업데이트 (포즈 기반)
            updated_bbox = tracker.update(keypoints, scores, image.shape)
            
            results.append((keypoints, scores, updated_bbox, tracker.track_id))
        
        # 비활성 트래커 제거
        active_track_ids = [t.track_id for t in active_trackers if t.is_active]
        self.trackers = {tid: self.trackers[tid] for tid in active_track_ids}
        
        tracking_time = time.time() - start_time
        self.inference_times['tracking'].append(tracking_time)
        
        return results
    
    def initialize_trackers(self, image: np.ndarray, person_boxes: List[List[float]]) -> List[Tuple[np.ndarray, np.ndarray, List[float], int]]:
        """트래커 초기화"""
        print(f"🎯 {len(person_boxes)}개 트래커 초기화 중...")
        
        self.trackers.clear()
        results = []
        
        for bbox in person_boxes:
            # 새 트래커 생성
            tracker = PersonTracker(
                track_id=self.next_track_id,
                initial_bbox=bbox,
                stability_factor=self.tracking_stability
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
            self.frame_count == 0 or  # 첫 프레임
            len(self.trackers) == 0 or  # 트래커 없음
            (self.frame_count - self.last_detection_frame) >= self.redetection_interval or  # 주기적 재검출
            all(not t.is_active for t in self.trackers.values())  # 모든 트래커 비활성
        )
        
        if need_detection:
            print(f"🔍 프레임 {self.frame_count}: 전체 검출 수행")
            # 전체 검출
            person_boxes = self.detect_persons_full(image)
            results = self.initialize_trackers(image, person_boxes)
            self.last_detection_frame = self.frame_count
        else:
            # 트래킹 기반 처리
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
                        # 손 키포인트는 더 크게
                        cv2.circle(image, (x, y), 4, base_color, -1)
                        if scores[idx] > 0.8:  # 고신뢰도는 테두리 추가
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
    
    def test_video(self, video_path: str, output_path: str = None, max_frames: int = None):
        """비디오 테스트"""
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
        
        # 트래킹 리셋
        self.reset_tracking()
        
        frame_idx = 0
        process_times = []
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if max_frames and frame_idx >= max_frames:
                    break
                
                # 프레임 처리
                start_time = time.time()
                vis_frame, results = self.process_frame(frame)
                process_time = time.time() - start_time
                process_times.append(process_time)
                
                # 출력 저장
                if output_path:
                    out.write(vis_frame)
                
                # 진행률 표시
                if frame_idx % 30 == 0:
                    avg_fps = 1.0 / np.mean(process_times[-30:]) if process_times else 0
                    print(f"   프레임 {frame_idx}/{total_frames}: {avg_fps:.1f}fps, {len(results)}명 추적")
                
                frame_idx += 1
                
                # ESC 키로 중단
                if cv2.waitKey(1) & 0xFF == 27:
                    break
                    
        except KeyboardInterrupt:
            print("\n⏹️ 사용자가 비디오 처리를 중단했습니다.")
        
        finally:
            cap.release()
            if output_path:
                out.release()
            cv2.destroyAllWindows()
        
        # 성능 통계
        if process_times:
            avg_fps = 1.0 / np.mean(process_times)
            print(f"\n📊 비디오 처리 완료:")
            print(f"   - 처리된 프레임: {frame_idx}")
            print(f"   - 평균 FPS: {avg_fps:.1f}")
            print(f"   - 총 처리 시간: {sum(process_times):.1f}초")
            if output_path:
                print(f"   - 출력 저장: {output_path}")

def main():
    """메인 테스트 함수"""
    # 모델 경로 설정
    rtmw_config = "../configs/wholebody_2d_keypoint/rtmpose/cocktail14/rtmw-x_8xb320-270e_cocktail14-384x288.py"
    rtmw_checkpoint = "../models/rtmw-x_simcc-cocktail14_pt-ucoco_270e-384x288-f840f204_20231122.pth"
    
    try:
        print("🚀 트래킹 기반 YOLO11L + RTMW 수화 인식 시스템 테스트")
        print("=" * 70)
        
        # 트래킹 하이브리드 추론기 생성
        inferencer = TrackingYOLO11LHybridInferencer(
            rtmw_config=rtmw_config,
            rtmw_checkpoint=rtmw_checkpoint,
            detection_device="auto",
            pose_device="auto",
            redetection_interval=30,  # 30프레임마다 재검출
            tracking_stability=0.8    # 높은 안정성
        )
        
        # 단일 이미지 테스트
        test_image = "winter01.jpg"
        if os.path.exists(test_image):
            print(f"\n📷 단일 이미지 테스트: {test_image}")
            image = cv2.imread(test_image)
            vis_image, results = inferencer.process_frame(image)
            
            print(f"✅ 검출/추적된 사람 수: {len(results)}")
            for keypoints, scores, bbox, track_id in results:
                valid_kpts = np.sum(scores > 0.3)
                hand_kpts = np.sum(scores[91:133] > 0.5)  # 손 키포인트
                print(f"   추적 ID {track_id}: {valid_kpts}/133 키포인트, {hand_kpts}/42 손 키포인트")
            
            # 결과 저장
            output_path = f"tracking_yolo11l_result_{os.path.basename(test_image)}"
            cv2.imwrite(output_path, vis_image)
            print(f"💾 결과 저장: {output_path}")
        
        print(f"\n🏆 트래킹 기반 시스템 특징:")
        print(f"   🎯 안정적 추적: 포즈 기반 바운딩박스 예측")
        print(f"   🤟 수화 최적화: 손과 상반신 키포인트 강조")
        print(f"   ⚡ 고속 처리: 첫 프레임만 검출, 이후 트래킹")
        print(f"   📐 일관된 크기: 바운딩박스 크기 안정화")
        print(f"   🔄 주기적 재검출: {inferencer.redetection_interval}프레임마다 갱신")
        
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
