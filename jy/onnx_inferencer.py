#!/usr/bin/env python3
"""
YOLO11L + RTMW ONNX 하이브리드 추론기
YOLO11L로 사람 검출 + RTMW ONNX로 포즈 추정
"""

import os
import cv2
import numpy as np
import time
from typing import List, Tuple, Optional
from collections import deque
import onnxruntime as ort

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    print("⚠️ ultralytics 미설치 - pip install ultralytics")
    YOLO_AVAILABLE = False

class YOLO11LONNXRTMWHybridInferencer:
    """YOLO11L + RTMW ONNX 하이브리드 추론기"""
    
    def __init__(self, 
                 rtmw_onnx_path: str,
                 detection_device: str = "auto",
                 onnx_providers: List[str] = None,
                 input_size: Tuple[int, int] = (288, 384),  # (H, W)
                 optimize_for_accuracy: bool = True):
        """
        Args:
            rtmw_onnx_path: RTMW ONNX 모델 파일 경로
            detection_device: 검출 디바이스 ('auto', 'cpu', 'cuda')
            onnx_providers: ONNX Runtime 프로바이더 리스트
            input_size: RTMW 입력 크기 (H, W)
            optimize_for_accuracy: 정확도 최적화 여부
        """
        if not YOLO_AVAILABLE:
            raise ImportError("ultralytics가 필요합니다: pip install ultralytics")
        
        self.rtmw_onnx_path = rtmw_onnx_path
        self.yolo_model_name = "yolo11l.pt"
        self.input_size = input_size
        self.optimize_for_accuracy = optimize_for_accuracy
        
        # 디바이스 설정
        self.detection_device = self._determine_yolo_device(detection_device)
        
        # ONNX Runtime 프로바이더 설정
        if onnx_providers is None:
            self.onnx_providers = self._get_available_providers()
        else:
            self.onnx_providers = onnx_providers
        
        print(f"🚀 YOLO11L + RTMW ONNX 하이브리드 추론기 초기화:")
        print(f"   - YOLO 모델: YOLO11L (디바이스: {self.detection_device})")
        print(f"   - RTMW ONNX: {os.path.basename(rtmw_onnx_path)}")
        print(f"   - ONNX 프로바이더: {self.onnx_providers}")
        print(f"   - 입력 크기: {input_size}")
        print(f"   - 정확도 최적화: {'ON' if optimize_for_accuracy else 'OFF'}")
        
        # 모델 초기화
        self._init_detection_model()
        self._init_pose_onnx_model()
        
        # 성능 통계
        self.inference_times = {
            'detection': [],
            'pose': [],
            'total': []
        }
        
        # 최적화 설정
        self._setup_optimization()
        
        # RTMW 키포인트 정보
        self._setup_keypoint_info()
    
    def _determine_yolo_device(self, device: str) -> str:
        """YOLO 디바이스 결정"""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            else:
                return "cpu"
        return device
    
    def _get_available_providers(self) -> List[str]:
        """사용 가능한 ONNX Runtime 프로바이더 확인"""
        available = ort.get_available_providers()
        
        # 우선순위: CUDA > CPU
        preferred_order = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        providers = []
        
        for provider in preferred_order:
            if provider in available:
                providers.append(provider)
        
        print(f"🔍 사용 가능한 ONNX 프로바이더: {available}")
        print(f"🎯 선택된 프로바이더: {providers}")
        
        return providers
    
    def _setup_optimization(self):
        """최적화 설정"""
        if self.optimize_for_accuracy:
            # YOLO11L 정확도 우선 파라미터
            self.yolo_conf_thresh = 0.4
            self.yolo_iou_thresh = 0.6
            self.yolo_max_det = 50
            self.yolo_classes = [0]  # 사람만
            self.detection_img_size = 832
        else:
            # 균형 설정
            self.yolo_conf_thresh = 0.5
            self.yolo_iou_thresh = 0.7
            self.yolo_max_det = 100
            self.yolo_classes = None
            self.detection_img_size = 640
    
    def _setup_keypoint_info(self):
        """RTMW 키포인트 정보 설정"""
        # RTMW는 133개 키포인트 (전신 + 얼굴 + 손)
        self.num_keypoints = 133
        
        # 주요 신체 부위 키포인트 인덱스 (COCO 기반)
        self.body_keypoints = list(range(17))  # 0-16: 신체
        self.face_keypoints = list(range(17, 17+68))  # 17-84: 얼굴
        self.left_hand_keypoints = list(range(17+68, 17+68+21))  # 85-105: 왼손
        self.right_hand_keypoints = list(range(17+68+21, 133))  # 106-126: 오른손
        
        # 신체 연결 정보 (그리기용)
        self.body_connections = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # 머리
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # 팔
            (5, 11), (6, 12), (11, 12),  # 몸통
            (11, 13), (13, 15), (12, 14), (14, 16)  # 다리
        ]
    
    def _init_detection_model(self):
        """YOLO11L 검출 모델 초기화"""
        print(f"🔧 YOLO11L 검출 모델 로딩 중...")
        try:
            self.detection_model = YOLO(self.yolo_model_name)
            self.detection_model.to(self.detection_device)
            print(f"✅ YOLO11L 로딩 완료")
        except Exception as e:
            print(f"❌ YOLO11L 로딩 실패: {e}")
            raise
    
    def _init_pose_onnx_model(self):
        """RTMW ONNX 모델 초기화"""
        print(f"🔧 RTMW ONNX 모델 로딩 중...")
        try:
            # ONNX Runtime 세션 옵션
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options.intra_op_num_threads = 0  # 자동 설정
            sess_options.inter_op_num_threads = 0  # 자동 설정
            
            # ONNX Runtime 세션 생성
            self.pose_session = ort.InferenceSession(
                self.rtmw_onnx_path, 
                sess_options=sess_options,
                providers=self.onnx_providers
            )
            
            # 입력/출력 정보 확인
            self.pose_input_name = self.pose_session.get_inputs()[0].name
            self.pose_output_names = [output.name for output in self.pose_session.get_outputs()]
            
            input_shape = self.pose_session.get_inputs()[0].shape
            output_shapes = [output.shape for output in self.pose_session.get_outputs()]
            
            print(f"✅ RTMW ONNX 로딩 완료")
            print(f"   - 입력: {self.pose_input_name} {input_shape}")
            print(f"   - 출력: {len(self.pose_output_names)}개 {output_shapes}")
            print(f"   - 프로바이더: {self.pose_session.get_providers()}")
            
        except Exception as e:
            print(f"❌ RTMW ONNX 로딩 실패: {e}")
            raise
    
    def detect_persons_high_accuracy(self, image: np.ndarray) -> List[List[float]]:
        """고정확도 사람 검출 (YOLO11L)"""
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
            
            # 검출 결과 추출
            person_boxes = []
            for result in results:
                boxes = result.boxes
                if boxes is not None and len(boxes) > 0:
                    person_coords = boxes.xyxy.cpu().numpy()
                    person_confs = boxes.conf.cpu().numpy()
                    
                    # 신뢰도 필터링
                    conf_mask = person_confs >= self.yolo_conf_thresh
                    if conf_mask.any():
                        filtered_boxes = person_coords[conf_mask]
                        filtered_confs = person_confs[conf_mask]
                        
                        # 신뢰도순 정렬
                        sorted_indices = np.argsort(filtered_confs)[::-1]
                        sorted_boxes = filtered_boxes[sorted_indices]
                        
                        person_boxes.extend(sorted_boxes.tolist())
            
            return person_boxes if person_boxes else []
            
        except Exception as e:
            print(f"❌ 사람 검출 실패: {e}")
            return []
    
    def preprocess_for_pose(self, image: np.ndarray, bbox: List[float]) -> np.ndarray:
        """포즈 추정을 위한 이미지 전처리"""
        try:
            x1, y1, x2, y2 = map(int, bbox)
            
            # 바운딩박스 확장 (여백 추가)
            img_h, img_w = image.shape[:2]
            margin_x = int((x2 - x1) * 0.1)
            margin_y = int((y2 - y1) * 0.1)
            
            x1 = max(0, x1 - margin_x)
            y1 = max(0, y1 - margin_y)
            x2 = min(img_w, x2 + margin_x)
            y2 = min(img_h, y2 + margin_y)
            
            # 크롭
            crop_img = image[y1:y2, x1:x2]
            
            # 리사이즈
            target_h, target_w = self.input_size
            resized_img = cv2.resize(crop_img, (target_w, target_h))
            
            # 정규화
            normalized_img = resized_img.astype(np.float32) / 255.0
            
            # 평균/표준편차 정규화 (ImageNet 기준)
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            normalized_img = (normalized_img - mean) / std
            
            # BGR to RGB
            normalized_img = normalized_img[:, :, ::-1]
            
            # 배치 차원 추가 및 채널 순서 변경 (NCHW)
            input_tensor = normalized_img.transpose(2, 0, 1)[np.newaxis, ...]
            
            return input_tensor, (x1, y1, x2, y2), (crop_img.shape[1], crop_img.shape[0])
            
        except Exception as e:
            print(f"❌ 전처리 실패: {e}")
            return None, None, None
    
    def estimate_pose_onnx(self, image: np.ndarray, bbox: List[float]) -> Tuple[np.ndarray, np.ndarray]:
        """RTMW ONNX를 사용한 포즈 추정"""
        try:
            start_time = time.time()
            
            # 전처리
            input_tensor, crop_bbox, crop_size = self.preprocess_for_pose(image, bbox)
            if input_tensor is None:
                return np.zeros((self.num_keypoints, 2)), np.zeros(self.num_keypoints)
            
            # ONNX 추론
            ort_inputs = {self.pose_input_name: input_tensor}
            ort_outputs = self.pose_session.run(self.pose_output_names, ort_inputs)
            
            # 후처리
            keypoints, scores = self.postprocess_pose_output(
                ort_outputs, crop_bbox, crop_size, self.input_size
            )
            
            pose_time = time.time() - start_time
            self.inference_times['pose'].append(pose_time)
            
            return keypoints, scores
            
        except Exception as e:
            print(f"❌ ONNX 포즈 추정 실패: {e}")
            return np.zeros((self.num_keypoints, 2)), np.zeros(self.num_keypoints)
    
    def postprocess_pose_output(self, ort_outputs, crop_bbox, crop_size, input_size):
        """포즈 추정 결과 후처리"""
        try:
            # RTMW 출력 형태에 따라 처리 (일반적으로 heatmap + simcc)
            if len(ort_outputs) >= 2:
                # SimCC 출력인 경우 (x, y 좌표 직접 출력)
                pred_x = ort_outputs[0][0]  # [133]
                pred_y = ort_outputs[1][0]  # [133] 
                
                # 키포인트 좌표 복원
                target_w, target_h = input_size[1], input_size[0]  # (W, H)
                crop_w, crop_h = crop_size
                x1, y1, x2, y2 = crop_bbox
                
                # 정규화된 좌표를 크롭 이미지 좌표로 변환
                keypoints_x = pred_x * crop_w / target_w
                keypoints_y = pred_y * crop_h / target_h
                
                # 크롭 좌표를 원본 이미지 좌표로 변환
                keypoints_x = keypoints_x + x1
                keypoints_y = keypoints_y + y1
                
                # 키포인트 배열 생성
                keypoints = np.stack([keypoints_x, keypoints_y], axis=1)  # [133, 2]
                
                # 점수 계산 (SimCC의 경우 좌표 신뢰도 기반)
                scores = np.ones(self.num_keypoints, dtype=np.float32) * 0.9  # 기본값
                
            else:
                # 히트맵 출력인 경우
                heatmap = ort_outputs[0][0]  # [133, H, W]
                
                keypoints = []
                scores = []
                
                for i in range(self.num_keypoints):
                    hm = heatmap[i]
                    
                    # 최대값 위치 찾기
                    max_val = np.max(hm)
                    if max_val > 0.1:  # 임계값
                        max_idx = np.unravel_index(np.argmax(hm), hm.shape)
                        
                        # 히트맵 좌표를 이미지 좌표로 변환
                        hm_h, hm_w = hm.shape
                        y_coord = max_idx[0] * crop_size[1] / hm_h + crop_bbox[1]
                        x_coord = max_idx[1] * crop_size[0] / hm_w + crop_bbox[0]
                        
                        keypoints.append([x_coord, y_coord])
                        scores.append(max_val)
                    else:
                        keypoints.append([0, 0])
                        scores.append(0.0)
                
                keypoints = np.array(keypoints, dtype=np.float32)
                scores = np.array(scores, dtype=np.float32)
            
            return keypoints, scores
            
        except Exception as e:
            print(f"❌ 후처리 실패: {e}")
            return np.zeros((self.num_keypoints, 2)), np.zeros(self.num_keypoints)
    
    def process_frame(self, image: np.ndarray) -> Tuple[np.ndarray, List[Tuple[np.ndarray, np.ndarray, List[float]]]]:
        """프레임 처리 (YOLO11L + RTMW ONNX)"""
        start_time = time.time()
        
        # 1. 사람 검출
        person_boxes = self.detect_persons_high_accuracy(image)
        
        # 2. 각 사람에 대해 포즈 추정
        results = []
        for bbox in person_boxes:
            keypoints, scores = self.estimate_pose_onnx(image, bbox)
            results.append((keypoints, scores, bbox))
        
        total_time = time.time() - start_time
        self.inference_times['total'].append(total_time)
        
        # 3. 시각화
        vis_image = self.visualize_results(image, results)
        
        return vis_image, results
    
    def visualize_results(self, image: np.ndarray, results: List[Tuple[np.ndarray, np.ndarray, List[float]]]) -> np.ndarray:
        """결과 시각화 - RTMW 133 키포인트"""
        vis_image = image.copy()
        
        for i, (keypoints, scores, bbox) in enumerate(results):
            # 바운딩박스 그리기
            x1, y1, x2, y2 = map(int, bbox)
            color = (0, 255, 0) if i == 0 else (255, 0, 255)
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(vis_image, f"Person {i+1}", (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # 신체 키포인트 그리기
            self._draw_body_keypoints(vis_image, keypoints, scores)
            
            # 얼굴 키포인트 그리기 (선택적)
            self._draw_face_keypoints(vis_image, keypoints, scores, draw_all=False)
            
            # 손 키포인트 그리기 (선택적)
            self._draw_hand_keypoints(vis_image, keypoints, scores)
        
        # 성능 정보 표시
        if self.inference_times['total']:
            fps = 1.0 / self.inference_times['total'][-1]
            cv2.putText(vis_image, f"FPS: {fps:.1f}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.putText(vis_image, "YOLO11L + RTMW ONNX", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(vis_image, f"ONNX: {self.pose_session.get_providers()[0]}", 
                   (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return vis_image
    
    def _draw_body_keypoints(self, image: np.ndarray, keypoints: np.ndarray, scores: np.ndarray):
        """신체 키포인트 그리기"""
        # 신체 연결선 그리기
        for connection in self.body_connections:
            kpt1_idx, kpt2_idx = connection
            if (kpt1_idx < len(keypoints) and kpt2_idx < len(keypoints) and 
                scores[kpt1_idx] > 0.3 and scores[kpt2_idx] > 0.3):
                
                pt1 = tuple(map(int, keypoints[kpt1_idx]))
                pt2 = tuple(map(int, keypoints[kpt2_idx]))
                cv2.line(image, pt1, pt2, (0, 255, 0), 2)
        
        # 신체 키포인트 그리기
        for i in self.body_keypoints:
            if i < len(keypoints) and scores[i] > 0.3:
                x, y = int(keypoints[i][0]), int(keypoints[i][1])
                if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                    if scores[i] > 0.8:
                        color = (0, 255, 0)  # 높은 신뢰도: 초록
                    elif scores[i] > 0.6:
                        color = (0, 255, 255)  # 중간 신뢰도: 노랑
                    else:
                        color = (0, 0, 255)  # 낮은 신뢰도: 빨강
                    
                    cv2.circle(image, (x, y), 4, color, -1)
    
    def _draw_face_keypoints(self, image: np.ndarray, keypoints: np.ndarray, scores: np.ndarray, draw_all: bool = False):
        """얼굴 키포인트 그리기"""
        if not draw_all:
            # 주요 얼굴 특징점만 그리기
            key_face_points = [30, 48, 54, 36, 45]  # 코끝, 입 끝, 눈 등
            for i in key_face_points:
                face_idx = 17 + i  # 얼굴 키포인트는 17번부터
                if (face_idx < len(keypoints) and scores[face_idx] > 0.5):
                    x, y = int(keypoints[face_idx][0]), int(keypoints[face_idx][1])
                    if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                        cv2.circle(image, (x, y), 2, (255, 255, 0), -1)
        else:
            # 모든 얼굴 키포인트 그리기
            for i in self.face_keypoints:
                if i < len(keypoints) and scores[i] > 0.4:
                    x, y = int(keypoints[i][0]), int(keypoints[i][1])
                    if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                        cv2.circle(image, (x, y), 1, (255, 255, 0), -1)
    
    def _draw_hand_keypoints(self, image: np.ndarray, keypoints: np.ndarray, scores: np.ndarray):
        """손 키포인트 그리기"""
        # 왼손
        for i in self.left_hand_keypoints:
            if i < len(keypoints) and scores[i] > 0.3:
                x, y = int(keypoints[i][0]), int(keypoints[i][1])
                if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                    cv2.circle(image, (x, y), 2, (0, 0, 255), -1)  # 빨강
        
        # 오른손
        for i in self.right_hand_keypoints:
            if i < len(keypoints) and scores[i] > 0.3:
                x, y = int(keypoints[i][0]), int(keypoints[i][1])
                if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                    cv2.circle(image, (x, y), 2, (255, 0, 0), -1)  # 파랑
    
    def benchmark_performance(self, image: np.ndarray, num_runs: int = 20) -> dict:
        """성능 벤치마크"""
        print(f"🏃 ONNX 성능 벤치마크 ({num_runs}회)...")
        
        # 워밍업
        for _ in range(3):
            self.process_frame(image)
        
        # 실제 벤치마크
        self.inference_times = {'detection': [], 'pose': [], 'total': []}
        
        for i in range(num_runs):
            self.process_frame(image)
            if (i + 1) % 5 == 0:
                print(f"   진행률: {i+1}/{num_runs}")
        
        # 통계 계산
        stats = {}
        for key, times in self.inference_times.items():
            if times:
                stats[key] = {
                    'mean': np.mean(times),
                    'std': np.std(times),
                    'min': np.min(times),
                    'max': np.max(times),
                    'fps': 1.0 / np.mean(times) if key == 'total' else None
                }
        
        return stats
    
    def test_single_image(self, image_path: str):
        """단일 이미지 테스트"""
        print(f"\n=== YOLO11L + RTMW ONNX 테스트: {os.path.basename(image_path)} ===")
        
        if not os.path.exists(image_path):
            print(f"❌ 이미지 파일 없음: {image_path}")
            return
            
        # 이미지 로드
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ 이미지 로드 실패: {image_path}")
            return
        
        print(f"📷 이미지 크기: {image.shape}")
        
        # 처리
        vis_image, results = self.process_frame(image)
        
        # 결과 출력
        print(f"✅ 검출된 사람 수: {len(results)}")
        
        for i, (keypoints, scores, bbox) in enumerate(results):
            body_valid = np.sum(scores[self.body_keypoints] > 0.3)
            face_valid = np.sum(scores[self.face_keypoints] > 0.3)
            left_hand_valid = np.sum(scores[self.left_hand_keypoints] > 0.3)
            right_hand_valid = np.sum(scores[self.right_hand_keypoints] > 0.3)
            
            print(f"   사람 {i+1}:")
            print(f"     - 신체: {body_valid}/17 키포인트")
            print(f"     - 얼굴: {face_valid}/68 키포인트")
            print(f"     - 왼손: {left_hand_valid}/21 키포인트")
            print(f"     - 오른손: {right_hand_valid}/21 키포인트")
        
        # 성능 벤치마크
        stats = self.benchmark_performance(image)
        
        print(f"\n📊 ONNX 성능 통계:")
        for stage, stat in stats.items():
            if stat:
                print(f"   {stage}:")
                print(f"     - 평균: {stat['mean']*1000:.1f}ms")
                print(f"     - 최소/최대: {stat['min']*1000:.1f}/{stat['max']*1000:.1f}ms")
                if stat['fps']:
                    print(f"     - FPS: {stat['fps']:.1f}")
        
        # 결과 저장
        output_path = f"onnx_result_{os.path.basename(image_path)}"
        cv2.imwrite(output_path, vis_image)
        print(f"💾 결과 저장: {output_path}")
        
        return vis_image, results, stats
    
    def test_webcam(self, camera_id: int = 0, window_size: Tuple[int, int] = (1280, 720)):
        """실시간 웹캠 테스트"""
        print(f"\n=== YOLO11L + RTMW ONNX 실시간 테스트 (카메라 ID: {camera_id}) ===")
        
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print(f"❌ 웹캠 열기 실패")
            return
        
        # 웹캠 설정
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, window_size[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, window_size[1])
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print(f"🎮 조작법: ESC(종료), S(스크린샷), SPACE(일시정지)")
        
        frame_count = 0
        fps_history = deque(maxlen=30)
        paused = False
        screenshot_count = 0
        
        try:
            while True:
                if not paused:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # 프레임 처리
                    start_time = time.time()
                    vis_frame, results = self.process_frame(frame)
                    process_time = time.time() - start_time
                    
                    # FPS 계산
                    fps = 1.0 / process_time if process_time > 0 else 0
                    fps_history.append(fps)
                    avg_fps = np.mean(fps_history) if fps_history else 0
                    
                    # 추가 정보 표시
                    cv2.putText(vis_frame, f"Avg FPS: {avg_fps:.1f}", 
                               (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    
                    if results:
                        total_keypoints = sum(np.sum(scores > 0.3) for _, scores, _ in results)
                        cv2.putText(vis_frame, f"Total keypoints: {total_keypoints}", 
                                   (10, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    
                    frame_count += 1
                
                # 화면 표시
                cv2.imshow('YOLO11L + RTMW ONNX Real-time', vis_frame)
                
                # 키 입력 처리
                key = cv2.waitKey(1) & 0xFF
                
                if key == 27:  # ESC
                    break
                elif key == ord('s'):  # 스크린샷
                    screenshot_name = f"onnx_webcam_screenshot_{screenshot_count:04d}.jpg"
                    cv2.imwrite(screenshot_name, vis_frame)
                    print(f"📸 스크린샷 저장: {screenshot_name}")
                    screenshot_count += 1
                elif key == ord(' '):  # 일시정지
                    paused = not paused
                    print(f"⏸️ {'일시정지' if paused else '재생'}")
                
                # 성능 통계 출력
                if frame_count % 60 == 0 and frame_count > 0:
                    print(f"📊 프레임 {frame_count}: 평균 {avg_fps:.1f}fps, {len(results)}명 검출")
                    
        except KeyboardInterrupt:
            print("\n⏹️ 사용자가 테스트를 중단했습니다.")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            
            if fps_history:
                final_avg_fps = np.mean(fps_history)
                print(f"\n📊 ONNX 웹캠 테스트 완료:")
                print(f"   - 처리된 프레임: {frame_count}")
                print(f"   - 평균 FPS: {final_avg_fps:.1f}")
                print(f"   - ONNX 프로바이더: {self.pose_session.get_providers()[0]}")
                print(f"   - 스크린샷: {screenshot_count}개 저장")

def main():
    """메인 테스트 함수"""
    # ONNX 모델 경로
    rtmw_onnx_path = "rtmw-x_simcc-cocktail14_pt-ucoco_270e-384x288-f840f204_20231122_384x288.onnx"
    
    # ONNX 파일 존재 확인
    if not os.path.exists(rtmw_onnx_path):
        print(f"❌ ONNX 파일 없음: {rtmw_onnx_path}")
        print(f"💡 먼저 RTMW 모델을 ONNX로 변환해야 합니다.")
        print(f"   python rtmw_onnx_converter.py 실행")
        return
    
    try:
        print("🚀 YOLO11L + RTMW ONNX 하이브리드 추론기 테스트")
        print("=" * 60)
        
        # ONNX 하이브리드 추론기 생성
        inferencer = YOLO11LONNXRTMWHybridInferencer(
            rtmw_onnx_path=rtmw_onnx_path,
            detection_device="auto",
            input_size=(288, 384),  # RTMW 기본 입력 크기
            optimize_for_accuracy=True
        )
        
        # 테스트 이미지
        test_image = "winter01.jpg"
        if os.path.exists(test_image):
            inferencer.test_single_image(test_image)
        else:
            print(f"⚠️ 테스트 이미지 없음: {test_image}")
        
        # 실시간 웹캠 테스트
        print(f"\n🎥 실시간 웹캠 테스트를 시작하시겠습니까? (y/n): ", end="")
        choice = input().strip().lower()
        
        if choice == 'y':
            inferencer.test_webcam()
        
        print(f"\n🏆 ONNX 하이브리드 시스템 특징:")
        print(f"   ⚡ 빠른 추론: ONNX Runtime 최적화")
        print(f"   🎯 고정확도: YOLO11L + RTMW 133 키포인트")
        print(f"   💻 다양한 백엔드: CUDA, CPU, OpenVINO 등")
        print(f"   📦 배포 용이: ONNX 표준 형식")
        
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()