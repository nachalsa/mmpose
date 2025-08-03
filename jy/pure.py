#!/usr/bin/env python3
"""
Intel XPU를 사용한 RTMW-x 추론
MMPose, MMCV 의존성 없이 순수 PyTorch + Intel XPU 사용
"""

import torch
import cv2
import numpy as np
import time
from typing import List, Tuple, Optional
import urllib.request
import os

class SimplePersonDetector:
    """간단한 YOLOv5 기반 인체 검출기"""
    
    def __init__(self, device='xpu:0'):
        self.device = device
        # YOLOv5s 모델 로드 (COCO 데이터셋 훈련)
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        self.model.to(device)
        self.model.eval()
        
    def detect_persons(self, image: np.ndarray, conf_thresh: float = 0.5) -> List[Tuple[int, int, int, int]]:
        """
        이미지에서 사람 바운딩 박스 검출
        
        Args:
            image: BGR 이미지 (OpenCV 형식)
            conf_thresh: 신뢰도 임계값
            
        Returns:
            List of (x1, y1, x2, y2) bounding boxes
        """
        # BGR -> RGB 변환
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 추론
        with torch.no_grad():
            results = self.model(rgb_image)
        
        # 결과 파싱 (class 0 = person)
        detections = results.pandas().xyxy[0]
        person_boxes = detections[
            (detections['class'] == 0) & 
            (detections['confidence'] > conf_thresh)
        ]
        
        boxes = []
        for _, row in person_boxes.iterrows():
            x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            boxes.append((x1, y1, x2, y2))
            
        return boxes

class RTMWXEstimator:
    """RTMW-x 포즈 추정기 (133 키포인트)"""
    
    def __init__(self, model_path: str, device='xpu:0'):
        self.device = device
        self.input_size = (384, 288)  # RTMW-x 입력 크기
        self.num_keypoints = 133  # WholeBody 키포인트 수
        
        # 모델 로드
        self.model = self._load_model(model_path)
        self.model.to(device)
        self.model.eval()
        
        # 정규화 파라미터 (ImageNet 기준)
        self.mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).to(device)
        
    def _load_model(self, model_path: str):
        """모델 로드 (PyTorch 체크포인트)"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")
            
        # 체크포인트 로드
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # 모델 구조는 실제 RTMW-x 구조로 대체해야 함
        # 여기서는 간단한 예시 구조 사용
        model = self._create_rtmw_model()
        
        # 상태 딕셔너리 로드
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
            
        return model
    
    def _create_rtmw_model(self):
        """RTMW-x 모델 구조 생성 (실제 구조로 대체 필요)"""
        # 실제로는 RTMPose의 정확한 구조를 구현해야 함
        # 여기서는 간단한 예시
        import torch.nn as nn
        
        class SimpleRTMW(nn.Module):
            def __init__(self):
                super().__init__()
                # 백본 (CSPNeXt 등)
                self.backbone = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=False)
                self.backbone.fc = nn.Identity()
                
                # 헤드 (SimCC)
                self.head_x = nn.Linear(2048, 384)  # x 좌표 예측
                self.head_y = nn.Linear(2048, 288)  # y 좌표 예측
                
            def forward(self, x):
                # 백본 특징 추출
                features = self.backbone(x)
                
                # SimCC 헤드로 좌표 예측
                pred_x = self.head_x(features)  # (B, 384)
                pred_y = self.head_y(features)  # (B, 288)
                
                return pred_x, pred_y
                
        return SimpleRTMW()
    
    def preprocess_image(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> torch.Tensor:
        """이미지 전처리"""
        x1, y1, x2, y2 = bbox
        
        # 바운딩 박스 크롭
        person_img = image[y1:y2, x1:x2]
        
        # 크기 조정
        person_img = cv2.resize(person_img, self.input_size)
        
        # BGR -> RGB 변환
        person_img = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)
        
        # 정규화
        person_img = person_img.astype(np.float32) / 255.0
        
        # 텐서 변환
        tensor_img = torch.from_numpy(person_img).permute(2, 0, 1)  # (C, H, W)
        
        # 정규화
        tensor_img = (tensor_img - self.mean.view(3, 1, 1)) / self.std.view(3, 1, 1)
        
        # 배치 차원 추가
        tensor_img = tensor_img.unsqueeze(0)  # (1, C, H, W)
        
        return tensor_img.to(self.device)
    
    def postprocess_keypoints(self, pred_x: torch.Tensor, pred_y: torch.Tensor, 
                            bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """키포인트 후처리"""
        x1, y1, x2, y2 = bbox
        
        # Softmax로 확률 분포 변환
        prob_x = torch.softmax(pred_x, dim=1)
        prob_y = torch.softmax(pred_y, dim=1)
        
        # 기댓값으로 좌표 계산
        x_coords = torch.sum(prob_x * torch.arange(384, device=self.device), dim=1)
        y_coords = torch.sum(prob_y * torch.arange(288, device=self.device), dim=1)
        
        # 원본 이미지 좌표로 변환
        scale_x = (x2 - x1) / 384
        scale_y = (y2 - y1) / 288
        
        keypoints = torch.stack([x_coords * scale_x + x1, y_coords * scale_y + y1], dim=1)
        
        return keypoints.cpu().numpy()
    
    def estimate_pose(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """단일 바운딩 박스에서 포즈 추정"""
        # 전처리
        input_tensor = self.preprocess_image(image, bbox)
        
        # 추론
        with torch.no_grad():
            pred_x, pred_y = self.model(input_tensor)
        
        # 후처리
        keypoints = self.postprocess_keypoints(pred_x, pred_y, bbox)
        
        return keypoints

def download_model(url: str, save_path: str):
    """모델 다운로드"""
    if not os.path.exists(save_path):
        print(f"모델 다운로드 중: {url}")
        urllib.request.urlretrieve(url, save_path)
        print(f"다운로드 완료: {save_path}")

def draw_keypoints(image: np.ndarray, keypoints: np.ndarray, 
                  connections: Optional[List[Tuple[int, int]]] = None) -> np.ndarray:
    """키포인트 시각화"""
    img_vis = image.copy()
    
    # 키포인트 그리기
    for i, (x, y) in enumerate(keypoints):
        cv2.circle(img_vis, (int(x), int(y)), 3, (0, 255, 0), -1)
        cv2.putText(img_vis, str(i), (int(x), int(y-5)), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    
    # 연결선 그리기 (간단한 신체 연결)
    if connections is None:
        # COCO 17 키포인트 연결 (Body 부분만)
        connections = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # 머리
            (5, 6), (5, 7), (6, 8), (7, 9), (8, 10),  # 팔
            (11, 12), (11, 13), (12, 14), (13, 15), (14, 16),  # 다리
            (5, 11), (6, 12)  # 몸통
        ]
    
    for start_idx, end_idx in connections:
        if start_idx < len(keypoints) and end_idx < len(keypoints):
            start_point = (int(keypoints[start_idx][0]), int(keypoints[start_idx][1]))
            end_point = (int(keypoints[end_idx][0]), int(keypoints[end_idx][1]))
            cv2.line(img_vis, start_point, end_point, (255, 0, 0), 2)
    
    return img_vis

def main():
    """메인 실행 함수"""
    print("=== RTMW-x Intel XPU 추론 ===")
    
    # XPU 가용성 확인
    if not torch.xpu.is_available():
        print("❌ Intel XPU를 사용할 수 없습니다.")
        return
    
    device = 'xpu:0'
    print(f"✅ 사용 디바이스: {device}")
    print(f"XPU 디바이스: {torch.xpu.get_device_name(0)}")
    
    # 모델 경로 설정 (실제 RTMW-x 모델 경로로 변경)
    model_path = "/path/to/rtmw-x_model.pth"  # 실제 모델 경로
    model_url = "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-x_simcc-coco-wholebody_pt-body7_270e-384x288-401dfc90_20230629.pth"
    
    # 모델 다운로드 (필요시)
    if not os.path.exists(model_path):
        print("실제 RTMW-x 모델 파일을 지정해주세요.")
        return
    
    try:
        # 검출기 및 포즈 추정기 초기화
        print("모델 로딩 중...")
        person_detector = SimplePersonDetector(device=device)
        pose_estimator = RTMWXEstimator(model_path, device=device)
        print("✅ 모델 로딩 완료")
        
        # 웹캠 초기화
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ 웹캠을 열 수 없습니다.")
            return
        
        print("웹캠 추론 시작 (ESC 키로 종료)")
        
        fps_counter = 0
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_start = time.time()
            
            # 1. 인체 검출
            person_boxes = person_detector.detect_persons(frame, conf_thresh=0.5)
            
            # 2. 각 검출된 사람에 대해 포즈 추정
            for bbox in person_boxes:
                keypoints = pose_estimator.estimate_pose(frame, bbox)
                frame = draw_keypoints(frame, keypoints)
                
                # 바운딩 박스 그리기
                x1, y1, x2, y2 = bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            
            frame_time = time.time() - frame_start
            fps = 1.0 / frame_time if frame_time > 0 else 0
            
            # FPS 표시
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Persons: {len(person_boxes)}", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # 결과 표시
            cv2.imshow('RTMW-x Intel XPU', frame)
            
            # ESC 키로 종료
            if cv2.waitKey(1) & 0xFF == 27:
                break
            
            fps_counter += 1
            
        # 평균 FPS 계산
        total_time = time.time() - start_time
        avg_fps = fps_counter / total_time if total_time > 0 else 0
        print(f"평균 FPS: {avg_fps:.2f}")
        
        # 정리
        cap.release()
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()