#!/usr/bin/env python3
"""
Pose Estimation Module for RTMW-x (256x192 기준)
RTMW-x WholeBody 포즈 추정기
"""

import torch
import cv2
import numpy as np
import os
from typing import Tuple

from rtmw_model import RTMWXModel
from config import (
    RTMW_INPUT_SIZE, RTMW_NUM_KEYPOINTS, RTMW_SIMCC_SPLIT_RATIO,
    IMAGENET_MEAN, IMAGENET_STD
)


class RTMWXEstimator:
    """RTMW-x WholeBody 포즈 추정기 (133 키포인트, 256x192)"""
    
    def __init__(self, model_path: str, device: str = 'xpu:0'):
        self.device = device
        self.input_size = RTMW_INPUT_SIZE  # (256, 192)
        self.num_keypoints = RTMW_NUM_KEYPOINTS
        self.simcc_split_ratio = RTMW_SIMCC_SPLIT_RATIO
        
        # 모델 로드
        self.model = self._load_rtmw_model(model_path)
        self.model.to(device)
        self.model.eval()
        
        # ImageNet 정규화 파라미터
        self.mean = torch.tensor(IMAGENET_MEAN).to(device)
        self.std = torch.tensor(IMAGENET_STD).to(device)
        
        print(f"✅ RTMW-x 모델 초기화 완료 - 입력 크기: {self.input_size}, 키포인트: {self.num_keypoints}개")
        
    def _load_rtmw_model(self, model_path: str):
        """RTMW-x 모델 로드"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"RTMW-x 모델 파일을 찾을 수 없습니다: {model_path}")
        
        # 체크포인트 로드
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            print("RTMW-x 체크포인트 구조 분석:")
            
            # 백본과 헤드 레이어 분석
            backbone_layers = [k for k in state_dict.keys() if k.startswith('backbone.')]
            head_layers = [k for k in state_dict.keys() if k.startswith('head.')]
            
            print(f"백본 레이어 수: {len(backbone_layers)}")
            print(f"헤드 레이어 수: {len(head_layers)}")
            
            # RTMW 헤드 구조 확인
            rtmw_head_keys = ['head.cls_x.weight', 'head.cls_y.weight']
            for key in rtmw_head_keys:
                if key in state_dict:
                    print(f"  {key}: {state_dict[key].shape}")
                    
        else:
            raise RuntimeError("체크포인트에 state_dict가 없습니다.")
        
        # RTMW-x 모델 구조 생성
        model = RTMWXModel.create_from_checkpoint(state_dict)
        
        # state_dict 로드 (strict=False로 불일치 허용)
        try:
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            print(f"✅ RTMW-x 모델 로드 완료")
            if len(missing_keys) > 0:
                print(f"⚠️ 백본 단순화로 인한 누락된 키: {len(missing_keys)}개")
            
        except Exception as e:
            print(f"❌ RTMW-x 모델 로드 실패: {e}")
            raise RuntimeError(f"RTMW-x 모델 로딩 중 오류 발생: {e}")
        
        return model
    
    def preprocess_image(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> torch.Tensor:
        """RTMW-x 입력을 위한 이미지 전처리 (256x192)"""
        x1, y1, x2, y2 = bbox
        
        # 바운딩 박스 영역 크롭
        person_img = image[y1:y2, x1:x2]
        
        # RTMW-x 표준 입력 크기로 리사이즈 (256x192)
        person_img = cv2.resize(person_img, self.input_size)
        
        # BGR -> RGB 변환
        person_img = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)
        
        # 0-1 정규화
        person_img = person_img.astype(np.float32) / 255.0
        
        # 텐서 변환
        tensor_img = torch.from_numpy(person_img).permute(2, 0, 1)  # (C, H, W)
        tensor_img = tensor_img.unsqueeze(0)  # (1, C, H, W)
        
        # XPU로 이동 후 ImageNet 정규화
        tensor_img = tensor_img.to(self.device)
        tensor_img = (tensor_img - self.mean.view(3, 1, 1)) / self.std.view(3, 1, 1)
        
        return tensor_img
    
    def postprocess_keypoints(self, pred_x: torch.Tensor, pred_y: torch.Tensor, 
                            bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """RTMW-x SimCC 출력 후처리 (384x288 기준)"""
        x1, y1, x2, y2 = bbox
        batch_size, num_keypoints, x_dim = pred_x.shape
        _, _, y_dim = pred_y.shape
        
        print(f"실제 bins: x_dim={x_dim}, y_dim={y_dim}")  # 디버깅용
        
        keypoints = []
        
        for i in range(num_keypoints):
            # SimCC: 각 키포인트별 1D 분포에서 expectation 계산
            x_probs = torch.softmax(pred_x[0, i], dim=0)  
            y_probs = torch.softmax(pred_y[0, i], dim=0)  
            
            # 좌표 인덱스 생성
            x_indices = torch.arange(x_dim, device=self.device, dtype=torch.float32)
            y_indices = torch.arange(y_dim, device=self.device, dtype=torch.float32)
            
            # Expectation으로 좌표 계산
            x_coord = torch.sum(x_probs * x_indices)  
            y_coord = torch.sum(y_probs * y_indices)  
            
            # bins을 실제 픽셀 좌표로 변환 (384x288 기준)
            # 실제 384x288 모델의 경우 bins가 더 클 수 있음
            pixel_x = (x_coord / (x_dim - 1)) * self.input_size[0]  # 384
            pixel_y = (y_coord / (y_dim - 1)) * self.input_size[1]  # 288
            
            # 바운딩 박스 기준으로 원본 이미지 좌표로 변환
            bbox_width = x2 - x1
            bbox_height = y2 - y1
            
            # 정규화 후 바운딩 박스에 맞춤
            norm_x = pixel_x / self.input_size[0]  # 0~1 범위
            norm_y = pixel_y / self.input_size[1]  # 0~1 범위
            
            # 좌표 클램핑
            norm_x = torch.clamp(norm_x, 0.0, 1.0)
            norm_y = torch.clamp(norm_y, 0.0, 1.0)
            
            final_x = x1 + norm_x * bbox_width
            final_y = y1 + norm_y * bbox_height
            
            keypoints.append([final_x.cpu().numpy(), final_y.cpu().numpy()])
        
        return np.array(keypoints)
    
    def estimate_pose(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """RTMW-x로 WholeBody 포즈 추정"""
        input_tensor = self.preprocess_image(image, bbox)
        
        with torch.no_grad():
            pred_x, pred_y = self.model(input_tensor)
            keypoints = self.postprocess_keypoints(pred_x, pred_y, bbox)
            return keypoints