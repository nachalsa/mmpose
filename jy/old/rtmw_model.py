#!/usr/bin/env python3
"""
RTMW-x 모델 구조 정의 (MMPose 호환)
Real-Time Multi-Person WholeBody Pose Estimation Model Structure
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RTMWXModel:
    """RTMW-x 모델 구조 정의"""
    
    @staticmethod
    def create_from_checkpoint(state_dict):
        """체크포인트에서 RTMW-x 모델 구조 생성"""
        
        class RTMWXFromCheckpoint(nn.Module):
            def __init__(self, state_dict):
                super().__init__()
                
                # 1. 백본 특징 추출기 구성
                self.backbone = self._build_simplified_backbone()
                
                # 2. RTMW-x 헤드 구조 생성  
                self.head = self._build_rtmw_head_from_state_dict(state_dict)
                
                # 3. RTMW-x 파라미터
                self.input_size = (384, 288)
                self.num_keypoints = 133
                self.simcc_split_ratio = 2.0
                
                # 4. 실제 출력 차원 분석
                self._analyze_output_dims(state_dict)
                
            def _analyze_output_dims(self, state_dict):
                """실제 출력 차원 분석"""
                if 'head.cls_x.weight' in state_dict and 'head.cls_y.weight' in state_dict:
                    self.x_output_size = state_dict['head.cls_x.weight'].shape[0]  # 576
                    self.y_output_size = state_dict['head.cls_y.weight'].shape[0]  # 768
                    
                    print(f"📊 실제 모델 출력: x={self.x_output_size}, y={self.y_output_size}")
                    
                    # MMPose 방식: 전체 출력이 키포인트별로 분할되는지 확인
                    if self.x_output_size % self.num_keypoints == 0:
                        self.x_bins_per_kpt = self.x_output_size // self.num_keypoints
                        print(f"📊 x 키포인트별 bins: {self.x_bins_per_kpt}")
                    else:
                        print(f"⚠️ x 출력이 {self.num_keypoints}로 나누어떨어지지 않음")
                    
                    if self.y_output_size % self.num_keypoints == 0:
                        self.y_bins_per_kpt = self.y_output_size // self.num_keypoints
                        print(f"📊 y 키포인트별 bins: {self.y_bins_per_kpt}")
                    else:
                        print(f"⚠️ y 출력이 {self.num_keypoints}로 나누어떨어지지 않음")
                
            def _build_simplified_backbone(self):
                """단순화된 백본"""
                backbone = nn.Sequential(
                    # 384x288 -> 192x144
                    nn.Conv2d(3, 64, 7, stride=2, padding=3),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(3, stride=2, padding=1),  # 96x72
                    
                    # 96x72 -> 48x36
                    nn.Conv2d(64, 128, 3, stride=2, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                    
                    # 48x36 -> 24x18
                    nn.Conv2d(128, 256, 3, stride=2, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True),
                    
                    # 24x18 -> 12x9
                    nn.Conv2d(256, 512, 3, stride=2, padding=1),
                    nn.BatchNorm2d(512),
                    nn.ReLU(inplace=True),
                    
                    # Global pooling
                    nn.AdaptiveAvgPool2d(1),
                    nn.Flatten(),
                    
                    # Feature projection
                    nn.Linear(512, 256),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.2)
                )
                
                return backbone
                
            def _build_rtmw_head_from_state_dict(self, state_dict):
                """실제 RTMW-x SimCC 헤드"""
                head = nn.Module()
                
                if 'head.cls_x.weight' in state_dict:
                    cls_x_weight = state_dict['head.cls_x.weight']
                    x_out_features, in_features = cls_x_weight.shape
                    
                    cls_x = nn.Linear(in_features, x_out_features)
                    cls_x.weight.data = cls_x_weight.clone()
                    if 'head.cls_x.bias' in state_dict:
                        cls_x.bias.data = state_dict['head.cls_x.bias'].clone()
                    head.cls_x = cls_x
                    
                    print(f"✅ RTMW-x cls_x: {in_features} -> {x_out_features}")
                
                if 'head.cls_y.weight' in state_dict:
                    cls_y_weight = state_dict['head.cls_y.weight']
                    y_out_features, in_features = cls_y_weight.shape
                    
                    cls_y = nn.Linear(in_features, y_out_features)
                    cls_y.weight.data = cls_y_weight.clone()
                    if 'head.cls_y.bias' in state_dict:
                        cls_y.bias.data = state_dict['head.cls_y.bias'].clone()
                    head.cls_y = cls_y
                    
                    print(f"✅ RTMW-x cls_y: {in_features} -> {y_out_features}")
                
                return head
            
            def forward(self, x):
                """RTMW-x 순전파 (원본 출력 그대로 반환)"""
                # 1. 백본을 통한 특징 추출
                features = self.backbone(x)
                
                # 2. SimCC 헤드 출력 (원본 그대로)
                cls_x_output = self.head.cls_x(features)  # [B, 576]
                cls_y_output = self.head.cls_y(features)  # [B, 768]
                
                # 3. 원본 출력 그대로 반환 (후처리는 디코더에서)
                return cls_x_output, cls_y_output
        
        return RTMWXFromCheckpoint(state_dict)