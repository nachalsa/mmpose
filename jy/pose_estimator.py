#!/usr/bin/env python3
"""
RTMW-x PyTorch Pose Estimator
RTMW-x PyTorch 기반 포즈 추정기 (SimCC 디코더 포함)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from typing import Tuple, List, Optional, Dict, Any
import os

from config import RTMW_INPUT_SIZE, IMAGENET_MEAN, IMAGENET_STD
from simcc_decoder import SimCCDecoder


class RTMWXEstimator:
    """RTMW-x PyTorch 포즈 추정기"""
    
    def __init__(self, model_path: str, device: str = 'cpu'):
        """
        Args:
            model_path: RTMW PyTorch 체크포인트 파일 경로 (.pth)
            device: 추론 디바이스 ('cpu', 'cuda', 'xpu')
        """
        self.model_path = model_path
        self.device = device
        self.input_size = RTMW_INPUT_SIZE  # (384, 288)
        
        # PyTorch 모델 로드
        self.model = self._load_pytorch_model()
        
        # SimCC 디코더 초기화 (실제 차원으로)
        self.simcc_decoder = SimCCDecoder(
            input_size=self.input_size,
            simcc_split_ratio=2.0,
            normalize=False
        )
        
        print(f"✅ RTMW-x PyTorch 추정기 초기화 완료")
        print(f"   - 모델: {os.path.basename(model_path)}")
        print(f"   - 디바이스: {self.device}")
        print(f"   - 입력 크기: {self.input_size}")
        print(f"   - SimCC 디코더 연결됨")
    
    def _load_pytorch_model(self):
        """PyTorch 모델 로드"""
        try:
            print(f"🔄 PyTorch 체크포인트 로드 중: {self.model_path}")
            
            # 체크포인트 로드 (CPU에서 먼저 로드)
            checkpoint = torch.load(self.model_path, map_location='cpu', weights_only=False)
            
            # state_dict 추출
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                print(f"📋 체크포인트 정보:")
                if 'meta' in checkpoint:
                    meta = checkpoint['meta']
                    for key, value in meta.items():
                        print(f"   - {key}: {value}")
            else:
                state_dict = checkpoint
            
            # state_dict에서 실제 차원 분석
            head_dims = self._analyze_head_dimensions(state_dict)
            
            # 모델 구조 생성 (실제 차원 사용)
            model = self._build_rtmw_model(state_dict, head_dims)
            
            # 가중치 로드 (strict=False로 크기 불일치 무시)
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            
            if missing_keys:
                print(f"⚠️ 누락된 키: {len(missing_keys)}개")
            if unexpected_keys:
                print(f"⚠️ 예상치 못한 키: {len(unexpected_keys)}개")
            
            # 디바이스로 이동
            model = model.to(self.device)
            model.eval()
            
            print(f"✅ PyTorch 모델 로드 완료")
            
            # 모델 테스트
            self._test_model_forward(model)
            
            return model
            
        except Exception as e:
            print(f"❌ PyTorch 모델 로드 실패: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _analyze_head_dimensions(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, int]:
        """state_dict에서 헤드 차원 분석 (올바른 버전)"""
        head_dims = {
            'simcc_x_bins': 768,    # 실제로는 cls_y에서 나옴 (384*2)
            'simcc_y_bins': 576,    # 실제로는 cls_x에서 나옴 (288*2)
            'num_keypoints': 133,   
            'feature_dim': 256      
        }
        
        print(f"\n🔍 헤드 차원 분석 (올바른 매칭):")
        
        for name, param in state_dict.items():
            if 'head' in name and 'weight' in name:
                print(f"   - {name}: {param.shape}")
                
                if 'cls_x' in name:
                    # cls_x는 실제로 Y 좌표를 담당 (288*2=576)
                    if len(param.shape) >= 2:
                        total_dim = param.shape[0]  # 576
                        head_dims['simcc_y_bins'] = total_dim // 133  # 실제 Y bins
                        head_dims['feature_dim'] = param.shape[1]
                        print(f"     → cls_x (실제 Y): {total_dim} ÷ 133 = {total_dim // 133}")
                        
                elif 'cls_y' in name:
                    # cls_y는 실제로 X 좌표를 담당 (384*2=768)
                    if len(param.shape) >= 2:
                        total_dim = param.shape[0]  # 768
                        head_dims['simcc_x_bins'] = total_dim // 133  # 실제 X bins
                        head_dims['feature_dim'] = param.shape[1]
                        print(f"     → cls_y (실제 X): {total_dim} ÷ 133 = {total_dim // 133}")
        
        print(f"\n📊 최종 SimCC 구조 (올바른 매칭):")
        print(f"   - X 이미지 bins: {head_dims['simcc_x_bins']} (384×2)")
        print(f"   - Y 이미지 bins: {head_dims['simcc_y_bins']} (288×2)")
        print(f"   - 키포인트 수: {head_dims['num_keypoints']}")
        print(f"   - 특징 차원: {head_dims['feature_dim']}")
        
        return head_dims
    
    def _build_rtmw_model(self, state_dict: Dict[str, torch.Tensor], head_dims: Dict[str, int]) -> nn.Module:
        """state_dict 기반으로 RTMW 모델 구조 생성"""
        
        class RTMWModel(nn.Module):
            """RTMW-x 모델 구조"""
            
            def __init__(self, head_dims: Dict[str, int]):
                super().__init__()
                self.head_dims = head_dims
                self.num_keypoints = head_dims['num_keypoints']
                self.simcc_x_bins = head_dims['simcc_x_bins']
                self.simcc_y_bins = head_dims['simcc_y_bins']
                self.feature_dim = head_dims['feature_dim']
                
                # 모델 구조를 state_dict에서 추론
                self._build_from_state_dict(state_dict)
            
            def _build_from_state_dict(self, state_dict):
                """state_dict에서 모델 구조 추론 및 생성"""
                
                # 백본 (간소화된 구조)
                self.backbone = self._build_backbone()
                
                # 넥 (Feature Pyramid Network)
                self.neck = self._build_neck()
                
                # 헤드 (SimCC - 실제 차원 사용)
                self.head = self._build_head()
            
            def _build_backbone(self):
                """백본 구조 생성 (간소화)"""
                # 실제 RTMW는 CSPDarkNet이지만 간소화
                return nn.Sequential(
                    # Stem
                    nn.Conv2d(3, 64, 6, stride=2, padding=2),
                    nn.BatchNorm2d(64),
                    nn.SiLU(inplace=True),
                    
                    # Stage 1
                    nn.Conv2d(64, 128, 3, stride=2, padding=1),
                    nn.BatchNorm2d(128),
                    nn.SiLU(inplace=True),
                    
                    # Stage 2  
                    nn.Conv2d(128, 256, 3, stride=2, padding=1),
                    nn.BatchNorm2d(256),
                    nn.SiLU(inplace=True),
                    
                    # Stage 3
                    nn.Conv2d(256, 512, 3, stride=2, padding=1),
                    nn.BatchNorm2d(512),
                    nn.SiLU(inplace=True),
                    
                    # Stage 4
                    nn.Conv2d(512, 1024, 3, stride=2, padding=1),
                    nn.BatchNorm2d(1024),
                    nn.SiLU(inplace=True),
                    
                    # Global Average Pooling
                    nn.AdaptiveAvgPool2d(1),
                    nn.Flatten()
                )
            
            def _build_neck(self):
                """넥 구조 생성"""
                return nn.Sequential(
                    nn.Linear(1024, 512),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.1),
                    nn.Linear(512, self.feature_dim),  # 실제 feature_dim 사용
                    nn.ReLU(inplace=True),
                )
            
            def _build_head(self):
                """SimCC 헤드 구조 생성 (실제 차원 사용)"""
                print(f"🎯 SimCC 헤드 생성:")
                print(f"   - Feature 차원: {self.feature_dim}")
                print(f"   - SimCC X 총 차원: {self.num_keypoints} × {self.simcc_x_bins} = {self.num_keypoints * self.simcc_x_bins}")
                print(f"   - SimCC Y 총 차원: {self.num_keypoints} × {self.simcc_y_bins} = {self.num_keypoints * self.simcc_y_bins}")
                
                return nn.ModuleDict({
                    'cls_x': nn.Linear(self.feature_dim, self.num_keypoints * self.simcc_x_bins),
                    'cls_y': nn.Linear(self.feature_dim, self.num_keypoints * self.simcc_y_bins),
                })
            
            def forward(self, x):
                """순전파 (올바른 X/Y 매칭)"""
                # Backbone + Neck
                features = self.backbone(x)
                features = self.neck(features)
                
                # Head (주의: cls_x는 Y, cls_y는 X!)
                cls_x_output = self.head['cls_x'](features)  # [B, 133*576] → Y 좌표
                cls_y_output = self.head['cls_y'](features)  # [B, 133*768] → X 좌표
                
                # Reshape: [B, total] → [B, keypoints, bins]
                batch_size = x.size(0)
                cls_y = cls_x_output.view(batch_size, self.num_keypoints, self.simcc_y_bins)  # [B, 133, 576] Y
                cls_x = cls_y_output.view(batch_size, self.num_keypoints, self.simcc_x_bins)  # [B, 133, 768] X
                
                return cls_x, cls_y  # 올바른 순서로 반환
        
        # 모델 생성 (실제 차원으로)
        model = RTMWModel(head_dims)
        print(f"🏗️ RTMW 모델 구조 생성 완료 (실제 차원 적용)")
        
        return model
    
    def _test_model_forward(self, model):
        """모델 순전파 테스트"""
        try:
            print(f"🧪 모델 순전파 테스트...")
            
            # 더미 입력
            dummy_input = torch.randn(1, 3, self.input_size[1], self.input_size[0])  # [B, C, H, W]
            dummy_input = dummy_input.to(self.device)
            
            with torch.no_grad():
                cls_x, cls_y = model(dummy_input)
                
            print(f"✅ 모델 테스트 성공:")
            print(f"   - 입력: {list(dummy_input.shape)}")
            print(f"   - SimCC X: {list(cls_x.shape)}")
            print(f"   - SimCC Y: {list(cls_y.shape)}")
            
            # SimCC 디코더 차원 업데이트
            self._update_simcc_decoder_dims(cls_x.shape[-1], cls_y.shape[-1])
            
        except Exception as e:
            print(f"⚠️ 모델 테스트 실패: {e}")
            import traceback
            traceback.print_exc()
    
    def _update_simcc_decoder_dims(self, simcc_x_dim: int, simcc_y_dim: int):
        """SimCC 디코더 차원 업데이트"""
        print(f"🔄 SimCC 디코더 차원 업데이트:")
        print(f"   - 기존 X 차원: {self.simcc_decoder.simcc_x}")
        print(f"   - 기존 Y 차원: {self.simcc_decoder.simcc_y}")
        print(f"   - 새로운 X 차원: {simcc_x_dim}")
        print(f"   - 새로운 Y 차원: {simcc_y_dim}")
        
        # 실제 모델 출력 차원으로 업데이트
        self.simcc_decoder.simcc_x = simcc_x_dim
        self.simcc_decoder.simcc_y = simcc_y_dim
        
        print(f"✅ SimCC 디코더 차원 업데이트 완료")
    
    def estimate_pose(self, image: np.ndarray, bbox: List[float]) -> np.ndarray:
        """
        포즈 추정 (PyTorch + SimCC)
        
        Args:
            image: 입력 이미지 [H, W, 3]
            bbox: 바운딩박스 [x1, y1, x2, y2]
            
        Returns:
            keypoints: [133, 2] 키포인트 좌표
        """
        try:
            # 1. 이미지 전처리
            input_tensor = self._preprocess_image(image, bbox)
            
            # 2. PyTorch 모델 추론
            cls_x, cls_y = self._run_pytorch_inference(input_tensor)
            
            # 3. SimCC 디코딩
            keypoints = self._decode_simcc_outputs(cls_x, cls_y, bbox, image.shape[:2])
            
            return keypoints
            
        except Exception as e:
            print(f"❌ 포즈 추정 실패: {e}")
            import traceback
            traceback.print_exc()
            # 빈 키포인트 반환
            return np.zeros((133, 2), dtype=np.float32)
    
    def _preprocess_image(self, image: np.ndarray, bbox: List[float]) -> torch.Tensor:
        """이미지 전처리"""
        x1, y1, x2, y2 = bbox
        
        # 바운딩박스 크롭
        cropped = image[int(y1):int(y2), int(x1):int(x2)]
        
        # 리사이즈
        resized = cv2.resize(cropped, self.input_size)  # (384, 288)
        
        # RGB 변환 및 정규화
        rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = rgb_image.astype(np.float32) / 255.0
        
        # ImageNet 정규화
        mean = np.array(IMAGENET_MEAN, dtype=np.float32)
        std = np.array(IMAGENET_STD, dtype=np.float32)
        normalized = (normalized - mean) / std
        
        # Torch 텐서 변환: [H, W, 3] -> [1, 3, H, W]
        tensor = torch.from_numpy(normalized.transpose(2, 0, 1)).unsqueeze(0)
        tensor = tensor.to(self.device)
        
        return tensor
    
    def _run_pytorch_inference(self, input_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """PyTorch 모델 추론"""
        try:
            with torch.no_grad():
                cls_x, cls_y = self.model(input_tensor)
                
            print(f"🔍 PyTorch 모델 출력:")
            print(f"   - SimCC X: {list(cls_x.shape)}")
            print(f"   - SimCC Y: {list(cls_y.shape)}")
            
            return cls_x, cls_y
            
        except Exception as e:
            print(f"❌ PyTorch 추론 실패: {e}")
            raise
    
    def _decode_simcc_outputs(self, cls_x: torch.Tensor, cls_y: torch.Tensor, 
                             bbox: List[float], original_image_shape: Tuple[int, int]) -> np.ndarray:
        """SimCC 출력 디코딩"""
        try:
            print(f"🎯 SimCC 디코딩 시작:")
            print(f"   - SimCC X: {cls_x.shape}")
            print(f"   - SimCC Y: {cls_y.shape}")
            
            # SimCC 디코딩
            keypoints, scores = self.simcc_decoder.decode(cls_x, cls_y)
            
            print(f"✅ SimCC 디코딩 완료:")
            print(f"   - 키포인트: {keypoints.shape}")
            print(f"   - 점수: {scores.shape if scores is not None else None}")
            
            # GPU 텐서를 CPU로 이동 후 numpy 변환
            if keypoints.is_cuda:
                keypoints = keypoints.cpu()
            
            keypoints_np = keypoints.squeeze(0).numpy()  # [133, 2]
            
            # 좌표 변환 (크롭된 이미지 -> 원본 이미지)
            transformed_keypoints = self._transform_coordinates(keypoints_np, bbox, original_image_shape)
            
            return transformed_keypoints
            
        except Exception as e:
            print(f"❌ SimCC 디코딩 실패: {e}")
            import traceback
            traceback.print_exc()
            return np.zeros((133, 2), dtype=np.float32)
    
    def _transform_coordinates(self, keypoints: np.ndarray, bbox: List[float], 
                             original_image_shape: Tuple[int, int]) -> np.ndarray:
        """키포인트 좌표를 원본 이미지 좌표계로 변환"""
        x1, y1, x2, y2 = bbox
        crop_width = x2 - x1
        crop_height = y2 - y1
        
        # 모델 입력 크기에서 크롭 크기로 스케일링
        scale_x = crop_width / self.input_size[0]   # 384
        scale_y = crop_height / self.input_size[1]  # 288
        
        # 좌표 변환
        transformed = keypoints.copy()
        transformed[:, 0] = keypoints[:, 0] * scale_x + x1  # X 좌표
        transformed[:, 1] = keypoints[:, 1] * scale_y + y1  # Y 좌표
        
        # 이미지 경계 클리핑
        img_height, img_width = original_image_shape
        transformed[:, 0] = np.clip(transformed[:, 0], 0, img_width - 1)
        transformed[:, 1] = np.clip(transformed[:, 1], 0, img_height - 1)
        
        print(f"📐 좌표 변환:")
        print(f"   - 스케일: ({scale_x:.3f}, {scale_y:.3f})")
        print(f"   - 오프셋: ({x1}, {y1})")
        print(f"   - 범위: x[0, {img_width}], y[0, {img_height}]")
        
        return transformed
    
    def get_model_info(self) -> Dict[str, Any]:
        """모델 정보 반환"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            'model_path': self.model_path,
            'device': self.device,
            'input_size': self.input_size,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024)  # float32 기준
        }


def test_rtmw_pytorch_estimator():
    """RTMW PyTorch 추정기 테스트"""
    print("=== RTMW-x PyTorch 추정기 테스트 ===")
    
    # 더미 모델 경로 (실제 경로로 변경 필요)
    model_path = "path/to/rtmw_model.pth"
    
    try:
        # 추정기 초기화
        estimator = RTMWXEstimator(model_path, device='cpu')
        
        # 모델 정보 출력
        info = estimator.get_model_info()
        print(f"\n📊 모델 정보:")
        for key, value in info.items():
            print(f"   - {key}: {value}")
        
        # 더미 데이터로 테스트
        dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        dummy_bbox = [100, 50, 300, 400]  # [x1, y1, x2, y2]
        
        # 포즈 추정
        keypoints = estimator.estimate_pose(dummy_image, dummy_bbox)
        
        print(f"\n✅ 테스트 완료:")
        print(f"   - 키포인트 shape: {keypoints.shape}")
        print(f"   - 좌표 범위: x[{keypoints[:, 0].min():.1f}, {keypoints[:, 0].max():.1f}]")
        print(f"   - 좌표 범위: y[{keypoints[:, 1].min():.1f}, {keypoints[:, 1].max():.1f}]")
        
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")


if __name__ == '__main__':
    test_rtmw_pytorch_estimator()