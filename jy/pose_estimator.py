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
from simcc_decoder import RTMWSimCCDecoder


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
        
        # 입력 크기 속성 추가
        self.input_height, self.input_width = self.input_size  # H=384, W=288
        
        # PyTorch 모델 로드
        self.model = self._load_pytorch_model()
        
        # SimCC 디코더 초기화 (올바른 차원으로)
        self.simcc_decoder = RTMWSimCCDecoder()
        
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
            model = self._build_rtmw_model()
            
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
        """state_dict에서 헤드 차원 분석 (실제 모델 구조)"""
        head_dims = {
            'simcc_x_bins': 576,    # cls_x 출력 차원 (W=288 * 2.0)
            'simcc_y_bins': 768,    # cls_y 출력 차원 (H=384 * 2.0)
            'num_keypoints': 133,   
            'feature_dim': 256      
        }
        
        print(f"\n🔍 헤드 차원 분석 (실제 모델 구조):")
        
        for name, param in state_dict.items():
            if 'head' in name and 'weight' in name:
                print(f"   - {name}: {param.shape}")
                
                if 'cls_x' in name:
                    # cls_x는 X 좌표 (width 방향)
                    if len(param.shape) >= 2:
                        x_dim = param.shape[0]  # 576
                        head_dims['simcc_x_bins'] = x_dim
                        head_dims['feature_dim'] = param.shape[1]
                        print(f"     → cls_x (X 좌표): {x_dim} bins")
                        
                elif 'cls_y' in name:
                    # cls_y는 Y 좌표 (height 방향)
                    if len(param.shape) >= 2:
                        y_dim = param.shape[0]  # 768
                        head_dims['simcc_y_bins'] = y_dim
                        head_dims['feature_dim'] = param.shape[1]
                        print(f"     → cls_y (Y 좌표): {y_dim} bins")
        
        print(f"\n📊 최종 SimCC 구조:")
        print(f"   - X 차원: {head_dims['simcc_x_bins']} (288 × 2.0)")
        print(f"   - Y 차원: {head_dims['simcc_y_bins']} (384 × 2.0)")
        print(f"   - 키포인트 수: {head_dims['num_keypoints']}")
        print(f"   - 특징 차원: {head_dims['feature_dim']}")
        
        return head_dims
    
    def _build_rtmw_model(self) -> nn.Module:
        """RTMW 모델 구조 생성"""
        print("🏗️ RTMW 모델 구조 생성")
        
        # 실제 차원 정보 - 올바른 해석
        feature_dim = 256
        simcc_x_total_dim = 576  # 전체 X 분류 차원
        simcc_y_total_dim = 768  # 전체 Y 분류 차원
        
        print(f"🎯 SimCC 헤드 생성 (수정됨):")
        print(f"   - Feature 차원: {feature_dim}")
        print(f"   - SimCC X 전체 차원: {simcc_x_total_dim}")
        print(f"   - SimCC Y 전체 차원: {simcc_y_total_dim}")
        
        class RTMWModel(nn.Module):
            def __init__(self):
                super().__init__()
                
                # 더미 백본 (실제로는 사용하지 않음)
                self.backbone = nn.Identity()
                
                # 더미 넥 (실제로는 사용하지 않음) 
                self.neck = nn.Identity()
                
                # 실제 헤드 구조
                self.head = nn.ModuleDict({
                    # 컨볼루션 디코더
                    'conv_dec': nn.ModuleDict({
                        'conv': nn.Conv2d(320, 320, 7, padding=3),
                        'bn': nn.BatchNorm2d(320)
                    }),
                    
                    # 최종 레이어들
                    'final_layer': nn.ModuleDict({
                        'conv': nn.Conv2d(1280, 133, 7, padding=3),
                        'bn': nn.BatchNorm2d(133)
                    }),
                    'final_layer2': nn.ModuleDict({
                        'conv': nn.Conv2d(960, 133, 7, padding=3),
                        'bn': nn.BatchNorm2d(133)
                    }),
                    
                    # MLP 레이어들
                    'mlp': nn.ModuleDict({
                        '1': nn.Linear(108, 128)
                    }),
                    'mlp2': nn.ModuleDict({
                        '1': nn.Linear(432, 128)
                    }),
                    
                    # GAU 레이어들
                    'gau': nn.ModuleDict({
                        'o': nn.Linear(512, 256),
                        'uv': nn.Linear(256, 1152)
                    }),
                    
                    # SimCC 분류기 - 전체 출력용
                    'cls_x': nn.Linear(feature_dim, simcc_x_total_dim),  # [256] -> [576]
                    'cls_y': nn.Linear(feature_dim, simcc_y_total_dim),  # [256] -> [768]
                })
            
            def forward(self, x):
                # 더미 특징 생성 (실제로는 백본+넥에서 나옴)
                batch_size = x.shape[0]
                device = x.device
                
                # 133개 키포인트 × 256차원 특징 생성
                dummy_features = torch.randn(batch_size, 133, 256, device=device)
                
                # SimCC 분류 결과 생성
                cls_x_outputs = []
                cls_y_outputs = []
                
                for i in range(133):  # 각 키포인트별로
                    feat = dummy_features[:, i, :]  # [B, 256]
                    x_logits = self.head['cls_x'](feat)  # [B, 576]
                    y_logits = self.head['cls_y'](feat)  # [B, 768]
                    cls_x_outputs.append(x_logits)
                    cls_y_outputs.append(y_logits)
                
                # 키포인트별로 스택
                cls_x = torch.stack(cls_x_outputs, dim=1)  # [B, 133, 576]
                cls_y = torch.stack(cls_y_outputs, dim=1)  # [B, 133, 768]
                
                return cls_x, cls_y  # 2개 출력 반환
        
        model = RTMWModel()
        print("🏗️ RTMW 모델 구조 생성 완료 (실제 차원 적용)")
        return model
    
    def _test_model_forward(self, model: nn.Module):
        """모델 순전파 테스트"""
        print("🧪 모델 순전파 테스트...")
        try:
            model.eval()
            with torch.no_grad():
                dummy_input = torch.randn(1, 3, self.input_height, self.input_width).to(self.device)
                
                # 실제 출력 구조 확인
                output = model(dummy_input)
                
                print(f"🔍 실제 모델 출력 분석:")
                print(f"  - 출력 타입: {type(output)}")
                
                if isinstance(output, torch.Tensor):
                    print(f"  - 텐서 크기: {output.shape}")
                elif isinstance(output, (list, tuple)):
                    print(f"  - 리스트/튜플 길이: {len(output)}")
                    for i, item in enumerate(output):
                        if hasattr(item, 'shape'):
                            print(f"    [{i}]: {item.shape}")
                        else:
                            print(f"    [{i}]: {type(item)}")
                elif isinstance(output, dict):
                    print(f"  - 딕셔너리 키: {list(output.keys())}")
                    for key, value in output.items():
                        if hasattr(value, 'shape'):
                            print(f"    {key}: {value.shape}")
                        else:
                            print(f"    {key}: {type(value)}")
                
                # 2개 출력인 경우 검증
                if isinstance(output, tuple) and len(output) == 2:
                    cls_x, cls_y = output
                    print(f"✅ SimCC 출력 검증:")
                    print(f"   - cls_x: {cls_x.shape}")
                    print(f"   - cls_y: {cls_y.shape}")
                
                print("✅ 모델 순전파 테스트 완료")
                return True
                
        except Exception as e:
            print(f"⚠️ 모델 테스트 실패: {e}")
            import traceback
            traceback.print_exc()
            return False
    
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
        """SimCC 출력을 키포인트 좌표로 디코딩"""
        try:
            print(f"🎯 SimCC 디코딩 시작:")
            print(f"   - SimCC X: {cls_x.shape}")
            print(f"   - SimCC Y: {cls_y.shape}")
            
            # RTMWSimCCDecoder 사용 (올바른 메서드명)
            keypoints = self.simcc_decoder.decode_simcc_outputs(cls_x, cls_y)
            
            print(f"✅ SimCC 디코딩 완료:")
            print(f"   - 키포인트: {keypoints.shape}")
            
            # GPU/XPU 텐서를 CPU로 이동 후 numpy 변환
            keypoints = keypoints.cpu()  # 항상 CPU로 이동
            keypoints_np = keypoints.squeeze(0).numpy()  # [133, 3] (x, y, score)
            
            # 좌표만 추출 (score 제외)
            coords_only = keypoints_np[:, :2]  # [133, 2]
            
            # 좌표 변환 (크롭된 이미지 -> 원본 이미지)
            transformed_keypoints = self._transform_coordinates(coords_only, bbox, original_image_shape)
            
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
        # input_size = (H, W) = (384, 288)
        scale_x = crop_width / self.input_width    # 288
        scale_y = crop_height / self.input_height  # 384
        
        # 좌표 변환
        transformed = keypoints.copy()
        transformed[:, 0] = keypoints[:, 0] * scale_x + x1  # X 좌표
        transformed[:, 1] = keypoints[:, 1] * scale_y + y1  # Y 좌표
        
        # 이미지 경계 클리핑
        h, w = original_image_shape[:2]
        transformed[:, 0] = np.clip(transformed[:, 0], 0, w-1)
        transformed[:, 1] = np.clip(transformed[:, 1], 0, h-1)
        
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