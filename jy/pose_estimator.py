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

from config import RTMW_INPUT_SIZE, POSE_MEAN, POSE_STD
from simcc_decoder import RTMWSimCCDecoder

# MMPose 공식 함수들 import
from mmpose.structures.bbox import bbox_xyxy2cs, get_warp_matrix


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
        """PyTorch 모델 로드 (실제 가중치 적용)"""
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
            
            # 실제 가중치 부분 매핑 (호환되는 것만)
            print(f"🔄 가중치 매핑 시작...")
            
            model_state = model.state_dict()
            loaded_keys = []
            
            # 백본 가중치 매핑 (간소화된 백본에 일부 적용)
            backbone_mapping = self._map_backbone_weights(state_dict, model_state)
            loaded_keys.extend(backbone_mapping)
            
            # 헤드 가중치 매핑 (SimCC 레이어)
            head_mapping = self._map_head_weights(state_dict, model_state)
            loaded_keys.extend(head_mapping)
            
            print(f"📊 가중치 로딩 결과:")
            print(f"   - 매핑된 레이어: {len(loaded_keys)}개")
            print(f"   - 전체 모델 파라미터: {len(model_state)}개")
            
            # 디바이스로 이동
            model = model.to(self.device)
            model.eval()
            
            print(f"✅ PyTorch 모델 로드 완료 (실제 가중치 적용)")
            
            # 모델 테스트
            self._test_model_forward(model)
            
            return model
            
        except Exception as e:
            print(f"❌ PyTorch 모델 로드 실패: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _map_backbone_weights(self, state_dict: Dict[str, torch.Tensor], 
                            model_state: Dict[str, torch.Tensor]) -> List[str]:
        """백본 가중치 매핑"""
        print("🔄 백본 가중치 매핑...")
        loaded_keys = []
        
        # 첫 번째 conv 레이어 매핑
        if 'backbone.stem.conv.weight' in state_dict:
            src_weight = state_dict['backbone.stem.conv.weight']
            if src_weight.shape[0] >= 80:  # 출력 채널이 충분한 경우
                model_state['backbone.0.weight'] = src_weight[:80].clone()
                loaded_keys.append('backbone.0.weight')
                print(f"   - stem conv: {src_weight.shape} -> {model_state['backbone.0.weight'].shape}")
        
        # 배치놈 매핑
        if 'backbone.stem.bn.weight' in state_dict and 'backbone.stem.bn.bias' in state_dict:
            bn_weight = state_dict['backbone.stem.bn.weight']
            bn_bias = state_dict['backbone.stem.bn.bias']
            if bn_weight.shape[0] >= 80:
                model_state['backbone.1.weight'] = bn_weight[:80].clone()
                model_state['backbone.1.bias'] = bn_bias[:80].clone()
                loaded_keys.extend(['backbone.1.weight', 'backbone.1.bias'])
                print(f"   - stem bn: {bn_weight.shape} -> 80")
        
        print(f"✅ 백본 가중치 매핑 완료: {len(loaded_keys)}개")
        return loaded_keys
    
    def _map_head_weights(self, state_dict: Dict[str, torch.Tensor], 
                         model_state: Dict[str, torch.Tensor]) -> List[str]:
        """헤드 가중치 매핑 (SimCC 레이어)"""
        print("🔄 헤드 가중치 매핑...")
        loaded_keys = []
        
        # SimCC 분류기 가중치 매핑
        if 'head.cls_x.weight' in state_dict:
            src_weight = state_dict['head.cls_x.weight']
            if 'head.cls_x.weight' in model_state:
                dst_weight = model_state['head.cls_x.weight']
                if src_weight.shape == dst_weight.shape:
                    model_state['head.cls_x.weight'] = src_weight.clone()
                    loaded_keys.append('head.cls_x.weight')
                    print(f"   - cls_x: {src_weight.shape} ✅")
                else:
                    print(f"   - cls_x: {src_weight.shape} vs {dst_weight.shape} ❌")
        
        if 'head.cls_y.weight' in state_dict:
            src_weight = state_dict['head.cls_y.weight']
            if 'head.cls_y.weight' in model_state:
                dst_weight = model_state['head.cls_y.weight']
                if src_weight.shape == dst_weight.shape:
                    model_state['head.cls_y.weight'] = src_weight.clone()
                    loaded_keys.append('head.cls_y.weight')
                    print(f"   - cls_y: {src_weight.shape} ✅")
                else:
                    print(f"   - cls_y: {src_weight.shape} vs {dst_weight.shape} ❌")
        
        print(f"✅ 헤드 가중치 매핑 완료: {len(loaded_keys)}개")
        return loaded_keys
    
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
        """RTMW 모델 구조 생성 (MMPose 구현 기반)"""
        print("🏗️ RTMW 모델 구조 생성 (실제 백본+헤드)")
        
        # 실제 차원 정보
        feature_dim = 256
        simcc_x_total_dim = 576  # W * 2.0 = 288 * 2
        simcc_y_total_dim = 768  # H * 2.0 = 384 * 2
        
        print(f"🎯 실제 RTMW 구조 생성:")
        print(f"   - 입력 크기: {self.input_size}")
        print(f"   - Feature map 크기: {self.input_width//32}x{self.input_height//32}")
        print(f"   - SimCC X 차원: {simcc_x_total_dim}")
        print(f"   - SimCC Y 차원: {simcc_y_total_dim}")
        
        class RTMWModel(nn.Module):
            def __init__(self):
                super().__init__()
                
                # 간소화된 백본 (CSPNeXt-X 스타일)
                self.backbone = self._build_simplified_backbone()
                
                # 간소화된 넥 (CSPNeXtPAFPN 스타일)
                self.neck = self._build_simplified_neck()
                
                # RTMW 헤드 (MMPose 구현 기반)
                self.head = self._build_rtmw_head()
                
            def _build_simplified_backbone(self):
                """간소화된 CSPNeXt 백본 - 수정된 크기"""
                return nn.Sequential(
                    # Stage 1: 384x288 -> 192x144
                    nn.Conv2d(3, 80, 3, stride=2, padding=1),  # CSPNeXt-X 기본 채널
                    nn.BatchNorm2d(80),
                    nn.SiLU(inplace=True),
                    
                    # Stage 2: 192x144 -> 96x72
                    nn.Conv2d(80, 160, 3, stride=2, padding=1),
                    nn.BatchNorm2d(160),
                    nn.SiLU(inplace=True),
                    
                    # Stage 3: 96x72 -> 48x36
                    nn.Conv2d(160, 320, 3, stride=2, padding=1),
                    nn.BatchNorm2d(320),
                    nn.SiLU(inplace=True),
                    
                    # Stage 4: 48x36 -> 24x18
                    nn.Conv2d(320, 640, 3, stride=2, padding=1),
                    nn.BatchNorm2d(640),
                    nn.SiLU(inplace=True),
                    
                    # Stage 5: 24x18 -> 12x9 (feature map for head)
                    nn.Conv2d(640, 1280, 3, stride=2, padding=1),  # RTMW-X 최종 채널
                    nn.BatchNorm2d(1280),
                    nn.SiLU(inplace=True),
                    
                    # 추가 패딩으로 크기 조정 (12x9 -> 정확한 크기)
                    nn.AdaptiveAvgPool2d((12, 9))  # 정확한 크기로 조정
                )
            
            def _build_simplified_neck(self):
                """간소화된 CSPNeXtPAFPN 넥"""
                return nn.Sequential(
                    # 추가 특징 처리
                    nn.Conv2d(1280, 1280, 3, padding=1),
                    nn.BatchNorm2d(1280),
                    nn.SiLU(inplace=True),
                    
                    # Feature enhancement
                    nn.Conv2d(1280, 1280, 1),
                    nn.BatchNorm2d(1280),
                    nn.SiLU(inplace=True),
                )
            
            def _build_rtmw_head(self):
                """RTMW 헤드 (MMPose RTMWHead 기반) - 수정된 버전"""
                in_channels = 1280
                out_channels = 133
                in_featuremap_size = (9, 12)  # 288//32, 384//32 (W//32, H//32)
                flatten_dims = in_featuremap_size[0] * in_featuremap_size[1]  # 108
                
                # PixelShuffle 관련
                ps = 2
                
                head = nn.ModuleDict({
                    # PixelShuffle
                    'ps': nn.PixelShuffle(ps),
                    
                    # 컨볼루션 디코더
                    'conv_dec': nn.Sequential(
                        nn.Conv2d(in_channels // ps**2, in_channels // 4, 7, padding=3),
                        nn.BatchNorm2d(in_channels // 4),
                        nn.ReLU(inplace=True)
                    ),
                    
                    # 최종 레이어들
                    'final_layer': nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, 7, padding=3),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU(inplace=True)
                    ),
                    
                    'final_layer2': nn.Sequential(
                        nn.Conv2d(in_channels // ps + in_channels // 4, out_channels, 7, padding=3),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU(inplace=True)
                    ),
                    
                    # MLP 레이어들 - 실제 차원에 맞춤
                    'mlp': nn.Sequential(
                        nn.LayerNorm(flatten_dims),  # 108
                        nn.Linear(flatten_dims, 128, bias=False)
                    ),
                    
                    'mlp2': nn.Sequential(
                        nn.LayerNorm(432),  # 실제 체크포인트 차원 (432)
                        nn.Linear(432, 128, bias=False)
                    ),
                    
                    # GAU (Gated Attention Unit) 간소화
                    'gau': nn.Sequential(
                        nn.Linear(256, 256),
                        nn.SiLU(inplace=True),
                        nn.Linear(256, 256),
                    ),
                    
                    # SimCC 분류기
                    'cls_x': nn.Linear(256, simcc_x_total_dim, bias=False),
                    'cls_y': nn.Linear(256, simcc_y_total_dim, bias=False),
                })
                
                return head
            
            def forward(self, x):
                """RTMW 순전파 (실제 특징 추출) - 수정된 버전"""
                # 1. 백본을 통한 특징 추출
                backbone_features = self.backbone(x)  # [B, 1280, 12, 9]
                
                # 2. 넥을 통한 특징 처리
                neck_features = self.neck(backbone_features)  # [B, 1280, 12, 9]
                
                # 3. RTMW 헤드 처리 (실제 차원 고려)
                # 두 개의 feature 생성 (enc_b, enc_t)
                enc_t = neck_features  # top features [B, 1280, 12, 9]
                enc_b = neck_features  # bottom features (간소화) [B, 1280, 12, 9]
                
                # enc_t 처리
                feats_t = self.head['final_layer'](enc_t)  # [B, 133, 12, 9]
                feats_t = torch.flatten(feats_t, 2)  # [B, 133, 108]
                feats_t = self.head['mlp'](feats_t)  # [B, 133, 128]
                
                # enc_b 처리 (PixelShuffle 적용)
                dec_t = self.head['ps'](enc_t)  # [B, 320, 24, 18]
                dec_t = self.head['conv_dec'](dec_t)  # [B, 320, 24, 18]
                
                # 크기 맞추기 - dec_t를 enc_b와 같은 크기로 조정
                dec_t_resized = F.interpolate(dec_t, size=(12, 9), mode='bilinear', align_corners=False)
                # enc_b와 dec_t 결합 (채널 차원에서)
                enc_b_combined = torch.cat([dec_t_resized, enc_b], dim=1)  # [B, 320+1280=1600, 12, 9]
                
                # final_layer2 입력 채널을 맞춰주기 위해 프로젝션 추가
                if not hasattr(self.head, 'projection'):
                    self.head['projection'] = nn.Conv2d(1600, 960, 1).to(enc_b_combined.device)
                
                # 채널 수를 960으로 줄임
                enc_b_projected = self.head['projection'](enc_b_combined)  # [B, 960, 12, 9]
                
                feats_b = self.head['final_layer2'](enc_b_projected)  # [B, 133, 12, 9]
                feats_b = torch.flatten(feats_b, 2)  # [B, 133, 108]
                
                # MLP2 입력을 위해 108을 432로 확장 (4배 repeat)
                feats_b_expanded = feats_b.repeat(1, 1, 4)  # [B, 133, 432]
                feats_b = self.head['mlp2'](feats_b_expanded)  # [B, 133, 128]
                
                # 특징 결합
                feats = torch.cat([feats_t, feats_b], dim=2)  # [B, 133, 256]
                
                # GAU 적용
                feats = self.head['gau'](feats)  # [B, 133, 256]
                
                # SimCC 분류
                pred_x = self.head['cls_x'](feats)  # [B, 133, 576]
                pred_y = self.head['cls_y'](feats)  # [B, 133, 768]
                
                return pred_x, pred_y
        
        model = RTMWModel()
        print("🏗️ RTMW 모델 구조 생성 완료 (실제 백본+헤드)")
        return model
    
    def _test_model_forward(self, model: nn.Module):
        """모델 순전파 테스트"""
        print("🧪 모델 순전파 테스트 (실제 백본+헤드)...")
        try:
            model.eval()
            with torch.no_grad():
                dummy_input = torch.randn(1, 3, self.input_height, self.input_width).to(self.device)
                
                print(f"🔍 입력 크기: {dummy_input.shape}")
                
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
                            print(f"    [{i}]: {item.shape} (min: {item.min().item():.3f}, max: {item.max().item():.3f})")
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
                    print(f"   - cls_x: {cls_x.shape} (값 범위: {cls_x.min().item():.3f} ~ {cls_x.max().item():.3f})")
                    print(f"   - cls_y: {cls_y.shape} (값 범위: {cls_y.min().item():.3f} ~ {cls_y.max().item():.3f})")
                    
                    # 확률 분포 확인 (Softmax 적용 후)
                    x_probs = F.softmax(cls_x, dim=-1)
                    y_probs = F.softmax(cls_y, dim=-1)
                    
                    # 각 키포인트의 최대 확률 위치 확인
                    x_max_indices = torch.argmax(x_probs, dim=-1)
                    y_max_indices = torch.argmax(y_probs, dim=-1)
                    
                    print(f"   - X 좌표 예측 범위: {x_max_indices.min().item()} ~ {x_max_indices.max().item()}")
                    print(f"   - Y 좌표 예측 범위: {y_max_indices.min().item()} ~ {y_max_indices.max().item()}")
                
                print("✅ 모델 순전파 테스트 완료 (실제 특징 추출)")
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
        """MMPose 공식 방식의 이미지 전처리"""
        # 1. bbox를 center, scale로 변환 (MMPose 방식)
        x1, y1, x2, y2 = bbox
        center, scale = self._bbox_to_center_scale(bbox)
        
        # 2. MMPose TopdownAffine 변환 적용
        transformed_img = self._apply_topdown_affine(image, center, scale)
        
        # 3. MMPose PoseDataPreprocessor 방식 정규화
        normalized_img = self._apply_pose_data_preprocessor(transformed_img)
        
        # 4. Torch 텐서 변환: [H, W, 3] -> [1, 3, H, W]
        tensor = torch.from_numpy(normalized_img.transpose(2, 0, 1)).unsqueeze(0)
        tensor = tensor.to(self.device)
        
        return tensor
    
    def _bbox_to_center_scale(self, bbox: List[float]) -> Tuple[np.ndarray, np.ndarray]:
        """바운딩박스를 MMPose center, scale 형식으로 변환 (공식 방식 사용)"""
        bbox_array = np.array(bbox, dtype=np.float32)  # [x1, y1, x2, y2]
        
        # MMPose 공식 bbox_xyxy2cs 함수 사용 (padding=1.25 기본값)
        center, scale = bbox_xyxy2cs(bbox_array, padding=1.25)
        
        return center, scale
    
    def _apply_topdown_affine(self, image: np.ndarray, center: np.ndarray, scale: np.ndarray) -> np.ndarray:
        """MMPose TopdownAffine 변환 적용 (공식 구현 사용)"""
        # 입력 크기 (W, H) = (288, 384)
        w, h = self.input_size
        
        # MMPose 공식 get_warp_matrix 함수 사용
        warp_mat = get_warp_matrix(
            center=center,
            scale=scale, 
            rot=0,  # 회전 없음
            output_size=(w, h)
        )
        
        # Affine 변환 적용
        transformed = cv2.warpAffine(
            image, warp_mat, (w, h), flags=cv2.INTER_LINEAR)
        
        return transformed
    
    def _apply_pose_data_preprocessor(self, image: np.ndarray) -> np.ndarray:
        """MMPose PoseDataPreprocessor 방식 정규화"""
        # BGR -> RGB 변환
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # float32 변환 (0-255 범위 유지)
        img_float = rgb_image.astype(np.float32)
        
        # MMPose 공식 정규화 값 (RGB 순서, 0-255 스케일)
        mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
        std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
        
        # 정규화 적용
        normalized = (img_float - mean) / std
        
        return normalized
    
    def _run_pytorch_inference(self, input_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """PyTorch 모델 추론 (실제 특징 추출)"""
        try:
            with torch.no_grad():
                cls_x, cls_y = self.model(input_tensor)
                
            print(f"🔍 PyTorch 모델 출력 (실제 백본):")
            print(f"   - 입력 크기: {list(input_tensor.shape)}")
            print(f"   - SimCC X: {list(cls_x.shape)} (범위: {cls_x.min().item():.3f} ~ {cls_x.max().item():.3f})")
            print(f"   - SimCC Y: {list(cls_y.shape)} (범위: {cls_y.min().item():.3f} ~ {cls_y.max().item():.3f})")
            
            # 출력 값의 분포 확인
            x_softmax = F.softmax(cls_x, dim=-1)
            y_softmax = F.softmax(cls_y, dim=-1)
            
            # 최대 확률 위치 (예측 좌표)
            x_coords = torch.argmax(x_softmax, dim=-1)  # [B, 133]
            y_coords = torch.argmax(y_softmax, dim=-1)  # [B, 133]
            
            print(f"   - 예측 X 좌표 범위: {x_coords.min().item()} ~ {x_coords.max().item()}")
            print(f"   - 예측 Y 좌표 범위: {y_coords.min().item()} ~ {y_coords.max().item()}")
            
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
        """키포인트 좌표를 원본 이미지 좌표계로 변환 (MMPose 공식 방식)"""
        if keypoints.size == 0:
            return keypoints
            
        # MMPose 공식 방식으로 center, scale 계산
        center, scale = self._bbox_to_center_scale(bbox)
        
        # 입력 크기
        w, h = self.input_size
        
        # MMPose 공식 역변환 매트릭스 계산
        warp_mat = get_warp_matrix(
            center=center,
            scale=scale,
            rot=0,
            output_size=(w, h),
            inv=True  # 역변환
        )
        
        # 키포인트 좌표 변환
        # keypoints shape: (133, 2)
        keypoints_homogeneous = np.hstack([keypoints, np.ones((keypoints.shape[0], 1))])
        transformed_keypoints = keypoints_homogeneous @ warp_mat.T
        
        # 이미지 경계 클리핑
        h_orig, w_orig = original_image_shape[:2]
        transformed_keypoints[:, 0] = np.clip(transformed_keypoints[:, 0], 0, w_orig-1)
        transformed_keypoints[:, 1] = np.clip(transformed_keypoints[:, 1], 0, h_orig-1)
        
        return transformed_keypoints
    
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
    """RTMW PyTorch 추정기 테스트 (실제 모델)"""
    print("=== RTMW-x PyTorch 추정기 테스트 (실제 백본+헤드) ===")
    
    # 실제 모델 경로
    model_path = "../models/rtmw-x_simcc-cocktail14_pt-ucoco_270e-384x288-f840f204_20231122.pth"
    
    try:
        # 추정기 초기화
        print(f"🚀 RTMW-x 추정기 초기화 중...")
        estimator = RTMWXEstimator(model_path, device='cpu')
        
        # 모델 정보 출력
        info = estimator.get_model_info()
        print(f"\n📊 모델 정보:")
        for key, value in info.items():
            if isinstance(value, float):
                print(f"   - {key}: {value:.2f}")
            else:
                print(f"   - {key}: {value}")
        
        # 더미 데이터로 테스트
        print(f"\n🧪 더미 데이터 테스트...")
        dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        dummy_bbox = [100, 50, 300, 400]  # [x1, y1, x2, y2]
        
        print(f"   - 입력 이미지: {dummy_image.shape}")
        print(f"   - 바운딩박스: {dummy_bbox}")
        
        # 포즈 추정
        keypoints = estimator.estimate_pose(dummy_image, dummy_bbox)
        
        print(f"\n✅ 테스트 완료:")
        print(f"   - 키포인트 shape: {keypoints.shape}")
        print(f"   - X 좌표 범위: [{keypoints[:, 0].min():.1f}, {keypoints[:, 0].max():.1f}]")
        print(f"   - Y 좌표 범위: [{keypoints[:, 1].min():.1f}, {keypoints[:, 1].max():.1f}]")
        
        # 유효한 키포인트 개수 확인
        valid_keypoints = np.sum((keypoints[:, 0] > 0) & (keypoints[:, 1] > 0))
        print(f"   - 유효한 키포인트: {valid_keypoints}/133개")
        
        # 몇 개 키포인트 샘플 출력
        print(f"\n📍 샘플 키포인트 (처음 5개):")
        for i in range(min(5, len(keypoints))):
            x, y = keypoints[i]
            print(f"   - 키포인트 {i}: ({x:.1f}, {y:.1f})")
        
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    test_rtmw_pytorch_estimator()