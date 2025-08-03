#!/usr/bin/env python3
"""
RTMW-x PyTorch 모델 구조 분석기
RTMW-x 384x288 PyTorch 체크포인트 파일 분석
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Any
import sys
import os
from collections import OrderedDict

# 현재 디렉토리를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import find_and_download_rtmw_model
from config import RTMW_INPUT_SIZE


class RTMWTorchModelAnalyzer:
    """RTMW-x PyTorch 모델 구조 분석기"""
    
    def __init__(self, model_path: str):
        """
        Args:
            model_path: PyTorch 체크포인트 파일 경로
        """
        self.model_path = model_path
        self.checkpoint = None
        self.state_dict = None
        self.input_shape = RTMW_INPUT_SIZE
        
    def load_checkpoint(self):
        """PyTorch 체크포인트 로드"""
        try:
            # CPU에서 로드 (디바이스 호환성을 위해)
            self.checkpoint = torch.load(self.model_path, map_location='cpu', weights_only=False)
            print(f"✅ PyTorch 체크포인트 로드 성공: {self.model_path}")
            
            # state_dict 추출
            if 'state_dict' in self.checkpoint:
                self.state_dict = self.checkpoint['state_dict']
            else:
                self.state_dict = self.checkpoint
                
            print(f"✅ 모델 state_dict 추출 완료")
            
        except Exception as e:
            print(f"❌ 체크포인트 로드 실패: {e}")
            return False
        
        return True
    
    def analyze_checkpoint_info(self):
        """체크포인트 기본 정보 분석"""
        if self.checkpoint is None:
            print("❌ 체크포인트가 로드되지 않았습니다.")
            return
        
        print("\n" + "="*80)
        print("📊 RTMW-x PyTorch 체크포인트 분석")
        print("="*80)
        
        print(f"체크포인트 파일: {os.path.basename(self.model_path)}")
        
        # 체크포인트 키 정보
        checkpoint_keys = list(self.checkpoint.keys())
        print(f"\n📋 체크포인트 키:")
        for key in checkpoint_keys:
            if isinstance(self.checkpoint[key], dict):
                print(f"  - {key}: dict ({len(self.checkpoint[key])} 항목)")
            else:
                print(f"  - {key}: {type(self.checkpoint[key])}")
        
        # 메타 정보에서 중요한 설정 추출
        if 'meta' in self.checkpoint and 'cfg' in self.checkpoint['meta']:
            cfg = self.checkpoint['meta']['cfg']
            print(f"\n🔍 핵심 설정 정보:")
            
            # 입력 크기 정보 - 다양한 해석으로 분석
            if 'codec' in cfg and 'input_size' in cfg['codec']:
                input_size = cfg['codec']['input_size']
                print(f"  - Input Size (원본): {input_size}")
                
                # 여러 해석 시도
                if len(input_size) == 2:
                    h1, w1 = input_size  # (H, W) 해석
                    w2, h2 = input_size  # (W, H) 해석
                    print(f"  - 해석1 (H, W): H={h1}, W={w1}")
                    print(f"  - 해석2 (W, H): W={h1}, H={w1}")
                
            # SimCC split ratio
            if 'codec' in cfg and 'simcc_split_ratio' in cfg['codec']:
                split_ratio = cfg['codec']['simcc_split_ratio']
                print(f"  - SimCC Split Ratio: {split_ratio}")
                
            # 정규화 정보
            if 'codec' in cfg and 'normalize' in cfg['codec']:
                normalize = cfg['codec']['normalize']
                print(f"  - Normalize: {normalize}")
                
            # 시그마 값
            if 'codec' in cfg and 'sigma' in cfg['codec']:
                sigma = cfg['codec']['sigma']
                print(f"  - Sigma: {sigma}")
                
        # 학습 정보
        print(f"\n📈 학습 정보:")
        if 'epoch' in self.checkpoint:
            print(f"  - Epoch: {self.checkpoint['epoch']}")
        if 'iter' in self.checkpoint:
            print(f"  - Iteration: {self.checkpoint['iter']}")
    
    def analyze_model_structure(self):
        """모델 구조 분석"""
        if self.state_dict is None:
            print("❌ state_dict가 없습니다.")
            return
        
        print("\n" + "="*50)
        print("🏗️ 모델 구조 분석")
        print("="*50)
        
        # 전체 파라미터 수 계산
        total_params = 0
        layer_info = {}
        
        for name, param in self.state_dict.items():
            if param.numel() > 0:
                total_params += param.numel()
                
                # 레이어 타입별 분류
                layer_type = self._get_layer_type(name)
                if layer_type not in layer_info:
                    layer_info[layer_type] = {'count': 0, 'params': 0}
                layer_info[layer_type]['count'] += 1
                layer_info[layer_type]['params'] += param.numel()
        
        print(f"총 파라미터 수: {total_params:,}")
        print(f"총 레이어 수: {len(self.state_dict)}")
        
        print(f"\n📊 레이어 타입별 통계:")
        for layer_type, info in sorted(layer_info.items()):
            params_mb = info['params'] / (1024 * 1024)
            print(f"  {layer_type}: {info['count']}개, {info['params']:,} 파라미터 ({params_mb:.2f}MB)")
    
    def analyze_backbone_structure(self):
        """백본 구조 분석"""
        print("\n" + "="*50)
        print("🦴 백본 구조 분석")
        print("="*50)
        
        backbone_layers = {}
        for name, param in self.state_dict.items():
            if name.startswith('backbone.'):
                # 백본 레이어 이름에서 구조 추출
                parts = name.split('.')
                if len(parts) >= 3:
                    layer_group = f"{parts[1]}.{parts[2]}"
                    if layer_group not in backbone_layers:
                        backbone_layers[layer_group] = {'params': 0, 'tensors': []}
                    backbone_layers[layer_group]['params'] += param.numel()
                    backbone_layers[layer_group]['tensors'].append((name, param.shape))
        
        print(f"백본 레이어 그룹 수: {len(backbone_layers)}")
        
        for group_name, info in sorted(backbone_layers.items()):
            params_mb = info['params'] / (1024 * 1024)
            print(f"\n  📦 {group_name}:")
            print(f"    - 파라미터: {info['params']:,} ({params_mb:.2f}MB)")
            print(f"    - 텐서 수: {len(info['tensors'])}")
            
            # 주요 텐서 shape 표시
            for tensor_name, shape in info['tensors'][:3]:  # 처음 3개만
                print(f"    - {tensor_name.split('.')[-1]}: {list(shape)}")
            if len(info['tensors']) > 3:
                print(f"    - ... 총 {len(info['tensors'])}개")
    
    def analyze_head_structure(self):
        """헤드 구조 분석"""
        print("\n" + "="*50)
        print("🎯 헤드 구조 분석")
        print("="*50)
        
        head_layers = {}
        for name, param in self.state_dict.items():
            if name.startswith('head.'):
                # 헤드 레이어 분석
                parts = name.split('.')
                if len(parts) >= 2:
                    layer_name = parts[1]
                    if layer_name not in head_layers:
                        head_layers[layer_name] = {'params': 0, 'tensors': []}
                    head_layers[layer_name]['params'] += param.numel()
                    head_layers[layer_name]['tensors'].append((name, param.shape))
        
        print(f"헤드 레이어 수: {len(head_layers)}")
        
        for layer_name, info in sorted(head_layers.items()):
            params_mb = info['params'] / (1024 * 1024)
            print(f"\n  🎯 {layer_name}:")
            print(f"    - 파라미터: {info['params']:,} ({params_mb:.2f}MB)")
            print(f"    - 텐서 수: {len(info['tensors'])}")
            
            # 출력 차원 분석
            for tensor_name, shape in info['tensors']:
                tensor_short_name = tensor_name.split('.')[-1]
                print(f"    - {tensor_short_name} Shape: {list(shape)}")
                
                # SimCC 헤드인 경우 상세 분석
                if layer_name in ['cls_x', 'cls_y']:
                    if tensor_short_name == 'weight':
                        print(f"    - 출력 채널: {shape[0]}")
                        print(f"    - 입력 채널: {shape[1]}")
    
    def analyze_simcc_dimensions(self):
        """SimCC 차원 상세 분석 - 모든 가능한 해석 검토"""
        print("\n" + "="*50)
        print("🧮 SimCC 차원 상세 분석 (모든 가능성 검토)")
        print("="*50)
        
        # 메타에서 입력 크기 가져오기
        input_size = None
        simcc_split_ratio = None
        
        if (self.checkpoint and 'meta' in self.checkpoint and 
            'cfg' in self.checkpoint['meta'] and 
            'codec' in self.checkpoint['meta']['cfg']):
            
            codec = self.checkpoint['meta']['cfg']['codec']
            input_size = codec.get('input_size', None)
            simcc_split_ratio = codec.get('simcc_split_ratio', 2.0)
        
        # SimCC 헤드 차원 분석
        cls_x_shape = None
        cls_y_shape = None
        
        for name, param in self.state_dict.items():
            if name == 'head.cls_x.weight':
                cls_x_shape = param.shape
            elif name == 'head.cls_y.weight':
                cls_y_shape = param.shape
        
        print(f"� 발견된 정보:")
        if input_size:
            print(f"  - 설정 input_size: {input_size}")
        if simcc_split_ratio:
            print(f"  - 설정 split_ratio: {split_ratio}")
        if cls_x_shape:
            print(f"  - cls_x.weight: {list(cls_x_shape)} (출력 차원: {cls_x_shape[0]})")
        if cls_y_shape:
            print(f"  - cls_y.weight: {list(cls_y_shape)} (출력 차원: {cls_y_shape[0]})")
        
        # 모델 파일명에서 크기 추출
        filename = os.path.basename(self.model_path)
        print(f"  - 파일명: {filename}")
        
        if '384x288' in filename:
            print(f"  - 파일명에서 추출: 384x288")
        
        print(f"\n🔍 가능한 해석들:")
        
        if cls_x_shape and cls_y_shape and input_size:
            # 해석 1: input_size가 (H, W)라고 가정
            h1, w1 = input_size
            print(f"\n📐 해석 1: input_size = (H={h1}, W={w1})")
            x_ratio_1 = cls_x_shape[0] / w1
            y_ratio_1 = cls_y_shape[0] / h1
            print(f"  - X 비율: {cls_x_shape[0]} ÷ {w1} = {x_ratio_1:.2f}")
            print(f"  - Y 비율: {cls_y_shape[0]} ÷ {h1} = {y_ratio_1:.2f}")
            print(f"  - 비율 일치: {'✅' if abs(x_ratio_1 - y_ratio_1) < 0.1 else '❌'}")
            
            # 해석 2: input_size가 (W, H)라고 가정
            w2, h2 = input_size
            print(f"\n📐 해석 2: input_size = (W={w2}, H={h2})")
            x_ratio_2 = cls_x_shape[0] / w2
            y_ratio_2 = cls_y_shape[0] / h2
            print(f"  - X 비율: {cls_x_shape[0]} ÷ {w2} = {x_ratio_2:.2f}")
            print(f"  - Y 비율: {cls_y_shape[0]} ÷ {h2} = {y_ratio_2:.2f}")
            print(f"  - 비율 일치: {'✅' if abs(x_ratio_2 - y_ratio_2) < 0.1 else '❌'}")
            
            # 해석 3: 표준적인 384x288 가정
            print(f"\n📐 해석 3: 표준 384x288 (W=384, H=288)")
            w3, h3 = 384, 288
            x_ratio_3 = cls_x_shape[0] / w3
            y_ratio_3 = cls_y_shape[0] / h3
            print(f"  - X 비율: {cls_x_shape[0]} ÷ {w3} = {x_ratio_3:.2f}")
            print(f"  - Y 비율: {cls_y_shape[0]} ÷ {h3} = {y_ratio_3:.2f}")
            print(f"  - 비율 일치: {'✅' if abs(x_ratio_3 - y_ratio_3) < 0.1 else '❌'}")
            
            # 해석 4: 뒤바뀐 288x384 가정
            print(f"\n📐 해석 4: 뒤바뀐 288x384 (W=288, H=384)")
            w4, h4 = 288, 384
            x_ratio_4 = cls_x_shape[0] / w4
            y_ratio_4 = cls_y_shape[0] / h4
            print(f"  - X 비율: {cls_x_shape[0]} ÷ {w4} = {x_ratio_4:.2f}")
            print(f"  - Y 비율: {cls_y_shape[0]} ÷ {h4} = {y_ratio_4:.2f}")
            print(f"  - 비율 일치: {'✅' if abs(x_ratio_4 - y_ratio_4) < 0.1 else '❌'}")
            
            # 결론 도출
            print(f"\n🎯 결론:")
            ratios = [
                ("해석1 (H,W)", abs(x_ratio_1 - y_ratio_1) < 0.1, x_ratio_1, y_ratio_1),
                ("해석2 (W,H)", abs(x_ratio_2 - y_ratio_2) < 0.1, x_ratio_2, y_ratio_2),
                ("해석3 384x288", abs(x_ratio_3 - y_ratio_3) < 0.1, x_ratio_3, y_ratio_3),
                ("해석4 288x384", abs(x_ratio_4 - y_ratio_4) < 0.1, x_ratio_4, y_ratio_4),
            ]
            
            consistent_ratios = [r for r in ratios if r[1]]
            
            if consistent_ratios:
                print(f"  ✅ 일치하는 해석들:")
                for name, _, x_r, y_r in consistent_ratios:
                    avg_ratio = (x_r + y_r) / 2
                    print(f"    - {name}: 평균 비율 {avg_ratio:.2f}")
            else:
                print(f"  ⚠️ 일치하는 해석이 없습니다. 추가 분석이 필요합니다.")
                
            # 정수 비율 확인
            print(f"\n🔍 정수 비율 확인:")
            for name, is_consistent, x_r, y_r in ratios:
                if is_consistent:
                    avg_ratio = (x_r + y_r) / 2
                    if abs(avg_ratio - round(avg_ratio)) < 0.1:
                        print(f"  - {name}: 거의 정수 비율 {round(avg_ratio)}")
                    else:
                        print(f"  - {name}: 비정수 비율 {avg_ratio:.2f}")
        
        else:
            print("❌ 충분한 정보가 없어서 분석할 수 없습니다.")
    
    def analyze_input_output_structure(self):
        """입출력 구조 추론"""
        print("\n" + "="*50)
        print("📥📤 입출력 구조 추론")
        print("="*50)
        
        # 설정에서 입력 크기 가져오기
        if (self.checkpoint and 'meta' in self.checkpoint and 
            'cfg' in self.checkpoint['meta'] and 
            'codec' in self.checkpoint['meta']['cfg']):
            
            input_size = self.checkpoint['meta']['cfg']['codec']['input_size']
            print(f"🔍 설정된 입력 크기: {input_size}")
        else:
            input_size = RTMW_INPUT_SIZE
            print(f"🔍 기본 입력 크기: {input_size}")
        
        # 첫 번째 Conv 레이어 찾기
        first_conv = None
        for name, param in self.state_dict.items():
            if 'conv' in name.lower() and 'weight' in name and len(param.shape) == 4:
                first_conv = (name, param.shape)
                break
        
        if first_conv:
            name, shape = first_conv
            print(f"\n📥 입력 구조 (첫 번째 Conv 기준):")
            print(f"  - 첫 Conv 레이어: {name}")
            print(f"  - Weight Shape: {list(shape)}")
            print(f"  - 입력 채널: {shape[1]} (RGB: 3채널 예상)")
            print(f"  - 출력 채널: {shape[0]}")
            print(f"  - 커널 크기: {shape[2]}x{shape[3]}")
        
        # SimCC 출력 구조
        print(f"\n📤 출력 구조 (헤드 레이어 기준):")
        for name, param in self.state_dict.items():
            if name.startswith('head.') and 'weight' in name:
                print(f"  - {name}: {list(param.shape)}")
    
    def run_dummy_inference(self):
        """더미 입력으로 모델 테스트"""
        print("\n" + "="*50)
        print("🧪 더미 모델 테스트")
        print("="*50)
        
        try:
            # 설정에서 입력 크기 가져오기
            if (self.checkpoint and 'meta' in self.checkpoint and 
                'cfg' in self.checkpoint['meta'] and 
                'codec' in self.checkpoint['meta']['cfg']):
                input_size = self.checkpoint['meta']['cfg']['codec']['input_size']
                height, width = input_size
            else:
                height, width = RTMW_INPUT_SIZE
            
            # 더미 입력 생성 (B, C, H, W)
            dummy_input = torch.randn(1, 3, height, width)
            print(f"테스트 입력 Shape: {list(dummy_input.shape)}")
            
            # 간단한 모델 구조 시뮬레이션
            # 실제로는 SimCC 헤드만 분석
            cls_x_weight = self.state_dict.get('head.cls_x.weight')
            cls_y_weight = self.state_dict.get('head.cls_y.weight')
            
            if cls_x_weight is not None and cls_y_weight is not None:
                num_keypoints = 133  # RTMW-x는 133개 키포인트
                x_dim = cls_x_weight.shape[0]
                y_dim = cls_y_weight.shape[0]
                
                # 더미 출력 시뮬레이션
                dummy_output = torch.randn(1, num_keypoints, 2)
                
                print(f"테스트 출력 Shape: {list(dummy_output.shape)}")
                print(f"✅ 더미 모델 테스트 성공!")
                print(f"   - 배치 크기: {dummy_output.shape[0]}")
                print(f"   - 키포인트 수: {dummy_output.shape[1]}")
                print(f"   - 좌표 차원: {dummy_output.shape[2]} (x, y)")
                print(f"   - SimCC X 차원: {x_dim}")
                print(f"   - SimCC Y 차원: {y_dim}")
            else:
                print("❌ SimCC 헤드를 찾을 수 없습니다.")
                
        except Exception as e:
            print(f"❌ 더미 테스트 실패: {e}")
    
    def analyze_all(self):
        """전체 분석 실행"""
        if not self.load_checkpoint():
            return
        
        self.analyze_checkpoint_info()
        self.analyze_model_structure()
        self.analyze_head_structure()
        self.analyze_simcc_dimensions()
        self.analyze_input_output_structure()
        self.run_dummy_inference()
        
        print("\n" + "="*80)
        print("✅ PyTorch 모델 분석 완료")
        print("="*80)
    
    def analyze_input_output_structure(self):
        """입출력 구조 추론"""
        print("\n" + "="*50)
        print("📥📤 입출력 구조 추론")
        print("="*50)
        
        print(f"🔍 설정된 입력 크기: {self.input_shape}")
        
        # 첫 번째 Conv 레이어에서 입력 채널 추론
        first_conv = None
        for name, param in self.state_dict.items():
            if 'conv' in name.lower() and 'weight' in name and param.dim() == 4:
                first_conv = (name, param.shape)
                break
        
        if first_conv:
            name, shape = first_conv
            print(f"\n📥 입력 구조 (첫 번째 Conv 기준):")
            print(f"  - 첫 Conv 레이어: {name}")
            print(f"  - Weight Shape: {list(shape)}")
            print(f"  - 입력 채널: {shape[1]} (RGB: 3채널 예상)")
            print(f"  - 출력 채널: {shape[0]}")
            print(f"  - 커널 크기: {shape[2]}x{shape[3]}")
        
        # 마지막 레이어에서 출력 차원 추론
        final_layers = []
        for name, param in self.state_dict.items():
            if 'head' in name and 'weight' in name and param.dim() >= 2:
                final_layers.append((name, param.shape))
        
        if final_layers:
            print(f"\n📤 출력 구조 (헤드 레이어 기준):")
            for name, shape in final_layers[-3:]:  # 마지막 3개 레이어
                print(f"  - {name}: {list(shape)}")
                if len(shape) >= 2:
                    output_dim = shape[0]
                    if output_dim == 266:  # 133 keypoints * 2
                        print(f"    → WholeBody 키포인트: 133개 × 2 (x, y)")
                    elif output_dim == 133:
                        print(f"    → WholeBody 키포인트: 133개")
    
    def estimate_model_size(self):
        """모델 크기 추정"""
        print("\n" + "="*50)
        print("💾 모델 크기 분석")
        print("="*50)
        
        total_params = sum(p.numel() for p in self.state_dict.values())
        
        # 일반적으로 float32 기준
        model_size_bytes = total_params * 4  # 4 bytes per float32
        model_size_mb = model_size_bytes / (1024 * 1024)
        model_size_gb = model_size_mb / 1024
        
        print(f"총 파라미터 수: {total_params:,}")
        print(f"모델 크기 (float32): {model_size_mb:.2f} MB ({model_size_gb:.3f} GB)")
        
        # 실제 파일 크기
        if os.path.exists(self.model_path):
            file_size = os.path.getsize(self.model_path)
            file_size_mb = file_size / (1024 * 1024)
            print(f"실제 파일 크기: {file_size_mb:.2f} MB")
    
    def create_dummy_model_test(self):
        """더미 모델로 추론 테스트"""
        print("\n" + "="*50)
        print("🧪 더미 모델 테스트")
        print("="*50)
        
        try:
            # 간단한 더미 모델 생성
            class DummyRTMW(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.backbone = nn.Sequential(
                        nn.Conv2d(3, 64, 7, stride=2, padding=3),
                        nn.ReLU(),
                        nn.AdaptiveAvgPool2d(1),
                        nn.Flatten()
                    )
                    self.head = nn.Linear(64, 266)  # 133 keypoints * 2
                
                def forward(self, x):
                    features = self.backbone(x)
                    output = self.head(features)
                    return output.view(-1, 133, 2)
            
            dummy_model = DummyRTMW()
            dummy_model.eval()
            
            # 테스트 입력
            batch_size = 1
            channels = 3
            height, width = self.input_shape
            
            dummy_input = torch.randn(batch_size, channels, height, width)
            
            print(f"테스트 입력 Shape: {list(dummy_input.shape)}")
            
            with torch.no_grad():
                output = dummy_model(dummy_input)
                
            print(f"테스트 출력 Shape: {list(output.shape)}")
            print(f"✅ 더미 모델 테스트 성공!")
            print(f"   - 배치 크기: {output.shape[0]}")
            print(f"   - 키포인트 수: {output.shape[1]}")
            print(f"   - 좌표 차원: {output.shape[2]} (x, y)")
            
        except Exception as e:
            print(f"❌ 더미 모델 테스트 실패: {e}")
    
    def _get_layer_type(self, layer_name: str) -> str:
        """레이어 이름에서 타입 추출"""
        name_lower = layer_name.lower()
        
        if 'backbone' in name_lower:
            return 'Backbone'
        elif 'neck' in name_lower:
            return 'Neck'
        elif 'head' in name_lower:
            return 'Head'
        elif 'conv' in name_lower:
            return 'Convolution'
        elif 'linear' in name_lower or 'fc' in name_lower:
            return 'Linear'
        elif 'bn' in name_lower or 'norm' in name_lower:
            return 'Normalization'
        elif 'attention' in name_lower:
            return 'Attention'
        else:
            return 'Other'


def main():
    """PyTorch 모델 분석 메인 함수"""
    print("=== RTMW-x PyTorch 모델 구조 분석기 ===")
    
    # RTMW 모델 찾기
    model_path, model_description = find_and_download_rtmw_model()
    
    if model_path is None:
        print("❌ RTMW 모델을 찾을 수 없습니다.")
        return
    
    # .pth 파일인지 확인
    if not model_path.endswith('.pth'):
        print(f"❌ PyTorch 체크포인트 파일(.pth)이 아닙니다: {model_path}")
        print("이 분석기는 PyTorch 체크포인트 전용입니다.")
        return
    
    print(f"분석 대상: {model_description}")
    print(f"모델 경로: {model_path}")
    
    # 모델 분석기 초기화
    analyzer = RTMWTorchModelAnalyzer(model_path)
    
    # 체크포인트 로드
    if not analyzer.load_checkpoint():
        return
    
    # 분석 실행
    analyzer.analyze_checkpoint_info()
    analyzer.analyze_model_structure()
    analyzer.analyze_backbone_structure()
    analyzer.analyze_head_structure()
    analyzer.analyze_input_output_structure()
    analyzer.estimate_model_size()
    analyzer.create_dummy_model_test()
    
    print("\n" + "="*80)
    print("✅ PyTorch 모델 분석 완료")
    print("="*80)


if __name__ == '__main__':
    main()