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
        
        # 메타 정보
        if 'meta' in self.checkpoint:
            meta = self.checkpoint['meta']
            print(f"\n🔍 메타 정보:")
            for key, value in meta.items():
                print(f"  - {key}: {value}")
        
        # 학습 정보
        if 'epoch' in self.checkpoint:
            print(f"\n📈 학습 정보:")
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
            if 'weight' in [t[0].split('.')[-1] for t in info['tensors']]:
                for tensor_name, shape in info['tensors']:
                    if tensor_name.endswith('.weight'):
                        print(f"    - Weight Shape: {list(shape)}")
                        if len(shape) >= 2:
                            print(f"    - 출력 채널: {shape[0]}")
                            print(f"    - 입력 채널: {shape[1] if len(shape) > 1 else 'N/A'}")
    
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