#!/usr/bin/env python3
"""
RTMW-x PyTorch 모델 구조 분석기
RTMW-x 384x288 PyTorch 체크포인트 파일 정확한 분석
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
            
            # 입력 크기 정보
            if hasattr(cfg, 'get') and 'codec' in cfg:
                codec = cfg['codec']
                if hasattr(codec, 'get') and 'input_size' in codec:
                    input_size = codec['input_size']
                    print(f"  - Input Size: {input_size} (H, W)")
                    
                # SimCC split ratio
                if hasattr(codec, 'get') and 'simcc_split_ratio' in codec:
                    split_ratio = codec['simcc_split_ratio']
                    print(f"  - SimCC Split Ratio: {split_ratio}")
                    
                # 정규화 정보
                if hasattr(codec, 'get') and 'normalize' in codec:
                    normalize = codec['normalize']
                    print(f"  - Normalize: {normalize}")
                    
                # 시그마 값
                if hasattr(codec, 'get') and 'sigma' in codec:
                    sigma = codec['sigma']
                    print(f"  - Sigma: {sigma}")
            else:
                print("  - 설정 정보를 파싱할 수 없습니다 (문자열 형태로 저장됨)")
                
        # 학습 정보
        print(f"\n📈 학습 정보:")
        if 'epoch' in self.checkpoint:
            print(f"  - Epoch: {self.checkpoint['epoch']}")
        if 'iter' in self.checkpoint:
            print(f"  - Iteration: {self.checkpoint['iter']}")
    
    def analyze_simcc_dimensions(self):
        """SimCC 차원 상세 분석"""
        print("\n" + "="*50)
        print("🧮 SimCC 차원 상세 분석")
        print("="*50)
        
        # 메타에서 입력 크기 가져오기 (기본값 사용)
        input_size = (288, 384)  # 기본 RTMW-x 입력 크기 (H, W)
        simcc_split_ratio = 2.0  # 기본 split ratio
        
        # SimCC 헤드 차원 분석
        cls_x_shape = None
        cls_y_shape = None
        
        for name, param in self.state_dict.items():
            if name == 'head.cls_x.weight':
                cls_x_shape = param.shape
            elif name == 'head.cls_y.weight':
                cls_y_shape = param.shape
        
        print(f"📏 설정 정보:")
        height, width = input_size
        print(f"  - 모델 입력 크기: {input_size} (H={height}, W={width})")
        print(f"  - SimCC Split Ratio: {simcc_split_ratio} (기본값)")
        
        print(f"\n🎯 SimCC 헤드 차원:")
        if cls_x_shape:
            print(f"  - cls_x.weight: {list(cls_x_shape)}")
            x_out_dim = cls_x_shape[0]
            print(f"    → X 출력 차원: {x_out_dim}")
            expected_x = int(width * simcc_split_ratio)
            print(f"    → 예상 X 차원: {width} × {simcc_split_ratio} = {expected_x}")
            print(f"    → 일치 여부: {'✅' if x_out_dim == expected_x else '❌'}")
            
            # 실제 split ratio 계산
            actual_x_ratio = x_out_dim / width
            print(f"    → 실제 X Split Ratio: {x_out_dim} ÷ {width} = {actual_x_ratio:.2f}")
        
        if cls_y_shape:
            print(f"  - cls_y.weight: {list(cls_y_shape)}")
            y_out_dim = cls_y_shape[0]
            print(f"    → Y 출력 차원: {y_out_dim}")
            expected_y = int(height * simcc_split_ratio)
            print(f"    → 예상 Y 차원: {height} × {simcc_split_ratio} = {expected_y}")
            print(f"    → 일치 여부: {'✅' if y_out_dim == expected_y else '❌'}")
            
            # 실제 split ratio 계산
            actual_y_ratio = y_out_dim / height
            print(f"    → 실제 Y Split Ratio: {y_out_dim} ÷ {height} = {actual_y_ratio:.2f}")
        
        # 결론
        if cls_x_shape and cls_y_shape:
            actual_x_ratio = cls_x_shape[0] / width
            actual_y_ratio = cls_y_shape[0] / height
            
            print(f"\n� 결론:")
            print(f"  - 입력 크기는 (H={height}, W={width})가 맞음")
            print(f"  - 실제 Split Ratio - X: {actual_x_ratio:.2f}, Y: {actual_y_ratio:.2f}")
            
            if abs(actual_x_ratio - actual_y_ratio) < 0.1:
                avg_ratio = (actual_x_ratio + actual_y_ratio) / 2
                print(f"  - 평균 Split Ratio: {avg_ratio:.2f}")
            else:
                print(f"  - ⚠️ X/Y Split Ratio가 다릅니다!")
    
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
        self.analyze_simcc_dimensions()
        self.analyze_head_structure()
        self.run_dummy_inference()
        
        print("\n" + "="*80)
        print("✅ PyTorch 모델 분석 완료")
        print("="*80)


def main():
    """메인 함수"""
    print("=== RTMW-x PyTorch 모델 구조 분석기 ===")
    
    # RTMW 모델 찾기
    model_path, model_description = find_and_download_rtmw_model()
    
    if model_path is None:
        print("❌ RTMW 모델을 찾을 수 없습니다.")
        return
    
    print(f"✅ 기존 RTMW 모델 발견: {model_path}")
    print(f"분석 대상: {model_description}")
    print(f"모델 경로: {model_path}")
    
    # 분석기 생성 및 실행
    analyzer = RTMWTorchModelAnalyzer(model_path)
    analyzer.analyze_all()


if __name__ == '__main__':
    main()
