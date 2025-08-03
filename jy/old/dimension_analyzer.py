#!/usr/bin/env python3
"""
RTMW-x 모델 차원 분석기 - 정확한 입력 크기 해석
"""

import torch
import os
from config import *

def analyze_model_dimensions():
    """모델 차원을 정확히 분석하여 올바른 입력 크기와 split ratio 찾기"""
    
    model_path = "./models/rtmw-x_simcc-cocktail14_pt-ucoco_270e-384x288-f840f204_20231122.pth"
    
    print("=" * 80)
    print("🔍 RTMW-x 모델 차원 분석기")
    print("=" * 80)
    
    # 체크포인트 로드
    print(f"📁 모델 로드: {model_path}")
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint['state_dict']
    
    # SimCC 헤드 차원 찾기
    cls_x_weight = None
    cls_y_weight = None
    
    for key in state_dict.keys():
        if 'cls_x.weight' in key:
            cls_x_weight = state_dict[key]
        elif 'cls_y.weight' in key:
            cls_y_weight = state_dict[key]
    
    if cls_x_weight is None or cls_y_weight is None:
        print("❌ SimCC 헤드를 찾을 수 없습니다!")
        return
    
    x_output_dim = cls_x_weight.shape[0]  # 576
    y_output_dim = cls_y_weight.shape[0]  # 768
    
    print(f"🎯 SimCC 헤드 출력 차원:")
    print(f"  - cls_x 출력: {x_output_dim}")
    print(f"  - cls_y 출력: {y_output_dim}")
    print()
    
    # 파일명에서 크기 추출: rtmw-x_simcc-cocktail14_pt-ucoco_270e-384x288
    # 일반적으로 이미지에서 WxH 순서
    filename_w, filename_h = 384, 288
    
    print("🔢 가능한 입력 크기 해석:")
    print("-" * 50)
    
    # 가능한 해석들
    scenarios = [
        ("시나리오 1: 파일명 384x288 = (W=384, H=288)", 384, 288),
        ("시나리오 2: 파일명 384x288 = (W=288, H=384)", 288, 384),
        ("시나리오 3: 현재 설정 (W=384, H=288)", RTMW_INPUT_SIZE[1], RTMW_INPUT_SIZE[0]),
    ]
    
    best_scenario = None
    min_diff = float('inf')
    
    for i, (desc, w, h) in enumerate(scenarios, 1):
        print(f"\n{desc}")
        print(f"  입력 크기: W={w}, H={h}")
        
        # X는 width 방향, Y는 height 방향
        # SimCC에서 보통 x좌표는 width와 관련, y좌표는 height와 관련
        x_ratio = x_output_dim / w
        y_ratio = y_output_dim / h
        
        print(f"  X Split Ratio: {x_output_dim}/{w} = {x_ratio:.3f}")
        print(f"  Y Split Ratio: {y_output_dim}/{h} = {y_ratio:.3f}")
        
        ratio_diff = abs(x_ratio - y_ratio)
        print(f"  비율 차이: {ratio_diff:.3f}")
        
        if ratio_diff < min_diff:
            min_diff = ratio_diff
            best_scenario = (desc, w, h, x_ratio, y_ratio)
        
        # 판정
        if ratio_diff < 0.1:
            print(f"  판정: ✅ 균등한 Split Ratio (우수)")
        elif ratio_diff < 0.5:
            print(f"  판정: ⚠️ 약간 다른 Split Ratio (보통)")
        else:
            print(f"  판정: ❌ 매우 다른 Split Ratio (부적절)")
    
    print("\n" + "=" * 80)
    print("🏆 최종 결론")
    print("=" * 80)
    
    if best_scenario:
        desc, w, h, x_ratio, y_ratio = best_scenario
        avg_ratio = (x_ratio + y_ratio) / 2
        
        print(f"✅ 최적 해석: {desc}")
        print(f"📐 권장 설정:")
        print(f"  - 입력 크기: (H={h}, W={w})")
        print(f"  - X Split Ratio: {x_ratio:.3f}")
        print(f"  - Y Split Ratio: {y_ratio:.3f}")
        print(f"  - 평균 Split Ratio: {avg_ratio:.3f}")
        
        # 정수 근사값 확인
        if abs(avg_ratio - 1.5) < 0.1:
            print(f"  - 권장 Split Ratio: 1.5")
        elif abs(avg_ratio - 2.0) < 0.1:
            print(f"  - 권장 Split Ratio: 2.0")
        elif abs(avg_ratio - 2.5) < 0.1:
            print(f"  - 권장 Split Ratio: 2.5")
        else:
            print(f"  - 권장 Split Ratio: {avg_ratio:.1f}")
        
        # 실제 계산 검증
        print(f"\n🧮 검증:")
        print(f"  - X: {w} × {x_ratio:.3f} = {w * x_ratio:.1f} (실제: {x_output_dim})")
        print(f"  - Y: {h} × {y_ratio:.3f} = {h * y_ratio:.1f} (실제: {y_output_dim})")
        
        return w, h, x_ratio, y_ratio
    
    print("❌ 적절한 해석을 찾을 수 없습니다!")
    return None

if __name__ == "__main__":
    result = analyze_model_dimensions()
