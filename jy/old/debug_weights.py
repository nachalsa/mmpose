#!/usr/bin/env python3
"""가중치 로딩 디버깅"""

import torch
import numpy as np

def check_checkpoint_weights():
    """체크포인트 가중치 확인"""
    print("=== 체크포인트 가중치 분석 ===")
    
    # 체크포인트 로드
    checkpoint = torch.load('../models/rtmw-x_simcc-cocktail14_pt-ucoco_270e-384x288-f840f204_20231122.pth', 
                           map_location='cpu', weights_only=False)
    state_dict = checkpoint['state_dict']
    
    print(f"총 레이어 수: {len(state_dict)}")
    
    # 백본 가중치 확인
    backbone_keys = [k for k in state_dict.keys() if 'backbone' in k and 'weight' in k]
    print(f"\n백본 가중치 개수: {len(backbone_keys)}")
    for k in sorted(backbone_keys)[:10]:  # 처음 10개
        weight = state_dict[k]
        print(f"  - {k}: {weight.shape} (범위: {weight.min().item():.4f} ~ {weight.max().item():.4f})")
    
    # 헤드 가중치 확인
    head_keys = [k for k in state_dict.keys() if 'head' in k and 'weight' in k]
    print(f"\n헤드 가중치 개수: {len(head_keys)}")
    for k in sorted(head_keys):
        weight = state_dict[k]
        print(f"  - {k}: {weight.shape} (범위: {weight.min().item():.4f} ~ {weight.max().item():.4f})")
    
    # 넥 가중치 확인
    neck_keys = [k for k in state_dict.keys() if 'neck' in k and 'weight' in k]
    print(f"\n넥 가중치 개수: {len(neck_keys)}")
    for k in sorted(neck_keys)[:5]:  # 처음 5개
        weight = state_dict[k]
        print(f"  - {k}: {weight.shape} (범위: {weight.min().item():.4f} ~ {weight.max().item():.4f})")

def test_weight_loading_effect():
    """가중치 로딩 효과 테스트"""
    print("\n=== 가중치 로딩 효과 테스트 ===")
    
    from pose_estimator import RTMWXEstimator
    
    # 추정기 생성
    estimator = RTMWXEstimator('../models/rtmw-x_simcc-cocktail14_pt-ucoco_270e-384x288-f840f204_20231122.pth', device='cpu')
    
    # 더미 입력으로 3번 테스트
    dummy_input = torch.randn(1, 3, 384, 288)
    
    results = []
    for i in range(3):
        with torch.no_grad():
            cls_x, cls_y = estimator.model(dummy_input)
            
            # 첫 번째 키포인트의 예측 좌표
            x_pred = torch.argmax(torch.softmax(cls_x[0, 0], dim=0))
            y_pred = torch.argmax(torch.softmax(cls_y[0, 0], dim=0))
            
            results.append((x_pred.item(), y_pred.item()))
            print(f"테스트 {i+1}: 첫 번째 키포인트 예측 = ({x_pred.item()}, {y_pred.item()})")
    
    # 결과가 동일한지 확인
    all_same = all(r == results[0] for r in results)
    print(f"모든 결과가 동일한가: {all_same}")
    
    if not all_same:
        print("⚠️ 문제: 모델이 랜덤한 결과를 생성하고 있습니다 (가중치가 제대로 로드되지 않음)")
    else:
        print("✅ 모델이 일관된 결과를 생성합니다")

if __name__ == "__main__":
    check_checkpoint_weights()
    test_weight_loading_effect()
