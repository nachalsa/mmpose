#!/usr/bin/env python3
"""간단한 좌표 변환 테스트"""

import torch
import numpy as np
from pose_estimator import RTMWXEstimator

def simple_coordinate_test():
    """간단한 좌표 변환 테스트"""
    print("=== 간단한 좌표 변환 테스트 ===")
    
    # 모델 로드
    model_path = "../models/rtmw-x_simcc-cocktail14_pt-ucoco_270e-384x288-f840f204_20231122.pth"
    estimator = RTMWXEstimator(model_path, device='cpu')
    
    print(f"✅ 모델 로드 완료")
    
    # 더미 이미지와 바운딩박스
    dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    dummy_bbox = [100, 50, 300, 400]  # [x1, y1, x2, y2]
    
    print(f"📐 입력:")
    print(f"  - 이미지: {dummy_image.shape}")
    print(f"  - 바운딩박스: {dummy_bbox}")
    
    # 포즈 추정
    keypoints = estimator.estimate_pose(dummy_image, dummy_bbox)
    
    print(f"\n📊 결과:")
    print(f"  - 키포인트 shape: {keypoints.shape}")
    print(f"  - X 좌표 범위: [{keypoints[:, 0].min():.1f}, {keypoints[:, 0].max():.1f}]")
    print(f"  - Y 좌표 범위: [{keypoints[:, 1].min():.1f}, {keypoints[:, 1].max():.1f}]")
    
    # 유효한 좌표 개수
    valid_count = np.sum((keypoints[:, 0] > 0) & (keypoints[:, 1] > 0))
    print(f"  - 유효한 키포인트: {valid_count}/133")
    
    # 바운딩박스 범위 체크
    x1, y1, x2, y2 = dummy_bbox
    in_bbox_x = np.sum((keypoints[:, 0] >= x1) & (keypoints[:, 0] <= x2))
    in_bbox_y = np.sum((keypoints[:, 1] >= y1) & (keypoints[:, 1] <= y2))
    
    print(f"  - 바운딩박스 내 X 좌표: {in_bbox_x}/133")
    print(f"  - 바운딩박스 내 Y 좌표: {in_bbox_y}/133")
    
    # 몇 개 샘플 출력
    print(f"\n📍 키포인트 샘플 (처음 10개):")
    for i in range(min(10, len(keypoints))):
        x, y = keypoints[i, 0], keypoints[i, 1]
        in_bbox = (x >= x1 and x <= x2 and y >= y1 and y <= y2)
        print(f"  [{i:2d}] ({x:6.1f}, {y:6.1f}) {'✓' if in_bbox else '✗'}")

if __name__ == '__main__':
    simple_coordinate_test()
