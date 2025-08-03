#!/usr/bin/env python3
"""좌표 변환 디버깅"""

import torch
import numpy as np
import cv2
from pose_estimator import RTMWXEstimator
from mmpose.structures.bbox import get_warp_matrix

def debug_coordinate_transformation():
    """좌표 변환 과정 단계별 디버깅"""
    print("=== 좌표 변환 디버깅 ===")
    
    # 실제 모델 로드
    model_path = "../models/rtmw-x_simcc-cocktail14_pt-ucoco_270e-384x288-f840f204_20231122.pth"
    estimator = RTMWXEstimator(model_path, device='cpu')
    
    # 테스트 이미지와 바운딩박스
    dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    dummy_bbox = [100, 50, 300, 400]  # [x1, y1, x2, y2]
    
    print(f"원본 이미지: {dummy_image.shape}")
    print(f"바운딩박스: {dummy_bbox}")
    
    # 단계 1: 바운딩박스 전처리
    x1, y1, x2, y2 = dummy_bbox
    w_bbox = x2 - x1
    h_bbox = y2 - y1
    
    # MMPose 공식: 센터와 스케일 계산
    center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
    scale = np.array([w_bbox, h_bbox]) * 1.25  # MMPose 패딩
    
    print(f"\n단계 1 - 바운딩박스 전처리:")
    print(f"  바운딩박스 크기: {w_bbox} x {h_bbox}")
    print(f"  센터: {center}")
    print(f"  스케일: {scale}")
    
    # 단계 2: topdown_affine 변환 매트릭스
    w, h = estimator.input_size  # (288, 384)
    warp_mat = get_warp_matrix(
        center=center,
        scale=scale,
        rot=0,
        output_size=(w, h),
        inv=False  # 순변환
    )
    
    print(f"\n단계 2 - 변환 매트릭스:")
    print(f"  타겟 크기: {w} x {h}")
    print(f"  변환 매트릭스:\n{warp_mat}")
    
    # 단계 3: 이미지 전처리
    cropped_image = estimator._apply_topdown_affine(
        dummy_image, center, scale
    )
    
    print(f"\n단계 3 - 이미지 전처리:")
    print(f"  크롭된 이미지: {cropped_image.shape}")
    
    # 단계 4: 모델 추론 (내부 로직 접근)
    # 이미지를 텐서로 변환
    input_tensor = torch.from_numpy(cropped_image).float()
    input_tensor = input_tensor.permute(2, 0, 1)  # HWC -> CHW
    input_tensor = input_tensor.unsqueeze(0)  # 배치 차원 추가
    input_tensor = input_tensor / 255.0  # 정규화
    
    print(f"\n단계 4 - 모델 입력:")
    print(f"  입력 텐서: {input_tensor.shape}")
    print(f"  입력 범위: [{input_tensor.min():.3f}, {input_tensor.max():.3f}]")
    
    # 모델 추론
    with torch.no_grad():
        outputs = estimator.model(input_tensor)
        simcc_x = outputs[0]  # [1, 133, 576]
        simcc_y = outputs[1]  # [1, 133, 768]
    
    print(f"\n단계 5 - 모델 출력:")
    print(f"  SimCC X: {simcc_x.shape}")
    print(f"  SimCC Y: {simcc_y.shape}")
    print(f"  SimCC X 범위: [{simcc_x.min():.3f}, {simcc_x.max():.3f}]")
    print(f"  SimCC Y 범위: [{simcc_y.min():.3f}, {simcc_y.max():.3f}]")
    
    # 단계 6: SimCC 디코딩
    keypoints_batch = estimator.simcc_decoder.decode_simcc_outputs(simcc_x, simcc_y)
    keypoints_model = keypoints_batch[0, :, :2].cpu().numpy()  # [133, 2]
    
    print(f"\n단계 6 - SimCC 디코딩:")
    print(f"  디코딩된 키포인트: {keypoints_model.shape}")
    print(f"  모델 좌표 X 범위: [{keypoints_model[:, 0].min():.1f}, {keypoints_model[:, 0].max():.1f}]")
    print(f"  모델 좌표 Y 범위: [{keypoints_model[:, 1].min():.1f}, {keypoints_model[:, 1].max():.1f}]")
    print(f"  모델 입력 크기: {w} x {h}")
    
    # 단계 7: 역변환 매트릭스
    inv_warp_mat = get_warp_matrix(
        center=center,
        scale=scale,
        rot=0,
        output_size=(w, h),
        inv=True  # 역변환
    )
    
    print(f"\n단계 7 - 역변환:")
    print(f"  역변환 매트릭스:\n{inv_warp_mat}")
    
    # 단계 8: 좌표 역변환
    keypoints_homogeneous = np.hstack([keypoints_model, np.ones((keypoints_model.shape[0], 1))])
    transformed_keypoints = keypoints_homogeneous @ inv_warp_mat.T
    
    print(f"\n단계 8 - 좌표 역변환:")
    print(f"  변환 전 좌표 (샘플 5개):")
    for i in range(5):
        print(f"    {i}: ({keypoints_model[i, 0]:.1f}, {keypoints_model[i, 1]:.1f})")
    
    print(f"  변환 후 좌표 (샘플 5개):")
    for i in range(5):
        print(f"    {i}: ({transformed_keypoints[i, 0]:.1f}, {transformed_keypoints[i, 1]:.1f})")
    
    print(f"  최종 X 범위: [{transformed_keypoints[:, 0].min():.1f}, {transformed_keypoints[:, 0].max():.1f}]")
    print(f"  최종 Y 범위: [{transformed_keypoints[:, 1].min():.1f}, {transformed_keypoints[:, 1].max():.1f}]")
    print(f"  원본 이미지 크기: {dummy_image.shape[1]} x {dummy_image.shape[0]}")
    
    # 좌표가 이미지 범위를 벗어나는지 확인
    h_orig, w_orig = dummy_image.shape[:2]
    out_of_bounds_x = np.sum((transformed_keypoints[:, 0] < 0) | (transformed_keypoints[:, 0] >= w_orig))
    out_of_bounds_y = np.sum((transformed_keypoints[:, 1] < 0) | (transformed_keypoints[:, 1] >= h_orig))
    
    print(f"\n범위 벗어난 좌표:")
    print(f"  X 좌표 범위 초과: {out_of_bounds_x}/133")
    print(f"  Y 좌표 범위 초과: {out_of_bounds_y}/133")
    
    # SimCC 디코더 설정 확인
    print(f"\nSimCC 디코더 설정:")
    print(f"  input_size: {estimator.simcc_decoder.input_size}")
    print(f"  simcc_split_ratio: {estimator.simcc_decoder.simcc_split_ratio}")
    print(f"  simcc_x_dim: {estimator.simcc_decoder.simcc_x_dim}")
    print(f"  simcc_y_dim: {estimator.simcc_decoder.simcc_y_dim}")

if __name__ == '__main__':
    debug_coordinate_transformation()
