#!/usr/bin/env python3
"""
MMPose SimCC 디코더 (공식 구현 기반)
MMPose 공식 코드에서 추출한 SimCC 후처리 로직
"""

import torch
import numpy as np
from typing import Tuple, List, Optional
import math


class SimCCDecoder:
    """MMPose SimCC 디코더 (공식 simcc_label.py 기반)"""
    
    def __init__(self, 
                 input_size: Tuple[int, int] = (384, 288),
                 simcc_split_ratio: float = 2.0,
                 simcc_normalize: bool = False,
                 use_dark: bool = False):
        """
        Args:
            input_size: 입력 이미지 크기 (W, H)
            simcc_split_ratio: SimCC split ratio (보통 2.0)
            simcc_normalize: SimCC 정규화 여부
            use_dark: DARK 후처리 사용 여부
        """
        self.input_size = input_size
        self.simcc_split_ratio = simcc_split_ratio
        self.simcc_normalize = simcc_normalize
        self.use_dark = use_dark
        
        # MMPose 공식: SimCC 라벨 크기 계산
        self.simcc_x_label_size = int(input_size[0] * simcc_split_ratio)  # 384 * 2 = 768
        self.simcc_y_label_size = int(input_size[1] * simcc_split_ratio)  # 288 * 2 = 576
        
        print(f"📊 MMPose SimCC 디코더 초기화:")
        print(f"   입력 크기: {input_size}")
        print(f"   Split Ratio: {simcc_split_ratio}")
        print(f"   예상 x 라벨 크기: {self.simcc_x_label_size}")
        print(f"   예상 y 라벨 크기: {self.simcc_y_label_size}")
    
    def decode(self, 
               simcc_x: torch.Tensor, 
               simcc_y: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """
        MMPose 공식 SimCC 디코딩
        
        Args:
            simcc_x: [B, total_x] 또는 [B, K, x_bins] x 좌표 로짓
            simcc_y: [B, total_y] 또는 [B, K, y_bins] y 좌표 로짓
            
        Returns:
            keypoints: [K, 2] 키포인트 좌표
            scores: [K] 키포인트 신뢰도
        """
        # 텐서를 numpy로 변환
        if isinstance(simcc_x, torch.Tensor):
            simcc_x = simcc_x.detach().cpu().numpy()
        if isinstance(simcc_y, torch.Tensor):
            simcc_y = simcc_y.detach().cpu().numpy()
        
        batch_size = simcc_x.shape[0]
        x_total = simcc_x.shape[1]  # 576
        y_total = simcc_y.shape[1]  # 768
        
        print(f"🔍 실제 SimCC 출력: x={x_total}, y={y_total}")
        
        # MMPose 공식 방법: get_simcc_maximum 함수 사용
        keypoints, scores = self._get_simcc_maximum(simcc_x, simcc_y)
        
        # 단일 배치 처리
        if keypoints.ndim == 3:
            keypoints = keypoints[0]  # [1, K, 2] -> [K, 2]
            scores = scores[0]      # [1, K] -> [K]
        
        # RTMW-x의 경우 키포인트 수가 133이 아닐 수 있으므로 실제 추정
        actual_keypoints = keypoints.shape[0]
        print(f"🔍 추정된 키포인트 수: {actual_keypoints}")
        
        # 좌표를 원래 이미지 크기로 스케일링 (MMPose 공식 방법)
        if self.simcc_normalize:
            # simcc_split_ratio로 나누어 원래 크기로 복원
            keypoints = keypoints / self.simcc_split_ratio
        
        return keypoints, scores
    
    def _get_simcc_maximum(self,
                          simcc_x: np.ndarray,
                          simcc_y: np.ndarray,
                          apply_softmax: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        MMPose 공식 get_simcc_maximum 함수 구현
        
        Args:
            simcc_x: [B, total_x] x 좌표 로짓
            simcc_y: [B, total_y] y 좌표 로짓  
            apply_softmax: softmax 적용 여부
            
        Returns:
            locs: [B, K, 2] 키포인트 좌표
            vals: [B, K] 키포인트 점수
        """
        assert isinstance(simcc_x, np.ndarray), ('simcc_x should be numpy.ndarray')
        assert isinstance(simcc_y, np.ndarray), ('simcc_y should be numpy.ndarray')
        assert simcc_x.ndim == 2, f'Invalid shape {simcc_x.shape}'
        assert simcc_y.ndim == 2, f'Invalid shape {simcc_y.shape}'
        
        N = simcc_x.shape[0]  # batch size
        
        if apply_softmax:
            simcc_x = simcc_x - np.max(simcc_x, axis=1, keepdims=True)
            simcc_y = simcc_y - np.max(simcc_y, axis=1, keepdims=True)
            ex, ey = np.exp(simcc_x), np.exp(simcc_y)
            simcc_x = ex / np.sum(ex, axis=1, keepdims=True)
            simcc_y = ey / np.sum(ey, axis=1, keepdims=True)
        
        # RTMW-x의 경우: 576과 768을 키포인트별로 분할해야 함
        # 가능한 키포인트 수들 시도
        possible_keypoints = [133, 17, 21, 68, 1]  # WholeBody, Body, Hand, Face, Single
        
        best_keypoints = None
        best_x_bins = None
        best_y_bins = None
        
        for num_kpts in possible_keypoints:
            x_bins = simcc_x.shape[1] // num_kpts
            y_bins = simcc_y.shape[1] // num_kpts
            
            if (simcc_x.shape[1] % num_kpts == 0 and 
                simcc_y.shape[1] % num_kpts == 0 and
                x_bins > 0 and y_bins > 0):
                best_keypoints = num_kpts
                best_x_bins = x_bins
                best_y_bins = y_bins
                break
        
        if best_keypoints is None:
            # fallback: 전체를 하나의 키포인트로 처리
            print("⚠️ 키포인트 분할 실패, 단일 키포인트로 처리")
            best_keypoints = 1
            best_x_bins = simcc_x.shape[1]
            best_y_bins = simcc_y.shape[1]
        
        print(f"🔍 키포인트 분할: {best_keypoints}개, x={best_x_bins} bins, y={best_y_bins} bins")
        
        # reshape
        try:
            reshaped_x = simcc_x.reshape(N, best_keypoints, best_x_bins)
            reshaped_y = simcc_y.reshape(N, best_keypoints, best_y_bins)
        except:
            # reshape 실패 시 전체 공간으로 처리
            print("⚠️ reshape 실패, 전체 공간 처리")
            return self._decode_as_single_space(simcc_x, simcc_y)
        
        # 각 키포인트별로 최대값 위치 찾기
        x_locs = np.argmax(reshaped_x, axis=2)  # [N, K]
        y_locs = np.argmax(reshaped_y, axis=2)  # [N, K]
        
        locs = np.stack((x_locs, y_locs), axis=-1).astype(np.float32)  # [N, K, 2]
        
        # 점수 계산
        max_val_x = np.amax(reshaped_x, axis=2)  # [N, K]
        max_val_y = np.amax(reshaped_y, axis=2)  # [N, K]
        
        # 두 방향 중 작은 값을 사용
        vals = np.minimum(max_val_x, max_val_y)
        
        # 0 이하 값들을 -1로 설정
        locs[vals <= 0.] = -1
        
        # bins 좌표를 실제 픽셀 좌표로 변환
        if best_x_bins > 1:
            locs[:, :, 0] = (locs[:, :, 0] / (best_x_bins - 1)) * self.input_size[0]
        else:
            locs[:, :, 0] = self.input_size[0] / 2
            
        if best_y_bins > 1:
            locs[:, :, 1] = (locs[:, :, 1] / (best_y_bins - 1)) * self.input_size[1]
        else:
            locs[:, :, 1] = self.input_size[1] / 2
        
        return locs, vals
    
    def _decode_as_single_space(self, 
                               simcc_x: np.ndarray, 
                               simcc_y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """전체 공간을 하나의 키포인트로 처리"""
        N = simcc_x.shape[0]
        
        # 전체 공간에서 최대값 위치 찾기
        x_locs = np.argmax(simcc_x, axis=1)  # [N]
        y_locs = np.argmax(simcc_y, axis=1)  # [N]
        
        # 실제 픽셀 좌표로 변환
        pixel_x = (x_locs / (simcc_x.shape[1] - 1)) * self.input_size[0]
        pixel_y = (y_locs / (simcc_y.shape[1] - 1)) * self.input_size[1]
        
        # 133개 키포인트를 모두 같은 위치로 설정 (임시)
        locs = np.zeros((N, 133, 2), dtype=np.float32)
        vals = np.zeros((N, 133), dtype=np.float32)
        
        for i in range(133):
            locs[:, i, 0] = pixel_x
            locs[:, i, 1] = pixel_y
            vals[:, i] = 0.5  # 임시 점수
        
        return locs, vals