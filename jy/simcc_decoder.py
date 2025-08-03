#!/usr/bin/env python3
"""
SimCC 디코더 모듈 - RTMW-x용
Coordinate Classification 방식으로 키포인트 좌표를 디코딩하는 모듈
"""

import torch
import numpy as np
from typing import Tuple, Union, Optional
from config import RTMW_INPUT_SIZE, RTMW_NUM_KEYPOINTS, RTMW_SIMCC_SPLIT_RATIO

class RTMWSimCCDecoder:
    """RTMW-x용 SimCC 디코더 - 올바른 차원으로 수정"""
    
    def __init__(self, 
                 input_size: Tuple[int, int] = RTMW_INPUT_SIZE,
                 num_keypoints: int = RTMW_NUM_KEYPOINTS,
                 simcc_split_ratio: float = RTMW_SIMCC_SPLIT_RATIO):
        """
        Args:
            input_size: 모델 입력 크기 (W, H) = (288, 384)
            num_keypoints: 키포인트 수 (133)
            simcc_split_ratio: SimCC split ratio (2.0, X/Y 공통)
        """
        self.input_width, self.input_height = input_size  # W=288, H=384
        self.num_keypoints = num_keypoints
        self.simcc_split_ratio = simcc_split_ratio
        
        # SimCC 차원 계산 - 올바른 공식 (W=288, H=384)
        self.simcc_x_dim = int(self.input_width * simcc_split_ratio)   # 288 * 2.0 = 576
        self.simcc_y_dim = int(self.input_height * simcc_split_ratio)  # 384 * 2.0 = 768
        
        print(f"🧮 RTMWSimCCDecoder 초기화 (수정됨):")
        print(f"  - 입력 크기: {input_size} (H={self.input_height}, W={self.input_width})")
        print(f"  - 키포인트 수: {num_keypoints}")
        print(f"  - SimCC Split Ratio: {simcc_split_ratio} (X/Y 공통)")
        print(f"  - SimCC X 차원: {self.simcc_x_dim} (W={self.input_width} × {simcc_split_ratio})")
        print(f"  - SimCC Y 차원: {self.simcc_y_dim} (H={self.input_height} × {simcc_split_ratio})")
        
    def decode_simcc_outputs(self, 
                           simcc_x: torch.Tensor, 
                           simcc_y: torch.Tensor) -> torch.Tensor:
        """
        SimCC 출력을 키포인트 좌표로 디코딩
        
        Args:
            simcc_x: X 좌표 분류 결과 [batch_size, num_keypoints, simcc_x_dim]
            simcc_y: Y 좌표 분류 결과 [batch_size, num_keypoints, simcc_y_dim]
            
        Returns:
            keypoints: 키포인트 좌표 [batch_size, num_keypoints, 3] (x, y, score)
        """
        batch_size = simcc_x.shape[0]
        device = simcc_x.device
        
        # 입력 차원 검증
        expected_x_shape = (batch_size, self.num_keypoints, self.simcc_x_dim)
        expected_y_shape = (batch_size, self.num_keypoints, self.simcc_y_dim)
        
        if simcc_x.shape != expected_x_shape:
            raise ValueError(f"SimCC X 차원 불일치: 예상 {expected_x_shape}, 실제 {simcc_x.shape}")
        if simcc_y.shape != expected_y_shape:
            raise ValueError(f"SimCC Y 차원 불일치: 예상 {expected_y_shape}, 실제 {simcc_y.shape}")
        
        # Softmax 적용하여 확률분포로 변환
        simcc_x_prob = torch.softmax(simcc_x, dim=-1)  # [B, K, 576]
        simcc_y_prob = torch.softmax(simcc_y, dim=-1)  # [B, K, 768]
        
        # 최대 확률의 인덱스 찾기
        x_indices = torch.argmax(simcc_x_prob, dim=-1)  # [B, K]
        y_indices = torch.argmax(simcc_y_prob, dim=-1)  # [B, K]
        
        # 최대 확률값 (신뢰도)
        x_scores = torch.max(simcc_x_prob, dim=-1)[0]  # [B, K]
        y_scores = torch.max(simcc_y_prob, dim=-1)[0]  # [B, K]
        
        # 인덱스를 실제 좌표로 변환 (MMPose 공식 방식)
        # MMPose: 인덱스를 float로 변환 후 마지막에 simcc_split_ratio로 나누기
        x_coords = x_indices.float()  # [B, K]
        y_coords = y_indices.float()  # [B, K]
        
        # MMPose 공식: 최종적으로 simcc_split_ratio로 나누기
        x_coords = x_coords / self.simcc_split_ratio  # [B, K]
        y_coords = y_coords / self.simcc_split_ratio  # [B, K]
        
        # 전체 신뢰도 계산 (MMPose 공식: min 방식)
        joint_scores = torch.minimum(x_scores, y_scores)  # [B, K]
        
        # 최종 키포인트 좌표 구성 [B, K, 3]
        keypoints = torch.stack([x_coords, y_coords, joint_scores], dim=-1)
        
        print(f"🔍 디코딩 결과:")
        print(f"   - X 좌표 범위: [{x_coords.min().item():.1f}, {x_coords.max().item():.1f}]")
        print(f"   - Y 좌표 범위: [{y_coords.min().item():.1f}, {y_coords.max().item():.1f}]")
        print(f"   - 신뢰도 범위: [{joint_scores.min().item():.3f}, {joint_scores.max().item():.3f}]")
        
        return keypoints

def test_simcc_decoder():
    """SimCC 디코더 테스트"""
    print("🧪 SimCC 디코더 테스트")
    print("=" * 50)
    
    # 디코더 초기화
    decoder = RTMWSimCCDecoder()
    
    # 더미 데이터 생성
    batch_size = 2
    device = torch.device('cpu')
    
    # SimCC 출력 시뮬레이션 (정규분포로 생성)
    simcc_x = torch.randn(batch_size, RTMW_NUM_KEYPOINTS, decoder.simcc_x_dim)
    simcc_y = torch.randn(batch_size, RTMW_NUM_KEYPOINTS, decoder.simcc_y_dim)
    
    print(f"입력 차원:")
    print(f"  - simcc_x: {simcc_x.shape}")
    print(f"  - simcc_y: {simcc_y.shape}")
    
    # 디코딩 테스트
    try:
        keypoints = decoder.decode_simcc_outputs(simcc_x, simcc_y)
        print(f"✅ 디코딩 성공!")
        print(f"  - 출력 차원: {keypoints.shape}")
        print(f"  - 첫 번째 키포인트 좌표: {keypoints[0, 0]}")
        
        # 좌표 범위 확인
        x_coords = keypoints[:, :, 0]
        y_coords = keypoints[:, :, 1]
        scores = keypoints[:, :, 2]
        
        print(f"좌표 범위:")
        print(f"  - X: [{x_coords.min().item():.2f}, {x_coords.max().item():.2f}]")
        print(f"  - Y: [{y_coords.min().item():.2f}, {y_coords.max().item():.2f}]")
        print(f"  - Score: [{scores.min().item():.3f}, {scores.max().item():.3f}]")
        
        return True
        
    except Exception as e:
        print(f"❌ 디코딩 실패: {e}")
        return False


if __name__ == "__main__":
    test_simcc_decoder()
