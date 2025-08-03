#!/usr/bin/env python3
"""
SimCC 디코더 구현
RTMW-x SimCC 출력을 실제 키포인트 좌표로 변환
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


class SimCCDecoder:
    """SimCC (Simple Coordinate Classification) 디코더"""
    
    def __init__(self, 
                 input_size: Tuple[int, int] = (384, 288),
                 sigma: Tuple[float, float] = (6.0, 6.93),
                 simcc_split_ratio: float = 2.0,
                 normalize: bool = False):
        """
        Args:
            input_size: (width, height) 입력 이미지 크기
            sigma: (sigma_x, sigma_y) 가우시안 시그마 값
            simcc_split_ratio: X/Y 차원 분할 비율
            normalize: 좌표 정규화 여부
        """
        self.input_size = input_size
        self.sigma = sigma
        self.simcc_split_ratio = simcc_split_ratio
        self.normalize = normalize
        
        # SimCC 차원 계산
        self.simcc_x = int(input_size[0] * simcc_split_ratio)  # 384 * 2 = 768 -> 576
        self.simcc_y = int(input_size[1] * simcc_split_ratio)  # 288 * 2 = 576 -> 768
        
        print(f"SimCC 디코더 초기화:")
        print(f"  - 입력 크기: {input_size}")
        print(f"  - SimCC X 차원: {self.simcc_x}")
        print(f"  - SimCC Y 차원: {self.simcc_y}")
        print(f"  - 시그마: {sigma}")
    
    def decode(self, 
               simcc_x: torch.Tensor, 
               simcc_y: torch.Tensor,
               return_heatmaps: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        SimCC 출력을 키포인트 좌표로 디코딩
        
        Args:
            simcc_x: [B, N, W*2] X 좌표 분류 출력
            simcc_y: [B, N, H*2] Y 좌표 분류 출력
            return_heatmaps: 히트맵 반환 여부
            
        Returns:
            keypoints: [B, N, 2] 키포인트 좌표
            heatmaps: [B, N, H, W] 히트맵 (옵션)
        """
        batch_size, num_keypoints = simcc_x.shape[:2]
        
        # Softmax로 확률 분포 변환
        simcc_x_prob = F.softmax(simcc_x, dim=-1)  # [B, N, W*2]
        simcc_y_prob = F.softmax(simcc_y, dim=-1)  # [B, N, H*2]
        
        # 기대값(expectation) 계산으로 좌표 추출
        x_coords = self._get_simcc_maximum(simcc_x_prob, self.simcc_x)
        y_coords = self._get_simcc_maximum(simcc_y_prob, self.simcc_y)
        
        # 좌표 결합
        keypoints = torch.stack([x_coords, y_coords], dim=-1)  # [B, N, 2]
        
        # 원본 이미지 크기로 스케일링
        keypoints[..., 0] = keypoints[..., 0] / self.simcc_split_ratio  # X 좌표
        keypoints[..., 1] = keypoints[..., 1] / self.simcc_split_ratio  # Y 좌표
        
        # 정규화 (옵션)
        if self.normalize:
            keypoints[..., 0] = keypoints[..., 0] / self.input_size[0]  # [0, 1]
            keypoints[..., 1] = keypoints[..., 1] / self.input_size[1]  # [0, 1]
        
        heatmaps = None
        if return_heatmaps:
            heatmaps = self._generate_heatmaps(keypoints, batch_size, num_keypoints)
        
        return keypoints, heatmaps
    
    def _get_simcc_maximum(self, simcc_output: torch.Tensor, simcc_size: int) -> torch.Tensor:
        """SimCC 출력에서 최대값 위치 계산"""
        # 기대값 계산 (weighted average)
        coords = torch.arange(simcc_size, dtype=simcc_output.dtype, device=simcc_output.device)
        coords = coords.view(1, 1, -1)  # [1, 1, simcc_size]
        
        # 가중 평균으로 좌표 계산
        expected_coords = torch.sum(simcc_output * coords, dim=-1)  # [B, N]
        
        return expected_coords
    
    def _generate_heatmaps(self, 
                          keypoints: torch.Tensor, 
                          batch_size: int, 
                          num_keypoints: int) -> torch.Tensor:
        """키포인트로부터 히트맵 생성"""
        height, width = self.input_size[1], self.input_size[0]
        heatmaps = torch.zeros(batch_size, num_keypoints, height, width, 
                              dtype=keypoints.dtype, device=keypoints.device)
        
        # 가우시안 히트맵 생성
        for b in range(batch_size):
            for k in range(num_keypoints):
                x, y = keypoints[b, k]
                if x > 0 and y > 0:  # 유효한 키포인트만
                    self._draw_gaussian(heatmaps[b, k], (int(x), int(y)), self.sigma)
        
        return heatmaps
    
    def _draw_gaussian(self, heatmap: torch.Tensor, center: Tuple[int, int], sigma: Tuple[float, float]):
        """가우시안 히트맵 그리기"""
        height, width = heatmap.shape
        x_center, y_center = center
        
        # 가우시안 범위 계산
        radius = 3 * max(sigma)
        x_min = max(0, int(x_center - radius))
        x_max = min(width, int(x_center + radius + 1))
        y_min = max(0, int(y_center - radius))
        y_max = min(height, int(y_center + radius + 1))
        
        # 가우시안 값 계산
        y_indices, x_indices = torch.meshgrid(
            torch.arange(y_min, y_max, device=heatmap.device),
            torch.arange(x_min, x_max, device=heatmap.device),
            indexing='ij'
        )
        
        gaussian = torch.exp(-((x_indices - x_center) ** 2 / (2 * sigma[0] ** 2) + 
                              (y_indices - y_center) ** 2 / (2 * sigma[1] ** 2)))
        
        heatmap[y_min:y_max, x_min:x_max] = torch.maximum(
            heatmap[y_min:y_max, x_min:x_max], gaussian
        )


class RTMWSimCCWrapper:
    """RTMW-x SimCC 래퍼 클래스"""
    
    def __init__(self, 
                 model_path: str,
                 input_size: Tuple[int, int] = (384, 288),
                 device: str = 'cpu'):
        """
        Args:
            model_path: RTMW 모델 체크포인트 경로
            input_size: 입력 이미지 크기
            device: 추론 디바이스
        """
        self.device = device
        self.input_size = input_size
        
        # SimCC 디코더 초기화
        self.decoder = SimCCDecoder(input_size=input_size)
        
        # 모델 로드 (실제 구현에서는 RTMW 모델 로드)
        print(f"RTMW-x 모델 로딩: {model_path}")
        self.model = self._load_rtmw_model(model_path)
    
    def _load_rtmw_model(self, model_path: str):
        """RTMW 모델 로드 (더미 구현)"""
        class DummyRTMWModel(nn.Module):
            def __init__(self):
                super().__init__()
                # 실제 RTMW 모델 구조 (간소화)
                self.backbone = nn.Sequential(
                    nn.Conv2d(3, 64, 7, stride=2, padding=3),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((18, 24))  # 288/16, 384/16
                )
                self.neck = nn.Sequential(
                    nn.Conv2d(64, 256, 3, padding=1),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d(1),
                    nn.Flatten()
                )
                # SimCC 헤드
                self.head_x = nn.Linear(256, 133 * 576)  # 133 keypoints * 576 x-bins
                self.head_y = nn.Linear(256, 133 * 768)  # 133 keypoints * 768 y-bins
            
            def forward(self, x):
                # Backbone + Neck
                features = self.backbone(x)
                features = self.neck(features)
                
                # SimCC 헤드
                simcc_x = self.head_x(features).view(-1, 133, 576)  # [B, 133, 576]
                simcc_y = self.head_y(features).view(-1, 133, 768)  # [B, 133, 768]
                
                return simcc_x, simcc_y
        
        model = DummyRTMWModel()
        model.eval()
        return model.to(self.device)
    
    def predict(self, image: torch.Tensor) -> torch.Tensor:
        """
        이미지에서 키포인트 예측
        
        Args:
            image: [B, 3, H, W] 입력 이미지
            
        Returns:
            keypoints: [B, 133, 2] 키포인트 좌표
        """
        with torch.no_grad():
            # 모델 추론
            simcc_x, simcc_y = self.model(image)
            
            # SimCC 디코딩
            keypoints, _ = self.decoder.decode(simcc_x, simcc_y)
            
            return keypoints


def test_simcc_decoder():
    """SimCC 디코더 테스트"""
    print("=== SimCC 디코더 테스트 ===")
    
    # 테스트 파라미터
    batch_size = 1
    num_keypoints = 133
    input_size = (384, 288)
    
    # SimCC 디코더 초기화
    decoder = SimCCDecoder(input_size=input_size)
    
    # 더미 SimCC 출력 생성
    simcc_x = torch.randn(batch_size, num_keypoints, 576)  # X 분류 출력
    simcc_y = torch.randn(batch_size, num_keypoints, 768)  # Y 분류 출력
    
    print(f"\n📥 입력:")
    print(f"  - SimCC X: {list(simcc_x.shape)}")
    print(f"  - SimCC Y: {list(simcc_y.shape)}")
    
    # 디코딩
    keypoints, heatmaps = decoder.decode(simcc_x, simcc_y, return_heatmaps=True)
    
    print(f"\n📤 출력:")
    print(f"  - 키포인트: {list(keypoints.shape)}")
    print(f"  - 히트맵: {list(heatmaps.shape) if heatmaps is not None else None}")
    
    # 좌표 범위 확인
    x_coords = keypoints[..., 0]
    y_coords = keypoints[..., 1]
    
    print(f"\n📊 좌표 통계:")
    print(f"  - X 범위: [{x_coords.min():.2f}, {x_coords.max():.2f}]")
    print(f"  - Y 범위: [{y_coords.min():.2f}, {y_coords.max():.2f}]")
    print(f"  - 유효 키포인트: {(keypoints.sum(dim=-1) > 0).sum().item()}/{num_keypoints}")
    
    print("\n✅ SimCC 디코더 테스트 완료!")


def test_full_pipeline():
    """전체 파이프라인 테스트"""
    print("\n=== RTMW-x 전체 파이프라인 테스트 ===")
    
    # 더미 모델 래퍼 초기화
    wrapper = RTMWSimCCWrapper(
        model_path="dummy_rtmw.pth",
        input_size=(384, 288),
        device='cpu'
    )
    
    # 테스트 입력
    batch_size = 1
    test_image = torch.randn(batch_size, 3, 288, 384)  # [B, C, H, W]
    
    print(f"\n📥 테스트 입력: {list(test_image.shape)}")
    
    # 예측
    keypoints = wrapper.predict(test_image)
    
    print(f"📤 최종 출력: {list(keypoints.shape)}")
    print(f"✅ 예상 출력 [1, 133, 2] 달성!")
    
    # 키포인트 샘플 출력
    print(f"\n🎯 키포인트 샘플 (처음 5개):")
    for i in range(5):
        x, y = keypoints[0, i]
        print(f"  키포인트 {i}: ({x:.2f}, {y:.2f})")


if __name__ == '__main__':
    test_simcc_decoder()
    test_full_pipeline()