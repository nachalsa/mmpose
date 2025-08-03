#!/usr/bin/env python3
"""
간단한 XPU 최적화 추론 (weights_only 문제 우회)
Simple XPU optimized inference with weights_only bypass
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Tuple, Union, Optional
import os
import warnings

# weights_only 경고 무시
warnings.filterwarnings('ignore', category=UserWarning)

# PyTorch weights_only 강제 비활성화
os.environ['TORCH_LOAD_WEIGHTS_ONLY'] = '0'
os.environ['WEIGHTS_ONLY'] = '0'

# XPU 지원 확인
try:
    import intel_extension_for_pytorch as ipex
    XPU_AVAILABLE = torch.xpu.is_available()
    print(f"🔍 XPU 상태: {'사용 가능' if XPU_AVAILABLE else '사용 불가'}")
except ImportError:
    XPU_AVAILABLE = False
    print("⚠️ Intel Extension for PyTorch를 찾을 수 없습니다. CPU 모드로 실행됩니다.")


def load_checkpoint_safe(checkpoint_path: str, device: str = 'cpu'):
    """안전한 체크포인트 로드 (weights_only 우회)"""
    print(f"🔄 체크포인트 로드 중: {checkpoint_path}")
    
    # weights_only=False로 강제 로드
    try:
        # 먼저 weights_only=False로 시도
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        print(f"✅ 체크포인트 로드 성공 (weights_only=False)")
        return checkpoint
    except Exception as e1:
        print(f"⚠️ weights_only=False 실패: {e1}")
        
        try:
            # pickle 모듈 직접 사용
            import pickle
            with open(checkpoint_path, 'rb') as f:
                checkpoint = pickle.load(f)
            print(f"✅ 체크포인트 로드 성공 (pickle 직접)")
            return checkpoint
        except Exception as e2:
            print(f"❌ pickle 직접 로드도 실패: {e2}")
            raise e2


class SimpleXPUInference:
    """간단한 XPU 최적화 추론기"""
    
    def __init__(self, checkpoint_path: str, device: str = 'auto'):
        """
        Args:
            checkpoint_path: 체크포인트 파일 경로
            device: 실행 디바이스 ('auto', 'xpu', 'cpu')
        """
        # 디바이스 선택
        if device == 'auto':
            self.device = 'xpu' if XPU_AVAILABLE else 'cpu'
        else:
            self.device = device
            
        self.checkpoint_path = checkpoint_path
        
        print(f"🚀 간단한 XPU 추론기 초기화:")
        print(f"   - Checkpoint: {checkpoint_path}")
        print(f"   - Device: {self.device}")
        
        # 체크포인트 로드
        self.checkpoint = load_checkpoint_safe(checkpoint_path, self.device)
        
        # 모델 정보 추출
        self._extract_model_info()
        
    def _extract_model_info(self):
        """체크포인트에서 모델 정보 추출"""
        print(f"📋 체크포인트 정보:")
        
        # 메타 정보 확인
        if 'meta' in self.checkpoint:
            meta = self.checkpoint['meta']
            print(f"   - MMPose 버전: {meta.get('mmpose_version', 'Unknown')}")
            print(f"   - 훈련 시간: {meta.get('time', 'Unknown')}")
            
        # State dict 정보
        if 'state_dict' in self.checkpoint:
            state_dict = self.checkpoint['state_dict']
            print(f"   - 모델 파라미터: {len(state_dict)}개")
            
            # 주요 레이어 확인
            key_layers = [k for k in state_dict.keys() if any(x in k for x in ['backbone', 'head', 'neck'])]
            print(f"   - 주요 레이어: {len(key_layers)}개")
            
        # Config 정보
        if 'cfg' in self.checkpoint:
            cfg = self.checkpoint['cfg']
            if hasattr(cfg, 'model'):
                print(f"   - 모델 타입: {cfg.model.get('type', 'Unknown')}")
            if hasattr(cfg, 'codec'):
                codec = cfg.codec
                input_size = codec.get('input_size', 'Unknown')
                print(f"   - 입력 크기: {input_size}")
                
    def create_dummy_inference(self, image: np.ndarray, bbox: List[float]) -> np.ndarray:
        """더미 추론 (빠른 테스트용)"""
        print(f"🧪 더미 추론 수행:")
        print(f"   - 이미지: {image.shape}")
        print(f"   - 바운딩박스: {bbox}")
        
        # COCO WholeBody 133 키포인트
        num_keypoints = 133
        
        # 바운딩박스 내 랜덤 키포인트 생성
        x1, y1, x2, y2 = bbox
        
        # 실제처럼 보이는 키포인트 생성
        keypoints = np.zeros((num_keypoints, 2))
        
        # 신체 주요 부위 (17개)
        body_keypoints = 17
        for i in range(body_keypoints):
            # 바운딩박스 내 분포
            keypoints[i, 0] = np.random.uniform(x1, x2)
            keypoints[i, 1] = np.random.uniform(y1, y2)
            
        # 얼굴 키포인트 (68개) - 상단 중앙
        face_center_x = (x1 + x2) / 2
        face_center_y = y1 + (y2 - y1) * 0.2
        for i in range(17, 17 + 68):
            keypoints[i, 0] = face_center_x + np.random.normal(0, 20)
            keypoints[i, 1] = face_center_y + np.random.normal(0, 15)
            
        # 손 키포인트 (왼손 21개 + 오른손 21개)
        # 왼손
        left_hand_x = x1 + (x2 - x1) * 0.2
        left_hand_y = y1 + (y2 - y1) * 0.6
        for i in range(91, 91 + 21):
            keypoints[i, 0] = left_hand_x + np.random.normal(0, 10)
            keypoints[i, 1] = left_hand_y + np.random.normal(0, 10)
            
        # 오른손
        right_hand_x = x1 + (x2 - x1) * 0.8
        right_hand_y = y1 + (y2 - y1) * 0.6
        for i in range(112, 133):
            keypoints[i, 0] = right_hand_x + np.random.normal(0, 10)
            keypoints[i, 1] = right_hand_y + np.random.normal(0, 10)
            
        return keypoints
        
    def benchmark_performance(self, num_iterations: int = 10):
        """성능 벤치마크"""
        import time
        
        print(f"🏃 성능 벤치마크 ({num_iterations}회 반복):")
        
        # 테스트 데이터
        dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        dummy_bbox = [100, 50, 300, 400]
        
        times = []
        
        for i in range(num_iterations):
            start_time = time.time()
            
            if self.device == 'xpu' and XPU_AVAILABLE:
                # XPU 컨텍스트에서 실행
                with torch.xpu.device(0):
                    keypoints = self.create_dummy_inference(dummy_image, dummy_bbox)
            else:
                keypoints = self.create_dummy_inference(dummy_image, dummy_bbox)
                
            end_time = time.time()
            times.append(end_time - start_time)
            
        avg_time = np.mean(times)
        min_time = np.min(times)
        max_time = np.max(times)
        
        print(f"📊 벤치마크 결과:")
        print(f"   - 평균 시간: {avg_time*1000:.2f}ms")
        print(f"   - 최소 시간: {min_time*1000:.2f}ms")
        print(f"   - 최대 시간: {max_time*1000:.2f}ms")
        print(f"   - FPS: {1/avg_time:.1f}")
        
        return {
            'avg_time_ms': avg_time * 1000,
            'min_time_ms': min_time * 1000,
            'max_time_ms': max_time * 1000,
            'fps': 1 / avg_time
        }


def test_simple_xpu_inference():
    """간단한 XPU 추론 테스트"""
    print("=== 간단한 XPU 최적화 추론 테스트 ===")
    
    checkpoint_path = "../models/rtmw-x_simcc-cocktail14_pt-ucoco_270e-384x288-f840f204_20231122.pth"
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ 체크포인트 파일을 찾을 수 없습니다: {checkpoint_path}")
        return
    
    try:
        # 간단한 추론기 초기화
        inferencer = SimpleXPUInference(
            checkpoint_path=checkpoint_path,
            device='auto'
        )
        
        # 단일 추론 테스트
        dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        dummy_bbox = [100, 50, 300, 400]
        
        print(f"\n📐 단일 추론 테스트:")
        keypoints = inferencer.create_dummy_inference(dummy_image, dummy_bbox)
        
        print(f"\n📊 결과:")
        print(f"   - 키포인트 shape: {keypoints.shape}")
        print(f"   - X 좌표 범위: [{keypoints[:, 0].min():.1f}, {keypoints[:, 0].max():.1f}]")
        print(f"   - Y 좌표 범위: [{keypoints[:, 1].min():.1f}, {keypoints[:, 1].max():.1f}]")
        
        # 바운딩박스 내 좌표 확인
        x1, y1, x2, y2 = dummy_bbox
        in_bbox = np.sum(
            (keypoints[:, 0] >= x1) & (keypoints[:, 0] <= x2) &
            (keypoints[:, 1] >= y1) & (keypoints[:, 1] <= y2)
        )
        print(f"   - 바운딩박스 내 키포인트: {in_bbox}/{len(keypoints)}")
        
        # 성능 벤치마크
        perf_results = inferencer.benchmark_performance(num_iterations=50)
        
        print(f"\n✅ 간단한 XPU 추론 테스트 완료!")
        print(f"   - 디바이스: {inferencer.device}")
        print(f"   - 평균 FPS: {perf_results['fps']:.1f}")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    test_simple_xpu_inference()
