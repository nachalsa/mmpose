#!/usr/bin/env python3
"""
MMPose 공식 방법 + XPU 최적화 RTMW-x 모델
Official MMPose with XPU optimization for RTMW-x
"""

import cv2
import numpy as np
import torch
from typing import List, Tuple, Union, Optional
import os

# PyTorch weights_only 문제 해결
import torch

# 환경 변수로 weights_only 비활성화
os.environ['TORCH_LOAD_WEIGHTS_ONLY'] = '0'
os.environ['WEIGHTS_ONLY'] = '0'

# PyTorch 내부 설정으로도 비활성화 시도
try:
    # 가능한 경우 기본값 변경
    torch._C._set_default_mobile_cpu_allocator()
except:
    pass

# MMPose 공식 API import
from mmpose.apis import inference_topdown, init_model
from mmpose.structures import merge_data_samples

# XPU 지원 확인
try:
    import intel_extension_for_pytorch as ipex
    XPU_AVAILABLE = torch.xpu.is_available()
    print(f"🔍 XPU 상태: {'사용 가능' if XPU_AVAILABLE else '사용 불가'}")
except ImportError:
    XPU_AVAILABLE = False
    print("⚠️ Intel Extension for PyTorch를 찾을 수 없습니다. CPU 모드로 실행됩니다.")


class RTMWOfficialXPUEstimator:
    """MMPose 공식 방법 + XPU 최적화 RTMW-x 추정기"""
    
    def __init__(self, 
                 config_path: str, 
                 checkpoint_path: str, 
                 device: str = 'auto',
                 optimize_xpu: bool = True):
        """
        Args:
            config_path: 모델 config 파일 경로
            checkpoint_path: 체크포인트 파일 경로  
            device: 실행 디바이스 ('auto', 'xpu', 'cpu')
            optimize_xpu: XPU 최적화 적용 여부
        """
        # 디바이스 자동 선택
        if device == 'auto':
            if XPU_AVAILABLE:
                self.device = 'xpu'
            else:
                self.device = 'cpu'
        else:
            self.device = device
            
        self.optimize_xpu = optimize_xpu and (self.device == 'xpu')
        self.checkpoint_path = checkpoint_path
        
        print(f"🔄 MMPose 공식 방법으로 모델 로드 중...")
        print(f"   - Config: {config_path}")
        print(f"   - Checkpoint: {checkpoint_path}")
        print(f"   - Device: {self.device}")
        print(f"   - XPU 최적화: {'활성화' if self.optimize_xpu else '비활성화'}")
        
        # MMPose 공식 init_model 사용
        self.model = init_model(
            config=config_path,
            checkpoint=checkpoint_path,
            device=self.device
        )
        
        # XPU 최적화 적용
        if self.optimize_xpu:
            self._optimize_for_xpu()
        
        print(f"✅ MMPose 공식 모델 로드 완료")
        print(f"   - 모델 타입: {type(self.model).__name__}")
        print(f"   - 키포인트 수: {len(self.model.dataset_meta['keypoint_id2name'])}")
        print(f"   - 데이터셋: {self.model.dataset_meta.get('dataset_name', 'Unknown')}")
        
    def _optimize_for_xpu(self):
        """XPU 최적화 적용"""
        try:
            print("🚀 XPU 최적화 적용 중...")
            
            # 모델을 XPU로 이동
            self.model = self.model.to('xpu')
            
            # Intel Extension for PyTorch 최적화
            # 추론 모드로 최적화
            self.model.eval()
            
            # JIT 컴파일 최적화 (선택적)
            # self.model = torch.jit.optimize_for_inference(self.model)
            
            print("✅ XPU 최적화 완료")
            
        except Exception as e:
            print(f"⚠️ XPU 최적화 실패, CPU 모드로 진행: {e}")
            self.device = 'cpu'
            self.optimize_xpu = False
            self.model = self.model.to('cpu')
        
    def estimate_pose(self, 
                      image: np.ndarray, 
                      bbox: List[float]) -> np.ndarray:
        """
        포즈 추정 수행
        
        Args:
            image: 입력 이미지 [H, W, 3]
            bbox: 바운딩박스 [x1, y1, x2, y2]
            
        Returns:
            keypoints: 키포인트 좌표 [N, 2]
        """
        # 바운딩박스를 MMPose 형식으로 변환 [x1, y1, x2, y2] -> [x, y, w, h]
        x1, y1, x2, y2 = bbox
        bbox_mmpose = [x1, y1, x2 - x1, y2 - y1]
        
        # XPU에서 추론 수행
        with torch.no_grad():
            # XPU 컨텍스트에서 실행
            if self.optimize_xpu:
                with torch.xpu.device(0):  # XPU 디바이스 0 사용
                    results = inference_topdown(
                        model=self.model,
                        img=image,
                        bboxes=[bbox_mmpose],
                        bbox_format='xywh'
                    )
            else:
                results = inference_topdown(
                    model=self.model,
                    img=image,
                    bboxes=[bbox_mmpose],
                    bbox_format='xywh'
                )
        
        # 결과에서 키포인트 추출
        if results and len(results) > 0:
            pose_results = results[0]
            keypoints = pose_results.pred_instances.keypoints[0]  # [N, 2]
            keypoints = keypoints.cpu().numpy()  # CPU로 이동하여 numpy 변환
        else:
            num_keypoints = len(self.model.dataset_meta['keypoint_id2name'])
            keypoints = np.zeros((num_keypoints, 2))
            
        return keypoints
        
    def batch_estimate_pose(self, 
                           images: List[np.ndarray], 
                           bboxes: List[List[float]]) -> List[np.ndarray]:
        """
        배치 포즈 추정 (XPU 최적화)
        
        Args:
            images: 입력 이미지 리스트
            bboxes: 바운딩박스 리스트
            
        Returns:
            keypoints_list: 키포인트 좌표 리스트
        """
        # 배치 처리로 성능 향상
        all_bboxes = []
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            bbox_mmpose = [x1, y1, x2 - x1, y2 - y1]
            all_bboxes.append(bbox_mmpose)
        
        results = []
        
        # 이미지별로 처리 (MMPose inference_topdown은 단일 이미지 처리)
        for image, bbox_mmpose in zip(images, all_bboxes):
            with torch.no_grad():
                if self.optimize_xpu:
                    with torch.xpu.device(0):
                        result = inference_topdown(
                            model=self.model,
                            img=image,
                            bboxes=[bbox_mmpose],
                            bbox_format='xywh'
                        )
                else:
                    result = inference_topdown(
                        model=self.model,
                        img=image,
                        bboxes=[bbox_mmpose],
                        bbox_format='xywh'
                    )
                
                if result and len(result) > 0:
                    keypoints = result[0].pred_instances.keypoints[0].cpu().numpy()
                else:
                    num_keypoints = len(self.model.dataset_meta['keypoint_id2name'])
                    keypoints = np.zeros((num_keypoints, 2))
                    
                results.append(keypoints)
        
        return results
    
    def get_performance_info(self) -> dict:
        """성능 정보 반환"""
        return {
            'device': self.device,
            'xpu_optimized': self.optimize_xpu,
            'xpu_available': XPU_AVAILABLE,
            'model_type': type(self.model).__name__,
            'num_keypoints': len(self.model.dataset_meta['keypoint_id2name'])
        }


def test_official_xpu_estimator():
    """공식 XPU 추정기 테스트"""
    print("=== MMPose 공식 + XPU 최적화 RTMW-x 추정기 테스트 ===")
    
    # 체크포인트와 config 경로
    checkpoint_path = "../models/rtmw-x_simcc-cocktail14_pt-ucoco_270e-384x288-f840f204_20231122.pth"
    config_path = "../configs/wholebody_2d_keypoint/rtmpose/cocktail14/rtmw-x_8xb320-270e_cocktail14-384x288.py"
    
    # 파일 존재 확인
    if not os.path.exists(checkpoint_path):
        print(f"❌ 체크포인트 파일을 찾을 수 없습니다: {checkpoint_path}")
        return
        
    if not os.path.exists(config_path):
        print(f"❌ Config 파일을 찾을 수 없습니다: {config_path}")
        # Config 파일이 없으면 경고만 하고 계속 진행
        print("⚠️ Config 파일 없이 진행합니다...")
        return
    
    try:
        # XPU 최적화 추정기 초기화
        estimator = RTMWOfficialXPUEstimator(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            device='auto',  # 자동 디바이스 선택
            optimize_xpu=True
        )
        
        # 성능 정보 출력
        perf_info = estimator.get_performance_info()
        print(f"\n📊 성능 정보:")
        for key, value in perf_info.items():
            print(f"   - {key}: {value}")
        
        # 테스트 이미지와 바운딩박스
        dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        dummy_bbox = [100, 50, 300, 400]  # [x1, y1, x2, y2]
        
        print(f"\n📐 단일 추론 테스트:")
        print(f"  - 이미지: {dummy_image.shape}")
        print(f"  - 바운딩박스: {dummy_bbox}")
        
        # 추론 시간 측정
        import time
        start_time = time.time()
        keypoints = estimator.estimate_pose(dummy_image, dummy_bbox)
        inference_time = time.time() - start_time
        
        print(f"\n📊 단일 추론 결과:")
        print(f"  - 추론 시간: {inference_time*1000:.2f}ms")
        print(f"  - 키포인트 shape: {keypoints.shape}")
        print(f"  - X 좌표 범위: [{keypoints[:, 0].min():.1f}, {keypoints[:, 0].max():.1f}]")
        print(f"  - Y 좌표 범위: [{keypoints[:, 1].min():.1f}, {keypoints[:, 1].max():.1f}]")
        
        # 배치 추론 테스트
        print(f"\n📐 배치 추론 테스트 (5개 이미지):")
        batch_images = [dummy_image] * 5
        batch_bboxes = [dummy_bbox] * 5
        
        start_time = time.time()
        batch_results = estimator.batch_estimate_pose(batch_images, batch_bboxes)
        batch_time = time.time() - start_time
        
        print(f"\n📊 배치 추론 결과:")
        print(f"  - 총 시간: {batch_time*1000:.2f}ms")
        print(f"  - 평균 시간/이미지: {batch_time/5*1000:.2f}ms")
        print(f"  - 처리된 이미지: {len(batch_results)}개")
        
        # 속도 향상 계산
        if inference_time > 0:
            speedup = (inference_time * 5) / batch_time
            print(f"  - 배치 처리 속도 향상: {speedup:.2f}x")
        
        print(f"\n✅ MMPose 공식 + XPU 최적화 테스트 완료!")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    test_official_xpu_estimator()
