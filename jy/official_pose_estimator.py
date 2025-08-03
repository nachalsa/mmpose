#!/usr/bin/env python3
"""
MMPose 공식 방법으로 RTMW-x 모델 로드
Official MMPose model loading approach for RTMW-x
"""

import cv2
import numpy as np
import torch
from typing import List, Tuple, Union, Optional
import os

# PyTorch weights_only 문제 해결 - 환경 변수로 설정
os.environ['TORCH_LOAD_WEIGHTS_ONLY'] = '0'

# MMPose 공식 API import
from mmpose.apis import inference_topdown, init_model
from mmpose.structures import merge_data_samples


class RTMWOfficialEstimator:
    """MMPose 공식 방법으로 RTMW-x 모델을 로드하는 추정기"""
    
    def __init__(self, 
                 config_path: str, 
                 checkpoint_path: str, 
                 device: str = 'cpu'):
        """
        Args:
            config_path: 모델 config 파일 경로
            checkpoint_path: 체크포인트 파일 경로  
            device: 실행 디바이스
        """
        self.device = device
        self.checkpoint_path = checkpoint_path
        
        print(f"🔄 MMPose 공식 방법으로 모델 로드 중...")
        print(f"   - Config: {config_path}")
        print(f"   - Checkpoint: {checkpoint_path}")
        print(f"   - Device: {device}")
        
        # MMPose 공식 init_model 사용
        self.model = init_model(
            config=config_path,
            checkpoint=checkpoint_path,
            device=device
        )
        
        print(f"✅ MMPose 공식 모델 로드 완료")
        print(f"   - 모델 타입: {type(self.model).__name__}")
        print(f"   - 키포인트 수: {len(self.model.dataset_meta['keypoint_id2name'])}")
        print(f"   - 데이터셋: {self.model.dataset_meta.get('dataset_name', 'Unknown')}")
        
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
        
        # MMPose 공식 inference_topdown 사용
        with torch.no_grad():
            results = inference_topdown(
                model=self.model,
                img=image,
                bboxes=[bbox_mmpose],  # 리스트로 전달
                bbox_format='xywh'
            )
        
        # 결과에서 키포인트 추출
        if results and len(results) > 0:
            # PoseDataSample에서 키포인트 추출
            pose_results = results[0]
            keypoints = pose_results.pred_instances.keypoints[0]  # [N, 2]
            keypoints = keypoints.cpu().numpy()
        else:
            # 결과가 없으면 빈 배열 반환
            num_keypoints = len(self.model.dataset_meta['keypoint_id2name'])
            keypoints = np.zeros((num_keypoints, 2))
            
        return keypoints
        
    def batch_estimate_pose(self, 
                           images: List[np.ndarray], 
                           bboxes: List[List[float]]) -> List[np.ndarray]:
        """
        배치 포즈 추정
        
        Args:
            images: 입력 이미지 리스트
            bboxes: 바운딩박스 리스트
            
        Returns:
            keypoints_list: 키포인트 좌표 리스트
        """
        results = []
        for image, bbox in zip(images, bboxes):
            keypoints = self.estimate_pose(image, bbox)
            results.append(keypoints)
        return results


def test_official_estimator():
    """공식 추정기 테스트"""
    print("=== MMPose 공식 RTMW-x 추정기 테스트 ===")
    
    # 체크포인트 경로 (정확한 크기는 288x384임)
    checkpoint_path = "../models/rtmw-x_simcc-cocktail14_pt-ucoco_270e-384x288-f840f204_20231122.pth"
    
    # RTMW-x 384x288 (실제로는 288x384) config 파일 경로
    config_path = "../configs/wholebody_2d_keypoint/rtmpose/cocktail14/rtmw-x_8xb320-270e_cocktail14-384x288.py"
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ 체크포인트 파일을 찾을 수 없습니다: {checkpoint_path}")
        return
        
    if not os.path.exists(config_path):
        print(f"❌ Config 파일을 찾을 수 없습니다: {config_path}")
        return
    
    try:
        # 공식 추정기 초기화
        estimator = RTMWOfficialEstimator(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            device='cpu'
        )
        
        # 테스트 이미지와 바운딩박스
        dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        dummy_bbox = [100, 50, 300, 400]  # [x1, y1, x2, y2]
        
        print(f"\n📐 테스트:")
        print(f"  - 이미지: {dummy_image.shape}")
        print(f"  - 바운딩박스: {dummy_bbox}")
        
        # 포즈 추정
        keypoints = estimator.estimate_pose(dummy_image, dummy_bbox)
        
        print(f"\n📊 결과:")
        print(f"  - 키포인트 shape: {keypoints.shape}")
        print(f"  - X 좌표 범위: [{keypoints[:, 0].min():.1f}, {keypoints[:, 0].max():.1f}]")
        print(f"  - Y 좌표 범위: [{keypoints[:, 1].min():.1f}, {keypoints[:, 1].max():.1f}]")
        
        # 바운딩박스 내 좌표 개수 확인
        x1, y1, x2, y2 = dummy_bbox
        in_bbox = np.sum(
            (keypoints[:, 0] >= x1) & (keypoints[:, 0] <= x2) &
            (keypoints[:, 1] >= y1) & (keypoints[:, 1] <= y2)
        )
        print(f"  - 바운딩박스 내 키포인트: {in_bbox}/{len(keypoints)}")
        
        # 몇 개 샘플 출력
        print(f"\n📍 키포인트 샘플 (처음 5개):")
        for i in range(min(5, len(keypoints))):
            x, y = keypoints[i, 0], keypoints[i, 1]
            in_bbox = (x >= x1 and x <= x2 and y >= y1 and y <= y2)
            print(f"  [{i:2d}] ({x:6.1f}, {y:6.1f}) {'✓' if in_bbox else '✗'}")
            
        print(f"\n✅ MMPose 공식 방법 테스트 완료!")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    test_official_estimator()
