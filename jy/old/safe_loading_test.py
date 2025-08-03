#!/usr/bin/env python3
"""
안전한 checkpoint 로딩을 통한 RTMW 모델 테스트
PyTorch 2.6에서 weights_only=True 기본값으로 인한 오류 해결
"""

import os
import torch
import numpy as np
from mmpose.apis import MMPoseInferencer, init_model
from mmengine.runner import load_checkpoint
import warnings

# Numpy 관련 safe globals 추가
torch.serialization.add_safe_globals([
    np.core.multiarray._reconstruct,
    np.ndarray,
    np.dtype,
    np.core.multiarray.scalar,
])

def safe_load_checkpoint(model, checkpoint_path):
    """안전한 checkpoint 로딩"""
    try:
        # 방법 1: weights_only=False로 로드 시도
        print(f"🔧 안전하지 않은 모드로 checkpoint 로드 시도...")
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        # 모델에 로드
        if hasattr(model, 'load_state_dict'):
            if 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'], strict=False)
            else:
                model.load_state_dict(checkpoint, strict=False)
        else:
            load_checkpoint(model, checkpoint_path, map_location='cpu')
            
        print("✅ Checkpoint 로드 성공!")
        return True
        
    except Exception as e:
        print(f"❌ Checkpoint 로드 실패: {e}")
        return False

def test_safe_mmpose_loading():
    """안전한 MMPose 로딩 테스트"""
    print("=== 안전한 RTMW 모델 로딩 테스트 ===\n")
    
    checkpoint_path = "../models/rtmw-x_simcc-cocktail14_pt-ucoco_270e-384x288-f840f204_20231122.pth"
    config_path = "../configs/wholebody_2d_keypoint/rtmpose/cocktail14/rtmw-x_8xb320-270e_cocktail14-384x288.py"
    test_image = "winter01.jpg"
    
    # PyTorch 보안 설정 완화
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        
        # 방법 1: MMPoseInferencer 재시도
        print("🔧 방법 1: MMPoseInferencer with weights_only=False")
        try:
            # Monkey patch torch.load to use weights_only=False
            original_load = torch.load
            torch.load = lambda *args, **kwargs: original_load(*args, **kwargs, weights_only=False) if 'weights_only' not in kwargs else original_load(*args, **kwargs)
            
            inferencer = MMPoseInferencer(
                pose2d=config_path,
                pose2d_weights=checkpoint_path,
                device='cpu'
            )
            
            # 복원
            torch.load = original_load
            
            print("✅ MMPoseInferencer 로드 성공!")
            
            # 테스트 이미지로 추론
            if os.path.exists(test_image):
                results = inferencer(test_image, show=False, out_dir=None)
                print(f"  추론 결과: {len(results)} 개 이미지 처리됨")
                
                # 첫 번째 결과 확인
                if results and len(results) > 0:
                    pred_instances = results[0].pred_instances
                    print(f"  탐지된 인스턴스: {len(pred_instances)}")
                    
                    if len(pred_instances) > 0:
                        keypoints = pred_instances.keypoints[0]  # 첫 번째 사람
                        print(f"  키포인트 형태: {keypoints.shape}")
                        
                        # 첫 번째 몇 개 키포인트 확인
                        valid_points = keypoints[keypoints[:, 2] > 0.5]  # confidence > 0.5
                        if len(valid_points) > 0:
                            print(f"  유효한 키포인트 수: {len(valid_points)}")
                            print(f"  좌표 범위 - X: [{valid_points[:, 0].min():.1f}, {valid_points[:, 0].max():.1f}]")
                            print(f"  좌표 범위 - Y: [{valid_points[:, 1].min():.1f}, {valid_points[:, 1].max():.1f}]")
                            print(f"  신뢰도 범위: [{valid_points[:, 2].min():.3f}, {valid_points[:, 2].max():.3f}]")
                        else:
                            print("  ⚠️ 신뢰도 높은 키포인트가 없음")
                    else:
                        print("  ⚠️ 탐지된 사람 없음")
                        
        except Exception as e:
            print(f"❌ MMPoseInferencer 실패: {e}")
            torch.load = original_load
            
        print("\n" + "="*50 + "\n")
        
        # 방법 2: init_model 재시도
        print("🔧 방법 2: init_model with safe loading")
        try:
            # mmengine의 load_checkpoint monkey patch
            from mmengine.runner import checkpoint
            original_torch_load = torch.load
            
            def safe_torch_load(*args, **kwargs):
                kwargs['weights_only'] = False
                return original_torch_load(*args, **kwargs)
            
            torch.load = safe_torch_load
            
            # 모델 초기화
            model = init_model(config_path, checkpoint_path, device='cpu')
            
            # 복원
            torch.load = original_torch_load
            
            print("✅ init_model 로드 성공!")
            print(f"  모델 타입: {type(model)}")
            print(f"  모델 상태: {'training' if model.training else 'evaluation'}")
            
        except Exception as e:
            print(f"❌ init_model 실패: {e}")
            torch.load = original_torch_load

if __name__ == "__main__":
    test_safe_mmpose_loading()
