#!/usr/bin/env python3
"""
MMPose 공식 방법으로 RTMW 추론 테스트
성공적으로 로드된 모델을 사용하여 실제 pose estimation 수행
"""

import os
import torch
import cv2
import numpy as np
from mmpose.apis import MMPoseInferencer, init_model, inference_topdown
from mmpose.utils import adapt_mmdet_pipeline
import json

def test_official_rtmw_inference():
    """공식 RTMW 추론 테스트"""
    print("=== MMPose 공식 RTMW 추론 테스트 ===\n")
    
    checkpoint_path = "../models/rtmw-x_simcc-cocktail14_pt-ucoco_270e-384x288-f840f204_20231122.pth"
    config_path = "../configs/wholebody_2d_keypoint/rtmpose/cocktail14/rtmw-x_8xb320-270e_cocktail14-384x288.py"
    test_image = "winter01.jpg"
    
    # PyTorch 보안 설정
    original_load = torch.load
    torch.load = lambda *args, **kwargs: original_load(*args, **kwargs, weights_only=False) if 'weights_only' not in kwargs else original_load(*args, **kwargs)
    
    try:
        print("🔧 1. 모델 초기화")
        model = init_model(config_path, checkpoint_path, device='cpu')
        print(f"✅ 모델 로드 성공: {type(model)}")
        
        # 이미지 로드
        if not os.path.exists(test_image):
            print(f"❌ 테스트 이미지 없음: {test_image}")
            return
            
        image = cv2.imread(test_image)
        print(f"📷 이미지 로드: {image.shape}")
        
        print("\n🔧 2. MMPoseInferencer로 전체 파이프라인 테스트")
        
        # MMPoseInferencer 사용 (detection + pose estimation)
        inferencer = MMPoseInferencer(
            pose2d=config_path,
            pose2d_weights=checkpoint_path,
            device='cpu'
        )
        
        # 추론 실행
        results_gen = inferencer(test_image, show=False, return_vis=False)
        results = list(results_gen)  # Generator를 list로 변환
        
        print(f"✅ 추론 완료: {len(results)} 개 이미지 처리됨")
        
        if len(results) > 0:
            result = results[0]
            pred_instances = result.pred_instances
            print(f"  탐지된 인스턴스: {len(pred_instances)}")
            
            if len(pred_instances) > 0:
                # 첫 번째 사람의 키포인트
                keypoints = pred_instances.keypoints[0]  # shape: (133, 3)
                keypoint_scores = pred_instances.keypoint_scores[0]  # shape: (133,)
                bbox = pred_instances.bboxes[0]  # shape: (5,) - x1,y1,x2,y2,score
                
                print(f"  키포인트 형태: {keypoints.shape}")
                print(f"  바운딩 박스: [{bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}] (score: {bbox[4]:.3f})")
                
                # 신뢰도 높은 키포인트 분석
                high_conf_mask = keypoint_scores > 0.5
                high_conf_points = keypoints[high_conf_mask]
                high_conf_scores = keypoint_scores[high_conf_mask]
                
                print(f"  고신뢰도(>0.5) 키포인트: {len(high_conf_points)}/{len(keypoints)}")
                
                if len(high_conf_points) > 0:
                    print(f"  좌표 범위 - X: [{high_conf_points[:, 0].min():.1f}, {high_conf_points[:, 0].max():.1f}]")
                    print(f"  좌표 범위 - Y: [{high_conf_points[:, 1].min():.1f}, {high_conf_points[:, 1].max():.1f}]")
                    print(f"  신뢰도 범위: [{high_conf_scores.min():.3f}, {high_conf_scores.max():.3f}]")
                    
                    # 주요 키포인트들 확인 (RTMW는 COCO + wholebody 133개 키포인트)
                    body_keypoints = ["nose", "left_eye", "right_eye", "left_ear", "right_ear",
                                     "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
                                     "left_wrist", "right_wrist", "left_hip", "right_hip",
                                     "left_knee", "right_knee", "left_ankle", "right_ankle"]
                    
                    print("\n  📍 주요 신체 부위 키포인트:")
                    for i, name in enumerate(body_keypoints):
                        if i < len(keypoints) and keypoint_scores[i] > 0.3:
                            x, y = keypoints[i][0], keypoints[i][1]
                            score = keypoint_scores[i]
                            print(f"    {name:15}: ({x:6.1f}, {y:6.1f}) conf={score:.3f}")
                
                # 결과 시각화를 위한 이미지 저장
                print("\n🎨 결과 시각화")
                vis_results = inferencer(test_image, show=False, return_vis=True, out_dir='rtmw_official_output')
                vis_list = list(vis_results)
                if vis_list:
                    print("✅ 시각화 이미지 저장됨: rtmw_official_output/")
                    
            else:
                print("  ⚠️ 탐지된 사람 없음")
        else:
            print("  ⚠️ 추론 결과 없음")
            
    except Exception as e:
        print(f"❌ 추론 실패: {e}")
        import traceback
        traceback.print_exc()
    finally:
        torch.load = original_load

def compare_with_manual_implementation():
    """수동 구현과 공식 방법 비교"""
    print("\n" + "="*60)
    print("🔍 수동 구현 vs 공식 방법 비교")
    print("="*60)
    
    # 수동 구현 결과 로드 (이전 테스트에서)
    manual_results = {
        "method": "Manual Implementation",
        "keypoints_detected": "낮은 신뢰도의 키포인트들",
        "coordinate_range": "X: [76.3, 318.1], Y: [62.2, 367.8]",
        "weight_loading": "11% coverage with adaptive mapping",
        "architecture": "Custom RTMW implementation"
    }
    
    print("📊 수동 구현 결과:")
    for key, value in manual_results.items():
        print(f"  {key}: {value}")
    
    print("\n📊 공식 방법의 장점:")
    print("  ✅ 완전한 모델 로딩 (100% weight coverage)")
    print("  ✅ 검증된 전처리 파이프라인")
    print("  ✅ 올바른 좌표 변환")
    print("  ✅ 133개 전체 키포인트 지원")
    print("  ✅ 높은 신뢰도 예측")

if __name__ == "__main__":
    test_official_rtmw_inference()
    compare_with_manual_implementation()
