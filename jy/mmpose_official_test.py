#!/usr/bin/env python3
"""MMPose 공식 방법으로 RTMW 모델 테스트"""

import os
import cv2
import numpy as np
import torch
from mmpose.apis import init_model, inference_topdown
from mmpose.apis import MMPoseInferencer

def test_mmpose_official_method():
    """MMPose 공식 방법으로 RTMW 모델 테스트"""
    print("=== MMPose 공식 방법으로 RTMW 모델 테스트 ===")
    
    # 방법 1: MMPoseInferencer 사용 (가장 간단)
    print("\n🔧 방법 1: MMPoseInferencer 사용")
    try:
        # RTMW inferencer 초기화
        inferencer = MMPoseInferencer(
            pose2d='rtmw-x_8xb320-270e_cocktail14-384x288',
            pose2d_weights='../models/rtmw-x_simcc-cocktail14_pt-ucoco_270e-384x288-f840f204_20231122.pth',
            device='cpu'
        )
        
        # 테스트 이미지
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # 추론
        results = next(inferencer(test_image))
        
        print("✅ MMPoseInferencer 성공!")
        print(f"  결과 타입: {type(results)}")
        if 'predictions' in results:
            predictions = results['predictions']
            print(f"  예측 개수: {len(predictions)}")
            if predictions:
                keypoints = predictions[0].pred_instances.keypoints
                print(f"  키포인트 shape: {keypoints.shape}")
                print(f"  X 범위: [{keypoints[..., 0].min():.1f}, {keypoints[..., 0].max():.1f}]")
                print(f"  Y 범위: [{keypoints[..., 1].min():.1f}, {keypoints[..., 1].max():.1f}]")
        
    except Exception as e:
        print(f"❌ MMPoseInferencer 실패: {e}")
        import traceback
        traceback.print_exc()
    
    # 방법 2: 직접 config와 checkpoint 로드
    print("\n🔧 방법 2: 직접 config와 checkpoint 로드")
    try:
        # RTMW config 찾기
        import os
        config_candidates = [
            # RTMW 설정 파일들
            "../mmpose/configs/wholebody_2d_keypoint/rtmpose/cocktail14/rtmw-x_8xb320-270e_cocktail14-384x288.py",
            "../configs/wholebody_2d_keypoint/rtmpose/cocktail14/rtmw-x_8xb320-270e_cocktail14-384x288.py",
            "../../configs/wholebody_2d_keypoint/rtmpose/cocktail14/rtmw-x_8xb320-270e_cocktail14-384x288.py",
        ]
        
        config_file = None
        for candidate in config_candidates:
            if os.path.exists(candidate):
                config_file = candidate
                break
        
        if config_file:
            print(f"  설정 파일 발견: {config_file}")
            
            # 모델 초기화
            model = init_model(
                config_file, 
                '../models/rtmw-x_simcc-cocktail14_pt-ucoco_270e-384x288-f840f204_20231122.pth',
                device='cpu'
            )
            
            print("✅ 모델 로드 성공!")
            print(f"  모델 타입: {type(model)}")
            print(f"  백본: {type(model.backbone)}")
            print(f"  헤드: {type(model.head)}")
            
            # 테스트 추론
            test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            bbox = [100, 50, 300, 400]  # [x1, y1, x2, y2]
            
            results = inference_topdown(model, test_image, [bbox])
            
            print(f"  추론 결과: {len(results)}")
            if results:
                keypoints = results[0]['keypoints']
                print(f"  키포인트 shape: {keypoints.shape}")
                print(f"  X 범위: [{keypoints[..., 0].min():.1f}, {keypoints[..., 0].max():.1f}]")
                print(f"  Y 범위: [{keypoints[..., 1].min():.1f}, {keypoints[..., 1].max():.1f}]")
                
        else:
            print("❌ RTMW 설정 파일을 찾을 수 없음")
            
    except Exception as e:
        print(f"❌ 직접 로드 실패: {e}")
        import traceback
        traceback.print_exc()

    # 방법 3: projects/rtmpose에서 설정 찾기
    print("\n🔧 방법 3: projects/rtmpose에서 설정 찾기")
    try:
        import sys
        sys.path.append('../../projects/rtmpose')
        
        # projects 경로에서 설정 파일 찾기
        project_configs = [
            "../../projects/rtmpose/rtmpose/wholebody_2d_keypoint/rtmw-x_8xb320-270e_cocktail14-384x288.py",
            "../../projects/rtmpose/rtmpose/wholebody_2d_keypoint/rtmw-l_8xb320-270e_cocktail14-384x288.py",
        ]
        
        for config_path in project_configs:
            if os.path.exists(config_path):
                print(f"  프로젝트 설정 파일 발견: {config_path}")
                
                model = init_model(
                    config_path,
                    '../models/rtmw-x_simcc-cocktail14_pt-ucoco_270e-384x288-f840f204_20231122.pth',
                    device='cpu'
                )
                
                print("✅ 프로젝트 설정으로 모델 로드 성공!")
                break
                
    except Exception as e:
        print(f"❌ 프로젝트 설정 실패: {e}")

def test_with_real_image():
    """실제 이미지로 테스트"""
    print("\n=== 실제 이미지로 테스트 ===")
    
    try:
        # 실제 이미지 로드
        image_path = "winter01.jpg"
        if not os.path.exists(image_path):
            print(f"❌ 이미지 파일 없음: {image_path}")
            return
            
        image = cv2.imread(image_path)
        print(f"📷 이미지 로드: {image.shape}")
        
        # MMPoseInferencer로 추론
        inferencer = MMPoseInferencer(pose2d='body')  # 기본 body pose
        results = next(inferencer(image))
        
        print("✅ 실제 이미지 추론 성공!")
        if 'predictions' in results:
            predictions = results['predictions']
            print(f"  탐지된 인스턴스: {len(predictions)}")
            
    except Exception as e:
        print(f"❌ 실제 이미지 테스트 실패: {e}")

if __name__ == '__main__':
    test_mmpose_official_method()
    test_with_real_image()
