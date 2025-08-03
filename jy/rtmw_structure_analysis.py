#!/usr/bin/env python3
"""
MMPose 공식 RTMW 결과 구조 분석 및 올바른 추론
"""

import os
import torch
import cv2
import numpy as np
from mmpose.apis import MMPoseInferencer, init_model
import json

def analyze_result_structure():
    """결과 구조 분석"""
    print("=== MMPose 결과 구조 분석 ===\n")
    
    checkpoint_path = "../models/rtmw-x_simcc-cocktail14_pt-ucoco_270e-384x288-f840f204_20231122.pth"
    config_path = "../configs/wholebody_2d_keypoint/rtmpose/cocktail14/rtmw-x_8xb320-270e_cocktail14-384x288.py"
    test_image = "winter01.jpg"
    
    # PyTorch 보안 설정
    original_load = torch.load
    torch.load = lambda *args, **kwargs: original_load(*args, **kwargs, weights_only=False) if 'weights_only' not in kwargs else original_load(*args, **kwargs)
    
    try:
        print("🔧 MMPoseInferencer 초기화")
        inferencer = MMPoseInferencer(
            pose2d=config_path,
            pose2d_weights=checkpoint_path,
            device='cpu'
        )
        
        # 추론 실행
        print("🔧 추론 실행")
        results_gen = inferencer(test_image, show=False, return_vis=False)
        results = list(results_gen)
        
        print(f"✅ 결과 개수: {len(results)}")
        
        if len(results) > 0:
            result = results[0]
            print(f"결과 타입: {type(result)}")
            print(f"결과 키들: {list(result.keys()) if hasattr(result, 'keys') else 'No keys method'}")
            
            # 각 키의 내용 확인
            for key in result.keys():
                value = result[key]
                print(f"\n키 '{key}':")
                print(f"  타입: {type(value)}")
                
                if hasattr(value, 'pred_instances'):
                    print(f"  pred_instances 있음: {type(value.pred_instances)}")
                    pred_instances = value.pred_instances
                    print(f"  인스턴스 수: {len(pred_instances)}")
                    
                    if len(pred_instances) > 0:
                        instance = pred_instances[0]
                        print(f"  첫 번째 인스턴스 속성: {dir(instance)}")
                        
                        # 키포인트 정보
                        if hasattr(instance, 'keypoints'):
                            keypoints = instance.keypoints
                            print(f"  키포인트 형태: {keypoints.shape}")
                            
                        if hasattr(instance, 'keypoint_scores'):
                            scores = instance.keypoint_scores
                            print(f"  점수 형태: {scores.shape}")
                            print(f"  점수 범위: [{scores.min():.3f}, {scores.max():.3f}]")
                            
                        if hasattr(instance, 'bboxes'):
                            bboxes = instance.bboxes
                            print(f"  바운딩박스 형태: {bboxes.shape}")
                            print(f"  바운딩박스: {bboxes}")
                            
                elif hasattr(value, 'shape'):
                    print(f"  형태: {value.shape}")
                elif isinstance(value, (list, tuple)):
                    print(f"  길이: {len(value)}")
                    if len(value) > 0:
                        print(f"  첫 번째 원소 타입: {type(value[0])}")
                else:
                    print(f"  값: {value}")
        
        # 시각화 결과도 확인
        print("\n🎨 시각화 결과 분석")
        vis_results_gen = inferencer(test_image, show=False, return_vis=True)
        vis_results = list(vis_results_gen)
        
        if len(vis_results) > 0:
            vis_result = vis_results[0]
            print(f"시각화 결과 타입: {type(vis_result)}")
            print(f"시각화 결과 키들: {list(vis_result.keys()) if hasattr(vis_result, 'keys') else 'No keys method'}")
            
    except Exception as e:
        print(f"❌ 분석 실패: {e}")
        import traceback
        traceback.print_exc()
    finally:
        torch.load = original_load

def correct_inference_test():
    """올바른 추론 테스트"""
    print("\n" + "="*50)
    print("=== 올바른 RTMW 추론 테스트 ===\n")
    
    checkpoint_path = "../models/rtmw-x_simcc-cocktail14_pt-ucoco_270e-384x288-f840f204_20231122.pth"
    config_path = "../configs/wholebody_2d_keypoint/rtmpose/cocktail14/rtmw-x_8xb320-270e_cocktail14-384x288.py"
    test_image = "winter01.jpg"
    
    # PyTorch 보안 설정
    original_load = torch.load
    torch.load = lambda *args, **kwargs: original_load(*args, **kwargs, weights_only=False) if 'weights_only' not in kwargs else original_load(*args, **kwargs)
    
    try:
        print("🔧 MMPoseInferencer 초기화")
        inferencer = MMPoseInferencer(
            pose2d=config_path,
            pose2d_weights=checkpoint_path,
            device='cpu'
        )
        
        # 추론 실행
        print("🔧 추론 실행")
        results_gen = inferencer(test_image, show=False, return_vis=False)
        results = list(results_gen)
        
        print(f"✅ 추론 완료: {len(results)} 개 이미지")
        
        if len(results) > 0:
            result = results[0]
            
            # 올바른 키 찾기
            predictions_key = None
            for key in result.keys():
                if hasattr(result[key], 'pred_instances'):
                    predictions_key = key
                    break
            
            if predictions_key:
                print(f"📍 예측 결과 키: '{predictions_key}'")
                prediction_result = result[predictions_key]
                pred_instances = prediction_result.pred_instances
                
                print(f"  탐지된 인스턴스: {len(pred_instances)}")
                
                if len(pred_instances) > 0:
                    # 첫 번째 사람 분석
                    instance = pred_instances[0]
                    
                    if hasattr(instance, 'keypoints'):
                        keypoints = instance.keypoints  # shape: (133, 3)
                        scores = instance.keypoint_scores if hasattr(instance, 'keypoint_scores') else keypoints[:, 2]
                        
                        print(f"  키포인트 형태: {keypoints.shape}")
                        print(f"  점수 범위: [{scores.min():.3f}, {scores.max():.3f}]")
                        
                        # 고신뢰도 키포인트
                        high_conf_mask = scores > 0.5
                        high_conf_points = keypoints[high_conf_mask]
                        high_conf_scores = scores[high_conf_mask]
                        
                        print(f"  고신뢰도(>0.5) 키포인트: {len(high_conf_points)}/{len(keypoints)}")
                        
                        if len(high_conf_points) > 0:
                            print(f"  좌표 범위 - X: [{high_conf_points[:, 0].min():.1f}, {high_conf_points[:, 0].max():.1f}]")
                            print(f"  좌표 범위 - Y: [{high_conf_points[:, 1].min():.1f}, {high_conf_points[:, 1].max():.1f}]")
                            
                            # 주요 신체 부위 (COCO 17개 키포인트)
                            body_parts = [
                                "nose", "left_eye", "right_eye", "left_ear", "right_ear",
                                "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
                                "left_wrist", "right_wrist", "left_hip", "right_hip",
                                "left_knee", "right_knee", "left_ankle", "right_ankle"
                            ]
                            
                            print("\n  📍 주요 신체 키포인트 (COCO 17개):")
                            for i, part in enumerate(body_parts):
                                if i < len(keypoints) and scores[i] > 0.3:
                                    x, y = keypoints[i, 0], keypoints[i, 1]
                                    score = scores[i]
                                    print(f"    {part:15}: ({x:6.1f}, {y:6.1f}) conf={score:.3f}")
                            
                            # 바운딩 박스 정보
                            if hasattr(instance, 'bboxes'):
                                bbox = instance.bboxes
                                print(f"\n  📦 바운딩박스: [{bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}]")
                                
                                # 키포인트가 바운딩박스 내에 있는지 확인
                                inside_bbox = (
                                    (high_conf_points[:, 0] >= bbox[0]) & 
                                    (high_conf_points[:, 0] <= bbox[2]) &
                                    (high_conf_points[:, 1] >= bbox[1]) & 
                                    (high_conf_points[:, 1] <= bbox[3])
                                )
                                print(f"  바운딩박스 내 키포인트: {inside_bbox.sum()}/{len(high_conf_points)}")
                                
                        else:
                            print("  ⚠️ 고신뢰도 키포인트 없음")
                    else:
                        print("  ❌ 키포인트 정보 없음")
                else:
                    print("  ⚠️ 탐지된 사람 없음")
            else:
                print("  ❌ 예측 결과를 찾을 수 없음")
                
        print("\n🎨 시각화 결과 저장")
        vis_results = inferencer(test_image, show=False, return_vis=True, out_dir='rtmw_official_output')
        vis_list = list(vis_results)
        if vis_list:
            print("✅ 시각화 저장 완료: rtmw_official_output/")
        
    except Exception as e:
        print(f"❌ 추론 실패: {e}")
        import traceback
        traceback.print_exc()
    finally:
        torch.load = original_load

if __name__ == "__main__":
    analyze_result_structure()
    correct_inference_test()
