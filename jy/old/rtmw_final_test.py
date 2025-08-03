#!/usr/bin/env python3
"""
MMPose RTMW 정확한 결과 파싱 및 분석
predictions 리스트에서 실제 키포인트 데이터 추출
"""

import os
import torch
import cv2
import numpy as np
from mmpose.apis import MMPoseInferencer
import json

def parse_rtmw_predictions():
    """RTMW 예측 결과 정확한 파싱"""
    print("=== RTMW 예측 결과 정확한 파싱 ===\n")
    
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
            predictions = result['predictions']
            
            print(f"📍 predictions 구조 분석:")
            print(f"  타입: {type(predictions)}")
            print(f"  길이: {len(predictions)}")
            
            if len(predictions) > 0:
                pred_list = predictions[0]  # 첫 번째 이미지
                print(f"  첫 번째 예측 타입: {type(pred_list)}")
                print(f"  첫 번째 예측 길이: {len(pred_list)}")
                
                if len(pred_list) > 0:
                    first_pred = pred_list[0]  # 첫 번째 사람
                    print(f"  첫 번째 인스턴스 타입: {type(first_pred)}")
                    
                    # 딕셔너리 형태라면 키 확인
                    if hasattr(first_pred, 'keys'):
                        print(f"  인스턴스 키들: {list(first_pred.keys())}")
                        
                        for key, value in first_pred.items():
                            print(f"\n  키 '{key}':")
                            print(f"    타입: {type(value)}")
                            if hasattr(value, 'shape'):
                                print(f"    형태: {value.shape}")
                            elif isinstance(value, (list, tuple)):
                                print(f"    길이: {len(value)}")
                            else:
                                print(f"    값: {value}")
                                
                    # pred_instances 속성이 있는지 확인
                    elif hasattr(first_pred, 'pred_instances'):
                        print(f"  pred_instances 발견!")
                        pred_instances = first_pred.pred_instances
                        analyze_pred_instances(pred_instances)
                        
                    # 직접 속성들 확인
                    else:
                        print(f"  인스턴스 속성들: {dir(first_pred)}")
                        
                        # 주요 속성들 확인
                        for attr in ['keypoints', 'keypoint_scores', 'bboxes', 'bbox_scores']:
                            if hasattr(first_pred, attr):
                                value = getattr(first_pred, attr)
                                print(f"  {attr}: {type(value)}, 형태: {getattr(value, 'shape', 'no shape')}")
                                
                        # 실제 키포인트 데이터 추출 시도
                        if hasattr(first_pred, 'keypoints'):
                            keypoints = first_pred.keypoints
                            print(f"\n🎯 키포인트 분석:")
                            print(f"  키포인트 형태: {keypoints.shape}")
                            
                            if hasattr(first_pred, 'keypoint_scores'):
                                scores = first_pred.keypoint_scores
                                print(f"  점수 형태: {scores.shape}")
                                analyze_keypoints(keypoints, scores)
                            else:
                                # 키포인트에 confidence가 포함된 경우
                                if keypoints.shape[-1] == 3:
                                    coords = keypoints[:, :2]
                                    scores = keypoints[:, 2]
                                    analyze_keypoints(coords, scores)
                                else:
                                    print(f"  점수 정보 없음, 좌표만 분석")
                                    analyze_keypoints(keypoints, None)
                            
                else:
                    print("  ⚠️ 예측 결과 비어있음")
            else:
                print("  ⚠️ predictions 비어있음")
                
    except Exception as e:
        print(f"❌ 파싱 실패: {e}")
        import traceback
        traceback.print_exc()
    finally:
        torch.load = original_load

def analyze_pred_instances(pred_instances):
    """pred_instances 분석"""
    print(f"  인스턴스 수: {len(pred_instances)}")
    
    if len(pred_instances) > 0:
        instance = pred_instances[0]
        print(f"  첫 번째 인스턴스 타입: {type(instance)}")
        
        if hasattr(instance, 'keypoints'):
            keypoints = instance.keypoints
            scores = getattr(instance, 'keypoint_scores', None)
            bbox = getattr(instance, 'bboxes', None)
            
            print(f"  키포인트 형태: {keypoints.shape}")
            if scores is not None:
                print(f"  점수 형태: {scores.shape}")
            if bbox is not None:
                print(f"  바운딩박스: {bbox}")
                
            analyze_keypoints(keypoints, scores)

def analyze_keypoints(keypoints, scores=None):
    """키포인트 상세 분석"""
    print(f"\n📍 키포인트 상세 분석:")
    
    if scores is not None:
        print(f"  점수 범위: [{scores.min():.3f}, {scores.max():.3f}]")
        
        # 고신뢰도 키포인트
        high_conf_mask = scores > 0.5
        medium_conf_mask = (scores > 0.3) & (scores <= 0.5)
        
        high_conf_points = keypoints[high_conf_mask]
        medium_conf_points = keypoints[medium_conf_mask]
        
        print(f"  고신뢰도(>0.5): {len(high_conf_points)}/{len(keypoints)}")
        print(f"  중신뢰도(0.3-0.5): {len(medium_conf_points)}/{len(keypoints)}")
        
        if len(high_conf_points) > 0:
            print(f"  고신뢰도 좌표 범위:")
            print(f"    X: [{high_conf_points[:, 0].min():.1f}, {high_conf_points[:, 0].max():.1f}]")
            print(f"    Y: [{high_conf_points[:, 1].min():.1f}, {high_conf_points[:, 1].max():.1f}]")
            
        # 주요 신체 부위 분석 (COCO 17개 + WholeBody)
        body_parts = [
            "nose", "left_eye", "right_eye", "left_ear", "right_ear",
            "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", 
            "left_wrist", "right_wrist", "left_hip", "right_hip",
            "left_knee", "right_knee", "left_ankle", "right_ankle"
        ]
        
        print(f"\n  📍 주요 신체 부위 (처음 17개):")
        valid_detections = 0
        for i, part in enumerate(body_parts):
            if i < len(keypoints) and scores[i] > 0.3:
                x, y = keypoints[i, 0], keypoints[i, 1]
                score = scores[i]
                print(f"    {part:15}: ({x:6.1f}, {y:6.1f}) conf={score:.3f}")
                valid_detections += 1
                
        print(f"\n  ✅ 유효한 주요 부위 탐지: {valid_detections}/17")
        
        # 전체 키포인트 통계
        all_valid = scores > 0.3
        if all_valid.sum() > 0:
            valid_keypoints = keypoints[all_valid]
            print(f"  전체 유효 키포인트: {all_valid.sum()}/133")
            print(f"  전체 좌표 범위:")
            print(f"    X: [{valid_keypoints[:, 0].min():.1f}, {valid_keypoints[:, 0].max():.1f}]")
            print(f"    Y: [{valid_keypoints[:, 1].min():.1f}, {valid_keypoints[:, 1].max():.1f}]")
    else:
        print(f"  점수 정보 없음")
        print(f"  좌표 범위:")
        print(f"    X: [{keypoints[:, 0].min():.1f}, {keypoints[:, 0].max():.1f}]")
        print(f"    Y: [{keypoints[:, 1].min():.1f}, {keypoints[:, 1].max():.1f}]")

def check_visualization_result():
    """시각화 결과 확인"""
    print("\n" + "="*50)
    print("🎨 시각화 결과 확인")
    
    vis_path = "rtmw_official_output/visualizations/winter01.jpg"
    if os.path.exists(vis_path):
        vis_img = cv2.imread(vis_path)
        print(f"✅ 시각화 이미지 확인: {vis_img.shape}")
        print(f"  저장 경로: {vis_path}")
    else:
        print(f"❌ 시각화 이미지 없음: {vis_path}")

if __name__ == "__main__":
    parse_rtmw_predictions()
    check_visualization_result()
