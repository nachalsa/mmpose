#!/usr/bin/env python3
"""RTMW-x 모델을 사용한 실제 이미지 테스트"""

import cv2
import numpy as np
from pose_estimator import RTMWXEstimator
from visualizer import RTMWVisualizer

def draw_keypoints_simple(image, keypoints):
    """간단한 키포인트 시각화"""
    output = image.copy()
    
    # 키포인트 그리기
    for i, (x, y) in enumerate(keypoints):
        if x > 0 and y > 0:  # 유효한 키포인트만
            cv2.circle(output, (int(x), int(y)), 3, (0, 255, 0), -1)
            # 키포인트 번호 표시 (선택사항)
            if i < 17:  # 주요 신체 키포인트만
                cv2.putText(output, str(i), (int(x+5), int(y-5)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    
    return output

def main():
    """실제 이미지로 RTMW 테스트"""
    print("=== 실제 이미지 RTMW 테스트 ===")
    
    # RTMW 추정기 초기화 (실제 모델 경로 사용)
    model_path = "../models/rtmw-x_simcc-cocktail14_pt-ucoco_270e-384x288-f840f204_20231122.pth"
    print(f"🔧 모델 로드 중: {model_path}")
    estimator = RTMWXEstimator(model_path, device='cpu')
    print(f"✅ 추정기 초기화 완료")
    
    # 이미지 로드
    image_path = "winter01.jpg"
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"❌ 이미지를 찾을 수 없습니다: {image_path}")
        return
    
    print(f"✅ 이미지 로드: {image.shape}")
    
    # 전체 이미지를 바운딩박스로 사용 (사람 전체)
    h, w = image.shape[:2]
    bbox = [50, 50, w-50, h-50]  # 여백을 둔 전체 영역
    
    # 포즈 추정
    print("🧪 포즈 추정 실행...")
    keypoints = estimator.estimate_pose(image, bbox)
    
    if keypoints is not None and len(keypoints) > 0:
        print(f"✅ 키포인트 감지: {len(keypoints)}개")
        
        # 결과 시각화
        output_image = draw_keypoints_simple(image.copy(), keypoints)
        
        # 결과 저장
        output_path = "rtmw_result_real_test.jpg"
        cv2.imwrite(output_path, output_image)
        print(f"✅ 결과 저장: {output_path}")
        
        # 키포인트 통계 출력
        valid_keypoints = keypoints[keypoints[:, 0] > 0]
        print(f"📊 유효 키포인트: {len(valid_keypoints)}/{len(keypoints)}개")
        
        if len(valid_keypoints) > 0:
            print(f"   - X 좌표 범위: [{valid_keypoints[:, 0].min():.1f}, {valid_keypoints[:, 0].max():.1f}]")
            print(f"   - Y 좌표 범위: [{valid_keypoints[:, 1].min():.1f}, {valid_keypoints[:, 1].max():.1f}]")
        
        # 주요 키포인트 출력 (몸, 얼굴, 손)
        keypoint_names = [
            "nose", "left_eye", "right_eye", "left_ear", "right_ear",
            "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", 
            "left_wrist", "right_wrist", "left_hip", "right_hip"
        ]
        
        print("\n📍 주요 키포인트 위치:")
        for i, name in enumerate(keypoint_names[:13]):
            if i < len(keypoints):
                x, y = keypoints[i][:2]
                print(f"   - {name}: ({x:.1f}, {y:.1f})")
    else:
        print("❌ 키포인트를 감지하지 못했습니다.")

if __name__ == "__main__":
    main()
