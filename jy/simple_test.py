#!/usr/bin/env python3
"""
간단한 트래킹 시스템 테스트
"""

print("🔍 트래킹 시스템 간단 테스트 시작...")

try:
    # 1. 기본 라이브러리 테스트
    print("1. 기본 라이브러리 import 테스트...")
    import cv2
    import numpy as np
    import torch
    print(f"   ✅ OpenCV: {cv2.__version__}")
    print(f"   ✅ NumPy: {np.__version__}")
    print(f"   ✅ PyTorch: {torch.__version__}")
    
    # 2. MMPose 라이브러리 테스트
    print("2. MMPose 라이브러리 테스트...")
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from mmpose.apis import init_model
    print("   ✅ MMPose import 성공")
    
    # 3. YOLO 라이브러리 테스트
    print("3. YOLO 라이브러리 테스트...")
    from ultralytics import YOLO
    print("   ✅ Ultralytics YOLO import 성공")
    
    # 4. 트래킹 모듈 테스트
    print("4. 트래킹 모듈 import 테스트...")
    from tracking_yolo11l_hybrid import (
        PersonTracker, 
        SignLanguageShoulderTracker,
        TrackingYOLO11LHybridInferencer,
        check_xpu_availability
    )
    print("   ✅ 트래킹 모듈 import 성공")
    
    # 5. XPU 가용성 확인
    print("5. XPU 가용성 확인...")
    xpu_available = check_xpu_availability()
    print(f"   ✅ XPU 상태: {xpu_available}")
    
    # 6. 기본 트래커 인스턴스 생성 테스트
    print("6. 기본 트래커 생성 테스트...")
    basic_tracker = PersonTracker(
        track_id=0,
        initial_bbox=[100, 100, 300, 400],
        stability_factor=0.3
    )
    print(f"   ✅ 기본 트래커 생성 성공 (ID: {basic_tracker.track_id})")
    
    # 7. 수화 특화 트래커 생성 테스트
    print("7. 수화 특화 트래커 생성 테스트...")
    shoulder_tracker = SignLanguageShoulderTracker(
        track_id=1,
        initial_bbox=[100, 100, 300, 400],
        stability_factor=0.3
    )
    print(f"   ✅ 수화 트래커 생성 성공 (ID: {shoulder_tracker.track_id})")
    
    # 8. 더미 키포인트 테스트
    print("8. 더미 키포인트 처리 테스트...")
    dummy_keypoints = np.random.rand(133, 2) * 300 + 100
    dummy_scores = np.random.rand(133) * 0.8 + 0.2
    
    # 어깨 키포인트 설정
    dummy_keypoints[5] = [150, 120]  # 왼쪽 어깨
    dummy_keypoints[6] = [250, 125]  # 오른쪽 어깨
    dummy_scores[5] = 0.9
    dummy_scores[6] = 0.9
    
    # 트래커 업데이트 테스트
    updated_bbox = basic_tracker.update(dummy_keypoints, dummy_scores, (480, 640))
    print(f"   ✅ 기본 트래커 업데이트: {[int(x) for x in updated_bbox]}")
    
    updated_bbox_shoulder = shoulder_tracker.update(dummy_keypoints, dummy_scores, (480, 640))
    print(f"   ✅ 수화 트래커 업데이트: {[int(x) for x in updated_bbox_shoulder]}")
    
    print("\n🎉 모든 기본 테스트 통과!")
    print("=" * 50)
    print("다음 단계: 실제 모델 로딩 테스트")
    
except Exception as e:
    print(f"❌ 테스트 실패: {e}")
    import traceback
    traceback.print_exc()
