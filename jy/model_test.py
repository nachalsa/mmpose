#!/usr/bin/env python3
"""
실제 모델 로딩 및 이미지 추론 테스트
"""

print("🚀 실제 모델 로딩 테스트 시작...")

try:
    # 1. 트래킹 시스템 import
    print("1. 트래킹 시스템 import...")
    from tracking_yolo11l_hybrid import TrackingYOLO11LHybridInferencer
    import cv2
    import numpy as np
    import os
    print("   ✅ 트래킹 시스템 import 성공")
    
    # 2. 테스트 이미지 확인
    print("2. 테스트 이미지 확인...")
    test_image_path = "winter01.jpg"
    if os.path.exists(test_image_path):
        image = cv2.imread(test_image_path)
        if image is not None:
            print(f"   ✅ 테스트 이미지 로드 성공: {image.shape}")
        else:
            print("   ❌ 이미지 읽기 실패")
            exit(1)
    else:
        print("   ❌ 테스트 이미지 파일 없음")
        # 더미 이미지 생성
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(image, (200, 100), (440, 400), (255, 255, 255), -1)  # 흰색 사각형
        print("   ✅ 더미 이미지 생성: (480, 640, 3)")
    
    # 3. 모델 경로 확인
    print("3. 모델 경로 확인...")
    rtmw_config = "../configs/wholebody_2d_keypoint/rtmpose/cocktail14/rtmw-l_8xb320-270e_cocktail14-384x288.py"
    rtmw_checkpoint = "../models/rtmw-x_simcc-cocktail14_pt-ucoco_270e-384x288-f840f204_20231122.pth"
    yolo_model = "../models/yolo11l.pt"
    
    print(f"   - RTMW 설정: {os.path.exists(rtmw_config)}")
    print(f"   - RTMW 체크포인트: {os.path.exists(rtmw_checkpoint)}")
    print(f"   - YOLO11L 모델: {os.path.exists(yolo_model)}")
    
    # 4. 트래킹 추론기 초기화 (XPU 우선 사용)
    print("4. 트래킹 추론기 초기화...")
    print("   (XPU 우선, CPU 폴백으로 초기화)")
    
    inferencer = TrackingYOLO11LHybridInferencer(
        rtmw_config=rtmw_config,
        rtmw_checkpoint=rtmw_checkpoint,
        detection_device="xpu",  # XPU 사용
        pose_device="xpu",       # XPU 사용
        yolo_model_name="yolo11l.pt",
        use_shoulder_tracking=True
    )
    print("   ✅ 트래킹 추론기 초기화 성공!")
    
    # 5. 단일 프레임 처리 테스트
    print("5. 단일 프레임 처리 테스트...")
    import time
    start_time = time.time()
    
    vis_image, results = inferencer.process_frame(image)
    
    process_time = time.time() - start_time
    print(f"   ✅ 프레임 처리 완료: {process_time:.3f}초")
    print(f"   - 검출된 사람: {len(results)}명")
    
    # 6. 결과 분석
    if results:
        print("6. 결과 분석...")
        for i, (keypoints, scores, bbox, track_id) in enumerate(results):
            valid_keypoints = np.sum(scores > 0.3)
            bbox_int = [int(x) for x in bbox]
            print(f"   - 사람 {i+1} (ID: {track_id})")
            print(f"     * 바운딩박스: {bbox_int}")
            print(f"     * 유효 키포인트: {valid_keypoints}/133")
            print(f"     * 평균 신뢰도: {np.mean(scores):.3f}")
    else:
        print("6. 결과: 검출된 사람 없음")
    
    # 7. 결과 이미지 저장
    print("7. 결과 이미지 저장...")
    output_path = "tracking_test_result.jpg"
    cv2.imwrite(output_path, vis_image)
    print(f"   ✅ 결과 저장: {output_path}")
    
    # 8. 성능 통계
    print("8. 성능 통계...")
    stats = inferencer.get_performance_stats()
    for key, value in stats.items():
        print(f"   - {key}: {value:.3f}")
    
    print("\n🎉 실제 모델 테스트 완료!")
    print("=" * 50)
    
except Exception as e:
    print(f"❌ 모델 테스트 실패: {e}")
    import traceback
    traceback.print_exc()
