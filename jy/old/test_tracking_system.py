#!/usr/bin/env python3
"""
구조화된 트래킹 시스템 테스트
"""

import cv2
import numpy as np
import time
from tracking_yolo11l_hybrid import TrackingYOLO11LHybridInferencer, test_webcam_tracking

def test_image_tracking():
    """이미지 기반 트래킹 테스트"""
    print("🔍 이미지 기반 트래킹 테스트 시작...")
    
    # 하이브리드 추론기 초기화
    try:
        inferencer = TrackingYOLO11LHybridInferencer(
            rtmw_config="../configs/wholebody_2d_keypoint/rtmw/coco-wholebody/rtmw-l_8xb64-270e_coco-wholebody-384x288.py",
            rtmw_checkpoint="../models/rtmw-x_simcc-cocktail14_pt-ucoco_270e-384x288-f840f204_20231122.pth",
            detection_device="auto",
            pose_device="auto",
            use_shoulder_tracking=True
        )
        print("✅ 트래킹 시스템 초기화 성공")
    except Exception as e:
        print(f"❌ 트래킹 시스템 초기화 실패: {e}")
        return False
    
    # 테스트 이미지 로드
    image_path = "winter01.jpg"
    try:
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ 이미지 로드 실패: {image_path}")
            return False
        print(f"✅ 이미지 로드 성공: {image.shape}")
    except Exception as e:
        print(f"❌ 이미지 로드 오류: {e}")
        return False
    
    # 프레임 처리 테스트 (여러 프레임 시뮬레이션)
    try:
        print("\n📸 프레임 처리 테스트 시작...")
        
        for frame_idx in range(5):  # 5프레임 테스트
            print(f"   프레임 {frame_idx + 1}/5 처리 중...")
            
            start_time = time.time()
            vis_image, results = inferencer.process_frame(image)
            process_time = time.time() - start_time
            
            print(f"   - 처리 시간: {process_time:.3f}초")
            print(f"   - 검출된 사람: {len(results)}명")
            
            # 결과 분석
            for keypoints, scores, bbox, track_id in results:
                valid_keypoints = np.sum(scores > 0.3)
                print(f"   - ID {track_id}: {valid_keypoints}개 키포인트, 바운딩박스: {[int(x) for x in bbox]}")
            
            # 결과 이미지 저장
            output_path = f"tracking_test_frame_{frame_idx + 1}.jpg"
            cv2.imwrite(output_path, vis_image)
            print(f"   - 결과 저장: {output_path}")
        
        # 성능 통계
        stats = inferencer.get_performance_stats()
        print(f"\n📊 성능 통계:")
        for key, value in stats.items():
            print(f"   - {key}: {value:.3f}")
        
        print("✅ 이미지 트래킹 테스트 완료")
        return True
        
    except Exception as e:
        print(f"❌ 프레임 처리 오류: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_simple_tracking_functionality():
    """기본 트래킹 기능 단위 테스트"""
    print("\n🧪 기본 트래킹 기능 단위 테스트...")
    
    try:
        from tracking_yolo11l_hybrid import PersonTracker, SignLanguageShoulderTracker
        
        # 기본 트래커 테스트
        print("1. PersonTracker 테스트...")
        basic_tracker = PersonTracker(
            track_id=0,
            initial_bbox=[100, 100, 300, 400],
            stability_factor=0.3
        )
        
        # 더미 키포인트 데이터
        dummy_keypoints = np.random.rand(133, 2) * 300 + 100
        dummy_scores = np.random.rand(133) * 0.8 + 0.2
        
        # 트래커 업데이트 테스트
        updated_bbox = basic_tracker.update(dummy_keypoints, dummy_scores, (480, 640))
        print(f"   ✅ 기본 트래커 업데이트: {[int(x) for x in updated_bbox]}")
        
        # 수화 특화 트래커 테스트
        print("2. SignLanguageShoulderTracker 테스트...")
        shoulder_tracker = SignLanguageShoulderTracker(
            track_id=1,
            initial_bbox=[100, 100, 300, 400],
            stability_factor=0.3
        )
        
        # 어깨 키포인트 시뮬레이션 (키포인트 5, 6)
        dummy_keypoints[5] = [150, 120]  # 왼쪽 어깨
        dummy_keypoints[6] = [250, 125]  # 오른쪽 어깨
        dummy_scores[5] = 0.9
        dummy_scores[6] = 0.9
        
        updated_bbox = shoulder_tracker.update(dummy_keypoints, dummy_scores, (480, 640))
        print(f"   ✅ 어깨 트래커 업데이트: {[int(x) for x in updated_bbox]}")
        
        # 어깨 움직임 특징 추출 테스트
        features = shoulder_tracker.get_shoulder_movement_features()
        print(f"   ✅ 어깨 움직임 특징: {len(features)}개 특징 추출됨")
        
        print("✅ 기본 트래킹 기능 테스트 완료")
        return True
        
    except Exception as e:
        print(f"❌ 단위 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🤟 구조화된 트래킹 시스템 테스트")
    print("=" * 50)
    
    # 1. 단위 테스트
    unit_test_result = test_simple_tracking_functionality()
    
    # 2. 이미지 테스트
    image_test_result = test_image_tracking()
    
    # 3. 웹캠 테스트 (선택적)
    print(f"\n📋 테스트 결과 요약:")
    print(f"   - 단위 테스트: {'✅ 성공' if unit_test_result else '❌ 실패'}")
    print(f"   - 이미지 테스트: {'✅ 성공' if image_test_result else '❌ 실패'}")
    
    if unit_test_result and image_test_result:
        print(f"\n🎉 모든 테스트 통과!")
        print(f"웹캠 실시간 테스트를 원하시면 다음 명령어를 실행하세요:")
        print(f"python3 -c \"from tracking_yolo11l_hybrid import test_webcam_tracking; test_webcam_tracking()\"")
    else:
        print(f"\n⚠️ 일부 테스트 실패 - 문제를 확인해주세요.")
