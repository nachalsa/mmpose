#!/usr/bin/env python3
"""
Enhanced Pose Visualizer 테스트 스크립트
"""

import os
import cv2
import numpy as np
from pathlib import Path
from enhanced_pose_visualizer import EnhancedPoseVisualizer

def create_test_image():
    """테스트용 간단한 이미지 생성"""
    # 640x480 흰색 배경 이미지 생성
    img = np.ones((480, 640, 3), dtype=np.uint8) * 255
    
    # 간단한 사람 형태 그리기 (테스트용)
    # 머리
    cv2.circle(img, (320, 120), 40, (100, 100, 100), -1)
    # 몸통
    cv2.rectangle(img, (280, 160), (360, 300), (100, 100, 100), -1)
    # 팔
    cv2.rectangle(img, (240, 180), (280, 260), (100, 100, 100), -1)
    cv2.rectangle(img, (360, 180), (400, 260), (100, 100, 100), -1)
    # 다리
    cv2.rectangle(img, (290, 300), (320, 420), (100, 100, 100), -1)
    cv2.rectangle(img, (330, 300), (360, 420), (100, 100, 100), -1)
    
    return img

def test_enhanced_visualizer():
    """Enhanced Pose Visualizer 테스트"""
    print("🧪 Enhanced Pose Visualizer 테스트 시작")
    
    # 테스트 디렉토리 생성
    test_dir = Path("test_images")
    test_dir.mkdir(exist_ok=True)
    
    # 테스트 이미지 생성
    test_img = create_test_image()
    test_img_path = test_dir / "test_person.jpg"
    cv2.imwrite(str(test_img_path), test_img)
    print(f"✅ 테스트 이미지 생성: {test_img_path}")
    
    try:
        # 1. 추론기 없이 이미지 시각화 테스트 (HDF5가 없는 경우)
        print("\n📸 1. 이미지 시각화 테스트 (추론기 없음)")
        visualizer = EnhancedPoseVisualizer(str(test_img_path), use_inferencer=False)
        
        output_dir = Path("test_output")
        output_dir.mkdir(exist_ok=True)
        
        # 시각화 실행 (포즈 데이터는 없지만 원본 이미지는 복사됨)
        visualizer.visualize_images(output_dir=str(output_dir))
        print("✅ 추론기 없는 이미지 시각화 완료")
        
        # 2. 추론기와 함께 이미지 시각화 테스트
        print("\n🤖 2. 이미지 시각화 테스트 (추론기 사용)")
        try:
            visualizer_with_ai = EnhancedPoseVisualizer(str(test_img_path), use_inferencer=True)
            visualizer_with_ai.visualize_images(output_dir=str(output_dir))
            print("✅ 추론기와 함께 이미지 시각화 완료")
        except Exception as e:
            print(f"⚠️ 추론기 테스트 실패 (정상): {e}")
        
        # 3. 디렉토리 모드 테스트
        print(f"\n📁 3. 디렉토리 시각화 테스트")
        visualizer_dir = EnhancedPoseVisualizer(str(test_dir), use_inferencer=False)
        visualizer_dir.visualize_images(output_dir=str(output_dir))
        print("✅ 디렉토리 시각화 완료")
        
        # 4. HDF5 파일이 있는 경우 테스트
        print(f"\n🗂️ 4. HDF5 파일 검색")
        hdf5_files = list(Path(".").glob("*_frames.h5"))
        if hdf5_files:
            frames_h5 = hdf5_files[0]
            poses_h5 = str(frames_h5).replace("_frames.h5", "_poses.h5")
            
            if Path(poses_h5).exists():
                print(f"   발견: {frames_h5} + {poses_h5}")
                try:
                    hdf5_visualizer = EnhancedPoseVisualizer((str(frames_h5), poses_h5))
                    print("✅ HDF5 시각화기 초기화 성공")
                    
                    # 첫 번째 비디오 시각화 (최대 5프레임)
                    if hdf5_visualizer.video_ids:
                        first_video = hdf5_visualizer.video_ids[0]
                        hdf5_visualizer.visualize_hdf5_video(
                            first_video, 
                            output_dir=str(output_dir),
                            max_frames=5
                        )
                        print(f"✅ HDF5 시각화 완료: {first_video}")
                except Exception as e:
                    print(f"⚠️ HDF5 테스트 실패: {e}")
            else:
                print(f"   poses 파일 없음: {poses_h5}")
        else:
            print("   HDF5 파일 없음")
        
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n🎯 테스트 결과:")
    print(f"   - 테스트 이미지: {test_img_path}")
    print(f"   - 출력 디렉토리: {output_dir}")
    print(f"   - 출력 파일들:")
    
    if output_dir.exists():
        for output_file in output_dir.glob("*"):
            print(f"     * {output_file.name}")

if __name__ == "__main__":
    test_enhanced_visualizer()
