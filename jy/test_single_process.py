#!/usr/bin/env python3
"""
Simple Single Process Test for Multi-ONNX Streamlined Processor
"""

import sys
import os
sys.path.insert(0, os.getcwd())

import cv2
import torch
from pathlib import Path
from multionnx_streamlined_processor_complete import StreamlinedVideoProcessor

def test_single_video():
    """단일 비디오 처리 테스트"""
    print("🧪 단일 비디오 처리 테스트")
    
    # 첫 번째 비디오 파일 찾기
    videos_dir = Path("/workspace01/team03/data/mmpose/jy/data/1.Training/videos")
    video_files = list(videos_dir.glob("**/*_F.mp4"))
    
    if not video_files:
        print("❌ 비디오 파일을 찾을 수 없습니다")
        return False
    
    test_video = video_files[0]
    print(f"📹 테스트 비디오: {test_video}")
    
    # 비디오 기본 정보 확인
    cap = cv2.VideoCapture(str(test_video))
    if not cap.isOpened():
        print("❌ 비디오 열기 실패")
        return False
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    print(f"   - 프레임 수: {frame_count}")
    print(f"   - FPS: {fps:.2f}")
    print(f"   - 해상도: {width}x{height}")
    
    # StreamlinedVideoProcessor 생성
    try:
        print("\n🚀 StreamlinedVideoProcessor 초기화 중...")
        processor = StreamlinedVideoProcessor(
            rtmw_model_name="rtmw-dw-x-l_simcc-cocktail14_270e-384x288.onnx",
            yolo_device="cuda" if torch.cuda.is_available() else "cpu",
            pose_device="cuda" if torch.cuda.is_available() else "cpu"
        )
        print("✅ 프로세서 초기화 완료")
        
        # 비디오 처리 실행
        print("\n🎬 비디오 처리 실행 중...")
        result = processor.process_video_to_arrays(str(test_video))
        
        if result:
            print("✅ 비디오 처리 성공!")
            print(f"   - JPEG 프레임: {len(result['jpeg_frames'])}개")
            print(f"   - 키포인트 프레임: {len(result['keypoints'])}개")
            print(f"   - 점수 프레임: {len(result['scores'])}개")
            print(f"   - 총 프레임 수: {result['frame_count']}")
            
            # 첫 번째 프레임의 키포인트 정보
            if result['keypoints'] and result['keypoints'][0]:
                first_keypoints = result['keypoints'][0][0] if result['keypoints'][0] else []
                print(f"   - 첫 번째 프레임 키포인트 수: {len(first_keypoints)}")
            
            return True
        else:
            print("❌ 비디오 처리 실패")
            return False
            
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_onnx_inferencer():
    """ONNX 추론기 직접 테스트"""
    print("\n🧪 ONNX 추론기 직접 테스트")
    
    try:
        from onnx_inferencer import YOLO11LRTMWONNXInferencer as ONNXInferencer
        
        # 추론기 초기화
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"   - 사용 디바이스: {device}")
        
        inferencer = ONNXInferencer(
            rtmw_onnx_path="rtmw-dw-x-l_simcc-cocktail14_270e-384x288.onnx",
            detection_device=device,
            pose_device=device,
            optimize_for_accuracy=True
        )
        print("✅ ONNX 추론기 초기화 완료")
        
        # 테스트 이미지 생성 (더미 프레임)
        import numpy as np
        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # 추론 실행
        result = inferencer.process_frame(test_frame)
        print(f"✅ 추론 실행 완료: {type(result)}")
        
        return True
        
    except Exception as e:
        print(f"❌ ONNX 추론기 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Single Process 테스트 시작")
    print(f"   - CUDA 사용 가능: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   - GPU 개수: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"   - GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # 1. ONNX 추론기 테스트
    onnx_ok = test_onnx_inferencer()
    
    if onnx_ok:
        # 2. 단일 비디오 처리 테스트
        video_ok = test_single_video()
        
        if video_ok:
            print("\n🎉 모든 단일 프로세스 테스트 통과!")
        else:
            print("\n⚠️ 비디오 처리 테스트에서 문제 발생")
    else:
        print("\n❌ ONNX 추론기부터 실패")
    
    print("\n테스트 완료.")
