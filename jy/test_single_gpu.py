#!/usr/bin/env python3
"""
Fast Multi-ONNX Processor 단일 비디오 GPU 테스트
"""

import os
import torch
from pathlib import Path
from fast_multionnx_processor import FastVideoProcessor

def test_single_video_gpu():
    """단일 비디오 GPU 테스트"""
    print("🚀 단일 비디오 GPU 테스트")
    print("=" * 50)
    
    # GPU 환경 확인
    if torch.cuda.is_available():
        print(f"✅ CUDA 사용 가능")
        print(f"   - GPU 개수: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_properties(i).name
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"   - GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
    else:
        print("❌ CUDA 사용 불가")
        return
    
    # 테스트 비디오 경로
    test_video = "/workspace01/team03/data/mmpose/jy/data/1.Training/videos/11/NIA_SL_WORD0001_REAL11_F.mp4"
    
    if not Path(test_video).exists():
        print(f"❌ 테스트 비디오 없음: {test_video}")
        return
    
    print(f"📹 테스트 비디오: {Path(test_video).name}")
    
    try:
        # GPU 0에서 테스트
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        torch.cuda.set_device(0)
        
        processor = FastVideoProcessor(
            rtmw_model_name="rtmw-dw-x-l_simcc-cocktail14_270e-384x288.onnx",
            yolo_device="cuda:0",
            pose_device="cuda",  # 명시적으로 CUDA 설정
            keypoint_scale=8,
            jpeg_quality=90
        )
        
        print(f"⚡ 비디오 처리 시작...")
        import time
        start_time = time.time()
        
        result = processor.process_video_fast(test_video)
        
        process_time = time.time() - start_time
        
        if result:
            print(f"✅ 처리 완료!")
            print(f"   - 프레임 수: {result['frame_count']}")
            print(f"   - 처리 시간: {process_time:.2f}초")
            print(f"   - 초당 프레임: {result['frame_count']/process_time:.1f} FPS")
            print(f"   - 키포인트 데이터: {len(result['keypoints'])}개")
        else:
            print(f"❌ 처리 실패")
            
    except Exception as e:
        print(f"💥 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_single_video_gpu()
