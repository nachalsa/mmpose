#!/usr/bin/env python3
"""
Fast Multi-ONNX 실제 처리 테스트 (5개 비디오)
"""

import sys
import os

print("=== Fast Multi-ONNX 실제 처리 테스트 ===")

try:
    # 경로 추가
    current_dir = os.getcwd()
    sys.path.insert(0, current_dir)
    
    from fast_multionnx_processor import FastMultiONNXProcessor
    print("✅ FastMultiONNXProcessor 임포트 성공")
    
    # 프로세서 초기화
    processor = FastMultiONNXProcessor(
        data_root="/workspace01/team03/data/mmpose/jy/data/1.Training",
        output_dir="/workspace01/team03/data/fast_processing_test",
        direction="F",
        item_types=["WORD"],
        num_cpu_workers=2,
        enable_hdf5_batch=False,  # 빠른 테스트를 위해 HDF5 비활성화
        keypoint_scale=8,
        jpeg_quality=90
    )
    print("✅ 프로세서 초기화 성공")
    
    print("\n🚀 실제 처리 시작 (최대 5개 비디오)")
    print("=" * 50)
    
    # 실제 병렬 처리 실행
    processor.process_videos_parallel(max_videos=5)
    
    print("\n🎉 실제 처리 테스트 완료!")
    
except Exception as e:
    print(f"❌ 오류 발생: {e}")
    import traceback
    traceback.print_exc()

print("\n=== 실제 처리 테스트 완료 ===")
