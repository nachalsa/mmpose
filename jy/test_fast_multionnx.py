#!/usr/bin/env python3
"""
Fast Multi-ONNX Processor 테스트
빠른 동작 확인용
"""

import sys
import os
sys.path.insert(0, os.getcwd())

from fast_multionnx_processor import FastMultiONNXProcessor

def test_fast_processor():
    """Fast Multi-ONNX Processor 테스트"""
    print("🧪 Fast Multi-ONNX Processor 테스트 시작")
    
    # 프로세서 초기화
    processor = FastMultiONNXProcessor(
        data_root="/workspace01/team03/data/mmpose/jy/data/1.Training",
        output_dir="/workspace01/team03/data/fast_test_output",
        direction="F",
        item_types=["WORD"],
        num_cpu_workers=2,  # 테스트용 최소 워커
        yolo_device="auto",
        pose_device="auto",
        keypoint_scale=8,
        jpeg_quality=90,
        enable_hdf5_batch=False  # 빠른 테스트를 위해 HDF5 비활성화
    )
    
    print("\n🔍 비디오 수집 테스트")
    all_videos = processor.collect_all_videos()
    
    if all_videos:
        print(f"✅ {len(all_videos)}개 비디오 발견!")
        
        # 처음 몇 개만 샘플 출력
        for i, (item_type, item_id, video_path) in enumerate(all_videos[:5]):
            print(f"   {i+1}. {item_type}{item_id:04d} -> {video_path}")
        
        if len(all_videos) > 5:
            print(f"   ... 및 {len(all_videos) - 5}개 더")
        
        # 빠른 테스트 (최대 3개만)
        print(f"\n🚀 빠른 처리 테스트 (최대 3개 비디오)")
        processor.process_videos_parallel(max_videos=3)
        
        return True
    else:
        print("❌ 비디오를 찾을 수 없습니다")
        return False

def test_customization():
    """커스터마이징 기능 테스트"""
    print("\n🔧 커스터마이징 기능 테스트")
    
    processor = FastMultiONNXProcessor()
    
    # 설정 변경 테스트
    processor.customize_keypoint_scale(16)
    processor.customize_jpeg_quality(95)
    processor.customize_output_dir("/workspace01/team03/data/custom_output")
    processor.enable_debug_mode()
    
    print("✅ 모든 커스터마이징 옵션 테스트 완료")

if __name__ == "__main__":
    print("🚀 Fast Multi-ONNX Processor 테스트")
    
    # 1. 메인 기능 테스트
    success = test_fast_processor()
    
    # 2. 커스터마이징 테스트
    if success:
        test_customization()
        print("\n🎉 모든 테스트 완료! Fast Multi-ONNX Processor가 정상 작동합니다.")
    else:
        print("\n❌ 기본 테스트 실패")
