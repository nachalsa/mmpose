#!/usr/bin/env python3
"""
Simple Fast Multi-ONNX Test
"""

import sys
import os

print("=== Fast Multi-ONNX Processor 테스트 ===")

try:
    # 경로 추가
    current_dir = os.getcwd()
    sys.path.insert(0, current_dir)
    
    print(f"현재 디렉토리: {current_dir}")
    print(f"Python 버전: {sys.version}")
    
    # 임포트 테스트
    from fast_multionnx_processor import FastMultiONNXProcessor
    print("✅ FastMultiONNXProcessor 임포트 성공")
    
    # 프로세서 초기화
    processor = FastMultiONNXProcessor(
        data_root="/workspace01/team03/data/mmpose/jy/data/1.Training",
        output_dir="/workspace01/team03/data/fast_test_output",
        direction="F",
        item_types=["WORD"],
        num_cpu_workers=2,
        enable_hdf5_batch=False
    )
    print("✅ 프로세서 초기화 성공")
    
    # 비디오 수집 테스트
    print("\n🔍 비디오 수집 중...")
    videos = processor.collect_all_videos()
    print(f"✅ 비디오 수집 완료: {len(videos)}개")
    
    if videos:
        print("\n📊 발견된 비디오 (처음 5개):")
        for i, (item_type, item_id, path) in enumerate(videos[:5]):
            print(f"  {i+1}. {item_type}{item_id:04d} -> {os.path.basename(path)}")
        
        if len(videos) > 5:
            print(f"  ... 및 {len(videos) - 5}개 더")
        
        # 커스터마이징 테스트
        print("\n🔧 커스터마이징 테스트:")
        processor.customize_keypoint_scale(16)
        processor.customize_jpeg_quality(95)
        
        print("\n🎉 모든 기본 테스트 성공!")
        print("실제 처리를 원하면 processor.process_videos_parallel() 호출")
        
    else:
        print("⚠️ 처리할 비디오가 없습니다")
        
except Exception as e:
    print(f"❌ 오류 발생: {e}")
    import traceback
    traceback.print_exc()

print("\n=== 테스트 완료 ===")
