#!/usr/bin/env python3
"""
Fast Multi-ONNX Processor GPU 테스트
"""

import os
import sys
import time
from pathlib import Path
from fast_multionnx_processor import FastMultiONNXProcessor

def test_gpu_performance():
    """GPU 성능 테스트"""
    print("🚀 Fast Multi-ONNX GPU 성능 테스트")
    print("=" * 60)
    
    # 기본 설정
    processor = FastMultiONNXProcessor(
        data_root="/workspace01/team03/data/mmpose/jy/data/1.Training",
        output_dir="/workspace01/team03/data/fast_gpu_test_output",
        direction="F",
        item_types=["WORD"],
        num_cpu_workers=4,  # 테스트용 적은 워커
        yolo_device="auto",
        pose_device="auto",
        keypoint_scale=8,
        jpeg_quality=90,
        enable_hdf5_batch=False  # 테스트용 배치 비활성화
    )
    
    print("📊 비디오 수집 중...")
    all_videos = processor.collect_all_videos()
    
    if not all_videos:
        print("❌ 테스트할 비디오가 없습니다")
        return
    
    print(f"📹 발견된 비디오: {len(all_videos)}개")
    
    # 첫 5개 비디오만 테스트
    test_videos = all_videos[:5]
    print(f"🎯 테스트 대상: {len(test_videos)}개 비디오")
    
    for i, (item_type, item_id, video_path) in enumerate(test_videos):
        print(f"   {i+1}. {item_type}{item_id:04d}: {Path(video_path).name}")
    
    start_time = time.time()
    
    # 병렬 처리 테스트
    processor.process_videos_parallel(max_videos=5)
    
    total_time = time.time() - start_time
    
    print(f"⏱️ 총 처리 시간: {total_time:.2f}초")
    print(f"⚡ 평균 비디오당: {total_time/len(test_videos):.2f}초")

if __name__ == "__main__":
    test_gpu_performance()
