#!/usr/bin/env python3
"""
Batch Fast Multi-ONNX GPU 배치 256 테스트
"""

import os
import torch
from pathlib import Path
from batch_fast_multionnx_processor import BatchFastMultiONNXProcessor

def test_batch_gpu_processing():
    """GPU 배치 256 처리 테스트"""
    print("🚀 GPU 배치 256 처리 테스트")
    print("=" * 60)
    
    # GPU 환경 확인
    if torch.cuda.is_available():
        print(f"✅ CUDA 사용 가능")
        print(f"   - GPU 개수: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_properties(i).name
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"   - GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
            
        # 총 VRAM 계산
        total_vram = sum(torch.cuda.get_device_properties(i).total_memory for i in range(torch.cuda.device_count())) / 1024**3
        print(f"   - 총 VRAM: {total_vram:.1f}GB")
        
        # 배치 256에 필요한 대략적인 메모리 계산
        # 384x288x3 = 331,776 pixels per frame
        # 256 frames = ~85MB per batch (float32)
        batch_memory_mb = (384 * 288 * 3 * 256 * 4) / (1024 * 1024)  # float32
        print(f"   - 예상 배치 메모리: {batch_memory_mb:.1f}MB (256 프레임)")
        
    else:
        print("❌ CUDA 사용 불가")
        return
    
    print("\\n🔧 배치 최적화 프로세서 초기화...")
    
    # 배치 최적화 설정
    processor = BatchFastMultiONNXProcessor(
        data_root="/workspace01/team03/data/mmpose/jy/data/1.Training",
        output_dir="/workspace01/team03/data/batch_gpu_test_output",
        direction="F",
        item_types=["WORD"],
        num_cpu_workers=4,      # 테스트용 적은 워커
        batch_size=256,         # GPU 배치 크기 256
        yolo_device="auto",
        pose_device="auto",
        keypoint_scale=8,
        jpeg_quality=90,
        enable_hdf5_batch=False # 테스트용 배치 비활성화
    )
    
    print("\\n📊 비디오 수집 중...")
    all_videos = processor.collect_all_videos()
    
    if not all_videos:
        print("❌ 테스트할 비디오가 없습니다")
        return
    
    print(f"📹 총 {len(all_videos)}개 비디오 발견")
    
    # 첫 3개 비디오만 테스트
    test_videos = all_videos[:3]
    print(f"🎯 테스트 대상: {len(test_videos)}개 비디오 (배치 256)")
    
    for i, (item_type, item_id, video_path) in enumerate(test_videos):
        print(f"   {i+1}. {item_type}{item_id:04d}: {Path(video_path).name}")
    
    print("\\n🚀 배치 GPU 처리 시작!")
    print(f"   - A6000 x2 GPU 병렬 처리")
    print(f"   - 배치 크기: 256 프레임")
    print(f"   - VRAM 최적화 활성화")
    print("="*60)
    
    import time
    start_time = time.time()
    
    # 배치 병렬 처리 테스트
    processor.process_videos_parallel(max_videos=3)
    
    total_time = time.time() - start_time
    
    print(f"\\n⏱️ 배치 처리 총 시간: {total_time:.2f}초")
    print(f"⚡ 평균 비디오당: {total_time/len(test_videos):.2f}초")
    print(f"🎯 배치 256 GPU 최적화 완료!")

if __name__ == "__main__":
    test_batch_gpu_processing()
