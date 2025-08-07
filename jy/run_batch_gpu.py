#!/usr/bin/env python3
"""
Simple Batch GPU Runner - 간단한 배치 GPU 실행기
A6000 x2 GPU 배치 256 최적화
"""

from batch_fast_multionnx_processor import BatchFastMultiONNXProcessor

def run_batch_gpu():
    """간단한 배치 GPU 실행"""
    print("🚀 Simple Batch GPU Runner")
    print("⚡ A6000 x2 GPU 배치 256 완전 최적화")
    print("=" * 60)
    
    # 배치 설정
    processor = BatchFastMultiONNXProcessor(
        data_root="/workspace01/team03/data/mmpose/jy/data/1.Training",
        output_dir="/workspace01/team03/data/simple_batch_output",
        direction="F",
        item_types=["WORD"],
        num_cpu_workers=6,       # CPU 워커 최적화
        batch_size=256,          # GPU 배치 256
        yolo_device="auto",
        pose_device="auto",
        keypoint_scale=8,
        jpeg_quality=90,
        enable_hdf5_batch=True   # 배치 파일 생성
    )
    
    # 빠른 테스트 (5개 비디오)
    print("\\n🎯 빠른 배치 테스트 (5개 비디오)")
    processor.process_videos_parallel(max_videos=5)

def run_full_batch():
    """전체 배치 처리"""
    print("🚀 Full Batch GPU Processing")
    print("⚡ A6000 x2 GPU 전체 39,000 비디오 처리")
    print("=" * 60)
    
    processor = BatchFastMultiONNXProcessor(
        data_root="/workspace01/team03/data/mmpose/jy/data/1.Training",
        output_dir="/workspace01/team03/data/full_batch_output",
        direction="F",
        item_types=["WORD"],
        num_cpu_workers=8,
        batch_size=256,
        yolo_device="auto",
        pose_device="auto",
        keypoint_scale=8,
        jpeg_quality=90,
        enable_hdf5_batch=True
    )
    
    choice = input("전체 39,000개 비디오를 처리하시겠습니까? (y/N): ").strip().lower()
    if choice == 'y':
        processor.process_videos_parallel()
    else:
        try:
            count = int(input("처리할 비디오 수 입력: "))
            processor.process_videos_parallel(max_videos=count)
        except ValueError:
            print("기본값 10개로 처리")
            processor.process_videos_parallel(max_videos=10)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "full":
        run_full_batch()
    else:
        run_batch_gpu()
