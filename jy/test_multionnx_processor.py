#!/usr/bin/env python3
"""
Multi-ONNX Streamlined Processor 테스트
"""

import sys
import os
sys.path.insert(0, os.getcwd())

from multionnx_streamlined_processor_complete import BatchProcessor

def test_video_collection():
    """비디오 수집 테스트"""
    print("🧪 비디오 수집 기능 테스트")
    
    processor = BatchProcessor(
        data_root="/workspace01/team03/data/mmpose/jy/data/1.Training",
        output_dir="/workspace01/team03/data/test_output",
        direction="F",
        item_types=["WORD"],
        gpu_batch_size=8,
        num_cpu_workers=2
    )
    
    # 비디오 수집 테스트
    folder_video_data = processor.collect_videos_by_folder()
    
    if folder_video_data:
        print("✅ 비디오 수집 성공!")
        total_videos = sum(len(videos) for videos in folder_video_data.values())
        print(f"📊 총 {total_videos}개 비디오 발견")
        
        # 샘플 출력
        for folder, videos in list(folder_video_data.items())[:3]:
            print(f"   - 폴더 {folder}: {len(videos)}개 비디오")
            if videos:
                sample = videos[0]
                print(f"     예시: {sample[0]}{sample[1]:04d} -> {sample[2]}")
        
        return True
    else:
        print("❌ 비디오 수집 실패")
        return False

def test_mini_processing():
    """미니 처리 테스트 (3개 비디오만)"""
    print("\n🧪 미니 처리 테스트 (최대 3개 비디오)")
    
    processor = BatchProcessor(
        data_root="/workspace01/team03/data/mmpose/jy/data/1.Training",
        output_dir="/workspace01/team03/data/test_output_mini",
        direction="F",
        item_types=["WORD"],
        gpu_batch_size=1,  # 작은 배치
        num_cpu_workers=1  # 최소 워커
    )
    
    # 비디오 수집
    folder_video_data = processor.collect_videos_by_folder()
    
    if not folder_video_data:
        print("❌ 비디오 없음")
        return False
    
    # 처음 3개 비디오만 선택
    limited_data = {}
    video_count = 0
    for folder, videos in folder_video_data.items():
        if video_count >= 3:
            break
        
        take_count = min(3 - video_count, len(videos))
        limited_data[folder] = videos[:take_count]
        video_count += take_count
    
    # 임시로 원본 데이터를 제한된 데이터로 교체
    original_collect = processor.collect_videos_by_folder
    processor.collect_videos_by_folder = lambda: limited_data
    
    try:
        processor.process_all_batches(cleanup_intermediate=False)
        print("✅ 미니 처리 테스트 완료")
        return True
    except Exception as e:
        print(f"❌ 미니 처리 테스트 실패: {e}")
        return False
    finally:
        processor.collect_videos_by_folder = original_collect

if __name__ == "__main__":
    print("🚀 Multi-ONNX Streamlined Processor 테스트 시작")
    
    # 1. 비디오 수집 테스트
    collection_ok = test_video_collection()
    
    if collection_ok:
        # 2. 미니 처리 테스트
        processing_ok = test_mini_processing()
        
        if processing_ok:
            print("\n🎉 모든 테스트 통과! Multi-ONNX Processor가 정상 작동합니다.")
        else:
            print("\n⚠️ 처리 테스트에서 문제 발생")
    else:
        print("\n❌ 비디오 수집부터 실패")
    
    print("\n테스트 완료.")
