#!/usr/bin/env python3
"""
Ultra Fast Processor 소규모 실행 테스트
실제 비디오 처리 파이프라인 검증 (3개 비디오만)
"""

import sys
import os
sys.path.append('/workspace01/team03/data/mmpose/jy')

from ultra_fast_processor_complete import UltraFastBatchProcessor
import time
from pathlib import Path

def create_small_test_data():
    """테스트용 소량의 비디오 데이터 준비"""
    source_dir = Path("/workspace01/team03/data/mmpose/jy/data/1.Training/videos/11")
    test_dir = Path("/tmp/test_mini_videos/1.Training/videos/11")
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # 처음 3개 비디오만 복사
    source_videos = list(source_dir.glob("*_F.mp4"))[:3]
    
    print("🔄 테스트용 소량 데이터 준비...")
    for video in source_videos:
        target = test_dir / video.name
        if not target.exists():
            import shutil
            shutil.copy2(video, target)
            print(f"   📋 복사: {video.name}")
    
    test_videos = list(test_dir.glob("*_F.mp4"))
    print(f"✅ 테스트 데이터 준비 완료: {len(test_videos)}개 비디오")
    return str(test_dir.parent.parent.parent)

def test_mini_processing():
    """초소형 처리 테스트"""
    print("\n🚀 Mini Processing 테스트 시작")
    print("=" * 60)
    
    # 테스트 데이터 준비
    test_data_root = create_small_test_data()
    
    # 프로세서 초기화 (최소 설정)
    processor = UltraFastBatchProcessor(
        data_root=test_data_root,
        output_dir="/tmp/mini_ultra_output",
        direction="F",
        item_types=["WORD"],
        num_gpu_workers=1,      # 단일 GPU
        gpu_batch_size=8,       # 작은 배치
        num_cpu_workers=2,      # 최소 CPU 워커
        num_video_loaders=1     # 단일 로더
    )
    
    print(f"🎯 테스트 설정:")
    print(f"   데이터 루트: {test_data_root}")
    print(f"   출력 디렉토리: /tmp/mini_ultra_output")
    print(f"   GPU 워커: 1개 (배치 크기: 8)")
    print(f"   CPU 워커: 2개")
    
    # 비디오 수집 확인
    all_videos = processor.collect_all_videos()
    print(f"   발견된 비디오: {len(all_videos)}개")
    
    if len(all_videos) == 0:
        print("❌ 처리할 비디오가 없습니다!")
        return False
    
    try:
        # 실제 처리 시작
        print("\n🔥 실제 처리 시작...")
        start_time = time.time()
        
        processor.process_all_videos_ultra_fast()
        
        elapsed_time = time.time() - start_time
        print(f"\n⏱️ 총 처리 시간: {elapsed_time:.2f}초")
        print(f"🚀 평균 속도: {len(all_videos)/elapsed_time:.2f} 비디오/초")
        
        # 결과 확인
        output_dir = Path("/tmp/mini_ultra_output/video_processing")
        if output_dir.exists():
            processed_items = list(output_dir.iterdir())
            print(f"✅ 처리된 아이템: {len(processed_items)}개")
            
            # 첫 번째 아이템 상세 확인
            if processed_items:
                first_item = processed_items[0]
                print(f"\n📁 첫 번째 처리 결과: {first_item.name}")
                
                files = list(first_item.glob("*"))
                for file in files:
                    if file.is_file():
                        size_mb = file.stat().st_size / (1024*1024)
                        print(f"   - {file.name}: {size_mb:.2f}MB")
            
            return len(processed_items) == len(all_videos)
        else:
            print("❌ 출력 디렉토리가 생성되지 않았습니다")
            return False
            
    except Exception as e:
        print(f"❌ 처리 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """메인 실행"""
    print("🧪 Ultra Fast Processor - Mini 실행 테스트")
    print("=" * 80)
    print("목표: 3개 비디오로 전체 파이프라인 검증")
    
    success = test_mini_processing()
    
    print("\n" + "=" * 80)
    if success:
        print("🎉 Mini 처리 테스트 성공!")
        print("   전체 처리 파이프라인이 정상 작동합니다!")
        print("   이제 대량 처리를 안전하게 실행할 수 있습니다.")
    else:
        print("❌ Mini 처리 테스트 실패!")
        print("   코드 수정이 필요합니다.")
    
    return success

if __name__ == "__main__":
    main()
