#!/usr/bin/env python3
"""
🚀 Ultra Fast Processor 테스트 스크립트
A6000 x2 환경에서 최적 성능 검증
"""

import os
import time
from pathlib import Path
from ultra_fast_processor import UltraFastVideoProcessor

def main():
    print("🚀 Ultra Fast Processor 테스트 시작")
    print("=" * 80)
    
    # 테스트할 비디오 파일 찾기
    test_videos = []
    data_dirs = ["/workspace01/team03/data/07", "/workspace01/team03/data/08"]
    
    for data_dir in data_dirs:
        if os.path.exists(data_dir):
            for ext in ['*.mp4', '*.avi', '*.mov']:
                videos = list(Path(data_dir).glob(f"**/{ext}"))
                test_videos.extend(videos[:2])  # 각 디렉토리에서 최대 2개
                if len(test_videos) >= 3:  # 총 3개로 제한
                    break
    
    if not test_videos:
        print("❌ 테스트할 비디오 파일을 찾을 수 없습니다.")
        return
        
    print(f"📹 테스트 대상 비디오: {len(test_videos)}개")
    for video in test_videos:
        print(f"   - {video}")
    
    # 프로세서 초기화
    print("\n🔧 Ultra Fast Processor 초기화 중...")
    
    processor = UltraFastVideoProcessor(
        gpu_batch_size=512,      # 추천 배치 크기 사용
        num_gpu_workers=2,       # A6000 x2 활용
        num_cpu_workers=32,      # CPU 코어의 약 30% 활용
        num_video_loaders=8,     # 비디오 로딩 병렬화
        accuracy_optimization=True
    )
    
    print("✅ 초기화 완료")
    
    # 각 비디오 처리 테스트
    for i, video_path in enumerate(test_videos, 1):
        print(f"\n🎯 테스트 {i}/{len(test_videos)}: {video_path.name}")
        print("-" * 60)
        
        output_path = f"/workspace01/team03/data/test_output_{i}.h5"
        
        start_time = time.time()
        
        try:
            # 처리 실행 (첫 100프레임만 테스트)
            result = processor.process_video(
                video_path=str(video_path),
                output_path=output_path,
                max_frames=100,  # 빠른 테스트를 위해 제한
                skip_existing=False
            )
            
            elapsed_time = time.time() - start_time
            
            if result["success"]:
                fps = result["frames_processed"] / elapsed_time
                print(f"✅ 처리 완료:")
                print(f"   📊 처리된 프레임: {result['frames_processed']}")
                print(f"   ⏱️  처리 시간: {elapsed_time:.2f}초")
                print(f"   🚀 평균 FPS: {fps:.1f}")
                print(f"   👥 검출된 인물: {result['total_people']}")
                print(f"   💾 출력 파일: {output_path}")
            else:
                print(f"❌ 처리 실패: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"❌ 에러 발생: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print(f"\n🎉 모든 테스트 완료!")
    print("=" * 80)
    
    # 리소스 정리
    processor.cleanup()

if __name__ == "__main__":
    main()
