#!/usr/bin/env python3
"""
Ultra Fast Processor 기능 테스트
실제 비디오 파일들로 경로 탐지 및 처리 테스트
"""

import sys
import os
sys.path.append('/workspace01/team03/data/mmpose/jy')

from ultra_fast_processor_complete import UltraFastBatchProcessor
import logging

def test_video_collection():
    """비디오 수집 기능 테스트"""
    print("🔍 Ultra Fast Processor - 비디오 수집 테스트")
    print("=" * 60)
    
    # 프로세서 초기화 (테스트 모드)
    processor = UltraFastBatchProcessor(
        data_root="/workspace01/team03/data/mmpose/jy/data/",
        output_dir="/tmp/test_ultra_fast_output",
        direction="F",
        item_types=["WORD"],
        num_gpu_workers=1,      # 테스트용으로 줄임
        gpu_batch_size=32,      # 테스트용으로 줄임
        num_cpu_workers=2,      # 테스트용으로 줄임
        num_video_loaders=1     # 테스트용으로 줄임
    )
    
    # 비디오 수집 테스트
    print("\n🎬 비디오 파일 수집 중...")
    videos = processor.collect_all_videos()
    
    print(f"\n📊 결과:")
    print(f"   총 발견된 비디오: {len(videos)}개")
    
    if videos:
        print(f"\n📋 발견된 비디오 샘플 (처음 10개):")
        for i, video_path in enumerate(videos[:10]):
            from pathlib import Path
            video_file = Path(video_path)
            folder = video_file.parent.name
            filename = video_file.name
            print(f"   {i+1:2d}. 폴더 {folder}: {filename}")
        
        if len(videos) > 10:
            print(f"       ... 외 {len(videos)-10}개")
        
        # 아이템 정보 추출 테스트
        print(f"\n🔍 아이템 정보 추출 테스트:")
        from ultra_fast_processor_complete import extract_item_info_from_path
        
        test_samples = videos[:5]
        for video_path in test_samples:
            item_info = extract_item_info_from_path(video_path)
            filename = Path(video_path).name
            if item_info:
                item_type, item_id = item_info
                print(f"   ✅ {filename} → {item_type}{item_id:04d}")
            else:
                print(f"   ❌ {filename} → 추출 실패")
        
        return True
    else:
        print("❌ 비디오 파일을 찾을 수 없습니다!")
        return False

def test_small_processing():
    """소규모 처리 테스트 (1개 비디오)"""
    print("\n🚀 소규모 처리 테스트")
    print("=" * 60)
    
    try:
        from pathlib import Path
        
        # 첫 번째 비디오 파일 찾기
        video_files = list(Path("/workspace01/team03/data/mmpose/jy/data/1.Training/videos").glob("**/*_F.mp4"))
        if not video_files:
            print("❌ 처리할 비디오 파일을 찾을 수 없습니다")
            return False
        
        test_video = str(video_files[0])
        print(f"🎯 테스트 대상: {Path(test_video).name}")
        print(f"   경로: {test_video}")
        
        # 비디오 프레임 수 확인
        from ultra_fast_processor_complete import get_video_frame_indices
        frame_indices = get_video_frame_indices(test_video)
        print(f"   프레임 수: {len(frame_indices)}개")
        
        if len(frame_indices) > 0:
            # 프레임 로딩 테스트 (처음 5개만)
            from ultra_fast_processor_complete import load_video_frames
            test_frames = load_video_frames(test_video, frame_indices[:5])
            print(f"   로드된 프레임: {len(test_frames)}개")
            
            if test_frames:
                frame_shape = test_frames[0].shape
                print(f"   프레임 크기: {frame_shape}")
                print("✅ 비디오 로딩 성공!")
                return True
            else:
                print("❌ 프레임 로딩 실패")
                return False
        else:
            print("❌ 비디오 프레임을 읽을 수 없습니다")
            return False
            
    except Exception as e:
        print(f"❌ 처리 테스트 실패: {e}")
        return False

def main():
    """메인 테스트 실행"""
    print("🧪 Ultra Fast Processor 기능 검증 테스트")
    print("=" * 80)
    
    # 로깅 레벨 설정
    logging.basicConfig(level=logging.INFO)
    
    tests = [
        ("비디오 수집", test_video_collection),
        ("소규모 처리", test_small_processing),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            print(f"\n🔄 {test_name} 테스트 시작...")
            result = test_func()
            results.append((test_name, result))
            
            if result:
                print(f"✅ {test_name} 테스트 성공!")
            else:
                print(f"❌ {test_name} 테스트 실패!")
                
        except Exception as e:
            print(f"❌ {test_name} 테스트 예외: {e}")
            results.append((test_name, False))
    
    # 최종 결과
    print("\n" + "=" * 80)
    print("📋 최종 테스트 결과")
    print("=" * 80)
    
    passed = 0
    for test_name, result in results:
        status = "✅ 통과" if result else "❌ 실패"
        print(f"   {test_name}: {status}")
        if result:
            passed += 1
    
    success_rate = (passed / len(results)) * 100 if results else 0
    print(f"\n🎯 성공률: {success_rate:.1f}% ({passed}/{len(results)})")
    
    if success_rate == 100:
        print("\n🎉 모든 테스트 통과! Ultra Fast Processor가 정상적으로 작동합니다.")
        print("   이제 전체 처리를 실행할 수 있습니다!")
    elif success_rate >= 50:
        print("\n⚠️ 부분적으로 성공했습니다. 일부 기능은 정상 작동합니다.")
    else:
        print("\n❌ 대부분의 테스트가 실패했습니다. 코드 수정이 필요합니다.")
        
    return success_rate >= 75

if __name__ == "__main__":
    main()
