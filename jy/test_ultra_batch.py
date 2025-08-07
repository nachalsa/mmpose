#!/usr/bin/env python3
"""
Ultra Batch 256 테스트 - A6000 x2 GPU 완전 활용
Phase 1: 배치 처리 아키텍처 구축 ⚡
Phase 2: 성능 모니터링 시스템 📊  
Phase 3: 고속 파이프라인 구현 🚀
"""

import os
import sys
import time
from pathlib import Path

# 현재 디렉토리를 Python 경로에 추가
sys.path.append(str(Path(__file__).parent))

from batch_fast_multionnx_processor import (
    BatchFastVideoProcessor,
    process_videos_dual_gpu_batch
)

def test_dual_gpu_batch():
    """듀얼 GPU 배치 256 테스트"""
    
    print("🚀 Ultra Batch 256 테스트 시작 - A6000 x2 GPU 완전 활용")
    print("=" * 60)
    
    # 테스트 비디오 경로들
    video_dir = Path("/workspace01/team03/data/mmpose/jy/data/1.Training/videos/05")
    video_paths = list(video_dir.glob("*.mp4"))[:10]  # 테스트용 10개
    
    if not video_paths:
        print("❌ 테스트 비디오를 찾을 수 없습니다")
        return False
    
    video_paths = [str(p) for p in video_paths]
    
    print(f"📹 테스트 비디오: {len(video_paths)}개")
    for i, path in enumerate(video_paths):
        print(f"   {i+1:2d}. {Path(path).name}")
    
    # 출력 디렉토리
    output_dir = "ultra_batch_test_output"
    
    # 설정
    config = {
        'rtmw_model_name': 'rtmw-dw-x-l_simcc-cocktail14_270e-384x288.onnx',
        'batch_size': 256,
        'keypoint_scale': 8,
        'jpeg_quality': 90,
        'gpu_warmup': True
    }
    
    print(f"\n⚙️ 설정:")
    for key, value in config.items():
        print(f"   - {key}: {value}")
    
    # 진행률 콜백
    def progress_callback(progress):
        progress_pct = progress * 100
        print(f"📊 전체 진행률: {progress_pct:.1f}%")
    
    print(f"\n🔥 Phase 1: 배치 처리 아키텍처 구축 ⚡")
    print(f"   - 256 프레임씩 VRAM 미리 로드")
    print(f"   - A6000 x2 완전 활용")
    print(f"   - 스마트 버퍼링 시스템")
    
    print(f"\n📊 Phase 2: 성능 모니터링 시스템 📊")
    print(f"   - 실시간 진행률 추적")
    print(f"   - GPU 사용률 모니터링")
    print(f"   - 처리 속도 분석")
    
    print(f"\n🚀 Phase 3: 고속 파이프라인 구현 🚀")
    print(f"   - 비동기 데이터 로딩")
    print(f"   - 스트리밍 처리")
    print(f"   - 결과 버퍼링")
    
    print(f"\n🚀 듀얼 GPU 배치 처리 시작...")
    start_time = time.time()
    
    try:
        # 듀얼 GPU 배치 처리 실행
        results = process_videos_dual_gpu_batch(
            video_paths=video_paths,
            output_dir=output_dir,
            config=config,
            progress_callback=progress_callback
        )
        
        total_time = time.time() - start_time
        
        print(f"\n🎉 Ultra Batch 256 테스트 완료!")
        print(f"=" * 60)
        print(f"📊 최종 결과:")
        print(f"   - 총 비디오: {results['total_videos']}개")
        print(f"   - 성공: {results['completed']}개")
        print(f"   - 실패: {results['failed']}개")
        print(f"   - 총 처리 시간: {total_time:.2f}초")
        print(f"   - 평균 시간/비디오: {results['average_time_per_video']:.2f}초")
        
        if results['completed'] > 0:
            success_rate = (results['completed'] / results['total_videos']) * 100
            print(f"   - 성공률: {success_rate:.1f}%")
        
        # 개별 결과 분석
        print(f"\n📈 개별 처리 결과:")
        for task_id, result_item in results['results'].items():
            if result_item['status'] == 'completed':
                fps = result_item.get('fps_achieved', 0)
                frames = result_item.get('frame_count', 0)
                time_taken = result_item.get('processing_time', 0)
                video_name = Path(result_item['video_path']).name
                gpu_id = result_item.get('gpu_id', -1)
                
                print(f"   ✅ GPU {gpu_id}: {video_name}")
                print(f"      📹 {frames}프레임, 🚀 {fps:.1f}FPS, ⏱️ {time_taken:.2f}초")
        
        return True
        
    except Exception as e:
        print(f"💥 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_single_gpu_batch():
    """단일 GPU 테스트"""
    
    print("🔥 단일 GPU 배치 테스트")
    
    # 테스트 비디오 1개
    video_dir = Path("/workspace01/team03/data/mmpose/jy/data/1.Training/videos/05")
    video_paths = list(video_dir.glob("*.mp4"))[:1]
    
    if not video_paths:
        print("❌ 테스트 비디오를 찾을 수 없습니다")
        return False
    
    video_path = str(video_paths[0])
    
    print(f"📹 테스트 비디오: {Path(video_path).name}")
    
    try:
        # 단일 GPU 처리기 초기화
        processor = BatchFastVideoProcessor(
            rtmw_model_name='rtmw-dw-x-l_simcc-cocktail14_270e-384x288.onnx',
            gpu_id=0,
            batch_size=256,
            keypoint_scale=8
        )
        
        # 진행률 콜백
        def progress_callback(progress):
            progress_pct = progress * 100
            print(f"📊 처리 진행률: {progress_pct:.1f}%")
        
        print(f"🔄 비디오 처리 시작...")
        start_time = time.time()
        
        # 비디오 처리
        result = processor.process_video_batch_optimized(video_path, progress_callback)
        
        processing_time = time.time() - start_time
        
        if result:
            print(f"✅ 처리 완료:")
            print(f"   - 총 프레임: {result['total_frames']}")
            print(f"   - 처리 시간: {processing_time:.2f}초")
            print(f"   - 달성 FPS: {result['fps']:.1f}")
            print(f"   - JPEG 프레임: {len(result['jpeg_frames'])}개")
            print(f"   - 키포인트: {len(result['keypoints'])}개")
            
            return True
        else:
            print(f"❌ 처리 실패")
            return False
            
    except Exception as e:
        print(f"💥 단일 GPU 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Ultra Batch Processor 테스트 시작")
    print("Phase 1: 배치 처리 아키텍처 구축 ⚡")
    print("Phase 2: 성능 모니터링 시스템 📊")  
    print("Phase 3: 고속 파이프라인 구현 🚀")
    print("=" * 60)
    
    # 1. 단일 GPU 테스트
    print("\n🔥 1단계: 단일 GPU 테스트")
    single_success = test_single_gpu_batch()
    
    if single_success:
        print("\n✅ 단일 GPU 테스트 성공! 듀얼 GPU 테스트 진행...")
        
        # 2. 듀얼 GPU 테스트
        print("\n🚀 2단계: 듀얼 GPU 테스트")
        dual_success = test_dual_gpu_batch()
        
        if dual_success:
            print("\n🎉 모든 테스트 성공! Ultra Batch 256 시스템 준비 완료!")
        else:
            print("\n⚠️ 듀얼 GPU 테스트 실패")
    else:
        print("\n❌ 단일 GPU 테스트 실패")
