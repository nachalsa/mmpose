#!/usr/bin/env python3
"""
🚀 A6000 x2 성능 벤치마크 테스트
Ultra Fast Processor vs 기존 Processor 비교
"""

import os
import time
import numpy as np
from ultra_fast_processor import UltraFastVideoProcessor
from multionnx_streamlined_processor import StreamlinedVideoProcessor
import cv2

def create_test_video(width=1920, height=1080, fps=30, duration=3):
    """테스트용 비디오 생성 (사람이 있는 장면 시뮬레이션)"""
    output_path = "/workspace01/team03/data/test_video.mp4"
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    total_frames = int(fps * duration)
    
    for frame_idx in range(total_frames):
        # 배경 생성 (그라데이션)
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :, 0] = 50  # 파란색 배경
        frame[:, :, 1] = 100
        frame[:, :, 2] = 150
        
        # 사람 모양의 간단한 도형들 그리기 (여러 사람)
        people_positions = [
            (width//4, height//2),
            (width//2, height//2),
            (3*width//4, height//2)
        ]
        
        for person_x, person_y in people_positions:
            # 움직임 시뮬레이션
            offset_x = int(20 * np.sin(frame_idx * 0.1))
            offset_y = int(10 * np.cos(frame_idx * 0.15))
            
            x, y = person_x + offset_x, person_y + offset_y
            
            # 머리 (원)
            cv2.circle(frame, (x, y-80), 25, (200, 180, 160), -1)
            
            # 몸체 (사각형)
            cv2.rectangle(frame, (x-30, y-50), (x+30, y+50), (100, 150, 200), -1)
            
            # 팔 (선)
            cv2.line(frame, (x-30, y-20), (x-50, y+10), (120, 120, 120), 8)
            cv2.line(frame, (x+30, y-20), (x+50, y+10), (120, 120, 120), 8)
            
            # 다리 (선)
            cv2.line(frame, (x-15, y+50), (x-25, y+100), (80, 80, 80), 8)
            cv2.line(frame, (x+15, y+50), (x+25, y+100), (80, 80, 80), 8)
        
        writer.write(frame)
    
    writer.release()
    print(f"✅ 테스트 비디오 생성: {output_path}")
    return output_path

def benchmark_processor(processor_class, name, video_path, **kwargs):
    """프로세서 성능 벤치마크"""
    print(f"\n{'='*60}")
    print(f"🧪 {name} 테스트 시작")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        processor = processor_class(**kwargs)
        
        output_path = f"/workspace01/team03/data/benchmark_{name.lower().replace(' ', '_')}.h5"
        
        result = processor.process_video(
            video_path=video_path,
            output_path=output_path,
            max_frames=100,  # 빠른 테스트를 위해 제한
            skip_existing=False
        )
        
        elapsed_time = time.time() - start_time
        
        if result["success"]:
            fps = result["frames_processed"] / elapsed_time
            print(f"✅ {name} 처리 완료:")
            print(f"   📊 처리된 프레임: {result['frames_processed']}")
            print(f"   ⏱️  총 처리 시간: {elapsed_time:.2f}초")
            print(f"   🚀 평균 FPS: {fps:.1f}")
            print(f"   👥 검출된 인물: {result['total_people']}")
            
            return {
                "success": True,
                "fps": fps,
                "elapsed_time": elapsed_time,
                "frames": result["frames_processed"],
                "people": result["total_people"]
            }
        else:
            print(f"❌ {name} 처리 실패: {result.get('error', 'Unknown error')}")
            return {"success": False, "error": result.get('error', 'Unknown error')}
            
    except Exception as e:
        print(f"❌ {name} 에러 발생: {str(e)}")
        return {"success": False, "error": str(e)}
    finally:
        if 'processor' in locals():
            processor.cleanup()

def main():
    print("🚀 A6000 x2 성능 벤치마크 시작")
    print("="*80)
    
    # 테스트 비디오 생성
    print("📹 테스트 비디오 생성 중...")
    test_video = create_test_video()
    
    # 테스트 설정들
    test_configs = [
        {
            "class": UltraFastVideoProcessor,
            "name": "Ultra Fast (GPU 512배치)",
            "kwargs": {
                "gpu_batch_size": 512,
                "num_gpu_workers": 2,
                "num_cpu_workers": 32,
                "num_video_loaders": 8
            }
        },
        {
            "class": UltraFastVideoProcessor,
            "name": "Ultra Fast (GPU 256배치)",
            "kwargs": {
                "gpu_batch_size": 256,
                "num_gpu_workers": 2,
                "num_cpu_workers": 16,
                "num_video_loaders": 4
            }
        },
        {
            "class": StreamlinedVideoProcessor,
            "name": "기존 스트림라인 프로세서",
            "kwargs": {
                "yolo_device": "cuda:0",
                "pose_device": "cuda"
            }
        }
    ]
    
    results = {}
    
    # 각 설정으로 벤치마크 실행
    for config in test_configs:
        result = benchmark_processor(
            processor_class=config["class"],
            name=config["name"],
            video_path=test_video,
            **config["kwargs"]
        )
        results[config["name"]] = result
    
    # 결과 비교
    print(f"\n{'='*80}")
    print("📊 성능 비교 결과")
    print(f"{'='*80}")
    
    successful_results = {name: result for name, result in results.items() if result["success"]}
    
    if successful_results:
        print(f"{'프로세서 이름':<30} {'FPS':<10} {'시간(초)':<10} {'프레임':<8} {'인물수':<8}")
        print("-" * 70)
        
        for name, result in successful_results.items():
            print(f"{name:<30} {result['fps']:<10.1f} {result['elapsed_time']:<10.2f} {result['frames']:<8} {result['people']:<8}")
        
        # 최고 성능 찾기
        best_fps = max(result['fps'] for result in successful_results.values())
        best_name = next(name for name, result in successful_results.items() if result['fps'] == best_fps)
        
        print(f"\n🏆 최고 성능: {best_name} ({best_fps:.1f} FPS)")
        
        # 성능 개선 비교
        if len(successful_results) > 1:
            baseline_fps = min(result['fps'] for result in successful_results.values())
            improvement = (best_fps / baseline_fps - 1) * 100
            print(f"🚀 성능 개선: {improvement:.1f}% 향상")
    
    else:
        print("❌ 모든 테스트가 실패했습니다.")
    
    # 테스트 파일 정리
    if os.path.exists(test_video):
        os.remove(test_video)
        print(f"\n🧹 테스트 파일 정리 완료")
    
    print(f"\n🎉 벤치마크 완료!")

if __name__ == "__main__":
    main()
