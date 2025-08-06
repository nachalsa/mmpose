#!/usr/bin/env python3
"""
Ultra Fast Processor 종합 테스트 (최신 버전)
사용자 요청: "서버 성능이 A6000 두장에 cpu코어 112개 있을정도로 좋은 환경"
목표: 최고 성능의 멀티 GPU 활용 테스트
"""

import cv2
import time
import torch
import psutil
import os
from ultra_fast_processor import UltraFastVideoProcessor
from onnx_inferencer import ONNXInferencer
import numpy as np

def create_benchmark_video(width=1920, height=1080, fps=30, duration=10, 
                          num_people=4, output_path="benchmark_video.mp4"):
    """성능 테스트용 고해상도 비디오 생성"""
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    total_frames = int(fps * duration)
    
    # 다중 인물 시뮬레이션을 위한 설정
    people_positions = []
    for i in range(num_people):
        x = int(width * (0.15 + 0.2 * i))
        y = int(height * 0.3)
        size = int(min(width, height) * 0.15)
        people_positions.append((x, y, size))
    
    print(f"🎬 생성 중: {width}x{height} @ {fps}fps, {duration}초, {num_people}명")
    
    for frame_num in range(total_frames):
        # 흰색 배경
        frame = np.ones((height, width, 3), dtype=np.uint8) * 240
        
        # 다중 인물 그리기
        for idx, (x, y, size) in enumerate(people_positions):
            # 움직임 시뮬레이션
            move_x = int(50 * np.sin(frame_num * 0.1 + idx * 2))
            move_y = int(20 * np.cos(frame_num * 0.05 + idx * 1.5))
            
            current_x = x + move_x
            current_y = y + move_y
            
            # 인물 모양 (타원)
            cv2.ellipse(frame, (current_x, current_y), 
                       (size//3, size//2), 0, 0, 360, 
                       (80 + idx * 40, 120, 160 - idx * 20), -1)
            
            # 머리 (원)
            cv2.circle(frame, (current_x, current_y - size//2), 
                      size//4, (200, 150 + idx * 20, 100), -1)
            
            # 팔다리 (선)
            arm_len = size // 3
            leg_len = size // 2
            
            # 팔
            cv2.line(frame, (current_x - size//3, current_y - size//4),
                    (current_x - size//2 - arm_len//2, current_y), 
                    (60, 60, 60), 8)
            cv2.line(frame, (current_x + size//3, current_y - size//4),
                    (current_x + size//2 + arm_len//2, current_y), 
                    (60, 60, 60), 8)
            
            # 다리
            cv2.line(frame, (current_x - size//6, current_y + size//2),
                    (current_x - size//4, current_y + size//2 + leg_len), 
                    (60, 60, 60), 8)
            cv2.line(frame, (current_x + size//6, current_y + size//2),
                    (current_x + size//4, current_y + size//2 + leg_len), 
                    (60, 60, 60), 8)
        
        # 프레임 번호
        cv2.putText(frame, f"Frame: {frame_num+1}/{total_frames}", 
                   (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
        
        out.write(frame)
        
        if frame_num % 60 == 0:
            print(f"   진행률: {frame_num/total_frames*100:.1f}%")
    
    out.release()
    print(f"✅ 벤치마크 비디오 생성 완료: {output_path}")
    return output_path

def benchmark_single_gpu():
    """단일 GPU 성능 테스트"""
    print("\n🔥 단일 GPU 베이스라인 성능 측정")
    print("=" * 60)
    
    inferencer = ONNXInferencer(device='cuda:0', accuracy_mode=True)
    
    # 고해상도 테스트 비디오 생성
    test_video = create_benchmark_video(
        width=1920, height=1080, fps=30, duration=5, 
        num_people=6, output_path="single_gpu_test.mp4"
    )
    
    cap = cv2.VideoCapture(test_video)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    
    print(f"📹 테스트 프레임 수: {len(frames)}")
    
    # 성능 측정
    start_time = time.time()
    total_detections = 0
    
    for i, frame in enumerate(frames):
        result = inferencer.process_frame(frame)
        if isinstance(result, tuple):
            _, pose_results = result
            if pose_results:
                total_detections += len(pose_results)
        else:
            if result:
                total_detections += len(result)
        
        if (i + 1) % 30 == 0:
            elapsed = time.time() - start_time
            current_fps = (i + 1) / elapsed
            print(f"   프레임 {i+1}/{len(frames)}: {current_fps:.1f} FPS")
    
    total_time = time.time() - start_time
    avg_fps = len(frames) / total_time
    
    print(f"\n📊 단일 GPU (CUDA:0) 결과:")
    print(f"   처리 시간: {total_time:.2f}초")
    print(f"   평균 FPS: {avg_fps:.1f}")
    print(f"   총 검출 인물: {total_detections}")
    
    # 정리
    os.remove(test_video)
    del inferencer
    torch.cuda.empty_cache()
    
    return avg_fps, total_detections

def benchmark_multi_gpu():
    """멀티 GPU 성능 테스트"""
    print("\n🚀 Ultra Fast Processor 멀티 GPU 성능")
    print("=" * 60)
    
    processor = UltraFastVideoProcessor(
        gpu_batch_size=512,     # A6000 x2 최대 성능
        cpu_workers=32,         # 112코어 활용
        detection_confidence=0.5,
        pose_confidence=0.3,
        accuracy_mode=True,
        enable_tensorrt=True    # 최고 성능 모드
    )
    
    # 고해상도 다중 인물 테스트
    test_video = create_benchmark_video(
        width=1920, height=1080, fps=30, duration=5,
        num_people=8, output_path="multi_gpu_test.mp4"
    )
    
    # 성능 측정
    start_time = time.time()
    
    results = processor.process_video(
        input_path=test_video,
        output_path="multi_gpu_output.mp4",
        enable_pose_tracking=True,
        enable_heatmap=True
    )
    
    total_time = time.time() - start_time
    
    print(f"\n🔥 Ultra Fast Processor 결과:")
    print(f"   처리 시간: {total_time:.2f}초")
    print(f"   평균 FPS: {results['average_fps']:.1f}")
    print(f"   총 처리 프레임: {results['total_frames']}")
    print(f"   총 검출 인물: {results['total_detections']}")
    print(f"   GPU 사용률: GPU0={results['gpu_utilization'][0]:.1f}%, GPU1={results['gpu_utilization'][1]:.1f}%")
    
    # 성능 향상 비율
    if 'baseline_fps' in globals():
        improvement = (results['average_fps'] / baseline_fps - 1) * 100
        print(f"   성능 향상: {improvement:+.1f}%")
    
    # 정리
    os.remove(test_video)
    if os.path.exists("multi_gpu_output.mp4"):
        os.remove("multi_gpu_output.mp4")
    
    return results['average_fps'], results['total_detections']

def system_analysis():
    """시스템 자원 분석"""
    print("\n🔍 시스템 자원 현황")
    print("=" * 50)
    
    # CPU 정보
    cpu_count = psutil.cpu_count()
    cpu_freq = psutil.cpu_freq()
    memory = psutil.virtual_memory()
    
    print(f"💻 CPU:")
    print(f"   코어 수: {cpu_count}")
    print(f"   클럭: {cpu_freq.current/1000:.1f} GHz")
    print(f"   메모리: {memory.total/1024**3:.1f} GB")
    
    # GPU 정보
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"\n🎮 GPU ({gpu_count}개):")
        
        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            memory_total = props.total_memory / 1024**3
            memory_reserved = torch.cuda.memory_reserved(i) / 1024**3
            memory_allocated = torch.cuda.memory_allocated(i) / 1024**3
            
            print(f"   GPU {i}: {props.name}")
            print(f"      VRAM: {memory_allocated:.1f}/{memory_total:.1f} GB")
            print(f"      컴퓨트 유닛: {props.multi_processor_count}")

def main():
    """메인 테스트 실행"""
    global baseline_fps
    
    print("🚀 Ultra Fast Video Processor 종합 성능 테스트")
    print("=" * 80)
    print("🎯 목표: A6000 x2 + 112 CPU 코어 최대 활용")
    print("=" * 80)
    
    # 시스템 분석
    system_analysis()
    
    try:
        # 1. 단일 GPU 베이스라인
        baseline_fps, baseline_detections = benchmark_single_gpu()
        
        # 2. 멀티 GPU 성능
        multi_fps, multi_detections = benchmark_multi_gpu()
        
        # 3. 최종 결과 비교
        print("\n" + "=" * 80)
        print("📊 최종 성능 비교")
        print("=" * 80)
        
        performance_improvement = (multi_fps / baseline_fps - 1) * 100
        
        print(f"🔹 단일 GPU (CUDA:0):")
        print(f"   평균 FPS: {baseline_fps:.1f}")
        print(f"   총 검출: {baseline_detections}")
        
        print(f"\n🔥 멀티 GPU (Ultra Fast):")
        print(f"   평균 FPS: {multi_fps:.1f}")
        print(f"   총 검출: {multi_detections}")
        
        print(f"\n🚀 성능 향상:")
        print(f"   FPS 증가: {performance_improvement:+.1f}%")
        print(f"   처리량: {multi_fps/baseline_fps:.1f}x")
        
        if performance_improvement > 100:
            print(f"   ✅ 목표 달성! A6000 x2 최적화 성공!")
        else:
            print(f"   ⚡ 추가 최적화 여지 있음")
            
    except Exception as e:
        print(f"❌ 테스트 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n🎉 종합 테스트 완료!")

if __name__ == "__main__":
    main()
