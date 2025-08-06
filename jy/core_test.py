#!/usr/bin/env python3
"""
🚀 Ultra Fast Processor 핵심 기능 테스트
A6000 x2 환경에서 실제 성능 검증
"""

import os
import time
import numpy as np
import torch
from onnx_inferencer import YOLO11LRTMWONNXInferencer
import cv2

def create_simple_test_video():
    """간단한 테스트 비디오 생성"""
    output_path = "/workspace01/team03/data/simple_test.mp4"
    
    # 720p 해상도로 작은 비디오 생성
    width, height = 1280, 720
    fps = 30
    duration = 2  # 2초
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    total_frames = int(fps * duration)
    
    for frame_idx in range(total_frames):
        # 단순한 배경
        frame = np.ones((height, width, 3), dtype=np.uint8) * 100
        
        # 사람 모양 그리기
        center_x, center_y = width//2, height//2
        
        # 움직임 추가
        offset_x = int(50 * np.sin(frame_idx * 0.2))
        offset_y = int(25 * np.cos(frame_idx * 0.1))
        
        x, y = center_x + offset_x, center_y + offset_y
        
        # 머리
        cv2.circle(frame, (x, y-60), 30, (200, 180, 160), -1)
        # 몸체
        cv2.rectangle(frame, (x-40, y-30), (x+40, y+80), (100, 150, 200), -1)
        # 팔
        cv2.line(frame, (x-40, y), (x-70, y+30), (120, 120, 120), 10)
        cv2.line(frame, (x+40, y), (x+70, y+30), (120, 120, 120), 10)
        # 다리
        cv2.line(frame, (x-20, y+80), (x-30, y+140), (80, 80, 80), 10)
        cv2.line(frame, (x+20, y+80), (x+30, y+140), (80, 80, 80), 10)
        
        writer.write(frame)
    
    writer.release()
    print(f"✅ 테스트 비디오 생성 완료: {output_path}")
    return output_path

def test_inference_performance():
    """추론 성능 직접 테스트"""
    print("🧪 추론 성능 직접 테스트")
    print("-" * 50)
    
    # 인퍼런서 초기화 (각 GPU별로)
    gpu_configs = [
        {"detection_device": "cuda:0", "pose_device": "cuda"},
        {"detection_device": "cuda:1", "pose_device": "cuda"}
    ]
    
    inferencers = []
    
    for i, config in enumerate(gpu_configs):
        print(f"🔧 GPU {i} 인퍼런서 초기화 중...")
        try:
            inferencer = YOLO11LRTMWONNXInferencer(
                rtmw_onnx_path="rtmw-dw-x-l_simcc-cocktail14_270e-384x288.onnx",
                detection_device=config["detection_device"],
                pose_device=config["pose_device"],
                optimize_for_accuracy=True
            )
            inferencers.append(inferencer)
            print(f"✅ GPU {i} 인퍼런서 준비 완료")
        except Exception as e:
            print(f"❌ GPU {i} 인퍼런서 초기화 실패: {e}")
    
    if not inferencers:
        print("❌ 사용 가능한 인퍼런서가 없습니다.")
        return
    
    # 테스트 비디오 생성
    test_video = create_simple_test_video()
    
    # 비디오 읽기
    cap = cv2.VideoCapture(test_video)
    frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    
    cap.release()
    
    print(f"📹 로딩된 프레임 수: {len(frames)}")
    
    # 각 인퍼런서로 성능 테스트
    for i, inferencer in enumerate(inferencers):
        print(f"\n🚀 GPU {i} 성능 테스트:")
        
        # 워밍업
        if frames:
            _ = inferencer.process_frame(frames[0])
        
        start_time = time.time()
        total_people = 0
        
        for frame_idx, frame in enumerate(frames):
            try:
                result = inferencer.process_frame(frame)
                # result가 tuple이면 결과 개수 확인
                if result is not None:
                    if isinstance(result, tuple) and len(result) >= 2:
                        # (vis_image, results) 형태
                        vis_image, pose_results = result
                        if pose_results is not None and len(pose_results) > 0:
                            total_people += len(pose_results)
                    elif isinstance(result, dict) and 'keypoints' in result:
                        total_people += len(result['keypoints'])
                    
                if frame_idx % 10 == 0:
                    print(f"   프레임 {frame_idx+1}/{len(frames)} 처리 중...")
                    
            except Exception as e:
                import traceback
                print(f"   ❌ 프레임 {frame_idx} 처리 실패: {e}")
                print(f"   스택트레이스: {traceback.format_exc()}")
                break  # 첫 번째 에러에서 중단
        
        elapsed_time = time.time() - start_time
        fps = len(frames) / elapsed_time
        
        print(f"   ✅ 완료:")
        print(f"      처리 시간: {elapsed_time:.2f}초")
        print(f"      평균 FPS: {fps:.1f}")
        print(f"      총 검출된 인물: {total_people}")
    
    # 정리
    os.remove(test_video)
    print(f"\n🧹 테스트 파일 정리 완료")

def test_multi_gpu_capability():
    """멀티 GPU 활용 능력 테스트"""
    print("\n🎮 멀티 GPU 활용 능력 테스트")
    print("=" * 50)
    
    if torch.cuda.device_count() < 2:
        print("❌ GPU가 2개 미만입니다.")
        return
    
    # 각 GPU 메모리 상태 확인
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        memory_allocated = torch.cuda.memory_allocated(i) / (1024**3)
        memory_total = torch.cuda.memory_reserved(i) / (1024**3)
        print(f"   메모리: {memory_allocated:.2f}GB / {memory_total:.2f}GB")
    
    # 간단한 GPU 작업 테스트
    print("\n⚡ GPU 병렬 작업 테스트:")
    
    # 각 GPU에서 간단한 연산 수행
    devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    
    for device in devices:
        try:
            # 큰 텐서 생성 및 연산
            with torch.cuda.device(device):
                x = torch.randn(1000, 1000, device=device)
                y = torch.randn(1000, 1000, device=device)
                
                start_time = time.time()
                z = torch.matmul(x, y)
                elapsed = time.time() - start_time
                
                print(f"   {device}: 연산 완료 ({elapsed:.3f}초)")
                
        except Exception as e:
            print(f"   {device}: 연산 실패 - {e}")

def main():
    print("🚀 Ultra Fast Processor 핵심 기능 테스트")
    print("=" * 80)
    
    # GPU 환경 확인
    print("🔍 GPU 환경 확인:")
    print(f"   CUDA 사용 가능: {torch.cuda.is_available()}")
    print(f"   GPU 개수: {torch.cuda.device_count()}")
    
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # 멀티 GPU 능력 테스트
    test_multi_gpu_capability()
    
    # 추론 성능 테스트
    test_inference_performance()
    
    print("\n🎉 모든 테스트 완료!")

if __name__ == "__main__":
    main()
