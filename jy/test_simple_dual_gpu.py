#!/usr/bin/env python3
import os
import sys
import time
import torch
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import multiprocessing as mp

# 경로 설정
sys.path.append('/workspace01/team03/data/mmpose/jy')
from batch_fast_multionnx_processor import BatchFastVideoProcessor

def process_on_gpu(gpu_id, video_path):
    """특정 GPU에서 비디오 처리"""
    try:
        print(f"🚀 GPU {gpu_id}에서 처리 시작: {Path(video_path).name}")
        
        # GPU 설정
        torch.cuda.set_device(gpu_id)
        device = f"cuda:{gpu_id}"
        
        # 프로세서 초기화 (올바른 매개변수 사용)
        processor = BatchFastVideoProcessor(
            batch_size=128,  # 작은 배치부터
            yolo_device=device,
            pose_device=device,
            gpu_warmup=True
        )
        
        # 처리 시작
        start_time = time.time()
        result = processor.process_video_batch_optimized(video_path)
        end_time = time.time()
        
        if result:
            print(f"✅ GPU {gpu_id} 완료: {end_time - start_time:.2f}초")
            print(f"   - 프레임: {result.get('total_frames', 0)}")
            fps = result.get('total_frames', 0) / max(end_time - start_time, 0.001)
            print(f"   - FPS: {fps:.1f}")
        else:
            print(f"❌ GPU {gpu_id} 처리 실패")
            
    except Exception as e:
        print(f"❌ GPU {gpu_id} 오류: {e}")
        import traceback
        traceback.print_exc()

def test_simple_dual_gpu():
    """간단한 듀얼 GPU 테스트"""
    print("🔥 간단한 듀얼 GPU 테스트")
    print("=" * 50)
    
    # CUDA multiprocessing 설정
    mp.set_start_method('spawn', force=True)
    
    # GPU 체크
    if not torch.cuda.is_available():
        print("❌ CUDA를 사용할 수 없습니다")
        return
    
    gpu_count = torch.cuda.device_count()
    print(f"✅ GPU 개수: {gpu_count}")
    
    for i in range(gpu_count):
        props = torch.cuda.get_device_properties(i)
        mem_gb = props.total_memory / 1024**3
        print(f"   - GPU {i}: {props.name} ({mem_gb:.1f}GB)")
    
    if gpu_count < 2:
        print("⚠️ 듀얼 GPU 테스트를 위해 2개 이상의 GPU가 필요합니다")
        return
    
    # 테스트 비디오 경로
    data_root = "/workspace01/team03/data/mmpose/jy/data/1.Training"
    test_videos = [
        f"{data_root}/NIA_SL_WORD0001_REAL11_F/NIA_SL_WORD0001_REAL11_F.mp4",
        f"{data_root}/NIA_SL_WORD0002_REAL11_F/NIA_SL_WORD0002_REAL11_F.mp4"
    ]
    
    # 각 GPU에서 하나씩 처리
    processes = []
    
    # GPU 0와 GPU 1에서 동시 처리
    for gpu_id, video_path in enumerate(test_videos[:2]):
        if gpu_id < gpu_count:
            p = mp.Process(target=process_on_gpu, args=(gpu_id, video_path))
            processes.append(p)
            p.start()
    
    # 모든 프로세스 완료 대기
    for p in processes:
        p.join()
    
    print("🎯 듀얼 GPU 테스트 완료!")

if __name__ == "__main__":
    test_simple_dual_gpu()
