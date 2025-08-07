#!/usr/bin/env python3
"""
개선된 배치 GPU 처리 테스트 - Production Ready 버전
- 진행률 추적 개선
- 실제 MP4 파일 처리
- 듀얼 GPU 최적화
- 통계 계산 오류 수정
"""
import os
import sys
import time
import h5py
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import multiprocessing as mp

# 경로 설정
sys.path.append('/workspace01/team03/data/mmpose/jy')
from batch_fast_multionnx_processor import BatchFastVideoProcessor

def process_video_on_gpu(gpu_id, video_files, result_queue, progress_queue):
    """특정 GPU에서 MP4 배치 처리"""
    try:
        print(f"🚀 GPU {gpu_id} 워커 시작: {len(video_files)}개 비디오 파일")
        
        # GPU 설정
        torch.cuda.set_device(gpu_id)
        device = f"cuda:{gpu_id}"
        
        # 프로세서 초기화
        processor = BatchFastVideoProcessor(
            batch_size=256,
            yolo_device=device,
            pose_device=device,
            gpu_warmup=True
        )
        
        # 각 비디오 파일 처리
        for i, video_path in enumerate(video_files):
            try:
                print(f"� GPU {gpu_id}: {Path(video_path).name} 처리 중... ({i+1}/{len(video_files)})")
                
                # 비디오 처리 시작
                start_time = time.time()
                result = processor.process_video_batch_optimized(str(video_path), 
                    progress_callback=lambda p: progress_queue.put({
                        'gpu_id': gpu_id,
                        'file': Path(video_path).name,
                        'progress': p,
                        'batch': f"처리중"
                    })
                )
                end_time = time.time()
                
                if result:
                    fps = result.get('total_frames', 0) / max(end_time - start_time, 0.001)
                    
                    result_data = {
                        'gpu_id': gpu_id,
                        'file': Path(video_path).name,
                        'frames': result.get('total_frames', 0),
                        'time': end_time - start_time,
                        'fps': fps,
                        'vram': torch.cuda.memory_allocated(gpu_id) / 1024**3
                    }
                    
                    result_queue.put(result_data)
                    print(f"✅ GPU {gpu_id}: {Path(video_path).name} 완료 ({fps:.1f} FPS)")
                    
                    # 완료 진행률 업데이트
                    progress_queue.put({
                        'gpu_id': gpu_id,
                        'file': Path(video_path).name,
                        'progress': 1.0,
                        'batch': f"완료"
                    })
                else:
                    print(f"❌ GPU {gpu_id} 처리 실패: {Path(video_path).name}")
                
            except Exception as e:
                print(f"❌ GPU {gpu_id} 파일 오류 {Path(video_path).name}: {e}")
                import traceback
                traceback.print_exc()
                
    except Exception as e:
        print(f"❌ GPU {gpu_id} 워커 오류: {e}")
        import traceback
        traceback.print_exc()

def monitor_progress(progress_queue, total_files):
    """진행률 모니터링"""
    completed = {}
    pbar = tqdm(total=total_files, desc="🚀 Dual GPU Video Processing")
    
    while len(completed) < total_files:
        try:
            progress_data = progress_queue.get(timeout=1)
            gpu_id = progress_data['gpu_id']
            file_name = progress_data['file']
            progress = progress_data['progress']
            batch_info = progress_data['batch']
            
            # 진행률 표시 업데이트
            pbar.set_description(f"🚀 GPU{gpu_id}: {file_name} {batch_info} ({progress*100:.1f}%)")
            
            # 파일 완료 체크
            if progress >= 1.0 and file_name not in completed:
                completed[file_name] = True
                pbar.update(1)
                
        except:
            continue
    
    pbar.close()

def test_improved_batch_gpu():
    """개선된 배치 GPU 테스트 - 실제 MP4 파일 처리"""
    print("🔥 개선된 배치 GPU 테스트 (실제 MP4)")
    print("=" * 60)
    
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
    
    # 실제 비디오 파일 찾기
    video_files = [
        "/workspace01/team03/data/mmpose/jy/data/1.Training/videos/05/NIA_SL_WORD0001_REAL05_F.mp4",
        "/workspace01/team03/data/mmpose/jy/data/1.Training/videos/05/NIA_SL_WORD0002_REAL05_F.mp4",
        "/workspace01/team03/data/mmpose/jy/data/1.Training/videos/05/NIA_SL_WORD0003_REAL05_F.mp4",
        "/workspace01/team03/data/mmpose/jy/data/1.Training/videos/05/NIA_SL_WORD0004_REAL05_F.mp4"
    ]
    
    # 실제로 존재하는 파일만 필터링
    existing_videos = []
    for video_path in video_files:
        if Path(video_path).exists():
            existing_videos.append(video_path)
        else:
            print(f"⚠️ 파일 없음: {Path(video_path).name}")
    
    if not existing_videos:
        print("❌ 처리할 비디오 파일을 찾을 수 없습니다")
        return
    
    print(f"� 테스트 비디오 파일: {len(existing_videos)}개")
    for f in existing_videos:
        print(f"   - {Path(f).name}")
    
    # 파일을 GPU에 분배
    files_per_gpu = len(existing_videos) // min(gpu_count, 2)
    gpu_0_files = existing_videos[:files_per_gpu] if files_per_gpu > 0 else existing_videos[:len(existing_videos)//2]
    gpu_1_files = existing_videos[files_per_gpu:] if files_per_gpu > 0 else existing_videos[len(existing_videos)//2:]
    
    print(f"📊 GPU 분배:")
    print(f"   - GPU 0: {len(gpu_0_files)}개 비디오")  
    print(f"   - GPU 1: {len(gpu_1_files)}개 비디오")
    
    # 큐 생성
    result_queue = mp.Queue()
    progress_queue = mp.Queue()
    
    # GPU 프로세스 시작
    processes = []
    
    if len(gpu_0_files) > 0:
        p0 = mp.Process(target=process_video_on_gpu, args=(0, gpu_0_files, result_queue, progress_queue))
        processes.append(p0)
        p0.start()
    
    if len(gpu_1_files) > 0 and gpu_count > 1:
        p1 = mp.Process(target=process_video_on_gpu, args=(1, gpu_1_files, result_queue, progress_queue))
        processes.append(p1)
        p1.start()
    
    # 진행률 모니터링 프로세스
    monitor_process = mp.Process(target=monitor_progress, args=(progress_queue, len(existing_videos)))
    monitor_process.start()
    
    # 결과 수집
    results = []
    total_start = time.time()
    
    # 모든 프로세스 완료 대기
    for p in processes:
        p.join()
    
    # 진행률 모니터 종료
    monitor_process.terminate()
    monitor_process.join()
    
    # 결과 수집
    while not result_queue.empty():
        results.append(result_queue.get())
    
    total_time = time.time() - total_start
    
    # 결과 출력
    print("\n" + "="*60)
    print("📊 처리 결과:")
    
    total_frames = 0
    total_fps = 0
    gpu_stats = {}
    
    for result in results:
        gpu_id = result['gpu_id']
        if gpu_id not in gpu_stats:
            gpu_stats[gpu_id] = {'files': 0, 'frames': 0, 'time': 0, 'vram': 0}
        
        gpu_stats[gpu_id]['files'] += 1
        gpu_stats[gpu_id]['frames'] += result['frames'] 
        gpu_stats[gpu_id]['time'] += result['time']
        gpu_stats[gpu_id]['vram'] = max(gpu_stats[gpu_id]['vram'], result['vram'])
        
        total_frames += result['frames']
        total_fps += result['fps']
        
        print(f"✅ GPU {result['gpu_id']}: {result['file']}")
        print(f"   - 프레임: {result['frames']}, 시간: {result['time']:.2f}s")
        print(f"   - FPS: {result['fps']:.1f}, VRAM: {result['vram']:.2f}GB")
    
    print(f"\n📈 전체 통계:")
    print(f"   - 총 처리 시간: {total_time:.2f}초")
    print(f"   - 총 프레임: {total_frames}")
    if total_time > 0:
        print(f"   - 평균 FPS: {total_frames/total_time:.1f}")
    else:
        print(f"   - 평균 FPS: 0.0")
    
    if total_frames > 0 and len(results) > 0:
        efficiency = (total_fps * total_time / total_frames)
        print(f"   - 병렬 효율성: {efficiency:.1f}x")
    else:
        print(f"   - 병렬 효율성: 0.0x")
    
    for gpu_id, stats in gpu_stats.items():
        if stats['files'] > 0 and stats['time'] > 0:
            avg_fps = stats['frames'] / stats['time']
            print(f"🔥 GPU {gpu_id}: {stats['files']}파일, {stats['frames']}프레임")
            print(f"   - 평균 FPS: {avg_fps:.1f}")
            print(f"   - 최대 VRAM: {stats['vram']:.2f}GB")
    
    print("🎯 개선된 배치 GPU 테스트 완료!")

if __name__ == "__main__":
    test_improved_batch_gpu()
