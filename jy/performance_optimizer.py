#!/usr/bin/env python3
"""
성능 테스트 및 최적화 스크립트
A6000 x2 환경에서 최적의 설정을 찾아보는 도구
"""

import torch
import time
import psutil
import os
from pathlib import Path

def check_system_specs():
    """시스템 사양 확인"""
    print("🔍 시스템 사양 확인:")
    print("="*60)
    
    # CPU 정보
    cpu_count = os.cpu_count()
    memory = psutil.virtual_memory()
    print(f"🖥️  CPU 코어: {cpu_count}개")
    print(f"🧠 RAM: {memory.total / (1024**3):.1f} GB")
    
    # GPU 정보
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"🎮 GPU 개수: {gpu_count}개")
        
        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            memory_gb = props.total_memory / (1024**3)
            print(f"   GPU {i}: {props.name} ({memory_gb:.1f} GB VRAM)")
    else:
        print("❌ CUDA 사용 불가")
    
    print("="*60)

def recommend_settings():
    """추천 설정 계산"""
    cpu_count = os.cpu_count()
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    
    print("💡 추천 설정:")
    print("="*60)
    
    if gpu_count >= 2:
        # A6000 x2 환경
        gpu_workers = 2
        gpu_batch = 256  # A6000 48GB VRAM 활용
        cpu_workers = min(32, cpu_count // 2)
        video_loaders = 8
        
        print(f"🚀 멀티 GPU 고성능 모드:")
        print(f"   - GPU 워커: {gpu_workers}개")
        print(f"   - GPU 배치: {gpu_batch}")
        print(f"   - CPU 워커: {cpu_workers}개")
        print(f"   - 비디오 로더: {video_loaders}개")
        
    elif gpu_count == 1:
        # 단일 GPU 환경
        gpu_workers = 1
        gpu_batch = 128
        cpu_workers = min(16, cpu_count // 4)
        video_loaders = 4
        
        print(f"⚡ 단일 GPU 모드:")
        print(f"   - GPU 워커: {gpu_workers}개")
        print(f"   - GPU 배치: {gpu_batch}")
        print(f"   - CPU 워커: {cpu_workers}개")
        print(f"   - 비디오 로더: {video_loaders}개")
        
    else:
        # CPU 전용
        gpu_workers = 0
        gpu_batch = 0
        cpu_workers = min(8, cpu_count)
        video_loaders = 2
        
        print(f"💻 CPU 전용 모드:")
        print(f"   - CPU 워커: {cpu_workers}개")
        print(f"   - 비디오 로더: {video_loaders}개")
    
    print("="*60)
    return gpu_workers, gpu_batch, cpu_workers, video_loaders

def test_gpu_memory():
    """GPU 메모리 테스트"""
    if not torch.cuda.is_available():
        print("❌ CUDA 사용 불가 - GPU 메모리 테스트 건너뜀")
        return
    
    print("🧪 GPU 메모리 테스트:")
    print("="*60)
    
    for gpu_id in range(torch.cuda.device_count()):
        try:
            torch.cuda.set_device(gpu_id)
            
            # 메모리 정보
            props = torch.cuda.get_device_properties(gpu_id)
            total_memory = props.total_memory / (1024**3)
            
            # 사용 가능한 메모리 확인
            torch.cuda.empty_cache()
            allocated = torch.cuda.memory_allocated(gpu_id) / (1024**3)
            cached = torch.cuda.memory_reserved(gpu_id) / (1024**3)
            free = total_memory - allocated - cached
            
            print(f"GPU {gpu_id} ({props.name}):")
            print(f"   - 총 메모리: {total_memory:.1f} GB")
            print(f"   - 사용 중: {allocated:.1f} GB")
            print(f"   - 예약됨: {cached:.1f} GB") 
            print(f"   - 사용 가능: {free:.1f} GB")
            
            # 추천 배치 크기 계산
            if free > 40:  # 40GB 이상 여유
                recommended_batch = 512
            elif free > 20:  # 20GB 이상 여유
                recommended_batch = 256
            elif free > 10:  # 10GB 이상 여유
                recommended_batch = 128
            else:
                recommended_batch = 64
                
            print(f"   - 추천 배치: {recommended_batch}")
            print()
            
        except Exception as e:
            print(f"GPU {gpu_id} 테스트 실패: {e}")
    
    print("="*60)

def benchmark_inference():
    """간단한 추론 벤치마크"""
    if not torch.cuda.is_available():
        print("❌ CUDA 사용 불가 - 벤치마크 건너뜀")
        return
        
    print("⚡ 추론 성능 벤치마크:")
    print("="*60)
    
    try:
        from onnx_inferencer import YOLO11LRTMWONNXInferencer
        import numpy as np
        
        # 테스트용 더미 이미지
        test_image = np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)
        
        for gpu_id in range(min(2, torch.cuda.device_count())):
            try:
                print(f"GPU {gpu_id} 테스트 중...")
                
                # 인퍼런서 초기화
                inferencer = YOLO11LRTMWONNXInferencer(
                    rtmw_onnx_path="models/rtmw-dw-x-l_simcc-cocktail14_270e-384x288.onnx",
                    detection_device=f'cuda:{gpu_id}',
                    pose_device='cuda',
                    optimize_for_accuracy=True
                )
                
                # 워밍업
                for _ in range(3):
                    try:
                        inferencer.process_frame(test_image)
                    except:
                        pass
                
                # 벤치마크
                times = []
                for i in range(10):
                    start = time.time()
                    vis_img, results = inferencer.process_frame(test_image)
                    end = time.time()
                    times.append(end - start)
                
                avg_time = sum(times) / len(times)
                fps = 1.0 / avg_time
                
                print(f"   - 평균 시간: {avg_time:.3f}초")
                print(f"   - FPS: {fps:.1f}")
                
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"   - GPU {gpu_id} 벤치마크 실패: {e}")
        
    except ImportError:
        print("onnx_inferencer 모듈을 찾을 수 없어 벤치마크를 건너뜁니다.")
    except Exception as e:
        print(f"벤치마크 오류: {e}")
    
    print("="*60)

def generate_optimal_config():
    """최적 설정 파일 생성"""
    gpu_workers, gpu_batch, cpu_workers, video_loaders = recommend_settings()
    
    config = f"""
# A6000 x2 환경 최적화 설정
export CUDA_VISIBLE_DEVICES=0,1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export OMP_NUM_THREADS={cpu_workers}

# 추천 파라미터:
# - GPU 워커: {gpu_workers}
# - GPU 배치 크기: {gpu_batch}  
# - CPU 워커: {cpu_workers}
# - 비디오 로더: {video_loaders}

# 실행 예시:
python ultra_fast_processor.py \\
    --gpu_workers {gpu_workers} \\
    --gpu_batch {gpu_batch} \\
    --cpu_workers {cpu_workers} \\
    --video_loaders {video_loaders}
"""
    
    with open("optimal_config.sh", "w") as f:
        f.write(config.strip())
    
    print("📝 최적 설정을 optimal_config.sh에 저장했습니다.")

def main():
    print("🚀 A6000 x2 환경 최적화 도구")
    print("="*80)
    
    # 시스템 사양 확인
    check_system_specs()
    
    # 추천 설정 계산
    recommend_settings()
    
    # GPU 메모리 테스트
    test_gpu_memory()
    
    # 벤치마크 (선택사항)
    run_benchmark = input("\n추론 벤치마크를 실행하시겠습니까? (y/N): ").strip().lower()
    if run_benchmark == 'y':
        benchmark_inference()
    
    # 최적 설정 생성
    generate_optimal_config()
    
    print("\n🎉 최적화 분석 완료!")

if __name__ == "__main__":
    main()
