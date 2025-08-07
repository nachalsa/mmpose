#!/usr/bin/env python3
import torch
import time
import multiprocessing as mp

def test_gpu_separation():
    """GPU 분리 테스트 - 각 GPU에서 독립적으로 텐서 연산"""
    print("🔥 GPU 분리 테스트")
    print("=" * 50)
    
    # CUDA multiprocessing 설정
    mp.set_start_method('spawn', force=True)
    
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

def gpu_compute_test(gpu_id):
    """GPU에서 계산 테스트"""
    try:
        print(f"🚀 GPU {gpu_id} 계산 시작")
        
        # GPU 설정
        torch.cuda.set_device(gpu_id)
        device = f"cuda:{gpu_id}"
        
        # 큰 텐서 생성 및 연산 (배치 256 시뮬레이션)
        batch_size = 256
        tensor_a = torch.randn(batch_size, 3, 384, 288).to(device)
        tensor_b = torch.randn(batch_size, 3, 384, 288).to(device)
        
        print(f"✅ GPU {gpu_id}: 텐서 생성 완료 ({batch_size}배치)")
        print(f"   - VRAM 사용: {torch.cuda.memory_allocated(gpu_id)/1024**3:.2f}GB")
        
        # 연산 수행 (포즈 추정과 유사한 계산량)
        start_time = time.time()
        for i in range(10):  # 10번 반복
            result = torch.nn.functional.conv2d(
                tensor_a, 
                torch.randn(64, 3, 3, 3).to(device),
                padding=1
            )
            result = torch.nn.functional.relu(result)
            result = torch.nn.functional.max_pool2d(result, 2)
            
            # GPU 동기화
            torch.cuda.synchronize(gpu_id)
            
        end_time = time.time()
        
        print(f"✅ GPU {gpu_id} 완료: {end_time - start_time:.2f}초")
        print(f"   - 처리량: {batch_size * 10 / (end_time - start_time):.1f} 프레임/초")
        print(f"   - 최대 VRAM: {torch.cuda.max_memory_allocated(gpu_id)/1024**3:.2f}GB")
        
    except Exception as e:
        print(f"❌ GPU {gpu_id} 오류: {e}")
        import traceback
        traceback.print_exc()

def main():
    test_gpu_separation()
    
    # 프로세스 시작
    processes = []
    for gpu_id in range(min(2, torch.cuda.device_count())):
        p = mp.Process(target=gpu_compute_test, args=(gpu_id,))
        processes.append(p)
        p.start()
    
    # 완료 대기
    for p in processes:
        p.join()
    
    print("🎯 GPU 분리 테스트 완료!")

if __name__ == "__main__":
    main()
