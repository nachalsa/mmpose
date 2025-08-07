#!/usr/bin/env python3
"""
듀얼 GPU 비동기 배치 프로세서 테스트 - 수정된 버전
"""

import sys
import os
import time
from pathlib import Path

# 현재 디렉토리를 패스에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def find_test_videos():
    """테스트용 비디오 파일 찾기"""
    base_paths = [
        '/workspace01/team03/data/mmpose/jy/data/1.Training/videos/11',
        '/workspace01/team03/data/1.Training/1.원천데이터/TS/WORD'
    ]
    
    video_files = []
    for base_path in base_paths:
        if os.path.exists(base_path):
            print(f"📁 탐색: {base_path}")
            for root, dirs, files in os.walk(base_path):
                for file in files:
                    if file.endswith('.mp4') and 'F_' in file:
                        full_path = os.path.join(root, file)
                        size_mb = os.path.getsize(full_path) / (1024 * 1024)
                        video_files.append((full_path, size_mb))
                        if len(video_files) >= 6:  # 6개만 수집
                            break
                if len(video_files) >= 6:
                    break
        if len(video_files) >= 6:
            break
    
    return video_files[:6]  # 최대 6개

def test_import():
    """모듈 import 테스트"""
    try:
        from batch_fast_multionnx_processor import process_videos_dual_gpu_async_batch
        print("✅ 함수 import 성공")
        return True
    except Exception as e:
        print(f"❌ Import 오류: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_pytorch_cuda():
    """PyTorch CUDA 환경 테스트"""
    try:
        import torch
        print(f"✅ PyTorch CUDA 사용 가능: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"📊 GPU 개수: {torch.cuda.device_count()}")
            for i in range(min(2, torch.cuda.device_count())):
                props = torch.cuda.get_device_properties(i)
                memory_gb = props.total_memory / (1024**3)
                print(f"   GPU {i}: {torch.cuda.get_device_name(i)} ({memory_gb:.1f}GB)")
        return torch.cuda.is_available() and torch.cuda.device_count() >= 2
    except Exception as e:
        print(f"❌ PyTorch 오류: {e}")
        return False

def test_dual_gpu_processing():
    """실제 듀얼 GPU 처리 테스트"""
    if not test_import():
        return False
    
    if not test_pytorch_cuda():
        print("⚠️ 듀얼 GPU 환경이 아닙니다. 테스트를 계속 진행합니다.")
    
    # 테스트 비디오 수집
    video_files = find_test_videos()
    if not video_files:
        print("❌ 테스트용 비디오 파일을 찾을 수 없습니다.")
        return False
    
    print(f"\n🔍 찾은 테스트 비디오: {len(video_files)}개")
    for i, (video_path, size_mb) in enumerate(video_files[:4]):
        print(f"   {i+1}. {os.path.basename(video_path)} ({size_mb:.1f}MB)")
    
    # 테스트 설정
    config = {
        'batch_size': 64,  # 테스트용으로 작게 설정
        'rtmw_model_name': 'rtmw-dw-x-l_8xb320-270e_cocktail14-384x288.onnx',
        'yolo_model_name': 'yolo11l',
        'crop_size': (384, 288),
        'jpeg_quality': 90,
        'max_vram_usage': 0.6,  # 안전하게 설정
        'keypoint_scale': 8
    }
    
    # 출력 디렉토리 설정
    output_dir = '/tmp/test_dual_gpu_fixed'
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 출력 디렉토리: {output_dir}")
    
    # 진행률 콜백
    def progress_callback(progress):
        print(f"📊 전체 진행률: {progress:.1%}")
    
    # 듀얼 GPU 비동기 처리 실행
    test_videos = [video_path for video_path, _ in video_files[:4]]  # 4개 비디오로 테스트
    
    print(f"\n🚀 듀얼 GPU 비동기 처리 시작: {len(test_videos)}개 비디오")
    start_time = time.time()
    
    try:
        from batch_fast_multionnx_processor import process_videos_dual_gpu_async_batch
        
        result = process_videos_dual_gpu_async_batch(
            video_paths=test_videos,
            output_dir=output_dir,
            config=config,
            progress_callback=progress_callback
        )
        
        processing_time = time.time() - start_time
        
        print("\n🏁 테스트 결과:")
        print(f"   ✅ 성공: {result['completed']}개")
        print(f"   ❌ 실패: {result['failed']}개")
        print(f"   ⏱️  총 시간: {result['total_time']:.2f}초")
        print(f"   ⚡ 평균 시간: {result['average_time_per_video']:.2f}초/비디오")
        print(f"   🖥️  GPU별 분배: {result['gpu_distribution']}")
        
        # 저장된 파일 확인
        saved_files = list(Path(output_dir).glob('*.h5'))
        print(f"\n💾 저장된 HDF5 파일: {len(saved_files)}개")
        for file in saved_files[:3]:
            size_mb = file.stat().st_size / (1024 * 1024)
            print(f"   - {file.name} ({size_mb:.1f}MB)")
        
        # 성공 여부 판단
        if result['completed'] > 0:
            print("\n🎉 테스트 성공!")
            if result['completed'] == len(test_videos):
                print("   - 모든 비디오 처리 완료 ✅")
            print("   - Streamlined HDF5 형식 저장 완료 ✅")
            
            # GPU 병렬 처리 확인
            gpu_dist = result['gpu_distribution']
            if len(gpu_dist) == 2 and all(count > 0 for count in gpu_dist.values()):
                print("   - 듀얼 GPU 병렬 처리 확인 ✅")
            else:
                print(f"   - GPU 분배 상황: {gpu_dist}")
            
            return True
        else:
            print("\n⚠️ 모든 비디오 처리 실패")
            return False
            
    except Exception as e:
        print(f"\n💥 처리 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """메인 함수"""
    print("🧪 듀얼 GPU 비동기 배치 프로세서 테스트")
    print("=" * 50)
    
    success = test_dual_gpu_processing()
    
    print("\n" + "=" * 50)
    if success:
        print("✅ 전체 테스트 성공!")
    else:
        print("❌ 테스트 실패")
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())
