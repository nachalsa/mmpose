#!/usr/bin/env python3
"""
비동기 GPU 배치 처리 테스트
확실한 병렬 처리 및 Streamlined HDF5 형식 확인
"""

import os
import sys
import time
import torch
from pathlib import Path

# 경로 추가
current_dir = os.getcwd()
sys.path.insert(0, current_dir)

print("🧪 비동기 GPU 배치 처리 테스트")
print("=" * 60)

def check_gpu_status():
    """GPU 상태 확인"""
    print("\n🔍 GPU 상태 확인:")
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"   - GPU 개수: {gpu_count}")
        
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"   - GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
        
        return gpu_count >= 2
    else:
        print("   - GPU 사용 불가")
        return False

def test_async_processing():
    """비동기 처리 테스트"""
    print("\n🚀 비동기 처리 테스트 시작")
    
    try:
        from batch_fast_multionnx_processor import process_videos_dual_gpu_async_batch
        print("✅ 비동기 함수 임포트 성공")
        
        # 테스트 비디오 파일들 찾기
        data_dir = Path("/workspace01/team03/data/mmpose/jy/data/1.Training/videos")
        video_files = []
        
        if data_dir.exists():
            for subdir in data_dir.iterdir():
                if subdir.is_dir():
                    for video_file in subdir.glob("*_F.mp4"):
                        video_files.append(str(video_file))
                        if len(video_files) >= 10:  # 테스트용 10개만
                            break
                    if len(video_files) >= 10:
                        break
        
        if not video_files:
            print("❌ 테스트용 비디오 파일을 찾을 수 없습니다")
            return False
        
        print(f"📹 테스트 비디오 {len(video_files)}개 발견")
        
        # 출력 디렉토리 설정
        output_dir = "/workspace01/team03/data/async_batch_test_output"
        os.makedirs(output_dir, exist_ok=True)
        
        # GPU 설정
        config = {
            'rtmw_model_name': 'rtmw-dw-x-l_simcc-cocktail14_270e-384x288.onnx',
            'batch_size': 256,
            'keypoint_scale': 8,
            'jpeg_quality': 90,
            'max_vram_usage': 0.8
        }
        
        print(f"\n⚡ 비동기 듀얼 GPU 처리 시작:")
        print(f"   - 비디오: {len(video_files)}개")
        print(f"   - 배치 크기: {config['batch_size']}")
        print(f"   - 출력: {output_dir}")
        
        # 진행률 추적
        progress_history = []
        
        def progress_callback(progress):
            progress_history.append(progress)
            percent = progress * 100
            if len(progress_history) % 10 == 0:  # 10번에 한 번만 출력
                print(f"📈 전체 진행률: {percent:.1f}%")
        
        # 비동기 처리 실행
        start_time = time.time()
        summary = process_videos_dual_gpu_async_batch(
            video_paths=video_files,
            output_dir=output_dir,
            config=config,
            progress_callback=progress_callback
        )
        total_time = time.time() - start_time
        
        # 결과 분석
        print(f"\n📊 비동기 처리 결과:")
        print(f"   - 총 처리: {summary['total_videos']}개")
        print(f"   - 성공: {summary['completed']}개") 
        print(f"   - 실패: {summary['failed']}개")
        print(f"   - 성공률: {summary['completed']/summary['total_videos']*100:.1f}%")
        print(f"   - 총 시간: {summary['total_time']:.2f}초")
        print(f"   - 평균 시간: {summary['average_time_per_video']:.2f}초/비디오")
        print(f"   - GPU별 분배: {summary['gpu_distribution']}")
        
        # GPU 병렬성 분석
        gpu_results = {}
        for result in summary['results'].values():
            gpu_id = result.get('gpu_id', -1)
            if gpu_id >= 0:
                if gpu_id not in gpu_results:
                    gpu_results[gpu_id] = []
                gpu_results[gpu_id].append(result.get('timestamp', 0))
        
        print(f"\n🔥 GPU 병렬성 분석:")
        for gpu_id, timestamps in gpu_results.items():
            if timestamps:
                timestamps.sort()
                print(f"   - GPU {gpu_id}: {len(timestamps)}개 작업")
                if len(timestamps) >= 2:
                    overlap_count = 0
                    for i in range(len(timestamps)-1):
                        time_diff = timestamps[i+1] - timestamps[i]
                        if time_diff < 5.0:  # 5초 이내 차이면 병렬 처리로 간주
                            overlap_count += 1
                    print(f"     → 병렬 처리된 작업: {overlap_count}개")
        
        # HDF5 파일 확인
        print(f"\n💾 생성된 HDF5 파일 확인:")
        output_path = Path(output_dir)
        frames_files = list(output_path.glob("*_frames.h5"))
        poses_files = list(output_path.glob("*_poses.h5"))
        
        print(f"   - 프레임 파일: {len(frames_files)}개")
        print(f"   - 포즈 파일: {len(poses_files)}개")
        
        if frames_files and poses_files:
            print("✅ Streamlined HDF5 형식 파일 생성 확인")
        else:
            print("⚠️ HDF5 파일 생성 문제 확인 필요")
        
        return summary['completed'] > 0
        
    except Exception as e:
        print(f"💥 비동기 처리 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """메인 테스트"""
    try:
        # GPU 상태 확인
        if not check_gpu_status():
            print("❌ 듀얼 GPU가 필요합니다")
            return
        
        # 비동기 처리 테스트
        if test_async_processing():
            print("\n🎉 모든 테스트 성공!")
        else:
            print("\n❌ 테스트 실패")
            
    except Exception as e:
        print(f"💥 메인 테스트 오류: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
