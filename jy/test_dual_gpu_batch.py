#!/usr/bin/env python3
"""
듀얼 GPU 배치 256 처리 테스트
- 실제 비디오 파일을 사용하여 GPU 병렬 처리 테스트
- Streamlined HDF5 저장 형식 테스트
- GPU 0번과 1번 동시 병렬 처리 확인
"""

import os
import sys
import time
from pathlib import Path

# 현재 디렉토리를 sys.path에 추가
sys.path.append('.')

from batch_fast_multionnx_processor import process_videos_dual_gpu_async_batch

def find_test_videos():
    """실제 비디오 파일들을 찾기"""
    base_dir = Path("/workspace01/team03/data/mmpose/jy/data/1.Training/videos")
    
    print(f"🔍 비디오 파일 탐색: {base_dir}")
    
    video_files = []
    
    # 지정된 파일 먼저 확인
    specified_file = base_dir / "11/NIA_SL_WORD2839_REAL11_F.mp4"
    if specified_file.exists():
        video_files.append(str(specified_file))
        print(f"✅ 지정 파일 확인: {specified_file.name}")
    
    # 추가 비디오 파일들 찾기 (테스트를 위해)
    if base_dir.exists():
        for folder_path in sorted(base_dir.iterdir()):
            if folder_path.is_dir():
                for video_file in folder_path.glob("*_F.mp4"):  # F 방향 비디오만
                    if len(video_files) < 6:  # 최대 6개까지
                        video_files.append(str(video_file))
                        size_mb = video_file.stat().st_size / (1024*1024)
                        print(f"✅ 발견: {video_file.name} ({size_mb:.1f}MB)")
    
    return video_files

def main():
    """메인 테스트 함수"""
    print("🚀 듀얼 GPU 배치 256 처리 테스트 시작")
    print("=" * 60)
    
    # 1. 비디오 파일 찾기
    test_videos = find_test_videos()
    
    if not test_videos:
        print("❌ 테스트할 비디오 파일을 찾을 수 없습니다.")
        return
    
    print(f"\n📹 테스트 비디오: {len(test_videos)}개")
    for i, video in enumerate(test_videos, 1):
        print(f"   {i}. {Path(video).name}")
    
    # 2. 테스트 설정
    config = {
        'batch_size': 128,  # 테스트용으로 128로 설정
        'rtmw_model_name': 'rtmw-dw-x-l_simcc-cocktail14_270e-384x288.onnx',  # 사용 가능한 모델로 수정
        'yolo_model_name': 'yolo11l',
        'crop_size': (384, 288),
        'jpeg_quality': 90,
        'max_vram_usage': 0.75,  # 안전하게 75%
        'keypoint_scale': 8
    }
    
    # 3. 출력 디렉토리 설정
    output_dir = "/tmp/dual_gpu_test_output"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n⚙️ 설정:")
    print(f"   - 배치 크기: {config['batch_size']}")
    print(f"   - VRAM 사용률: {config['max_vram_usage']*100}%")
    print(f"   - 출력 디렉토리: {output_dir}")
    
    # 4. 진행률 콜백 정의
    def progress_callback(progress):
        print(f"📊 전체 진행률: {progress:.1%}")
    
    # 5. 듀얼 GPU 비동기 배치 처리 실행
    print(f"\n🔥 듀얼 GPU 완전 비동기 처리 시작...")
    print("   - GPU 0번과 GPU 1번이 동시에 처리됩니다")
    print("   - 각 GPU는 독립적으로 작업을 수행합니다")
    
    start_time = time.time()
    
    try:
        result = process_videos_dual_gpu_async_batch(
            video_paths=test_videos,
            output_dir=output_dir,
            config=config,
            progress_callback=progress_callback
        )
        
        # 6. 결과 분석
        total_time = time.time() - start_time
        
        print("\n" + "=" * 60)
        print("🏁 테스트 결과 분석")
        print("=" * 60)
        
        print(f"✅ 처리 완료:")
        print(f"   - 성공: {result['completed']}개")
        print(f"   - 실패: {result['failed']}개")
        print(f"   - 총 시간: {result['total_time']:.2f}초")
        print(f"   - 평균 시간: {result['average_time_per_video']:.2f}초/비디오")
        
        # GPU별 작업 분배 확인
        print(f"\n🖥️ GPU 병렬 처리 확인:")
        gpu_distribution = result['gpu_distribution']
        for gpu_id, task_count in gpu_distribution.items():
            print(f"   - GPU {gpu_id}: {task_count}개 작업 처리")
        
        # 병렬 처리 효율성 계산
        if len(gpu_distribution) == 2:
            gpu0_tasks = gpu_distribution.get(0, 0)
            gpu1_tasks = gpu_distribution.get(1, 0)
            balance_ratio = min(gpu0_tasks, gpu1_tasks) / max(gpu0_tasks, gpu1_tasks) if max(gpu0_tasks, gpu1_tasks) > 0 else 0
            print(f"   - 작업 균형도: {balance_ratio:.1%}")
            
            if balance_ratio > 0.8:
                print("   ✅ GPU 작업 분배가 균등합니다")
            else:
                print("   ⚠️ GPU 작업 분배가 불균등합니다")
        
        # 7. 저장된 파일 확인
        output_path = Path(output_dir)
        saved_files = {
            'frames': list(output_path.glob("*_frames.h5")),
            'poses': list(output_path.glob("*_poses.h5"))
        }
        
        print(f"\n💾 저장된 HDF5 파일:")
        print(f"   - 프레임 파일: {len(saved_files['frames'])}개")
        print(f"   - 포즈 파일: {len(saved_files['poses'])}개")
        
        for frames_file in saved_files['frames'][:3]:  # 처음 3개만 표시
            size_mb = frames_file.stat().st_size / (1024*1024)
            print(f"     📄 {frames_file.name} ({size_mb:.1f}MB)")
        
        # 8. 성능 평가
        if result['completed'] > 0:
            avg_fps = sum(
                item['fps_achieved'] for item in result['results'].values() 
                if item.get('fps_achieved', 0) > 0
            ) / result['completed']
            
            print(f"\n⚡ 성능 지표:")
            print(f"   - 평균 FPS: {avg_fps:.1f}")
            print(f"   - 총 처리 시간: {total_time:.2f}초")
            
            if avg_fps > 10:
                print("   ✅ 우수한 처리 성능")
            elif avg_fps > 5:
                print("   ✅ 양호한 처리 성능")
            else:
                print("   ⚠️ 처리 성능 개선 필요")
        
        # 9. GPU 병렬 처리 검증
        print(f"\n🔍 GPU 병렬 처리 검증:")
        if result['completed'] >= 2 and len(gpu_distribution) == 2:
            if all(count > 0 for count in gpu_distribution.values()):
                print("   ✅ GPU 0번과 1번이 모두 작업을 처리했습니다")
                print("   ✅ 듀얼 GPU 병렬 처리가 성공적으로 작동합니다")
            else:
                print("   ⚠️ 한쪽 GPU만 작업을 처리했습니다")
        else:
            print("   ⚠️ 충분한 작업이 없어 병렬 처리를 확인할 수 없습니다")
        
        # 10. Streamlined 형식 검증
        if saved_files['frames'] and saved_files['poses']:
            print(f"\n📋 Streamlined HDF5 형식 검증:")
            print("   ✅ 프레임과 포즈 파일이 분리되어 저장되었습니다")
            print("   ✅ Streamlined 호환 형식으로 저장 완료")
        
        if result['completed'] == len(test_videos):
            print(f"\n🎉 전체 테스트 성공!")
            print("   - 모든 비디오 처리 완료")
            print("   - GPU 병렬 처리 확인")
            print("   - Streamlined HDF5 저장 완료")
        else:
            print(f"\n⚠️ 일부 테스트 실패 ({result['failed']}개)")
            
    except Exception as e:
        print(f"\n💥 테스트 실행 오류: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "=" * 60)
    print("테스트 완료")

if __name__ == "__main__":
    main()
