#!/usr/bin/env python3
"""
완성된 시스템 통합 테스트
1. 모델 다운로드 캐싱 테스트
2. HDF5 포즈 시각화 테스트  
3. JPEG 이미지 시각화 테스트
4. 완전한 처리 파이프라인 검증
"""

import os
import sys
import time
import subprocess
from pathlib import Path

def test_model_caching():
    """모델 다운로드 캐싱 테스트"""
    print("🔄 1. 모델 다운로드 캐싱 테스트")
    print("=" * 50)
    
    from onnx_inferencer import YOLO11LRTMWONNXInferencer
    
    # 첫 번째 실행 - 다운로드 시간 측정
    print("📥 첫 번째 모델 로딩 (다운로드 포함):")
    start_time = time.time()
    
    try:
        inferencer1 = YOLO11LRTMWONNXInferencer(
            rtmw_onnx_path="rtmw-dw-x-l_simcc-cocktail14_270e-384x288.onnx",
            detection_device="cpu",
            pose_device="cpu"
        )
        first_load_time = time.time() - start_time
        print(f"✅ 첫 번째 로딩 완료: {first_load_time:.2f}초")
        del inferencer1
    except Exception as e:
        print(f"❌ 첫 번째 로딩 실패: {e}")
        return False
    
    # 두 번째 실행 - 캐시 활용 시간 측정
    print("\n🚀 두 번째 모델 로딩 (캐시 활용):")
    start_time = time.time()
    
    try:
        inferencer2 = YOLO11LRTMWONNXInferencer(
            rtmw_onnx_path="rtmw-dw-x-l_simcc-cocktail14_270e-384x288.onnx",
            detection_device="cpu",
            pose_device="cpu"
        )
        second_load_time = time.time() - start_time
        print(f"✅ 두 번째 로딩 완료: {second_load_time:.2f}초")
        del inferencer2
    except Exception as e:
        print(f"❌ 두 번째 로딩 실패: {e}")
        return False
    
    # 성능 개선 확인
    if second_load_time < first_load_time * 0.5:  # 50% 이상 빨라져야 함
        speedup = first_load_time / second_load_time
        print(f"🎉 캐싱 성공! {speedup:.1f}배 빨라짐")
        return True
    else:
        print(f"⚠️ 캐싱 효과 부족: {first_load_time:.2f}s → {second_load_time:.2f}s")
        return False

def test_hdf5_visualization():
    """HDF5 시각화 테스트"""
    print("\n🎨 2. HDF5 포즈 시각화 테스트")
    print("=" * 50)
    
    # HDF5 파일 찾기
    hdf5_files = list(Path("/workspace01/team03/data/word").glob("*_frames.h5"))
    if not hdf5_files:
        print("⚠️ HDF5 파일을 찾을 수 없습니다")
        return False
    
    frames_file = hdf5_files[0]
    poses_file = str(frames_file).replace("_frames.h5", "_poses.h5")
    
    if not Path(poses_file).exists():
        print(f"⚠️ 대응하는 포즈 파일이 없습니다: {poses_file}")
        return False
    
    print(f"📁 테스트 파일: {frames_file.name}")
    
    # 시각화 실행
    cmd = [
        "python", "hdf5_pose_visualizer.py",
        "--input", str(frames_file),
        "--poses", poses_file,
        "--output", "test_hdf5_viz",
        "--max-frames", "5",
        "--analysis"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            print("✅ HDF5 시각화 성공")
            print(f"   출력: test_hdf5_viz/")
            
            # 출력 파일 확인
            output_dir = Path("test_hdf5_viz")
            if output_dir.exists():
                image_files = list(output_dir.rglob("*.jpg"))
                json_files = list(output_dir.rglob("*.json"))
                print(f"   생성된 이미지: {len(image_files)}개")
                print(f"   분석 파일: {len(json_files)}개")
            
            return True
        else:
            print(f"❌ HDF5 시각화 실패: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("❌ HDF5 시각화 타임아웃")
        return False
    except Exception as e:
        print(f"❌ HDF5 시각화 오류: {e}")
        return False

def test_jpeg_visualization():
    """JPEG 시각화 테스트"""
    print("\n📷 3. JPEG 이미지 시각화 테스트")
    print("=" * 50)
    
    # 테스트 이미지 디렉토리 확인
    test_images_dir = Path("test_images")
    if not test_images_dir.exists() or not list(test_images_dir.glob("*.jpg")):
        print("⚠️ 테스트 이미지가 없습니다. 생성 중...")
        
        # 테스트 이미지 생성
        try:
            import cv2
            import numpy as np
            
            test_images_dir.mkdir(exist_ok=True)
            
            for i in range(3):
                img = np.zeros((288, 384, 3), dtype=np.uint8)
                img[:] = (50 + i*30, 100 + i*20, 150 + i*10)
                
                cv2.putText(img, f'JPEG Test {i+1}', (50, 150), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                cv2.imwrite(str(test_images_dir / f"test_{i+1:03d}.jpg"), img)
            
            print(f"✅ 테스트 이미지 {len(list(test_images_dir.glob('*.jpg')))}개 생성")
        except Exception as e:
            print(f"❌ 테스트 이미지 생성 실패: {e}")
            return False
    
    # JPEG 시각화 실행
    cmd = [
        "python", "hdf5_pose_visualizer.py",
        "--input", str(test_images_dir),
        "--output", "test_jpeg_viz"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            print("✅ JPEG 시각화 성공")
            print(f"   출력: test_jpeg_viz/")
            
            # 출력 파일 확인
            output_dir = Path("test_jpeg_viz")
            if output_dir.exists():
                image_files = list(output_dir.glob("*.jpg"))
                print(f"   생성된 시각화 이미지: {len(image_files)}개")
            
            return True
        else:
            print(f"❌ JPEG 시각화 실패: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("❌ JPEG 시각화 타임아웃")
        return False
    except Exception as e:
        print(f"❌ JPEG 시각화 오류: {e}")
        return False

def test_processor_integration():
    """프로세서 통합 테스트"""
    print("\n🚀 4. 프로세서 통합 테스트")
    print("=" * 50)
    
    try:
        # 완성된 프로세서 임포트 테스트
        from ultra_fast_processor_complete import UltraFastBatchProcessor
        from multionnx_streamlined_processor_complete import StreamlinedVideoProcessor
        
        print("✅ 프로세서 클래스 임포트 성공")
        
        # 기본 초기화 테스트
        ultra_processor = UltraFastBatchProcessor(
            data_root="/workspace01/team03/data",
            output_dir="/tmp/test_output",
            num_gpu_workers=1,
            gpu_batch_size=64,
            num_cpu_workers=4,
            num_video_loaders=2
        )
        
        streamlined_processor = StreamlinedVideoProcessor()
        
        print("✅ 프로세서 초기화 성공")
        print(f"   UltraFastBatchProcessor: GPU 워커 {ultra_processor.num_gpu_workers}개")
        print(f"   StreamlinedVideoProcessor: ONNX 추론기 준비 완료")
        
        return True
        
    except Exception as e:
        print(f"❌ 프로세서 통합 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """메인 테스트 실행"""
    print("🧪 완성된 시스템 통합 테스트")
    print("=" * 80)
    
    test_results = []
    
    # 1. 모델 캐싱 테스트
    try:
        result1 = test_model_caching()
        test_results.append(("모델 캐싱", result1))
    except Exception as e:
        print(f"❌ 모델 캐싱 테스트 오류: {e}")
        test_results.append(("모델 캐싱", False))
    
    # 2. HDF5 시각화 테스트
    try:
        result2 = test_hdf5_visualization()
        test_results.append(("HDF5 시각화", result2))
    except Exception as e:
        print(f"❌ HDF5 시각화 테스트 오류: {e}")
        test_results.append(("HDF5 시각화", False))
    
    # 3. JPEG 시각화 테스트
    try:
        result3 = test_jpeg_visualization()
        test_results.append(("JPEG 시각화", result3))
    except Exception as e:
        print(f"❌ JPEG 시각화 테스트 오류: {e}")
        test_results.append(("JPEG 시각화", False))
    
    # 4. 프로세서 통합 테스트
    try:
        result4 = test_processor_integration()
        test_results.append(("프로세서 통합", result4))
    except Exception as e:
        print(f"❌ 프로세서 통합 테스트 오류: {e}")
        test_results.append(("프로세서 통합", False))
    
    # 최종 결과
    print("\n" + "=" * 80)
    print("🏁 최종 테스트 결과")
    print("=" * 80)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ 통과" if result else "❌ 실패"
        print(f"   {test_name:<20} {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 성공률: {passed}/{total} ({100*passed//total}%)")
    
    if passed == total:
        print("🎉 모든 테스트 통과! 시스템이 완전히 작동합니다.")
        print("\n📋 사용 준비 상태:")
        print("   ✅ A6000 x2 + 112 CPU 최적화 프로세서")
        print("   ✅ 모델 자동 다운로드 및 캐싱")
        print("   ✅ HDF5 및 JPEG 시각화 도구")
        print("   ✅ 통합 처리 파이프라인")
    else:
        print(f"⚠️ {total-passed}개 테스트 실패 - 문제를 확인하세요")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
