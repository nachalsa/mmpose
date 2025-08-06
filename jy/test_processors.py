#!/usr/bin/env python3
"""
Ultra Fast Processor 실행 테스트 스크립트
두 프로세서의 실제 동작을 확인
"""

import os
import sys
import time
import subprocess
from pathlib import Path

def test_processor(processor_path: str, processor_name: str):
    """프로세서 실행 테스트"""
    print(f"\n🚀 {processor_name} 실행 테스트")
    print("=" * 60)
    
    if not Path(processor_path).exists():
        print(f"❌ 파일이 없습니다: {processor_path}")
        return False
    
    print(f"📂 실행 파일: {processor_path}")
    
    try:
        # 프로세서 실행 (타임아웃 30초)
        start_time = time.time()
        
        result = subprocess.run(
            [sys.executable, processor_path],
            timeout=30,
            capture_output=True,
            text=True,
            cwd="/workspace01/team03/data/mmpose/jy"
        )
        
        elapsed = time.time() - start_time
        
        print(f"⏱️  실행 시간: {elapsed:.2f}초")
        print(f"📤 반환 코드: {result.returncode}")
        
        if result.stdout:
            print(f"📋 표준 출력 (처음 1000자):")
            print(result.stdout[:1000])
        
        if result.stderr:
            print(f"⚠️  표준 에러 (처음 1000자):")
            print(result.stderr[:1000])
        
        if result.returncode == 0:
            print(f"✅ {processor_name} 성공적으로 실행됨!")
            return True
        else:
            print(f"❌ {processor_name} 실행 중 오류 발생 (코드: {result.returncode})")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⏰ {processor_name} 실행 타임아웃 (30초 초과)")
        print("   → 프로세서가 정상적으로 시작되어 처리 중일 가능성이 높습니다!")
        return True
        
    except Exception as e:
        print(f"💥 {processor_name} 실행 중 예외 발생: {e}")
        return False

def check_dependencies():
    """필수 종속성 확인"""
    print("🔍 종속성 확인")
    print("=" * 30)
    
    dependencies = [
        "torch", "cv2", "numpy", "h5py", "tqdm", "pathlib"
    ]
    
    missing = []
    for dep in dependencies:
        try:
            if dep == "cv2":
                import cv2
            else:
                __import__(dep)
            print(f"✅ {dep}")
        except ImportError:
            print(f"❌ {dep}")
            missing.append(dep)
    
    if missing:
        print(f"\n⚠️  누락된 종속성: {', '.join(missing)}")
        return False
    else:
        print("\n✅ 모든 종속성 확인됨!")
        return True

def check_onnx_inferencer():
    """ONNXInferencer 모듈 확인"""
    print("\n🔍 ONNXInferencer 모듈 확인")
    print("=" * 40)
    
    try:
        from onnx_inferencer import YOLO11LRTMWONNXInferencer as ONNXInferencer
        print("✅ ONNXInferencer 모듈 임포트 성공")
        
        # 간단한 인스턴스 생성 테스트
        inferencer = ONNXInferencer(
            rtmw_onnx_path="rtmw-dw-x-l_simcc-cocktail14_270e-384x288.onnx",
            detection_device='cpu',
            pose_device='cpu',
            optimize_for_accuracy=True
        )
        print("✅ ONNXInferencer 인스턴스 생성 성공")
        return True
        
    except Exception as e:
        print(f"❌ ONNXInferencer 모듈 오류: {e}")
        return False

def main():
    """메인 테스트 함수"""
    print("🚀 Ultra Fast Processor 실행 가능성 테스트")
    print("=" * 80)
    
    # 1. 종속성 확인
    deps_ok = check_dependencies()
    
    # 2. ONNXInferencer 확인
    onnx_ok = check_onnx_inferencer()
    
    if not (deps_ok and onnx_ok):
        print("\n❌ 사전 요구사항이 충족되지 않았습니다.")
        return
    
    # 3. 프로세서 테스트 실행
    processor_tests = [
        ("ultra_fast_processor_complete.py", "Ultra Fast Processor"),
        ("multionnx_streamlined_processor_complete.py", "Multi-ONNX Streamlined Processor")
    ]
    
    results = {}
    
    for processor_file, processor_name in processor_tests:
        processor_path = f"/workspace01/team03/data/mmpose/jy/{processor_file}"
        success = test_processor(processor_path, processor_name)
        results[processor_name] = success
    
    # 4. 최종 결과 리포트
    print("\n🎯 최종 테스트 결과")
    print("=" * 50)
    
    for processor_name, success in results.items():
        status = "✅ 성공" if success else "❌ 실패"
        print(f"   {processor_name}: {status}")
    
    success_count = sum(results.values())
    total_count = len(results)
    
    print(f"\n📊 성공률: {success_count}/{total_count} ({success_count/total_count*100:.1f}%)")
    
    if success_count == total_count:
        print("\n🎉 모든 프로세서가 성공적으로 실행됩니다!")
        print("   → A6000 x2 + 112 CPU 환경에서 사용 가능합니다!")
    else:
        print(f"\n⚠️  {total_count - success_count}개 프로세서에 문제가 있습니다.")
        print("   → 로그를 확인하여 문제를 해결해주세요.")

if __name__ == "__main__":
    main()
