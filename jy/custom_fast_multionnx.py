#!/usr/bin/env python3
"""
커스터마이징 가능한 Fast Multi-ONNX 프로세서
사용법: python custom_fast_multionnx.py
"""

from simple_fast_multionnx import SimpleFastProcessor

def main():
    """커스터마이징 가능한 메인 실행"""
    print("🚀 커스터마이징 가능한 Fast Multi-ONNX Processor")
    print("=" * 80)
    
    # 기본 설정 출력
    print("\n📋 기본 설정:")
    print("   - 데이터 루트: /workspace01/team03/data/mmpose/jy/data/1.Training")
    print("   - 출력 디렉토리: /workspace01/team03/data/custom_output")
    print("   - 방향: F (정면)")
    print("   - 타입: WORD")
    print("   - 키포인트 스케일: 8")
    print("   - JPEG 품질: 90")
    
    # 커스터마이징 옵션
    print("\n🔧 커스터마이징 옵션:")
    
    # 출력 디렉토리 커스터마이징
    custom_output = input("출력 디렉토리를 변경하시겠습니까? (현재: custom_output, 엔터: 기본값 유지): ").strip()
    if not custom_output:
        custom_output = "custom_output"
    output_dir = f"/workspace01/team03/data/{custom_output}"
    
    # 키포인트 스케일 커스터마이징
    scale_input = input("키포인트 스케일을 변경하시겠습니까? (현재: 8, 엔터: 기본값 유지): ").strip()
    keypoint_scale = 8
    if scale_input:
        try:
            keypoint_scale = int(scale_input)
            if keypoint_scale <= 0:
                keypoint_scale = 8
        except ValueError:
            keypoint_scale = 8
    
    # JPEG 품질 커스터마이징
    quality_input = input("JPEG 품질을 변경하시겠습니까? (현재: 90, 1-100, 엔터: 기본값 유지): ").strip()
    jpeg_quality = 90
    if quality_input:
        try:
            jpeg_quality = int(quality_input)
            if jpeg_quality < 1 or jpeg_quality > 100:
                jpeg_quality = 90
        except ValueError:
            jpeg_quality = 90
    
    # 처리할 비디오 수 제한
    limit_input = input("처리할 비디오 수를 제한하시겠습니까? (엔터: 제한없음): ").strip()
    max_videos = None
    if limit_input:
        try:
            max_videos = int(limit_input)
            if max_videos <= 0:
                max_videos = None
        except ValueError:
            max_videos = None
    
    print("\n✅ 최종 설정:")
    print(f"   - 출력 디렉토리: {output_dir}")
    print(f"   - 키포인트 스케일: {keypoint_scale}")
    print(f"   - JPEG 품질: {jpeg_quality}")
    print(f"   - 처리 제한: {'없음' if max_videos is None else f'{max_videos}개'}")
    
    confirm = input("\n이 설정으로 처리를 시작하시겠습니까? (y/N): ").strip().lower()
    if confirm != 'y':
        print("❌ 처리가 취소되었습니다.")
        return
    
    # 프로세서 초기화
    try:
        processor = SimpleFastProcessor(
            data_root="/workspace01/team03/data/mmpose/jy/data/1.Training",
            output_dir=output_dir,
            direction="F",
            item_types=["WORD"],
            keypoint_scale=keypoint_scale,
            jpeg_quality=jpeg_quality,
            gpu_ids=[0, 1]  # A6000 x2 사용
        )
        
        print(f"\n🚀 처리 시작...")
        processor.process_videos_simple(max_videos=max_videos)
        
    except KeyboardInterrupt:
        print("\n⚠️ 사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")

if __name__ == "__main__":
    main()
