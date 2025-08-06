#!/usr/bin/env python3
"""
Fast Multi-ONNX Processor - 커스터마이징 친화적 실행
A6000 x2 GPU 최적화, 빠른 실행, 완전 GPU 가속
"""

from fast_multionnx_processor import FastMultiONNXProcessor

def run_fast_multionnx():
    """Fast Multi-ONNX 실행 - 커스터마이징 가능"""
    print("🚀 Fast Multi-ONNX Processor")
    print("⚡ A6000 x2 GPU 완전 가속 버전")
    print("=" * 60)
    
    # ============================
    # 커스터마이징 설정
    # ============================
    
    # 기본 설정
    DATA_ROOT = "/workspace01/team03/data/mmpose/jy/data/1.Training"
    OUTPUT_DIR = "/workspace01/team03/data/fast_multionnx_output"
    DIRECTION = "F"  # F 또는 B
    ITEM_TYPES = ["WORD"]  # ["WORD", "MORPH"] 등
    
    # 성능 설정 (A6000 x2 최적화)
    NUM_CPU_WORKERS = 8      # CPU 워커 수 (112코어 환경에 적합)
    KEYPOINT_SCALE = 8       # 키포인트 스케일링
    JPEG_QUALITY = 90        # JPEG 압축 품질 (90=고품질)
    ENABLE_HDF5_BATCH = True # HDF5 배치 파일 생성 여부
    
    # 제한 설정
    MAX_VIDEOS = None        # None = 전체, 숫자 = 제한
    
    # ============================
    # 프로세서 초기화
    # ============================
    
    processor = FastMultiONNXProcessor(
        data_root=DATA_ROOT,
        output_dir=OUTPUT_DIR,
        rtmw_model_name="rtmw-dw-x-l_simcc-cocktail14_270e-384x288.onnx",
        direction=DIRECTION,
        item_types=ITEM_TYPES,
        num_cpu_workers=NUM_CPU_WORKERS,
        yolo_device="auto",
        pose_device="auto", 
        keypoint_scale=KEYPOINT_SCALE,
        jpeg_quality=JPEG_QUALITY,
        enable_hdf5_batch=ENABLE_HDF5_BATCH
    )
    
    # ============================
    # 사용자 상호작용
    # ============================
    
    print(f"📊 비디오 수집 중...")
    all_videos = processor.collect_all_videos()
    
    if not all_videos:
        print("❌ 처리할 비디오가 없습니다")
        return
    
    print(f"📹 총 {len(all_videos)}개 비디오 발견")
    
    if MAX_VIDEOS is None:
        # 사용자에게 제한 여부 물어보기
        choice = input("\\n모든 비디오를 처리하시겠습니까? (y/N): ").strip().lower()
        if choice != 'y':
            try:
                max_count = int(input("처리할 최대 비디오 수 입력: "))
                MAX_VIDEOS = max_count
            except ValueError:
                print("❌ 잘못된 입력. 전체 처리합니다.")
                MAX_VIDEOS = None
    
    if MAX_VIDEOS:
        print(f"🔢 처리 제한: {MAX_VIDEOS}개 비디오")
    
    # ============================
    # 실행
    # ============================
    
    print("\\n🚀 Fast Multi-ONNX 처리 시작!")
    print(f"   - A6000 x2 GPU 완전 가속")
    print(f"   - {NUM_CPU_WORKERS}개 CPU 워커") 
    print(f"   - 키포인트 스케일: {KEYPOINT_SCALE}")
    print(f"   - JPEG 품질: {JPEG_QUALITY}")
    print("="*60)
    
    # 처리 실행
    processor.process_videos_parallel(max_videos=MAX_VIDEOS)

def run_quick_test():
    """빠른 테스트 실행"""
    print("🚀 Fast Multi-ONNX 빠른 테스트")
    print("=" * 50)
    
    processor = FastMultiONNXProcessor(
        data_root="/workspace01/team03/data/mmpose/jy/data/1.Training",
        output_dir="/workspace01/team03/data/fast_test_output",
        direction="F",
        item_types=["WORD"],
        num_cpu_workers=4,
        keypoint_scale=8,
        jpeg_quality=90,
        enable_hdf5_batch=False
    )
    
    print("⚡ 10개 비디오 빠른 테스트 시작!")
    processor.process_videos_parallel(max_videos=10)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        run_quick_test()
    else:
        run_fast_multionnx()
