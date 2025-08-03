#!/usr/bin/env python3
"""
결합 데이터 시각화 테스트 스크립트
"""

import sys
import os
sys.path.append('/home/ty/rtmw/02/mmpose/jy')

from video_processor_yolo11l import VideoProcessorYOLO11L
import matplotlib
matplotlib.use('Agg')  # GUI 없이 실행

def test_visualization():
    """시각화 기능 테스트"""
    # 모델 경로 설정
    rtmw_config = "../configs/wholebody_2d_keypoint/rtmpose/cocktail14/rtmw-x_8xb320-270e_cocktail14-384x288.py"
    rtmw_checkpoint = "../models/rtmw-x_simcc-cocktail14_pt-ucoco_270e-384x288-f840f204_20231122.pth"
    
    # 처리기 생성
    processor = VideoProcessorYOLO11L(
        rtmw_config=rtmw_config,
        rtmw_checkpoint=rtmw_checkpoint,
        output_dir="video_processing_outputs"
    )
    
    # 결합 데이터 파일 찾기
    combined_dir = processor.output_dir / "combined_data"
    json_files = list(combined_dir.glob("*_combined.json"))
    
    if json_files:
        print(f"📁 {len(json_files)}개의 결합 데이터 파일 발견")
        
        # 첫 번째 파일로 시각화 테스트
        test_file = json_files[0]
        print(f"🎨 시각화 테스트: {test_file.name}")
        
        # 단일 데이터 시각화
        vis_path = processor.visualize_combined_data(str(test_file))
        if vis_path:
            print(f"✅ 단일 데이터 시각화 완료: {vis_path}")
        else:
            print("❌ 단일 데이터 시각화 실패")
        
        # 전체 요약 시각화
        print(f"📊 전체 요약 시각화 생성 중...")
        summary_path = processor.create_combined_data_summary()
        if summary_path:
            print(f"✅ 요약 시각화 완료: {summary_path}")
        else:
            print("❌ 요약 시각화 실패")
            
        # 결합 데이터 로드 테스트
        print(f"🧪 결합 데이터 로드 테스트...")
        crop_image, keypoints, scores, metadata = processor.load_combined_data(str(test_file))
        
        if crop_image is not None:
            print(f"✅ 로드 성공:")
            print(f"   - 이미지 크기: {crop_image.shape}")
            print(f"   - 키포인트 수: {len(keypoints)}")
            print(f"   - 평균 신뢰도: {scores.mean():.3f}")
            print(f"   - 프레임 인덱스: {metadata['frame_info']['frame_idx']}")
        else:
            print("❌ 로드 실패")
    else:
        print("❌ 결합 데이터 파일이 없습니다.")

if __name__ == "__main__":
    test_visualization()
