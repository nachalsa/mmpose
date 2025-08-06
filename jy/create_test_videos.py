#!/usr/bin/env python3
"""
테스트용 비디오 파일 생성
실제 NIA_SL    # 여러 폴더에 테스트 비디오 생성
    test_videos = [
        ("11", "NIA_SL_WORD1070_REAL11_D.mp4"),
        ("12", "NIA_SL_WORD1071_REAL12_D.mp4"),
        ("13", "NIA_SL_WORD1072_REAL13_D.mp4"),
    ]070_REAL11_D.mp4 형태의 테스트 비디오 생성
"""

import cv2
import numpy as np
from pathlib import Path

def create_test_video(output_path: str, duration_seconds: int = 3, fps: int = 30):
    """테스트용 비디오 생성"""
    
    # 비디오 설정
    width, height = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    total_frames = duration_seconds * fps
    
    print(f"🎬 테스트 비디오 생성: {output_path}")
    print(f"   - 해상도: {width}x{height}")
    print(f"   - FPS: {fps}")
    print(f"   - 총 프레임: {total_frames}")
    
    for i in range(total_frames):
        # 간단한 애니메이션 프레임 생성
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # 배경 색상 변화
        bg_color = int((i / total_frames) * 100 + 50)
        frame[:] = (bg_color, bg_color + 20, bg_color + 40)
        
        # 움직이는 원 (사람 시뮬레이션)
        center_x = int(width/2 + 100 * np.sin(i * 0.1))
        center_y = int(height/2 + 50 * np.cos(i * 0.1))
        cv2.circle(frame, (center_x, center_y), 30, (255, 255, 255), -1)
        
        # 텍스트 추가
        cv2.putText(frame, f'Test Frame {i+1}', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        out.write(frame)
    
    out.release()
    print(f"✅ 비디오 생성 완료: {Path(output_path).stat().st_size / (1024*1024):.1f}MB")

def main():
    """테스트 비디오들 생성"""
    base_dir = Path("/workspace01/team03/data/mmpose/jy/data/1.Training/videos")
    
    # 여러 폴더에 테스트 비디오 생성
    test_videos = [
        ("11", "NIA_SL_WORD1070_REAL11_D.mp4"),
        ("12", "NIA_SL_WORD1071_REAL12_D.mp4"),
        ("13", "NIA_SL_WORD1072_REAL13_D.mp4"),
    ]
    
    for folder, filename in test_videos:
        folder_path = base_dir / folder
        folder_path.mkdir(parents=True, exist_ok=True)
        
        video_path = folder_path / filename
        create_test_video(str(video_path), duration_seconds=2)
    
    print(f"\n🎉 모든 테스트 비디오 생성 완료!")
    print(f"📁 생성 위치: {base_dir}")

if __name__ == "__main__":
    main()
