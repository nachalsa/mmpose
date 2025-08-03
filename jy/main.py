#!/usr/bin/env python3
"""
Main Module for RTMW-x
RTMW-x Intel XPU 추론 메인 실행 파일
"""

import cv2
import time
import traceback

from detector import SimplePersonDetector
from pose_estimator import RTMWXEstimator
from visualizer import RTMWVisualizer
from utils import find_and_download_rtmw_model, check_xpu_availability
from config import DEFAULT_DEVICE, RTMW_INPUT_SIZE, DEFAULT_CONF_THRESH, FPS_UPDATE_INTERVAL


def main():
    """RTMW-x Intel XPU 추론 메인 함수"""
    print("=== RTMW-x Intel XPU WholeBody 포즈 추론 ===")
    print("RTMW-x: Real-Time Multi-Person WholeBody Pose Estimation")
    print("키포인트: Body(17) + Face(68) + Hands(48) = 133개")
    
    # Intel XPU 가용성 확인
    if not check_xpu_availability():
        return
    
    device = DEFAULT_DEVICE
    
    # RTMW 모델 찾기 및 다운로드
    model_path, model_description = find_and_download_rtmw_model()
    
    if model_path is None:
        print("❌ RTMW 모델을 찾을 수 없습니다.")
        return
    
    print(f"📏 RTMW-x 입력 크기: {RTMW_INPUT_SIZE}")
    
    try:
        # 검출기, 포즈 추정기, 시각화기 초기화
        print("모델 로딩 중...")
        person_detector = SimplePersonDetector(device=device)
        rtmw_estimator = RTMWXEstimator(model_path, device=device)
        visualizer = RTMWVisualizer()
        print(f"✅ RTMW-x 모델 로딩 완료: {model_description}")
        
        # 웹캠 초기화
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ 웹캠을 열 수 없습니다.")
            return
        
        print("RTMW-x 웹캠 추론 시작 (ESC 키로 종료)")
        print(f"사용 모델: {model_description}")
        print(f"입력 해상도: {RTMW_INPUT_SIZE}")
        
        frame_count = 0
        fps_counter = 0
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            fps_counter += 1
            
            # 1. 인체 검출
            person_boxes = person_detector.detect_persons(frame, conf_thresh=DEFAULT_CONF_THRESH)
            
            # 2. 각 검출된 사람에 대해 RTMW-x WholeBody 포즈 추정 및 시각화
            for i, bbox in enumerate(person_boxes):
                try:
                    keypoints = rtmw_estimator.estimate_pose(frame, bbox)
                    
                    # 시각화 (키포인트 + 스켈레톤 + 바운딩박스)
                    frame = visualizer.visualize_pose(frame, keypoints, bbox, person_id=i)
                    
                except Exception as e:
                    print(f"❌ 포즈 추정 실패: {e}")
                    print("프로그램을 종료합니다.")
                    cap.release()
                    cv2.destroyAllWindows()
                    return
            
            # FPS 계산
            if fps_counter % FPS_UPDATE_INTERVAL == 0:
                elapsed = time.time() - start_time
                fps = fps_counter / elapsed if elapsed > 0 else 0
                fps_counter = 0
                start_time = time.time()
            else:
                fps = 0
            
            # 정보 표시
            frame = visualizer.draw_info(frame, frame_count, len(person_boxes), fps, "RTMW-x")
            
            # 결과 표시
            cv2.imshow(f'RTMW-x WholeBody ({RTMW_INPUT_SIZE[0]}x{RTMW_INPUT_SIZE[1]})', frame)
            
            # ESC 키로 종료
            if cv2.waitKey(1) & 0xFF == 27:
                break
        
        # 정리
        cap.release()
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        traceback.print_exc()


if __name__ == '__main__':
    main()