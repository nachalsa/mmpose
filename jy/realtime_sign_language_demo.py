#!/usr/bin/env python3
"""
실시간 수화 인식 트래킹 데모
웹캠을 사용한 트래킹 기반 YOLO11L + RTMW 시스템
"""

import cv2
import numpy as np
import time
import argparse
from tracking_yolo11l_hybrid import TrackingYOLO11LHybridInferencer

def parse_arguments():
    """명령행 인수 파싱"""
    parser = argparse.ArgumentParser(description="실시간 수화 인식 트래킹 데모")
    parser.add_argument("--camera", type=int, default=0, help="카메라 디바이스 번호 (기본: 0)")
    parser.add_argument("--width", type=int, default=1280, help="캠 해상도 너비 (기본: 1280)")
    parser.add_argument("--height", type=int, default=720, help="캠 해상도 높이 (기본: 720)")
    parser.add_argument("--fps", type=int, default=30, help="목표 FPS (기본: 30)")
    parser.add_argument("--redetect-interval", type=int, default=30, help="재검출 간격 (기본: 30프레임)")
    parser.add_argument("--save-video", type=str, help="결과 비디오 저장 경로")
    parser.add_argument("--max-time", type=int, help="최대 실행 시간 (초)")
    return parser.parse_args()

class SignLanguageTrackingDemo:
    """수화 인식 트래킹 데모 클래스"""
    
    def __init__(self, args):
        self.args = args
        
        # 모델 경로
        rtmw_config = "../configs/wholebody_2d_keypoint/rtmpose/cocktail14/rtmw-x_8xb320-270e_cocktail14-384x288.py"
        rtmw_checkpoint = "../models/rtmw-x_simcc-cocktail14_pt-ucoco_270e-384x288-f840f204_20231122.pth"
        
        print("🚀 수화 인식 트래킹 시스템 초기화 중...")
        
        # 트래킹 하이브리드 추론기 생성
        self.inferencer = TrackingYOLO11LHybridInferencer(
            rtmw_config=rtmw_config,
            rtmw_checkpoint=rtmw_checkpoint,
            detection_device="auto",
            pose_device="auto",
            redetection_interval=args.redetect_interval,
            tracking_stability=0.8
        )
        
        # 성능 통계
        self.fps_history = []
        self.detection_count_history = []
        self.processing_times = []
        
        # UI 설정
        self.show_keypoints = True
        self.show_hands_only = False
        self.show_stats = True
        self.show_bbox = True
        
        print("✅ 시스템 초기화 완료!")
    
    def setup_camera(self):
        """카메라 설정"""
        print(f"📷 카메라 {self.args.camera} 설정 중...")
        
        self.cap = cv2.VideoCapture(self.args.camera)
        if not self.cap.isOpened():
            raise RuntimeError(f"카메라 {self.args.camera} 열기 실패")
        
        # 해상도 설정
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.args.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.args.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.args.fps)
        
        # 실제 설정값 확인
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        print(f"✅ 카메라 설정: {actual_width}x{actual_height}, {actual_fps:.1f}fps")
        
        return actual_width, actual_height
    
    def setup_video_writer(self, width: int, height: int):
        """비디오 저장 설정"""
        if self.args.save_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(
                self.args.save_video, fourcc, self.args.fps, (width, height)
            )
            print(f"📹 비디오 저장: {self.args.save_video}")
        else:
            self.video_writer = None
    
    def draw_enhanced_visualization(self, image: np.ndarray, results) -> np.ndarray:
        """향상된 시각화"""
        vis_image = image.copy()
        
        # 트래커별 색상
        colors = [
            (0, 255, 0),    # 초록
            (255, 0, 0),    # 파랑  
            (0, 0, 255),    # 빨강
            (255, 255, 0),  # 시안
            (255, 0, 255),  # 마젠타
        ]
        
        for keypoints, scores, bbox, track_id in results:
            color = colors[track_id % len(colors)]
            
            # 바운딩박스
            if self.show_bbox:
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
                cv2.putText(vis_image, f"Person {track_id}", (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # 키포인트 그리기
            if self.show_keypoints:
                self._draw_keypoints_enhanced(vis_image, keypoints, scores, color)
        
        # 성능 정보
        if self.show_stats:
            self._draw_performance_stats(vis_image)
        
        # 컨트롤 정보
        self._draw_controls(vis_image)
        
        return vis_image
    
    def _draw_keypoints_enhanced(self, image: np.ndarray, keypoints: np.ndarray, scores: np.ndarray, color: tuple):
        """향상된 키포인트 그리기"""
        # 손 키포인트 인덱스
        left_hand_indices = list(range(91, 112))   # 왼손 (21개)
        right_hand_indices = list(range(112, 133)) # 오른손 (21개)
        face_indices = list(range(17, 91))         # 얼굴 (70개)
        body_indices = list(range(0, 17))          # 몸통 (17개)
        
        if self.show_hands_only:
            # 손만 표시
            indices_to_draw = left_hand_indices + right_hand_indices
            point_sizes = [6] * len(indices_to_draw)
        else:
            # 전체 표시 (손 강조)
            indices_to_draw = list(range(len(keypoints)))
            point_sizes = []
            for i in range(len(keypoints)):
                if i in left_hand_indices or i in right_hand_indices:
                    point_sizes.append(5)  # 손: 큰 점
                elif i in face_indices:
                    point_sizes.append(2)  # 얼굴: 작은 점
                else:
                    point_sizes.append(3)  # 몸통: 중간 점
        
        # 키포인트 그리기
        for idx, size in zip(indices_to_draw, point_sizes):
            if idx < len(keypoints) and scores[idx] > 0.3:
                x, y = int(keypoints[idx][0]), int(keypoints[idx][1])
                if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                    # 신뢰도에 따른 색상
                    if scores[idx] > 0.8:
                        kpt_color = color  # 고신뢰도: 원래 색상
                    elif scores[idx] > 0.6:
                        kpt_color = tuple(int(c*0.8) for c in color)  # 중신뢰도: 약간 어둡게
                    else:
                        kpt_color = tuple(int(c*0.6) for c in color)  # 저신뢰도: 더 어둡게
                    
                    cv2.circle(image, (x, y), size, kpt_color, -1)
                    
                    # 손 키포인트는 흰 테두리 추가
                    if idx in left_hand_indices or idx in right_hand_indices:
                        cv2.circle(image, (x, y), size+1, (255, 255, 255), 1)
    
    def _draw_performance_stats(self, image: np.ndarray):
        """성능 통계 표시"""
        # 현재 FPS
        if self.fps_history:
            current_fps = self.fps_history[-1]
            avg_fps = np.mean(self.fps_history[-30:])  # 최근 30프레임 평균
            
            cv2.putText(image, f"FPS: {current_fps:.1f} (avg: {avg_fps:.1f})", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 트래킹 정보
        active_trackers = len(self.inferencer.trackers)
        cv2.putText(image, f"Tracking: {active_trackers} persons", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # 프레임 정보
        cv2.putText(image, f"Frame: {self.inferencer.frame_count}", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # 마지막 검출 정보
        frames_since_detection = self.inferencer.frame_count - self.inferencer.last_detection_frame
        next_detection = max(0, self.inferencer.redetection_interval - frames_since_detection)
        cv2.putText(image, f"Next detection in: {next_detection} frames", 
                   (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
    
    def _draw_controls(self, image: np.ndarray):
        """컨트롤 정보 표시"""
        h, w = image.shape[:2]
        controls = [
            "Controls:",
            "  [H] Toggle hands only",
            "  [K] Toggle keypoints", 
            "  [B] Toggle bounding box",
            "  [S] Toggle stats",
            "  [R] Reset tracking",
            "  [ESC] Quit"
        ]
        
        for i, text in enumerate(controls):
            y_pos = h - 20 - (len(controls) - i - 1) * 25
            color = (255, 255, 255) if i == 0 else (200, 200, 200)
            thickness = 2 if i == 0 else 1
            cv2.putText(image, text, (w - 250, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)
    
    def handle_keyboard_input(self, key: int) -> bool:
        """키보드 입력 처리"""
        if key == 27:  # ESC
            return False
        elif key == ord('h') or key == ord('H'):
            self.show_hands_only = not self.show_hands_only
            print(f"손만 표시: {self.show_hands_only}")
        elif key == ord('k') or key == ord('K'):
            self.show_keypoints = not self.show_keypoints
            print(f"키포인트 표시: {self.show_keypoints}")
        elif key == ord('b') or key == ord('B'):
            self.show_bbox = not self.show_bbox
            print(f"바운딩박스 표시: {self.show_bbox}")
        elif key == ord('s') or key == ord('S'):
            self.show_stats = not self.show_stats
            print(f"통계 표시: {self.show_stats}")
        elif key == ord('r') or key == ord('R'):
            self.inferencer.reset_tracking()
            print("트래킹 리셋")
        
        return True
    
    def run(self):
        """메인 실행 루프"""
        try:
            # 카메라 설정
            width, height = self.setup_camera()
            self.setup_video_writer(width, height)
            
            print("\n🎬 실시간 수화 인식 트래킹 시작!")
            print("컨트롤: H(손만), K(키포인트), B(박스), S(통계), R(리셋), ESC(종료)")
            
            start_time = time.time()
            frame_count = 0
            
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("❌ 프레임 읽기 실패")
                    break
                
                # 프레임 처리
                process_start = time.time()
                vis_frame, results = self.inferencer.process_frame(frame)
                process_time = time.time() - process_start
                
                # 향상된 시각화
                final_frame = self.draw_enhanced_visualization(vis_frame, results)
                
                # 성능 통계 업데이트
                fps = 1.0 / process_time if process_time > 0 else 0
                self.fps_history.append(fps)
                self.detection_count_history.append(len(results))
                self.processing_times.append(process_time)
                
                # 히스토리 크기 제한
                if len(self.fps_history) > 300:  # 10초치 (30fps 기준)
                    self.fps_history = self.fps_history[-300:]
                    self.detection_count_history = self.detection_count_history[-300:]
                    self.processing_times = self.processing_times[-300:]
                
                # 화면 표시
                cv2.imshow("Sign Language Tracking Demo", final_frame)
                
                # 비디오 저장
                if self.video_writer:
                    self.video_writer.write(final_frame)
                
                # 키보드 입력 처리
                key = cv2.waitKey(1) & 0xFF
                if not self.handle_keyboard_input(key):
                    break
                
                frame_count += 1
                
                # 최대 시간 체크
                if self.args.max_time:
                    elapsed = time.time() - start_time
                    if elapsed > self.args.max_time:
                        print(f"⏰ 최대 실행 시간 {self.args.max_time}초 도달")
                        break
                
                # 주기적 통계 출력
                if frame_count % 300 == 0:  # 10초마다
                    avg_fps = np.mean(self.fps_history[-300:])
                    avg_detections = np.mean(self.detection_count_history[-300:])
                    print(f"📊 {frame_count}프레임: {avg_fps:.1f}fps, {avg_detections:.1f}명 평균")
        
        except KeyboardInterrupt:
            print("\n⏹️ 사용자가 데모를 중단했습니다.")
        
        except Exception as e:
            print(f"\n❌ 데모 실행 중 오류: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """정리 작업"""
        print("\n🧹 정리 작업 중...")
        
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        
        if self.video_writer:
            self.video_writer.release()
        
        cv2.destroyAllWindows()
        
        # 최종 통계
        if self.fps_history:
            avg_fps = np.mean(self.fps_history)
            avg_processing_time = np.mean(self.processing_times)
            total_frames = len(self.fps_history)
            
            print(f"\n📊 최종 통계:")
            print(f"   - 총 처리 프레임: {total_frames}")
            print(f"   - 평균 FPS: {avg_fps:.1f}")
            print(f"   - 평균 처리 시간: {avg_processing_time*1000:.1f}ms")
            print(f"   - 평균 검출/추적: {np.mean(self.detection_count_history):.1f}명")
            
            if self.args.save_video:
                print(f"   - 저장된 비디오: {self.args.save_video}")

def main():
    """메인 함수"""
    args = parse_arguments()
    demo = SignLanguageTrackingDemo(args)
    demo.run()

if __name__ == "__main__":
    main()
