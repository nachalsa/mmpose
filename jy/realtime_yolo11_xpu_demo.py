#!/usr/bin/env python3
"""
YOLO11 + RTMW XPU 하이브리드 실시간 웹캠 데모
실시간 사람 검출 + 포즈 추정 (XPU 최적화)
"""

import cv2
import time
import numpy as np
from typing import Optional
from yolo11_xpu_hybrid_inferencer import YOLO11XPUHybridInferencer

class RealTimeHybridDemo:
    """실시간 하이브리드 포즈 추정 데모"""
    
    def __init__(self, 
                 rtmw_config: str,
                 rtmw_checkpoint: str, 
                 yolo_model: str = "yolo11m.pt",
                 webcam_id: int = 0,
                 target_fps: int = 30):
        """
        Args:
            rtmw_config: RTMW 설정 파일 경로
            rtmw_checkpoint: RTMW 체크포인트 경로
            yolo_model: YOLO11 모델 파일명
            webcam_id: 웹캠 ID
            target_fps: 목표 FPS
        """
        self.webcam_id = webcam_id
        self.target_fps = target_fps
        self.frame_time = 1.0 / target_fps
        
        print(f"🚀 실시간 하이브리드 포즈 추정 데모 초기화")
        print(f"   - 웹캠 ID: {webcam_id}")
        print(f"   - 목표 FPS: {target_fps}")
        
        # 하이브리드 추론기 초기화
        self.inferencer = YOLO11XPUHybridInferencer(
            rtmw_config=rtmw_config,
            rtmw_checkpoint=rtmw_checkpoint,
            yolo_model=yolo_model,
            detection_device="auto",
            pose_device="auto"
        )
        
        # 성능 통계
        self.fps_history = []
        self.max_history = 30
        
        # 시각화 설정
        self.show_bbox = True
        self.show_keypoints = True
        self.min_confidence = 0.3
        
    def init_webcam(self) -> Optional[cv2.VideoCapture]:
        """웹캠 초기화"""
        print(f"📷 웹캠 초기화 중... (ID: {self.webcam_id})")
        
        cap = cv2.VideoCapture(self.webcam_id)
        
        if not cap.isOpened():
            print(f"❌ 웹캠 열기 실패 (ID: {self.webcam_id})")
            return None
        
        # 웹캠 설정 최적화
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, self.target_fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 지연 최소화
        
        # 실제 설정값 확인
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"✅ 웹캠 초기화 완료:")
        print(f"   - 해상도: {actual_width}x{actual_height}")
        print(f"   - FPS: {actual_fps}")
        
        return cap
    
    def process_frame_optimized(self, frame: np.ndarray) -> np.ndarray:
        """최적화된 프레임 처리"""
        # 프레임 크기 조정으로 성능 향상 (선택적)
        original_shape = frame.shape[:2]
        
        # 큰 이미지는 크기 조정 (1080p 이상)
        if original_shape[0] > 1080 or original_shape[1] > 1920:
            scale_factor = min(1080 / original_shape[0], 1920 / original_shape[1])
            new_width = int(original_shape[1] * scale_factor)
            new_height = int(original_shape[0] * scale_factor)
            
            frame_resized = cv2.resize(frame, (new_width, new_height))
            
            # 추론
            vis_frame, results = self.inferencer.process_frame(frame_resized, conf_thresh=0.6)
            
            # 결과를 원본 크기로 복원
            vis_frame = cv2.resize(vis_frame, (original_shape[1], original_shape[0]))
        else:
            # 원본 크기로 처리
            vis_frame, results = self.inferencer.process_frame(frame, conf_thresh=0.6)
        
        return vis_frame
    
    def add_performance_overlay(self, frame: np.ndarray, fps: float) -> np.ndarray:
        """성능 정보 오버레이 추가"""
        overlay = frame.copy()
        
        # FPS 히스토리 업데이트
        self.fps_history.append(fps)
        if len(self.fps_history) > self.max_history:
            self.fps_history.pop(0)
        
        avg_fps = np.mean(self.fps_history)
        
        # 성능 정보 텍스트
        info_texts = [
            f"FPS: {fps:.1f} (평균: {avg_fps:.1f})",
            f"검출: {self.inferencer.detection_device.upper()}",
            f"포즈: {self.inferencer.pose_device.upper()}",
            f"모델: YOLO11 + RTMW",
        ]
        
        # 반투명 배경
        overlay_height = len(info_texts) * 25 + 20
        cv2.rectangle(overlay, (10, 10), (320, overlay_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # 텍스트 추가
        for i, text in enumerate(info_texts):
            y_pos = 35 + i * 25
            color = (0, 255, 0) if fps >= self.target_fps * 0.8 else (0, 255, 255)
            cv2.putText(frame, text, (15, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # 컨트롤 도움말
        help_texts = [
            "컨트롤:",
            "q: 종료",
            "b: 바운딩박스 토글", 
            "k: 키포인트 토글",
            "r: 통계 리셋"
        ]
        
        help_start_y = frame.shape[0] - len(help_texts) * 20 - 10
        for i, text in enumerate(help_texts):
            y_pos = help_start_y + i * 20
            cv2.putText(frame, text, (15, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def run(self):
        """실시간 데모 실행"""
        # 웹캠 초기화
        cap = self.init_webcam()
        if cap is None:
            return
        
        print(f"\n🎥 실시간 데모 시작!")
        print(f"   컨트롤: q=종료, b=박스토글, k=키포인트토글, r=리셋")
        
        try:
            frame_count = 0
            
            while True:
                start_time = time.time()
                
                # 프레임 읽기
                ret, frame = cap.read()
                if not ret:
                    print("❌ 프레임 읽기 실패")
                    break
                
                # 프레임 처리
                try:
                    processed_frame = self.process_frame_optimized(frame)
                except Exception as e:
                    print(f"⚠️ 프레임 처리 오류: {e}")
                    processed_frame = frame
                
                # FPS 계산
                process_time = time.time() - start_time
                fps = 1.0 / process_time if process_time > 0 else 0
                
                # 성능 정보 오버레이
                display_frame = self.add_performance_overlay(processed_frame, fps)
                
                # 화면 표시
                cv2.imshow('YOLO11 + RTMW XPU 하이브리드 데모', display_frame)
                
                # 키보드 입력 처리
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print("🛑 사용자 종료 요청")
                    break
                elif key == ord('b'):
                    self.show_bbox = not self.show_bbox
                    print(f"📦 바운딩박스: {'ON' if self.show_bbox else 'OFF'}")
                elif key == ord('k'):
                    self.show_keypoints = not self.show_keypoints
                    print(f"🎯 키포인트: {'ON' if self.show_keypoints else 'OFF'}")
                elif key == ord('r'):
                    self.fps_history.clear()
                    self.inferencer.inference_times = {'detection': [], 'pose': [], 'total': []}
                    print("📊 통계 리셋")
                
                # FPS 제한
                elapsed_time = time.time() - start_time
                sleep_time = max(0, self.frame_time - elapsed_time)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
                frame_count += 1
                
                # 5초마다 성능 통계 출력
                if frame_count % (self.target_fps * 5) == 0:
                    if self.fps_history:
                        avg_fps = np.mean(self.fps_history[-self.target_fps*5:])
                        print(f"📊 지난 5초 평균 FPS: {avg_fps:.1f}")
        
        except KeyboardInterrupt:
            print("\n🛑 Ctrl+C로 종료")
        
        finally:
            # 정리
            cap.release()
            cv2.destroyAllWindows()
            
            # 최종 통계
            if self.fps_history:
                print(f"\n📊 최종 성능 통계:")
                print(f"   - 평균 FPS: {np.mean(self.fps_history):.1f}")
                print(f"   - 최대 FPS: {np.max(self.fps_history):.1f}")
                print(f"   - 최소 FPS: {np.min(self.fps_history):.1f}")
            
            print("✅ 데모 종료")

def main():
    """메인 함수"""
    # 모델 경로 설정
    rtmw_config = "../configs/wholebody_2d_keypoint/rtmpose/cocktail14/rtmw-x_8xb320-270e_cocktail14-384x288.py"
    rtmw_checkpoint = "../models/rtmw-x_simcc-cocktail14_pt-ucoco_270e-384x288-f840f204_20231122.pth"
    
    try:
        # 실시간 데모 생성
        demo = RealTimeHybridDemo(
            rtmw_config=rtmw_config,
            rtmw_checkpoint=rtmw_checkpoint,
            yolo_model="yolo11m.pt",
            webcam_id=0,  # 기본 웹캠
            target_fps=30
        )
        
        # 데모 실행
        demo.run()
        
    except Exception as e:
        print(f"❌ 데모 실행 실패: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
