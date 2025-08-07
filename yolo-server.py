#!/usr/bin/env python3
"""
엣지 서버: 웹캠 입력 → YOLO 검출 → 박스 크롭 → JPEG 스트림 전송
"""

import cv2
import numpy as np
import requests
import time
import json
import argparse
from typing import List, Optional, Tuple
from pathlib import Path
import logging
from collections import deque

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    print("⚠️ ultralytics 미설치 - pip install ultralytics")
    YOLO_AVAILABLE = False

# RTMW 전처리 함수들 (streamlined_processor.py에서 가져옴)
def bbox_xyxy2cs(bbox: np.ndarray, padding: float = 1.10) -> Tuple[np.ndarray, np.ndarray]:
    """바운딩박스를 center, scale로 변환"""
    dim = bbox.ndim
    if dim == 1:
        bbox = bbox[None, :]
    
    scale = (bbox[..., 2:] - bbox[..., :2]) * padding
    center = (bbox[..., 2:] + bbox[..., :2]) * 0.5
    
    if dim == 1:
        center = center[0]
        scale = scale[0]
    
    return center, scale

def _rotate_point(pt: np.ndarray, angle_rad: float) -> np.ndarray:
    """점을 회전"""
    cos_val = np.cos(angle_rad)
    sin_val = np.sin(angle_rad)
    return np.array([pt[0] * cos_val - pt[1] * sin_val,
                     pt[0] * sin_val + pt[1] * cos_val])

def _get_3rd_point(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """세 번째 점을 계산 (직교점)"""
    direction = a - b
    return b + np.array([-direction[1], direction[0]])

def get_warp_matrix(center: np.ndarray, scale: np.ndarray, rot: float, 
                   output_size: Tuple[int, int]) -> np.ndarray:
    """아핀 변환 매트릭스 계산"""
    src_w, src_h = scale[:2]
    dst_w, dst_h = output_size[:2]
    
    rot_rad = np.deg2rad(rot)
    src_dir = _rotate_point(np.array([src_w * -0.5, 0.]), rot_rad)
    dst_dir = np.array([dst_w * -0.5, 0.])
    
    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center
    src[1, :] = center + src_dir
    
    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir
    
    src[2, :] = _get_3rd_point(src[0, :], src[1, :])
    dst[2, :] = _get_3rd_point(dst[0, :], dst[1, :])
    
    warp_mat = cv2.getAffineTransform(src, dst)
    return warp_mat

def fix_aspect_ratio(bbox_scale: np.ndarray, aspect_ratio: float) -> np.ndarray:
    """bbox를 고정 종횡비로 조정"""
    w, h = bbox_scale[0], bbox_scale[1]
    if w > h * aspect_ratio:
        new_h = w / aspect_ratio
        bbox_scale = np.array([w, new_h])
    else:
        new_w = h * aspect_ratio
        bbox_scale = np.array([new_w, h])
    return bbox_scale

class EdgeYOLODetector:
    """엣지 디바이스용 YOLO 검출기"""
    
    def __init__(self, 
                 yolo_model: str = "yolo11l.pt",
                 conf_thresh: float = 0.4,
                 iou_thresh: float = 0.6,
                 max_det: int = 50,
                 img_size: int = 832):
        
        if not YOLO_AVAILABLE:
            raise ImportError("ultralytics가 필요합니다: pip install ultralytics")
        
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.max_det = max_det
        self.img_size = img_size
        
        # YOLO 모델 로드
        print(f"🔧 YOLO11L 모델 로딩 중...")
        self.model = YOLO(yolo_model)
        
        # GPU 사용 가능하면 GPU로
        if hasattr(self.model.model, 'to'):
            try:
                import torch
                if torch.cuda.is_available():
                    self.model.to('cuda')
                    print(f"✅ YOLO11L CUDA 모드 활성화")
                else:
                    print(f"✅ YOLO11L CPU 모드")
            except:
                print(f"✅ YOLO11L CPU 모드")
        
        print(f"✅ YOLO11L 모델 로딩 완료")
    
    def detect_persons(self, image: np.ndarray) -> List[List[float]]:
        """사람 검출"""
        try:
            results = self.model(
                image,
                conf=self.conf_thresh,
                iou=self.iou_thresh,
                max_det=self.max_det,
                classes=[0],  # 사람 클래스만
                verbose=False,
                imgsz=self.img_size
            )
            
            person_boxes = []
            for result in results:
                boxes = result.boxes
                if boxes is not None and len(boxes) > 0:
                    person_coords = boxes.xyxy
                    person_confs = boxes.conf
                    
                    # 신뢰도 재필터링
                    conf_mask = person_confs >= self.conf_thresh
                    if conf_mask.any():
                        filtered_boxes = person_coords[conf_mask]
                        filtered_confs = person_confs[conf_mask]
                        
                        # numpy 변환
                        if hasattr(filtered_boxes, 'cpu'):
                            filtered_boxes = filtered_boxes.cpu().numpy()
                            filtered_confs = filtered_confs.cpu().numpy()
                        
                        # 신뢰도순 정렬
                        sorted_indices = np.argsort(filtered_confs)[::-1]
                        sorted_boxes = filtered_boxes[sorted_indices]
                        
                        person_boxes.extend(sorted_boxes.tolist())
            
            return person_boxes
            
        except Exception as e:
            print(f"❌ YOLO 검출 실패: {e}")
            return []
    
    def crop_person_image_rtmw(self, image: np.ndarray, bbox: List[float]) -> Optional[np.ndarray]:
        """RTMW 방식으로 사람 이미지 크롭"""
        try:
            # RTMW 설정: width=288, height=384
            input_width, input_height = 288, 384
            
            # 1. bbox를 center, scale로 변환
            bbox_array = np.array(bbox, dtype=np.float32)
            center, scale = bbox_xyxy2cs(bbox_array)
            
            # 2. aspect ratio 고정
            aspect_ratio = input_width / input_height  # 0.75
            scale = fix_aspect_ratio(scale, aspect_ratio)
            
            # 3. 아핀 변환 매트릭스 계산
            warp_mat = get_warp_matrix(
                center=center,
                scale=scale,
                rot=0.0,
                output_size=(input_width, input_height)
            )
            
            # 4. 아핀 변환 적용
            cropped_image = cv2.warpAffine(
                image, 
                warp_mat, 
                (input_width, input_height), 
                flags=cv2.INTER_LINEAR
            )
            
            return cropped_image
                
        except Exception as e:
            print(f"⚠️ 크롭 실패: {e}")
            return None

class EdgeServer:
    """엣지 서버 - 웹캠 캡처 + YOLO 검출 + 크롭 + 전송"""
    
    def __init__(self, 
                 pose_server_url: str,
                 camera_id: int = 0,
                 window_size: Tuple[int, int] = (1280, 720),
                 jpeg_quality: int = 85,
                 max_fps: int = 30):
        
        self.pose_server_url = pose_server_url.rstrip('/')
        self.camera_id = camera_id
        self.window_size = window_size
        self.jpeg_quality = jpeg_quality
        self.max_fps = max_fps
        
        # YOLO 검출기 초기화
        self.detector = EdgeYOLODetector()
        
        # 웹캠 초기화
        self.cap = None
        self.init_camera()
        
        # 성능 측정
        self.fps_history = deque(maxlen=30)
        self.frame_count = 0
        
        print(f"✅ 엣지 서버 초기화 완료")
        print(f"   - 포즈 서버: {self.pose_server_url}")
        print(f"   - 카메라: {camera_id}")
        print(f"   - 해상도: {window_size}")
        print(f"   - JPEG 품질: {jpeg_quality}%")
    
    def init_camera(self):
        """웹캠 초기화"""
        print(f"📹 웹캠 연결 중... (카메라 ID: {self.camera_id})")
        
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"❌ 웹캠 열기 실패 (카메라 ID: {self.camera_id})")
        
        # 웹캠 설정
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.window_size[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.window_size[1])
        self.cap.set(cv2.CAP_PROP_FPS, self.max_fps)
        
        # 실제 웹캠 해상도 확인
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        print(f"✅ 웹캠 연결 성공: {actual_width}x{actual_height}, {actual_fps:.1f}fps")
    
    def send_crop_to_pose_server(self, crop_image: np.ndarray, bbox: List[float], frame_id: int) -> Optional[dict]:
        """크롭된 이미지를 포즈 서버로 전송"""
        try:
            # JPEG 인코딩
            ret, jpeg_data = cv2.imencode('.jpg', crop_image, 
                                        [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
            if not ret:
                return None
            
            # 요청 데이터 준비
            files = {'image': ('crop.jpg', jpeg_data.tobytes(), 'image/jpeg')}
            data = {
                'frame_id': frame_id,
                'bbox': json.dumps(bbox),
                'timestamp': time.time()
            }
            
            # POST 요청
            response = requests.post(
                f"{self.pose_server_url}/estimate_pose",
                files=files,
                data=data,
                timeout=5.0
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"⚠️ 포즈 서버 응답 오류: {response.status_code}")
                return None
                
        except requests.RequestException as e:
            print(f"⚠️ 포즈 서버 통신 실패: {e}")
            return None
        except Exception as e:
            print(f"⚠️ 이미지 전송 실패: {e}")
            return None
    
    def visualize_results(self, image: np.ndarray, person_boxes: List[List[float]], 
                         pose_results: List[dict]) -> np.ndarray:
        """결과 시각화"""
        vis_image = image.copy()
        
        # 검출된 바운딩박스 그리기
        for i, bbox in enumerate(person_boxes):
            x1, y1, x2, y2 = map(int, bbox)
            color = (0, 255, 0) if i == 0 else (255, 0, 255)
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(vis_image, f"Person {i+1}", (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # 포즈 결과가 있으면 키포인트 그리기 (원본 좌표로 변환 필요)
        for i, pose_result in enumerate(pose_results):
            if pose_result and 'keypoints' in pose_result:
                keypoints = np.array(pose_result['keypoints'])
                scores = np.array(pose_result['scores'])
                
                # 크롭 이미지 좌표를 원본 이미지 좌표로 변환
                if i < len(person_boxes):
                    bbox = person_boxes[i]
                    # 간단한 스케일링 (실제로는 아핀 변환 역변환 필요)
                    bbox_w = bbox[2] - bbox[0]
                    bbox_h = bbox[3] - bbox[1]
                    
                    for j, (kpt, score) in enumerate(zip(keypoints, scores)):
                        if score > 0.3:
                            # 크롭 좌표(288x384)를 원본 바운딩박스 좌표로 변환
                            x = int(bbox[0] + (kpt[0] / 288.0) * bbox_w)
                            y = int(bbox[1] + (kpt[1] / 384.0) * bbox_h)
                            
                            if 0 <= x < vis_image.shape[1] and 0 <= y < vis_image.shape[0]:
                                if score > 0.8:
                                    kpt_color = (0, 255, 0)    # 높은 신뢰도: 초록
                                elif score > 0.6:
                                    kpt_color = (0, 255, 255)  # 중간 신뢰도: 노랑
                                else:
                                    kpt_color = (0, 0, 255)    # 낮은 신뢰도: 빨강
                                
                                cv2.circle(vis_image, (x, y), 3, kpt_color, -1)
        
        # 성능 정보 표시
        if self.fps_history:
            avg_fps = np.mean(self.fps_history)
            cv2.putText(vis_image, f"Edge FPS: {avg_fps:.1f}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.putText(vis_image, f"YOLO11L Edge Server", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(vis_image, f"Pose Server: {self.pose_server_url.split('://')[-1]}", 
                   (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(vis_image, f"Detections: {len(person_boxes)}", 
                   (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        return vis_image
    
    def run(self):
        """메인 실행 루프"""
        print(f"\n🚀 엣지 서버 실행 시작")
        print(f"🎮 조작법:")
        print(f"   - ESC: 종료")
        print(f"   - S: 스크린샷")
        print(f"   - SPACE: 일시정지/재생")
        
        paused = False
        screenshot_count = 0
        
        try:
            while True:
                if not paused:
                    ret, frame = self.cap.read()
                    if not ret:
                        print("❌ 프레임 읽기 실패")
                        break
                    
                    start_time = time.time()
                    
                    # 1. YOLO 검출
                    person_boxes = self.detector.detect_persons(frame)
                    
                    # 2. 각 사람에 대해 크롭 + 포즈 서버로 전송
                    pose_results = []
                    for i, bbox in enumerate(person_boxes):
                        crop_image = self.detector.crop_person_image_rtmw(frame, bbox)
                        if crop_image is not None:
                            # 포즈 서버로 전송
                            pose_result = self.send_crop_to_pose_server(
                                crop_image, bbox, self.frame_count * 1000 + i
                            )
                            pose_results.append(pose_result)
                        else:
                            pose_results.append(None)
                    
                    # 3. FPS 계산
                    process_time = time.time() - start_time
                    fps = 1.0 / process_time if process_time > 0 else 0
                    self.fps_history.append(fps)
                    
                    # 4. 시각화
                    vis_frame = self.visualize_results(frame, person_boxes, pose_results)
                    self.frame_count += 1
                
                # 화면 표시
                cv2.imshow('YOLO Edge Server', vis_frame)
                
                # 키 입력 처리
                key = cv2.waitKey(1) & 0xFF
                
                if key == 27:  # ESC
                    break
                elif key == ord('s') or key == ord('S'):  # 스크린샷
                    screenshot_name = f"edge_screenshot_{screenshot_count:04d}.jpg"
                    cv2.imwrite(screenshot_name, vis_frame)
                    print(f"📸 스크린샷 저장: {screenshot_name}")
                    screenshot_count += 1
                elif key == ord(' '):  # 일시정지/재생
                    paused = not paused
                    print(f"⏸️ {'일시정지' if paused else '재생'}")
                
                # 주기적 통계 출력
                if self.frame_count % 60 == 0 and self.frame_count > 0:
                    avg_fps = np.mean(self.fps_history) if self.fps_history else 0
                    print(f"📊 프레임 {self.frame_count}: {avg_fps:.1f}fps, "
                          f"{len(person_boxes)}명 검출")
                    
        except KeyboardInterrupt:
            print("\n⏹️ 사용자가 엣지 서버를 중단했습니다.")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """리소스 정리"""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        
        if self.fps_history:
            avg_fps = np.mean(self.fps_history)
            print(f"\n📊 엣지 서버 완료:")
            print(f"   - 처리된 프레임: {self.frame_count}")
            print(f"   - 평균 FPS: {avg_fps:.1f}")

def main():
    parser = argparse.ArgumentParser(description="YOLO Edge Server")
    parser.add_argument("--pose-server", type=str, default="http://192.168.1.100:5000",
                       help="포즈 서버 URL (기본값: http://192.168.1.100:5000)")
    parser.add_argument("--camera", type=int, default=0,
                       help="카메라 ID (기본값: 0)")
    parser.add_argument("--width", type=int, default=1280,
                       help="웹캠 가로 해상도 (기본값: 1280)")
    parser.add_argument("--height", type=int, default=720,
                       help="웹캠 세로 해상도 (기본값: 720)")
    parser.add_argument("--jpeg-quality", type=int, default=85,
                       help="JPEG 전송 품질 (기본값: 85)")
    parser.add_argument("--fps", type=int, default=30,
                       help="최대 FPS (기본값: 30)")
    
    args = parser.parse_args()
    
    try:
        edge_server = EdgeServer(
            pose_server_url=args.pose_server,
            camera_id=args.camera,
            window_size=(args.width, args.height),
            jpeg_quality=args.jpeg_quality,
            max_fps=args.fps
        )
        
        edge_server.run()
        
    except Exception as e:
        print(f"❌ 엣지 서버 실행 실패: {e}")

if __name__ == "__main__":
    main()