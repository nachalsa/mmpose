#!/usr/bin/env python3
"""
사람 검출기 모듈
YOLOv11을 사용한 사람 검출 - config 중심 관리
"""

import cv2
import numpy as np
from typing import List, Tuple
import torch
import os
import shutil

try:
    from ultralytics import YOLO
except ImportError:
    print("❌ ultralytics가 설치되지 않았습니다: pip install ultralytics")
    raise

from config import (
    get_yolo_config, 
    check_model_exists, 
    ensure_models_dir,
    MODELS_DIR
)

class SimplePersonDetector:
    """YOLOv11 기반 사람 검출기 - config 중심 관리"""
    
    def __init__(self, device: str = 'cpu'):
        self.device = device
        
        # config에서 YOLO 모델 정보 가져오기
        self.model_config = get_yolo_config()
        self.model_filename = self.model_config["filename"]
        self.model_path = self.model_config["path"]
        
        print(f"🔧 YOLO 검출기 초기화:")
        print(f"   - 모델: {self.model_config['description']}")
        print(f"   - 파일명: {self.model_filename}")
        print(f"   - 대상 경로: {self.model_path}")
        
        # models 디렉토리 확인
        ensure_models_dir()
        
        # 모델 로드
        self.model = self._load_model()
        
        # COCO 클래스 (0: person)
        self.person_class_id = 0
        
        print(f"✅ 사람 검출기 초기화 완료")
    
    def _load_model(self) -> YOLO:
        """config 기반 YOLO 모델 로드"""
        try:
            # 모델 파일이 models 디렉토리에 있는지 확인
            if check_model_exists('yolo'):
                print(f"✅ 기존 YOLO 모델 발견: {self.model_path}")
                return YOLO(self.model_path)
            
            # 모델이 없으면 다운로드 후 models 디렉토리로 이동
            print(f"📥 YOLO 모델 다운로드 및 설치 중: {self.model_filename}")
            
            # 1. 모델 다운로드 (ultralytics가 자동으로 캐시에 다운로드)
            print(f"🔄 모델 다운로드 중...")
            temp_model = YOLO(self.model_filename)
            
            # 2. 다운로드된 파일을 models 디렉토리로 복사
            ultralytics_cache = os.path.expanduser(f"~/.ultralytics/weights/{self.model_filename}")
            
            if os.path.exists(ultralytics_cache):
                print(f"📁 캐시에서 모델 발견: {ultralytics_cache}")
                print(f"📂 models 디렉토리로 복사 중: {self.model_path}")
                
                # 복사 실행
                shutil.copy2(ultralytics_cache, self.model_path)
                print(f"✅ 모델 복사 완료: {self.model_path}")
                
                # 복사된 파일로 모델 재로드
                model = YOLO(self.model_path)
                print(f"✅ models 디렉토리의 모델로 로드 완료")
                
            else:
                print(f"⚠️ 캐시에서 모델을 찾을 수 없음: {ultralytics_cache}")
                print(f"🔄 임시 모델로 계속 진행...")
                model = temp_model
            
            # 최종 확인
            print(f"📊 최종 모델 상태:")
            print(f"   - 사용 중인 모델: {model.model_name if hasattr(model, 'model_name') else 'Unknown'}")
            print(f"   - models 디렉토리 파일 존재: {os.path.exists(self.model_path)}")
            
            return model
            
        except Exception as e:
            print(f"❌ YOLO 모델 로드 실패: {e}")
            print(f"🔄 기본 방식으로 재시도...")
            try:
                return YOLO(self.model_filename)
            except Exception as e2:
                print(f"❌ 기본 로드도 실패: {e2}")
                raise
    
    def detect_persons(self, image: np.ndarray, conf_thresh: float = 0.5) -> List[List[float]]:
        """
        이미지에서 사람 검출
        
        Args:
            image: 입력 이미지 (BGR)
            conf_thresh: 검출 신뢰도 임계값
            
        Returns:
            List of bounding boxes [x1, y1, x2, y2]
        """
        try:
            # YOLO 추론
            results = self.model(image, verbose=False)
            
            person_boxes = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # 사람 클래스만 필터링
                        if int(box.cls[0]) == self.person_class_id and float(box.conf[0]) >= conf_thresh:
                            # 좌표 추출 [x1, y1, x2, y2]
                            coords = box.xyxy[0].cpu().numpy()
                            person_boxes.append(coords.tolist())
            
            return person_boxes
            
        except Exception as e:
            print(f"❌ 사람 검출 실패: {e}")
            return []
    
    def visualize_detections(self, image: np.ndarray, boxes: List[List[float]]) -> np.ndarray:
        """검출 결과 시각화"""
        vis_image = image.copy()
        
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            
            # 바운딩 박스 그리기
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 레이블 추가
            label = f"Person {i+1}"
            cv2.putText(vis_image, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return vis_image


def test_detector():
    """검출기 테스트"""
    print("🧪 사람 검출기 테스트")
    
    # 검출기 초기화
    detector = SimplePersonDetector()
    
    # 웹캠으로 테스트
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ 웹캠을 열 수 없습니다")
        return
    
    print("웹캠 테스트 시작 (ESC로 종료)")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 사람 검출
            boxes = detector.detect_persons(frame, conf_thresh=0.5)
            
            # 시각화
            vis_frame = detector.visualize_detections(frame, boxes)
            
            # 정보 표시
            info_text = f"Detected: {len(boxes)} persons"
            cv2.putText(vis_frame, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow('Person Detection Test', vis_frame)
            
            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                break
                
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    test_detector()