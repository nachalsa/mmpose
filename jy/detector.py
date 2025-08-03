#!/usr/bin/env python3
"""
Person Detection Module
YOLOv11 기반 사람 검출기
"""

import torch
import cv2
import numpy as np
import os
import shutil
from pathlib import Path
from typing import List, Tuple
from ultralytics import YOLO

from config import MODELS_DIR, YOLO_MODEL_FILENAME


class SimplePersonDetector:
    """YOLOv11 Person 전용 검출기"""
    
    def __init__(self, device='xpu:0'):
        self.device = device
        
        # 모델 경로 설정
        yolo_model_path = os.path.join(MODELS_DIR, YOLO_MODEL_FILENAME)
        
        # models 디렉토리 생성
        os.makedirs(MODELS_DIR, exist_ok=True)
        
        try:
            # 로컬 모델 파일 확인
            if os.path.exists(yolo_model_path):
                print(f"✅ YOLOv11 모델 발견: {yolo_model_path}")
                self.model = YOLO(yolo_model_path)
            else:
                print(f"YOLOv11 모델이 없습니다. 다운로드 중: {YOLO_MODEL_FILENAME}")
                self.model = YOLO('yolo11m.pt')
                
                # 다운로드된 모델을 models 폴더로 복사
                ultralytics_cache = Path.home() / '.ultralytics' / 'models'
                downloaded_model = ultralytics_cache / 'yolo11m.pt'
                
                if downloaded_model.exists():
                    shutil.copy2(str(downloaded_model), yolo_model_path)
                    print(f"✅ YOLOv11 모델 저장: {yolo_model_path}")
                else:
                    print("⚠️ 모델 다운로드는 완료되었지만 복사에 실패했습니다.")
            
            # 모델을 XPU로 이동
            self.model.to(device)
            print("✅ YOLOv11 모델 로드 완료")
            
        except ImportError:
            print("❌ ultralytics 패키지가 필요합니다: pip install ultralytics")
            raise
        except Exception as e:
            print(f"❌ YOLOv11 모델 로드 실패: {e}")
            raise
    
    def detect_persons(self, image: np.ndarray, conf_thresh: float = 0.5) -> List[Tuple[int, int, int, int]]:
        """YOLOv11으로 사람 검출
        
        Args:
            image: 입력 이미지 (BGR 형식)
            conf_thresh: 신뢰도 임계값
            
        Returns:
            List of bounding boxes: [(x1, y1, x2, y2), ...]
        """
        results = self.model(image, classes=[0], conf=conf_thresh, verbose=False)
        
        boxes = []
        if len(results[0].boxes) > 0:
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                
                if conf > conf_thresh:
                    boxes.append((int(x1), int(y1), int(x2), int(y2)))
        
        return boxes