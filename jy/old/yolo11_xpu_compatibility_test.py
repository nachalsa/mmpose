#!/usr/bin/env python3
"""
YOLO11 XPU 호환성 및 최적화 테스트
"""

import torch
import time
import numpy as np
import cv2
import os

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    print("❌ ultralytics 미설치: pip install ultralytics")
    YOLO_AVAILABLE = False

def check_device_compatibility():
    """디바이스 호환성 확인"""
    print("=== 디바이스 호환성 확인 ===")
    
    devices = {
        'CPU': True,
        'CUDA': torch.cuda.is_available(),
        'XPU': torch.xpu.is_available() if hasattr(torch, 'xpu') else False
    }
    
    for device, available in devices.items():
        status = "✅ 사용가능" if available else "❌ 사용불가"
        print(f"{device}: {status}")
        
        if device == 'XPU' and available:
            try:
                xpu_count = torch.xpu.device_count()
                print(f"   XPU 디바이스 수: {xpu_count}")
                
                for i in range(xpu_count):
                    props = torch.xpu.get_device_properties(i)
                    print(f"   XPU {i}: {props.name}")
            except Exception as e:
                print(f"   XPU 상세 정보 확인 실패: {e}")
    
    return devices

def test_yolo11_devices():
    """YOLO11의 각 디바이스 지원 테스트"""
    if not YOLO_AVAILABLE:
        return {}
    
    print("\n=== YOLO11 디바이스 지원 테스트 ===")
    
    # 테스트용 더미 이미지
    dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    
    # YOLO11 모델 로드
    model_path = "../models/yolo11m.pt"
    if not os.path.exists(model_path):
        print(f"📥 YOLO11 모델 다운로드: {model_path}")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    try:
        model = YOLO(model_path)
        print(f"✅ YOLO11 모델 로드 완료")
    except Exception as e:
        print(f"❌ YOLO11 모델 로드 실패: {e}")
        return {}
    
    devices_to_test = ['cpu']
    
    # CUDA 테스트
    if torch.cuda.is_available():
        devices_to_test.append('cuda')
    
    # XPU 테스트 시도
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        devices_to_test.append('xpu')
    
    results = {}
    
    for device in devices_to_test:
        print(f"\n🧪 {device.upper()} 테스트...")
        
        try:
            # 모델을 디바이스로 이동 시도
            if device != 'cpu':
                model.to(device)
            
            # 워밍업
            for _ in range(3):
                _ = model(dummy_image, verbose=False)
            
            # 성능 측정
            times = []
            for _ in range(10):
                start = time.time()
                results_pred = model(dummy_image, verbose=False)
                end = time.time()
                times.append(end - start)
            
            avg_time = np.mean(times)
            fps = 1.0 / avg_time
            
            results[device] = {
                'success': True,
                'avg_time': avg_time,
                'fps': fps,
                'detections': len(results_pred[0].boxes) if results_pred[0].boxes is not None else 0
            }
            
            print(f"   ✅ 성공: {avg_time*1000:.1f}ms, {fps:.1f} FPS")
            
        except Exception as e:
            print(f"   ❌ 실패: {e}")
            results[device] = {
                'success': False,
                'error': str(e)
            }
    
    return results

def optimize_yolo11_for_xpu():
    """YOLO11 XPU 최적화 설정"""
    print("\n=== YOLO11 XPU 최적화 ===")
    
    if not YOLO_AVAILABLE:
        return None
    
    optimizations = {
        'half_precision': False,  # FP16 사용 여부
        'dynamic_shapes': False,  # 동적 모양 최적화
        'batch_size': 1,          # 배치 크기
        'max_det': 100,           # 최대 검출 수
        'conf': 0.5,              # 신뢰도 임계값
        'iou': 0.7                # IoU 임계값
    }
    
    # XPU에서 지원되는 최적화 확인
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        print("🔧 XPU 최적화 설정 적용 중...")
        
        try:
            # XPU 특화 설정
            optimizations['half_precision'] = False  # XPU FP16 지원 확인 필요
            optimizations['batch_size'] = 1          # 단일 배치로 시작
            
            print("✅ XPU 최적화 설정 완료")
            
        except Exception as e:
            print(f"⚠️ XPU 최적화 설정 실패: {e}")
    
    return optimizations

def create_optimized_yolo_wrapper():
    """최적화된 YOLO11 래퍼 클래스 생성"""
    
    class OptimizedYOLO11:
        """XPU 최적화된 YOLO11 래퍼"""
        
        def __init__(self, model_path: str, device: str = 'auto'):
            self.model_path = model_path
            self.device = device
            
            # 디바이스 자동 결정
            if device == 'auto':
                if hasattr(torch, 'xpu') and torch.xpu.is_available():
                    self.device = 'xpu'
                elif torch.cuda.is_available():
                    self.device = 'cuda'
                else:
                    self.device = 'cpu'
            
            # 모델 로드
            self.model = YOLO(model_path)
            
            # 디바이스 설정
            if self.device != 'cpu':
                try:
                    self.model.to(self.device)
                    print(f"✅ YOLO11 {self.device.upper()} 모드 활성화")
                except Exception as e:
                    print(f"⚠️ {self.device.upper()} 설정 실패, CPU로 폴백: {e}")
                    self.device = 'cpu'
            
            # 최적화 설정
            self.optimizations = optimize_yolo11_for_xpu()
            
        def predict(self, image, conf=0.5, classes=None):
            """최적화된 예측"""
            try:
                # YOLO11 추론 with 최적화 파라미터
                results = self.model(
                    image,
                    conf=conf,
                    classes=classes,
                    max_det=self.optimizations.get('max_det', 100),
                    verbose=False
                )
                
                return results
                
            except Exception as e:
                print(f"❌ 예측 실패: {e}")
                return None
        
        def detect_persons(self, image, conf=0.5):
            """사람만 검출"""
            results = self.predict(image, conf=conf, classes=[0])  # 클래스 0: person
            
            person_boxes = []
            if results and results[0].boxes is not None:
                boxes = results[0].boxes
                person_coords = boxes.xyxy
                person_confs = boxes.conf
                
                # 신뢰도 필터링
                conf_mask = person_confs >= conf
                if conf_mask.any():
                    filtered_boxes = person_coords[conf_mask]
                    
                    # numpy 변환
                    if isinstance(filtered_boxes, torch.Tensor):
                        filtered_boxes = filtered_boxes.cpu().numpy()
                    
                    person_boxes = filtered_boxes.tolist()
            
            return person_boxes
    
    return OptimizedYOLO11

def main():
    """메인 테스트"""
    print("🚀 YOLO11 XPU 호환성 및 최적화 테스트")
    
    # 1. 디바이스 호환성 확인
    device_info = check_device_compatibility()
    
    # 2. YOLO11 디바이스 지원 테스트
    yolo_results = test_yolo11_devices()
    
    # 3. 결과 요약
    print("\n=== 테스트 결과 요약 ===")
    
    print("📊 YOLO11 성능 비교:")
    best_device = None
    best_fps = 0
    
    for device, result in yolo_results.items():
        if result['success']:
            fps = result['fps']
            print(f"   {device.upper()}: {fps:.1f} FPS")
            
            if fps > best_fps:
                best_fps = fps
                best_device = device
        else:
            print(f"   {device.upper()}: ❌ 실패")
    
    if best_device:
        print(f"🏆 최고 성능: {best_device.upper()} ({best_fps:.1f} FPS)")
        
        # 최적화된 래퍼 테스트
        print(f"\n🧪 최적화된 YOLO11 래퍼 테스트...")
        
        OptimizedYOLO11 = create_optimized_yolo_wrapper()
        
        try:
            # 테스트 이미지
            test_image = "winter01.jpg"
            if os.path.exists(test_image):
                image = cv2.imread(test_image)
                
                # 최적화된 검출기 생성
                detector = OptimizedYOLO11("../models/yolo11m.pt", device='auto')
                
                # 사람 검출 테스트
                start_time = time.time()
                person_boxes = detector.detect_persons(image, conf=0.5)
                detection_time = time.time() - start_time
                
                print(f"✅ 검출 완료: {len(person_boxes)}명, {detection_time*1000:.1f}ms")
                
                # 검출된 박스 정보
                for i, box in enumerate(person_boxes):
                    x1, y1, x2, y2 = box
                    w, h = x2 - x1, y2 - y1
                    print(f"   사람 {i+1}: ({x1:.0f}, {y1:.0f}) {w:.0f}x{h:.0f}")
                
            else:
                print(f"⚠️ 테스트 이미지 없음: {test_image}")
                
        except Exception as e:
            print(f"❌ 최적화된 래퍼 테스트 실패: {e}")
    else:
        print("❌ 사용 가능한 디바이스 없음")

if __name__ == "__main__":
    main()
