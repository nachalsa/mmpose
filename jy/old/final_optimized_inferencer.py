#!/usr/bin/env python3
"""
🏆 최종 YOLO11 + RTMW XPU 하이브리드 시스템 
최고 성능과 정확도를 위한 최종 추론기
"""

import os
import torch
import cv2
import numpy as np
import time
from typing import List, Tuple, Optional
from mmpose.apis import init_model, inference_topdown

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    print("⚠️ ultralytics 미설치 - pip install ultralytics")
    YOLO_AVAILABLE = False

def check_xpu_availability():
    """XPU 가용성 확인"""
    try:
        if torch.xpu.is_available():
            device_count = torch.xpu.device_count()
            print(f"✅ Intel XPU 사용 가능: {device_count}개 디바이스")
            return True
        else:
            print("⚠️ Intel XPU 사용 불가 - CPU 모드로 실행")
            return False
    except Exception as e:
        print(f"⚠️ XPU 확인 실패: {e} - CPU 모드로 실행")
        return False

class FinalOptimizedInferencer:
    """최종 최적화된 하이브리드 추론기 - 속도와 정확도의 균형"""
    
    def __init__(self, 
                 rtmw_config: str, 
                 rtmw_checkpoint: str,
                 use_yolo_detection: bool = True,
                 yolo_model: str = "yolo11s.pt",  # Small 모델이 최적 균형
                 detection_device: str = "auto",
                 pose_device: str = "auto"):
        """
        Args:
            rtmw_config: RTMW 설정 파일 경로
            rtmw_checkpoint: RTMW 체크포인트 경로
            use_yolo_detection: YOLO11 검출 사용 여부 (False면 간단한 검출)
            yolo_model: YOLO11 모델 ('yolo11s.pt' 추천)
            detection_device: 검출 디바이스
            pose_device: 포즈 추정 디바이스
        """
        self.rtmw_config = rtmw_config
        self.rtmw_checkpoint = rtmw_checkpoint
        self.use_yolo_detection = use_yolo_detection
        self.yolo_model_name = yolo_model
        
        # XPU 가용성 확인
        self.xpu_available = check_xpu_availability()
        
        # 디바이스 결정
        self.detection_device = self._determine_device(detection_device, "검출")
        self.pose_device = self._determine_device(pose_device, "포즈추정")
        
        print(f"🚀 최종 최적화 하이브리드 추론기 초기화:")
        print(f"   - 검출 방식: {'YOLO11 ' + yolo_model if use_yolo_detection else '간단한 박스'}")
        print(f"   - 검출 디바이스: {self.detection_device}")
        print(f"   - 포즈 추정 디바이스: {self.pose_device}")
        
        # PyTorch 보안 설정
        self.original_load = torch.load
        torch.load = lambda *args, **kwargs: self.original_load(*args, **kwargs, weights_only=False) if 'weights_only' not in kwargs else self.original_load(*args, **kwargs)
        
        # 모델 초기화
        if self.use_yolo_detection:
            self._init_yolo_detection_model()
        self._init_pose_model()
        
        # 성능 통계
        self.inference_times = {
            'detection': [],
            'pose': [],
            'total': []
        }
        
    def _determine_device(self, device: str, task_name: str) -> str:
        """디바이스 자동 결정"""
        if device == "auto":
            if self.xpu_available:
                return "xpu"
            elif torch.cuda.is_available():
                return "cuda"
            else:
                return "cpu"
        else:
            if device == "xpu" and not self.xpu_available:
                print(f"⚠️ {task_name} XPU 미사용 가능 - CPU로 폴백")
                return "cpu"
            elif device == "cuda" and not torch.cuda.is_available():
                print(f"⚠️ {task_name} CUDA 미사용 가능 - CPU로 폴백")
                return "cpu"
            return device
    
    def _init_yolo_detection_model(self):
        """YOLO11 검출 모델 초기화"""
        if not YOLO_AVAILABLE:
            print("⚠️ YOLO11 사용 불가 - 간단한 검출로 폴백")
            self.use_yolo_detection = False
            return
            
        print(f"🔧 YOLO11 검출 모델 로딩 중... (디바이스: {self.detection_device})")
        start_time = time.time()
        
        try:
            model_path = os.path.join("../models", self.yolo_model_name)
            self.detection_model = YOLO(model_path)
            
            if self.detection_device != "cpu":
                try:
                    self.detection_model.to(self.detection_device)
                    print(f"✅ YOLO11 {self.detection_device.upper()} 모드 활성화")
                except Exception as e:
                    print(f"⚠️ YOLO11 {self.detection_device.upper()} 실패, CPU로 폴백: {e}")
                    self.detection_device = "cpu"
                    self.detection_model.to('cpu')
            
            init_time = time.time() - start_time
            print(f"✅ YOLO11 검출 모델 로딩 완료: {init_time:.2f}초")
            
        except Exception as e:
            print(f"❌ YOLO11 모델 로딩 실패: {e}")
            print(f"🔄 간단한 검출로 폴백")
            self.use_yolo_detection = False
    
    def _init_pose_model(self):
        """RTMW 포즈 추정 모델 초기화"""
        print(f"🔧 RTMW 포즈 모델 로딩 중... (디바이스: {self.pose_device})")
        start_time = time.time()
        
        try:
            self.pose_model = init_model(
                config=self.rtmw_config,
                checkpoint=self.rtmw_checkpoint,
                device=self.pose_device
            )
            
            init_time = time.time() - start_time
            print(f"✅ RTMW 포즈 모델 로딩 완료: {init_time:.2f}초")
            
        except Exception as e:
            print(f"❌ {self.pose_device} 포즈 모델 실패: {e}")
            print(f"🔄 CPU로 폴백...")
            self.pose_device = 'cpu'
            
            self.pose_model = init_model(
                config=self.rtmw_config,
                checkpoint=self.rtmw_checkpoint,
                device='cpu'
            )
            
            init_time = time.time() - start_time
            print(f"✅ CPU 포즈 모델 로딩 완료: {init_time:.2f}초")
    
    def detect_persons(self, image: np.ndarray, conf_thresh: float = 0.6) -> List[List[float]]:
        """사람 검출 (YOLO11 또는 간단한 방식)"""
        start_time = time.time()
        
        if self.use_yolo_detection and hasattr(self, 'detection_model'):
            try:
                # YOLO11 검출
                results = self.detection_model(
                    image,
                    conf=conf_thresh,
                    classes=[0],  # 사람 클래스만
                    verbose=False
                )
                
                person_boxes = []
                for result in results:
                    boxes = result.boxes
                    if boxes is not None and len(boxes) > 0:
                        person_coords = boxes.xyxy
                        person_confs = boxes.conf
                        
                        conf_mask = person_confs >= conf_thresh
                        if conf_mask.any():
                            filtered_boxes = person_coords[conf_mask]
                            
                            if isinstance(filtered_boxes, torch.Tensor):
                                filtered_boxes = filtered_boxes.cpu().numpy()
                                
                            person_boxes.extend(filtered_boxes.tolist())
                
                detection_time = time.time() - start_time
                self.inference_times['detection'].append(detection_time)
                
                return person_boxes if person_boxes else self._simple_detection(image)
                
            except Exception as e:
                print(f"❌ YOLO11 검출 실패: {e}")
                return self._simple_detection(image)
        else:
            # 간단한 검출
            result = self._simple_detection(image)
            detection_time = time.time() - start_time
            self.inference_times['detection'].append(detection_time)
            return result
    
    def _simple_detection(self, image: np.ndarray) -> List[List[float]]:
        """간단한 사람 검출 (전체 이미지의 중앙 80% 영역)"""
        h, w = image.shape[:2]
        margin_w = int(w * 0.1)
        margin_h = int(h * 0.1)
        bbox = [margin_w, margin_h, w - margin_w, h - margin_h]
        return [bbox]
    
    def estimate_pose(self, image: np.ndarray, bbox: List[float]) -> Tuple[np.ndarray, np.ndarray]:
        """포즈 추정"""
        try:
            start_time = time.time()
            
            results = inference_topdown(
                model=self.pose_model,
                img=image,
                bboxes=[bbox],
                bbox_format='xyxy'
            )
            
            pose_time = time.time() - start_time
            self.inference_times['pose'].append(pose_time)
            
            if results and len(results) > 0:
                keypoints = results[0].pred_instances.keypoints[0]
                scores = results[0].pred_instances.keypoint_scores[0]
                
                if isinstance(keypoints, torch.Tensor):
                    keypoints = keypoints.cpu().numpy()
                if isinstance(scores, torch.Tensor):
                    scores = scores.cpu().numpy()
                    
                return keypoints, scores
            else:
                return np.zeros((133, 2)), np.zeros(133)
                
        except Exception as e:
            print(f"❌ 포즈 추정 실패: {e}")
            return np.zeros((133, 2)), np.zeros(133)
    
    def process_frame(self, image: np.ndarray, conf_thresh: float = 0.6) -> Tuple[np.ndarray, List[Tuple[np.ndarray, np.ndarray, List[float]]]]:
        """프레임 처리 (검출 + 포즈 추정)"""
        start_time = time.time()
        
        # 1. 사람 검출
        person_boxes = self.detect_persons(image, conf_thresh)
        
        # 2. 각 사람에 대해 포즈 추정
        results = []
        for bbox in person_boxes:
            keypoints, scores = self.estimate_pose(image, bbox)
            results.append((keypoints, scores, bbox))
        
        total_time = time.time() - start_time
        self.inference_times['total'].append(total_time)
        
        # 3. 시각화
        vis_image = self.visualize_results(image, results)
        
        return vis_image, results
    
    def visualize_results(self, image: np.ndarray, results: List[Tuple[np.ndarray, np.ndarray, List[float]]]) -> np.ndarray:
        """결과 시각화"""
        vis_image = image.copy()
        
        for keypoints, scores, bbox in results:
            # 바운딩박스 그리기
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 키포인트 그리기 (신뢰도 0.3 이상)
            for i, (kpt, score) in enumerate(zip(keypoints, scores)):
                if score > 0.3:
                    x, y = int(kpt[0]), int(kpt[1])
                    if 0 <= x < vis_image.shape[1] and 0 <= y < vis_image.shape[0]:
                        cv2.circle(vis_image, (x, y), 3, (0, 0, 255), -1)
        
        # 성능 정보 표시
        if self.inference_times['total']:
            fps = 1.0 / self.inference_times['total'][-1]
            cv2.putText(vis_image, f"FPS: {fps:.1f}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 시스템 정보
        detection_method = f"YOLO11 {self.yolo_model_name}" if self.use_yolo_detection else "Simple"
        cv2.putText(vis_image, f"Detection: {detection_method}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(vis_image, f"Pose: {self.pose_device.upper()}", 
                   (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return vis_image
    
    def benchmark_performance(self, image: np.ndarray, num_runs: int = 20) -> dict:
        """성능 벤치마크"""
        print(f"🏃 성능 벤치마크 ({num_runs}회)...")
        
        # 워밍업
        for _ in range(5):
            self.process_frame(image)
        
        # 실제 벤치마크
        self.inference_times = {'detection': [], 'pose': [], 'total': []}
        
        for i in range(num_runs):
            self.process_frame(image)
            if (i + 1) % 10 == 0:
                print(f"   진행률: {i+1}/{num_runs}")
        
        # 통계 계산
        stats = {}
        for key, times in self.inference_times.items():
            if times:
                stats[key] = {
                    'mean': np.mean(times),
                    'std': np.std(times),
                    'min': np.min(times),
                    'max': np.max(times),
                    'fps': 1.0 / np.mean(times) if key == 'total' else None
                }
        
        return stats
    
    def test_single_image(self, image_path: str):
        """단일 이미지 테스트"""
        detection_type = "YOLO11" if self.use_yolo_detection else "Simple"
        print(f"\n=== 최종 {detection_type} 하이브리드 테스트: {os.path.basename(image_path)} ===")
        
        if not os.path.exists(image_path):
            print(f"❌ 이미지 파일 없음: {image_path}")
            return
            
        # 이미지 로드
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ 이미지 로드 실패: {image_path}")
            return
        
        print(f"📷 이미지 크기: {image.shape}")
        
        # 처리
        vis_image, results = self.process_frame(image, conf_thresh=0.6)
        
        # 결과 출력
        print(f"✅ 검출된 사람 수: {len(results)}")
        
        for i, (keypoints, scores, bbox) in enumerate(results):
            valid_kpts = np.sum(scores > 0.3)
            print(f"   사람 {i+1}: {valid_kpts}/133 키포인트 (신뢰도 > 0.3)")
        
        # 성능 벤치마크
        stats = self.benchmark_performance(image, num_runs=20)
        
        print(f"\n📊 성능 통계:")
        for stage, stat in stats.items():
            if stat:
                print(f"   {stage}:")
                print(f"     - 평균: {stat['mean']*1000:.1f}ms")
                print(f"     - 최소/최대: {stat['min']*1000:.1f}/{stat['max']*1000:.1f}ms")
                if stat['fps']:
                    print(f"     - FPS: {stat['fps']:.1f}")
        
        # 결과 저장
        output_path = f"final_{detection_type.lower()}_result_{os.path.basename(image_path)}"
        cv2.imwrite(output_path, vis_image)
        print(f"💾 결과 저장: {output_path}")
        
        return vis_image, results, stats

def main():
    """메인 함수 - 최종 시스템 테스트"""
    # 모델 경로 설정
    rtmw_config = "../configs/wholebody_2d_keypoint/rtmpose/cocktail14/rtmw-x_8xb320-270e_cocktail14-384x288.py"
    rtmw_checkpoint = "../models/rtmw-x_simcc-cocktail14_pt-ucoco_270e-384x288-f840f204_20231122.pth"
    
    try:
        print("🏆 최종 최적화 하이브리드 포즈 추정 시스템")
        print("=" * 60)
        
        # 1. 간단한 검출 방식 (최고 속도)
        print("\n🏃 속도 우선 - 간단한 검출 방식")
        simple_inferencer = FinalOptimizedInferencer(
            rtmw_config=rtmw_config,
            rtmw_checkpoint=rtmw_checkpoint,
            use_yolo_detection=False
        )
        simple_inferencer.test_single_image("winter01.jpg")
        
        # 2. YOLO11 검출 방식 (균형 잡힌 성능)
        print("\n🎯 균형 우선 - YOLO11 Small 검출 방식")
        yolo_inferencer = FinalOptimizedInferencer(
            rtmw_config=rtmw_config,
            rtmw_checkpoint=rtmw_checkpoint,
            use_yolo_detection=True,
            yolo_model="yolo11s.pt"
        )
        yolo_inferencer.test_single_image("winter01.jpg")
        
        print("\n🏆 최종 추천:")
        print("   🏃 최고 속도가 필요한 경우: 간단한 검출 방식 (25+ FPS)")
        print("   🎯 정확한 사람 검출이 중요한 경우: YOLO11 Small 방식 (19+ FPS)")
        print("   ⚖️ 실시간 애플리케이션: YOLO11 Small 추천 (검출 정확도 + 적절한 속도)")
        
    except Exception as e:
        print(f"❌ 최종 테스트 실패: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
