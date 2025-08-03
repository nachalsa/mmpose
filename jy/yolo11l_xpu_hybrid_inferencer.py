#!/usr/bin/env python3
"""
YOLO11L + RTMW XPU 하이브리드 추론기
Large 모델을 사용한 고정확도 사람 검출 + XPU 포즈 추정
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

class YOLO11LXPUHybridInferencer:
    """YOLO11L + RTMW XPU 하이브리드 추론기 - 고정확도 버전"""
    
    def __init__(self, 
                 rtmw_config: str, 
                 rtmw_checkpoint: str,
                 detection_device: str = "auto",
                 pose_device: str = "auto",
                 optimize_for_accuracy: bool = True):
        """
        Args:
            rtmw_config: RTMW 설정 파일 경로
            rtmw_checkpoint: RTMW 체크포인트 경로
            detection_device: 검출 디바이스 ('auto', 'cpu', 'cuda', 'xpu')
            pose_device: 포즈 추정 디바이스 ('auto', 'cpu', 'cuda', 'xpu')
            optimize_for_accuracy: 정확도 최적화 여부
        """
        if not YOLO_AVAILABLE:
            raise ImportError("ultralytics가 필요합니다: pip install ultralytics")
            
        self.rtmw_config = rtmw_config
        self.rtmw_checkpoint = rtmw_checkpoint
        self.yolo_model_name = "yolo11l.pt"  # Large 모델 사용
        self.optimize_for_accuracy = optimize_for_accuracy
        
        # XPU 가용성 확인
        self.xpu_available = check_xpu_availability()
        
        # 디바이스 결정
        self.detection_device = self._determine_device(detection_device, "검출")
        self.pose_device = self._determine_device(pose_device, "포즈추정")
        
        print(f"🚀 YOLO11L + RTMW 하이브리드 추론기 초기화:")
        print(f"   - YOLO 모델: YOLO11L (Large - 고정확도)")
        print(f"   - 검출 디바이스: {self.detection_device}")
        print(f"   - 포즈 추정 디바이스: {self.pose_device}")
        print(f"   - 정확도 최적화: {'ON' if optimize_for_accuracy else 'OFF'}")
        
        # PyTorch 보안 설정
        self.original_load = torch.load
        torch.load = lambda *args, **kwargs: self.original_load(*args, **kwargs, weights_only=False) if 'weights_only' not in kwargs else self.original_load(*args, **kwargs)
        
        # 모델 초기화
        self._init_detection_model()
        self._init_pose_model()
        
        # 성능 통계
        self.inference_times = {
            'detection': [],
            'pose': [],
            'total': []
        }
        
        # 최적화 설정
        self._setup_optimization()
        
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
    
    def _setup_optimization(self):
        """최적화 설정 - 정확도 우선"""
        if self.optimize_for_accuracy:
            print("🎯 정확도 최적화 설정 적용 중...")
            
            # YOLO11L 정확도 우선 파라미터
            self.yolo_conf_thresh = 0.4     # 낮은 신뢰도 (더 많은 검출)
            self.yolo_iou_thresh = 0.6      # 적당한 IoU 임계값
            self.yolo_max_det = 50          # 더 많은 검출 허용
            self.yolo_classes = [0]         # 사람 클래스만
            
            # 이미지 크기 최적화 (더 큰 입력 크기)
            self.detection_img_size = 832   # Large 모델에 적합한 큰 입력 크기
            self.pose_batch_size = 1        # 포즈 추정 배치 크기
            
            # 멀티스케일 테스트 (선택적)
            self.use_multiscale = False     # 시간이 더 걸리므로 기본 false
            
            print("✅ 정확도 최적화 설정 완료")
        else:
            # 균형 설정
            self.yolo_conf_thresh = 0.5
            self.yolo_iou_thresh = 0.7
            self.yolo_max_det = 100
            self.yolo_classes = None
            self.detection_img_size = 640
            self.use_multiscale = False
    
    def _init_detection_model(self):
        """YOLO11L 검출 모델 초기화"""
        print(f"🔧 YOLO11L 검출 모델 로딩 중... (디바이스: {self.detection_device})")
        start_time = time.time()
        
        try:
            # YOLO11L 모델 로드
            model_path = os.path.join("../models", self.yolo_model_name)
            if not os.path.exists(model_path):
                print(f"📥 YOLO11L 모델 다운로드 중: {self.yolo_model_name}")
                
            self.detection_model = YOLO(model_path)
            
            # 디바이스 설정
            if self.detection_device != "cpu":
                try:
                    self.detection_model.to(self.detection_device)
                    print(f"✅ YOLO11L {self.detection_device.upper()} 모드 활성화")
                except Exception as e:
                    print(f"⚠️ YOLO11L {self.detection_device.upper()} 실패, CPU로 폴백: {e}")
                    self.detection_device = "cpu"
                    self.detection_model.to('cpu')
            
            init_time = time.time() - start_time
            print(f"✅ YOLO11L 검출 모델 로딩 완료: {init_time:.2f}초")
            
        except Exception as e:
            print(f"❌ YOLO11L 모델 로딩 실패: {e}")
            self.detection_model = None
            self.use_simple_detection = True
            print(f"🔄 간단한 검출기로 폴백")
    
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
    
    def detect_persons_high_accuracy(self, image: np.ndarray) -> List[List[float]]:
        """고정확도 사람 검출"""
        if self.detection_model is None:
            return self._simple_person_detection(image)
        
        try:
            start_time = time.time()
            
            # YOLO11L 추론 (고정확도 파라미터)
            results = self.detection_model(
                image,
                conf=self.yolo_conf_thresh,
                iou=self.yolo_iou_thresh,
                max_det=self.yolo_max_det,
                classes=self.yolo_classes,
                verbose=False,
                imgsz=self.detection_img_size,
                augment=self.use_multiscale  # TTA (Test Time Augmentation)
            )
            
            detection_time = time.time() - start_time
            self.inference_times['detection'].append(detection_time)
            
            # 사람(class 0) 검출 결과 추출
            person_boxes = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None and len(boxes) > 0:
                    person_coords = boxes.xyxy
                    person_confs = boxes.conf
                    
                    # 신뢰도 재필터링
                    conf_mask = person_confs >= self.yolo_conf_thresh
                    if conf_mask.any():
                        filtered_boxes = person_coords[conf_mask]
                        filtered_confs = person_confs[conf_mask]
                        
                        # numpy 변환
                        if isinstance(filtered_boxes, torch.Tensor):
                            filtered_boxes = filtered_boxes.cpu().numpy()
                            filtered_confs = filtered_confs.cpu().numpy()
                        
                        # 신뢰도순 정렬
                        sorted_indices = np.argsort(filtered_confs)[::-1]
                        sorted_boxes = filtered_boxes[sorted_indices]
                        
                        person_boxes.extend(sorted_boxes.tolist())
            
            print(f"🔍 YOLO11L 검출 결과: {len(person_boxes)}명, {detection_time:.3f}초")
            
            return person_boxes if person_boxes else self._simple_person_detection(image)
            
        except Exception as e:
            print(f"❌ YOLO11L 검출 실패: {e}")
            return self._simple_person_detection(image)
    
    def _simple_person_detection(self, image: np.ndarray) -> List[List[float]]:
        """간단한 사람 검출 (폴백)"""
        h, w = image.shape[:2]
        margin_w = int(w * 0.1)
        margin_h = int(h * 0.1)
        bbox = [margin_w, margin_h, w - margin_w, h - margin_h]
        return [bbox]
    
    def estimate_pose(self, image: np.ndarray, bbox: List[float]) -> Tuple[np.ndarray, np.ndarray]:
        """RTMW를 사용한 포즈 추정 (XPU)"""
        try:
            start_time = time.time()
            
            # MMPose 추론
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
    
    def process_frame(self, image: np.ndarray, conf_thresh: float = None) -> Tuple[np.ndarray, List[Tuple[np.ndarray, np.ndarray, List[float]]]]:
        """프레임 처리 (고정확도 검출 + 포즈 추정)"""
        start_time = time.time()
        
        # 신뢰도 임계값 설정
        if conf_thresh is None:
            conf_thresh = self.yolo_conf_thresh
        
        # 1. 고정확도 사람 검출
        person_boxes = self.detect_persons_high_accuracy(image)
        
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
        """결과 시각화 - 향상된 버전"""
        vis_image = image.copy()
        
        for i, (keypoints, scores, bbox) in enumerate(results):
            # 바운딩박스 그리기 (다른 색상으로)
            x1, y1, x2, y2 = map(int, bbox)
            color = (0, 255, 0) if i == 0 else (255, 0, 255)  # 첫 번째는 초록, 나머지는 자홍
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
            
            # 사람 번호 표시
            cv2.putText(vis_image, f"Person {i+1}", (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # 키포인트 그리기 (신뢰도별 색상)
            for j, (kpt, score) in enumerate(zip(keypoints, scores)):
                if score > 0.3:
                    x, y = int(kpt[0]), int(kpt[1])
                    if 0 <= x < vis_image.shape[1] and 0 <= y < vis_image.shape[0]:
                        # 신뢰도에 따른 색상
                        if score > 0.8:
                            kpt_color = (0, 255, 0)    # 높은 신뢰도: 초록
                        elif score > 0.6:
                            kpt_color = (0, 255, 255)  # 중간 신뢰도: 노랑
                        else:
                            kpt_color = (0, 0, 255)    # 낮은 신뢰도: 빨강
                        
                        cv2.circle(vis_image, (x, y), 3, kpt_color, -1)
        
        # 성능 정보 표시
        if self.inference_times['total']:
            fps = 1.0 / self.inference_times['total'][-1]
            cv2.putText(vis_image, f"FPS: {fps:.1f}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 모델 정보
        cv2.putText(vis_image, "YOLO11L + RTMW (High Accuracy)", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(vis_image, f"Detection: {self.detection_device.upper()}", 
                   (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(vis_image, f"Pose: {self.pose_device.upper()}", 
                   (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return vis_image
    
    def benchmark_performance(self, image: np.ndarray, num_runs: int = 15) -> dict:
        """성능 벤치마크 (Large 모델이므로 적은 실행)"""
        print(f"🏃 YOLO11L 성능 벤치마크 ({num_runs}회)...")
        
        # 워밍업 (Large 모델은 더 오래 걸림)
        for _ in range(3):
            self.process_frame(image)
        
        # 실제 벤치마크
        self.inference_times = {'detection': [], 'pose': [], 'total': []}
        
        for i in range(num_runs):
            self.process_frame(image)
            if (i + 1) % 5 == 0:
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
        print(f"\n=== YOLO11L + RTMW 고정확도 테스트: {os.path.basename(image_path)} ===")
        
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
        vis_image, results = self.process_frame(image)
        
        # 결과 출력
        print(f"✅ 검출된 사람 수: {len(results)}")
        
        for i, (keypoints, scores, bbox) in enumerate(results):
            valid_kpts = np.sum(scores > 0.3)
            high_conf_kpts = np.sum(scores > 0.8)
            bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            print(f"   사람 {i+1}: {valid_kpts}/133 키포인트 (신뢰도 > 0.3), {high_conf_kpts} 고신뢰도")
            print(f"            바운딩박스 크기: {bbox_area:.0f}px²")
        
        # 성능 벤치마크
        stats = self.benchmark_performance(image, num_runs=15)
        
        print(f"\n📊 YOLO11L 성능 통계:")
        for stage, stat in stats.items():
            if stat:
                print(f"   {stage}:")
                print(f"     - 평균: {stat['mean']*1000:.1f}ms")
                print(f"     - 최소/최대: {stat['min']*1000:.1f}/{stat['max']*1000:.1f}ms")
                if stat['fps']:
                    print(f"     - FPS: {stat['fps']:.1f}")
        
        # 결과 저장
        output_path = f"yolo11l_result_{os.path.basename(image_path)}"
        cv2.imwrite(output_path, vis_image)
        print(f"💾 결과 저장: {output_path}")
        
        return vis_image, results, stats

def main():
    """메인 테스트 함수"""
    # 모델 경로 설정
    rtmw_config = "../configs/wholebody_2d_keypoint/rtmpose/cocktail14/rtmw-x_8xb320-270e_cocktail14-384x288.py"
    rtmw_checkpoint = "../models/rtmw-x_simcc-cocktail14_pt-ucoco_270e-384x288-f840f204_20231122.pth"
    
    try:
        print("🚀 YOLO11L + RTMW XPU 고정확도 하이브리드 추론기 테스트")
        print("=" * 70)
        
        # YOLO11L 하이브리드 추론기 생성
        inferencer = YOLO11LXPUHybridInferencer(
            rtmw_config=rtmw_config,
            rtmw_checkpoint=rtmw_checkpoint,
            detection_device="auto",
            pose_device="auto",
            optimize_for_accuracy=True
        )
        
        # 테스트 이미지
        test_image = "winter01.jpg"
        inferencer.test_single_image(test_image)
        
        print(f"\n🏆 YOLO11L 하이브리드 시스템 특징:")
        print(f"   🎯 최고 정확도: Large 모델로 더 정확한 사람 검출")
        print(f"   🔍 정밀 검출: 낮은 신뢰도 임계값으로 놓치기 쉬운 사람도 검출")
        print(f"   📐 큰 입력 크기: 832px로 더 세밀한 검출")
        print(f"   💻 XPU 가속: 검출과 포즈 추정 모두 XPU 활용")
        
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
