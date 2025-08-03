#!/usr/bin/env python3
"""
최적화된 YOLO11 + RTMW XPU 하이브리드 추론기
성능 최적화 버전 (더 빠른 검출 + 더 정확한 포즈 추정)
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

class OptimizedYOLO11XPUInferencer:
    """최적화된 YOLO11 + RTMW XPU 하이브리드 추론기"""
    
    def __init__(self, 
                 rtmw_config: str, 
                 rtmw_checkpoint: str,
                 yolo_model: str = "yolo11n.pt",  # 더 가벼운 nano 모델 사용
                 detection_device: str = "auto",
                 pose_device: str = "auto",
                 optimize_for_speed: bool = True):
        """
        Args:
            rtmw_config: RTMW 설정 파일 경로
            rtmw_checkpoint: RTMW 체크포인트 경로
            yolo_model: YOLO11 모델 ('yolo11n.pt' = nano, 'yolo11s.pt' = small, 'yolo11m.pt' = medium)
            detection_device: 검출 디바이스 ('auto', 'cpu', 'cuda', 'xpu')
            pose_device: 포즈 추정 디바이스 ('auto', 'cpu', 'cuda', 'xpu')
            optimize_for_speed: 속도 최적화 여부
        """
        if not YOLO_AVAILABLE:
            raise ImportError("ultralytics가 필요합니다: pip install ultralytics")
            
        self.rtmw_config = rtmw_config
        self.rtmw_checkpoint = rtmw_checkpoint
        self.yolo_model_name = yolo_model
        self.optimize_for_speed = optimize_for_speed
        
        # XPU 가용성 확인
        self.xpu_available = check_xpu_availability()
        
        # 디바이스 결정
        self.detection_device = self._determine_device(detection_device, "검출")
        self.pose_device = self._determine_device(pose_device, "포즈추정")
        
        print(f"🚀 최적화된 YOLO11 + RTMW 하이브리드 추론기 초기화:")
        print(f"   - YOLO 모델: {yolo_model}")
        print(f"   - 검출 디바이스: {self.detection_device}")
        print(f"   - 포즈 추정 디바이스: {self.pose_device}")
        print(f"   - 속도 최적화: {'ON' if optimize_for_speed else 'OFF'}")
        
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
            # 지정된 디바이스가 사용 가능한지 확인
            if device == "xpu" and not self.xpu_available:
                print(f"⚠️ {task_name} XPU 미사용 가능 - CPU로 폴백")
                return "cpu"
            elif device == "cuda" and not torch.cuda.is_available():
                print(f"⚠️ {task_name} CUDA 미사용 가능 - CPU로 폴백")
                return "cpu"
            return device
    
    def _setup_optimization(self):
        """최적화 설정"""
        if self.optimize_for_speed:
            print("⚡ 속도 최적화 설정 적용 중...")
            
            # YOLO11 최적화 파라미터
            self.yolo_conf_thresh = 0.7    # 더 높은 신뢰도 (false positive 감소)
            self.yolo_iou_thresh = 0.5     # IoU 임계값
            self.yolo_max_det = 10         # 최대 검출 수 (사람만 찾으므로 낮게)
            self.yolo_classes = [0]        # 사람 클래스만
            
            # 이미지 크기 최적화
            self.detection_img_size = 640  # YOLO 입력 크기 (기본: 640)
            self.pose_batch_size = 1       # 포즈 추정 배치 크기
            
            # 스킵 프레임 설정 (선택적)
            self.detection_skip_frames = 1  # 매 N프레임마다 검출 (1=모든 프레임)
            self.frame_count = 0
            self.last_bboxes = []           # 마지막 검출 결과
            
            print("✅ 속도 최적화 설정 완료")
        else:
            # 정확도 우선 설정
            self.yolo_conf_thresh = 0.5
            self.yolo_iou_thresh = 0.7
            self.yolo_max_det = 100
            self.yolo_classes = None  # 모든 클래스
            self.detection_img_size = 640
            self.detection_skip_frames = 1
    
    def _init_detection_model(self):
        """YOLO11 검출 모델 초기화"""
        print(f"🔧 YOLO11 검출 모델 로딩 중... (디바이스: {self.detection_device})")
        start_time = time.time()
        
        try:
            # YOLO11 모델 로드
            model_path = os.path.join("../models", self.yolo_model_name)
            if not os.path.exists(model_path):
                print(f"📥 YOLO11 모델 다운로드 중: {self.yolo_model_name}")
                
            self.detection_model = YOLO(model_path)
            
            # 디바이스 설정
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
    
    def detect_persons_optimized(self, image: np.ndarray) -> List[List[float]]:
        """최적화된 사람 검출"""
        self.frame_count += 1
        
        # 스킵 프레임 처리 (선택적)
        if self.optimize_for_speed and self.detection_skip_frames > 1:
            if self.frame_count % self.detection_skip_frames != 0 and self.last_bboxes:
                return self.last_bboxes
        
        if self.detection_model is None:
            return self._simple_person_detection(image)
        
        try:
            start_time = time.time()
            
            # YOLO11 추론 (최적화된 파라미터)
            results = self.detection_model(
                image,
                conf=self.yolo_conf_thresh,
                iou=self.yolo_iou_thresh,
                max_det=self.yolo_max_det,
                classes=self.yolo_classes,
                verbose=False,
                imgsz=self.detection_img_size
            )
            
            detection_time = time.time() - start_time
            self.inference_times['detection'].append(detection_time)
            
            # 사람(class 0) 검출 결과 추출
            person_boxes = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None and len(boxes) > 0:
                    # 이미 사람 클래스만 필터링됨
                    person_coords = boxes.xyxy
                    person_confs = boxes.conf
                    
                    # 신뢰도 재필터링
                    conf_mask = person_confs >= self.yolo_conf_thresh
                    if conf_mask.any():
                        filtered_boxes = person_coords[conf_mask]
                        
                        # numpy 변환
                        if isinstance(filtered_boxes, torch.Tensor):
                            filtered_boxes = filtered_boxes.cpu().numpy()
                            
                        person_boxes.extend(filtered_boxes.tolist())
            
            # 검출 결과 캐시 (스킵 프레임용)
            if person_boxes:
                self.last_bboxes = person_boxes
            elif not self.last_bboxes:  # 첫 번째 검출이 실패한 경우
                self.last_bboxes = self._simple_person_detection(image)
            
            return person_boxes if person_boxes else self.last_bboxes
            
        except Exception as e:
            print(f"❌ YOLO11 검출 실패: {e}")
            return self._simple_person_detection(image)
    
    def _simple_person_detection(self, image: np.ndarray) -> List[List[float]]:
        """간단한 사람 검출 (폴백)"""
        h, w = image.shape[:2]
        margin_w = int(w * 0.15)
        margin_h = int(h * 0.15)
        bbox = [margin_w, margin_h, w - margin_w, h - margin_h]
        return [bbox]
    
    def estimate_pose_optimized(self, image: np.ndarray, bbox: List[float]) -> Tuple[np.ndarray, np.ndarray]:
        """최적화된 포즈 추정"""
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
                # 키포인트 추출
                keypoints = results[0].pred_instances.keypoints[0]
                scores = results[0].pred_instances.keypoint_scores[0]
                
                # numpy 변환
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
    
    def process_frame_fast(self, image: np.ndarray, conf_thresh: float = None) -> Tuple[np.ndarray, List[Tuple[np.ndarray, np.ndarray, List[float]]]]:
        """빠른 프레임 처리"""
        start_time = time.time()
        
        # 신뢰도 임계값 설정
        if conf_thresh is None:
            conf_thresh = self.yolo_conf_thresh
        
        # 1. 최적화된 사람 검출
        person_boxes = self.detect_persons_optimized(image)
        
        # 2. 각 사람에 대해 포즈 추정
        results = []
        for bbox in person_boxes:
            keypoints, scores = self.estimate_pose_optimized(image, bbox)
            results.append((keypoints, scores, bbox))
        
        total_time = time.time() - start_time
        self.inference_times['total'].append(total_time)
        
        # 3. 빠른 시각화
        vis_image = self.visualize_results_fast(image, results)
        
        return vis_image, results
    
    def visualize_results_fast(self, image: np.ndarray, results: List[Tuple[np.ndarray, np.ndarray, List[float]]]) -> np.ndarray:
        """빠른 결과 시각화"""
        vis_image = image.copy()
        
        for keypoints, scores, bbox in results:
            # 바운딩박스 그리기
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 주요 키포인트만 그리기 (속도 향상)
            if self.optimize_for_speed:
                # 얼굴, 손목, 발목만 표시
                key_indices = [0, 1, 2, 3, 4, 9, 10, 15, 16, 17, 18, 19, 20]  # 주요 포인트
                for i in key_indices:
                    if i < len(keypoints) and scores[i] > 0.5:
                        x, y = int(keypoints[i][0]), int(keypoints[i][1])
                        if 0 <= x < vis_image.shape[1] and 0 <= y < vis_image.shape[0]:
                            cv2.circle(vis_image, (x, y), 3, (0, 0, 255), -1)
            else:
                # 모든 키포인트 표시
                for i, (kpt, score) in enumerate(zip(keypoints, scores)):
                    if score > 0.3:
                        x, y = int(kpt[0]), int(kpt[1])
                        if 0 <= x < vis_image.shape[1] and 0 <= y < vis_image.shape[0]:
                            cv2.circle(vis_image, (x, y), 2, (0, 0, 255), -1)
        
        # 성능 정보 표시
        if self.inference_times['total']:
            fps = 1.0 / self.inference_times['total'][-1]
            cv2.putText(vis_image, f"FPS: {fps:.1f}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 모델 정보
        model_info = f"YOLO11{self.yolo_model_name[-5]} + RTMW"
        cv2.putText(vis_image, model_info, 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return vis_image
    
    def benchmark_performance(self, image: np.ndarray, num_runs: int = 20) -> dict:
        """성능 벤치마크 (더 많은 실행)"""
        print(f"🏃 최적화된 성능 벤치마크 ({num_runs}회)...")
        
        # 워밍업 (더 많이)
        for _ in range(5):
            self.process_frame_fast(image)
        
        # 실제 벤치마크
        self.inference_times = {'detection': [], 'pose': [], 'total': []}
        
        for i in range(num_runs):
            self.process_frame_fast(image)
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
        print(f"\n=== 최적화된 YOLO11 + RTMW 테스트: {os.path.basename(image_path)} ===")
        
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
        vis_image, results = self.process_frame_fast(image)
        
        # 결과 출력
        print(f"✅ 검출된 사람 수: {len(results)}")
        
        for i, (keypoints, scores, bbox) in enumerate(results):
            valid_kpts = np.sum(scores > 0.3)
            print(f"   사람 {i+1}: {valid_kpts}/133 키포인트 (신뢰도 > 0.3)")
        
        # 성능 벤치마크
        stats = self.benchmark_performance(image, num_runs=20)
        
        print(f"\n📊 최적화된 성능 통계:")
        for stage, stat in stats.items():
            if stat:
                print(f"   {stage}:")
                print(f"     - 평균: {stat['mean']*1000:.1f}ms")
                print(f"     - 최소/최대: {stat['min']*1000:.1f}/{stat['max']*1000:.1f}ms")
                if stat['fps']:
                    print(f"     - FPS: {stat['fps']:.1f}")
        
        # 결과 저장
        output_path = f"optimized_yolo11_result_{os.path.basename(image_path)}"
        cv2.imwrite(output_path, vis_image)
        print(f"💾 결과 저장: {output_path}")
        
        return vis_image, results, stats

def main():
    """메인 테스트 함수"""
    # 모델 경로 설정
    rtmw_config = "../configs/wholebody_2d_keypoint/rtmpose/cocktail14/rtmw-x_8xb320-270e_cocktail14-384x288.py"
    rtmw_checkpoint = "../models/rtmw-x_simcc-cocktail14_pt-ucoco_270e-384x288-f840f204_20231122.pth"
    
    try:
        print("🚀 최적화된 YOLO11 + RTMW XPU 하이브리드 추론기 테스트")
        
        # 다양한 YOLO 모델 테스트
        yolo_models = [
            ("yolo11n.pt", "Nano - 초고속"),
            ("yolo11s.pt", "Small - 고속"),
            ("yolo11m.pt", "Medium - 균형")
        ]
        
        for yolo_model, description in yolo_models:
            print(f"\n{'='*60}")
            print(f"🧪 {description} 모델 테스트: {yolo_model}")
            print(f"{'='*60}")
            
            try:
                # 최적화된 추론기 생성
                inferencer = OptimizedYOLO11XPUInferencer(
                    rtmw_config=rtmw_config,
                    rtmw_checkpoint=rtmw_checkpoint,
                    yolo_model=yolo_model,
                    detection_device="auto",
                    pose_device="auto",
                    optimize_for_speed=True
                )
                
                # 테스트 이미지
                test_image = "winter01.jpg"
                inferencer.test_single_image(test_image)
                
            except Exception as e:
                print(f"❌ {yolo_model} 테스트 실패: {e}")
                continue
        
    except Exception as e:
        print(f"❌ 전체 테스트 실패: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
