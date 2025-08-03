#!/usr/bin/env python3
"""
RTMW XPU 하이브리드 추론기 - 검출은 CPU, 포즈 추정은 XPU
"""

import os
import torch
import cv2
import numpy as np
import time
from mmpose.apis import init_model, inference_topdown

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

class RTMWHybridInferencer:
    """RTMW 하이브리드 추론기 - 검출은 CPU, 포즈는 XPU"""
    
    def __init__(self, config_path: str, checkpoint_path: str):
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        
        # XPU 가용성 확인
        self.xpu_available = check_xpu_availability()
        self.pose_device = 'xpu' if self.xpu_available else 'cpu'
        
        print(f"🚀 RTMW 하이브리드 추론기 초기화:")
        print(f"   - 포즈 추정 디바이스: {self.pose_device}")
        
        # PyTorch 보안 설정
        self.original_load = torch.load
        torch.load = lambda *args, **kwargs: self.original_load(*args, **kwargs, weights_only=False) if 'weights_only' not in kwargs else self.original_load(*args, **kwargs)
        
        # 포즈 추정 모델 초기화 (XPU 사용)
        self._init_pose_model()
        
    def _init_pose_model(self):
        """포즈 추정 모델 초기화"""
        print(f"🔧 포즈 추정 모델 로딩 중... (디바이스: {self.pose_device})")
        start_time = time.time()
        
        try:
            self.pose_model = init_model(
                config=self.config_path,
                checkpoint=self.checkpoint_path,
                device=self.pose_device
            )
            
            init_time = time.time() - start_time
            print(f"✅ 포즈 모델 로딩 완료: {init_time:.2f}초")
            
        except Exception as e:
            print(f"❌ XPU 포즈 모델 실패: {e}")
            print(f"🔄 CPU로 폴백...")
            self.pose_device = 'cpu'
            
            self.pose_model = init_model(
                config=self.config_path,
                checkpoint=self.checkpoint_path,
                device='cpu'
            )
            
            init_time = time.time() - start_time
            print(f"✅ CPU 포즈 모델 로딩 완료: {init_time:.2f}초")
            
    def simple_person_detection(self, image: np.ndarray, conf_thresh: float = 0.5):
        """간단한 사람 검출 (전체 이미지를 바운딩박스로 사용)"""
        h, w = image.shape[:2]
        
        # 이미지 중앙 80% 영역을 바운딩박스로 사용
        margin_w = int(w * 0.1)
        margin_h = int(h * 0.1)
        
        bbox = [margin_w, margin_h, w - margin_w, h - margin_h]
        return [bbox]
        
    def estimate_pose(self, image: np.ndarray, bbox: list):
        """포즈 추정 (XPU 사용)"""
        try:
            # MMPose 추론
            results = inference_topdown(
                model=self.pose_model,
                img=image,
                bboxes=[bbox],
                bbox_format='xyxy'
            )
            
            if results and len(results) > 0:
                # 키포인트 추출
                keypoints = results[0].pred_instances.keypoints[0]
                scores = results[0].pred_instances.keypoint_scores[0]
                
                # numpy 배열로 변환
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
            
    def visualize_results(self, image: np.ndarray, keypoints: np.ndarray, scores: np.ndarray, bbox: list):
        """결과 시각화"""
        vis_image = image.copy()
        
        # 바운딩박스 그리기
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # 키포인트 그리기 (신뢰도 0.3 이상)
        for i, (kpt, score) in enumerate(zip(keypoints, scores)):
            if score > 0.3:
                x, y = int(kpt[0]), int(kpt[1])
                cv2.circle(vis_image, (x, y), 3, (0, 0, 255), -1)
                
        # 디바이스 정보 표시
        cv2.putText(vis_image, f"Pose Device: {self.pose_device.upper()}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return vis_image
        
    def test_single_image(self, image_path: str):
        """단일 이미지 테스트"""
        print(f"\n=== 단일 이미지 테스트: {image_path} ===")
        
        if not os.path.exists(image_path):
            print(f"❌ 이미지 파일 없음: {image_path}")
            return
            
        # 이미지 로드
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ 이미지 로드 실패: {image_path}")
            return
            
        print(f"📷 이미지 크기: {image.shape}")
        
        # 1. 사람 검출 (간단한 방식)
        print("🔍 사람 검출 중...")
        bboxes = self.simple_person_detection(image)
        print(f"🧑 검출된 영역: {len(bboxes)}개")
        
        # 2. 포즈 추정
        for i, bbox in enumerate(bboxes):
            print(f"🎯 포즈 추정 중... bbox: {bbox}")
            
            start_time = time.time()
            keypoints, scores = self.estimate_pose(image, bbox)
            inference_time = time.time() - start_time
            
            print(f"  ⏱️ 추론 시간: {inference_time*1000:.2f}ms")
            print(f"  📊 키포인트: {keypoints.shape}")
            print(f"  📈 평균 신뢰도: {scores.mean():.3f}")
            
            # 고신뢰도 키포인트
            high_conf_mask = scores > 0.5
            high_conf_count = high_conf_mask.sum()
            print(f"  🎯 고신뢰도(>0.5) 키포인트: {high_conf_count}/{len(scores)}")
            
            # 시각화
            vis_image = self.visualize_results(image, keypoints, scores, bbox)
            
            # 결과 저장
            output_path = f"rtmw_hybrid_result_{int(time.time())}_{i}.jpg"
            cv2.imwrite(output_path, vis_image)
            print(f"💾 결과 저장: {output_path}")
            
        return True
        
    def benchmark_performance(self, image_path: str, num_iterations: int = 10):
        """성능 벤치마크"""
        print(f"\n=== 성능 벤치마크 ({num_iterations}회) ===")
        
        # 이미지 로드
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ 이미지 로드 실패: {image_path}")
            return
            
        bboxes = self.simple_person_detection(image)
        bbox = bboxes[0] if bboxes else [0, 0, image.shape[1], image.shape[0]]
        
        # 워밍업
        print("🔥 워밍업 중...")
        for _ in range(3):
            self.estimate_pose(image, bbox)
            
        # 벤치마크
        print(f"📊 벤치마킹 중...")
        times = []
        
        for i in range(num_iterations):
            start_time = time.time()
            keypoints, scores = self.estimate_pose(image, bbox)
            end_time = time.time()
            
            iteration_time = end_time - start_time
            times.append(iteration_time)
            print(f"  반복 {i+1}/{num_iterations}: {iteration_time*1000:.2f}ms")
            
        # 결과 분석
        avg_time = np.mean(times)
        min_time = np.min(times)
        max_time = np.max(times)
        std_time = np.std(times)
        
        print(f"\n📊 벤치마크 결과:")
        print(f"   - 포즈 디바이스: {self.pose_device}")
        print(f"   - 평균 시간: {avg_time*1000:.2f}ms (±{std_time*1000:.2f}ms)")
        print(f"   - 최소 시간: {min_time*1000:.2f}ms")
        print(f"   - 최대 시간: {max_time*1000:.2f}ms")
        print(f"   - 평균 FPS: {1/avg_time:.1f}")
        print(f"   - XPU 가속: {'✅' if self.pose_device == 'xpu' else '❌'}")
        
    def __del__(self):
        """소멸자"""
        if hasattr(self, 'original_load'):
            torch.load = self.original_load

def main():
    """메인 함수"""
    print("=== RTMW XPU 하이브리드 추론기 ===")
    print("검출: CPU, 포즈 추정: XPU")
    
    config_path = "../configs/wholebody_2d_keypoint/rtmpose/cocktail14/rtmw-x_8xb320-270e_cocktail14-384x288.py"
    checkpoint_path = "../models/rtmw-x_simcc-cocktail14_pt-ucoco_270e-384x288-f840f204_20231122.pth"
    test_image = "winter01.jpg"
    
    # 파일 존재 확인
    if not os.path.exists(config_path):
        print(f"❌ 설정 파일 없음: {config_path}")
        return
        
    if not os.path.exists(checkpoint_path):
        print(f"❌ 체크포인트 파일 없음: {checkpoint_path}")
        return
        
    if not os.path.exists(test_image):
        print(f"❌ 테스트 이미지 없음: {test_image}")
        return
    
    try:
        # 하이브리드 추론기 초기화
        inferencer = RTMWHybridInferencer(config_path, checkpoint_path)
        
        # 단일 이미지 테스트
        inferencer.test_single_image(test_image)
        
        # 성능 벤치마크
        inferencer.benchmark_performance(test_image, num_iterations=10)
        
        print("\n✅ 하이브리드 추론기 테스트 완료!")
        print(f"   💡 요약: 포즈 추정은 {inferencer.pose_device.upper()}에서 실행")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
