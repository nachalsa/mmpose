#!/usr/bin/env python3
"""
최종 작동하는 XPU 최적화 MMPose 추론기
Final Working XPU Optimized MMPose Inferencer
"""

import cv2
import numpy as np
import torch
import os
import time
import warnings
from typing import List, Tuple, Union, Optional

# PyTorch safe_globals 사전 설정
safe_globals_list = [
    'numpy._core.multiarray._reconstruct',
    'numpy.core.multiarray._reconstruct', 
    'numpy.ndarray',
    'numpy.dtype',
    'collections.OrderedDict',
    'torch._utils._rebuild_tensor_v2',
    'torch._utils._rebuild_parameter',
    'torch.nn.parameter.Parameter',
    'torch.Tensor',
    'torch.Size',
    'torch.device',
    'torch.dtype',
]

for global_name in safe_globals_list:
    try:
        torch.serialization.add_safe_globals([global_name])
    except:
        pass

# 경고 무시
warnings.filterwarnings('ignore')

print(f"🚀 PyTorch 버전: {torch.__version__}")
print(f"🔍 XPU 사용 가능: {torch.xpu.is_available()}")

XPU_AVAILABLE = torch.xpu.is_available()

# MMPose 모듈들
try:
    from mmpose.apis import init_model, inference_topdown
    from mmpose.registry import VISUALIZERS
    from mmpose.structures import merge_data_samples
    print("✅ MMPose 모듈 로드 성공")
except ImportError as e:
    print(f"❌ MMPose 모듈 로드 실패: {e}")
    raise e


class FinalWorkingXPUInferencer:
    """최종 작동하는 XPU 최적화 추론기"""
    
    def __init__(self, config_path: str, checkpoint_path: str, device: str = 'auto'):
        """
        Args:
            config_path: 설정 파일 경로
            checkpoint_path: 체크포인트 파일 경로  
            device: 실행 디바이스 ('auto', 'xpu', 'cpu')
        """
        # 디바이스 선택
        if device == 'auto':
            self.device = 'xpu' if XPU_AVAILABLE else 'cpu'
        else:
            self.device = device
            
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        
        print(f"🚀 최종 작동하는 XPU 추론기 초기화:")
        print(f"   - Config: {os.path.basename(config_path)}")
        print(f"   - Checkpoint: {os.path.basename(checkpoint_path)}")
        print(f"   - Device: {self.device}")
        
        # 안전한 체크포인트 준비
        self.safe_checkpoint_path = self._prepare_safe_checkpoint()
        
        # MMPose 모델 초기화
        self._init_model()
        
    def _prepare_safe_checkpoint(self):
        """안전한 체크포인트 준비"""
        safe_path = self.checkpoint_path.replace('.pth', '_safe.pth')
        
        if os.path.exists(safe_path):
            print(f"📂 기존 안전한 체크포인트 사용: {os.path.basename(safe_path)}")
            return safe_path
            
        print(f"🔄 안전한 체크포인트 생성 중...")
        
        # 원본 체크포인트 로드
        checkpoint = torch.load(self.checkpoint_path, map_location='cpu', weights_only=False)
        
        # 안전한 형식으로 저장
        torch.save(checkpoint, safe_path, _use_new_zipfile_serialization=False)
        
        print(f"💾 안전한 체크포인트 생성 완료: {os.path.basename(safe_path)}")
        return safe_path
        
    def _init_model(self):
        """MMPose 모델 초기화"""
        print(f"🔄 MMPose 모델 초기화 중...")
        
        try:
            # MMPose 모델 초기화 (안전한 체크포인트 사용)
            self.model = init_model(
                config=self.config_path,
                checkpoint=self.safe_checkpoint_path,
                device=self.device
            )
            
            print(f"✅ MMPose 모델 초기화 성공!")
            
            # XPU 최적화
            if self.device == 'xpu' and XPU_AVAILABLE:
                self._optimize_for_xpu()
                
        except Exception as e:
            print(f"❌ 모델 초기화 실패: {e}")
            # CPU로 폴백
            print(f"🔄 CPU로 폴백 시도...")
            self.device = 'cpu'
            try:
                self.model = init_model(
                    config=self.config_path,
                    checkpoint=self.safe_checkpoint_path,
                    device='cpu'
                )
                print(f"✅ CPU 모델 초기화 성공!")
            except Exception as e2:
                print(f"❌ CPU 모델도 실패: {e2}")
                raise e2
            
    def _optimize_for_xpu(self):
        """XPU 최적화"""
        print(f"⚡ XPU 최적화 적용 중...")
        
        try:
            # 모델을 XPU로 이동 (이미 init_model에서 됨)
            print(f"   - 모델 디바이스: {next(self.model.parameters()).device}")
            
            # XPU 최적화 (Intel Extension for PyTorch 있는 경우)
            try:
                import intel_extension_for_pytorch as ipex
                self.model = ipex.optimize(self.model)
                print(f"✅ Intel Extension for PyTorch 최적화 적용")
            except ImportError:
                print(f"⚠️ Intel Extension for PyTorch 없음 - 기본 XPU 사용")
                
            # 평가 모드로 설정
            self.model.eval()
            
            print(f"✅ XPU 최적화 완료")
            
        except Exception as e:
            print(f"⚠️ XPU 최적화 실패: {e}")
            
    def estimate_pose(self, image: np.ndarray, bbox: List[float], visualize: bool = False) -> np.ndarray:
        """포즈 추정"""
        try:
            # 바운딩박스 형식 변환 [x1, y1, x2, y2] -> [x1, y1, w, h]
            x1, y1, x2, y2 = bbox
            bbox_xywh = [x1, y1, x2 - x1, y2 - y1]
            
            # MMPose 추론
            results = inference_topdown(
                model=self.model,
                img=image,
                bboxes=[bbox_xywh],
                bbox_format='xywh'
            )
            
            if results and len(results) > 0:
                # 키포인트 추출
                keypoints = results[0].pred_instances.keypoints[0]  # 첫 번째 사람
                scores = results[0].pred_instances.keypoint_scores[0]  # 신뢰도
                
                # numpy 배열로 변환
                if isinstance(keypoints, torch.Tensor):
                    keypoints = keypoints.cpu().numpy()
                if isinstance(scores, torch.Tensor):
                    scores = scores.cpu().numpy()
                    
                # 시각화
                if visualize:
                    self._visualize_keypoints(image, keypoints, scores, bbox)
                    
                return keypoints, scores
            else:
                print(f"⚠️ 추론 결과 없음")
                return np.zeros((133, 2)), np.zeros(133)
                
        except Exception as e:
            print(f"❌ 포즈 추정 실패: {e}")
            import traceback
            traceback.print_exc()
            return np.zeros((133, 2)), np.zeros(133)
            
    def _visualize_keypoints(self, image: np.ndarray, keypoints: np.ndarray, scores: np.ndarray, bbox: List[float]):
        """키포인트 시각화"""
        vis_image = image.copy()
        
        # 바운딩박스 그리기
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # 키포인트 그리기 (신뢰도 0.3 이상만)
        for i, (kpt, score) in enumerate(zip(keypoints, scores)):
            if score > 0.3:
                x, y = int(kpt[0]), int(kpt[1])
                cv2.circle(vis_image, (x, y), 3, (0, 0, 255), -1)
                
        # 결과 저장
        output_path = f"/home/ty/rtmw/02/mmpose/jy/final_result_{int(time.time())}.jpg"
        cv2.imwrite(output_path, vis_image)
        print(f"📷 시각화 결과 저장: {os.path.basename(output_path)}")
        
    def batch_estimate_pose(self, images: List[np.ndarray], bboxes: List[List[float]]) -> List[Tuple[np.ndarray, np.ndarray]]:
        """배치 포즈 추정"""
        results = []
        
        print(f"🔄 배치 추론 ({len(images)}개 이미지)")
        
        for i, (image, bbox) in enumerate(zip(images, bboxes)):
            keypoints, scores = self.estimate_pose(image, bbox)
            results.append((keypoints, scores))
            
        return results
        
    def benchmark_performance(self, num_iterations: int = 20):
        """성능 벤치마크"""
        print(f"🏃 성능 벤치마크 ({num_iterations}회 반복):")
        
        # 테스트 데이터
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        test_bbox = [100, 50, 300, 400]
        
        times = []
        
        # 워밍업
        print(f"🔥 워밍업 중...")
        for _ in range(3):
            self.estimate_pose(test_image, test_bbox)
            
        # 실제 벤치마크
        print(f"📊 벤치마킹 중...")
        for i in range(num_iterations):
            start_time = time.time()
            keypoints, scores = self.estimate_pose(test_image, test_bbox)
            end_time = time.time()
            
            times.append(end_time - start_time)
            
        avg_time = np.mean(times)
        min_time = np.min(times)
        max_time = np.max(times)
        std_time = np.std(times)
        
        print(f"\n📊 벤치마크 결과:")
        print(f"   - 평균 시간: {avg_time*1000:.2f}ms (±{std_time*1000:.2f}ms)")
        print(f"   - 최소 시간: {min_time*1000:.2f}ms")
        print(f"   - 최대 시간: {max_time*1000:.2f}ms")
        print(f"   - 평균 FPS: {1/avg_time:.1f}")
        print(f"   - 디바이스: {self.device}")
        
        return {
            'avg_time_ms': avg_time * 1000,
            'min_time_ms': min_time * 1000,
            'max_time_ms': max_time * 1000,
            'std_time_ms': std_time * 1000,
            'fps': 1 / avg_time,
            'device': self.device
        }


def test_final_working_xpu_inferencer():
    """최종 작동하는 XPU 추론기 테스트"""
    print("=== 최종 작동하는 XPU 추론기 테스트 ===")
    
    config_path = "/home/ty/rtmw/02/mmpose/configs/wholebody_2d_keypoint/rtmpose/cocktail14/rtmw-x_8xb320-270e_cocktail14-384x288.py"
    checkpoint_path = "/home/ty/rtmw/02/mmpose/models/rtmw-x_simcc-cocktail14_pt-ucoco_270e-384x288-f840f204_20231122.pth"
    
    # 파일 존재 확인
    if not os.path.exists(config_path):
        print(f"❌ 설정 파일을 찾을 수 없습니다: {config_path}")
        return
        
    if not os.path.exists(checkpoint_path):
        print(f"❌ 체크포인트 파일을 찾을 수 없습니다: {checkpoint_path}")
        return
    
    try:
        # 추론기 초기화
        inferencer = FinalWorkingXPUInferencer(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            device='auto'
        )
        
        # 테스트 이미지 로드
        test_image_path = "/home/ty/rtmw/02/mmpose/jy/winter01.jpg"
        if os.path.exists(test_image_path):
            image = cv2.imread(test_image_path)
            print(f"📷 테스트 이미지 로드: {image.shape}")
        else:
            # 더미 이미지 생성
            image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            print(f"📷 더미 이미지 생성: {image.shape}")
        
        # 테스트 바운딩박스
        bbox = [100, 50, 400, 450]
        
        print(f"\n📐 단일 추론 테스트:")
        start_time = time.time()
        keypoints, scores = inferencer.estimate_pose(image, bbox, visualize=True)
        end_time = time.time()
        
        print(f"\n📊 추론 결과:")
        print(f"   - 키포인트 shape: {keypoints.shape}")
        print(f"   - 신뢰도 shape: {scores.shape}")
        print(f"   - 추론 시간: {(end_time - start_time)*1000:.2f}ms")
        
        if keypoints.shape[0] > 0:
            print(f"   - X 좌표 범위: [{keypoints[:, 0].min():.1f}, {keypoints[:, 0].max():.1f}]")
            print(f"   - Y 좌표 범위: [{keypoints[:, 1].min():.1f}, {keypoints[:, 1].max():.1f}]")
            print(f"   - 평균 신뢰도: {scores.mean():.3f}")
            
            # 주요 키포인트 확인
            if keypoints.shape[0] >= 17:
                print(f"   - 코 위치: ({keypoints[0, 0]:.1f}, {keypoints[0, 1]:.1f}) 신뢰도: {scores[0]:.3f}")
                print(f"   - 왼쪽 어깨: ({keypoints[5, 0]:.1f}, {keypoints[5, 1]:.1f}) 신뢰도: {scores[5]:.3f}")
                print(f"   - 오른쪽 어깨: ({keypoints[6, 0]:.1f}, {keypoints[6, 1]:.1f}) 신뢰도: {scores[6]:.3f}")
        
        # 성능 벤치마크
        perf_results = inferencer.benchmark_performance(num_iterations=10)
        
        print(f"\n✅ 최종 작동하는 XPU 추론기 테스트 완료!")
        print(f"   - 디바이스: {perf_results['device']}")
        print(f"   - 평균 FPS: {perf_results['fps']:.1f}")
        print(f"   - XPU 사용: {'✅' if perf_results['device'] == 'xpu' else '❌'}")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    test_final_working_xpu_inferencer()
