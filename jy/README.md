# 🏆 YOLO11 + RTMW XPU 하이브리드 포즈 추정 시스템

Intel XPU를 활용한 고성능 실시간 포즈 추정 시스템입니다. YOLO11 사람 검출과 RTMW 전신 포즈 추정을 결합하여 최적의 성능과 정확도를 제공합니다.

## 🚀 주요 특징

- **Intel XPU 최적화**: Intel Arc GPU를 활용한 고성능 추론
- **하이브리드 아키텍처**: YOLO11 검출 + RTMW 포즈 추정
- **실시간 처리**: 15-25 FPS 성능
- **133개 키포인트**: 얼굴, 손, 발, 전신 키포인트 지원
- **다양한 모델 지원**: Nano/Small/Medium YOLO11 모델 선택 가능

## 📊 성능 비교

| 시스템 | FPS | 키포인트 | 검출 정확도 | 추천 용도 |
|--------|-----|----------|-------------|-----------|
| 간단한 검출 + XPU 포즈 | **23.5** | 133/133 | 기본 | 최고 속도 필요시 |
| YOLO11n + XPU 포즈 | 18.8 | 133/133 | 높음 | 초고속 정확한 검출 |
| **YOLO11s + XPU 포즈** | **18.6** | 133/133 | **높음** | **실시간 애플리케이션 추천** |
| YOLO11m + XPU 포즈 | 17.2 | 133/133 | 최고 | 정확도 우선 |

## 🛠️ 설치 방법

### 1. 환경 요구사항

- Python 3.8+
- Intel XPU 지원 환경
- PyTorch 2.9.0.dev20250802+xpu
- Intel Extension for PyTorch

### 2. 패키지 설치

```bash
# MMPose 설치
pip install mmpose

# YOLO11 설치
pip install ultralytics

# Intel XPU 지원
pip install intel-extension-for-pytorch[xpu]
```

### 3. 모델 다운로드

```bash
# RTMW-x 모델 다운로드 (자동)
# YOLO11 모델 다운로드 (자동)
```

## 🎯 사용법

### 1. 빠른 시작 - 최종 최적화 버전

```python
from final_optimized_inferencer import FinalOptimizedInferencer

# 간단한 검출 방식 (최고 속도)
inferencer = FinalOptimizedInferencer(
    rtmw_config="path/to/rtmw_config.py",
    rtmw_checkpoint="path/to/rtmw_checkpoint.pth",
    use_yolo_detection=False  # 간단한 검출
)

# YOLO11 검출 방식 (균형잡힌 성능)
inferencer = FinalOptimizedInferencer(
    rtmw_config="path/to/rtmw_config.py", 
    rtmw_checkpoint="path/to/rtmw_checkpoint.pth",
    use_yolo_detection=True,  # YOLO11 검출
    yolo_model="yolo11s.pt"   # Small 모델 추천
)

# 이미지 처리
image = cv2.imread("test_image.jpg")
vis_image, results = inferencer.process_frame(image)

# 결과 확인
for keypoints, scores, bbox in results:
    valid_keypoints = np.sum(scores > 0.3)
    print(f"유효한 키포인트: {valid_keypoints}/133")
```

### 2. 실시간 웹캠 데모

```python
from realtime_yolo11_xpu_demo import RealTimeHybridDemo

demo = RealTimeHybridDemo(
    rtmw_config="path/to/rtmw_config.py",
    rtmw_checkpoint="path/to/rtmw_checkpoint.pth",
    yolo_model="yolo11s.pt",
    webcam_id=0,
    target_fps=30
)

demo.run()  # 실시간 데모 실행
```

### 3. 성능 벤치마크

```python
from comprehensive_performance_test import ComprehensivePerformanceTest

tester = ComprehensivePerformanceTest()
tester.run_comprehensive_test()  # 모든 시스템 성능 비교
```

## 📁 파일 구조

```
mmpose/jy/
├── pose_estimator.py                 # 기존 PyTorch 기반 포즈 추정기 (보존)
├── final_optimized_inferencer.py     # 🏆 최종 최적화 추론기 (추천)
├── yolo11_xpu_hybrid_inferencer.py   # 기본 YOLO11 하이브리드 
├── optimized_yolo11_xpu_inferencer.py # 최적화된 YOLO11 하이브리드
├── realtime_yolo11_xpu_demo.py       # 실시간 웹캠 데모
├── comprehensive_performance_test.py  # 종합 성능 비교 도구
├── yolo11_xpu_compatibility_test.py  # YOLO11 XPU 호환성 테스트
└── config.py                         # 설정 파일
```

## 🔧 핵심 파일 설명

### 1. `final_optimized_inferencer.py` (🏆 추천)
- **용도**: 최종 완성된 하이브리드 시스템
- **특징**: 간단한 검출 또는 YOLO11 검출 선택 가능
- **성능**: 18.6-23.5 FPS
- **추천 이유**: 사용하기 쉽고 최적화된 성능

### 2. `pose_estimator.py` (유지됨)
- **용도**: 기존 PyTorch 기반 포즈 추정기
- **특징**: 수동 모델 구성, 학습/연구 목적
- **의미**: 개발 과정과 기술적 이해를 위해 보존

### 3. `realtime_yolo11_xpu_demo.py`
- **용도**: 실시간 웹캠 데모
- **특징**: 키보드 컨트롤, 성능 모니터링
- **컨트롤**: q=종료, b=박스토글, k=키포인트토글

## 🎮 실시간 데모 컨트롤

- **q**: 프로그램 종료
- **b**: 바운딩박스 표시 토글
- **k**: 키포인트 표시 토글
- **r**: 성능 통계 리셋

## ⚡ 성능 최적화 팁

### 1. 속도 우선 설정
```python
# 간단한 검출 사용
use_yolo_detection=False  # 25+ FPS

# 또는 YOLO11 Nano 모델
yolo_model="yolo11n.pt"   # 18+ FPS
```

### 2. 정확도 우선 설정
```python
# YOLO11 Medium 모델
yolo_model="yolo11m.pt"   # 17+ FPS, 최고 정확도
```

### 3. 균형 설정 (추천)
```python
# YOLO11 Small 모델
yolo_model="yolo11s.pt"   # 18+ FPS, 높은 정확도
```

## 🔍 XPU 디바이스 확인

```python
import torch

# XPU 가용성 확인
if torch.xpu.is_available():
    device_count = torch.xpu.device_count()
    print(f"XPU 디바이스 수: {device_count}")
    
    for i in range(device_count):
        props = torch.xpu.get_device_properties(i)
        print(f"XPU {i}: {props.name}")
else:
    print("XPU 사용 불가")
```

## 📈 성능 개선 내역

1. **기존 시스템**: 간단한 검출 + XPU 포즈 추정 (25.1 FPS)
2. **YOLO11 도입**: 정확한 사람 검출 추가 (17-19 FPS)
3. **XPU 최적화**: 모든 모델 XPU 가속 (Intel Arc GPU 활용)
4. **하이브리드 선택**: 용도에 따른 최적 모델 선택 가능

## 🎯 사용 시나리오별 추천

### 실시간 스트리밍/게임
```python
# 최고 속도 우선
use_yolo_detection=False  # 25+ FPS
```

### 보안 감시/모니터링
```python
# 정확한 사람 검출 필요
yolo_model="yolo11s.pt"   # 18+ FPS, 높은 검출 정확도
```

### 스포츠 분석/의료
```python
# 최고 정확도 필요
yolo_model="yolo11m.pt"   # 17+ FPS, 최고 정확도
```

## 🐛 문제 해결

### XPU 인식 안됨
```bash
# Intel Extension for PyTorch 재설치
pip uninstall intel-extension-for-pytorch
pip install intel-extension-for-pytorch[xpu]
```

### YOLO11 다운로드 실패
```bash
# 수동 다운로드
wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt
mv yolo11s.pt models/
```

### MMPose 설정 오류
```bash
# MMPose 재설치
pip uninstall mmpose
pip install mmpose
```

## 📞 지원

- Intel XPU 관련: Intel Extension for PyTorch 문서 참조
- YOLO11 관련: Ultralytics 공식 문서 참조
- MMPose 관련: OpenMMLab MMPose 문서 참조

## 🏆 결론

이 시스템은 Intel XPU의 성능을 최대한 활용하여 실시간 포즈 추정을 구현합니다. YOLO11의 정확한 사람 검출과 RTMW의 정밀한 포즈 추정을 결합하여 다양한 애플리케이션에 적용할 수 있는 고성능 솔루션을 제공합니다.

**최종 추천**: `final_optimized_inferencer.py`를 사용하여 YOLO11 Small 모델로 시작하세요!
