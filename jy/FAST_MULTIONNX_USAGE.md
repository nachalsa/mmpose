# Fast Multi-ONNX Processor 사용법

## 🚀 개요
A6000 x2 GPU 환경에 최적화된 커스터마이징 가능한 빠른 포즈 추정 프로세서입니다.

## 📊 성능
- **평균 처리 속도**: 11-12 fps
- **GPU 활용**: A6000 x2 병렬 처리
- **처리 방식**: 라운드로빈 GPU 스케줄링
- **안정성**: 멀티프로세싱 오류 없는 단일 프로세스 구조

## 🔧 주요 파일들

### 1. `simple_fast_multionnx.py` (핵심 엔진)
- 메인 프로세서 클래스
- GPU 병렬 처리 로직
- HDF5 저장 기능

### 2. `custom_fast_multionnx.py` (사용자 인터페이스)
- 커스터마이징 가능한 실행 스크립트
- 대화형 설정 변경
- 사용자 친화적 인터페이스

## 💻 사용 방법

### 기본 실행
```bash
cd /workspace01/team03/data/mmpose/jy
python custom_fast_multionnx.py
```

### 프로그래밍 방식 사용
```python
from simple_fast_multionnx import SimpleFastProcessor

# 프로세서 초기화
processor = SimpleFastProcessor(
    data_root="/workspace01/team03/data/mmpose/jy/data/1.Training",
    output_dir="/workspace01/team03/data/my_output",
    direction="F",
    item_types=["WORD"],
    keypoint_scale=8,
    jpeg_quality=90,
    gpu_ids=[0, 1]
)

# 처리 실행
processor.process_videos_simple(max_videos=10)  # 10개만 처리
```

## ⚙️ 커스터마이징 옵션

### 키포인트 스케일 변경
```python
processor.customize_keypoint_scale(16)  # 기본값: 8
```

### JPEG 품질 변경
```python
processor.customize_jpeg_quality(95)  # 기본값: 90 (1-100)
```

### 출력 디렉토리 변경
```python
processor.customize_output_dir("/path/to/output")
```

## 📁 출력 구조
```
output_directory/
├── WORD0001/
│   ├── frames.h5     # JPEG 인코딩된 프레임
│   └── poses.h5      # 키포인트와 스코어
├── WORD0002/
│   ├── frames.h5
│   └── poses.h5
└── ...
```

## 🔍 모니터링

### 실시간 처리 상태
- 진행률 바를 통한 시각적 모니터링
- GPU별 처리 속도 표시
- 프레임 수와 FPS 정보 제공

### 최종 통계
- 처리 성공률
- 사용된 GPU 정보
- 총 처리 시간

## 🐛 문제 해결

### GPU 메모리 부족
```python
# GPU ID를 하나만 사용
processor = SimpleFastProcessor(gpu_ids=[0])
```

### 처리 속도가 느림
```python
# JPEG 품질 낮춰서 속도 향상
processor.customize_jpeg_quality(70)
```

### 키포인트 정확도 문제
```python
# 키포인트 스케일 높여서 정밀도 향상
processor.customize_keypoint_scale(16)
```

## 📋 시스템 요구사항
- CUDA 지원 GPU (A6000 권장)
- Python 3.10+
- PyTorch with CUDA
- OpenCV, h5py, numpy
- ONNX Runtime with CUDA support

## 🎯 사용 예시

### 빠른 테스트 (5개 비디오)
```bash
echo "5" | python custom_fast_multionnx.py
```

### 전체 데이터셋 처리
```bash
python custom_fast_multionnx.py
# 처리 제한: 엔터 (제한없음)
```

### 고품질 설정
```bash
python custom_fast_multionnx.py
# JPEG 품질: 95
# 키포인트 스케일: 16
```

## ✅ 확인된 성능
- ✅ 39,000개 비디오 인식 성공
- ✅ A6000 x2 GPU 병렬 처리 동작
- ✅ 평균 11-12 fps 안정적 처리
- ✅ HDF5 형태 안전한 데이터 저장
- ✅ 멀티프로세싱 오류 해결됨
- ✅ 커스터마이징 기능 완벽 동작

## 🔧 고급 설정
더 세밀한 설정이 필요한 경우 `SimpleFastProcessor` 클래스를 직접 수정하여 사용하세요.
