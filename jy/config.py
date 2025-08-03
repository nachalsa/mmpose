#!/usr/bin/env python3
"""
Configuration Module for RTMW-x
RTMW-x 설정 관리 모듈 - 통합 모델 관리
"""

import os

# 경로 설정
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # /home/ty/rtmw/02
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# 디바이스 설정
DEFAULT_DEVICE = 'xpu:0'

# 모델 설정 - 올바른 288x384 해석 (W=288, H=384)
RTMW_INPUT_SIZE = (288, 384)  # Width=288, Height=384 (MMPose 표준)
RTMW_NUM_KEYPOINTS = 133
RTMW_SIMCC_SPLIT_RATIO = 2.0  # X/Y 공통 split ratio

# 정규화 설정 - MMPose 공식 방식
POSE_MEAN = [123.675, 116.28, 103.53]   # MMPose PoseDataPreprocessor 표준 (RGB, 0-255)
POSE_STD = [58.395, 57.12, 57.375]      # MMPose PoseDataPreprocessor 표준 (RGB, 0-255)

# 기존 ImageNet 정규화 (참고용, 더 이상 사용하지 않음)
IMAGENET_MEAN = [0.485, 0.456, 0.406]   # 0-1 스케일
IMAGENET_STD = [0.229, 0.224, 0.225]    # 0-1 스케일

# =============================================================================
# 통합 모델 관리 - 모든 모델을 models 디렉토리에서 관리
# =============================================================================

# YOLO 모델 설정 - 명확한 경로 지정
YOLO_MODEL_CONFIG = {
    "filename": "yolo11m.pt",
    "path": os.path.join(MODELS_DIR, "yolo11m.pt"),  # 명확한 절대 경로
    "url": None,  # ultralytics에서 자동 다운로드
    "description": "YOLOv11 Medium - 사람 검출용"
}

# RTMW 모델 옵션들
RTMW_MODEL_OPTIONS = [
    {
        "filename": "rtmw-x_simcc-cocktail14_pt-ucoco_270e-384x288-f840f204_20231122.pth",
        "path": os.path.join(MODELS_DIR, "rtmw-x_simcc-cocktail14_pt-ucoco_270e-384x288-f840f204_20231122.pth"),
        "url": "https://download.openmmlab.com/mmpose/v1/projects/rtmw/rtmw-x_simcc-cocktail14_pt-ucoco_270e-384x288-f840f204_20231122.pth",
        "description": "RTMW-x 384x288 (최고 성능)",
        "input_size": (384, 288),
        "keypoints": 133
    },
    {
        "filename": "rtmw-dw-x-l_simcc-cocktail14_270e-384x288-20231122.pth",
        "path": os.path.join(MODELS_DIR, "rtmw-dw-x-l_simcc-cocktail14_270e-384x288-20231122.pth"),
        "url": "https://download.openmmlab.com/mmpose/v1/projects/rtmw/rtmw-dw-x-l_simcc-cocktail14_270e-384x288-20231122.pth",
        "description": "RTMW-l 384x288 (균형)",
        "input_size": (384, 288),
        "keypoints": 133
    }
]

# 편의성을 위한 별칭들
YOLO_MODEL_FILENAME = YOLO_MODEL_CONFIG["filename"]
YOLO_MODEL_PATH = YOLO_MODEL_CONFIG["path"]

# 기본 RTMW 모델 (첫 번째 옵션)
DEFAULT_RTMW_MODEL = RTMW_MODEL_OPTIONS[0]

# 하위 호환성을 위한 기존 변수들
RTMW_MODEL_INFO = {
    model["filename"]: model["description"] 
    for model in RTMW_MODEL_OPTIONS
}

# =============================================================================
# 검출 및 추론 설정
# =============================================================================

# 검출 설정
DEFAULT_CONF_THRESH = 0.5

# FPS 계산 설정
FPS_UPDATE_INTERVAL = 30

# 키포인트 범위 설정 (RTMW-x 133 keypoints)
BODY_KEYPOINTS_RANGE = (0, 17)      # Body: 17개
FACE_KEYPOINTS_RANGE = (17, 85)     # Face: 68개 (17+68=85)
HANDS_KEYPOINTS_RANGE = (85, 133)   # Hands: 48개 (85+48=133)

# =============================================================================
# 유틸리티 함수들
# =============================================================================

def get_model_path(model_type: str, model_name: str = None) -> str:
    """모델 파일 경로 반환"""
    if model_type.lower() == 'yolo':
        return YOLO_MODEL_PATH
    elif model_type.lower() == 'rtmw':
        if model_name:
            for model in RTMW_MODEL_OPTIONS:
                if model_name in model["filename"]:
                    return model["path"]
        return DEFAULT_RTMW_MODEL["path"]
    else:
        raise ValueError(f"지원하지 않는 모델 타입: {model_type}")

def get_yolo_config() -> dict:
    """YOLO 모델 설정 반환"""
    return YOLO_MODEL_CONFIG.copy()

def check_model_exists(model_type: str, model_name: str = None) -> bool:
    """모델 파일 존재 여부 확인"""
    model_path = get_model_path(model_type, model_name)
    return os.path.exists(model_path)

def ensure_models_dir():
    """models 디렉토리 존재 확인 및 생성"""
    os.makedirs(MODELS_DIR, exist_ok=True)
    return MODELS_DIR

# =============================================================================
# 디렉토리 생성
# =============================================================================

# 필요한 디렉토리들 생성
ensure_models_dir()
os.makedirs(DATA_DIR, exist_ok=True)

# 초기화 로그
if __name__ == "__main__":
    print("📁 RTMW-x 설정 정보:")
    print(f"   - 프로젝트 루트: {PROJECT_ROOT}")
    print(f"   - 모델 디렉토리: {MODELS_DIR}")
    print(f"   - 데이터 디렉토리: {DATA_DIR}")
    print(f"   - 기본 디바이스: {DEFAULT_DEVICE}")
    print(f"   - RTMW 입력 크기: {RTMW_INPUT_SIZE}")
    print(f"   - YOLO 모델 경로: {YOLO_MODEL_PATH}")
    
    for i, model in enumerate(RTMW_MODEL_OPTIONS):
        exists = "✅" if os.path.exists(model["path"]) else "❌"
        print(f"     {exists} {model['description']}")