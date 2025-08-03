#!/usr/bin/env python3
"""
Configuration Module for RTMW-x
RTMW-x 설정 관리 모듈
"""

# 디바이스 설정
DEFAULT_DEVICE = 'xpu:0'

# 모델 설정 - 실제 384x288 사용
RTMW_INPUT_SIZE = (384, 288)  # 384x288로 복원
RTMW_NUM_KEYPOINTS = 133
RTMW_SIMCC_SPLIT_RATIO = 2.0

# ImageNet 정규화 파라미터
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# 파일 경로 설정
MODELS_DIR = "./models"
YOLO_MODEL_FILENAME = "yolo11m.pt"

# RTMW 모델 옵션 - 실제 384x288 모델들
RTMW_MODEL_OPTIONS = [
    {
        "filename": "rtmw-x_simcc-cocktail14_pt-ucoco_270e-384x288-f840f204_20231122.pth",
        "url": "https://download.openmmlab.com/mmpose/v1/projects/rtmw/rtmw-x_simcc-cocktail14_pt-ucoco_270e-384x288-f840f204_20231122.pth",
        "description": "RTMW-x 384x288 (최고 성능)"
    },
    {
        "filename": "rtmw-dw-x-l_simcc-cocktail14_270e-384x288-20231122.pth",
        "url": "https://download.openmmlab.com/mmpose/v1/projects/rtmw/rtmw-dw-x-l_simcc-cocktail14_270e-384x288-20231122.pth",
        "description": "RTMW-l 384x288 (균형)"
    },
    {
        "filename": "rtmpose-l_simcc-body7_pt-body7_420e-384x288-3f5a1437_20230504.pth",
        "url": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-l_simcc-body7_pt-body7_420e-384x288-3f5a1437_20230504.pth",
        "description": "RTMPose-l 384x288 (Body7)"
    }
]

# 검출 설정
DEFAULT_CONF_THRESH = 0.5

# FPS 계산 설정
FPS_UPDATE_INTERVAL = 30

# 키포인트 범위 설정
BODY_KEYPOINTS_RANGE = (0, 17)
FACE_KEYPOINTS_RANGE = (17, 85)
HANDS_KEYPOINTS_RANGE = (85, 133)