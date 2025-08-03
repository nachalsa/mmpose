#!/usr/bin/env python3
"""
Utility Functions for RTMW-x
RTMW-x 유틸리티 함수들
"""

import os
import urllib.request
from typing import Optional, Tuple
import torch

from config import RTMW_MODEL_OPTIONS, MODELS_DIR


def download_model(url: str, save_path: str) -> None:
    """모델 다운로드
    
    Args:
        url: 다운로드할 모델 URL
        save_path: 저장할 경로
    """
    if not os.path.exists(save_path):
        print(f"모델 다운로드 중: {url}")
        urllib.request.urlretrieve(url, save_path)
        print(f"다운로드 완료: {save_path}")


def find_and_download_rtmw_model() -> Tuple[Optional[str], Optional[str]]:
    """RTMW 모델 찾기 및 다운로드
    
    Returns:
        Tuple[model_path, model_description]: 모델 경로와 설명
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # 기존 모델 찾기
    for model_option in RTMW_MODEL_OPTIONS:
        temp_path = os.path.join(MODELS_DIR, model_option["filename"])
        
        if os.path.exists(temp_path):
            print(f"✅ 기존 RTMW 모델 발견: {temp_path}")
            return temp_path, model_option["description"]
    
    # 모델 다운로드 시도
    for model_option in RTMW_MODEL_OPTIONS:
        temp_path = os.path.join(MODELS_DIR, model_option["filename"])
        
        print(f"RTMW 모델 다운로드 시도: {model_option['description']}")
        try:
            download_model(model_option["url"], temp_path)
            print(f"✅ RTMW 모델 다운로드 성공: {model_option['description']}")
            return temp_path, model_option["description"]
        except Exception as e:
            print(f"⚠️ 다운로드 실패: {e}")
            if os.path.exists(temp_path):
                os.remove(temp_path)
            continue
    
    return None, None


def check_xpu_availability() -> bool:
    """Intel XPU 가용성 확인
    
    Returns:
        bool: XPU 사용 가능 여부
    """
    if not torch.xpu.is_available():
        print("❌ Intel XPU를 사용할 수 없습니다.")
        return False
    
    print(f"✅ 사용 디바이스: xpu:0")
    print(f"XPU 디바이스: {torch.xpu.get_device_name(0)}")
    return True