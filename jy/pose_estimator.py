#!/usr/bin/env python3
"""
Pose Estimation Module for RTMW-x (MMPose 공식 호환)
RTMW-x WholeBody 포즈 추정기
"""

import torch
import cv2
import numpy as np
import os
from typing import Tuple

from rtmw_model import RTMWXModel
from mmpose_simcc_decoder import SimCCDecoder
from config import (
    RTMW_INPUT_SIZE, RTMW_NUM_KEYPOINTS, RTMW_SIMCC_SPLIT_RATIO,
    IMAGENET_MEAN, IMAGENET_STD
)


class RTMWXEstimator:
    """RTMW-x WholeBody 포즈 추정기 (MMPose 공식 호환)"""
    
    def __init__(self, model_path: str, device: str = 'xpu:0'):
        self.device = device
        self.input_size = RTMW_INPUT_SIZE
        self.num_keypoints = RTMW_NUM_KEYPOINTS
        
        # MMPose 공식 SimCC 디코더 초기화
        self.simcc_decoder = SimCCDecoder(
            input_size=self.input_size,
            simcc_split_ratio=RTMW_SIMCC_SPLIT_RATIO,
            simcc_normalize=True,  # MMPose 공식: 정규화 활성화
            use_dark=False  # 일단 비활성화
        )
        
        # 모델 로드
        self.model = self._load_rtmw_model(model_path)
        self.model.to(device)
        self.model.eval()
        
        # ImageNet 정규화 파라미터
        self.mean = torch.tensor(IMAGENET_MEAN).to(device)
        self.std = torch.tensor(IMAGENET_STD).to(device)
        
        print(f"✅ RTMW-x 모델 초기화 완료 - 입력 크기: {self.input_size}, 키포인트: {self.num_keypoints}개")
        
    def _load_rtmw_model(self, model_path: str):
        """RTMW-x 모델 로드"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"RTMW-x 모델 파일을 찾을 수 없습니다: {model_path}")
        
        # 체크포인트 로드
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            print("RTMW-x 체크포인트 구조 분석:")
            
            # 백본과 헤드 레이어 분석
            backbone_layers = [k for k in state_dict.keys() if k.startswith('backbone.')]
            head_layers = [k for k in state_dict.keys() if k.startswith('head.')]
            
            print(f"백본 레이어 수: {len(backbone_layers)}")
            print(f"헤드 레이어 수: {len(head_layers)}")
            
            # RTMW 헤드 구조 확인
            rtmw_head_keys = ['head.cls_x.weight', 'head.cls_y.weight']
            for key in rtmw_head_keys:
                if key in state_dict:
                    print(f"  {key}: {state_dict[key].shape}")
                    
        else:
            raise RuntimeError("체크포인트에 state_dict가 없습니다.")
        
        # RTMW-x 모델 구조 생성
        model = RTMWXModel.create_from_checkpoint(state_dict)
        
        # state_dict 로드
        try:
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            print(f"✅ RTMW-x 모델 로드 완료")
            if len(missing_keys) > 0:
                print(f"⚠️ 백본 단순화로 인한 누락된 키: {len(missing_keys)}개")
            
        except Exception as e:
            print(f"❌ RTMW-x 모델 로드 실패: {e}")
            raise RuntimeError(f"RTMW-x 모델 로딩 중 오류 발생: {e}")
        
        return model
    
    def preprocess_image(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> torch.Tensor:
        """RTMW-x 입력을 위한 이미지 전처리"""
        x1, y1, x2, y2 = bbox
        
        # 바운딩 박스 영역 크롭
        person_img = image[y1:y2, x1:x2]
        
        # RTMW-x 표준 입력 크기로 리사이즈
        person_img = cv2.resize(person_img, self.input_size)
        
        # BGR -> RGB 변환
        person_img = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)
        
        # 0-1 정규화
        person_img = person_img.astype(np.float32) / 255.0
        
        # 텐서 변환
        tensor_img = torch.from_numpy(person_img).permute(2, 0, 1)  # (C, H, W)
        tensor_img = tensor_img.unsqueeze(0)  # (1, C, H, W)
        
        # XPU로 이동 후 ImageNet 정규화
        tensor_img = tensor_img.to(self.device)
        tensor_img = (tensor_img - self.mean.view(3, 1, 1)) / self.std.view(3, 1, 1)
        
        return tensor_img
    
    def postprocess_keypoints(self, 
                            cls_x_output: torch.Tensor, 
                            cls_y_output: torch.Tensor,
                            bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """MMPose 공식 SimCC 디코더를 사용한 후처리"""
        x1, y1, x2, y2 = bbox
        
        # MMPose 공식 SimCC 디코더로 키포인트 좌표 추출
        keypoints, scores = self.simcc_decoder.decode(cls_x_output, cls_y_output)
        
        # 바운딩 박스 기준으로 원본 이미지 좌표로 변환
        bbox_width = x2 - x1
        bbox_height = y2 - y1
        
        final_keypoints = []
        for i, (kpt, score) in enumerate(zip(keypoints, scores)):
            x, y = kpt
            
            # 정규화된 좌표를 바운딩 박스에 맞춤
            norm_x = x / self.input_size[0]  # 0~1 범위
            norm_y = y / self.input_size[1]  # 0~1 범위
            
            # 좌표 클램핑
            norm_x = np.clip(norm_x, 0.0, 1.0)
            norm_y = np.clip(norm_y, 0.0, 1.0)
            
            final_x = x1 + norm_x * bbox_width
            final_y = y1 + norm_y * bbox_height
            
            final_keypoints.append([final_x, final_y])
        
        return np.array(final_keypoints)
    
    def estimate_pose(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """RTMW-x로 WholeBody 포즈 추정 (MMPose 공식 호환)"""
        input_tensor = self.preprocess_image(image, bbox)
        
        with torch.no_grad():
            cls_x_output, cls_y_output = self.model(input_tensor)
            
            # MMPose 공식 디코더로 후처리
            keypoints = self.postprocess_keypoints(cls_x_output, cls_y_output, bbox)
            
            return keypoints