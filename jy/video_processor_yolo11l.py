#!/usr/bin/env python3
"""
YOLO11L + RTMW 영상 처리기
영상 입력을 처리하고 모델에 들어가는 형태로 이미지를 잘라서 저장
"""

import os
import torch
import cv2
import numpy as np
import time
from typing import List, Tuple, Optional, Dict, Any
from collections import deque
from pathlib import Path
import json
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap

from yolo11l_xpu_hybrid_inferencer import YOLO11LXPUHybridInferencer

# RTMW 전처리 함수들 (MMPose에서 가져옴)
def bbox_xyxy2cs(bbox: np.ndarray, padding: float = 1.25) -> Tuple[np.ndarray, np.ndarray]:
    """바운딩박스를 center, scale로 변환"""
    dim = bbox.ndim
    if dim == 1:
        bbox = bbox[None, :]
    
    scale = (bbox[..., 2:] - bbox[..., :2]) * padding
    center = (bbox[..., 2:] + bbox[..., :2]) * 0.5
    
    if dim == 1:
        center = center[0]
        scale = scale[0]
    
    return center, scale

def _rotate_point(pt: np.ndarray, angle_rad: float) -> np.ndarray:
    """점을 회전"""
    cos_val = np.cos(angle_rad)
    sin_val = np.sin(angle_rad)
    return np.array([pt[0] * cos_val - pt[1] * sin_val,
                     pt[0] * sin_val + pt[1] * cos_val])

def _get_3rd_point(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """세 번째 점을 계산 (직교점)"""
    direction = a - b
    return b + np.array([-direction[1], direction[0]])

def get_warp_matrix(center: np.ndarray, scale: np.ndarray, rot: float, 
                   output_size: Tuple[int, int]) -> np.ndarray:
    """아핀 변환 매트릭스 계산"""
    src_w, src_h = scale[:2]
    dst_w, dst_h = output_size[:2]
    
    rot_rad = np.deg2rad(rot)
    src_dir = _rotate_point(np.array([src_w * -0.5, 0.]), rot_rad)
    dst_dir = np.array([dst_w * -0.5, 0.])
    
    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center
    src[1, :] = center + src_dir
    
    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir
    
    # aspect ratio 고정
    src[2, :] = _get_3rd_point(src[0, :], src[1, :])
    dst[2, :] = _get_3rd_point(dst[0, :], dst[1, :])
    
    warp_mat = cv2.getAffineTransform(src, dst)
    return warp_mat

def fix_aspect_ratio(bbox_scale: np.ndarray, aspect_ratio: float) -> np.ndarray:
    """bbox를 고정 종횡비로 조정"""
    w, h = bbox_scale[0], bbox_scale[1]
    if w > h * aspect_ratio:
        new_h = w / aspect_ratio
        bbox_scale = np.array([w, new_h])
    else:
        new_w = h * aspect_ratio
        bbox_scale = np.array([new_w, h])
    return bbox_scale

class VideoProcessorYOLO11L:
    """YOLO11L 기반 영상 처리 및 크롭 저장 시스템"""
    
    def __init__(self, 
                 rtmw_config: str,
                 rtmw_checkpoint: str,
                 output_dir: str = "video_outputs",
                 save_crops: bool = True,
                 save_pose_data: bool = True,
                 crop_padding: float = 0.1):
        """
        Args:
            rtmw_config: RTMW 설정 파일 경로
            rtmw_checkpoint: RTMW 체크포인트 경로
            output_dir: 출력 디렉토리
            save_crops: 크롭 이미지 저장 여부
            save_pose_data: 포즈 데이터 저장 여부
            crop_padding: 바운딩박스 패딩 비율
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.save_crops = save_crops
        self.save_pose_data = save_pose_data
        self.crop_padding = crop_padding
        
        # 하위 디렉토리 생성
        if self.save_crops:
            (self.output_dir / "crops").mkdir(exist_ok=True)
            (self.output_dir / "person_crops").mkdir(exist_ok=True)
        
        if self.save_pose_data:
            (self.output_dir / "pose_data").mkdir(exist_ok=True)
        
        (self.output_dir / "processed_frames").mkdir(exist_ok=True)
        (self.output_dir / "metadata").mkdir(exist_ok=True)
        (self.output_dir / "combined_data").mkdir(exist_ok=True)  # 결합 데이터 디렉토리
        
        print(f"📁 출력 디렉토리 설정: {self.output_dir}")
        
        # YOLO11L 추론기 초기화
        self.inferencer = YOLO11LXPUHybridInferencer(
            rtmw_config=rtmw_config,
            rtmw_checkpoint=rtmw_checkpoint,
            detection_device="auto",
            pose_device="auto",
            optimize_for_accuracy=True
        )
        
        # 처리 통계
        self.stats = {
            'total_frames': 0,
            'processed_frames': 0,
            'detected_persons': 0,
            'saved_crops': 0,
            'processing_times': [],
            'detection_times': [],
            'pose_times': []
        }
        
        print(f"✅ YOLO11L 영상 처리기 초기화 완료")
    
    def _expand_bbox(self, bbox: List[float], image_shape: Tuple[int, int], padding: float = None) -> List[int]:
        """바운딩박스를 패딩으로 확장"""
        if padding is None:
            padding = self.crop_padding
            
        h, w = image_shape[:2]
        x1, y1, x2, y2 = bbox
        
        # 바운딩박스 크기 계산
        bbox_w = x2 - x1
        bbox_h = y2 - y1
        
        # 패딩 적용
        pad_w = bbox_w * padding
        pad_h = bbox_h * padding
        
        # 확장된 바운딩박스
        new_x1 = max(0, int(x1 - pad_w))
        new_y1 = max(0, int(y1 - pad_h))
        new_x2 = min(w, int(x2 + pad_w))
        new_y2 = min(h, int(y2 + pad_h))
        
        return [new_x1, new_y1, new_x2, new_y2]
    
    def _crop_person_image(self, image: np.ndarray, bbox: List[float], person_id: int, frame_idx: int) -> np.ndarray:
        """사람 이미지 크롭 - RTMW 모델의 정확한 TopdownAffine 전처리 방식 사용"""
        try:
            # RTMW 설정: width=288, height=384
            input_width, input_height = 288, 384
            
            # 1. bbox를 center, scale로 변환 (padding=1.25 적용)
            bbox_array = np.array(bbox, dtype=np.float32)
            center, scale = bbox_xyxy2cs(bbox_array, padding=1.25)
            
            # 2. aspect ratio 고정 (width/height = 288/384 = 0.75)
            aspect_ratio = input_width / input_height  # 0.75
            scale = fix_aspect_ratio(scale, aspect_ratio)
            
            # 3. 아핀 변환 매트릭스 계산
            warp_mat = get_warp_matrix(
                center=center,
                scale=scale,
                rot=0.0,  # 회전 없음
                output_size=(input_width, input_height)
            )
            
            # 4. 아핀 변환 적용
            cropped_image = cv2.warpAffine(
                image, 
                warp_mat, 
                (input_width, input_height), 
                flags=cv2.INTER_LINEAR
            )
            
            # 5. 크기 검증
            h, w = cropped_image.shape[:2]
            if h == input_height and w == input_width:
                return cropped_image
            else:
                print(f"⚠️ 크기 오류: {h}x{w}, 예상: {input_height}x{input_width}")
                return cropped_image
                
        except Exception as e:
            print(f"⚠️ RTMW 전처리 실패 (Person {person_id}): {e}")
            # 폴백: 단순 크롭 + 리사이즈
            expanded_bbox = self._expand_bbox(bbox, image.shape)
            x1, y1, x2, y2 = expanded_bbox
            cropped = image[y1:y2, x1:x2]
            
            if cropped.size == 0:
                return None
            
            # 288x384 (WxH)로 리사이즈
            return cv2.resize(cropped, (288, 384), interpolation=cv2.INTER_LINEAR)
    
    def _save_crop_image(self, crop_image: np.ndarray, frame_idx: int, person_id: int, bbox: List[float], confidence: float = None) -> str:
        """크롭 이미지 저장"""
        if crop_image is None:
            return None
            
        # 파일명 생성
        timestamp = datetime.now().strftime("%H%M%S")
        if confidence is not None:
            filename = f"frame_{frame_idx:06d}_person_{person_id}_conf_{confidence:.2f}_{timestamp}.jpg"
        else:
            filename = f"frame_{frame_idx:06d}_person_{person_id}_{timestamp}.jpg"
        
        # 저장 경로
        crop_path = self.output_dir / "person_crops" / filename
        
        # 저장
        success = cv2.imwrite(str(crop_path), crop_image)
        
        if success:
            self.stats['saved_crops'] += 1
            return str(crop_path)
        else:
            print(f"❌ 크롭 이미지 저장 실패: {crop_path}")
            return None
    
    def _save_pose_data(self, frame_idx: int, person_id: int, keypoints: np.ndarray, scores: np.ndarray, bbox: List[float]) -> str:
        """포즈 데이터 JSON으로 저장"""
        # 데이터 구성
        pose_data = {
            'frame_idx': frame_idx,
            'person_id': person_id,
            'timestamp': datetime.now().isoformat(),
            'bbox': bbox,
            'keypoints': keypoints.tolist(),
            'scores': scores.tolist(),
            'stats': {
                'total_keypoints': len(keypoints),
                'valid_keypoints': int(np.sum(scores > 0.3)),
                'high_confidence_keypoints': int(np.sum(scores > 0.8)),
                'average_confidence': float(np.mean(scores)),
                'max_confidence': float(np.max(scores)),
                'min_confidence': float(np.min(scores))
            }
        }
        
        # 파일명 생성
        filename = f"frame_{frame_idx:06d}_person_{person_id}_pose.json"
        pose_path = self.output_dir / "pose_data" / filename
        
        # JSON 저장
        try:
            with open(pose_path, 'w', encoding='utf-8') as f:
                json.dump(pose_data, f, indent=2, ensure_ascii=False)
            return str(pose_path)
        except Exception as e:
            print(f"❌ 포즈 데이터 저장 실패: {e}")
            return None
    
    def _save_metadata(self, frame_idx: int, original_image_shape: Tuple[int, int], results: List[Tuple], processing_time: float):
        """프레임 메타데이터 저장"""
        metadata = {
            'frame_idx': frame_idx,
            'timestamp': datetime.now().isoformat(),
            'original_shape': {
                'height': original_image_shape[0],
                'width': original_image_shape[1],
                'channels': original_image_shape[2] if len(original_image_shape) > 2 else 1
            },
            'processing_time': processing_time,
            'detected_persons': len(results),
            'persons': []
        }
        
        # 각 사람 정보
        for i, (keypoints, scores, bbox) in enumerate(results):
            person_info = {
                'person_id': i,
                'bbox': bbox,
                'bbox_area': (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]),
                'keypoint_stats': {
                    'total': len(scores),
                    'valid': int(np.sum(scores > 0.3)),
                    'high_confidence': int(np.sum(scores > 0.8)),
                    'avg_confidence': float(np.mean(scores))
                }
            }
            metadata['persons'].append(person_info)
        
        # 파일 저장
        filename = f"frame_{frame_idx:06d}_metadata.json"
        metadata_path = self.output_dir / "metadata" / filename
        
        try:
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"❌ 메타데이터 저장 실패: {e}")
    
    def _save_combined_data(self, frame_idx: int, person_id: int, crop_image: np.ndarray, 
                           keypoints: np.ndarray, scores: np.ndarray, bbox: List[float], 
                           original_image_shape: Tuple[int, int] = None) -> str:
        """이미지와 포즈 데이터를 결합한 파일 저장 - RTMW 정확한 전처리 기반"""
        try:
            # RTMW 입력 크기: width=288, height=384
            input_width, input_height = 288, 384
            
            # bbox를 center, scale로 변환 (RTMW와 동일한 방식)
            bbox_array = np.array(bbox, dtype=np.float32)
            center, scale = bbox_xyxy2cs(bbox_array, padding=1.25)
            
            # aspect ratio 고정
            aspect_ratio = input_width / input_height  # 0.75
            scale = fix_aspect_ratio(scale, aspect_ratio)
            
            # 결합 데이터 구성
            combined_data = {
                'frame_info': {
                    'frame_idx': frame_idx,
                    'person_id': person_id,
                    'timestamp': datetime.now().isoformat(),
                    'bbox': bbox
                },
                'crop_transform': {
                    'bbox_center': center.tolist(),
                    'bbox_scale': scale.tolist(),
                    'input_size': [input_width, input_height],  # [288, 384]
                    'aspect_ratio': aspect_ratio,
                    'padding_factor': 1.25
                },
                'pose_data': {
                    # keypoints는 크롭 이미지 좌표계에서의 x,y,score만 저장
                    'keypoints_2d': [[float(kpt[0]), float(kpt[1]), float(score)] 
                                   for kpt, score in zip(keypoints, scores)],
                    'stats': {
                        'total_keypoints': len(keypoints),
                        'valid_keypoints': int(np.sum(scores > 0.3)),
                        'high_confidence_keypoints': int(np.sum(scores > 0.8)),
                        'average_confidence': float(np.mean(scores)),
                        'max_confidence': float(np.max(scores)),
                        'min_confidence': float(np.min(scores))
                    }
                },
                'image_info': {
                    'shape': list(crop_image.shape),  # [height, width, channels]
                    'dtype': str(crop_image.dtype),
                    'size_bytes': crop_image.nbytes
                }
            }
            
            # 파일명 생성
            base_filename = f"frame_{frame_idx:06d}_person_{person_id}_combined"
            
            # JSON 저장 (메타데이터)
            json_path = self.output_dir / "combined_data" / f"{base_filename}.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(combined_data, f, indent=2, ensure_ascii=False)
            
            # 크롭 이미지 저장 (RTMW 전처리된 이미지)
            npy_path = self.output_dir / "combined_data" / f"{base_filename}_image.npy"
            np.save(npy_path, crop_image)
            
            # 포즈 데이터 저장 (x, y, score)
            pose_npy_path = self.output_dir / "combined_data" / f"{base_filename}_pose.npy"
            pose_array = np.column_stack([keypoints, scores.reshape(-1, 1)])  # (133, 3) - [x, y, score]
            np.save(pose_npy_path, pose_array)
            
            # 시각화용 JPEG 저장
            vis_path = self.output_dir / "combined_data" / f"{base_filename}_visualization.jpg"
            self._save_pose_visualization(crop_image, keypoints, scores, vis_path)
            
            # 통합 크롭 이미지도 JPEG로 저장
            crop_img_path = self.output_dir / "combined_data" / f"{base_filename}_crop.jpg"
            cv2.imwrite(str(crop_img_path), crop_image)
            
            return str(json_path)
            
        except Exception as e:
            print(f"❌ 결합 데이터 저장 실패: {e}")
            return None
    
    def _transform_keypoints_to_crop_coords(self, keypoints: np.ndarray, crop_transform: dict) -> np.ndarray:
        """원본 이미지 좌표계의 키포인트를 크롭 이미지 좌표계로 변환"""
        if crop_transform is None:
            return keypoints
        
        # 변환 정보 추출
        expanded_bbox = crop_transform.get('expanded_bbox', [0, 0, 384, 384])
        padding = crop_transform.get('padding', [0, 0])
        resize_ratio = crop_transform.get('resize_ratio', 1.0)
        
        x1, y1, x2, y2 = expanded_bbox
        pad_w, pad_h = padding
        
        # 좌표 변환 수행
        transformed_keypoints = keypoints.copy()
        
        # 1. 크롭 영역으로 좌표 이동 (원본 -> 크롭)
        transformed_keypoints[:, 0] = keypoints[:, 0] - x1
        transformed_keypoints[:, 1] = keypoints[:, 1] - y1
        
        # 2. 정사각형 패딩 적용
        transformed_keypoints[:, 0] = transformed_keypoints[:, 0] + pad_w
        transformed_keypoints[:, 1] = transformed_keypoints[:, 1] + pad_h
        
        # 3. 384x384로 리사이즈 비율 적용
        transformed_keypoints[:, 0] = transformed_keypoints[:, 0] * resize_ratio
        transformed_keypoints[:, 1] = transformed_keypoints[:, 1] * resize_ratio
        
        return transformed_keypoints
    
    def _process_person_with_crop_pose(self, image: np.ndarray, bbox: List[float], person_id: int, frame_idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """크롭된 이미지에서 포즈 추정을 수행하여 일관된 좌표계 사용"""
        # 1. 크롭 이미지 생성
        crop_image = self._crop_person_image(image, bbox, person_id, frame_idx)
        if crop_image is None:
            return None, None, None
        
        # 2. 크롭된 이미지에서 직접 포즈 추정
        keypoints, scores = self.inferencer.estimate_pose_on_crop(crop_image)
        
        print(f"🎯 크롭 이미지 포즈 추정: Person {person_id}, 유효 키포인트: {np.sum(scores > 0.3)}/133")
        
        return crop_image, keypoints, scores
    
    def load_combined_data(self, combined_json_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        """결합 데이터 로드"""
        try:
            # JSON 메타데이터 로드
            with open(combined_json_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            # 파일 경로 추출
            base_path = Path(combined_json_path).parent / Path(combined_json_path).stem
            
            # 이미지 로드
            image_path = str(base_path) + "_image.npy"
            crop_image = np.load(image_path)
            
            # 포즈 데이터 로드
            pose_path = str(base_path) + "_pose.npy"
            pose_data = np.load(pose_path)  # (133, 3) - [x, y, score]
            
            keypoints = pose_data[:, :2]  # (133, 2)
            scores = pose_data[:, 2]      # (133,)
            
            return crop_image, keypoints, scores, metadata
            
        except Exception as e:
            print(f"❌ 결합 데이터 로드 실패: {e}")
            return None, None, None, None
    
    def create_dataset_from_combined_data(self, combined_data_dir: str = None) -> dict:
        """결합 데이터들로부터 데이터셋 생성"""
        if combined_data_dir is None:
            combined_data_dir = self.output_dir / "combined_data"
        
        combined_data_dir = Path(combined_data_dir)
        
        if not combined_data_dir.exists():
            print(f"❌ 결합 데이터 디렉토리 없음: {combined_data_dir}")
            return None
        
        # JSON 파일들 찾기
        json_files = list(combined_data_dir.glob("*_combined.json"))
        
        if not json_files:
            print(f"❌ 결합 데이터 파일 없음: {combined_data_dir}")
            return None
        
        print(f"📊 {len(json_files)}개의 결합 데이터 파일 발견")
        
        # 데이터셋 생성
        dataset = {
            'images': [],
            'keypoints': [],
            'scores': [],
            'metadata': [],
            'info': {
                'total_samples': len(json_files),
                'created_at': datetime.now().isoformat(),
                'source_dir': str(combined_data_dir)
            }
        }
        
        for json_file in sorted(json_files):
            crop_image, keypoints, scores, metadata = self.load_combined_data(str(json_file))
            
            if crop_image is not None:
                dataset['images'].append(crop_image)
                dataset['keypoints'].append(keypoints)
                dataset['scores'].append(scores)
                dataset['metadata'].append(metadata)
        
        # 넘파이 배열로 변환
        if dataset['images']:
            dataset['images'] = np.array(dataset['images'])
            dataset['keypoints'] = np.array(dataset['keypoints'])
            dataset['scores'] = np.array(dataset['scores'])
        
        # 데이터셋 저장
        dataset_path = combined_data_dir / "complete_dataset.npy"
        np.save(dataset_path, dataset)
        
        # 메타정보 저장
        info_path = combined_data_dir / "dataset_info.json"
        with open(info_path, 'w', encoding='utf-8') as f:
            dataset_info = dataset.copy()
            # 넘파이 배열은 정보만 저장
            if isinstance(dataset_info['images'], np.ndarray):
                dataset_info['images'] = f"numpy array shape: {dataset_info['images'].shape}"
                dataset_info['keypoints'] = f"numpy array shape: {dataset_info['keypoints'].shape}"
                dataset_info['scores'] = f"numpy array shape: {dataset_info['scores'].shape}"
            json.dump(dataset_info, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 데이터셋 생성 완료:")
        print(f"   - 이미지: {dataset['images'].shape if isinstance(dataset['images'], np.ndarray) else len(dataset['images'])}")
        print(f"   - 키포인트: {dataset['keypoints'].shape if isinstance(dataset['keypoints'], np.ndarray) else len(dataset['keypoints'])}")
        print(f"   - 저장 경로: {dataset_path}")
        
        return dataset
    
    def visualize_combined_data(self, combined_json_path: str, save_path: str = None, show_confidence: bool = True) -> str:
        """결합 데이터를 시각화 (이미지 + 포즈 오버레이) - 크롭 좌표계 직접 사용"""
        try:
            # 결합 데이터 로드
            crop_image, keypoints, scores, metadata = self.load_combined_data(combined_json_path)
            
            if crop_image is None:
                print(f"❌ 결합 데이터 로드 실패: {combined_json_path}")
                return None
            
            # 크롭 좌표계 정보 확인
            crop_transform = metadata.get('crop_transform', None)
            if crop_transform:
                print(f"🎯 크롭 좌표계 데이터 사용 (변환 불필요)")
            else:
                print(f"⚠️ 크롭 변환 정보 없음 - 레거시 데이터일 수 있음")
            
            # 시각화 생성
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'Combined Data Visualization - Frame {metadata["frame_info"]["frame_idx"]}, Person {metadata["frame_info"]["person_id"]}', fontsize=16)
            
            # 1. 원본 크롭 이미지
            ax1 = axes[0, 0]
            ax1.imshow(cv2.cvtColor(crop_image, cv2.COLOR_BGR2RGB))
            ax1.set_title('Original Crop Image (288x384 - HxW)')
            ax1.axis('off')
            
            # 2. 포즈 키포인트 오버레이 (크롭 좌표계 직접 사용)
            ax2 = axes[0, 1]
            ax2.imshow(cv2.cvtColor(crop_image, cv2.COLOR_BGR2RGB))
            
            # 키포인트 그리기 (크롭 좌표계 직접 사용)
            valid_count = 0
            for i, (kpt, score) in enumerate(zip(keypoints, scores)):
                if score > 0.3:  # 유효한 키포인트만
                    x, y = kpt[0], kpt[1]
                    
                    # 이미지 범위 내에 있는지 확인 (288x384)
                    if 0 <= x <= 384 and 0 <= y <= 288:
                        valid_count += 1
                        
                        # 신뢰도에 따른 색상 및 크기
                        if score > 0.8:
                            color = 'lime'
                            size = 30
                        elif score > 0.6:
                            color = 'yellow'
                            size = 20
                        else:
                            color = 'red'
                            size = 15
                        
                        ax2.scatter(x, y, c=color, s=size, alpha=0.8, edgecolors='black', linewidth=1)
                        
                        # 신뢰도 표시
                        if show_confidence and score > 0.7:
                            ax2.annotate(f'{score:.2f}', (x, y), xytext=(5, 5), 
                                       textcoords='offset points', fontsize=8, 
                                       color='white', weight='bold',
                                       bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))
            
            ax2.set_title(f'Pose Keypoints Overlay\n(Valid: {valid_count}/133, RTMW coords)')
            ax2.axis('off')
            
            # 3. 신뢰도 히트맵 (크롭 좌표계 직접 사용)
            ax3 = axes[1, 0]
            
            # 키포인트를 이미지 위에 히트맵으로 표시 (288x384)
            heatmap = np.zeros((288, 384))
            for kpt, score in zip(keypoints, scores):
                if score > 0.3:
                    x, y = int(kpt[0]), int(kpt[1])
                    if 0 <= x < 384 and 0 <= y < 288:
                        # 가우시안 블러로 히트맵 생성
                        size = int(score * 15)  # 더 작은 히트맵 사이즈
                        y_start, y_end = max(0, y-size), min(288, y+size)
                        x_start, x_end = max(0, x-size), min(384, x+size)
                        heatmap[y_start:y_end, x_start:x_end] = np.maximum(
                            heatmap[y_start:y_end, x_start:x_end], score
                        )
            
            im3 = ax3.imshow(heatmap, cmap='jet', alpha=0.7, vmin=0, vmax=1)
            ax3.imshow(cv2.cvtColor(crop_image, cv2.COLOR_BGR2RGB), alpha=0.3)
            ax3.set_title('Confidence Heatmap (Crop coords)')
            ax3.axis('off')
            plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
            
            # 4. 통계 정보
            ax4 = axes[1, 1]
            ax4.axis('off')
            
            # 통계 텍스트 생성
            stats = metadata["pose_data"]["stats"]
            bbox = metadata["frame_info"]["bbox"]
            
            stats_text = f"""
📊 Pose Statistics:
• Total Keypoints: {stats['total_keypoints']}
• Valid (>0.3): {stats['valid_keypoints']}
• High Confidence (>0.8): {stats['high_confidence_keypoints']}
• Average Confidence: {stats['average_confidence']:.3f}
• Max Confidence: {stats['max_confidence']:.3f}
• Min Confidence: {stats['min_confidence']:.3f}

📷 Image Info:
• Crop Size: {crop_image.shape[1]}x{crop_image.shape[0]} (WxH)
• RTMW Format: 384x288 (WxH)
• Channels: {crop_image.shape[2]}
• Data Type: {crop_image.dtype}

🎯 Coordinate System:
• Type: RTMW Model Coordinates
• X Range: 0-384 pixels
• Y Range: 0-288 pixels
• Note: Direct model input coords

�📐 Original Bounding Box:
• X: {bbox[0]:.1f} ~ {bbox[2]:.1f}
• Y: {bbox[1]:.1f} ~ {bbox[3]:.1f}
• Width: {bbox[2]-bbox[0]:.1f}
• Height: {bbox[3]-bbox[1]:.1f}

⏰ Timestamp:
{metadata['frame_info']['timestamp']}
            """
            
            ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, fontsize=11,
                    verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
            
            # 저장 경로 결정
            if save_path is None:
                base_name = Path(combined_json_path).stem
                save_path = Path(combined_json_path).parent / f"{base_name}_visualization.png"
            
            # 저장
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✅ 시각화 저장: {save_path}")
            return str(save_path)
            
        except Exception as e:
            print(f"❌ 시각화 실패: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def create_combined_data_summary(self, combined_data_dir: str = None) -> str:
        """결합 데이터 전체 요약 시각화"""
        if combined_data_dir is None:
            combined_data_dir = self.output_dir / "combined_data"
        
        combined_data_dir = Path(combined_data_dir)
        json_files = list(combined_data_dir.glob("*_combined.json"))
        
        if not json_files:
            print(f"❌ 결합 데이터 파일 없음: {combined_data_dir}")
            return None
        
        print(f"📊 {len(json_files)}개 결합 데이터 분석 중...")
        
        # 데이터 수집
        all_confidences = []
        all_valid_kpts = []
        all_high_conf_kpts = []
        frame_indices = []
        person_ids = []
        
        for json_file in sorted(json_files):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                
                stats = metadata["pose_data"]["stats"]
                all_confidences.append(stats["average_confidence"])
                all_valid_kpts.append(stats["valid_keypoints"])
                all_high_conf_kpts.append(stats["high_confidence_keypoints"])
                frame_indices.append(metadata["frame_info"]["frame_idx"])
                person_ids.append(metadata["frame_info"]["person_id"])
                
            except Exception as e:
                print(f"⚠️ 파일 읽기 실패: {json_file} - {e}")
                continue
        
        if not all_confidences:
            print("❌ 유효한 데이터가 없습니다.")
            return None
        
        # 요약 시각화 생성
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Combined Data Summary ({len(json_files)} samples)', fontsize=16)
        
        # 1. 신뢰도 분포
        ax1 = axes[0, 0]
        ax1.hist(all_confidences, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_title('Average Confidence Distribution')
        ax1.set_xlabel('Confidence Score')
        ax1.set_ylabel('Frequency')
        ax1.axvline(np.mean(all_confidences), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(all_confidences):.3f}')
        ax1.legend()
        
        # 2. 유효 키포인트 수 분포
        ax2 = axes[0, 1]
        ax2.hist(all_valid_kpts, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
        ax2.set_title('Valid Keypoints Distribution')
        ax2.set_xlabel('Number of Valid Keypoints')
        ax2.set_ylabel('Frequency')
        ax2.axvline(np.mean(all_valid_kpts), color='red', linestyle='--',
                   label=f'Mean: {np.mean(all_valid_kpts):.1f}')
        ax2.legend()
        
        # 3. 고신뢰도 키포인트 수 분포
        ax3 = axes[0, 2]
        ax3.hist(all_high_conf_kpts, bins=20, alpha=0.7, color='orange', edgecolor='black')
        ax3.set_title('High Confidence Keypoints Distribution')
        ax3.set_xlabel('Number of High Conf. Keypoints')
        ax3.set_ylabel('Frequency')
        ax3.axvline(np.mean(all_high_conf_kpts), color='red', linestyle='--',
                   label=f'Mean: {np.mean(all_high_conf_kpts):.1f}')
        ax3.legend()
        
        # 4. 시간별 신뢰도 변화
        ax4 = axes[1, 0]
        ax4.plot(frame_indices, all_confidences, 'o-', alpha=0.7, markersize=3)
        ax4.set_title('Confidence Over Time')
        ax4.set_xlabel('Frame Index')
        ax4.set_ylabel('Average Confidence')
        ax4.grid(True, alpha=0.3)
        
        # 5. 유효 키포인트 수 시간 변화
        ax5 = axes[1, 1]
        ax5.plot(frame_indices, all_valid_kpts, 'o-', alpha=0.7, markersize=3, color='green')
        ax5.set_title('Valid Keypoints Over Time')
        ax5.set_xlabel('Frame Index')
        ax5.set_ylabel('Number of Valid Keypoints')
        ax5.grid(True, alpha=0.3)
        
        # 6. 통계 요약
        ax6 = axes[1, 2]
        ax6.axis('off')
        
        summary_text = f"""
📊 Dataset Summary Statistics:

🎯 Confidence Scores:
  • Mean: {np.mean(all_confidences):.3f}
  • Std: {np.std(all_confidences):.3f}
  • Min: {np.min(all_confidences):.3f}
  • Max: {np.max(all_confidences):.3f}

🔢 Valid Keypoints:
  • Mean: {np.mean(all_valid_kpts):.1f}
  • Std: {np.std(all_valid_kpts):.1f}
  • Min: {np.min(all_valid_kpts)}
  • Max: {np.max(all_valid_kpts)}

⭐ High Conf. Keypoints:
  • Mean: {np.mean(all_high_conf_kpts):.1f}
  • Std: {np.std(all_high_conf_kpts):.1f}
  • Min: {np.min(all_high_conf_kpts)}
  • Max: {np.max(all_high_conf_kpts)}

📝 Dataset Info:
  • Total Samples: {len(json_files)}
  • Frame Range: {min(frame_indices)} ~ {max(frame_indices)}
  • Unique Persons: {len(set(person_ids))}
        """
        
        ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
        
        # 저장
        summary_path = combined_data_dir / "dataset_summary_visualization.png"
        plt.tight_layout()
        plt.savefig(summary_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 요약 시각화 저장: {summary_path}")
        return str(summary_path)
    
    def process_video_file(self, video_path: str, 
                          start_frame: int = 0, 
                          end_frame: int = None,
                          frame_interval: int = 1,
                          show_progress: bool = True) -> Dict[str, Any]:
        """비디오 파일 처리"""
        print(f"\n=== YOLO11L 영상 처리 시작: {os.path.basename(video_path)} ===")
        
        if not os.path.exists(video_path):
            print(f"❌ 비디오 파일 없음: {video_path}")
            return None
        
        # 비디오 캡처 초기화
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ 비디오 열기 실패: {video_path}")
            return None
        
        # 비디오 정보
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"📹 비디오 정보: {width}x{height}, {fps:.1f}fps, {total_frames}프레임")
        
        # 처리 범위 설정
        if end_frame is None:
            end_frame = total_frames
        end_frame = min(end_frame, total_frames)
        
        print(f"🎯 처리 범위: {start_frame} ~ {end_frame} (간격: {frame_interval})")
        
        # 시작 프레임으로 이동
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # 처리 시작
        process_start_time = time.time()
        
        try:
            for frame_idx in range(start_frame, end_frame, frame_interval):
                # 프레임 읽기
                ret, frame = cap.read()
                if not ret:
                    print(f"⚠️ 프레임 {frame_idx} 읽기 실패")
                    break
                
                self.stats['total_frames'] += 1
                
                # 프레임 처리
                frame_start_time = time.time()
                vis_frame, results = self.inferencer.process_frame(frame)
                frame_processing_time = time.time() - frame_start_time
                
                self.stats['processing_times'].append(frame_processing_time)
                self.stats['processed_frames'] += 1
                self.stats['detected_persons'] += len(results)
                
                # 결과 저장
                if results:
                    for person_id, (original_keypoints, original_scores, bbox) in enumerate(results):
                        # 크롭 이미지에서 포즈 추정 수행 (일관된 좌표계)
                        crop_image, crop_keypoints, crop_scores = self._process_person_with_crop_pose(frame, bbox, person_id, frame_idx)
                        
                        if crop_image is not None:
                            # 크롭 이미지 저장
                            if self.save_crops:
                                crop_path = self._save_crop_image(crop_image, frame_idx, person_id, bbox)
                            
                            # 포즈 데이터 저장 (크롭 좌표계)
                            if self.save_pose_data:
                                pose_path = self._save_pose_data(frame_idx, person_id, crop_keypoints, crop_scores, bbox)
                            
                            # 결합 데이터 저장 (크롭과 포즈 모두 크롭 좌표계)
                            if self.save_pose_data:
                                combined_path = self._save_combined_data(frame_idx, person_id, crop_image, crop_keypoints, crop_scores, bbox, frame.shape)
                
                # 처리된 프레임 저장
                processed_frame_path = self.output_dir / "processed_frames" / f"frame_{frame_idx:06d}_processed.jpg"
                cv2.imwrite(str(processed_frame_path), vis_frame)
                
                # 메타데이터 저장
                self._save_metadata(frame_idx, frame.shape, results, frame_processing_time)
                
                # 진행률 표시
                if show_progress and (frame_idx - start_frame) % 30 == 0:
                    progress = (frame_idx - start_frame) / (end_frame - start_frame) * 100
                    avg_fps = 1.0 / np.mean(self.stats['processing_times'][-30:]) if self.stats['processing_times'] else 0
                    print(f"📊 진행률: {progress:.1f}% | 프레임 {frame_idx}/{end_frame} | "
                          f"검출: {len(results)}명 | 처리속도: {avg_fps:.1f}fps")
                
                # 프레임 간격만큼 건너뛰기
                if frame_interval > 1:
                    for _ in range(frame_interval - 1):
                        cap.read()
        
        except KeyboardInterrupt:
            print("\n⏹️ 사용자가 처리를 중단했습니다.")
        
        finally:
            cap.release()
            
            # 처리 완료
            total_processing_time = time.time() - process_start_time
            
            # 최종 통계 저장
            final_stats = {
                'video_info': {
                    'path': video_path,
                    'total_frames': total_frames,
                    'fps': fps,
                    'resolution': f"{width}x{height}"
                },
                'processing_info': {
                    'start_frame': start_frame,
                    'end_frame': end_frame,
                    'frame_interval': frame_interval,
                    'processed_frames': self.stats['processed_frames'],
                    'total_processing_time': total_processing_time,
                    'average_fps': self.stats['processed_frames'] / total_processing_time if total_processing_time > 0 else 0
                },
                'detection_stats': {
                    'total_persons_detected': self.stats['detected_persons'],
                    'average_persons_per_frame': self.stats['detected_persons'] / self.stats['processed_frames'] if self.stats['processed_frames'] > 0 else 0,
                    'saved_crops': self.stats['saved_crops']
                },
                'performance': {
                    'avg_processing_time': np.mean(self.stats['processing_times']) if self.stats['processing_times'] else 0,
                    'min_processing_time': np.min(self.stats['processing_times']) if self.stats['processing_times'] else 0,
                    'max_processing_time': np.max(self.stats['processing_times']) if self.stats['processing_times'] else 0
                },
                'output_paths': {
                    'crops': str(self.output_dir / "person_crops") if self.save_crops else None,
                    'pose_data': str(self.output_dir / "pose_data") if self.save_pose_data else None,
                    'combined_data': str(self.output_dir / "combined_data"),
                    'processed_frames': str(self.output_dir / "processed_frames"),
                    'metadata': str(self.output_dir / "metadata")
                }
            }
            
            # 최종 통계 저장
            stats_path = self.output_dir / f"final_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(stats_path, 'w', encoding='utf-8') as f:
                json.dump(final_stats, f, indent=2, ensure_ascii=False)
            
            # 결과 출력
            print(f"\n✅ 영상 처리 완료!")
            print(f"📊 처리 통계:")
            print(f"   - 처리 프레임: {self.stats['processed_frames']}/{total_frames}")
            print(f"   - 총 처리시간: {total_processing_time:.1f}초")
            print(f"   - 평균 처리속도: {final_stats['processing_info']['average_fps']:.1f}fps")
            print(f"   - 검출된 사람: {self.stats['detected_persons']}명")
            print(f"   - 저장된 크롭: {self.stats['saved_crops']}개")
            print(f"📁 출력 경로: {self.output_dir}")
            
            return final_stats
    
    def process_webcam_recording(self, camera_id: int = 0, 
                               duration_seconds: int = 30,
                               save_interval: int = 5) -> Dict[str, Any]:
        """웹캠 녹화 및 실시간 처리"""
        print(f"\n=== YOLO11L 웹캠 녹화 처리 (카메라 ID: {camera_id}) ===")
        print(f"⏱️ 녹화 시간: {duration_seconds}초, 저장 간격: {save_interval}프레임")
        
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print(f"❌ 웹캠 열기 실패 (카메라 ID: {camera_id})")
            return None
        
        # 웹캠 설정 - 30fps 강제 설정
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        
        # FPS 강제 설정 재시도
        for _ in range(3):
            cap.set(cv2.CAP_PROP_FPS, 30)
            current_fps = cap.get(cv2.CAP_PROP_FPS)
            if current_fps >= 29:
                break
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"📹 웹캠 설정: {width}x{height}, {fps:.1f}fps")
        print(f"🎮 ESC: 종료, SPACE: 일시정지, S: 즉시 저장")
        
        # 녹화 시작
        frame_count = 0
        start_time = time.time()
        paused = False
        
        try:
            while True:
                current_time = time.time()
                elapsed_time = current_time - start_time
                
                # 시간 제한 확인
                if elapsed_time >= duration_seconds:
                    print(f"⏱️ 설정된 시간({duration_seconds}초) 완료")
                    break
                
                if not paused:
                    ret, frame = cap.read()
                    if not ret:
                        print("❌ 프레임 읽기 실패")
                        break
                    
                    # 프레임 처리
                    vis_frame, results = self.inferencer.process_frame(frame)
                    
                    # 저장 여부 결정
                    save_this_frame = (frame_count % save_interval == 0) or results
                    
                    if save_this_frame and results:
                        for person_id, (original_keypoints, original_scores, bbox) in enumerate(results):
                            # 크롭 이미지에서 포즈 추정 수행 (일관된 좌표계)
                            crop_image, crop_keypoints, crop_scores = self._process_person_with_crop_pose(frame, bbox, person_id, frame_count)
                            
                            if crop_image is not None:
                                # 크롭 저장
                                if self.save_crops:
                                    self._save_crop_image(crop_image, frame_count, person_id, bbox)
                                
                                # 포즈 데이터 저장 (크롭 좌표계)
                                if self.save_pose_data:
                                    self._save_pose_data(frame_count, person_id, crop_keypoints, crop_scores, bbox)
                                
                                # 결합 데이터 저장
                                if self.save_pose_data:
                                    self._save_combined_data(frame_count, person_id, crop_image, crop_keypoints, crop_scores, bbox, frame.shape)
                    
                    # 정보 표시
                    remaining_time = duration_seconds - elapsed_time
                    cv2.putText(vis_frame, f"Recording: {remaining_time:.1f}s left", 
                               (10, height - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.putText(vis_frame, f"Frame: {frame_count} | Saved: {self.stats['saved_crops']}", 
                               (10, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    frame_count += 1
                    self.stats['total_frames'] += 1
                    self.stats['processed_frames'] += 1
                    self.stats['detected_persons'] += len(results)
                
                # 화면 표시
                cv2.imshow('YOLO11L Webcam Recording', vis_frame)
                
                # 키 입력
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    break
                elif key == ord(' '):  # 일시정지
                    paused = not paused
                    print(f"⏸️ {'일시정지' if paused else '재생'}")
                elif key == ord('s') or key == ord('S'):  # 즉시 저장
                    if results:
                        for person_id, (original_keypoints, original_scores, bbox) in enumerate(results):
                            # 크롭 이미지에서 포즈 추정 수행 (일관된 좌표계)
                            crop_image, crop_keypoints, crop_scores = self._process_person_with_crop_pose(frame, bbox, person_id, frame_count)
                            
                            if crop_image is not None:
                                if self.save_crops:
                                    path = self._save_crop_image(crop_image, frame_count, person_id, bbox)
                                    print(f"📸 즉시 저장: {os.path.basename(path)}")
                                
                                # 포즈 데이터 즉시 저장 (크롭 좌표계)
                                if self.save_pose_data:
                                    pose_path = self._save_pose_data(frame_count, person_id, crop_keypoints, crop_scores, bbox)
                                    if pose_path:
                                        print(f"📊 포즈 데이터 저장: {os.path.basename(pose_path)}")
                                
                                # 결합 데이터 즉시 저장
                                if self.save_pose_data:
                                    combined_path = self._save_combined_data(frame_count, person_id, crop_image, crop_keypoints, crop_scores, bbox, frame.shape)
                                    if combined_path:
                                        print(f"🔗 결합 데이터 저장: {os.path.basename(combined_path)}")
        
        except KeyboardInterrupt:
            print("\n⏹️ 사용자가 녹화를 중단했습니다.")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            
            total_time = time.time() - start_time
            print(f"\n✅ 웹캠 녹화 완료!")
            print(f"   - 총 시간: {total_time:.1f}초")
            print(f"   - 처리 프레임: {frame_count}")
            print(f"   - 저장된 크롭: {self.stats['saved_crops']}개")
            
            return {
                'total_time': total_time,
                'frames_processed': frame_count,
                'crops_saved': self.stats['saved_crops'],
                'persons_detected': self.stats['detected_persons']
            }

def main():
    """메인 실행 함수"""
    # 모델 경로 설정
    rtmw_config = "../configs/wholebody_2d_keypoint/rtmpose/cocktail14/rtmw-x_8xb320-270e_cocktail14-384x288.py"
    rtmw_checkpoint = "../models/rtmw-x_simcc-cocktail14_pt-ucoco_270e-384x288-f840f204_20231122.pth"
    
    try:
        print("🎬 YOLO11L 영상 처리기 테스트")
        print("=" * 60)
        
        # 영상 처리기 생성
        processor = VideoProcessorYOLO11L(
            rtmw_config=rtmw_config,
            rtmw_checkpoint=rtmw_checkpoint,
            output_dir="video_processing_outputs",
            save_crops=True,
            save_pose_data=True,
            crop_padding=0.15
        )
        
        print("\n처리 모드를 선택하세요:")
        print("1. 비디오 파일 처리")
        print("2. 웹캠 실시간 녹화 처리 (30fps)")
        print("3. 테스트 이미지로 크롭 테스트")
        print("4. 결합 데이터셋 생성 (기존 결합 데이터에서)")
        print("5. 결합 데이터 로드 테스트")
        print("6. 결합 데이터 시각화 (단일 파일)")
        print("7. 결합 데이터 전체 요약 시각화")
        print("8. 결합 데이터 구조 설명")
        
        choice = input("선택 (1-8): ").strip()
        
        if choice == "1":
            # 비디오 파일 처리
            video_path = input("비디오 파일 경로를 입력하세요: ").strip()
            if not video_path:
                print("⚠️ 기본 테스트 비디오를 찾는 중...")
                # 테스트용 비디오 파일들 확인
                test_videos = ["test_video.mp4", "sample.mp4", "demo.avi"]
                for test_video in test_videos:
                    if os.path.exists(test_video):
                        video_path = test_video
                        break
            
            if os.path.exists(video_path):
                # 처리 옵션
                start_frame = int(input("시작 프레임 (기본 0): ") or "0")
                end_frame_input = input("끝 프레임 (기본 전체): ").strip()
                end_frame = int(end_frame_input) if end_frame_input else None
                frame_interval = int(input("프레임 간격 (기본 1): ") or "1")
                
                # 처리 실행
                stats = processor.process_video_file(
                    video_path=video_path,
                    start_frame=start_frame,
                    end_frame=end_frame,
                    frame_interval=frame_interval
                )
                
                # 처리 완료 후 데이터셋 생성 여부 확인
                create_dataset = input("\n결합 데이터셋을 생성하시겠습니까? (y/n): ").strip().lower()
                if create_dataset == 'y':
                    dataset = processor.create_dataset_from_combined_data()
            else:
                print(f"❌ 비디오 파일을 찾을 수 없습니다: {video_path}")
        
        elif choice == "2":
            # 웹캠 처리
            camera_id = int(input("카메라 ID (기본 0): ") or "0")
            duration = int(input("녹화 시간(초) (기본 30): ") or "30")
            save_interval = int(input("저장 간격(프레임) (기본 10): ") or "10")
            
            stats = processor.process_webcam_recording(
                camera_id=camera_id,
                duration_seconds=duration,
                save_interval=save_interval
            )
            
            # 처리 완료 후 데이터셋 생성 여부 확인
            create_dataset = input("\n결합 데이터셋을 생성하시겠습니까? (y/n): ").strip().lower()
            if create_dataset == 'y':
                dataset = processor.create_dataset_from_combined_data()
        
        elif choice == "3":
            # 크롭 테스트
            test_image = "winter01.jpg"
            if os.path.exists(test_image):
                print(f"🧪 크롭 테스트: {test_image}")
                
                image = cv2.imread(test_image)
                vis_frame, results = processor.inferencer.process_frame(image)
                
                if results:
                    for person_id, (original_keypoints, original_scores, bbox) in enumerate(results):
                        # 크롭 이미지에서 포즈 추정 수행 (일관된 좌표계)
                        crop_image, crop_keypoints, crop_scores = processor._process_person_with_crop_pose(image, bbox, person_id, 0)
                        
                        if crop_image is not None:
                            crop_path = processor._save_crop_image(crop_image, 0, person_id, bbox)
                            print(f"✅ 크롭 저장: {crop_path}")
                            
                            # 결합 데이터도 저장 (크롭 좌표계)
                            combined_path = processor._save_combined_data(0, person_id, crop_image, crop_keypoints, crop_scores, bbox, image.shape)
                            if combined_path:
                                print(f"🔗 결합 데이터 저장: {combined_path}")
                            
                            # 크롭 이미지 표시
                            cv2.imshow(f'Person {person_id} Crop', crop_image)
                    
                    cv2.imshow('Original with Detection', vis_frame)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                else:
                    print("❌ 사람을 검출하지 못했습니다.")
            else:
                print(f"❌ 테스트 이미지 없음: {test_image}")
        
        elif choice == "4":
            # 결합 데이터셋 생성
            data_dir = input("결합 데이터 디렉토리 경로 (기본: video_processing_outputs/combined_data): ").strip()
            if not data_dir:
                data_dir = None
            
            dataset = processor.create_dataset_from_combined_data(data_dir)
            
        elif choice == "5":
            # 결합 데이터 로드 테스트
            combined_dir = processor.output_dir / "combined_data"
            if combined_dir.exists():
                json_files = list(combined_dir.glob("*_combined.json"))
                if json_files:
                    # 첫 번째 파일로 테스트
                    test_file = json_files[0]
                    print(f"🧪 결합 데이터 로드 테스트: {test_file.name}")
                    
                    crop_image, keypoints, scores, metadata = processor.load_combined_data(str(test_file))
                    
                    if crop_image is not None:
                        print(f"✅ 로드 성공:")
                        print(f"   - 이미지 크기: {crop_image.shape}")
                        print(f"   - 키포인트 수: {len(keypoints)}")
                        print(f"   - 평균 신뢰도: {np.mean(scores):.3f}")
                        print(f"   - 프레임 인덱스: {metadata['frame_info']['frame_idx']}")
                        
                        # 이미지 표시
                        cv2.imshow('Loaded Crop Image', crop_image)
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()
                    else:
                        print("❌ 로드 실패")
                else:
                    print("❌ 결합 데이터 파일이 없습니다.")
            else:
                print("❌ 결합 데이터 디렉토리가 없습니다.")
        
        elif choice == "6":
            # 단일 결합 데이터 시각화
            combined_dir = processor.output_dir / "combined_data"
            if combined_dir.exists():
                json_files = list(combined_dir.glob("*_combined.json"))
                if json_files:
                    print(f"\n📁 사용 가능한 결합 데이터 파일:")
                    for i, file in enumerate(json_files):
                        print(f"   {i+1}. {file.name}")
                    
                    try:
                        choice_idx = int(input(f"시각화할 파일 선택 (1-{len(json_files)}): ")) - 1
                        if 0 <= choice_idx < len(json_files):
                            selected_file = json_files[choice_idx]
                            print(f"🎨 시각화 생성 중: {selected_file.name}")
                            
                            vis_path = processor.visualize_combined_data(str(selected_file))
                            if vis_path:
                                print(f"✅ 시각화 완료! 파일: {vis_path}")
                            else:
                                print("❌ 시각화 실패")
                        else:
                            print("❌ 잘못된 선택입니다.")
                    except ValueError:
                        print("❌ 숫자를 입력해주세요.")
                else:
                    print("❌ 결합 데이터 파일이 없습니다.")
            else:
                print("❌ 결합 데이터 디렉토리가 없습니다.")
        
        elif choice == "7":
            # 전체 요약 시각화
            data_dir = input("결합 데이터 디렉토리 경로 (기본: video_processing_outputs/combined_data): ").strip()
            if not data_dir:
                data_dir = None
            
            print("📊 전체 데이터셋 요약 시각화 생성 중...")
            summary_path = processor.create_combined_data_summary(data_dir)
            if summary_path:
                print(f"✅ 요약 시각화 완료! 파일: {summary_path}")
            else:
                print("❌ 요약 시각화 실패")
        
        elif choice == "8":
            # 결합 데이터 구조 설명
            print("\n" + "="*80)
            print("📚 결합 데이터 구조 설명")
            print("="*80)
            
            print("""
🎯 결합 데이터란?
  YOLO11L 검출 결과와 RTMW 포즈 추정 결과를 하나로 합친 데이터셋입니다.
  이미지 크롭과 포즈 키포인트가 함께 저장되어 머신러닝 학습에 최적화되어 있습니다.

📁 저장 파일 구조 (각 사람마다 4개 파일):
  1. frame_XXXXXX_person_X_combined.json       - 메타데이터 및 통계
  2. frame_XXXXXX_person_X_combined_image.npy  - 크롭 이미지 (288x384 넘파이)
  3. frame_XXXXXX_person_X_combined_pose.npy   - 포즈 데이터 (133x3: x,y,score)
  4. frame_XXXXXX_person_X_combined_crop.jpg   - 시각화용 JPEG 이미지

📊 JSON 메타데이터 구조:
  {
    "frame_info": {
      "frame_idx": 프레임 인덱스,
      "person_id": 사람 ID,
      "timestamp": 생성 시간,
      "bbox": [x1, y1, x2, y2] 바운딩박스
    },
    "pose_data": {
      "keypoints": [[x, y], ...] 133개 키포인트 좌표,
      "scores": [score, ...] 각 키포인트 신뢰도,
      "stats": {
        "total_keypoints": 133,
        "valid_keypoints": 신뢰도 > 0.3인 키포인트 수,
        "high_confidence_keypoints": 신뢰도 > 0.8인 키포인트 수,
        "average_confidence": 평균 신뢰도,
        "max_confidence": 최대 신뢰도,
        "min_confidence": 최소 신뢰도
      }
    },
    "image_info": {
      "original_shape": [288, 384, 3],
      "dtype": "uint8",
      "channels": 3,
      "size_bytes": 이미지 바이트 크기
    }
  }

🔄 데이터 처리 과정:
  1. YOLO11L로 사람 검출 → 바운딩박스 생성
  2. 바운딩박스 15% 패딩 확장 → RTMW 비율(3:4)로 변환
  3. 288x384로 리사이즈 (RTMW 정확한 입력 크기)
  4. RTMW로 크롭 이미지에서 직접 133개 키포인트 추정
  5. 이미지 + 포즈 + 메타데이터 결합 저장

💡 활용 방법:
  • 머신러닝 데이터셋으로 사용
  • 포즈 분석 및 시각화
  • 동작 인식 학습 데이터
  • 수화 인식 데이터셋
  • 스포츠 동작 분석

🛠️ 지원 기능:
  • load_combined_data(): 저장된 데이터 로드
  • visualize_combined_data(): 단일 데이터 시각화
  • create_combined_data_summary(): 전체 데이터셋 통계 시각화
  • create_dataset_from_combined_data(): 넘파이 데이터셋 생성
            """)
            
            print("="*80)
        
        else:
            print("❌ 잘못된 선택입니다.")
    
    except Exception as e:
        print(f"❌ 처리 실패: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
