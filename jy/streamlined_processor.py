#!/usr/bin/env python3
"""
스트림라인 비디오 처리기 - HDF5 배치 처리용
SEN ID 기반으로 깔끔하게 처리하여 불필요한 중간 파일 제거
"""

import os
import cv2
import json
import h5py
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from tqdm import tqdm
import time
import shutil
import urllib.request

# 설정 및 MMPose 관련 임포트
from config import MODELS_DIR, YOLO_MODEL_CONFIG, RTMW_MODEL_OPTIONS
from yolo11l_xpu_hybrid_inferencer import YOLO11LXPUHybridInferencer

# RTMW 전처리 함수들 (video_processor_yolo11l.py에서 가져옴)
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

class StreamlinedVideoProcessor:
    """HDF5용 간소화된 비디오 처리기"""
    
    def __init__(self, 
                 rtmw_config_path: str = "configs/wholebody_2d_keypoint/rtmpose/cocktail14/rtmw-x_8xb320-270e_cocktail14-384x288.py",
                 rtmw_model_name: str = "rtmw-x"):  # 모델명으로 선택
        
        self.logger = logging.getLogger(__name__)
        self.keypoint_scale = 8  # 키포인트 x,y 좌표 8배 스케일링
        
        # 상대 경로를 절대 경로로 변환
        base_dir = Path(__file__).parent.parent  # mmpose/jy/.. -> mmpose
        rtmw_config_path = str(base_dir / rtmw_config_path)
        
        # 모델 파일들 확인 및 다운로드
        yolo_model_path = self._ensure_yolo_model()
        rtmw_model_path = self._ensure_rtmw_model(rtmw_model_name)
        
        # YOLO11L + RTMW 하이브리드 추론기 초기화
        self.inferencer = YOLO11LXPUHybridInferencer(
            rtmw_config=rtmw_config_path,
            rtmw_checkpoint=rtmw_model_path,
            detection_device='xpu',
            pose_device='xpu',
            optimize_for_accuracy=True
        )
        
        self.logger.info("✅ 스트림라인 비디오 처리기 초기화 완료")

    def _ensure_yolo_model(self) -> str:
        """YOLO 모델 파일 확인 및 다운로드"""
        yolo_config = YOLO_MODEL_CONFIG
        model_path = Path(MODELS_DIR) / yolo_config["filename"]
        
        if model_path.exists():
            self.logger.info(f"✅ 기존 YOLO 모델 발견: {model_path}")
            return str(model_path)
        
        self.logger.info(f"📥 YOLO 모델 다운로드 시작: {yolo_config['filename']}")
        
        # models 디렉토리 생성
        Path(MODELS_DIR).mkdir(parents=True, exist_ok=True)
        
        try:
            # YOLO 모델은 ultralytics에서 자동 다운로드되므로 
            # 일시적으로 해당 경로에서 다운로드 후 models로 복사
            from ultralytics import YOLO
            
            # 임시로 YOLO 모델 로드 (자동 다운로드됨)
            temp_model = YOLO(yolo_config["filename"])
            
            # ultralytics 캐시에서 모델 파일 찾기
            import torch
            from ultralytics.utils import ASSETS
            
            # 다운로드된 모델 찾기
            cache_dir = Path.home() / '.cache' / 'ultralytics'
            downloaded_model = None
            
            for weights_dir in [cache_dir, cache_dir / 'weights']:
                if weights_dir.exists():
                    for model_file in weights_dir.glob(yolo_config["filename"]):
                        downloaded_model = model_file
                        break
                if downloaded_model:
                    break
            
            if downloaded_model and downloaded_model.exists():
                # models 디렉토리로 복사
                shutil.copy2(downloaded_model, model_path)
                self.logger.info(f"✅ YOLO 모델 복사 완료: {model_path}")
                self.logger.info(f"   파일 크기: {model_path.stat().st_size / (1024*1024):.1f} MB")
                return str(model_path)
            else:
                self.logger.warning(f"⚠️ YOLO 모델 다운로드 위치를 찾을 수 없음")
                return yolo_config["filename"]  # ultralytics가 자동으로 처리하도록
            
        except Exception as e:
            self.logger.warning(f"⚠️ YOLO 모델 다운로드 실패: {e}")
            self.logger.info("   ultralytics가 자동으로 다운로드할 예정")
            return yolo_config["filename"]

    def _ensure_rtmw_model(self, model_name: str = "rtmw-x") -> str:
        """RTMW 모델 파일 확인 및 다운로드"""
        # 모델명으로 설정 찾기
        rtmw_config = None
        for config in RTMW_MODEL_OPTIONS:
            if model_name in config["filename"]:
                rtmw_config = config
                break
        
        if not rtmw_config:
            self.logger.error(f"❌ 알 수 없는 RTMW 모델명: {model_name}")
            rtmw_config = RTMW_MODEL_OPTIONS[0]  # 기본값 사용
        
        model_path = Path(rtmw_config["path"])
        
        if model_path.exists():
            self.logger.info(f"✅ 기존 RTMW 모델 발견: {model_path}")
            return str(model_path)
        
        self.logger.info(f"📥 RTMW 모델 다운로드 시작: {rtmw_config['filename']}")
        
        # models 디렉토리 생성
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            download_url = rtmw_config["url"]
            if not download_url:
                self.logger.error(f"❌ 다운로드 URL이 없음: {rtmw_config['filename']}")
                return str(model_path)
            
            self.logger.info(f"🔄 다운로드 중: {download_url}")
            
            # 진행률 표시가 있는 다운로드
            def download_progress_hook(block_num, block_size, total_size):
                if total_size > 0:
                    percent = min(100, (block_num * block_size * 100) // total_size)
                    if block_num % 50 == 0:  # 50블록마다 출력 (더 자주)
                        self.logger.info(f"   다운로드 진행률: {percent}%")
            
            urllib.request.urlretrieve(download_url, model_path, download_progress_hook)
            
            # 파일 크기 검증
            if model_path.exists() and model_path.stat().st_size > 1024 * 1024:  # 1MB 이상
                self.logger.info(f"✅ RTMW 모델 다운로드 완료: {model_path}")
                self.logger.info(f"   파일 크기: {model_path.stat().st_size / (1024*1024):.1f} MB")
                return str(model_path)
            else:
                self.logger.error(f"❌ 다운로드된 파일이 유효하지 않음: {model_path}")
                if model_path.exists():
                    model_path.unlink()  # 손상된 파일 삭제
                return str(model_path)
            
        except Exception as e:
            self.logger.error(f"❌ RTMW 모델 다운로드 실패: {e}")
            if model_path.exists():
                model_path.unlink()  # 부분 다운로드 파일 삭제
            return str(model_path)

    def _crop_person_image_rtmw(self, image: np.ndarray, bbox: List[float]) -> Optional[np.ndarray]:
        """RTMW 방식으로 사람 이미지 크롭"""
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
                self.logger.warning(f"⚠️ 크기 오류: {h}x{w}, 예상: {input_height}x{input_width}")
                return cropped_image
                
        except Exception as e:
            self.logger.warning(f"⚠️ RTMW 전처리 실패: {e}")
            return None

    def process_video_to_arrays(self, video_path: str) -> Optional[Dict[str, np.ndarray]]:
        """
        비디오를 넘파이 배열로 직접 변환
        
        Args:
            video_path: 비디오 파일 경로
            
        Returns:
            Dict containing:
            - crop_images: (N, 288, 384, 3) uint8
            - keypoints: (N, 133, 2) float32  
            - scores: (N, 133) float32
            - frame_count: int
        """
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                self.logger.error(f"❌ 비디오 열기 실패: {video_path}")
                return None
            
            all_crop_images = []
            all_keypoints = []
            all_scores = []
            
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 1. YOLO 검출로 사람 찾기
                try:
                    vis_image, results = self.inferencer.process_frame(frame)
                    if not results or len(results) == 0:
                        frame_idx += 1
                        continue
                    
                    # 첫 번째 사람의 bbox만 사용
                    _, _, bbox = results[0]
                    
                    # 2. RTMW 방식으로 크롭 이미지 생성
                    crop_image = self._crop_person_image_rtmw(frame, bbox)
                    if crop_image is None:
                        frame_idx += 1
                        continue
                    
                    # 3. 크롭된 이미지에서 직접 포즈 추정 (288x384 좌표계)
                    keypoints, scores = self.inferencer.estimate_pose_on_crop(crop_image)
                    
                    # 4. 배열에 추가
                    all_crop_images.append(crop_image)
                    all_keypoints.append(keypoints)
                    all_scores.append(scores)
                    
                except Exception as e:
                    self.logger.warning(f"프레임 {frame_idx} 처리 실패: {e}")
                    frame_idx += 1
                    continue
                
                frame_idx += 1
            
            cap.release()
            
            if len(all_crop_images) == 0:
                self.logger.warning(f"⚠️ 유효한 프레임이 없습니다: {video_path}")
                return None
            
            return {
                'crop_images': np.stack(all_crop_images),      # (N, 288, 384, 3)
                'keypoints': np.stack(all_keypoints),          # (N, 133, 2)
                'scores': np.stack(all_scores),                # (N, 133)
                'frame_count': len(all_crop_images)
            }
            
        except Exception as e:
            self.logger.error(f"❌ 비디오 처리 실패: {video_path}, 오류: {e}")
            return None

    def process_sen_video(self, sen_id: int, video_path: str, output_dir: Path) -> bool:
        """
        SEN ID 기반으로 비디오 처리하고 저장
        
        Args:
            sen_id: SEN 번호 (예: 1)
            video_path: 비디오 파일 경로
            output_dir: 출력 디렉토리
            
        Returns:
            bool: 성공 여부
        """
        try:
            # 비디오 처리
            self.logger.info(f"🎬 처리 중: SEN{sen_id:04d} - {Path(video_path).name}")
            start_time = time.time()
            
            arrays = self.process_video_to_arrays(video_path)
            if arrays is None:
                return False
            
            processing_time = time.time() - start_time
            
            # SEN ID 폴더 생성
            sen_dir = output_dir / f"SEN{sen_id:04d}"
            sen_dir.mkdir(parents=True, exist_ok=True)
            
            # 넘파이 배열 저장 (키포인트는 8배 스케일링)
            np.save(sen_dir / "crop_images.npy", arrays['crop_images'])
            
            # 키포인트 8배 스케일링하여 정수로 저장
            keypoints_scaled = np.round(arrays['keypoints'] * self.keypoint_scale).astype(np.int32)
            np.save(sen_dir / "keypoints_scaled.npy", keypoints_scaled)
            np.save(sen_dir / "keypoints_original.npy", arrays['keypoints'])  # 원본도 보존
            
            np.save(sen_dir / "scores.npy", arrays['scores'])
            
            # 최소한의 메타데이터
            metadata = {
                'sen_id': sen_id,
                'video_path': str(video_path),
                'video_filename': Path(video_path).name,
                'frame_count': arrays['frame_count'],
                'processing_time': processing_time,
                'shape_info': {
                    'crop_images': list(arrays['crop_images'].shape),
                    'keypoints_original': list(arrays['keypoints'].shape),
                    'keypoints_scaled': list(keypoints_scaled.shape),
                    'scores': list(arrays['scores'].shape)
                },
                'keypoint_scale': self.keypoint_scale,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            with open(sen_dir / "metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.logger.info(f"✅ SEN{sen_id:04d} 완료: {arrays['frame_count']}프레임, {processing_time:.2f}초")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ SEN{sen_id:04d} 처리 실패: {e}")
            return False


class BatchProcessor:
    """250개 단위 배치 처리기"""
    
    def __init__(self, 
                 data_root: str = "data/1.Training",  # 상대 경로로 변경
                 output_dir: str = "sign_language_dataset",
                 batch_size: int = 250,
                 rtmw_model_name: str = "rtmw-x"):  # RTMW 모델 선택 옵션 추가
        
        self.data_root = Path(data_root)
        self.output_dir = Path(output_dir)
        self.batch_size = batch_size
        self.rtmw_model_name = rtmw_model_name
        
        # 로깅 설정
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('batch_processing.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # 출력 디렉토리 생성
        self.sen_output_dir = self.output_dir / "sen_processing"
        self.hdf5_output_dir = self.output_dir / "hdf5_batches"
        
        self.sen_output_dir.mkdir(parents=True, exist_ok=True)
        self.hdf5_output_dir.mkdir(parents=True, exist_ok=True)
        
        # 스트림라인 처리기 초기화 (모델 자동 다운로드 포함)
        self.logger.info(f"🚀 모델 초기화 시작 (RTMW: {rtmw_model_name})")
        self.processor = StreamlinedVideoProcessor(rtmw_model_name=rtmw_model_name)
        self.logger.info("✅ 배치 처리기 초기화 완료")

    def collect_f_videos(self) -> List[Tuple[int, str]]:
        """F 방향 영상 수집 (라벨 무관)"""
        videos_base_dir = self.data_root / "videos"
        
        video_data = []
        
        if not videos_base_dir.exists():
            self.logger.error(f"❌ videos 폴더 없음: {videos_base_dir}")
            return video_data
        
        # videos 하위의 모든 폴더 자동 검색
        for sub_dir in videos_base_dir.iterdir():
            if not sub_dir.is_dir():
                continue
                
            self.logger.info(f"🔍 검색 중: {sub_dir}")
            
            # F 방향 영상 파일 검색
            for video_file in sub_dir.glob("*_F.mp4"):
                try:
                    # SEN 번호 추출 (예: NIA_SL_SEN0001_REAL03_F.mp4 → 1)
                    filename = video_file.stem
                    if '_SEN' not in filename:
                        self.logger.warning(f"⚠️ SEN 번호를 찾을 수 없음: {filename}")
                        continue
                        
                    sen_match = filename.split('_SEN')[1].split('_')[0]
                    sen_id = int(sen_match)
                    
                    video_data.append((sen_id, str(video_file)))
                    self.logger.debug(f"✅ 발견: SEN{sen_id:04d} - {video_file.name}")
                    
                except (IndexError, ValueError) as e:
                    self.logger.warning(f"⚠️ SEN ID 추출 실패: {filename} - {e}")
                    continue
        
        # SEN ID로 정렬
        video_data.sort(key=lambda x: x[0])
        
        self.logger.info(f"🎬 F 방향 영상 총 {len(video_data)}개 발견")
        
        # 상세 통계
        if video_data:
            min_sen = min(video_data, key=lambda x: x[0])[0]
            max_sen = max(video_data, key=lambda x: x[0])[0]
            self.logger.info(f"📊 SEN 범위: SEN{min_sen:04d} ~ SEN{max_sen:04d}")
        
        return video_data

    def create_batches(self, video_data: List[Tuple[int, str]]) -> List[List[Tuple[int, str]]]:
        """비디오 데이터를 배치로 분할"""
        batches = []
        for i in range(0, len(video_data), self.batch_size):
            batch = video_data[i:i + self.batch_size]
            batches.append(batch)
        
        self.logger.info(f"📦 총 {len(batches)}개 배치로 분할 (배치 크기: {self.batch_size})")
        return batches

    def process_sen_batch(self, batch_data: List[Tuple[int, str]], batch_idx: int) -> List[int]:
        """SEN 배치 처리 (비디오 → 넘파이 배열)"""
        successful_sen_ids = []
        
        self.logger.info(f"🔄 배치 {batch_idx} SEN 처리 시작 ({len(batch_data)}개 영상)")
        
        for sen_id, video_path in tqdm(batch_data, desc=f"배치 {batch_idx} 처리"):
            success = self.processor.process_sen_video(sen_id, video_path, self.sen_output_dir)
            if success:
                successful_sen_ids.append(sen_id)
        
        self.logger.info(f"✅ 배치 {batch_idx} SEN 처리 완료: {len(successful_sen_ids)}/{len(batch_data)}개 성공")
        return successful_sen_ids

    def create_hdf5_batch(self, sen_ids: List[int], batch_data: List[Tuple[int, str]], batch_idx: int):
        """SEN 배열들을 HDF5 배치로 변환 (라벨 없이)"""
        try:
            self.logger.info(f"📦 배치 {batch_idx} HDF5 생성 시작")
            
            # HDF5 파일 경로
            frames_h5 = self.hdf5_output_dir / f"batch_{batch_idx:02d}_F_frames.h5"
            poses_h5 = self.hdf5_output_dir / f"batch_{batch_idx:02d}_F_poses.h5"
            
            with h5py.File(frames_h5, 'w') as f_frames, \
                 h5py.File(poses_h5, 'w') as f_poses:
                
                for sen_id in tqdm(sen_ids, desc=f"HDF5 배치 {batch_idx}"):
                    sen_dir = self.sen_output_dir / f"SEN{sen_id:04d}"
                    
                    # 넘파이 배열 로드
                    crop_images = np.load(sen_dir / "crop_images.npy")
                    keypoints_scaled = np.load(sen_dir / "keypoints_scaled.npy")  # 8배 스케일링된 정수 좌표
                    keypoints_original = np.load(sen_dir / "keypoints_original.npy")  # 원본 float 좌표
                    scores = np.load(sen_dir / "scores.npy")
                    
                    # 메타데이터 로드
                    with open(sen_dir / "metadata.json", 'r') as f:
                        metadata = json.load(f)
                    
                    # HDF5에 저장
                    video_group = f"video_{sen_id:04d}"
                    
                    # 프레임 데이터
                    f_frames.create_dataset(f"{video_group}/frames", 
                                          data=crop_images, 
                                          compression='gzip', compression_opts=1)
                    f_frames.create_dataset(f"{video_group}/metadata", 
                                          data=json.dumps(metadata))
                    
                    # 포즈 데이터 (스케일링된 정수 좌표와 원본 좌표 모두 저장)
                    f_poses.create_dataset(f"{video_group}/keypoints_scaled", 
                                         data=keypoints_scaled, 
                                         compression='gzip', compression_opts=1)
                    f_poses.create_dataset(f"{video_group}/keypoints_original", 
                                         data=keypoints_original, 
                                         compression='gzip', compression_opts=1)
                    f_poses.create_dataset(f"{video_group}/scores", 
                                         data=scores, 
                                         compression='gzip', compression_opts=1)
            
            self.logger.info(f"✅ 배치 {batch_idx} HDF5 생성 완료")
            self.logger.info(f"   - 프레임: {frames_h5}")
            self.logger.info(f"   - 포즈: {poses_h5}")
            
        except Exception as e:
            self.logger.error(f"❌ 배치 {batch_idx} HDF5 생성 실패: {e}")

    def cleanup_sen_files(self, sen_ids: List[int]):
        """SEN 중간 파일들 정리 (선택적)"""
        for sen_id in sen_ids:
            sen_dir = self.sen_output_dir / f"SEN{sen_id:04d}"
            if sen_dir.exists():
                import shutil
                shutil.rmtree(sen_dir)
        
        self.logger.info(f"🧹 SEN 중간 파일 {len(sen_ids)}개 정리 완료")

    def process_all_batches(self, cleanup_intermediate: bool = False):
        """전체 배치 처리 파이프라인"""
        # 1. F 방향 영상 수집
        video_data = self.collect_f_videos()
        if not video_data:
            self.logger.error("❌ 처리할 영상이 없습니다")
            return
        
        # 2. 배치 분할
        batches = self.create_batches(video_data)
        
        # 3. 각 배치 처리
        for batch_idx, batch_data in enumerate(batches):
            self.logger.info(f"\n🚀 배치 {batch_idx + 1}/{len(batches)} 처리 시작")
            
            # 3-1. SEN 처리 (비디오 → 넘파이)
            successful_sen_ids = self.process_sen_batch(batch_data, batch_idx)
            
            if successful_sen_ids:
                # 3-2. HDF5 생성 (넘파이 → HDF5)
                self.create_hdf5_batch(successful_sen_ids, batch_data, batch_idx)
                
                # 3-3. 중간 파일 정리 (선택적)
                if cleanup_intermediate:
                    self.cleanup_sen_files(successful_sen_ids)
            
            self.logger.info(f"✅ 배치 {batch_idx + 1} 완료\n")
        
        self.logger.info("🎉 전체 배치 처리 완료!")

    def process_test_batch(self, test_count: int = 5):
        """테스트용 소규모 배치 처리"""
        video_data = self.collect_f_videos()[:test_count]
        
        self.logger.info(f"🧪 테스트 배치 처리 시작 ({test_count}개 영상)")
        
        # SEN 처리
        successful_sen_ids = []
        for sen_id, video_path in video_data:
            success = self.processor.process_sen_video(sen_id, video_path, self.sen_output_dir)
            if success:
                successful_sen_ids.append(sen_id)
        
        # HDF5 생성
        if successful_sen_ids:
            self.create_hdf5_batch(successful_sen_ids, video_data, 99)  # 테스트용 배치 번호
        
        self.logger.info(f"✅ 테스트 배치 완료 ({len(successful_sen_ids)}/{test_count}개 성공)")


def main():
    """메인 실행 함수"""
    print("🚀 스트림라인 배치 처리기")
    print("=" * 50)
    
    # RTMW 모델 선택
    print("\n사용할 RTMW 모델을 선택하세요:")
    print("1. RTMW-x (최고 성능, 기본값)")
    print("2. RTMW-l (균형)")
    
    model_choice = input("모델 선택 (1-2, 기본값: 1): ").strip()
    
    rtmw_model_map = {
        '1': 'rtmw-x',
        '2': 'rtmw-dw-x-l',
        '': 'rtmw-x'  # 기본값
    }
    
    rtmw_model_name = rtmw_model_map.get(model_choice, 'rtmw-x')
    print(f"✅ 선택된 모델: {rtmw_model_name}")
    
    # 배치 처리기 초기화 (모델 자동 다운로드 포함)
    print("\n📥 모델 다운로드 및 초기화 중...")
    try:
        batch_processor = BatchProcessor(rtmw_model_name=rtmw_model_name)
        print("✅ 초기화 완료!")
    except Exception as e:
        print(f"❌ 초기화 실패: {e}")
        return
    
    while True:
        print("\n처리 모드를 선택하세요:")
        print("1. 테스트 처리 (5개 영상)")
        print("2. 전체 배치 처리 (250개씩)")
        print("3. 전체 배치 처리 + 중간파일 정리")
        print("4. 영상 목록만 확인")
        print("5. 모델 정보 확인")
        print("0. 종료")
        
        choice = input("선택 (0-5): ").strip()
        
        if choice == '1':
            batch_processor.process_test_batch(5)
        elif choice == '2':
            batch_processor.process_all_batches(cleanup_intermediate=False)
        elif choice == '3':
            batch_processor.process_all_batches(cleanup_intermediate=True)
        elif choice == '4':
            video_data = batch_processor.collect_f_videos()
            print(f"\n📊 총 {len(video_data)}개 영상 발견:")
            for i, (sen_id, video_path) in enumerate(video_data[:10]):
                print(f"  {i+1}. SEN{sen_id:04d} - {Path(video_path).name}")
            if len(video_data) > 10:
                print(f"  ... 외 {len(video_data) - 10}개")
        elif choice == '5':
            print(f"\n📋 현재 모델 정보:")
            print(f"  - RTMW 모델: {rtmw_model_name}")
            print(f"  - YOLO 모델: {YOLO_MODEL_CONFIG['filename']}")
            print(f"  - 모델 디렉토리: {MODELS_DIR}")
            
            # 모델 파일 존재 확인
            yolo_path = Path(MODELS_DIR) / YOLO_MODEL_CONFIG["filename"]
            print(f"  - YOLO 파일 존재: {'✅' if yolo_path.exists() else '❌'}")
            
            for config in RTMW_MODEL_OPTIONS:
                if rtmw_model_name in config["filename"]:
                    rtmw_path = Path(config["path"])
                    print(f"  - RTMW 파일 존재: {'✅' if rtmw_path.exists() else '❌'}")
                    if rtmw_path.exists():
                        size_mb = rtmw_path.stat().st_size / (1024*1024)
                        print(f"    크기: {size_mb:.1f} MB")
                    break
        elif choice == '0':
            print("👋 종료합니다.")
            break
        else:
            print("❌ 잘못된 선택입니다.")


if __name__ == "__main__":
    main()
