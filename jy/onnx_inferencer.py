#!/usr/bin/env python3
"""
YOLO11L + RTMW-L ONNX 하이브리드 추론기
Large 모델을 사용한 고정확도 사람 검출 + ONNX 포즈 추정
"""

import os
import torch
import cv2
import numpy as np
import time
import urllib.request
import hashlib
import zipfile
import tempfile
from pathlib import Path
from typing import List, Tuple, Optional
from collections import deque
import onnxruntime as ort

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    print("⚠️ ultralytics 미설치 - pip install ultralytics")
    YOLO_AVAILABLE = False

# 모델 다운로드 관련 상수
MODEL_URLS = {
    "rtmw-dw-x-l_simcc-cocktail14_270e-384x288.onnx": {
        "type": "zip",
        "url": "https://download.openmmlab.com/mmpose/v1/projects/rtmw/onnx_sdk/rtmw-dw-x-l_simcc-cocktail14_270e-384x288_20231122.zip",
        "extracted_name": "end2end.onnx",
        "md5": None  # zip 파일의 MD5는 별도로 확인하지 않음
    },
    "rtmw-l_simcc-cocktail14_pt-ucoco_270e-384x288.onnx": {
        "type": "direct",
        "url": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmw-l_simcc-cocktail14_pt-ucoco_270e-384x288.onnx",
        "md5": "6708b5b97b65d476b982a6e8b2fc56e1"
    }
}

def download_file_with_progress(url: str, filepath: str, expected_md5: str = None):
    """진행률 표시하면서 파일 다운로드"""
    print(f"📥 다운로드 중: {os.path.basename(filepath)}")
    print(f"   URL: {url}")
    
    def show_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(downloaded * 100.0 / total_size, 100.0)
            downloaded_mb = downloaded / (1024 * 1024)
            total_mb = total_size / (1024 * 1024)
            print(f"\r   진행률: {percent:.1f}% ({downloaded_mb:.1f}/{total_mb:.1f} MB)", end='')
    
    try:
        # 디렉토리 생성
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # 파일 다운로드
        urllib.request.urlretrieve(url, filepath, show_progress)
        print()  # 새 줄
        
        # MD5 체크섬 확인
        if expected_md5:
            print("   MD5 체크섬 확인 중...")
            with open(filepath, 'rb') as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
            
            if file_hash != expected_md5:
                os.remove(filepath)
                raise ValueError(f"MD5 체크섬 불일치: 예상({expected_md5}) != 실제({file_hash})")
            print("   ✅ MD5 체크섬 확인 완료")
        
        print(f"✅ 다운로드 완료: {os.path.basename(filepath)}")
        return True
        
    except Exception as e:
        print(f"\n❌ 다운로드 실패: {e}")
        if os.path.exists(filepath):
            os.remove(filepath)
        return False

def download_and_extract_zip(url: str, target_filepath: str, extracted_name: str = "end2end.onnx"):
    """ZIP 파일 다운로드 및 특정 파일 압축 해제"""
    print(f"📦 ZIP 파일 다운로드 및 압축 해제 중...")
    print(f"   URL: {url}")
    print(f"   대상 파일: {extracted_name}")
    print(f"   저장 경로: {target_filepath}")
    
    try:
        # 대상 디렉토리 생성
        os.makedirs(os.path.dirname(target_filepath), exist_ok=True)
        
        # 임시 디렉토리 생성
        with tempfile.TemporaryDirectory() as temp_dir:
            zip_path = os.path.join(temp_dir, "model.zip")
            
            # ZIP 파일 다운로드
            def show_progress(block_num, block_size, total_size):
                if total_size > 0:
                    downloaded = block_num * block_size
                    percent = min(downloaded * 100.0 / total_size, 100.0)
                    downloaded_mb = downloaded / (1024 * 1024)
                    total_mb = total_size / (1024 * 1024)
                    print(f"\r   다운로드 진행률: {percent:.1f}% ({downloaded_mb:.1f}/{total_mb:.1f} MB)", end='', flush=True)
            
            print("   🌐 다운로드 시작...")
            urllib.request.urlretrieve(url, zip_path, show_progress)
            print("\n   ✅ 다운로드 완료")
            
            # ZIP 파일 압축 해제
            print("   📂 압축 해제 중...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # ZIP 파일 내용 확인
                file_list = zip_ref.namelist()
                print(f"   📋 ZIP 파일 내용: {len(file_list)}개 파일")
                
                # 대상 파일 찾기 - 여러 패턴으로 시도
                target_file_in_zip = None
                possible_names = [extracted_name, f"*/{extracted_name}", f"**/{extracted_name}"]
                
                for file_path in file_list:
                    # 정확한 이름 매치
                    if os.path.basename(file_path) == extracted_name:
                        target_file_in_zip = file_path
                        break
                    # .onnx 확장자로 끝나는 파일 찾기
                    elif file_path.endswith('.onnx') and 'end2end' in file_path:
                        target_file_in_zip = file_path
                        break
                
                if target_file_in_zip is None:
                    print(f"   ❌ ZIP 파일에서 ONNX 모델을 찾을 수 없습니다.")
                    print(f"   📋 사용 가능한 파일들:")
                    for file_path in file_list[:20]:  # 처음 20개만 출력
                        print(f"      - {file_path}")
                    if len(file_list) > 20:
                        print(f"      ... 총 {len(file_list)}개 파일")
                    return False
                
                print(f"   ✅ 대상 파일 발견: {target_file_in_zip}")
                
                # 파일 압축 해제
                with zip_ref.open(target_file_in_zip) as source, open(target_filepath, 'wb') as target:
                    file_size = zip_ref.getinfo(target_file_in_zip).file_size
                    extracted_size = 0
                    chunk_size = 8192  # 8KB 청크
                    
                    while True:
                        chunk = source.read(chunk_size)
                        if not chunk:
                            break
                        target.write(chunk)
                        extracted_size += len(chunk)
                        
                        if file_size > 0:
                            percent = min(extracted_size * 100.0 / file_size, 100.0)
                            print(f"\r   압축 해제 진행률: {percent:.1f}%", end='', flush=True)
                
                print("\n   ✅ 압축 해제 완료")
                
                # 파일 크기 확인
                if os.path.exists(target_filepath):
                    file_size_mb = os.path.getsize(target_filepath) / (1024 * 1024)
                    print(f"   📄 최종 파일 크기: {file_size_mb:.1f} MB")
                    if file_size_mb > 10:  # 10MB 이상이면 성공으로 간주
                        return True
                    else:
                        print(f"   ⚠️ 파일 크기가 너무 작습니다 ({file_size_mb:.1f}MB < 10MB)")
                        return False
                else:
                    print(f"   ❌ 압축 해제된 파일을 찾을 수 없습니다.")
                    return False
    
    except Exception as e:
        print(f"\n❌ ZIP 다운로드/압축 해제 실패: {e}")
        if os.path.exists(target_filepath):
            try:
                os.remove(target_filepath)
            except:
                pass
        return False

def ensure_model_exists(model_path: str, model_name: str = None) -> bool:
    """모델 파일 존재 확인 및 자동 다운로드 (캐싱 개선)"""
    # 파일 존재 및 크기 검증
    if os.path.exists(model_path):
        try:
            file_size = os.path.getsize(model_path)
            if file_size > 50 * 1024 * 1024:  # 50MB 이상이어야 유효한 ONNX 모델
                print(f"✅ 모델 파일 존재: {os.path.basename(model_path)} ({file_size/1024/1024:.1f}MB)")
                return True
            else:
                print(f"⚠️ 불완전한 모델 파일 감지 ({file_size/1024/1024:.1f}MB < 50MB), 재다운로드 필요")
                os.remove(model_path)  # 불완전한 파일 삭제
        except OSError as e:
            print(f"⚠️ 모델 파일 검증 실패: {e}, 재다운로드 시도")
    
    if model_name is None:
        model_name = os.path.basename(model_path)
    
    # 알려진 모델인지 확인
    if model_name in MODEL_URLS:
        model_info = MODEL_URLS[model_name]
        print(f"🔍 모델 자동 다운로드 시작: {model_name}")
        
        # 디렉토리 생성
        model_dir = os.path.dirname(model_path)
        if model_dir:  # 빈 문자열이 아닌 경우만
            os.makedirs(model_dir, exist_ok=True)
        else:
            # 현재 디렉토리에 모델 저장
            model_path = os.path.basename(model_path)
            print(f"⚠️ 모델 경로 수정: {model_path}")
            
        # 다운로드 타입에 따라 처리
        if model_info.get("type") == "zip":
            # ZIP 파일 다운로드 및 압축 해제
            success = download_and_extract_zip(
                model_info["url"],
                model_path,
                model_info["extracted_name"]
            )
            # 다운로드 성공 후 파일 크기 재검증
            if success and os.path.exists(model_path):
                final_size = os.path.getsize(model_path)
                print(f"✅ 모델 다운로드 완료 (최종 크기: {final_size/1024/1024:.1f}MB)")
                return True
            return False
        else:
            # 직접 파일 다운로드
            success = download_file_with_progress(
                model_info["url"], 
                model_path, 
                model_info.get("md5")
            )
            # 다운로드 성공 후 파일 크기 재검증
            if success and os.path.exists(model_path):
                final_size = os.path.getsize(model_path)
                print(f"✅ 모델 다운로드 완료 (최종 크기: {final_size/1024/1024:.1f}MB)")
                return True
            return False
    else:
        print(f"❌ 알 수 없는 모델: {model_name}")
        print(f"   수동으로 다운로드하여 다음 경로에 저장하세요: {model_path}")
        
        # 사용 가능한 모델 목록 출력
        print(f"   📋 사용 가능한 모델들:")
        for available_model in MODEL_URLS.keys():
            print(f"      - {available_model}")
        return False

def get_available_providers():
    """사용 가능한 ONNX 실행 제공자 확인"""
    available = ort.get_available_providers()
    print(f"🔧 사용 가능한 ONNX Providers: {', '.join(available)}")
    return available

def select_best_provider():
    """최적의 ONNX 실행 제공자 선택"""
    available = get_available_providers()
    
    # 우선순위: OpenVINO > CUDA > DirectML > CPU
    priority_order = [
        'OpenVINOExecutionProvider',    # Intel GPU/CPU 최적화
        'CUDAExecutionProvider',        # NVIDIA GPU
        'DmlExecutionProvider',         # DirectML (Windows GPU)
        'CPUExecutionProvider'          # CPU 폴백
    ]
    
    for provider in priority_order:
        if provider in available:
            print(f"✅ 선택된 ONNX Provider: {provider}")
            return provider
    
    return 'CPUExecutionProvider'

class YOLO11LRTMWONNXInferencer:
    """YOLO11L + RTMW ONNX 하이브리드 추론기 (ZIP 다운로드 지원)"""
    
    def __init__(self, 
                 rtmw_onnx_path: str,
                 detection_device: str = "auto",
                 pose_device: str = "auto",
                 optimize_for_accuracy: bool = True):
        """
        Args:
            rtmw_onnx_path: RTMW ONNX 모델 경로 (ZIP 파일 자동 다운로드/압축해제 지원)
            detection_device: 검출 디바이스 ('auto', 'cpu', 'cuda', 'xpu')
            pose_device: 포즈 추정 디바이스 ('auto', 'cpu', 'cuda', 'openvino')
            optimize_for_accuracy: 정확도 최적화 여부
        """
        if not YOLO_AVAILABLE:
            raise ImportError("ultralytics가 필요합니다: pip install ultralytics")
            
        self.rtmw_onnx_path = rtmw_onnx_path
        self.yolo_model_name = "yolo11l.pt"  # Large 모델 사용
        self.optimize_for_accuracy = optimize_for_accuracy
        
        # 디바이스 결정
        self.detection_device = self._determine_detection_device(detection_device)
        self.pose_provider = self._determine_pose_provider(pose_device)
        
        # 모델 이름 확인 (경로에서 추출)
        model_filename = os.path.basename(rtmw_onnx_path)
        model_type = "RTMW-DW-X-L" if "dw-x-l" in model_filename else "RTMW"
        
        print(f"🚀 YOLO11L + {model_type} ONNX 하이브리드 추론기 초기화:")
        print(f"   - YOLO 모델: YOLO11L (Large - 고정확도)")
        print(f"   - RTMW 모델: {model_type} 384x288 ONNX (ZIP 다운로드 지원)")
        print(f"   - RTMW 모델: RTMW-L 384x288 ONNX")
        print(f"   - 검출 디바이스: {self.detection_device}")
        print(f"   - 포즈 Provider: {self.pose_provider}")
        print(f"   - 정확도 최적화: {'ON' if optimize_for_accuracy else 'OFF'}")
        
        # 모델 초기화
        self._init_detection_model()
        self._init_pose_model()
        
        # 성능 통계
        self.inference_times = {
            'detection': [],
            'pose': [],
            'total': []
        }
        
        # 최적화 설정
        self._setup_optimization()
        
    def _determine_detection_device(self, device: str) -> str:
        """검출 디바이스 자동 결정"""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            else:
                return "cpu"
        return device
    
    def _determine_pose_provider(self, device: str) -> str:
        """포즈 추정 Provider 자동 결정"""
        if device == "auto":
            return select_best_provider()
        elif device == "openvino":
            return 'OpenVINOExecutionProvider'
        elif device == "cuda":
            return 'CUDAExecutionProvider'
        elif device == "directml":
            return 'DmlExecutionProvider'
        else:
            return 'CPUExecutionProvider'
    
    def _setup_optimization(self):
        """최적화 설정 - 정확도 우선"""
        if self.optimize_for_accuracy:
            print("🎯 정확도 최적화 설정 적용 중...")
            
            # YOLO11L 정확도 우선 파라미터
            self.yolo_conf_thresh = 0.4     # 낮은 신뢰도 (더 많은 검출)
            self.yolo_iou_thresh = 0.6      # 적당한 IoU 임계값
            self.yolo_max_det = 50          # 더 많은 검출 허용
            self.yolo_classes = [0]         # 사람 클래스만
            
            # 이미지 크기 최적화
            self.detection_img_size = 832   # Large 모델에 적합한 큰 입력 크기
            self.pose_input_size = (288, 384)  # RTMW-L 384x288 입력 크기
            
            print("✅ 정확도 최적화 설정 완료")
        else:
            # 균형 설정
            self.yolo_conf_thresh = 0.5
            self.yolo_iou_thresh = 0.7
            self.yolo_max_det = 100
            self.yolo_classes = None
            self.detection_img_size = 640
            self.pose_input_size = (288, 384)
    
    def _init_detection_model(self):
        """YOLO11L 검출 모델 초기화"""
        print(f"🔧 YOLO11L 검출 모델 로딩 중... (디바이스: {self.detection_device})")
        start_time = time.time()
        
        try:
            # 모델 디렉토리 생성
            models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
            os.makedirs(models_dir, exist_ok=True)
            
            # YOLO11L 모델 경로
            model_path = os.path.join(models_dir, self.yolo_model_name)
            
            # 모델 파일 존재 확인
            if not os.path.exists(model_path):
                print(f"📥 YOLO11L 모델 자동 다운로드 중: {self.yolo_model_name}")
                print("   (Ultralytics에서 자동으로 다운로드됩니다)")
            
            # YOLO 모델 로드 (자동 다운로드 포함)
            self.detection_model = YOLO(self.yolo_model_name)
            
            # 모델을 지정된 위치에 복사 (다음에 더 빠른 로딩을 위해)
            if not os.path.exists(model_path) and hasattr(self.detection_model, 'ckpt_path'):
                try:
                    import shutil
                    if os.path.exists(self.detection_model.ckpt_path):
                        shutil.copy2(self.detection_model.ckpt_path, model_path)
                        print(f"💾 모델 복사됨: {model_path}")
                except Exception as e:
                    print(f"⚠️ 모델 복사 실패 (정상 작동): {e}")
            
            # 디바이스 설정
            if self.detection_device != "cpu":
                try:
                    self.detection_model.to(self.detection_device)
                    print(f"✅ YOLO11L {self.detection_device.upper()} 모드 활성화")
                except Exception as e:
                    print(f"⚠️ YOLO11L {self.detection_device.upper()} 실패, CPU로 폴백: {e}")
                    self.detection_device = "cpu"
                    self.detection_model.to('cpu')
            
            init_time = time.time() - start_time
            print(f"✅ YOLO11L 검출 모델 로딩 완료: {init_time:.2f}초")
            
        except Exception as e:
            print(f"❌ YOLO11L 모델 로딩 실패: {e}")
            self.detection_model = None
            print(f"🔄 간단한 검출기로 폴백")
    
    def _init_pose_model(self):
        """RTMW-L ONNX 포즈 추정 모델 초기화"""
        print(f"🔧 RTMW-L ONNX 포즈 모델 로딩 중... (Provider: {self.pose_provider})")
        start_time = time.time()
        
        try:
            # 모델 파일 존재 확인 및 자동 다운로드
            model_name = os.path.basename(self.rtmw_onnx_path)
            if not ensure_model_exists(self.rtmw_onnx_path, model_name):
                # 기본 경로에서도 시도
                default_path = os.path.join(os.path.dirname(__file__), "..", "models", model_name)
                if not ensure_model_exists(default_path, model_name):
                    raise FileNotFoundError(f"RTMW-L ONNX 모델을 찾을 수 없습니다: {self.rtmw_onnx_path}")
                else:
                    self.rtmw_onnx_path = default_path
            
            # ONNX 세션 옵션 설정
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            # Provider 설정
            providers = [self.pose_provider]
            if self.pose_provider != 'CPUExecutionProvider':
                providers.append('CPUExecutionProvider')  # 폴백용
            
            # ONNX 세션 생성
            self.pose_session = ort.InferenceSession(
                self.rtmw_onnx_path,
                sess_options=sess_options,
                providers=providers
            )
            
            # 입력/출력 정보 확인
            self.input_name = self.pose_session.get_inputs()[0].name
            self.input_shape = self.pose_session.get_inputs()[0].shape
            self.output_names = [output.name for output in self.pose_session.get_outputs()]
            
            print(f"📋 ONNX 모델 정보:")
            print(f"   - 입력: {self.input_name} {self.input_shape}")
            print(f"   - 출력: {len(self.output_names)}개")
            print(f"   - Provider: {self.pose_session.get_providers()[0]}")
            
            init_time = time.time() - start_time
            print(f"✅ RTMW-L ONNX 포즈 모델 로딩 완료: {init_time:.2f}초")
            
        except Exception as e:
            print(f"❌ RTMW-L ONNX 모델 로딩 실패: {e}")
            # CPU 폴백 시도
            print("🔄 CPU Provider로 폴백 시도...")
            try:
                self.pose_session = ort.InferenceSession(
                    self.rtmw_onnx_path,
                    providers=['CPUExecutionProvider']
                )
                self.input_name = self.pose_session.get_inputs()[0].name
                self.input_shape = self.pose_session.get_inputs()[0].shape
                self.output_names = [output.name for output in self.pose_session.get_outputs()]
                self.pose_provider = 'CPUExecutionProvider'
                print(f"✅ CPU 폴백 성공")
            except Exception as e2:
                print(f"❌ CPU 폴백도 실패: {e2}")
                raise e2
    
    def detect_persons_high_accuracy(self, image: np.ndarray) -> List[List[float]]:
        """고정확도 사람 검출"""
        if self.detection_model is None:
            return self._simple_person_detection(image)
        
        try:
            start_time = time.time()
            
            # YOLO11L 추론 (고정확도 파라미터)
            results = self.detection_model(
                image,
                conf=self.yolo_conf_thresh,
                iou=self.yolo_iou_thresh,
                max_det=self.yolo_max_det,
                classes=self.yolo_classes,
                verbose=False,
                imgsz=self.detection_img_size
            )
            
            detection_time = time.time() - start_time
            self.inference_times['detection'].append(detection_time)
            
            # 사람(class 0) 검출 결과 추출
            person_boxes = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None and len(boxes) > 0:
                    person_coords = boxes.xyxy
                    person_confs = boxes.conf
                    
                    # 신뢰도 재필터링
                    conf_mask = person_confs >= self.yolo_conf_thresh
                    if conf_mask.any():
                        filtered_boxes = person_coords[conf_mask]
                        filtered_confs = person_confs[conf_mask]
                        
                        # numpy 변환
                        if isinstance(filtered_boxes, torch.Tensor):
                            filtered_boxes = filtered_boxes.cpu().numpy()
                            filtered_confs = filtered_confs.cpu().numpy()
                        
                        # 신뢰도순 정렬
                        sorted_indices = np.argsort(filtered_confs)[::-1]
                        sorted_boxes = filtered_boxes[sorted_indices]
                        
                        person_boxes.extend(sorted_boxes.tolist())
            
            return person_boxes if person_boxes else self._simple_person_detection(image)
            
        except Exception as e:
            print(f"❌ YOLO11L 검출 실패: {e}")
            return self._simple_person_detection(image)
    
    def _simple_person_detection(self, image: np.ndarray) -> List[List[float]]:
        """간단한 사람 검출 (폴백)"""
        h, w = image.shape[:2]
        margin_w = int(w * 0.1)
        margin_h = int(h * 0.1)
        bbox = [margin_w, margin_h, w - margin_w, h - margin_h]
        return [bbox]
    
    def _preprocess_image_for_pose(self, crop_image: np.ndarray) -> Optional[np.ndarray]:
        """포즈 추정용 이미지 전처리"""
        try:
            # 입력 이미지 유효성 검사
            if crop_image is None or crop_image.size == 0:
                print(f"⚠️ 빈 크롭 이미지가 전달됨")
                return None
            
            h, w = crop_image.shape[:2]
            if h == 0 or w == 0:
                print(f"⚠️ 크기가 0인 크롭 이미지: {h}x{w}")
                return None
            
            # RTMW-L 384x288 크기로 리사이즈
            try:
                resized = cv2.resize(crop_image, self.pose_input_size)  # (288, 384)
            except cv2.error as e:
                print(f"⚠️ 리사이즈 실패: {e}, 이미지 크기: {crop_image.shape}")
                return None
            
            # BGR to RGB
            rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            
            # 정규화 (0-1 범위)
            normalized = rgb_image.astype(np.float32) / 255.0
            
            # 표준화 (ImageNet 평균/표준편차)
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            standardized = (normalized - mean) / std
            
            # 차원 변경: HWC -> CHW
            transposed = standardized.transpose(2, 0, 1)
            
            # 배치 차원 추가: CHW -> BCHW
            batched = np.expand_dims(transposed, axis=0).astype(np.float32)
            
            return batched
        
        except Exception as e:
            print(f"⚠️ 이미지 전처리 실패: {e}")
            return None
    
    def _postprocess_pose_output(self, outputs: List[np.ndarray], 
                               original_crop_shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        """포즈 추정 출력 후처리"""
        try:
            # RTMW 출력 형태에 따라 조정 필요
            # 일반적으로 keypoints와 heatmap/simcc 출력이 있음
            
            if len(outputs) >= 2:
                # SimCC 방식의 경우
                pred_x = outputs[0]  # x 좌표 예측
                pred_y = outputs[1]  # y 좌표 예측
                
                # 최대값 위치 찾기
                keypoints_x = np.argmax(pred_x[0], axis=1)
                keypoints_y = np.argmax(pred_y[0], axis=1)
                
                # 신뢰도 계산
                scores_x = np.max(pred_x[0], axis=1)
                scores_y = np.max(pred_y[0], axis=1)
                scores = np.minimum(scores_x, scores_y)
                
                # 좌표 스케일링 (모델 입력 크기에서 크롭 이미지 크기로)
                scale_x = original_crop_shape[1] / self.pose_input_size[0]  # width
                scale_y = original_crop_shape[0] / self.pose_input_size[1]  # height
                
                keypoints_x = keypoints_x * scale_x
                keypoints_y = keypoints_y * scale_y
                
                # 키포인트 배열 생성
                keypoints = np.stack([keypoints_x, keypoints_y], axis=1)
                
                return keypoints, scores
            else:
                # 다른 출력 형태의 경우
                print(f"⚠️ 예상하지 못한 출력 형태: {len(outputs)}개 출력")
                return np.zeros((133, 2)), np.zeros(133)
                
        except Exception as e:
            print(f"❌ 포즈 출력 후처리 실패: {e}")
            return np.zeros((133, 2)), np.zeros(133)
    
    def estimate_pose_on_crop(self, crop_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """크롭된 이미지에서 ONNX 포즈 추정"""
        try:
            start_time = time.time()
            
            # 전처리
            input_tensor = self._preprocess_image_for_pose(crop_image)
            
            # ONNX 추론
            outputs = self.pose_session.run(
                self.output_names,
                {self.input_name: input_tensor}
            )
            
            # 후처리
            keypoints, scores = self._postprocess_pose_output(outputs, crop_image.shape[:2])
            
            pose_time = time.time() - start_time
            self.inference_times['pose'].append(pose_time)
            
            return keypoints, scores
            
        except Exception as e:
            print(f"❌ ONNX 포즈 추정 실패: {e}")
            return np.zeros((133, 2)), np.zeros(133)
    
    def estimate_pose_batch(self, crop_images: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """배치 포즈 추정 (ONNX)"""
        if not crop_images:
            return np.array([]), np.array([])
        
        batch_keypoints = []
        batch_scores = []
        
        try:
            # 배치 전처리
            batch_inputs = []
            valid_indices = []  # 유효한 이미지 인덱스 추적
            original_shapes = []  # 원본 이미지 크기 추적
            
            for i, crop_image in enumerate(crop_images):
                # 이미 전처리된 텐서인지 확인
                if len(crop_image.shape) == 4 and crop_image.shape[1] == 3:
                    # 이미 전처리된 배치 텐서 (B, C, H, W)
                    batch_inputs.append(crop_image[0])  # 배치 차원 제거
                    valid_indices.append(i)
                    original_shapes.append((384, 288))  # RTMW 입력 크기
                elif len(crop_image.shape) == 3 and crop_image.shape[0] == 3:
                    # 이미 전처리된 단일 텐서 (C, H, W)
                    batch_inputs.append(crop_image)
                    valid_indices.append(i)
                    original_shapes.append((384, 288))  # RTMW 입력 크기
                else:
                    # 원본 이미지인 경우 전처리 필요
                    input_tensor = self._preprocess_image_for_pose(crop_image)
                    if input_tensor is not None:
                        batch_inputs.append(input_tensor[0])  # 배치 차원 제거
                        valid_indices.append(i)
                        original_shapes.append(crop_image.shape[:2])
                    else:
                        print(f"⚠️ 배치 처리에서 이미지 {i} 전처리 실패")
            
            if not batch_inputs:
                print(f"⚠️ 유효한 입력 이미지가 없음")
                return np.array([]), np.array([])
            
            batch_tensor = np.stack(batch_inputs, axis=0)
            
            # 배치 추론
            outputs = self.pose_session.run(
                self.output_names,
                {self.input_name: batch_tensor}
            )
            
            # 각 유효한 이미지에 대해 후처리
            batch_idx = 0
            for i, crop_image in enumerate(crop_images):
                if i in valid_indices:
                    # 배치 출력에서 해당 결과 추출
                    image_outputs = [output[batch_idx:batch_idx+1] for output in outputs]
                    keypoints, scores = self._postprocess_pose_output(image_outputs, original_shapes[batch_idx])
                    batch_keypoints.append(keypoints)
                    batch_scores.append(scores)
                    batch_idx += 1
                else:
                    # 실패한 이미지는 기본값
                    batch_keypoints.append(np.zeros((133, 2)))
                    batch_scores.append(np.zeros(133))
            
            return np.array(batch_keypoints), np.array(batch_scores)
        
        except Exception as e:
            print(f"배치 처리 실패, 개별 처리로 폴백: {e}")
            # 폴백: 개별 처리
            for crop_image in crop_images:
                keypoints, scores = self.estimate_pose_on_crop(crop_image)
                batch_keypoints.append(keypoints)
                batch_scores.append(scores)
        
            return np.array(batch_keypoints), np.array(batch_scores)
    
    def process_frame(self, image: np.ndarray, conf_thresh: float = None) -> Tuple[np.ndarray, List[Tuple[np.ndarray, np.ndarray, List[float]]]]:
        """프레임 처리 (고정확도 검출 + ONNX 포즈 추정)"""
        start_time = time.time()
        
        # 신뢰도 임계값 설정
        if conf_thresh is None:
            conf_thresh = self.yolo_conf_thresh
        
        # 1. 고정확도 사람 검출
        person_boxes = self.detect_persons_high_accuracy(image)
        
        # 2. 각 사람에 대해 포즈 추정
        results = []
        for bbox in person_boxes:
            # 바운딩박스에서 크롭
            x1, y1, x2, y2 = map(int, bbox)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
            
            if x2 > x1 and y2 > y1:
                crop_image = image[y1:y2, x1:x2]
                keypoints, scores = self.estimate_pose_on_crop(crop_image)
                
                # 키포인트를 원본 이미지 좌표계로 변환
                keypoints[:, 0] += x1
                keypoints[:, 1] += y1
                
                results.append((keypoints, scores, bbox))
        
        total_time = time.time() - start_time
        self.inference_times['total'].append(total_time)
        
        # 3. 시각화
        vis_image = self.visualize_results(image, results)
        
        return vis_image, results
    
    def visualize_results(self, image: np.ndarray, results: List[Tuple[np.ndarray, np.ndarray, List[float]]]) -> np.ndarray:
        """결과 시각화"""
        vis_image = image.copy()
        
        for i, (keypoints, scores, bbox) in enumerate(results):
            # 바운딩박스 그리기
            x1, y1, x2, y2 = map(int, bbox)
            color = (0, 255, 0) if i == 0 else (255, 0, 255)
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
            
            # 사람 번호 표시
            cv2.putText(vis_image, f"Person {i+1}", (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # 키포인트 그리기 (신뢰도별 색상)
            for j, (kpt, score) in enumerate(zip(keypoints, scores)):
                if score > 0.3:
                    x, y = int(kpt[0]), int(kpt[1])
                    if 0 <= x < vis_image.shape[1] and 0 <= y < vis_image.shape[0]:
                        # 신뢰도에 따른 색상
                        if score > 0.8:
                            kpt_color = (0, 255, 0)    # 높은 신뢰도: 초록
                        elif score > 0.6:
                            kpt_color = (0, 255, 255)  # 중간 신뢰도: 노랑
                        else:
                            kpt_color = (0, 0, 255)    # 낮은 신뢰도: 빨강
                        
                        cv2.circle(vis_image, (x, y), 3, kpt_color, -1)
        
        # 성능 정보 표시
        if self.inference_times['total']:
            fps = 1.0 / self.inference_times['total'][-1]
            cv2.putText(vis_image, f"FPS: {fps:.1f}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 모델 정보
        cv2.putText(vis_image, "YOLO11L + RTMW-L ONNX", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(vis_image, f"Detection: {self.detection_device.upper()}", 
                   (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(vis_image, f"Pose: {self.pose_provider.split('ExecutionProvider')[0]}", 
                   (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return vis_image
    
    def benchmark_performance(self, image: np.ndarray, num_runs: int = 15) -> dict:
        """성능 벤치마크"""
        print(f"🏃 YOLO11L + RTMW-L ONNX 성능 벤치마크 ({num_runs}회)...")
        
        # 워밍업
        for _ in range(3):
            self.process_frame(image)
        
        # 실제 벤치마크
        self.inference_times = {'detection': [], 'pose': [], 'total': []}
        
        for i in range(num_runs):
            self.process_frame(image)
            if (i + 1) % 5 == 0:
                print(f"   진행률: {i+1}/{num_runs}")
        
        # 통계 계산
        stats = {}
        for key, times in self.inference_times.items():
            if times:
                stats[key] = {
                    'mean': np.mean(times),
                    'std': np.std(times),
                    'min': np.min(times),
                    'max': np.max(times),
                    'fps': 1.0 / np.mean(times) if key == 'total' else None
                }
        
        return stats
    
    def test_single_image(self, image_path: str):
        """단일 이미지 테스트"""
        print(f"\n=== YOLO11L + RTMW-L ONNX 테스트: {os.path.basename(image_path)} ===")
        
        if not os.path.exists(image_path):
            print(f"❌ 이미지 파일 없음: {image_path}")
            return
            
        # 이미지 로드
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ 이미지 로드 실패: {image_path}")
            return
        
        print(f"📷 이미지 크기: {image.shape}")
        
        # 처리
        vis_image, results = self.process_frame(image)
        
        # 결과 출력
        print(f"✅ 검출된 사람 수: {len(results)}")
        
        for i, (keypoints, scores, bbox) in enumerate(results):
            valid_kpts = np.sum(scores > 0.3)
            high_conf_kpts = np.sum(scores > 0.8)
            bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            print(f"   사람 {i+1}: {valid_kpts}/133 키포인트 (신뢰도 > 0.3), {high_conf_kpts} 고신뢰도")
            print(f"            바운딩박스 크기: {bbox_area:.0f}px²")
        
        # 성능 벤치마크
        stats = self.benchmark_performance(image, num_runs=15)
        
        print(f"\n📊 성능 통계:")
        for stage, stat in stats.items():
            if stat:
                print(f"   {stage}:")
                print(f"     - 평균: {stat['mean']*1000:.1f}ms")
                print(f"     - 최소/최대: {stat['min']*1000:.1f}/{stat['max']*1000:.1f}ms")
                if stat['fps']:
                    print(f"     - FPS: {stat['fps']:.1f}")
        
        # 결과 저장
        output_path = f"yolo11l_rtmw_onnx_result_{os.path.basename(image_path)}"
        cv2.imwrite(output_path, vis_image)
        print(f"💾 결과 저장: {output_path}")
        
        return vis_image, results, stats

def main():
    """메인 테스트 함수"""
    # 모델 경로 설정 - 여러 가능한 경로 확인
    models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    # 새 고성능 모델을 기본으로 사용
    rtmw_model_name = "rtmw-dw-x-l_simcc-cocktail14_270e-384x288.onnx"
    
    # 가능한 모델 경로들
    possible_paths = [
        os.path.join(models_dir, rtmw_model_name),
        os.path.join("../models", rtmw_model_name),
        os.path.join("models", rtmw_model_name),
        rtmw_model_name  # 현재 디렉토리
    ]
    
    rtmw_onnx_path = None
    for path in possible_paths:
        if os.path.exists(path):
            rtmw_onnx_path = path
            break
    
    # 모델이 없으면 기본 경로 사용 (자동 다운로드됨)
    if rtmw_onnx_path is None:
        rtmw_onnx_path = possible_paths[0]  # 첫 번째 경로 사용
    
    try:
        print("🚀 YOLO11L + RTMW-DW-X-L ONNX 하이브리드 추론기 테스트")
        print("=" * 70)
        
        # 사용 가능한 ONNX Provider 확인
        get_available_providers()
        
        # YOLO11L + RTMW-DW-X-L ONNX 하이브리드 추론기 생성
        inferencer = YOLO11LRTMWONNXInferencer(
            rtmw_onnx_path=rtmw_onnx_path,
            detection_device="auto",
            pose_device="auto",
            optimize_for_accuracy=True
        )
        
        # 테스트 이미지 - 여러 가능한 경로 확인
        test_image_names = ["winter01.jpg", "test.jpg", "demo.jpg", "sample.jpg"]
        test_image = None
        
        for img_name in test_image_names:
            possible_img_paths = [
                img_name,  # 현재 디렉토리
                os.path.join("../demo/resources", img_name),
                os.path.join("demo/resources", img_name),
                os.path.join("resources", img_name)
            ]
            
            for img_path in possible_img_paths:
                if os.path.exists(img_path):
                    test_image = img_path
                    break
            
            if test_image:
                break
        
        if test_image:
            inferencer.test_single_image(test_image)
        else:
            print("⚠️ 테스트 이미지를 찾을 수 없습니다. 웹캠 테스트만 진행합니다.")
            print("   테스트 이미지를 준비하려면 다음 중 하나를 현재 디렉토리에 저장하세요:")
            for img_name in test_image_names:
                print(f"   - {img_name}")
        
        # 실시간 웹캠 테스트
        print(f"\n🎥 YOLO11L + RTMW-L ONNX 실시간 웹캠 테스트를 시작하시겠습니까? (y/n): ", end="")
        try:
            choice = input().strip().lower()
        except (KeyboardInterrupt, EOFError):
            choice = 'n'
        
        if choice == 'y':
            inferencer.test_webcam()
        
        print(f"\n🏆 YOLO11L + RTMW ONNX 시스템 특징:")
        print(f"   🎯 최고 정확도: Large 모델로 더 정확한 사람 검출")
        print(f"   🔍 정밀 검출: 낮은 신뢰도 임계값으로 놓치기 쉬운 사람도 검출")
        print(f"   📐 최적 입력: YOLO 832px, RTMW 384x288")
        print(f"   ⚡ ONNX 가속: OpenVINO/CUDA/DirectML 활용")
        print(f"   💨 빠른 추론: ONNX Runtime 최적화")
        print(f"   📦 ZIP 지원: 자동 ZIP 다운로드 및 압축 해제")
        print(f"   📥 자동 다운로드: 필요한 모델 자동 설치")
        
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()

    def test_webcam(self, camera_id: int = 0, window_size: Tuple[int, int] = (1280, 720)):
        """실시간 웹캠 테스트 - YOLO11L + RTMW-L ONNX"""
        print(f"\n=== YOLO11L + RTMW-L ONNX 실시간 웹캠 테스트 (카메라 ID: {camera_id}) ===")
        print("📹 웹캠 연결 중...")
        
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print(f"❌ 웹캠 열기 실패 (카메라 ID: {camera_id})")
            return
        
        # 웹캠 설정
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, window_size[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, window_size[1])
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        
        # 실제 웹캠 해상도 확인
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"✅ 웹캠 연결 성공: {actual_width}x{actual_height}, {actual_fps:.1f}fps")
        print(f"🎮 조작법:")
        print(f"   - ESC: 종료")
        print(f"   - S: 현재 프레임 스크린샷")
        print(f"   - SPACE: 일시정지/재생")
        print(f"   - A: 정확도 모드 토글 (높음/표준)")
        print(f"   - +/-: 신뢰도 임계값 조정")
        
        # 성능 측정 변수
        frame_count = 0
        fps_history = deque(maxlen=30)
        paused = False
        screenshot_count = 0
        accuracy_mode = True
        current_conf_thresh = self.yolo_conf_thresh
        
        try:
            while True:
                if not paused:
                    ret, frame = cap.read()
                    if not ret:
                        print("❌ 프레임 읽기 실패")
                        break
                    
                    # 프레임 처리
                    start_time = time.time()
                    vis_frame, results = self.process_frame(frame, conf_thresh=current_conf_thresh)
                    process_time = time.time() - start_time
                    
                    # FPS 계산
                    fps = 1.0 / process_time if process_time > 0 else 0
                    fps_history.append(fps)
                    avg_fps = np.mean(fps_history) if fps_history else 0
                    
                    # 추가 정보 표시
                    info_y = 140
                    cv2.putText(vis_frame, f"Avg FPS: {avg_fps:.1f}", 
                               (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    cv2.putText(vis_frame, f"Provider: {self.pose_provider.split('ExecutionProvider')[0]}", 
                               (10, info_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    cv2.putText(vis_frame, f"Mode: {'High Accuracy' if accuracy_mode else 'Standard'}", 
                               (10, info_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    cv2.putText(vis_frame, f"Conf: {current_conf_thresh:.2f}", 
                               (10, info_y + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    
                    # 검출된 사람 정보
                    if results is not None and len(results) > 0:
                        person_info = f"Persons: {len(results)}"
                        for i, (_, scores, _) in enumerate(results):
                            valid_kpts = np.sum(scores > 0.3)
                            high_conf_kpts = np.sum(scores > 0.8)
                            person_info += f" | P{i+1}: {valid_kpts}/133 ({high_conf_kpts} high)"
                        cv2.putText(vis_frame, person_info, 
                                   (10, info_y + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                    
                    frame_count += 1
                
                # 화면 표시
                cv2.imshow('YOLO11L + RTMW-L ONNX Real-time', vis_frame)
                
                # 키 입력 처리
                key = cv2.waitKey(1) & 0xFF
                
                if key == 27:  # ESC
                    break
                elif key == ord('s') or key == ord('S'):  # 스크린샷
                    screenshot_name = f"yolo11l_rtmw_onnx_screenshot_{screenshot_count:04d}.jpg"
                    cv2.imwrite(screenshot_name, vis_frame)
                    print(f"📸 스크린샷 저장: {screenshot_name}")
                    screenshot_count += 1
                elif key == ord(' '):  # 일시정지/재생
                    paused = not paused
                    print(f"⏸️ {'일시정지' if paused else '재생'}")
                elif key == ord('a') or key == ord('A'):  # 정확도 모드 토글
                    accuracy_mode = not accuracy_mode
                    if accuracy_mode:
                        current_conf_thresh = 0.4
                        self.detection_img_size = 832
                        print("🎯 고정확도 모드 활성화")
                    else:
                        current_conf_thresh = 0.6
                        self.detection_img_size = 640
                        print("⚡ 표준 속도 모드 활성화")
                elif key == ord('+') or key == ord('='):  # 신뢰도 증가
                    current_conf_thresh = min(0.9, current_conf_thresh + 0.05)
                    print(f"📈 신뢰도 임계값: {current_conf_thresh:.2f}")
                elif key == ord('-'):  # 신뢰도 감소
                    current_conf_thresh = max(0.1, current_conf_thresh - 0.05)
                    print(f"📉 신뢰도 임계값: {current_conf_thresh:.2f}")
                
                # 성능 통계 주기적 출력
                if frame_count % 60 == 0 and frame_count > 0:
                    print(f"📊 프레임 {frame_count}: 평균 {avg_fps:.1f}fps, "
                          f"{len(results)}명 검출 (Provider: {self.pose_provider.split('ExecutionProvider')[0]})")
                    
        except KeyboardInterrupt:
            print("\n⏹️ 사용자가 웹캠 테스트를 중단했습니다.")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            
            # 최종 통계
            if fps_history:
                final_avg_fps = np.mean(fps_history)
                print(f"\n📊 YOLO11L + RTMW-L ONNX 웹캠 테스트 완료:")
                print(f"   - 처리된 프레임: {frame_count}")
                print(f"   - 평균 FPS: {final_avg_fps:.1f}")
                print(f"   - 검출 디바이스: {self.detection_device}")
                print(f"   - 포즈 Provider: {self.pose_provider}")
                print(f"   - 스크린샷: {screenshot_count}개 저장")
                print(f"   - 최종 신뢰도 임계값: {current_conf_thresh:.2f}")

if __name__ == "__main__":
    main()