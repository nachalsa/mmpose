#!/usr/bin/env python3
"""
포즈 추정 서버: JPEG 이미지 수신 → RTMW 포즈 추정 → 결과 반환
"""

import cv2
import numpy as np
import time
import json
import argparse
from pathlib import Path
import logging
from typing import Optional, Tuple, Dict, Any
from collections import deque
import os

# Flask 웹 서버
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import threading
import queue

# MMPose 관련
from mmpose.apis import init_model, inference_topdown
import torch

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RTMWPoseEstimator:
    """RTMW 포즈 추정기"""
    
    def __init__(self, 
                 rtmw_config: str,
                 rtmw_checkpoint: str,
                 device: str = "auto"):
        
        self.rtmw_config = rtmw_config
        self.rtmw_checkpoint = rtmw_checkpoint
        self.device = self._determine_device(device)
        
        # PyTorch 보안 설정
        self.original_load = torch.load
        torch.load = lambda *args, **kwargs: self.original_load(
            *args, **kwargs, weights_only=False
        ) if 'weights_only' not in kwargs else self.original_load(*args, **kwargs)
        
        print(f"🔧 RTMW 포즈 모델 로딩 중... (디바이스: {self.device})")
        start_time = time.time()
        
        try:
            self.pose_model = init_model(
                config=self.rtmw_config,
                checkpoint=self.rtmw_checkpoint,
                device=self.device
            )
            
            init_time = time.time() - start_time
            print(f"✅ RTMW 포즈 모델 로딩 완료: {init_time:.2f}초")
            
        except Exception as e:
            print(f"❌ {self.device} 포즈 모델 실패: {e}")
            if self.device != 'cpu':
                print(f"🔄 CPU로 폴백...")
                self.device = 'cpu'
                
                self.pose_model = init_model(
                    config=self.rtmw_config,
                    checkpoint=self.rtmw_checkpoint,
                    device='cpu'
                )
                
                init_time = time.time() - start_time
                print(f"✅ CPU 포즈 모델 로딩 완료: {init_time:.2f}초")
            else:
                raise
    
    def _determine_device(self, device: str) -> str:
        """디바이스 자동 결정"""
        if device == "auto":
            # XPU 확인
            try:
                if torch.xpu.is_available():
                    return "xpu"
            except:
                pass
            
            # CUDA 확인
            if torch.cuda.is_available():
                return "cuda"
            
            return "cpu"
        else:
            # 사용자 지정 디바이스 검증
            if device == "xpu":
                try:
                    if not torch.xpu.is_available():
                        print("⚠️ XPU 미사용 가능 - CPU로 폴백")
                        return "cpu"
                except:
                    print("⚠️ XPU 확인 실패 - CPU로 폴백")
                    return "cpu"
            elif device == "cuda":
                if not torch.cuda.is_available():
                    print("⚠️ CUDA 미사용 가능 - CPU로 폴백")
                    return "cpu"
            
            return device
    
    def estimate_pose_on_crop(self, crop_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """크롭된 이미지에서 포즈 추정"""
        try:
            start_time = time.time()
            
            # 크롭 이미지 전체를 바운딩박스로 사용 (288x384)
            h, w = crop_image.shape[:2]
            full_bbox = [0, 0, w, h]
            
            # MMPose 추론
            results = inference_topdown(
                model=self.pose_model,
                img=crop_image,
                bboxes=[full_bbox],
                bbox_format='xyxy'
            )
            
            pose_time = time.time() - start_time
            
            if results and len(results) > 0:
                keypoints = results[0].pred_instances.keypoints[0]
                scores = results[0].pred_instances.keypoint_scores[0]
                
                if isinstance(keypoints, torch.Tensor):
                    keypoints = keypoints.cpu().numpy()
                if isinstance(scores, torch.Tensor):
                    scores = scores.cpu().numpy()
                
                return keypoints, scores, pose_time
            else:
                return np.zeros((133, 2)), np.zeros(133), pose_time
                
        except Exception as e:
            logger.error(f"❌ 포즈 추정 실패: {e}")
            return np.zeros((133, 2)), np.zeros(133), 0.0

class PoseServer:
    """포즈 추정 서버"""
    
    def __init__(self, 
                 rtmw_config: str,
                 rtmw_checkpoint: str,
                 device: str = "auto",
                 port: int = 5000,
                 host: str = "0.0.0.0"):
        
        self.port = port
        self.host = host
        
        # RTMW 추정기 초기화
        self.estimator = RTMWPoseEstimator(rtmw_config, rtmw_checkpoint, device)
        
        # Flask 앱 설정
        self.app = Flask(__name__)
        self.setup_routes()
        
        # 성능 통계
        self.request_count = 0
        self.processing_times = deque(maxlen=100)
        
        print(f"✅ 포즈 서버 초기화 완료")
        print(f"   - 디바이스: {self.estimator.device}")
        print(f"   - 서버 주소: {host}:{port}")
    
    def setup_routes(self):
        """Flask 라우트 설정"""
        
        @self.app.route('/health', methods=['GET'])
        def health_check():
            """헬스 체크"""
            return jsonify({
                'status': 'healthy',
                'device': self.estimator.device,
                'request_count': self.request_count,
                'avg_processing_time': np.mean(self.processing_times) if self.processing_times else 0
            })
        
        @self.app.route('/estimate_pose', methods=['POST'])
        def estimate_pose():
            """포즈 추정 엔드포인트"""
            try:
                start_time = time.time()
                
                # 요청 데이터 검증
                if 'image' not in request.files:
                    return jsonify({'error': 'No image provided'}), 400
                
                image_file = request.files['image']
                if image_file.filename == '':
                    return jsonify({'error': 'Empty image file'}), 400
                
                # 메타데이터 파싱
                frame_id = request.form.get('frame_id', 0)
                bbox_str = request.form.get('bbox', '[]')
                timestamp = float(request.form.get('timestamp', time.time()))
                
                try:
                    bbox = json.loads(bbox_str)
                except:
                    bbox = []
                
                # 이미지 디코딩
                image_data = image_file.read()
                nparr = np.frombuffer(image_data, np.uint8)
                crop_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if crop_image is None:
                    return jsonify({'error': 'Failed to decode image'}), 400
                
                # 크기 검증 (RTMW 입력 크기: 288x384)
                h, w = crop_image.shape[:2]
                if w != 288 or h != 384:
                    logger.warning(f"⚠️ 예상과 다른 크롭 크기: {w}x{h} (예상: 288x384)")
                
                # 포즈 추정
                keypoints, scores, pose_time = self.estimator.estimate_pose_on_crop(crop_image)
                
                # 응답 데이터 구성
                total_time = time.time() - start_time
                self.processing_times.append(total_time)
                self.request_count += 1
                
                response = {
                    'frame_id': frame_id,
                    'keypoints': keypoints.tolist(),
                    'scores': scores.tolist(),
                    'bbox': bbox,
                    'processing_time': pose_time,
                    'total_time': total_time,
                    'timestamp': timestamp,
                    'received_at': start_time,
                    'device': self.estimator.device,
                    'image_size': [w, h]
                }
                
                # 주기적 로그 출력
                if self.request_count % 30 == 0:
                    avg_time = np.mean(self.processing_times) if self.processing_times else 0
                    logger.info(f"📊 요청 {self.request_count}: 평균 처리시간 {avg_time*1000:.1f}ms")
                
                return jsonify(response)
                
            except Exception as e:
                logger.error(f"❌ 포즈 추정 요청 실패: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/stats', methods=['GET'])
        def get_stats():
            """통계 정보"""
            return jsonify({
                'request_count': self.request_count,
                'device': self.estimator.device,
                'processing_times': {
                    'count': len(self.processing_times),
                    'mean': float(np.mean(self.processing_times)) if self.processing_times else 0,
                    'min': float(np.min(self.processing_times)) if self.processing_times else 0,
                    'max': float(np.max(self.processing_times)) if self.processing_times else 0,
                    'std': float(np.std(self.processing_times)) if self.processing_times else 0
                }
            })
    
    def run(self):
        """서버 실행"""
        print(f"\n🚀 포즈 추정 서버 시작")
        print(f"   - 주소: http://{self.host}:{self.port}")
        print(f"   - 헬스체크: http://{self.host}:{self.port}/health")
        print(f"   - 통계: http://{self.host}:{self.port}/stats")
        print(f"   - Ctrl+C로 종료")
        
        try:
            # Flask 앱 실행 (디버그 모드 비활성화, 프로덕션용)
            self.app.run(
                host=self.host,
                port=self.port,
                debug=False,
                threaded=True,  # 멀티스레드 지원
                use_reloader=False
            )
        except KeyboardInterrupt:
            print("\n⏹️ 포즈 서버 종료")
        except Exception as e:
            logger.error(f"❌ 서버 실행 실패: {e}")

def main():
    parser = argparse.ArgumentParser(description="RTMW Pose Estimation Server")
    parser.add_argument("--config", type=str, 
                       default="configs/wholebody_2d_keypoint/rtmpose/cocktail14/rtmw-l_8xb320-270e_cocktail14-384x288.py",
                       help="RTMW 설정 파일 경로")
    parser.add_argument("--checkpoint", type=str,
                       default="models/rtmw-dw-x-l_simcc-cocktail14_270e-384x288-f840f204_20231122.pth",
                       help="RTMW 체크포인트 파일 경로") 
    parser.add_argument("--device", type=str, default="auto",
                       choices=["auto", "cpu", "cuda", "xpu"],
                       help="추론 디바이스 (기본값: auto)")
    parser.add_argument("--port", type=int, default=5000,
                       help="서버 포트 (기본값: 5000)")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                       help="서버 호스트 (기본값: 0.0.0.0)")
    
    args = parser.parse_args()
    
    # 파일 존재 확인
    if not Path(args.config).exists():
        print(f"❌ 설정 파일 없음: {args.config}")
        return
    
    if not Path(args.checkpoint).exists():
        print(f"❌ 체크포인트 파일 없음: {args.checkpoint}")
        return
    
    try:
        pose_server = PoseServer(
            rtmw_config=args.config,
            rtmw_checkpoint=args.checkpoint,
            device=args.device,
            port=args.port,
            host=args.host
        )
        
        pose_server.run()
        
    except Exception as e:
        print(f"❌ 포즈 서버 실행 실패: {e}")

if __name__ == "__main__":
    main()