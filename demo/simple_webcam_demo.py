#!/usr/bin/env python3
# Copyright (c) OpenMMLab. All rights reserved.
"""
RTMPose-x 384x288 Intel XPU 웹캠 데모

빠른 시작:
    python demo/simple_webcam_demo.py --device xpu:0

요구사항:
    - OpenCV
    - PyTorch with XPU support
    - Intel Extension for PyTorch
    - MMPose
    - 웹캠
"""

import cv2
import time
import argparse
import sys
import os

# Intel XPU 지원 먼저 import
try:
    import torch
    XPU_AVAILABLE = torch.xpu.is_available()
    if XPU_AVAILABLE:
        print(f"Intel XPU 지원됨: {torch.xpu.get_device_name(0)}")
    else:
        print("Intel XPU를 찾을 수 없습니다.")
except Exception as e:
    XPU_AVAILABLE = False
    print(f"Intel XPU 초기화 실패: {e}")

# MMPose 경로 추가
# sys.path.insert(0, '/home/ty/rtmw/02')
sys.path.insert(0, '/home/ty/rtmw/02/mmpose')

# pycocotools를 xtcocotools로 alias (MMPose 호환성)
try:
    import pycocotools
    import pycocotools.coco
    import pycocotools.cocoeval
    import pycocotools.mask
    
    sys.modules['xtcocotools'] = pycocotools
    sys.modules['xtcocotools.coco'] = pycocotools.coco
    sys.modules['xtcocotools.cocoeval'] = pycocotools.cocoeval
    sys.modules['xtcocotools.mask'] = pycocotools.mask
except ImportError:
    print("pycocotools import 실패")

# 간단한 Inferencer 대신 직접 모델 로딩
try:
    from mmpose.apis.inferencers import MMPoseInferencer
    MMPOSE_AVAILABLE = True
except ImportError as e:
    print(f"MMPose Inferencer import 실패: {e}")
    MMPOSE_AVAILABLE = False


def main():
    parser = argparse.ArgumentParser(description='RTMPose-x Intel XPU Webcam Demo')
    parser.add_argument('--camera', type=int, default=0, help='Camera ID')
    parser.add_argument('--device', type=str, default='xpu:0', help='Device (xpu:0, cuda:0 or cpu)')
    parser.add_argument('--config', type=str, 
                       default='/home/ty/rtmw/02/mmpose/configs/body_2d_keypoint/rtmpose/body8/rtmpose-x_8xb256-700e_body8-halpe26-384x288.py',
                       help='Config file path')
    parser.add_argument('--checkpoint', type=str,
                       default='https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-x_simcc-body7_pt-body7-halpe26_700e-384x288-7fb6e239_20230606.pth',
                       help='Checkpoint file path')
    parser.add_argument('--force-xpu', action='store_true', help='XPU 강제 사용 (NMS 오류 무시)')
    args = parser.parse_args()

    # 디바이스 정보 출력
    print(f"요청된 디바이스: {args.device}")
    
    # XPU NMS 문제 때문에 일시적으로 CPU 사용 권장
    if args.device.startswith('xpu'):
        if not args.force_xpu:
            print("\n⚠️  경고: Intel XPU에서 NMS 연산이 지원되지 않아 CPU로 전환합니다.")
            print("XPU를 강제로 사용하려면 --force-xpu 옵션을 추가하세요.")
            print("(단, 추론 중 오류가 발생할 수 있습니다)\n")
            args.device = 'cpu'
        else:
            print("\n⚠️  XPU 강제 사용 모드: NMS 오류가 발생할 수 있습니다.\n")
    
    if args.device.startswith('xpu') and not XPU_AVAILABLE:
        print("경고: XPU를 요청했지만 사용할 수 없습니다. CPU로 변경합니다.")
        args.device = 'cpu'

    if not MMPOSE_AVAILABLE:
        print("오류: MMPose를 import할 수 없습니다.")
        return

    print(f"최종 사용 디바이스: {args.device}")

    # RTMPose-x 모델 초기화 (여러 detection 모델 시도)
    print("RTMPose-x 모델 로딩 중...")
    detection_models = [
        'yolox_s_8x8_300e_coco',  # YOLOX-S
        'yolox_tiny_8x8_300e_coco',  # YOLOX-Tiny
        'rtmdet_tiny_8xb32-300e_coco',  # RTMDet-Tiny
        'yolox',  # 기본 YOLOX
        None  # Detection 없이 수동 bbox
    ]
    
    inferencer = None
    for det_model in detection_models:
        try:
            print(f"Detection 모델 시도: {det_model}")
            inferencer = MMPoseInferencer(
                pose2d=args.config,
                pose2d_weights=args.checkpoint,
                det_model=det_model,
                device=args.device,
                show_progress=False
            )
            print(f"모델 로딩 완료! (Detection: {det_model})")
            break
        except Exception as e:
            print(f"Detection 모델 {det_model} 실패: {e}")
            continue
    
    if inferencer is None:
        print("모든 detection 모델 로딩 실패. detection 없이 시도합니다.")
        try:
            # Detection 없이 전체 이미지에서 포즈 추정
            inferencer = MMPoseInferencer(
                pose2d=args.config,
                pose2d_weights=args.checkpoint,
                device=args.device,
                show_progress=False
            )
            print("Detection 없는 모델 로딩 완료!")
        except Exception as e:
            print(f"모델 로딩 완전 실패: {e}")
            return

    # 웹캠 초기화
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"웹캠 {args.camera}를 열 수 없습니다.")
        return

    print(f"\n=== RTMPose-x 웹캠 데모 ===")
    print(f"디바이스: {args.device}")
    if XPU_AVAILABLE and args.device.startswith('xpu'):
        print(f"XPU 디바이스: {torch.xpu.get_device_name(0)}")
    elif XPU_AVAILABLE:
        print(f"XPU 사용 가능: {torch.xpu.get_device_name(0)} (현재 CPU 사용 중)")
    print("'q' 키를 눌러 종료하세요.")
    print("===================================\n")

    fps_counter = 0
    start_time = time.time()
    error_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("웹캠에서 프레임을 읽을 수 없습니다.")
                break

            # 프레임 좌우 반전 (거울 효과)
            frame = cv2.flip(frame, 1)

            # 포즈 추정 수행
            try:
                # MMPose Inferencer 호출 (return_vis=True로 시각화 결과 받기)
                results = list(inferencer(
                    frame,
                    show=False,
                    return_vis=True,  # 시각화 결과 반환
                    draw_bbox=True,
                    bbox_thr=0.5,
                    kpt_thr=0.3,
                    radius=4,
                    thickness=2
                ))
                
                display_frame = frame.copy()
                
                if results and len(results) > 0:
                    result = results[0]
                    
                    # 방법 1: visualization 키가 있는 경우
                    if isinstance(result, dict) and 'visualization' in result:
                        vis_data = result['visualization']
                        if isinstance(vis_data, list) and len(vis_data) > 0:
                            display_frame = vis_data[0]
                        elif isinstance(vis_data, np.ndarray):
                            display_frame = vis_data
                        print("시각화 데이터 사용됨")
                    
                    # 방법 2: predictions가 있는 경우 수동으로 시각화
                    elif isinstance(result, dict) and 'predictions' in result:
                        predictions = result['predictions']
                        if len(predictions) > 0:
                            # MMPose visualizer 사용해서 직접 그리기
                            try:
                                vis_result = inferencer.visualize(
                                    [frame], 
                                    predictions,
                                    return_vis=True,
                                    show=False,
                                    draw_bbox=True,
                                    radius=4,
                                    thickness=2,
                                    kpt_thr=0.3
                                )
                                if vis_result and len(vis_result) > 0:
                                    display_frame = vis_result[0]
                                    print("수동 시각화 성공")
                            except Exception as vis_e:
                                print(f"수동 시각화 실패: {vis_e}")
                                # 간단한 키포인트 그리기
                                display_frame = draw_simple_pose(frame, predictions)
                    
                    # 포즈 검출 상태 표시
                    cv2.putText(display_frame, f"Poses: {len(predictions) if 'predictions' in locals() else 'Unknown'}", 
                               (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                else:
                    cv2.putText(display_frame, "No Pose Detected", (10, 120),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    
                error_count = 0  # 성공하면 오류 카운터 리셋
                
            except Exception as e:
                error_count += 1
                if error_count <= 3:  # 처음 3개 오류만 상세 출력
                    print(f"추론 오류: {e}")
                    
                    # 디버깅: 결과 구조 확인
                    try:
                        debug_results = list(inferencer(frame, show=False, return_vis=False))
                        print(f"결과 개수: {len(debug_results)}")
                        if debug_results:
                            print(f"첫 번째 결과 타입: {type(debug_results[0])}")
                            if isinstance(debug_results[0], dict):
                                print(f"결과 키들: {list(debug_results[0].keys())}")
                                if 'predictions' in debug_results[0]:
                                    pred_count = len(debug_results[0]['predictions'])
                                    print(f"예측 결과 개수: {pred_count}")
                    except Exception as debug_e:
                        print(f"디버깅 실패: {debug_e}")
                        
                elif error_count == 4:
                    print("추론 오류가 계속 발생합니다. 상세 출력을 중단합니다.")
                    
                display_frame = frame.copy()
                cv2.putText(display_frame, "Inference Error", (10, 120),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # FPS 계산 및 표시
            fps_counter += 1
            if fps_counter % 30 == 0:
                elapsed = time.time() - start_time
                fps = 30 / elapsed
                start_time = time.time()
                print(f"FPS: {fps:.1f} (오류: {error_count})")

            # FPS와 디바이스 정보를 화면에 표시
            device_text = f"Device: {args.device}"
            if XPU_AVAILABLE and not args.device.startswith('xpu'):
                device_text += " (XPU Available)"
                
            cv2.putText(display_frame, device_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            if fps_counter >= 30:
                elapsed = time.time() - start_time + 1
                fps = fps_counter / elapsed
                cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # 오류 발생 시 화면에 표시
            if error_count > 0:
                cv2.putText(display_frame, f"Inference Errors: {error_count}", (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            cv2.imshow('RTMPose-x Demo', display_frame)

            # 'q' 키로 종료
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\n키보드 인터럽트로 종료")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("정리 완료!")


def draw_simple_pose(image, predictions):
    """간단한 포즈 그리기 함수"""
    import numpy as np
    
    display_img = image.copy()
    
    try:
        for pred in predictions:
            if hasattr(pred, 'pred_instances'):
                instances = pred.pred_instances
                if hasattr(instances, 'keypoints'):
                    keypoints = instances.keypoints
                    scores = getattr(instances, 'keypoint_scores', None)
                    
                    # 키포인트 그리기
                    for i, (kpt, score) in enumerate(zip(keypoints[0], scores[0] if scores is not None else [1.0]*len(keypoints[0]))):
                        if score > 0.3:  # 임계값 이상인 키포인트만 그리기
                            x, y = int(kpt[0]), int(kpt[1])
                            if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                                cv2.circle(display_img, (x, y), 4, (0, 255, 0), -1)
                                cv2.putText(display_img, str(i), (x+5, y-5), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
                    
                    # 간단한 스켈레톤 연결 (COCO 17 포인트 기준)
                    skeleton_links = [
                        (0, 1), (0, 2), (1, 3), (2, 4),  # 머리
                        (5, 6), (5, 7), (6, 8), (7, 9), (8, 10),  # 팔
                        (5, 11), (6, 12), (11, 12),  # 몸통
                        (11, 13), (12, 14), (13, 15), (14, 16)  # 다리
                    ]
                    
                    for link in skeleton_links:
                        if link[0] < len(keypoints[0]) and link[1] < len(keypoints[0]):
                            kpt1, kpt2 = keypoints[0][link[0]], keypoints[0][link[1]]
                            score1 = scores[0][link[0]] if scores is not None else 1.0
                            score2 = scores[0][link[1]] if scores is not None else 1.0
                            
                            if score1 > 0.3 and score2 > 0.3:
                                x1, y1 = int(kpt1[0]), int(kpt1[1])
                                x2, y2 = int(kpt2[0]), int(kpt2[1])
                                cv2.line(display_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
    except Exception as e:
        print(f"간단한 포즈 그리기 실패: {e}")
        
    return display_img


if __name__ == '__main__':
    main()
