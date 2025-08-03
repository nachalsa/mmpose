#!/usr/bin/env python3
# Copyright (c) OpenMMLab. All rights reserved.
"""
RTMPose-x WholeBody 133점 Intel XPU 웹캠 데모

빠른 시작:
    python demo/simple_webcam_demo.py --device cpu

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

# Qt 플랫폼 설정 (GUI 문제 해결)
os.environ['QT_QPA_PLATFORM'] = 'xcb'  # X11 사용 강제
# 또는 headless 환경이면
# os.environ['QT_QPA_PLATFORM'] = 'offscreen'

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

# MMPose와 시각화 모듈 import
try:
    from mmpose.apis.inferencers import MMPoseInferencer
    MMPOSE_AVAILABLE = True
except ImportError as e:
    print(f"MMPose Inferencer import 실패: {e}")
    MMPOSE_AVAILABLE = False

# 시각화 모듈 import
try:
    from pose_visualizer import (
        PoseVisualizer, 
        create_fps_counter, 
        update_fps_counter,
        handle_mmpose_visualization
    )
    VISUALIZER_AVAILABLE = True
except ImportError as e:
    print(f"시각화 모듈 import 실패: {e}")
    VISUALIZER_AVAILABLE = False


def test_opencv_display():
    """OpenCV 디스플레이 테스트"""
    print("🔍 OpenCV 디스플레이 테스트 중...")
    try:
        # 간단한 테스트 이미지 생성
        test_img = cv2.imread('/dev/null')  # 존재하지 않는 파일
        if test_img is None:
            import numpy as np
            test_img = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(test_img, "OpenCV Test", (200, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        
        cv2.imshow('OpenCV Test', test_img)
        print("✅ OpenCV 창 생성 성공! 3초 후 자동 닫힘...")
        cv2.waitKey(3000)  # 3초 대기
        cv2.destroyAllWindows()
        return True
    except Exception as e:
        print(f"❌ OpenCV 디스플레이 테스트 실패: {e}")
        return False


def test_webcam():
    """웹캠 접근 테스트"""
    print("📹 웹캠 접근 테스트 중...")
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ 웹캠 열기 실패")
            return False
        
        ret, frame = cap.read()
        if not ret:
            print("❌ 웹캠에서 프레임 읽기 실패")
            cap.release()
            return False
        
        print(f"✅ 웹캠 성공! 프레임 크기: {frame.shape}")
        cap.release()
        return True
    except Exception as e:
        print(f"❌ 웹캠 테스트 실패: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='RTMPose-x WholeBody Intel XPU Webcam Demo')
    parser.add_argument('--camera', type=int, default=0, help='Camera ID')
    parser.add_argument('--device', type=str, default='cpu', help='Device (xpu:0, cuda:0 or cpu)')
    
    # 모델 설정 - WholeBody가 기본값
    parser.add_argument('--wholebody', action='store_true', default=True, 
                       help='WholeBody 133점 모드 사용 (기본값)')
    parser.add_argument('--body-only', action='store_true', 
                       help='Body 17점 모드만 사용 (WholeBody 비활성화)')
    
    # 테스트 옵션 추가
    parser.add_argument('--test-only', action='store_true', 
                       help='시스템 테스트만 수행 (추론 없음)')
    parser.add_argument('--save-frames', action='store_true', 
                       help='프레임을 파일로 저장 (GUI 없음)')
    parser.add_argument('--verbose', action='store_true', 
                       help='상세 디버그 출력')
    
    # Body 17점 모델 설정
    parser.add_argument('--body-config', type=str, 
                       default='/home/ty/rtmw/02/mmpose/configs/body_2d_keypoint/rtmpose/body8/rtmpose-x_8xb256-700e_body8-halpe26-384x288.py',
                       help='Body config file path')
    parser.add_argument('--body-checkpoint', type=str,
                       default='https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-x_simcc-body7_pt-body7-halpe26_700e-384x288-7fb6e239_20230606.pth',
                       help='Body checkpoint file path')
    
    # WholeBody 모델 설정 (기본값)
    parser.add_argument('--wholebody-config', type=str,
                       default='/home/ty/rtmw/02/mmpose/configs/wholebody_2d_keypoint/rtmpose/coco-wholebody/rtmpose-l_8xb32-270e_coco-wholebody-384x288.py',
                       help='WholeBody config file path')
    parser.add_argument('--wholebody-checkpoint', type=str,
                       default='https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-l_simcc-coco-wholebody_pt-aic-coco_270e-384x288-eaeb96c8_20230125.pth',
                       help='WholeBody checkpoint file path')
    
    parser.add_argument('--force-xpu', action='store_true', help='XPU 강제 사용 (NMS 오류 무시)')
    parser.add_argument('--kpt-thr', type=float, default=0.3, help='키포인트 임계값')
    parser.add_argument('--radius', type=int, default=4, help='키포인트 반지름')
    parser.add_argument('--thickness', type=int, default=2, help='스켈레톤 두께')
    parser.add_argument('--show-labels', action='store_true', help='키포인트 번호 표시')
    args = parser.parse_args()

    # 테스트 모드
    if args.test_only:
        print("🔧 시스템 테스트 모드")
        opencv_ok = test_opencv_display()
        webcam_ok = test_webcam()
        
        if opencv_ok and webcam_ok:
            print("✅ 모든 테스트 통과!")
        else:
            print("❌ 일부 테스트 실패")
        return

    # --body-only 옵션이 있으면 wholebody 비활성화
    if args.body_only:
        args.wholebody = False

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
        
    if not VISUALIZER_AVAILABLE:
        print("오류: 시각화 모듈을 import할 수 없습니다.")
        return

    print(f"최종 사용 디바이스: {args.device}")

    # 모델 설정 선택 (기본값은 WholeBody)
    if args.wholebody:
        config = args.wholebody_config
        checkpoint = args.wholebody_checkpoint
        mode_text = "WholeBody 133점"
        print("📍 WholeBody 133점 모드 사용 (기본값)")
    else:
        config = args.body_config
        checkpoint = args.body_checkpoint
        mode_text = "Body 17점"
        print("📍 Body 17점 모드 사용")

    # 시각화 객체 초기화
    visualizer = PoseVisualizer(
        keypoint_threshold=args.kpt_thr,
        keypoint_radius=args.radius,
        skeleton_thickness=args.thickness,
        show_keypoint_labels=args.show_labels,
        wholebody_mode=args.wholebody  # 기본값 True
    )

    # RTMPose 모델 초기화
    print("RTMPose 모델 로딩 중...")
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
                pose2d=config,
                pose2d_weights=checkpoint,
                det_model=det_model,
                device=args.device,
                show_progress=False
            )
            print(f"✅ 모델 로딩 완료! (Detection: {det_model})")
            break
        except Exception as e:
            print(f"❌ Detection 모델 {det_model} 실패: {e}")
            continue
    
    if inferencer is None:
        print("모든 detection 모델 로딩 실패. detection 없이 시도합니다.")
        try:
            # Detection 없이 전체 이미지에서 포즈 추정
            inferencer = MMPoseInferencer(
                pose2d=config,
                pose2d_weights=checkpoint,
                device=args.device,
                show_progress=False
            )
            print("✅ Detection 없는 모델 로딩 완료!")
        except Exception as e:
            print(f"❌ 모델 로딩 완전 실패: {e}")
            return

    # 웹캠 초기화
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"❌ 웹캠 {args.camera}를 열 수 없습니다.")
        return

    # 웹캠 설정
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    print(f"\n{'='*50}")
    print(f"🎥 RTMPose {mode_text} 웹캠 데모")
    print(f"{'='*50}")
    print(f"🖥️  디바이스: {args.device}")
    if XPU_AVAILABLE and args.device.startswith('xpu'):
        print(f"🔥 XPU 디바이스: {torch.xpu.get_device_name(0)}")
    elif XPU_AVAILABLE:
        print(f"💡 XPU 사용 가능: {torch.xpu.get_device_name(0)} (현재 CPU 사용 중)")
    print(f"👤 모드: {mode_text}")
    if args.wholebody:
        print("🎨 WholeBody 색상 구분:")
        print("   - 초록색: Body (17점)")
        print("   - 마젠타: Face (68점)")
        print("   - 시안색: Left Hand (21점)")
        print("   - 노란색: Right Hand (21점)")
        print("   - 주황색: Left Foot (6점)")
        print("   - 보라색: Right Foot (6점)")
    
    if args.save_frames:
        print("💾 프레임 저장 모드 활성화")
    else:
        print("⌨️  'q' 키를 눌러 종료하세요.")
    print(f"{'='*50}\n")

    # FPS 카운터 초기화
    fps_counter = create_fps_counter()
    error_count = 0
    frame_count = 0

    try:
        print("🚀 실시간 추론 시작...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ 웹캠에서 프레임을 읽을 수 없습니다.")
                break

            frame_count += 1
            if args.verbose and frame_count % 30 == 0:
                print(f"📊 처리된 프레임: {frame_count}")

            # 프레임 좌우 반전 (거울 효과)
            frame = cv2.flip(frame, 1)

            # 포즈 추정 수행
            try:
                # MMPose Inferencer 호출
                results = list(inferencer(
                    frame,
                    show=False,
                    return_vis=True,
                    draw_bbox=True,
                    bbox_thr=0.5,
                    kpt_thr=args.kpt_thr,
                    radius=args.radius,
                    thickness=args.thickness
                ))
                
                # 시각화 처리
                display_frame = handle_mmpose_visualization(
                    frame, results, inferencer, visualizer)
                    
                error_count = 0  # 성공하면 오류 카운터 리셋
                
            except Exception as e:
                error_count += 1
                if error_count <= 3:  # 처음 3개 오류만 상세 출력
                    print(f"⚠️ 추론 오류: {e}")
                elif error_count == 4:
                    print("⚠️ 추론 오류가 계속 발생합니다. 상세 출력을 중단합니다.")
                    
                display_frame = frame.copy()
                display_frame = visualizer.add_info_text(
                    display_frame, ["Inference Error"], 
                    position=(10, 120), color=visualizer.COLORS['error'])

            # FPS 계산 및 업데이트
            current_fps = update_fps_counter(fps_counter)
            
            # 시스템 정보를 화면에 표시
            info_texts = []
            
            # 디바이스 정보
            device_text = f"Device: {args.device}"
            if XPU_AVAILABLE and not args.device.startswith('xpu'):
                device_text += " (XPU Available)"
            info_texts.append(device_text)
            
            # 모드 정보
            info_texts.append(f"Mode: {mode_text}")
            
            # FPS 정보
            if current_fps > 0:
                info_texts.append(f"FPS: {current_fps:.1f}")
            
            # 프레임 정보
            info_texts.append(f"Frame: {frame_count}")
            
            # 오류 정보
            if error_count > 0:
                info_texts.append(f"Errors: {error_count}")
            
            # 정보 텍스트 추가
            display_frame = visualizer.add_info_text(display_frame, info_texts)

            # 파일 저장 모드
            if args.save_frames:
                if frame_count % 30 == 0:  # 30프레임마다 저장
                    filename = f"output_frame_{frame_count:06d}.jpg"
                    cv2.imwrite(filename, display_frame)
                    print(f"💾 프레임 저장: {filename}")
                
                # 파일 저장 모드에서는 100프레임 후 종료
                if frame_count >= 100:
                    print("📁 100프레임 저장 완료!")
                    break
            else:
                # GUI 모드
                try:
                    cv2.imshow(f'RTMPose {mode_text} Demo', display_frame)
                    
                    # 'q' 키로 종료
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("\n⌨️ 'q' 키로 종료")
                        break
                    elif key == ord('s'):  # 's' 키로 스크린샷
                        filename = f"screenshot_{int(time.time())}.jpg"
                        cv2.imwrite(filename, display_frame)
                        print(f"📸 스크린샷 저장: {filename}")
                        
                except Exception as gui_e:
                    print(f"⚠️ GUI 오류: {gui_e}")
                    print("💾 파일 저장 모드로 전환합니다...")
                    args.save_frames = True

    except KeyboardInterrupt:
        print("\n⌨️ 키보드 인터럽트로 종료")

    finally:
        cap.release()
        if not args.save_frames:
            cv2.destroyAllWindows()
        print("✅ 정리 완료!")


if __name__ == '__main__':
    main()
