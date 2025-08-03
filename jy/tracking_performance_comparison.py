#!/usr/bin/env python3
"""
트래킹 vs 일반 시스템 성능 비교
수화 인식을 위한 안정성과 속도 분석
"""

import os
import cv2
import numpy as np
import time
import json
from typing import Dict, List, Any
import matplotlib.pyplot as plt

# 시스템 임포트
from final_optimized_inferencer import FinalOptimizedInferencer
from yolo11l_xpu_hybrid_inferencer import YOLO11LXPUHybridInferencer
from tracking_yolo11l_hybrid import TrackingYOLO11LHybridInferencer

def create_test_sequence(base_image_path: str, num_frames: int = 60) -> List[np.ndarray]:
    """테스트 시퀀스 생성 (실제 비디오 시뮬레이션)"""
    print(f"📷 테스트 시퀀스 생성: {num_frames}프레임")
    
    base_image = cv2.imread(base_image_path)
    if base_image is None:
        raise ValueError(f"이미지 로드 실패: {base_image_path}")
    
    frames = []
    h, w = base_image.shape[:2]
    
    # 약간의 변화를 주어 실제 비디오와 유사하게 만들기
    for i in range(num_frames):
        # 약간의 이동과 크기 변화 시뮬레이션
        dx = int(5 * np.sin(i * 0.1))  # 좌우 이동
        dy = int(3 * np.cos(i * 0.1))  # 상하 이동
        scale = 1.0 + 0.02 * np.sin(i * 0.05)  # 크기 변화
        
        # 변환 행렬
        center = (w//2, h//2)
        M = cv2.getRotationMatrix2D(center, 0, scale)
        M[0, 2] += dx
        M[1, 2] += dy
        
        # 변환 적용
        transformed = cv2.warpAffine(base_image, M, (w, h))
        
        # 약간의 노이즈 추가
        noise = np.random.normal(0, 1, transformed.shape).astype(np.uint8)
        noisy_frame = cv2.add(transformed, noise)
        
        frames.append(noisy_frame)
    
    print(f"✅ {len(frames)}개 프레임 생성 완료")
    return frames

def test_system_on_sequence(system_name: str, inferencer, frames: List[np.ndarray]) -> Dict[str, Any]:
    """시스템을 시퀀스에 대해 테스트"""
    print(f"\n🔧 {system_name} 시퀀스 테스트 ({len(frames)}프레임)")
    
    # 트래킹 시스템인 경우 리셋
    if hasattr(inferencer, 'reset_tracking'):
        inferencer.reset_tracking()
    
    results = {
        'frame_times': [],
        'detection_counts': [],
        'keypoint_counts': [],
        'bbox_areas': [],
        'bbox_stability': [],
        'processing_times': {
            'detection': [],
            'pose': [],
            'total': []
        }
    }
    
    prev_bbox = None
    
    for frame_idx, frame in enumerate(frames):
        start_time = time.time()
        
        # 프레임 처리
        vis_frame, frame_results = inferencer.process_frame(frame)
        
        total_time = time.time() - start_time
        results['frame_times'].append(total_time)
        
        # 결과 분석
        detection_count = len(frame_results)
        results['detection_counts'].append(detection_count)
        
        total_keypoints = 0
        bbox_areas = []
        
        for result in frame_results:
            if len(result) == 3:  # (keypoints, scores, bbox)
                keypoints, scores, bbox = result
            elif len(result) == 4:  # (keypoints, scores, bbox, track_id)
                keypoints, scores, bbox, _ = result
            else:
                continue
            
            # 키포인트 수 계산
            valid_keypoints = np.sum(scores > 0.3)
            total_keypoints += valid_keypoints
            
            # 바운딩박스 면적
            if len(bbox) >= 4:
                area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                bbox_areas.append(area)
                
                # 바운딩박스 안정성 측정
                if prev_bbox is not None:
                    stability = calculate_bbox_stability(prev_bbox, bbox)
                    results['bbox_stability'].append(stability)
                
                prev_bbox = bbox
        
        results['keypoint_counts'].append(total_keypoints)
        results['bbox_areas'].extend(bbox_areas)
        
        # 추론기의 성능 통계 수집
        if hasattr(inferencer, 'inference_times'):
            times = inferencer.inference_times
            if 'detection' in times and times['detection']:
                results['processing_times']['detection'].append(times['detection'][-1])
            if 'pose' in times and times['pose']:
                results['processing_times']['pose'].append(times['pose'][-1])
            if 'total' in times and times['total']:
                results['processing_times']['total'].append(times['total'][-1])
        
        # 진행률 표시
        if (frame_idx + 1) % 10 == 0:
            fps = 1.0 / np.mean(results['frame_times'][-10:])
            print(f"   진행률: {frame_idx+1}/{len(frames)}, FPS: {fps:.1f}")
    
    # 통계 계산
    results['stats'] = {
        'avg_fps': 1.0 / np.mean(results['frame_times']) if results['frame_times'] else 0,
        'avg_detection_count': np.mean(results['detection_counts']) if results['detection_counts'] else 0,
        'avg_keypoint_count': np.mean(results['keypoint_counts']) if results['keypoint_counts'] else 0,
        'avg_bbox_area': np.mean(results['bbox_areas']) if results['bbox_areas'] else 0,
        'bbox_stability_score': np.mean(results['bbox_stability']) if results['bbox_stability'] else 1.0,
        'fps_std': np.std([1.0/t for t in results['frame_times']]) if results['frame_times'] else 0
    }
    
    print(f"✅ {system_name} 완료: {results['stats']['avg_fps']:.1f}fps, 안정성: {results['stats']['bbox_stability_score']:.3f}")
    
    return results

def calculate_bbox_stability(bbox1: List[float], bbox2: List[float]) -> float:
    """바운딩박스 안정성 계산 (0~1, 1이 가장 안정)"""
    if len(bbox1) < 4 or len(bbox2) < 4:
        return 0.0
    
    # 중심점 이동
    center1 = [(bbox1[0] + bbox1[2])/2, (bbox1[1] + bbox1[3])/2]
    center2 = [(bbox2[0] + bbox2[2])/2, (bbox2[1] + bbox2[3])/2]
    center_distance = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
    
    # 크기 변화
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    area_ratio = min(area1, area2) / max(area1, area2) if max(area1, area2) > 0 else 0
    
    # 모양 변화
    aspect1 = (bbox1[2] - bbox1[0]) / (bbox1[3] - bbox1[1]) if bbox1[3] != bbox1[1] else 1
    aspect2 = (bbox2[2] - bbox2[0]) / (bbox2[3] - bbox2[1]) if bbox2[3] != bbox2[1] else 1
    aspect_ratio = min(aspect1, aspect2) / max(aspect1, aspect2) if max(aspect1, aspect2) > 0 else 0
    
    # 종합 안정성 점수
    center_stability = max(0, 1 - center_distance / 100)  # 100픽셀 이상 이동 시 0점
    size_stability = area_ratio
    shape_stability = aspect_ratio
    
    return (center_stability + size_stability + shape_stability) / 3

def compare_systems():
    """시스템 비교 실행"""
    print("🚀 트래킹 vs 일반 시스템 성능 비교 시작")
    
    # 모델 경로
    rtmw_config = "../configs/wholebody_2d_keypoint/rtmpose/cocktail14/rtmw-x_8xb320-270e_cocktail14-384x288.py"
    rtmw_checkpoint = "../models/rtmw-x_simcc-cocktail14_pt-ucoco_270e-384x288-f840f204_20231122.pth"
    
    # 테스트 시퀀스 생성
    test_image = "winter01.jpg"
    frames = create_test_sequence(test_image, num_frames=90)  # 3초 분량 (30fps 기준)
    
    # 테스트할 시스템들
    systems_to_test = [
        {
            'name': '기존 하이브리드 (간단한 검출)',
            'class': FinalOptimizedInferencer,
            'args': {
                'rtmw_config': rtmw_config,
                'rtmw_checkpoint': rtmw_checkpoint,
                'detection_method': 'simple',
                'detection_device': 'auto',
                'pose_device': 'auto'
            }
        },
        {
            'name': 'YOLO11L 하이브리드 (매번 검출)',
            'class': YOLO11LXPUHybridInferencer,
            'args': {
                'rtmw_config': rtmw_config,
                'rtmw_checkpoint': rtmw_checkpoint,
                'detection_device': 'auto',
                'pose_device': 'auto',
                'optimize_for_accuracy': True
            }
        },
        {
            'name': '트래킹 YOLO11L (30프레임 간격)',
            'class': TrackingYOLO11LHybridInferencer,
            'args': {
                'rtmw_config': rtmw_config,
                'rtmw_checkpoint': rtmw_checkpoint,
                'detection_device': 'auto',
                'pose_device': 'auto',
                'redetection_interval': 30,
                'tracking_stability': 0.8
            }
        },
        {
            'name': '트래킹 YOLO11L (60프레임 간격)',
            'class': TrackingYOLO11LHybridInferencer,
            'args': {
                'rtmw_config': rtmw_config,
                'rtmw_checkpoint': rtmw_checkpoint,
                'detection_device': 'auto',
                'pose_device': 'auto',
                'redetection_interval': 60,
                'tracking_stability': 0.9
            }
        }
    ]
    
    # 각 시스템 테스트
    all_results = {}
    
    for system_info in systems_to_test:
        try:
            system_name = system_info['name']
            print(f"\n{'='*60}")
            print(f"🧪 테스트 중: {system_name}")
            print(f"{'='*60}")
            
            # 시스템 초기화
            inferencer = system_info['class'](**system_info['args'])
            
            # 테스트 실행
            results = test_system_on_sequence(system_name, inferencer, frames)
            all_results[system_name] = results
            
            # 메모리 정리
            del inferencer
            
        except Exception as e:
            print(f"❌ {system_name} 테스트 실패: {e}")
            continue
    
    # 결과 분석 및 출력
    print_comparison_results(all_results)
    
    # 결과 저장
    timestamp = int(time.time())
    result_file = f"tracking_vs_normal_comparison_{timestamp}.json"
    
    # JSON 직렬화 가능하도록 변환
    json_results = {}
    for system_name, results in all_results.items():
        json_results[system_name] = {
            'stats': results['stats'],
            'frame_count': len(results['frame_times']),
            'avg_processing_times': {
                key: float(np.mean(times)) if times else 0.0
                for key, times in results['processing_times'].items()
            }
        }
    
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(json_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 상세 결과 저장: {result_file}")
    
    # 시각화 생성
    create_performance_visualization(all_results)

def print_comparison_results(all_results: Dict[str, Dict[str, Any]]):
    """비교 결과 출력"""
    print(f"\n{'='*80}")
    print("📊 트래킹 vs 일반 시스템 성능 비교 결과")
    print(f"{'='*80}")
    
    # 테이블 헤더
    print(f"{'시스템':<35} {'FPS':<8} {'안정성':<8} {'키포인트':<10} {'FPS분산':<8}")
    print("-" * 80)
    
    # 결과 정렬 (FPS 기준)
    sorted_systems = sorted(all_results.items(), key=lambda x: x[1]['stats']['avg_fps'], reverse=True)
    
    for system_name, results in sorted_systems:
        stats = results['stats']
        print(f"{system_name:<35} {stats['avg_fps']:<8.1f} {stats['bbox_stability_score']:<8.3f} "
              f"{stats['avg_keypoint_count']:<10.1f} {stats['fps_std']:<8.2f}")
    
    print("-" * 80)
    
    # 분석
    print(f"\n🏆 성능 분석:")
    
    if sorted_systems:
        # 최고 FPS
        fastest_system = sorted_systems[0]
        print(f"   🚀 최고 속도: {fastest_system[0]} ({fastest_system[1]['stats']['avg_fps']:.1f} FPS)")
        
        # 최고 안정성
        most_stable = max(sorted_systems, key=lambda x: x[1]['stats']['bbox_stability_score'])
        print(f"   🎯 최고 안정성: {most_stable[0]} (안정성: {most_stable[1]['stats']['bbox_stability_score']:.3f})")
        
        # 가장 일관된 성능
        most_consistent = min(sorted_systems, key=lambda x: x[1]['stats']['fps_std'])
        print(f"   📊 가장 일관된 성능: {most_consistent[0]} (FPS 분산: {most_consistent[1]['stats']['fps_std']:.2f})")
    
    # 트래킹 시스템의 장점 분석
    print(f"\n💡 트래킹 시스템 장점:")
    
    tracking_systems = [name for name in all_results.keys() if '트래킹' in name]
    normal_systems = [name for name in all_results.keys() if '트래킹' not in name]
    
    if tracking_systems and normal_systems:
        # 안정성 비교
        tracking_stability = np.mean([all_results[name]['stats']['bbox_stability_score'] for name in tracking_systems])
        normal_stability = np.mean([all_results[name]['stats']['bbox_stability_score'] for name in normal_systems])
        
        print(f"   📐 바운딩박스 안정성: 트래킹 {tracking_stability:.3f} vs 일반 {normal_stability:.3f}")
        
        # FPS 일관성 비교
        tracking_consistency = np.mean([all_results[name]['stats']['fps_std'] for name in tracking_systems])
        normal_consistency = np.mean([all_results[name]['stats']['fps_std'] for name in normal_systems])
        
        print(f"   ⚡ FPS 일관성: 트래킹 {tracking_consistency:.2f} vs 일반 {normal_consistency:.2f} (낮을수록 좋음)")
    
    print(f"\n🤟 수화 인식을 위한 추천:")
    print(f"   🎯 안정성 우선: 트래킹 기반 시스템 (60프레임 간격)")
    print(f"   ⚡ 속도 우선: 기존 하이브리드 시스템")
    print(f"   ⚖️ 균형: 트래킹 기반 시스템 (30프레임 간격)")

def create_performance_visualization(all_results: Dict[str, Dict[str, Any]]):
    """성능 시각화 생성"""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.font_manager as fm
        
        # 한글 폰트 설정
        plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.rcParams['axes.unicode_minus'] = False
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        systems = list(all_results.keys())
        
        # 1. FPS 비교
        fps_values = [all_results[sys]['stats']['avg_fps'] for sys in systems]
        bars1 = ax1.bar(systems, fps_values, color=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
        ax1.set_title('Average FPS Comparison', fontsize=14, fontweight='bold')
        ax1.set_ylabel('FPS')
        ax1.tick_params(axis='x', rotation=45)
        for bar, fps in zip(bars1, fps_values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    f'{fps:.1f}', ha='center', va='bottom')
        
        # 2. 안정성 비교
        stability_values = [all_results[sys]['stats']['bbox_stability_score'] for sys in systems]
        bars2 = ax2.bar(systems, stability_values, color=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
        ax2.set_title('Bounding Box Stability Comparison', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Stability Score (0-1)')
        ax2.tick_params(axis='x', rotation=45)
        ax2.set_ylim(0, 1)
        for bar, stability in zip(bars2, stability_values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                    f'{stability:.3f}', ha='center', va='bottom')
        
        # 3. FPS 일관성 (분산)
        fps_std_values = [all_results[sys]['stats']['fps_std'] for sys in systems]
        bars3 = ax3.bar(systems, fps_std_values, color=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
        ax3.set_title('FPS Consistency (Lower is Better)', fontsize=14, fontweight='bold')
        ax3.set_ylabel('FPS Standard Deviation')
        ax3.tick_params(axis='x', rotation=45)
        for bar, std in zip(bars3, fps_std_values):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                    f'{std:.2f}', ha='center', va='bottom')
        
        # 4. 키포인트 수 비교
        keypoint_values = [all_results[sys]['stats']['avg_keypoint_count'] for sys in systems]
        bars4 = ax4.bar(systems, keypoint_values, color=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
        ax4.set_title('Average Keypoints Detected', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Keypoint Count')
        ax4.tick_params(axis='x', rotation=45)
        for bar, kpts in zip(bars4, keypoint_values):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f'{kpts:.0f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # 저장
        timestamp = int(time.time())
        plot_file = f"tracking_performance_comparison_{timestamp}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"📈 성능 비교 그래프 저장: {plot_file}")
        plt.close()
        
    except ImportError:
        print("⚠️ matplotlib 미설치 - 시각화 생략")
    except Exception as e:
        print(f"⚠️ 시각화 생성 실패: {e}")

def main():
    """메인 함수"""
    try:
        compare_systems()
    except KeyboardInterrupt:
        print("\n⏹️ 사용자가 비교를 중단했습니다.")
    except Exception as e:
        print(f"\n❌ 비교 실행 중 오류: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
