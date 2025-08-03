#!/usr/bin/env python3
"""
종합 성능 테스트 - YOLO11L 포함 확장 버전
기존 시스템 + YOLO11 (n/s/m/l) + 최적화 버전 전체 비교
"""

import os
import cv2
import numpy as np
import json
import time
from typing import Dict, List, Any

# 기존 시스템 임포트
from final_optimized_inferencer import FinalOptimizedInferencer
from yolo11_xpu_hybrid_inferencer import YOLO11XPUHybridInferencer  
from optimized_yolo11_xpu_inferencer import OptimizedYOLO11XPUInferencer
from yolo11l_xpu_hybrid_inferencer import YOLO11LXPUHybridInferencer

def prepare_test_images():
    """테스트 이미지 준비"""
    print("📷 테스트 이미지 준비 중...")
    
    base_image = "winter01.jpg"
    if not os.path.exists(base_image):
        print(f"❌ 기본 이미지 없음: {base_image}")
        return []
    
    # 기본 이미지만 사용 (성능 테스트 중심)
    test_images = [base_image]
    
    for img_path in test_images:
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            if img is not None:
                print(f"   ✅ {img_path}: {img.shape}")
            else:
                print(f"   ❌ {img_path}: 로드 실패")
        else:
            print(f"   ❌ {img_path}: 파일 없음")
    
    print(f"📊 총 {len(test_images)}개 테스트 이미지 준비 완료")
    return test_images

def test_system(system_name: str, inferencer, test_images: List[str], num_runs: int = 5) -> Dict[str, Any]:
    """개별 시스템 테스트"""
    print(f"\n🔧 {system_name} 벤치마크 ({num_runs}회)")
    
    results = {}
    
    for img_path in test_images:
        print(f"   📷 테스트 중: {os.path.basename(img_path)}")
        
        image = cv2.imread(img_path)
        if image is None:
            continue
            
        # 워밍업
        for _ in range(2):
            inferencer.process_frame(image)
        
        # 실제 측정
        times = []
        keypoint_counts = []
        detection_counts = []
        
        for _ in range(num_runs + 2):  # 처음 2개는 제외
            start_time = time.time()
            _, results_data = inferencer.process_frame(image)
            end_time = time.time()
            
            # 처음 2회는 제외 (추가 워밍업)
            if len(times) < num_runs:
                times.append(end_time - start_time)
                detection_counts.append(len(results_data))
                
                # 키포인트 수 계산
                total_keypoints = 0
                for kpts, scores, bbox in results_data:
                    valid_kpts = np.sum(scores > 0.3)
                    total_keypoints += valid_kpts
                keypoint_counts.append(total_keypoints)
        
        # 통계 계산
        if times:
            avg_time = np.mean(times)
            fps = 1.0 / avg_time
            avg_detections = np.mean(detection_counts)
            avg_keypoints = np.mean(keypoint_counts)
            
            results[os.path.basename(img_path)] = {
                'fps': fps,
                'avg_time': avg_time,
                'detections': avg_detections,
                'keypoints': avg_keypoints,
                'times': times
            }
            
            print(f"      ✅ 평균: {avg_time*1000:.1f}ms, FPS: {fps:.1f}")
    
    return results

def run_comprehensive_test():
    """확장된 종합 성능 테스트"""
    print("🚀 하이브리드 포즈 추정 시스템 종합 성능 비교 시작 (YOLO11L 포함)")
    
    # 모델 경로
    rtmw_config = "../configs/wholebody_2d_keypoint/rtmpose/cocktail14/rtmw-x_8xb320-270e_cocktail14-384x288.py"
    rtmw_checkpoint = "../models/rtmw-x_simcc-cocktail14_pt-ucoco_270e-384x288-f840f204_20231122.pth"
    
    # 테스트 이미지 준비
    test_images = prepare_test_images()
    if not test_images:
        print("❌ 테스트 이미지 없음")
        return
    
    # 테스트할 시스템들
    systems_to_test = [
        {
            'name': '기존 하이브리드 (간단한 검출 + XPU 포즈)',
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
            'name': 'YOLO11 하이브리드 (Nano)',
            'class': YOLO11XPUHybridInferencer,
            'args': {
                'rtmw_config': rtmw_config,
                'rtmw_checkpoint': rtmw_checkpoint,
                'yolo_model': 'yolo11n.pt',
                'detection_device': 'auto',
                'pose_device': 'auto'
            }
        },
        {
            'name': 'YOLO11 하이브리드 (Small)',
            'class': YOLO11XPUHybridInferencer,
            'args': {
                'rtmw_config': rtmw_config,
                'rtmw_checkpoint': rtmw_checkpoint,
                'yolo_model': 'yolo11s.pt',
                'detection_device': 'auto',
                'pose_device': 'auto'
            }
        },
        {
            'name': 'YOLO11 하이브리드 (Medium)',
            'class': YOLO11XPUHybridInferencer,
            'args': {
                'rtmw_config': rtmw_config,
                'rtmw_checkpoint': rtmw_checkpoint,
                'yolo_model': 'yolo11m.pt',
                'detection_device': 'auto',
                'pose_device': 'auto'
            }
        },
        {
            'name': 'YOLO11 하이브리드 (Large)',
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
            'name': '최적화된 YOLO11 (Nano)',
            'class': OptimizedYOLO11XPUInferencer,
            'args': {
                'rtmw_config': rtmw_config,
                'rtmw_checkpoint': rtmw_checkpoint,
                'yolo_model': 'yolo11n.pt',
                'detection_device': 'auto',
                'pose_device': 'auto',
                'optimize_for_speed': True
            }
        },
        {
            'name': '최적화된 YOLO11 (Small)',
            'class': OptimizedYOLO11XPUInferencer,
            'args': {
                'rtmw_config': rtmw_config,
                'rtmw_checkpoint': rtmw_checkpoint,
                'yolo_model': 'yolo11s.pt',
                'detection_device': 'auto',
                'pose_device': 'auto',
                'optimize_for_speed': True
            }
        },
        {
            'name': '최적화된 YOLO11 (Medium)',
            'class': OptimizedYOLO11XPUInferencer,
            'args': {
                'rtmw_config': rtmw_config,
                'rtmw_checkpoint': rtmw_checkpoint,
                'yolo_model': 'yolo11m.pt',
                'detection_device': 'auto',
                'pose_device': 'auto',
                'optimize_for_speed': True
            }
        },
        {
            'name': '최적화된 YOLO11 (Large)',
            'class': OptimizedYOLO11XPUInferencer,
            'args': {
                'rtmw_config': rtmw_config,
                'rtmw_checkpoint': rtmw_checkpoint,
                'yolo_model': 'yolo11l.pt',
                'detection_device': 'auto',
                'pose_device': 'auto',
                'optimize_for_speed': False  # Large는 정확도 우선
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
            results = test_system(system_name, inferencer, test_images, num_runs=5)
            all_results[system_name] = results
            
            # 메모리 정리
            del inferencer
            
        except Exception as e:
            print(f"❌ {system_name} 테스트 실패: {e}")
            continue
    
    # 결과 분석 및 출력
    print_comprehensive_comparison(all_results)
    
    # 결과 저장
    timestamp = int(time.time())
    result_file = f"comprehensive_performance_test_with_large_{timestamp}.json"
    
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 상세 결과 저장: {result_file}")

def print_comprehensive_comparison(all_results: Dict[str, Dict[str, Any]]):
    """종합 비교 결과 출력"""
    print(f"\n{'='*100}")
    print("📊 종합 성능 비교 결과 (YOLO11L 포함)")
    print(f"{'='*100}")
    
    # 시스템별 평균 성능 계산
    system_stats = {}
    
    for system_name, results in all_results.items():
        total_fps = 0
        total_keypoints = 0
        total_detections = 0
        image_count = 0
        
        for img_name, img_results in results.items():
            total_fps += img_results['fps']
            total_keypoints += img_results['keypoints']
            total_detections += img_results['detections']
            image_count += 1
        
        if image_count > 0:
            system_stats[system_name] = {
                'avg_fps': total_fps / image_count,
                'avg_keypoints': total_keypoints / image_count,
                'avg_detections': total_detections / image_count
            }
    
    # FPS 기준 정렬
    sorted_systems = sorted(system_stats.items(), key=lambda x: x[1]['avg_fps'], reverse=True)
    
    # 테이블 출력
    print(f"{'시스템':<35} {'FPS':<8} {'키포인트':<10} {'검출수':<8} {'검출방식':<25}")
    print("-" * 100)
    
    for system_name, stats in sorted_systems:
        detection_method = "간단한 박스" if "간단한 검출" in system_name else ""
        if "Nano" in system_name:
            detection_method = f"{'최적화된 ' if '최적화된' in system_name else ''}YOLO11 yolo11n.pt"
        elif "Small" in system_name:
            detection_method = f"{'최적화된 ' if '최적화된' in system_name else ''}YOLO11 yolo11s.pt"
        elif "Medium" in system_name:
            detection_method = f"{'최적화된 ' if '최적화된' in system_name else ''}YOLO11 yolo11m.pt"
        elif "Large" in system_name:
            detection_method = f"{'최적화된 ' if '최적화된' in system_name else ''}YOLO11 yolo11l.pt"
        
        print(f"{system_name:<35} {stats['avg_fps']:<8.1f} {stats['avg_keypoints']:<10.1f} "
              f"{stats['avg_detections']:<8.1f} {detection_method:<25}")
    
    print("-" * 100)
    
    # 성능 분석
    if sorted_systems:
        best_fps_system = sorted_systems[0]
        best_accuracy_system = max(sorted_systems, key=lambda x: x[1]['avg_keypoints'])
        
        print(f"\n🏆 성능 분석:")
        print(f"   🚀 최고 FPS: {best_fps_system[0]} ({best_fps_system[1]['avg_fps']:.1f} FPS)")
        print(f"   🎯 최고 정확도: {best_accuracy_system[0]} ({best_accuracy_system[1]['avg_keypoints']:.1f} 키포인트)")
        
        # 기존 시스템 대비 개선률
        baseline_system = None
        for system_name, stats in sorted_systems:
            if "기존 하이브리드" in system_name:
                baseline_system = (system_name, stats)
                break
        
        if baseline_system:
            baseline_fps = baseline_system[1]['avg_fps']
            print(f"\n📈 기존 시스템 대비 성능 변화:")
            
            improvements = []
            for system_name, stats in sorted_systems:
                if system_name != baseline_system[0]:
                    improvement = ((stats['avg_fps'] - baseline_fps) / baseline_fps) * 100
                    improvements.append((improvement, system_name, stats['avg_fps']))
            
            # 개선률 순 정렬
            improvements.sort(reverse=True)
            
            for improvement, system_name, fps in improvements:
                arrow = "⬆️" if improvement > 0 else "⬇️"
                print(f"   {arrow} {system_name}: {improvement:+.1f}%")
        
        # 추천 시스템
        print(f"\n💡 추천 시스템:")
        
        # 속도 우선
        speed_winner = sorted_systems[0]
        print(f"   🏃 속도 우선: {speed_winner[0]} ({speed_winner[1]['avg_fps']:.1f} FPS)")
        
        # 균형 우선 (종합 점수)
        balance_scores = []
        for system_name, stats in sorted_systems:
            # 종합 점수 = FPS * 키포인트 수
            score = stats['avg_fps'] * stats['avg_keypoints']
            balance_scores.append((score, system_name, stats))
        
        balance_winner = max(balance_scores, key=lambda x: x[0])
        print(f"   ⚖️ 균형 우선: {balance_winner[1]} (종합점수: {balance_winner[0]:.1f})")
        
        # 정확도 우선
        accuracy_winner = max(sorted_systems, key=lambda x: x[1]['avg_keypoints'])
        print(f"   🎯 정확도 우선: {accuracy_winner[0]} ({accuracy_winner[1]['avg_keypoints']:.1f} 키포인트)")
    
    print(f"\n✅ 종합 성능 테스트 완료!")
    print(f"🎯 YOLO11L 추가: 최고 정확도 옵션 제공")
    print(f"⚡ 실시간 용도: 기존 하이브리드 또는 최적화된 YOLO11n 추천")
    print(f"🎪 최고 정확도: YOLO11L 하이브리드 추천")

def main():
    """메인 함수"""
    try:
        run_comprehensive_test()
    except KeyboardInterrupt:
        print("\n⏹️ 사용자가 테스트를 중단했습니다.")
    except Exception as e:
        print(f"\n❌ 테스트 실행 중 오류: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
