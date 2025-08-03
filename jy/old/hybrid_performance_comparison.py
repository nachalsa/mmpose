#!/usr/bin/env python3
"""
하이브리드 포즈 추정 시스템 성능 비교
기존 방식 vs YOLO11 XPU 하이브리드 방식
"""

import time
import cv2
import numpy as np
import os
from typing import Dict, List, Tuple

# 기존 하이브리드 추론기
from rtmw_hybrid_xpu_inferencer import RTMWHybridInferencer

# 새로운 YOLO11 하이브리드 추론기  
from yolo11_xpu_hybrid_inferencer import YOLO11XPUHybridInferencer

class HybridPerformanceComparison:
    """하이브리드 시스템 성능 비교"""
    
    def __init__(self):
        self.rtmw_config = "../configs/wholebody_2d_keypoint/rtmpose/cocktail14/rtmw-x_8xb320-270e_cocktail14-384x288.py"
        self.rtmw_checkpoint = "../models/rtmw-x_simcc-cocktail14_pt-ucoco_270e-384x288-f840f204_20231122.pth"
        
        self.test_images = []
        self.results = {}
        
    def setup_test_images(self):
        """테스트 이미지 준비"""
        print("📷 테스트 이미지 준비 중...")
        
        # 기본 테스트 이미지
        base_images = ["winter01.jpg", "winter01 copy.jpg"]
        
        for img_name in base_images:
            img_path = os.path.join(".", img_name)
            if os.path.exists(img_path):
                image = cv2.imread(img_path)
                if image is not None:
                    self.test_images.append((img_name, image))
                    print(f"   ✅ {img_name}: {image.shape}")
        
        # 추가 해상도 테스트용 이미지 생성
        if self.test_images:
            base_name, base_image = self.test_images[0]
            
            # 다양한 해상도로 리사이즈
            resolutions = [
                ("640x480", (640, 480)),
                ("1280x720", (1280, 720)),
                ("1920x1080", (1920, 1080)),
            ]
            
            for res_name, (w, h) in resolutions:
                resized = cv2.resize(base_image, (w, h))
                self.test_images.append((f"{base_name}_{res_name}", resized))
                print(f"   ✅ {base_name}_{res_name}: {resized.shape}")
        
        print(f"📊 총 {len(self.test_images)}개 테스트 이미지 준비 완료")
    
    def benchmark_legacy_system(self, num_runs: int = 5) -> Dict:
        """기존 하이브리드 시스템 벤치마크"""
        print(f"\n🔧 기존 하이브리드 시스템 벤치마크 ({num_runs}회)")
        
        try:
            # 기존 시스템 초기화
            legacy_inferencer = RTMWHybridInferencer(
                config_path=self.rtmw_config,
                checkpoint_path=self.rtmw_checkpoint
            )
            
            results = {}
            
            for img_name, image in self.test_images:
                print(f"   📷 테스트 중: {img_name}")
                
                times = []
                detection_counts = []
                keypoint_counts = []
                
                # 워밍업
                for _ in range(2):
                    try:
                        bboxes = legacy_inferencer.simple_person_detection(image)
                        if bboxes:
                            legacy_inferencer.estimate_pose(image, bboxes[0])
                    except Exception as e:
                        print(f"      ⚠️ 워밍업 실패: {e}")
                
                # 실제 벤치마크
                for run in range(num_runs):
                    start_time = time.time()
                    
                    try:
                        # 사람 검출
                        bboxes = legacy_inferencer.simple_person_detection(image)
                        detection_count = len(bboxes)
                        
                        # 포즈 추정
                        total_keypoints = 0
                        if bboxes:
                            keypoints, scores = legacy_inferencer.estimate_pose(image, bboxes[0])
                            valid_keypoints = np.sum(scores > 0.3)
                            total_keypoints = valid_keypoints
                        
                        end_time = time.time()
                        times.append(end_time - start_time)
                        detection_counts.append(detection_count)
                        keypoint_counts.append(total_keypoints)
                        
                    except Exception as e:
                        print(f"      ❌ 런 {run+1} 실패: {e}")
                        times.append(float('inf'))
                        detection_counts.append(0)
                        keypoint_counts.append(0)
                
                # 통계 계산
                valid_times = [t for t in times if t != float('inf')]
                if valid_times:
                    results[img_name] = {
                        'avg_time': np.mean(valid_times),
                        'std_time': np.std(valid_times),
                        'min_time': np.min(valid_times),
                        'max_time': np.max(valid_times),
                        'fps': 1.0 / np.mean(valid_times),
                        'success_rate': len(valid_times) / num_runs,
                        'avg_detections': np.mean(detection_counts),
                        'avg_keypoints': np.mean(keypoint_counts)
                    }
                    
                    print(f"      ✅ 평균: {np.mean(valid_times)*1000:.1f}ms, FPS: {1.0/np.mean(valid_times):.1f}")
                else:
                    results[img_name] = {'error': 'All runs failed'}
                    print(f"      ❌ 모든 런 실패")
            
            return results
            
        except Exception as e:
            print(f"❌ 기존 시스템 벤치마크 실패: {e}")
            return {}
    
    def benchmark_yolo11_system(self, num_runs: int = 5) -> Dict:
        """YOLO11 하이브리드 시스템 벤치마크"""
        print(f"\n🚀 YOLO11 하이브리드 시스템 벤치마크 ({num_runs}회)")
        
        try:
            # YOLO11 시스템 초기화
            yolo11_inferencer = YOLO11XPUHybridInferencer(
                rtmw_config=self.rtmw_config,
                rtmw_checkpoint=self.rtmw_checkpoint,
                yolo_model="yolo11m.pt",
                detection_device="auto",
                pose_device="auto"
            )
            
            results = {}
            
            for img_name, image in self.test_images:
                print(f"   📷 테스트 중: {img_name}")
                
                times = []
                detection_counts = []
                keypoint_counts = []
                
                # 워밍업
                for _ in range(2):
                    try:
                        yolo11_inferencer.process_frame(image, conf_thresh=0.5)
                    except Exception as e:
                        print(f"      ⚠️ 워밍업 실패: {e}")
                
                # 실제 벤치마크
                for run in range(num_runs):
                    start_time = time.time()
                    
                    try:
                        # 통합 처리
                        vis_image, pose_results = yolo11_inferencer.process_frame(image, conf_thresh=0.5)
                        
                        end_time = time.time()
                        times.append(end_time - start_time)
                        detection_counts.append(len(pose_results))
                        
                        # 키포인트 수 계산
                        total_keypoints = 0
                        for keypoints, scores, bbox in pose_results:
                            valid_keypoints = np.sum(scores > 0.3)
                            total_keypoints += valid_keypoints
                        keypoint_counts.append(total_keypoints)
                        
                    except Exception as e:
                        print(f"      ❌ 런 {run+1} 실패: {e}")
                        times.append(float('inf'))
                        detection_counts.append(0)
                        keypoint_counts.append(0)
                
                # 통계 계산
                valid_times = [t for t in times if t != float('inf')]
                if valid_times:
                    results[img_name] = {
                        'avg_time': np.mean(valid_times),
                        'std_time': np.std(valid_times),
                        'min_time': np.min(valid_times),
                        'max_time': np.max(valid_times),
                        'fps': 1.0 / np.mean(valid_times),
                        'success_rate': len(valid_times) / num_runs,
                        'avg_detections': np.mean(detection_counts),
                        'avg_keypoints': np.mean(keypoint_counts)
                    }
                    
                    print(f"      ✅ 평균: {np.mean(valid_times)*1000:.1f}ms, FPS: {1.0/np.mean(valid_times):.1f}")
                else:
                    results[img_name] = {'error': 'All runs failed'}
                    print(f"      ❌ 모든 런 실패")
            
            return results
            
        except Exception as e:
            print(f"❌ YOLO11 시스템 벤치마크 실패: {e}")
            return {}
    
    def compare_results(self, legacy_results: Dict, yolo11_results: Dict):
        """결과 비교 분석"""
        print(f"\n📊 성능 비교 분석")
        print("=" * 80)
        
        # 테이블 헤더
        print(f"{'이미지':<20} {'기존 FPS':<10} {'YOLO11 FPS':<12} {'향상률':<10} {'검출 정확도':<12}")
        print("-" * 80)
        
        overall_legacy_fps = []
        overall_yolo11_fps = []
        
        for img_name in self.test_images:
            img_name_key = img_name[0]  # 튜플의 첫 번째 요소 (이름)
            
            legacy_result = legacy_results.get(img_name_key, {})
            yolo11_result = yolo11_results.get(img_name_key, {})
            
            if 'error' not in legacy_result and 'error' not in yolo11_result:
                legacy_fps = legacy_result.get('fps', 0)
                yolo11_fps = yolo11_result.get('fps', 0)
                
                improvement = (yolo11_fps / legacy_fps - 1) * 100 if legacy_fps > 0 else 0
                
                legacy_detections = legacy_result.get('avg_detections', 0)
                yolo11_detections = yolo11_result.get('avg_detections', 0)
                
                print(f"{img_name_key:<20} {legacy_fps:<10.1f} {yolo11_fps:<12.1f} {improvement:<10.1f}% {yolo11_detections:<12.1f}")
                
                overall_legacy_fps.append(legacy_fps)
                overall_yolo11_fps.append(yolo11_fps)
            else:
                print(f"{img_name_key:<20} {'ERROR':<10} {'ERROR':<12} {'N/A':<10} {'N/A':<12}")
        
        print("-" * 80)
        
        # 전체 평균
        if overall_legacy_fps and overall_yolo11_fps:
            avg_legacy_fps = np.mean(overall_legacy_fps)
            avg_yolo11_fps = np.mean(overall_yolo11_fps)
            avg_improvement = (avg_yolo11_fps / avg_legacy_fps - 1) * 100
            
            print(f"{'평균':<20} {avg_legacy_fps:<10.1f} {avg_yolo11_fps:<12.1f} {avg_improvement:<10.1f}%")
            
            print(f"\n🏆 성능 개선 요약:")
            print(f"   - 기존 시스템 평균 FPS: {avg_legacy_fps:.1f}")
            print(f"   - YOLO11 하이브리드 평균 FPS: {avg_yolo11_fps:.1f}")
            print(f"   - 성능 향상: {avg_improvement:.1f}%")
            
            # 디바이스별 정보
            print(f"\n💻 디바이스 정보:")
            print(f"   - 기존 시스템: 간단한 검출 + XPU 포즈 추정")
            print(f"   - YOLO11 시스템: XPU 검출 + XPU 포즈 추정")
    
    def run_full_comparison(self):
        """전체 비교 실행"""
        print("🚀 하이브리드 포즈 추정 시스템 성능 비교 시작")
        
        # 테스트 이미지 준비
        self.setup_test_images()
        
        if not self.test_images:
            print("❌ 테스트 이미지가 없습니다")
            return
        
        # 기존 시스템 벤치마크
        legacy_results = self.benchmark_legacy_system(num_runs=5)
        
        # YOLO11 시스템 벤치마크  
        yolo11_results = self.benchmark_yolo11_system(num_runs=5)
        
        # 결과 비교
        self.compare_results(legacy_results, yolo11_results)
        
        # 상세 결과 저장
        self.save_detailed_results(legacy_results, yolo11_results)
        
        print("\n✅ 성능 비교 완료")
    
    def save_detailed_results(self, legacy_results: Dict, yolo11_results: Dict):
        """상세 결과 저장"""
        try:
            import json
            
            detailed_results = {
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'legacy_system': legacy_results,
                'yolo11_system': yolo11_results,
                'test_info': {
                    'num_images': len(self.test_images),
                    'rtmw_config': self.rtmw_config,
                    'rtmw_checkpoint': os.path.basename(self.rtmw_checkpoint)
                }
            }
            
            output_file = f"hybrid_performance_comparison_{int(time.time())}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(detailed_results, f, indent=2, ensure_ascii=False)
            
            print(f"💾 상세 결과 저장: {output_file}")
            
        except Exception as e:
            print(f"⚠️ 결과 저장 실패: {e}")

def main():
    """메인 함수"""
    try:
        comparison = HybridPerformanceComparison()
        comparison.run_full_comparison()
        
    except Exception as e:
        print(f"❌ 비교 실행 실패: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
