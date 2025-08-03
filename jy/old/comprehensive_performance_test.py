#!/usr/bin/env python3
"""
최종 하이브리드 포즈 추정 시스템 성능 종합 비교
모든 버전들의 성능과 정확도 비교
"""

import time
import cv2
import numpy as np
import os
from typing import Dict, List
import json

# 모든 하이브리드 추론기들
from rtmw_hybrid_xpu_inferencer import RTMWHybridInferencer
from yolo11_xpu_hybrid_inferencer import YOLO11XPUHybridInferencer  
from optimized_yolo11_xpu_inferencer import OptimizedYOLO11XPUInferencer

class ComprehensivePerformanceTest:
    """종합 성능 테스트"""
    
    def __init__(self):
        self.rtmw_config = "../configs/wholebody_2d_keypoint/rtmpose/cocktail14/rtmw-x_8xb320-270e_cocktail14-384x288.py"
        self.rtmw_checkpoint = "../models/rtmw-x_simcc-cocktail14_pt-ucoco_270e-384x288-f840f204_20231122.pth"
        
        # 테스트 설정
        self.test_image = "winter01.jpg"
        self.num_runs = 10
        
        # 결과 저장
        self.results = {}
        
    def test_legacy_hybrid(self):
        """기존 하이브리드 시스템 테스트"""
        print("\n🔧 기존 하이브리드 시스템 테스트")
        print("-" * 50)
        
        try:
            inferencer = RTMWHybridInferencer(
                config_path=self.rtmw_config,
                checkpoint_path=self.rtmw_checkpoint
            )
            
            image = cv2.imread(self.test_image)
            if image is None:
                return None
            
            # 워밍업
            for _ in range(3):
                bboxes = inferencer.simple_person_detection(image)
                if bboxes:
                    inferencer.estimate_pose(image, bboxes[0])
            
            # 벤치마크
            times = []
            keypoint_counts = []
            
            for i in range(self.num_runs):
                start_time = time.time()
                
                bboxes = inferencer.simple_person_detection(image)
                keypoints, scores = None, None
                if bboxes:
                    keypoints, scores = inferencer.estimate_pose(image, bboxes[0])
                
                end_time = time.time()
                times.append(end_time - start_time)
                
                if scores is not None:
                    valid_kpts = np.sum(scores > 0.3)
                    keypoint_counts.append(valid_kpts)
                else:
                    keypoint_counts.append(0)
            
            result = {
                'name': '기존 하이브리드 (간단한 검출 + XPU 포즈)',
                'avg_time': np.mean(times),
                'fps': 1.0 / np.mean(times),
                'std_time': np.std(times),
                'avg_keypoints': np.mean(keypoint_counts),
                'detection_method': '간단한 박스',
                'pose_device': 'XPU',
                'model_size': 'N/A'
            }
            
            print(f"✅ 평균 FPS: {result['fps']:.1f}")
            print(f"✅ 평균 키포인트: {result['avg_keypoints']:.1f}/133")
            
            return result
            
        except Exception as e:
            print(f"❌ 기존 하이브리드 테스트 실패: {e}")
            return None
    
    def test_yolo11_hybrid(self, yolo_model: str, description: str):
        """YOLO11 하이브리드 시스템 테스트"""
        print(f"\n🚀 YOLO11 하이브리드 테스트: {description}")
        print("-" * 50)
        
        try:
            inferencer = YOLO11XPUHybridInferencer(
                rtmw_config=self.rtmw_config,
                rtmw_checkpoint=self.rtmw_checkpoint,
                yolo_model=yolo_model,
                detection_device="auto",
                pose_device="auto"
            )
            
            image = cv2.imread(self.test_image)
            if image is None:
                return None
            
            # 워밍업
            for _ in range(3):
                inferencer.process_frame(image)
            
            # 벤치마크
            times = []
            keypoint_counts = []
            detection_counts = []
            
            for i in range(self.num_runs):
                start_time = time.time()
                
                vis_image, results = inferencer.process_frame(image, conf_thresh=0.5)
                
                end_time = time.time()
                times.append(end_time - start_time)
                
                detection_counts.append(len(results))
                
                total_keypoints = 0
                for keypoints, scores, bbox in results:
                    valid_kpts = np.sum(scores > 0.3)
                    total_keypoints += valid_kpts
                keypoint_counts.append(total_keypoints)
            
            result = {
                'name': f'YOLO11 하이브리드 ({description})',
                'avg_time': np.mean(times),
                'fps': 1.0 / np.mean(times),
                'std_time': np.std(times),
                'avg_keypoints': np.mean(keypoint_counts),
                'avg_detections': np.mean(detection_counts),
                'detection_method': f'YOLO11 {yolo_model}',
                'pose_device': 'XPU',
                'model_size': yolo_model
            }
            
            print(f"✅ 평균 FPS: {result['fps']:.1f}")
            print(f"✅ 평균 검출: {result['avg_detections']:.1f}명")
            print(f"✅ 평균 키포인트: {result['avg_keypoints']:.1f}/133")
            
            return result
            
        except Exception as e:
            print(f"❌ YOLO11 하이브리드 테스트 실패: {e}")
            return None
    
    def test_optimized_yolo11(self, yolo_model: str, description: str):
        """최적화된 YOLO11 시스템 테스트"""
        print(f"\n⚡ 최적화된 YOLO11 테스트: {description}")
        print("-" * 50)
        
        try:
            inferencer = OptimizedYOLO11XPUInferencer(
                rtmw_config=self.rtmw_config,
                rtmw_checkpoint=self.rtmw_checkpoint,
                yolo_model=yolo_model,
                detection_device="auto",
                pose_device="auto",
                optimize_for_speed=True
            )
            
            image = cv2.imread(self.test_image)
            if image is None:
                return None
            
            # 워밍업
            for _ in range(3):
                inferencer.process_frame_fast(image)
            
            # 벤치마크
            times = []
            keypoint_counts = []
            detection_counts = []
            
            for i in range(self.num_runs):
                start_time = time.time()
                
                vis_image, results = inferencer.process_frame_fast(image)
                
                end_time = time.time()
                times.append(end_time - start_time)
                
                detection_counts.append(len(results))
                
                total_keypoints = 0
                for keypoints, scores, bbox in results:
                    valid_kpts = np.sum(scores > 0.3)
                    total_keypoints += valid_kpts
                keypoint_counts.append(total_keypoints)
            
            result = {
                'name': f'최적화된 YOLO11 ({description})',
                'avg_time': np.mean(times),
                'fps': 1.0 / np.mean(times),
                'std_time': np.std(times),
                'avg_keypoints': np.mean(keypoint_counts),
                'avg_detections': np.mean(detection_counts),
                'detection_method': f'최적화된 YOLO11 {yolo_model}',
                'pose_device': 'XPU',
                'model_size': yolo_model
            }
            
            print(f"✅ 평균 FPS: {result['fps']:.1f}")
            print(f"✅ 평균 검출: {result['avg_detections']:.1f}명")
            print(f"✅ 평균 키포인트: {result['avg_keypoints']:.1f}/133")
            
            return result
            
        except Exception as e:
            print(f"❌ 최적화된 YOLO11 테스트 실패: {e}")
            return None
    
    def run_comprehensive_test(self):
        """종합 테스트 실행"""
        print("🏁 종합 하이브리드 포즈 추정 시스템 성능 테스트")
        print("=" * 80)
        
        if not os.path.exists(self.test_image):
            print(f"❌ 테스트 이미지 없음: {self.test_image}")
            return
        
        all_results = []
        
        # 1. 기존 하이브리드 시스템
        legacy_result = self.test_legacy_hybrid()
        if legacy_result:
            all_results.append(legacy_result)
        
        # 2. 기본 YOLO11 하이브리드 시스템들
        yolo_models = [
            ("yolo11n.pt", "Nano"),
            ("yolo11s.pt", "Small"), 
            ("yolo11m.pt", "Medium")
        ]
        
        for yolo_model, desc in yolo_models:
            result = self.test_yolo11_hybrid(yolo_model, desc)
            if result:
                all_results.append(result)
        
        # 3. 최적화된 YOLO11 시스템들
        for yolo_model, desc in yolo_models:
            result = self.test_optimized_yolo11(yolo_model, desc)
            if result:
                all_results.append(result)
        
        # 4. 결과 분석 및 출력
        self.analyze_results(all_results)
        
        # 5. 결과 저장
        self.save_results(all_results)
    
    def analyze_results(self, results: List[Dict]):
        """결과 분석 및 출력"""
        print("\n" + "=" * 80)
        print("📊 종합 성능 비교 결과")
        print("=" * 80)
        
        # 테이블 헤더
        print(f"{'시스템':<35} {'FPS':<8} {'키포인트':<10} {'검출수':<8} {'검출방식':<20}")
        print("-" * 80)
        
        # 결과 정렬 (FPS 기준)
        sorted_results = sorted(results, key=lambda x: x['fps'], reverse=True)
        
        for result in sorted_results:
            name = result['name']
            fps = result['fps']
            keypoints = result['avg_keypoints']
            detections = result.get('avg_detections', 1.0)
            detection_method = result['detection_method']
            
            print(f"{name:<35} {fps:<8.1f} {keypoints:<10.1f} {detections:<8.1f} {detection_method:<20}")
        
        print("-" * 80)
        
        # 최고 성능 분석
        best_fps = max(results, key=lambda x: x['fps'])
        best_accuracy = max(results, key=lambda x: x['avg_keypoints'])
        
        print(f"\n🏆 성능 분석:")
        print(f"   🚀 최고 FPS: {best_fps['name']} ({best_fps['fps']:.1f} FPS)")
        print(f"   🎯 최고 정확도: {best_accuracy['name']} ({best_accuracy['avg_keypoints']:.1f} 키포인트)")
        
        # 성능 개선 분석
        legacy_fps = next((r['fps'] for r in results if '기존 하이브리드' in r['name']), None)
        if legacy_fps:
            print(f"\n📈 기존 시스템 대비 개선률:")
            for result in sorted_results:
                if '기존 하이브리드' not in result['name']:
                    improvement = (result['fps'] / legacy_fps - 1) * 100
                    status = "⬆️" if improvement > 0 else "⬇️"
                    print(f"   {status} {result['name']}: {improvement:+.1f}%")
        
        # 추천 시스템
        print(f"\n💡 추천 시스템:")
        
        # 속도 우선
        speed_focused = max(results, key=lambda x: x['fps'])
        print(f"   🏃 속도 우선: {speed_focused['name']} ({speed_focused['fps']:.1f} FPS)")
        
        # 균형 우선 (FPS * 키포인트 수)
        balanced_score = [(r['fps'] * r['avg_keypoints'], r) for r in results]
        balanced_best = max(balanced_score, key=lambda x: x[0])[1]
        print(f"   ⚖️ 균형 우선: {balanced_best['name']} (종합점수: {balanced_best['fps'] * balanced_best['avg_keypoints']:.1f})")
        
        # 정확도 우선
        accuracy_focused = max(results, key=lambda x: x['avg_keypoints'])
        print(f"   🎯 정확도 우선: {accuracy_focused['name']} ({accuracy_focused['avg_keypoints']:.1f} 키포인트)")
    
    def save_results(self, results: List[Dict]):
        """결과 저장"""
        try:
            output_data = {
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'test_image': self.test_image,
                'num_runs': self.num_runs,
                'results': results,
                'summary': {
                    'total_systems_tested': len(results),
                    'best_fps': max(results, key=lambda x: x['fps']),
                    'best_accuracy': max(results, key=lambda x: x['avg_keypoints'])
                }
            }
            
            output_file = f"comprehensive_performance_test_{int(time.time())}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            
            print(f"\n💾 상세 결과 저장: {output_file}")
            
        except Exception as e:
            print(f"⚠️ 결과 저장 실패: {e}")

def main():
    """메인 함수"""
    try:
        tester = ComprehensivePerformanceTest()
        tester.run_comprehensive_test()
        
        print(f"\n✅ 종합 성능 테스트 완료!")
        print(f"🎯 추천: 실시간 용도에는 최적화된 YOLO11s 사용")
        print(f"⚡ 최고 속도가 필요하면 최적화된 YOLO11n 사용")
        print(f"🎪 최고 정확도가 필요하면 최적화된 YOLO11m 사용")
        
    except Exception as e:
        print(f"❌ 종합 테스트 실패: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
