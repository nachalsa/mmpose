#!/usr/bin/env python3
"""
RTMW-x 모델 차원 디버깅
실제 체크포인트의 정확한 차원 분석
"""

import torch
import numpy as np
from collections import OrderedDict
from utils import find_and_download_rtmw_model


def analyze_rtmw_checkpoint():
    """RTMW 체크포인트의 실제 차원 분석"""
    
    # RTMW 모델 찾기
    model_path, model_description = find_and_download_rtmw_model()
    
    if model_path is None:
        print("❌ RTMW 모델을 찾을 수 없습니다.")
        return
    
    print(f"🔍 분석 대상: {model_description}")
    print(f"📁 모델 경로: {model_path}")
    
    try:
        # 체크포인트 로드
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        
        # state_dict 추출
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        print(f"\n" + "="*80)
        print("📊 RTMW-x 체크포인트 상세 분석")
        print("="*80)
        
        # 1. 전체 키 목록
        print(f"\n📋 전체 레이어 수: {len(state_dict)}")
        
        # 2. 헤드 관련 레이어만 필터링
        head_layers = {}
        for name, param in state_dict.items():
            if 'head' in name:
                head_layers[name] = param.shape
        
        print(f"\n🎯 헤드 레이어 분석:")
        print(f"헤드 레이어 수: {len(head_layers)}")
        
        for name, shape in head_layers.items():
            print(f"   - {name}: {shape}")
        
        # 3. SimCC 관련 레이어 찾기
        simcc_layers = {}
        for name, shape in head_layers.items():
            if any(keyword in name.lower() for keyword in ['cls_x', 'cls_y', 'simcc', 'x_', 'y_']):
                simcc_layers[name] = shape
        
        print(f"\n🔍 SimCC 관련 레이어:")
        for name, shape in simcc_layers.items():
            print(f"   - {name}: {shape}")
            
            if 'weight' in name and len(shape) >= 2:
                output_dim = shape[0]
                input_dim = shape[1]
                print(f"     → 출력: {output_dim}, 입력: {input_dim}")
                
                # 133으로 나누어 떨어지는지 확인
                if output_dim % 133 == 0:
                    per_keypoint = output_dim // 133
                    print(f"     → 키포인트당: {per_keypoint} (133개 키포인트 기준)")
                else:
                    print(f"     → 133으로 나누어떨어지지 않음")
        
        # 4. 가능한 출력 패턴 분석
        print(f"\n📈 출력 패턴 분석:")
        
        # SimCC 표준 차원들
        standard_dims = {
            # 384x288 기준
            "384*1.5": int(384 * 1.5),    # 576
            "384*2": int(384 * 2),        # 768
            "288*2": int(288 * 2),        # 576  
            "288*2.67": int(288 * 2.67),  # 768
            # 다른 가능한 패턴들
            "133*4": 133 * 4,             # 532
            "133*5": 133 * 5,             # 665
            "133*6": 133 * 6,             # 798
        }
        
        print("가능한 SimCC 차원들:")
        for desc, dim in standard_dims.items():
            print(f"   - {desc}: {dim}")
        
        # 5. 실제 발견된 차원과 비교
        found_dims = set()
        for name, shape in simcc_layers.items():
            if 'weight' in name and len(shape) >= 2:
                found_dims.add(shape[0])
        
        print(f"\n✅ 실제 발견된 출력 차원: {sorted(found_dims)}")
        
        # 6. 각 차원이 어떤 패턴인지 분석
        for dim in sorted(found_dims):
            print(f"\n🔍 차원 {dim} 분석:")
            
            # 133으로 나눈 결과
            if dim % 133 == 0:
                per_kpt = dim // 133
                print(f"   - 133개 키포인트 × {per_kpt} = {dim}")
            
            # 입력 크기와의 관계
            if dim == 384 * 1.5:
                print(f"   - 384 × 1.5 = {dim} (X 좌표, 1.5배)")
            elif dim == 384 * 2:
                print(f"   - 384 × 2 = {dim} (X 좌표, 2배)")
            elif dim == 288 * 2:
                print(f"   - 288 × 2 = {dim} (Y 좌표, 2배)")
            elif dim == int(288 * 2.67):
                print(f"   - 288 × 2.67 = {dim} (Y 좌표, 2.67배)")
            
            # 가장 가까운 표준 차원
            closest_std = min(standard_dims.values(), key=lambda x: abs(x - dim))
            closest_name = [k for k, v in standard_dims.items() if v == closest_std][0]
            print(f"   - 가장 가까운 표준: {closest_name} ({closest_std})")
        
        # 7. 추정되는 올바른 차원
        print(f"\n🎯 추정되는 SimCC 구조:")
        
        x_dims = [dim for dim in found_dims if dim in [576, 768, 532]]
        y_dims = [dim for dim in found_dims if dim in [576, 768, 665]]
        
        if x_dims and y_dims:
            print(f"   - SimCC X 차원: {x_dims[0]}")
            print(f"   - SimCC Y 차원: {y_dims[0]}")
            
            if x_dims[0] % 133 == 0 and y_dims[0] % 133 == 0:
                x_per_kpt = x_dims[0] // 133
                y_per_kpt = y_dims[0] // 133
                print(f"   - X 키포인트당: {x_per_kpt} bins")
                print(f"   - Y 키포인트당: {y_per_kpt} bins")
        
        # 8. 수정 권장사항
        print(f"\n💡 수정 권장사항:")
        print(f"1. pose_estimator.py의 head_dims 기본값을 실제 값으로 수정")
        print(f"2. SimCC 디코더의 예상 차원 업데이트")
        
        if found_dims:
            dims_list = sorted(found_dims)
            if len(dims_list) >= 2:
                print(f"3. 추천 설정:")
                print(f"   'simcc_x_dim': {dims_list[0] // 133} (총 {dims_list[0]})")
                print(f"   'simcc_y_dim': {dims_list[1] // 133} (총 {dims_list[1]})")
        
        return sorted(found_dims)
        
    except Exception as e:
        print(f"❌ 분석 실패: {e}")
        import traceback
        traceback.print_exc()
        return None


def create_corrected_pose_estimator():
    """수정된 pose_estimator 생성"""
    
    # 실제 차원 분석
    actual_dims = analyze_rtmw_checkpoint()
    
    if not actual_dims or len(actual_dims) < 2:
        print("❌ 차원 분석 실패")
        return
    
    # 실제 차원으로 수정된 코드 생성
    x_total = min(actual_dims)
    y_total = max(actual_dims)
    
    x_per_kpt = x_total // 133
    y_per_kpt = y_total // 133
    
    print(f"\n" + "="*80)
    print("🔧 수정된 pose_estimator.py 코드")
    print("="*80)
    
    corrected_code = f'''
    def _analyze_head_dimensions(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, int]:
        """state_dict에서 헤드 차원 분석 (수정된 버전)"""
        head_dims = {{
            'simcc_x_dim': {x_per_kpt},  # 실제 값: {x_total} / 133
            'simcc_y_dim': {y_per_kpt},  # 실제 값: {y_total} / 133
            'num_keypoints': 133,
            'feature_dim': 256
        }}
        
        print(f"\\n🔍 헤드 차원 분석 (실제 값 사용):")
        
        for name, param in state_dict.items():
            if 'head' in name and 'weight' in name:
                print(f"   - {{name}}: {{param.shape}}")
                
                # 실제 차원 확인
                if 'cls_x' in name or 'simcc_x' in name:
                    if len(param.shape) >= 2:
                        actual_x_total = param.shape[0]
                        head_dims['simcc_x_dim'] = actual_x_total // 133
                        head_dims['feature_dim'] = param.shape[1]
                        print(f"     → 실제 SimCC X: {{actual_x_total}} = 133 × {{actual_x_total // 133}}")
                
                elif 'cls_y' in name or 'simcc_y' in name:
                    if len(param.shape) >= 2:
                        actual_y_total = param.shape[0]
                        head_dims['simcc_y_dim'] = actual_y_total // 133
                        head_dims['feature_dim'] = param.shape[1]
                        print(f"     → 실제 SimCC Y: {{actual_y_total}} = 133 × {{actual_y_total // 133}}")
        
        print(f"\\n📊 최종 헤드 차원:")
        for key, value in head_dims.items():
            print(f"   - {{key}}: {{value}}")
        
        return head_dims
    '''
    
    print(corrected_code)
    
    print(f"\n✅ 수정사항:")
    print(f"1. simcc_x_dim 기본값: {x_per_kpt} (총 {x_total})")
    print(f"2. simcc_y_dim 기본값: {y_per_kpt} (총 {y_total})")
    print(f"3. 이 값들을 pose_estimator.py에 적용하세요")


if __name__ == '__main__':
    print("=== RTMW-x 모델 차원 디버깅 ===")
    
    # 실제 차원 분석
    create_corrected_pose_estimator()