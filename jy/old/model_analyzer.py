#!/usr/bin/env python3
"""
RTMW-x 모델 구조 분석기
RTMW-x 384x288 모델의 입력/출력 구조와 모델 아키텍처 분석
"""

import torch
import numpy as np
import onnx
import onnxruntime as ort
from typing import Dict, List, Tuple, Any
import sys
import os

# 현재 디렉토리를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import find_and_download_rtmw_model
from config import RTMW_INPUT_SIZE


class RTMWModelAnalyzer:
    """RTMW-x 모델 구조 분석기"""
    
    def __init__(self, model_path: str):
        """
        Args:
            model_path: ONNX 모델 파일 경로
        """
        self.model_path = model_path
        self.onnx_model = None
        self.ort_session = None
        self.input_shape = RTMW_INPUT_SIZE
        
    def load_model(self):
        """ONNX 모델 로드"""
        try:
            # ONNX 모델 로드
            self.onnx_model = onnx.load(self.model_path)
            print(f"✅ ONNX 모델 로드 성공: {self.model_path}")
            
            # ONNX Runtime 세션 생성
            providers = ['CPUExecutionProvider']
            try:
                # Intel XPU 지원 확인
                if 'OpenVINOExecutionProvider' in ort.get_available_providers():
                    providers.insert(0, 'OpenVINOExecutionProvider')
                    print("🚀 OpenVINO Provider 사용")
            except:
                pass
            
            self.ort_session = ort.InferenceSession(self.model_path, providers=providers)
            print(f"✅ ONNX Runtime 세션 생성 완료")
            
        except Exception as e:
            print(f"❌ 모델 로드 실패: {e}")
            return False
        
        return True
    
    def analyze_model_structure(self):
        """모델 전체 구조 분석"""
        if self.onnx_model is None:
            print("❌ 모델이 로드되지 않았습니다.")
            return
        
        print("\n" + "="*80)
        print("📊 RTMW-x 모델 구조 분석")
        print("="*80)
        
        # 기본 정보
        print(f"모델 파일: {os.path.basename(self.model_path)}")
        print(f"ONNX 버전: {self.onnx_model.opset_import[0].version}")
        print(f"Producer: {self.onnx_model.producer_name}")
        print(f"Model Version: {self.onnx_model.model_version}")
        
        # 그래프 정보
        graph = self.onnx_model.graph
        print(f"\n📈 그래프 정보:")
        print(f"- 노드 수: {len(graph.node)}")
        print(f"- 입력 수: {len(graph.input)}")
        print(f"- 출력 수: {len(graph.output)}")
        print(f"- 초기화 텐서 수: {len(graph.initializer)}")
        
    def analyze_input_output(self):
        """입력/출력 구조 분석"""
        if self.onnx_model is None:
            print("❌ 모델이 로드되지 않았습니다.")
            return
        
        graph = self.onnx_model.graph
        
        print("\n" + "="*50)
        print("📥 입력 구조 분석")
        print("="*50)
        
        for i, input_tensor in enumerate(graph.input):
            print(f"\n입력 {i+1}: {input_tensor.name}")
            shape = [dim.dim_value if dim.dim_value > 0 else dim.dim_param 
                    for dim in input_tensor.type.tensor_type.shape.dim]
            data_type = input_tensor.type.tensor_type.elem_type
            print(f"  - Shape: {shape}")
            print(f"  - Data Type: {self._get_data_type_name(data_type)}")
            
            # 배치, 채널, 높이, 너비 해석
            if len(shape) == 4:
                print(f"  - 배치 크기: {shape[0]}")
                print(f"  - 채널 수: {shape[1]}")
                print(f"  - 높이: {shape[2]}")
                print(f"  - 너비: {shape[3]}")
                print(f"  - 예상 입력: RGB 이미지 (H×W×C → C×H×W)")
        
        print("\n" + "="*50)
        print("📤 출력 구조 분석")
        print("="*50)
        
        for i, output_tensor in enumerate(graph.output):
            print(f"\n출력 {i+1}: {output_tensor.name}")
            shape = [dim.dim_value if dim.dim_value > 0 else dim.dim_param 
                    for dim in output_tensor.type.tensor_type.shape.dim]
            data_type = output_tensor.type.tensor_type.elem_type
            print(f"  - Shape: {shape}")
            print(f"  - Data Type: {self._get_data_type_name(data_type)}")
            
            # RTMW 출력 해석
            if len(shape) >= 2:
                if len(shape) == 3 and shape[2] == 2:
                    print(f"  - 키포인트 좌표: {shape[1]}개 키포인트 × 2 (x, y)")
                elif len(shape) == 2 and shape[1] > 100:
                    print(f"  - 키포인트 수: {shape[1] // 2}개 (x, y 쌍)")
                    print(f"  - WholeBody 구성:")
                    print(f"    * Body: 17개 키포인트")
                    print(f"    * Face: 68개 키포인트") 
                    print(f"    * Left Hand: 21개 키포인트")
                    print(f"    * Right Hand: 21개 키포인트")
                    print(f"    * 총 133개 키포인트 예상")
    
    def test_inference(self):
        """테스트 추론으로 실제 입출력 확인"""
        if self.ort_session is None:
            print("❌ ONNX Runtime 세션이 없습니다.")
            return
        
        print("\n" + "="*50)
        print("🧪 테스트 추론")
        print("="*50)
        
        # 입력 준비
        input_name = self.ort_session.get_inputs()[0].name
        input_shape = self.ort_session.get_inputs()[0].shape
        
        # 동적 차원 처리
        actual_shape = []
        for dim in input_shape:
            if isinstance(dim, str) or dim == -1:
                actual_shape.append(1)  # 배치 크기를 1로 설정
            else:
                actual_shape.append(dim)
        
        print(f"입력 이름: {input_name}")
        print(f"입력 shape: {actual_shape}")
        
        # 더미 입력 데이터 생성 (정규화된 RGB 이미지)
        dummy_input = np.random.randn(*actual_shape).astype(np.float32)
        # 일반적인 이미지 정규화 범위로 조정
        dummy_input = (dummy_input * 0.5) + 0.5  # [0, 1] 범위
        
        try:
            # 추론 실행
            outputs = self.ort_session.run(None, {input_name: dummy_input})
            
            print(f"\n✅ 추론 성공!")
            print(f"출력 개수: {len(outputs)}")
            
            for i, output in enumerate(outputs):
                print(f"\n출력 {i+1}:")
                print(f"  - Shape: {output.shape}")
                print(f"  - Data Type: {output.dtype}")
                print(f"  - Min: {output.min():.6f}")
                print(f"  - Max: {output.max():.6f}")
                print(f"  - Mean: {output.mean():.6f}")
                
                # 키포인트 출력 해석
                if len(output.shape) == 3 and output.shape[2] == 2:
                    num_keypoints = output.shape[1]
                    print(f"  - 키포인트 수: {num_keypoints}")
                    print(f"  - 좌표 범위 (x): [{output[:,:,0].min():.2f}, {output[:,:,0].max():.2f}]")
                    print(f"  - 좌표 범위 (y): [{output[:,:,1].min():.2f}, {output[:,:,1].max():.2f}]")
                elif len(output.shape) == 2:
                    if output.shape[1] == 266:  # 133 keypoints * 2
                        print(f"  - WholeBody 키포인트: 133개 × 2 (x, y)")
                        keypoints = output.reshape(-1, 133, 2)
                        print(f"  - 좌표 범위 (x): [{keypoints[:,:,0].min():.2f}, {keypoints[:,:,0].max():.2f}]")
                        print(f"  - 좌표 범위 (y): [{keypoints[:,:,1].min():.2f}, {keypoints[:,:,1].max():.2f}]")
                    
        except Exception as e:
            print(f"❌ 추론 실패: {e}")
    
    def analyze_layer_structure(self):
        """레이어 구조 분석"""
        if self.onnx_model is None:
            print("❌ 모델이 로드되지 않았습니다.")
            return
        
        print("\n" + "="*50)
        print("🔍 주요 레이어 구조 분석")
        print("="*50)
        
        graph = self.onnx_model.graph
        
        # 연산자 타입별 카운트
        op_counts = {}
        for node in graph.node:
            op_type = node.op_type
            op_counts[op_type] = op_counts.get(op_type, 0) + 1
        
        print(f"\n📊 연산자 통계:")
        for op_type, count in sorted(op_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {op_type}: {count}개")
        
        # 주요 레이어 정보
        print(f"\n🏗️ 주요 구조 분석:")
        conv_count = op_counts.get('Conv', 0)
        relu_count = op_counts.get('Relu', 0)
        bn_count = op_counts.get('BatchNormalization', 0)
        
        print(f"  - Convolution 레이어: {conv_count}개")
        print(f"  - ReLU 활성화: {relu_count}개") 
        print(f"  - Batch Normalization: {bn_count}개")
        
        if 'Gemm' in op_counts or 'MatMul' in op_counts:
            print(f"  - Dense/Linear 레이어: {op_counts.get('Gemm', 0) + op_counts.get('MatMul', 0)}개")
    
    def _get_data_type_name(self, data_type: int) -> str:
        """ONNX 데이터 타입을 문자열로 변환"""
        type_map = {
            1: 'float32',
            2: 'uint8', 
            3: 'int8',
            6: 'int32',
            7: 'int64',
            9: 'bool',
            10: 'float16',
            11: 'double'
        }
        return type_map.get(data_type, f'unknown({data_type})')


def main():
    """모델 분석 메인 함수"""
    print("=== RTMW-x 모델 구조 분석기 ===")
    
    # RTMW 모델 찾기
    model_path, model_description = find_and_download_rtmw_model()
    
    if model_path is None:
        print("❌ RTMW 모델을 찾을 수 없습니다.")
        return
    
    print(f"분석 대상: {model_description}")
    print(f"모델 경로: {model_path}")
    
    # 모델 분석기 초기화
    analyzer = RTMWModelAnalyzer(model_path)
    
    # 모델 로드
    if not analyzer.load_model():
        return
    
    # 전체 구조 분석
    analyzer.analyze_model_structure()
    
    # 입출력 구조 분석
    analyzer.analyze_input_output()
    
    # 레이어 구조 분석
    analyzer.analyze_layer_structure()
    
    # 테스트 추론
    analyzer.test_inference()
    
    print("\n" + "="*80)
    print("✅ 모델 분석 완료")
    print("="*80)


if __name__ == '__main__':
    main()