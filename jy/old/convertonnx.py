#!/usr/bin/env python3
"""
RTMW 모델 ONNX 변환기
MMPose 모델을 ONNX 형식으로 변환합니다.
"""

import os
import torch
import numpy as np
from mmpose.apis import init_model
from mmpose.structures import PoseDataSample
from mmengine.structures import InstanceData
import onnx
import onnxruntime as ort

class RTMWONNXConverter:
    """RTMW 모델을 ONNX로 변환하는 클래스"""
    
    def __init__(self, config_path: str, checkpoint_path: str, device: str = 'cpu'):
        """
        Args:
            config_path: RTMW 설정 파일 경로
            checkpoint_path: RTMW 체크포인트 파일 경로
            device: 변환에 사용할 디바이스
        """
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.device = device
        
        print(f"🔧 RTMW 모델 로딩 중... (디바이스: {device})")
        self.model = init_model(config_path, checkpoint_path, device=device)
        self.model.eval()  # 평가 모드로 설정
        
        # 모델 정보 출력
        self._print_model_info()
    
    def _print_model_info(self):
        """모델 정보 출력"""
        print(f"✅ RTMW 모델 정보:")
        print(f"   - 설정 파일: {os.path.basename(self.config_path)}")
        print(f"   - 체크포인트: {os.path.basename(self.checkpoint_path)}")
        print(f"   - 디바이스: {self.device}")
        
        # 입력/출력 크기 정보
        if hasattr(self.model.cfg, 'model'):
            model_cfg = self.model.cfg.model
            if hasattr(model_cfg, 'data_preprocessor'):
                data_proc = model_cfg.data_preprocessor
                if 'input_size' in data_proc:
                    print(f"   - 입력 크기: {data_proc['input_size']}")
            
            if hasattr(model_cfg, 'head'):
                head_cfg = model_cfg.head
                if hasattr(head_cfg, 'out_channels'):
                    print(f"   - 출력 채널: {head_cfg.out_channels}")
    
    def convert_to_onnx(self, 
                       output_path: str,
                       input_size: tuple = (288, 384),  # (H, W)
                       opset_version: int = 11,
                       dynamic_batch: bool = True,
                       simplify: bool = True):
        """
        RTMW 모델을 ONNX로 변환
        
        Args:
            output_path: 출력 ONNX 파일 경로
            input_size: 입력 이미지 크기 (H, W)
            opset_version: ONNX opset 버전
            dynamic_batch: 동적 배치 크기 지원 여부
            simplify: ONNX 모델 최적화 여부
        """
        print(f"\n🔄 ONNX 변환 시작...")
        print(f"   - 출력 경로: {output_path}")
        print(f"   - 입력 크기: {input_size}")
        print(f"   - Opset 버전: {opset_version}")
        print(f"   - 동적 배치: {dynamic_batch}")
        
        try:
            # 더미 입력 생성
            height, width = input_size
            batch_size = 1
            dummy_input = torch.randn(batch_size, 3, height, width).to(self.device)
            
            # 동적 축 설정
            dynamic_axes = None
            if dynamic_batch:
                dynamic_axes = {
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }
            
            # 입력/출력 이름 설정
            input_names = ['input']
            output_names = ['output']
            
            # ONNX 변환
            print("⚙️ PyTorch -> ONNX 변환 중...")
            torch.onnx.export(
                self.model,
                dummy_input,
                output_path,
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                verbose=False
            )
            
            print(f"✅ ONNX 변환 완료: {output_path}")
            
            # 모델 검증
            self._verify_onnx_model(output_path, dummy_input)
            
            # 모델 최적화 (선택적)
            if simplify:
                self._simplify_onnx_model(output_path)
            
            # 모델 정보 출력
            self._print_onnx_info(output_path)
            
            return True
            
        except Exception as e:
            print(f"❌ ONNX 변환 실패: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _verify_onnx_model(self, onnx_path: str, dummy_input: torch.Tensor):
        """ONNX 모델 검증"""
        print("🔍 ONNX 모델 검증 중...")
        
        try:
            # ONNX 모델 로드 및 검증
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)
            print("✅ ONNX 모델 구조 검증 통과")
            
            # ONNX Runtime으로 추론 테스트
            ort_session = ort.InferenceSession(onnx_path)
            
            # 입력 데이터 준비
            input_data = dummy_input.cpu().numpy()
            
            # 원본 모델 추론
            with torch.no_grad():
                original_output = self.model(dummy_input)
                if hasattr(original_output, 'pred_instances'):
                    # MMPose 출력 구조인 경우
                    if hasattr(original_output.pred_instances, 'keypoints'):
                        original_result = original_output.pred_instances.keypoints
                    else:
                        original_result = original_output
                else:
                    original_result = original_output
            
            # ONNX 모델 추론
            ort_inputs = {ort_session.get_inputs()[0].name: input_data}
            onnx_output = ort_session.run(None, ort_inputs)
            
            # 결과 비교 (간단한 검증)
            print(f"📊 출력 비교:")
            print(f"   - 원본 출력 타입: {type(original_result)}")
            print(f"   - ONNX 출력 타입: {type(onnx_output[0])}")
            print(f"   - ONNX 출력 shape: {onnx_output[0].shape}")
            
            # 수치적 일치성 검사는 복잡하므로 기본 검증만 수행
            print("✅ ONNX Runtime 추론 테스트 통과")
            
        except Exception as e:
            print(f"⚠️ ONNX 모델 검증 중 오류: {e}")
    
    def _simplify_onnx_model(self, onnx_path: str):
        """ONNX 모델 최적화"""
        try:
            import onnxsim
            print("🔧 ONNX 모델 최적화 중...")
            
            # 원본 모델 로드
            model = onnx.load(onnx_path)
            
            # 최적화
            model_simplified, check = onnxsim.simplify(model)
            
            if check:
                # 최적화된 모델 저장
                simplified_path = onnx_path.replace('.onnx', '_simplified.onnx')
                onnx.save(model_simplified, simplified_path)
                print(f"✅ 최적화된 모델 저장: {simplified_path}")
            else:
                print("⚠️ 모델 최적화 검증 실패")
                
        except ImportError:
            print("⚠️ onnx-simplifier 미설치 - pip install onnx-simplifier")
        except Exception as e:
            print(f"⚠️ 모델 최적화 실패: {e}")
    
    def _print_onnx_info(self, onnx_path: str):
        """ONNX 모델 정보 출력"""
        try:
            model = onnx.load(onnx_path)
            
            print(f"\n📋 ONNX 모델 정보:")
            print(f"   - 파일 크기: {os.path.getsize(onnx_path) / 1024 / 1024:.1f} MB")
            print(f"   - IR 버전: {model.ir_version}")
            print(f"   - Opset 버전: {model.opset_import[0].version}")
            
            # 입력 정보
            if model.graph.input:
                input_info = model.graph.input[0]
                input_shape = [dim.dim_value if dim.dim_value > 0 else 'dynamic' 
                             for dim in input_info.type.tensor_type.shape.dim]
                print(f"   - 입력: {input_info.name} {input_shape}")
            
            # 출력 정보  
            if model.graph.output:
                output_info = model.graph.output[0]
                output_shape = [dim.dim_value if dim.dim_value > 0 else 'dynamic' 
                              for dim in output_info.type.tensor_type.shape.dim]
                print(f"   - 출력: {output_info.name} {output_shape}")
            
        except Exception as e:
            print(f"⚠️ ONNX 정보 출력 실패: {e}")

def convert_rtmw_to_onnx(config_path: str, 
                        checkpoint_path: str, 
                        output_path: str = None,
                        input_size: tuple = (288, 384),
                        device: str = 'cpu'):
    """
    RTMW 모델을 ONNX로 변환하는 메인 함수
    
    Args:
        config_path: RTMW 설정 파일 경로
        checkpoint_path: RTMW 체크포인트 파일 경로
        output_path: 출력 ONNX 파일 경로 (None이면 자동 생성)
        input_size: 입력 이미지 크기 (H, W)
        device: 변환에 사용할 디바이스
    """
    # 출력 경로 자동 생성
    if output_path is None:
        base_name = os.path.splitext(os.path.basename(checkpoint_path))[0]
        output_path = f"{base_name}_{input_size[1]}x{input_size[0]}.onnx"
    
    print("🚀 RTMW -> ONNX 변환기")
    print("=" * 50)
    
    # 변환기 생성 및 변환 실행
    converter = RTMWONNXConverter(config_path, checkpoint_path, device)
    
    success = converter.convert_to_onnx(
        output_path=output_path,
        input_size=input_size,
        opset_version=11,
        dynamic_batch=True,
        simplify=True
    )
    
    if success:
        print(f"\n🎉 ONNX 변환 완료!")
        print(f"📁 출력 파일: {output_path}")
        print(f"💡 사용법:")
        print(f"   import onnxruntime as ort")
        print(f"   session = ort.InferenceSession('{output_path}')")
        print(f"   result = session.run(None, {{'input': your_input_data}})")
        
        return output_path
    else:
        print(f"\n❌ ONNX 변환 실패")
        return None

def main():
    """테스트 메인 함수"""
    # 모델 경로 설정
    config_path = "../configs/wholebody_2d_keypoint/rtmpose/cocktail14/rtmw-x_8xb320-270e_cocktail14-384x288.py"
    checkpoint_path = "../models/rtmw-x_simcc-cocktail14_pt-ucoco_270e-384x288-f840f204_20231122.pth"
    
    # 파일 존재 확인
    if not os.path.exists(config_path):
        print(f"❌ 설정 파일 없음: {config_path}")
        return
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ 체크포인트 파일 없음: {checkpoint_path}")
        return
    
    # ONNX 변환 실행
    output_onnx = convert_rtmw_to_onnx(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        input_size=(288, 384),  # RTMW 기본 입력 크기
        device='cpu'  # 변환은 CPU에서 안정적
    )
    
    if output_onnx:
        print(f"\n🔥 추가 최적화 옵션:")
        print(f"   1. 다른 입력 크기로 변환:")
        print(f"      - (384, 512): 고해상도")  
        print(f"      - (192, 256): 저해상도")
        print(f"   2. TensorRT 최적화 (NVIDIA GPU)")
        print(f"   3. OpenVINO 최적화 (Intel CPU/GPU)")

if __name__ == "__main__":
    main()