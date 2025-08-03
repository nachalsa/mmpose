import torch
import os

def check_model_type(model_path):
    """모델 파일 타입 확인"""
    try:
        # 파일 로드
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        
        print(f"파일 타입: {type(checkpoint)}")
        print(f"내용: {checkpoint.keys() if isinstance(checkpoint, dict) else 'Not a dict'}")
        
        if isinstance(checkpoint, dict):
            print("\n=== 딕셔너리 기반 체크포인트 (가중치만) ===")
            for key in checkpoint.keys():
                print(f"  - {key}: {type(checkpoint[key])}")
            
            if 'state_dict' in checkpoint:
                print(f"  state_dict 크기: {len(checkpoint['state_dict'])}")
                print(f"  첫 번째 가중치 키: {list(checkpoint['state_dict'].keys())[0]}")
        else:
            print("\n=== 모델 객체 (구조 + 가중치) ===")
            print(f"모델 타입: {type(checkpoint)}")
            if hasattr(checkpoint, '__dict__'):
                print(f"모델 속성들: {list(checkpoint.__dict__.keys())}")
        
        return checkpoint
        
    except Exception as e:
        print(f"오류: {e}")
        return None

def analyze_checkpoint_structure(model_path):
    """체크포인트 구조 상세 분석"""
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    print("=" * 50)
    print("체크포인트 구조 분석")
    print("=" * 50)
    
    if isinstance(checkpoint, dict):
        print("📁 딕셔너리 기반 체크포인트")
        print(f"   키들: {list(checkpoint.keys())}")
        
        # 메타 정보 확인
        if 'meta' in checkpoint:
            meta = checkpoint['meta']
            print(f"\n📋 메타 정보:")
            for key, value in meta.items():
                print(f"   {key}: {value}")
        
        # state_dict 확인
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            print(f"\n⚖️ 가중치 정보:")
            print(f"   총 레이어 수: {len(state_dict)}")
            print(f"   첫 5개 레이어:")
            for i, (key, tensor) in enumerate(list(state_dict.items())[:5]):
                print(f"     {key}: {tensor.shape}")
            
            # 구조 추론
            backbone_layers = [k for k in state_dict.keys() if k.startswith('backbone')]
            head_layers = [k for k in state_dict.keys() if k.startswith('head')]
            
            print(f"\n🏗️ 구조 추론:")
            print(f"   백본 레이어: {len(backbone_layers)}개")
            print(f"   헤드 레이어: {len(head_layers)}개")
            
            if backbone_layers:
                print(f"   백본 첫 레이어: {backbone_layers[0]}")
            if head_layers:
                print(f"   헤드 첫 레이어: {head_layers[0]}")
        
        print(f"\n❌ 결론: 가중치만 저장된 체크포인트 (구조 없음)")
        
    else:
        print("🏛️ 모델 객체")
        print(f"   타입: {type(checkpoint)}")
        
        # 모델 메서드 확인
        if hasattr(checkpoint, 'forward'):
            print("   ✅ forward 메서드 있음")
        if hasattr(checkpoint, 'state_dict'):
            print("   ✅ state_dict 메서드 있음")
        if hasattr(checkpoint, 'eval'):
            print("   ✅ eval 메서드 있음")
            
        print(f"\n✅ 결론: 구조 + 가중치 모두 저장된 모델")
    
    return checkpoint

def test_model_loading(model_path):
    """실제 로딩 테스트로 확인"""
    print("=" * 50)
    print("모델 로딩 테스트")
    print("=" * 50)
    
    try:
        # 방법 1: 직접 로드 (구조 포함된 경우)
        print("🧪 테스트 1: 직접 로드")
        model = torch.load(model_path, map_location='cpu')
        
        if hasattr(model, 'forward'):
            print("   ✅ 성공: 구조 + 가중치 모델")
            print(f"   모델 타입: {type(model)}")
            return True
        else:
            print("   ❌ 실패: 구조 없음")
            
    except Exception as e:
        print(f"   ❌ 실패: {e}")
    
    try:
        # 방법 2: 체크포인트로 로드 (가중치만 있는 경우)
        print("\n🧪 테스트 2: 체크포인트 로드")
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            print("   ✅ 성공: 가중치 체크포인트")
            print("   ⚠️ 구조를 별도로 정의해야 함")
            return False
        else:
            print("   ❌ 알 수 없는 형식")
            
    except Exception as e:
        print(f"   ❌ 실패: {e}")
    
    return None

def complete_model_analysis(model_path):
    """모델 파일 완전 분석"""
    
    print("🔍 RTMPose-x 모델 파일 분석")
    print("=" * 60)
    
    # 파일 기본 정보
    file_size = os.path.getsize(model_path) / (1024**3)  # GB
    print(f"📁 파일 정보:")
    print(f"   경로: {model_path}")
    print(f"   크기: {file_size:.2f} GB")
    
    # 로드 및 분석
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    print(f"\n📊 구조 분석:")
    print(f"   파일 타입: {type(checkpoint)}")
    
    if isinstance(checkpoint, dict):
        print(f"   딕셔너리 키: {list(checkpoint.keys())}")
        
        # 메타데이터
        if 'meta' in checkpoint:
            meta = checkpoint['meta']
            print(f"\n📋 메타데이터:")
            print(f"   MMPose 버전: {meta.get('mmpose_version', 'N/A')}")
            print(f"   설정: {meta.get('config', {}).get('model', {}).get('type', 'N/A')}")
        
        # 가중치 정보  
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            print(f"\n⚖️ 가중치 정보:")
            print(f"   총 파라미터: {len(state_dict):,}개")
            
            # 파라미터 개수 계산
            total_params = sum(p.numel() for p in state_dict.values() if isinstance(p, torch.Tensor))
            print(f"   총 파라미터 개수: {total_params:,}")
            print(f"   메모리 사용량: {total_params * 4 / (1024**2):.1f} MB")
    
    print(f"\n🏁 결론:")
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        print("   ❌ 가중치만 저장됨 (구조 정의 필요)")
        print("   📝 해결책: MMPose config + 체크포인트 또는 ONNX 변환")
    else:
        print("   ✅ 구조 포함 (바로 사용 가능)")

# 현재 모델 확인
model_path = "./models/rtmpose-x_simcc-coco-wholebody_pt-body7_270e-384x288-401dfc90_20230629.pth"
checkpoint = check_model_type(model_path)
# 분석 실행
analyze_checkpoint_structure(model_path)
# 테스트 실행
has_structure = test_model_loading(model_path)
# 완전 분석 실행
complete_model_analysis(model_path)