#!/usr/bin/env python3
"""
PyTorch 버전별 체크포인트 로딩 테스트
PyTorch version specific checkpoint loading test
"""

import torch
import os
import warnings

print("=== PyTorch 체크포인트 로딩 테스트 ===")
print(f"PyTorch 버전: {torch.__version__}")
print(f"CUDA 사용 가능: {torch.cuda.is_available()}")

# XPU 확인
try:
    print(f"XPU 사용 가능: {torch.xpu.is_available()}")
except:
    print("XPU 사용 불가")

checkpoint_path = "/home/ty/rtmw/02/mmpose/models/rtmw-x_simcc-cocktail14_pt-ucoco_270e-384x288-f840f204_20231122.pth"

if not os.path.exists(checkpoint_path):
    print(f"❌ 체크포인트 파일을 찾을 수 없습니다: {checkpoint_path}")
    exit(1)

print(f"\n📂 체크포인트 파일: {checkpoint_path}")
print(f"📁 파일 크기: {os.path.getsize(checkpoint_path) / (1024*1024):.1f} MB")

# Method 1: weights_only=False (강제)
print(f"\n🔄 방법 1: weights_only=False")
try:
    # 모든 경고 무시
    warnings.filterwarnings('ignore')
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    print(f"✅ 성공!")
    print(f"   - 키: {list(checkpoint.keys())}")
    
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        print(f"   - State dict 크기: {len(state_dict)}")
        
        # 첫 번째 몇 개 키 출력
        first_keys = list(state_dict.keys())[:5]
        print(f"   - 첫 5개 키: {first_keys}")
        
    if 'meta' in checkpoint:
        meta = checkpoint['meta']
        print(f"   - Meta 정보: {list(meta.keys())}")
        
    if 'cfg' in checkpoint:
        print(f"   - Config 존재: ✅")
        
except Exception as e:
    print(f"❌ 실패: {e}")

# Method 2: torch.jit.load (TorchScript용)
print(f"\n🔄 방법 2: torch.jit.load")
try:
    model = torch.jit.load(checkpoint_path, map_location='cpu')
    print(f"✅ TorchScript 모델로 로드 성공!")
except Exception as e:
    print(f"❌ TorchScript 실패: {e}")

# Method 3: pickle 직접 사용
print(f"\n🔄 방법 3: pickle 직접")
try:
    import pickle
    with open(checkpoint_path, 'rb') as f:
        checkpoint = pickle.load(f)
    print(f"✅ Pickle 직접 로드 성공!")
    print(f"   - 타입: {type(checkpoint)}")
    if isinstance(checkpoint, dict):
        print(f"   - 키: {list(checkpoint.keys())}")
except Exception as e:
    print(f"❌ Pickle 실패: {e}")

# Method 4: 환경 변수 설정 후 재시도
print(f"\n🔄 방법 4: 환경 변수 설정")
os.environ['TORCH_LOAD_WEIGHTS_ONLY'] = '0'
os.environ['WEIGHTS_ONLY'] = '0'
torch._C._set_load_weights_only_false()  # PyTorch 내부 함수

try:
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    print(f"✅ 환경 변수 설정 후 성공!")
    print(f"   - 키: {list(checkpoint.keys())}")
except Exception as e:
    print(f"❌ 환경 변수 설정 후에도 실패: {e}")

print("\n=== 테스트 완료 ===")
