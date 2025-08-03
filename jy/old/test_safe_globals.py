#!/usr/bin/env python3
"""
PyTorch safe_globals를 사용한 체크포인트 로딩 해결
Checkpoint loading solution using PyTorch safe_globals
"""

import torch
import numpy as np
import os

# PyTorch safe_globals 설정
print("🔧 PyTorch safe_globals 설정 중...")

# numpy 관련 안전한 글로벌 추가
safe_globals_list = [
    # numpy core
    'numpy._core.multiarray._reconstruct',
    'numpy.core.multiarray._reconstruct', 
    'numpy.ndarray',
    'numpy.dtype',
    'numpy.core.numeric.ndarray',
    # collections
    'collections.OrderedDict',
    'collections.defaultdict',
    # 기본 타입들
    'builtins.dict',
    'builtins.list',
    'builtins.tuple',
    'builtins.set',
    'builtins.frozenset',
    'builtins.int',
    'builtins.float',
    'builtins.str',
    'builtins.bool',
    'builtins.bytes',
    'builtins.bytearray',
    'builtins.complex',
    # torch 관련
    'torch._utils._rebuild_tensor_v2',
    'torch._utils._rebuild_parameter',
    'torch._utils._rebuild_tensor',
    'torch.nn.parameter.Parameter',
    'torch.Tensor',
    'torch.Size',
    'torch.device',
    'torch.dtype',
]

# 안전한 글로벌들 추가
for global_name in safe_globals_list:
    try:
        torch.serialization.add_safe_globals([global_name])
        print(f"✅ {global_name} 추가됨")
    except Exception as e:
        print(f"⚠️ {global_name} 추가 실패: {e}")

print("🔧 안전한 글로벌 설정 완료")

# 체크포인트 테스트
checkpoint_path = "/home/ty/rtmw/02/mmpose/models/rtmw-x_simcc-cocktail14_pt-ucoco_270e-384x288-f840f204_20231122.pth"

print(f"\n📂 체크포인트 로딩 테스트: {checkpoint_path}")

# Method 1: safe_globals 컨텍스트 매니저 사용
print(f"\n🔄 방법 1: safe_globals 컨텍스트 매니저")
try:
    with torch.serialization.safe_globals(['numpy._core.multiarray._reconstruct']):
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    print(f"✅ safe_globals 컨텍스트 매니저 성공!")
    print(f"   - 키: {list(checkpoint.keys())}")
except Exception as e:
    print(f"❌ safe_globals 실패: {e}")

# Method 2: weights_only=False (신뢰할 수 있는 소스)
print(f"\n🔄 방법 2: weights_only=False")
try:
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    print(f"✅ weights_only=False 성공!")
    print(f"   - 키: {list(checkpoint.keys())}")
    
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        print(f"   - State dict 크기: {len(state_dict)}")
        
    # 새로운 형식으로 저장 (안전한 방식)
    safe_checkpoint_path = checkpoint_path.replace('.pth', '_safe.pth')
    
    # 새로운 직렬화 방식 사용
    torch.save(checkpoint, safe_checkpoint_path, _use_new_zipfile_serialization=False)
    print(f"💾 안전한 체크포인트 저장: {safe_checkpoint_path}")
    
    # 안전한 체크포인트 로드 테스트
    print(f"\n🔄 안전한 체크포인트 로드 테스트")
    try:
        safe_checkpoint = torch.load(safe_checkpoint_path, map_location='cpu', weights_only=False)
        print(f"✅ 안전한 체크포인트 로드 성공!")
        print(f"   - 키: {list(safe_checkpoint.keys())}")
    except Exception as e:
        print(f"❌ 안전한 체크포인트 로드 실패: {e}")
        
except Exception as e:
    print(f"❌ weights_only=False 실패: {e}")

print("\n=== 체크포인트 로딩 테스트 완료 ===")
