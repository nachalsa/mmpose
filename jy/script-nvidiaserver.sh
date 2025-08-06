cd .. && python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install uv pip
uv pip install torchvision torchaudio torch
uv pip install opencv-python-headless h5py tqdm
uv pip install mmdet mmengine
pip install mmcv==2.1.0
uv pip install ultralytics
# ONNX 추론기를 위한 추가 의존성
uv pip install onnxruntime-gpu  # GPU 가속을 위한 ONNX Runtime
uv pip install onnxruntime      # CPU 폴백용 ONNX Runtime  
uv pip install onnx             # ONNX 모델 처리
uv pip install Pillow          # 이미지 처리
uv pip install requests         # 모델 다운로드용
python -m pip install -e .
cd jy
# ONNX 추론기 실행 (또는 원하는 스크립트로 변경)
python onnx_inferencer.py



# cd .. && python3 -m venv .venv
# . .venv/bin/activate
# python3 -m pip install --upgrade pip setuptools wheel 
# pip3 install uv 
# uv pip install torchvision torchaudio torch --system
# uv pip install opencv-python-headless h5py tqdm --system
# uv pip install mmdet mmengine --system
# uv pip install mmcv==2.1.0 --system
# uv pip install ultralytics --system

# uv pip3 install chumpy --no-build-isolation --system
# uv pip3 install xtcocotools --no-build-isolation --system


# python3 -m pip install -e .
# cd jy
# python stream_processor.py