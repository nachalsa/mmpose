cd .. && python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install uv pip
uv pip install torchvision torchaudio torch
uv pip install opencv-python-headless h5py tqdm
uv pip install mmdet mmengine
pip install mmcv==2.1.0
uv pip install ultralytics
python -m pip install -e .
cd jy
python stream_processor.py