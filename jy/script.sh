cd .. && python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install uv pip
uv pip install --pre torchvision torchaudio torch --index-url https://download.pytorch.org/whl/nightly/xpu
uv pip install opencv-python h5py tqdm
uv pip install mmcv==2.1.0 mmdet mmengine
uv pip install ultralytics
python -m pip install -e .
cd jy
python stream_processor.py