cd .. && python3 -m venv .venv
. .venv/bin/activate
pip install uv pip
uv pip install --pre torchvision torchaudio torch --index-url https://download.pytorch.org/whl/nightly/xpu
uv pip install opencv-python h5py tqdm
python -m pip install -e .
cd jy
python stream_processor.py