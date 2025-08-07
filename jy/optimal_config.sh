# A6000 x2 환경 최적화 설정
export CUDA_VISIBLE_DEVICES=0,1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export OMP_NUM_THREADS=32

# 추천 파라미터:
# - GPU 워커: 2
# - GPU 배치 크기: 256  
# - CPU 워커: 32
# - 비디오 로더: 8

# 실행 예시:
python ultra_fast_processor.py \
    --gpu_workers 2 \
    --gpu_batch 256 \
    --cpu_workers 32 \
    --video_loaders 8