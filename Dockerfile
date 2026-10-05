FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --timeout 120 --retries 5 'torch==2.6.0+cu124' --index-url https://download.pytorch.org/whl/cu124 \
    && pip install --no-cache-dir -r requirements.txt \
    && python -c "import torch; assert torch.version.cuda == '12.4', 'expected CUDA 12.4 PyTorch'"
COPY video_search ./video_search
COPY web ./web
ENV VIDEO_SEARCH_DATA=/data VIDEO_SEARCH_DEVICE=cuda MPLCONFIGDIR=/tmp/matplotlib
EXPOSE 8000
CMD ["uvicorn", "video_search.api:app", "--host", "0.0.0.0", "--port", "8000"]
