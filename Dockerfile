FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir 'torch==2.6.0+cpu' --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt \
    && python -c "import torch; assert torch.version.cuda is None, 'expected CPU PyTorch'"
COPY video_search ./video_search
COPY web ./web
ENV VIDEO_SEARCH_DATA=/data VIDEO_SEARCH_DEVICE=cpu MPLCONFIGDIR=/tmp/matplotlib
EXPOSE 8000
CMD ["uvicorn", "video_search.api:app", "--host", "0.0.0.0", "--port", "8000"]
