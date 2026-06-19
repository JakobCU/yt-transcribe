# Image for the FUTURE Linux GPU server (docker compose --profile full up -d --build).
# NOT used on Windows — there the app runs natively (conda + CUDA); see deploy/README.md.
#
# CUDA runtime base so Whisper + pyannote use the GPU. Torch is installed from the
# matching CUDA wheel index; the rest of the project comes from pyproject extras.
FROM nvidia/cuda:12.6.2-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip ffmpeg git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependency layer first (better build caching), then the source.
COPY pyproject.toml ./
COPY yt_transcribe ./yt_transcribe
COPY server ./server
COPY tool ./tool

RUN python3 -m pip install --upgrade pip \
    && python3 -m pip install torch --index-url https://download.pytorch.org/whl/cu126 \
    && python3 -m pip install -e ".[server,postgres]"

EXPOSE 8000

# EXACTLY one worker: the in-process job queue and the resident GPU models are
# per-process state — multiple workers would duplicate models and lose job tracking.
# --proxy-headers + forwarded-allow-ips so the app trusts Caddy's X-Forwarded-Proto
# (needed for Secure cookies behind TLS).
CMD ["uvicorn", "server.app:app", \
     "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "1", \
     "--proxy-headers", "--forwarded-allow-ips", "*"]
