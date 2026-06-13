# VoxBox inference server image (the GPU-box side).
# The core is stdlib-only; extras are installed via EXTRAS build-arg.
#   docker build -t voxbox --build-arg EXTRAS="server" .
#   docker build -t voxbox --build-arg EXTRAS="server,whisper,piper" .
FROM python:3.11-slim

ARG EXTRAS=server
WORKDIR /app

# ffmpeg helps faster-whisper decode arbitrary audio; harmless for the mock path.
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY voxbox ./voxbox
COPY scripts ./scripts

RUN pip install --no-cache-dir ".[${EXTRAS}]"

ENV VOX_HOST=0.0.0.0 VOX_PORT=8765
EXPOSE 8765

# Backends are selected at runtime via VOX_* env vars (see README).
CMD ["python", "scripts/serve.py"]
