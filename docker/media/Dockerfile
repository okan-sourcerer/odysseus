# Odysseus media server: scripts/diffusion_server.py with CUDA, run as the
# `media` service by docker/media.nvidia.yml. Versions match a setup verified
# with FLUX.2 [klein] 4B (nf4) on an 8 GB NVIDIA GPU.
FROM python:3.12-slim

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/hf

# torch's CUDA 12.8 wheel bundles the CUDA runtime; the host only needs the
# NVIDIA driver (passed through by the overlay's GPU reservation).
RUN pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.11.0

RUN pip install \
    diffusers==0.41.0 transformers==5.18.0 accelerate==1.15.0 \
    bitsandbytes==0.50.2 safetensors==0.8.0 huggingface_hub==1.33.0 \
    tokenizers==0.23.2 numpy==2.5.3 pillow==12.3.0 \
    fastapi==0.142.2 starlette==1.7.0 uvicorn==0.54.0 pydantic==2.13.5 \
    python-multipart==0.0.32 httpx==0.28.1

RUN useradd --create-home --uid 1000 media
COPY --chown=media scripts/diffusion_server.py /app/diffusion_server.py
USER media
WORKDIR /app
EXPOSE 8100
ENTRYPOINT ["python", "/app/diffusion_server.py"]
