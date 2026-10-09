# Odysseus TTS server: scripts/tts_server.py with Kokoro (CPU) and Chatterbox
# Multilingual (GPU when free), run as the `tts` service by
# docker/media.nvidia.yml. Separate from the image server's image because
# chatterbox-tts pins torch 2.6 and older transformers/diffusers.
FROM python:3.12-slim

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/hf

# espeak-ng: Kokoro's pronunciation fallback for words outside its dictionary.
RUN apt-get update \
 && apt-get install -y --no-install-recommends espeak-ng \
 && rm -rf /var/lib/apt/lists/*

RUN pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.6.0 torchaudio==2.6.0

RUN pip install chatterbox-tts==0.1.7 kokoro==0.9.4 "misaki[en]==0.9.4" \
    soundfile fastapi uvicorn

# Kokoro's English G2P loads this spaCy model; install it now rather than at
# first request (the container user cannot install packages).
RUN python -m spacy download en_core_web_sm

RUN useradd --create-home --uid 1000 tts && mkdir -p /hf /voices && chown tts /hf /voices
COPY --chown=tts scripts/tts_server.py /app/tts_server.py
USER tts
WORKDIR /app
EXPOSE 8200
ENTRYPOINT ["python", "/app/tts_server.py"]
