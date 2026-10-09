#!/usr/bin/env python3
"""OpenAI-compatible local text-to-speech server: Kokoro + Chatterbox Multilingual.

Odysseus' "API endpoint" TTS provider POSTs {model, input, voice, speed,
response_format} to <endpoint>/audio/speech; this server answers that call.

- Kokoro-82M (Apache-2.0) runs on the CPU: fast, English (and a few other
  languages), 50+ preset voices.
- Chatterbox Multilingual (MIT) covers 23 languages including Turkish and can
  clone a voice from a short reference clip. It runs on the GPU when there is
  room, else on the CPU, and is loaded on first use and released when idle.

model "tts-auto" (the default) picks per request: Turkish text goes to
Chatterbox, everything else to Kokoro; "tts-kokoro" and "tts-chatterbox"
force one. OpenAI model names (tts-1, ...) mean tts-auto, and OpenAI voice
names map to the matching Kokoro voices, so OpenAI-style clients work as is.

The server never asks Ollama to unload: read-aloud runs while the chat model
is still streaming the reply it reads.
"""
import argparse
import io
import logging
import re
import threading
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("tts_server")

KOKORO_SAMPLE_RATE = 24000
# Chatterbox occasionally keeps generating after the text ends (extra
# syllables or babble), worst on short inputs. Takes far beyond this budget
# (~2x normal pacing plus leading/trailing silence) are regenerated once,
# keeping the shorter take. A tighter budget mostly rejected normal takes
# and multiplied latency, so it only catches gross overruns.
CHATTERBOX_SECONDS_PER_CHAR = 0.15
CHATTERBOX_SECONDS_SLACK = 1.5
CHATTERBOX_MAX_TAKES = 2
KOKORO_DEFAULT_VOICE = "af_heart"
MAX_INPUT_CHARS = 5000

_args = None
app = FastAPI(title="Odysseus TTS Server")


# ---------------------------------------------------------------------------
# Language routing
# ---------------------------------------------------------------------------

_TURKISH_LETTERS = set("çğıöşüÇĞİÖŞÜ")
# Common Turkish words written without Turkish-only letters, so short ASCII
# sentences ("Merhaba, nasil gidiyor?") still route to the Turkish voice.
_TURKISH_WORDS = {
    "ve", "bir", "bu", "da", "de", "ne", "ile", "ama", "evet", "hayir", "merhaba",
    "nasil", "neden", "icin", "gibi", "daha", "cok", "sen", "ben", "biz", "siz",
    "var", "yok", "mi", "mu", "olan", "olarak", "kadar", "sonra", "once", "simdi",
    "tamam", "tesekkurler", "lutfen", "iyi", "kotu", "ki", "diye", "veya", "hem",
}
_WORD_RE = re.compile(r"[A-Za-zÇĞİÖŞÜçğıöşü]+")


def detect_language(text: str) -> str:
    """'tr' for Turkish text, else 'en'. A heuristic, tuned to tell the two apart."""
    if any(ch in _TURKISH_LETTERS for ch in text):
        return "tr"
    words = [w.lower() for w in _WORD_RE.findall(text)]
    if not words:
        return "en"
    hits = sum(1 for w in words if w in _TURKISH_WORDS)
    return "tr" if hits >= 2 or hits / len(words) >= 0.3 else "en"


MODELS = ("tts-auto", "tts-kokoro", "tts-chatterbox")

# OpenAI voice names -> Kokoro voices (Kokoro ships several of the same names).
OPENAI_VOICES = {
    "alloy": "af_alloy", "echo": "am_echo", "fable": "bm_fable", "onyx": "am_onyx",
    "nova": "af_nova", "shimmer": "af_heart", "ash": "am_adam", "coral": "af_bella",
    "sage": "af_sarah", "ballad": "bm_george", "verse": "am_michael",
}


def max_plausible_seconds(text: str) -> float:
    """Longest believable speech for ``text``; longer output means overrun."""
    return CHATTERBOX_SECONDS_SLACK + CHATTERBOX_SECONDS_PER_CHAR * len(text)


def resolve_engine(model: str, text: str) -> tuple[str, str]:
    """(engine, language) for a request. engine is 'kokoro' or 'chatterbox'."""
    model = (model or "tts-auto").strip().lower()
    language = detect_language(text)
    if model in ("tts-kokoro", "kokoro"):
        return "kokoro", language
    if model in ("tts-chatterbox", "chatterbox"):
        return "chatterbox", language
    return ("chatterbox" if language == "tr" else "kokoro"), language


def kokoro_voice(voice: str) -> str:
    """A Kokoro voice name for ``voice`` (Kokoro or OpenAI naming)."""
    voice = (voice or "").strip().lower()
    voice = OPENAI_VOICES.get(voice, voice)
    return voice if re.fullmatch(r"[a-z]{2}_[a-z0-9]+", voice) else KOKORO_DEFAULT_VOICE


# ---------------------------------------------------------------------------
# Engines
# ---------------------------------------------------------------------------

class _Kokoro:
    """Kokoro-82M on the CPU, one pipeline per language code (voice prefix)."""

    def __init__(self):
        self._pipelines = {}
        self._lock = threading.Lock()

    def _pipeline(self, lang_code: str):
        with self._lock:
            if lang_code not in self._pipelines:
                from kokoro import KPipeline
                logger.info("Loading Kokoro pipeline (lang_code=%s) on cpu", lang_code)
                self._pipelines[lang_code] = KPipeline(lang_code=lang_code, device="cpu")
            return self._pipelines[lang_code]

    def synthesize(self, text: str, voice: str, speed: float) -> tuple[np.ndarray, int]:
        voice = kokoro_voice(voice)
        # Kokoro voices are named <lang><gender>_<name>; the first letter is
        # the pipeline language (a = American English, b = British, ...).
        pipeline = self._pipeline(voice[0])
        chunks = [audio for _, _, audio in pipeline(text, voice=voice, speed=speed) if audio is not None]
        if not chunks:
            raise RuntimeError("Kokoro produced no audio")
        audio = np.concatenate([np.asarray(chunk, dtype=np.float32) for chunk in chunks])
        return audio, KOKORO_SAMPLE_RATE


class _Chatterbox:
    """Chatterbox Multilingual: loaded on first use, released after idle."""

    def __init__(self, voices_dir: Path, idle_unload: float, cfg_weight: float, temperature: float):
        self._voices_dir = voices_dir
        self._idle_unload = idle_unload
        self._cfg_weight = cfg_weight
        self._temperature = temperature
        self._model = None
        self._device = None
        self._lock = threading.RLock()
        self._last_used = 0.0
        if idle_unload > 0:
            threading.Thread(target=self._idle_loop, name="chatterbox-idle", daemon=True).start()

    def _load(self):
        import torch
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS

        if torch.cuda.is_available():
            try:
                logger.info("Loading Chatterbox Multilingual on cuda")
                self._model = ChatterboxMultilingualTTS.from_pretrained(device="cuda")
                self._device = "cuda"
                return
            except torch.cuda.OutOfMemoryError:
                # The chat model holds the GPU; read-aloud must not evict it.
                logger.warning("Not enough free GPU memory for Chatterbox; using the CPU")
                self._model = None
                torch.cuda.empty_cache()
        logger.info("Loading Chatterbox Multilingual on cpu")
        self._model = ChatterboxMultilingualTTS.from_pretrained(device="cpu")
        self._device = "cpu"

    def _reference_clip(self, voice: str):
        """A <voice>.wav in the voices folder is cloned; otherwise the built-in voice."""
        if voice and re.fullmatch(r"[A-Za-z0-9_.-]+", voice):
            clip = self._voices_dir / f"{voice}.wav"
            if clip.is_file():
                return str(clip)
        return None

    def synthesize(self, text: str, voice: str, language: str) -> tuple[np.ndarray, int]:
        with self._lock:
            if self._model is None:
                self._load()
            sample_rate = int(self._model.sr)
            budget = max_plausible_seconds(text)
            best = None
            try:
                for take in range(1, CHATTERBOX_MAX_TAKES + 1):
                    wav = self._model.generate(
                        text, language_id=language, audio_prompt_path=self._reference_clip(voice),
                        cfg_weight=self._cfg_weight, temperature=self._temperature,
                    )
                    audio = wav.squeeze().detach().cpu().numpy().astype(np.float32)
                    if best is None or len(audio) < len(best):
                        best = audio
                    seconds = len(audio) / sample_rate
                    if seconds <= budget:
                        break
                    logger.info("Chatterbox take %d ran %.1fs for %d chars (budget %.1fs); retrying",
                                take, seconds, len(text), budget)
            finally:
                self._last_used = time.monotonic()
            return best, sample_rate

    def unload(self) -> bool:
        """Release the model (and its GPU memory). True if one was loaded."""
        with self._lock:
            if self._model is None:
                return False
            self._model = None
            import gc
            gc.collect()
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass
            return True

    def _idle_loop(self):
        while True:
            time.sleep(min(15.0, max(1.0, self._idle_unload / 4)))
            with self._lock:
                if self._model is not None and time.monotonic() - self._last_used >= self._idle_unload:
                    self.unload()
                    logger.info("Unloaded Chatterbox after %ss idle", self._idle_unload)

    @property
    def state(self) -> str:
        return self._device if self._model is not None else "unloaded"


_kokoro = _Kokoro()
_chatterbox = None


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

class SpeechRequest(BaseModel):
    model: str = "tts-auto"
    input: str
    voice: str = KOKORO_DEFAULT_VOICE
    speed: float = 1.0
    response_format: str = "mp3"


def encode_audio(audio: np.ndarray, sample_rate: int, response_format: str) -> tuple[bytes, str]:
    fmt = (response_format or "mp3").lower()
    formats = {"mp3": ("MP3", "audio/mpeg"), "wav": ("WAV", "audio/wav"), "flac": ("FLAC", "audio/flac")}
    if fmt not in formats:
        raise HTTPException(400, f"response_format must be one of {sorted(formats)}")
    sf_format, media_type = formats[fmt]
    buf = io.BytesIO()
    sf.write(buf, np.clip(audio, -1.0, 1.0), sample_rate, format=sf_format)
    return buf.getvalue(), media_type


@app.post("/v1/audio/speech")
def speech(req: SpeechRequest):
    text = (req.input or "").strip()
    if not text:
        raise HTTPException(400, "input is empty")
    text = text[:MAX_INPUT_CHARS]
    engine, language = resolve_engine(req.model, text)
    speed = min(max(float(req.speed or 1.0), 0.5), 2.0)
    start = time.time()
    if engine == "kokoro":
        audio, sample_rate = _kokoro.synthesize(text, req.voice, speed)
    else:
        audio, sample_rate = _chatterbox.synthesize(text, req.voice, language)
    data, media_type = encode_audio(audio, sample_rate, req.response_format)
    logger.info("%s/%s: %d chars -> %.1fs audio in %.1fs", engine, language, len(text),
                len(audio) / sample_rate, time.time() - start)
    return Response(content=data, media_type=media_type, headers={"X-TTS-Engine": engine})


@app.post("/v1/unload")
def unload():
    """Free Chatterbox's GPU memory, e.g. before an image job needs the GPU."""
    released = _chatterbox.unload() if _chatterbox else False
    if released:
        logger.info("Unloaded Chatterbox on request")
    return {"released": released}


@app.get("/v1/models")
def list_models():
    return {"object": "list", "data": [
        {"id": model_id, "object": "model", "owned_by": "local"}
        for model_id in MODELS
    ]}


@app.get("/health")
def health():
    return {"status": "ok", "kokoro": "cpu", "chatterbox": _chatterbox.state if _chatterbox else "off"}


def main():
    global _args, _chatterbox
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8200)
    parser.add_argument("--voices-dir", default="/voices",
                        help="Folder of <name>.wav reference clips Chatterbox can clone")
    parser.add_argument("--idle-unload", type=float, default=300,
                        help="Release Chatterbox after this many idle seconds (0 = keep loaded)")
    parser.add_argument("--chatterbox-cfg", type=float, default=0.5,
                        help="Chatterbox cfg_weight (lower = looser pacing)")
    parser.add_argument("--chatterbox-temperature", type=float, default=0.8,
                        help="Chatterbox sampling temperature")
    parser.add_argument("--warm-kokoro", action="store_true",
                        help="Load Kokoro's English pipeline at startup")
    _args = parser.parse_args()
    _chatterbox = _Chatterbox(Path(_args.voices_dir), _args.idle_unload,
                              _args.chatterbox_cfg, _args.chatterbox_temperature)
    if _args.warm_kokoro:
        _kokoro._pipeline("a")
    uvicorn.run(app, host=_args.host, port=_args.port)


if __name__ == "__main__":
    main()
