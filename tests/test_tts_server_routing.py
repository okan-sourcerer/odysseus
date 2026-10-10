"""scripts/tts_server.py: language detection, engine routing and voice names.

The routing helpers are compiled out of the script via AST so Kokoro,
Chatterbox and torch need not be installed.
"""
import ast
import re
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "tts_server.py"
_FUNCS = ("detect_language", "resolve_engine", "kokoro_voice", "max_plausible_seconds",
          "split_for_speech")
_CONSTS = ("_TURKISH_LETTERS", "_TURKISH_WORDS", "_WORD_RE", "MODELS", "OPENAI_VOICES",
           "KOKORO_DEFAULT_VOICE", "CHATTERBOX_SECONDS_PER_CHAR", "CHATTERBOX_SECONDS_SLACK",
           "CHATTERBOX_CHUNK_CHARS", "_SENTENCE_RE")


def _load():
    tree = ast.parse(_SCRIPT.read_text(encoding="utf-8"))
    body = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in _FUNCS:
            body.append(node)
        elif isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id in _CONSTS for t in node.targets
        ):
            body.append(node)
    ns = {"re": re}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(_SCRIPT), "exec"), ns)
    return ns


NS = _load()


@pytest.mark.parametrize("text,language", [
    ("Hello, how are you doing today?", "en"),
    ("The weather in Izmir is lovely.", "en"),
    ("Bugün hava çok güzel.", "tr"),
    ("İstanbul'a gidiyorum.", "tr"),
    ("Merhaba, nasil gidiyor?", "tr"),           # ASCII Turkish
    ("Bu bir test ve sonra devam edecegiz.", "tr"),
    ("I like the movie Ne Zha.", "en"),           # one stray Turkish-looking word
    ("", "en"),
    ("12345 !!!", "en"),
])
def test_detect_language(text, language):
    assert NS["detect_language"](text) == language


@pytest.mark.parametrize("model,text,engine", [
    ("tts-auto", "Hello there", "kokoro"),
    ("tts-auto", "Günaydın, nasılsın?", "chatterbox"),
    ("tts-1", "Günaydın", "chatterbox"),           # OpenAI names mean auto
    ("gpt-4o-mini-tts", "Hello", "kokoro"),
    ("tts-kokoro", "Günaydın", "kokoro"),          # forced
    ("tts-chatterbox", "Hello", "chatterbox"),     # forced
    ("", "Hello", "kokoro"),
])
def test_resolve_engine(model, text, engine):
    assert NS["resolve_engine"](model, text)[0] == engine


def test_resolve_engine_passes_the_detected_language():
    assert NS["resolve_engine"]("tts-chatterbox", "Hello")[1] == "en"
    assert NS["resolve_engine"]("tts-auto", "Teşekkürler")[1] == "tr"


@pytest.mark.parametrize("voice,kokoro", [
    ("alloy", "af_alloy"), ("Nova", "af_nova"), ("onyx", "am_onyx"),
    ("bm_george", "bm_george"),       # Kokoro names pass through
    ("../etc/passwd", "af_heart"),    # anything else falls back
    ("", "af_heart"),
])
def test_kokoro_voice(voice, kokoro):
    assert NS["kokoro_voice"](voice) == kokoro


def test_models_are_recognised_as_tts_by_odysseus_settings():
    # static/js/settings/speech.js lists an endpoint as a TTS provider only
    # when a model id contains "tts" or "audio".
    assert all("tts" in model for model in NS["MODELS"])


def test_overrun_budget_only_catches_gross_overruns():
    budget = NS["max_plausible_seconds"]
    # Measured normal Chatterbox takes: 19 chars -> 1.2-4.2 s, 40 chars -> 2.1-3.8 s.
    assert budget("Tamam, teşekkürler!") >= 4.2
    assert budget("Bugün hava çok güzel, dışarı çıkalım mı?") >= 3.8
    # A babbling take (9.3 s for those 19 characters) is out of budget.
    assert budget("Tamam, teşekkürler!") < 9.3


def test_long_replies_are_split_into_sentence_chunks():
    split = NS["split_for_speech"]
    reply = "Merhaba! Bugün hava çok güzel. " + "Uzun bir cümle daha geliyor ve oldukça uzun olacak. " * 20
    chunks = split(reply)
    assert len(chunks) > 1
    assert all(len(chunk) <= 250 for chunk in chunks)
    assert " ".join(chunks).split() == reply.split()  # nothing lost or reordered


def test_a_sentence_longer_than_a_chunk_is_cut_at_a_space():
    chunks = NS["split_for_speech"]("kelime " * 100)
    assert len(chunks) > 1 and all(len(c) <= 250 for c in chunks)
    assert " ".join(chunks).split() == ("kelime " * 100).split()


def test_short_text_stays_one_chunk():
    assert NS["split_for_speech"]("Tamam, teşekkürler!") == ["Tamam, teşekkürler!"]
    assert NS["split_for_speech"]("no punctuation here") == ["no punctuation here"]
