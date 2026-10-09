"""Read-aloud (TTS) is reachable in the UI and TTS endpoints stay out of chat pickers."""
import re
import threading
from pathlib import Path

from routes import model_routes
from tests.test_model_routes import (
    _ImmediateThread,
    _RouteDb,
    _RouteModelEndpoint,
    _route_endpoint,
    _route_ep,
    _route_request,
)

ROOT = Path(__file__).resolve().parents[1]
INDEX = (ROOT / "static" / "index.html").read_text(encoding="utf-8")
SPEECH = (ROOT / "static" / "js" / "settings" / "speech.js").read_text(encoding="utf-8")


def test_tts_endpoints_are_not_offered_as_chat_models(monkeypatch):
    chat = _route_ep("ollama", "http://127.0.0.1:11434/v1", cached_models=["qwen3.5:9b"])
    tts = _route_ep("voices", "http://tts:8200/v1", cached_models=["tts-auto", "tts-kokoro"])
    tts.model_type = "tts"
    db = _RouteDb([chat, tts])
    router = model_routes.setup_model_routes(model_discovery=None)
    monkeypatch.setattr(model_routes, "ModelEndpoint", _RouteModelEndpoint)
    monkeypatch.setattr(model_routes, "SessionLocal", lambda: db)
    monkeypatch.setattr(model_routes, "_auth_disabled", lambda: True)
    monkeypatch.setattr(model_routes, "build_chat_url", lambda base: f"{base}/chat/completions")
    monkeypatch.setattr(model_routes, "_probe_endpoint", lambda *a, **k: [])
    monkeypatch.setattr(threading, "Thread", _ImmediateThread)

    result = _route_endpoint(router, "/api/models")(_route_request())

    assert [item["endpoint_name"] for item in result["items"]] == ["ollama"]


def _tag(element_id: str) -> str:
    return re.search(rf'<[^>]*id="{element_id}"[^>]*>', INDEX).group(0)


def test_tts_settings_panel_and_mode_button_are_not_force_hidden():
    panel = INDEX.split('id="set-ttsEnabledToggle"', 1)[0].rsplit('<div class="admin-card"', 1)[1]
    assert not panel.lstrip(">").startswith(" hidden") and "hidden" not in panel.split(">", 1)[0]
    # app.js shows the button once a provider is configured; a `hidden`
    # attribute would override that.
    assert " hidden" not in _tag("overflow-tts-btn")


def test_settings_lists_the_endpoints_own_tts_models():
    assert '<option value="local">' not in INDEX  # built-in Kokoro needs a GPU in the app container
    assert "endpointModels[ep.id] = ttsModels;" in SPEECH
    assert "fillModelOptions();" in SPEECH
