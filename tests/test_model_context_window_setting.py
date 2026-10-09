"""Per-model context window set in Settings (model_context_windows).

Odysseus budgets against the setting. On a local Ollama it also makes Ollama
load the model with that window: Ollama's /v1 ignores num_ctx, so requests go
to a derived model created FROM the original with only num_ctx changed.
"""
import asyncio
import json

import pytest
from fastapi import HTTPException

import src.model_context as mc
import src.ollama_context_variants as variants
from routes import model_routes
from src.agent_runtime import context_resolution as cr
from src.agent_runtime.context_resolution import ContextEvidence, resolve_effective_context
from tests.test_model_routes import _PinnedFakeDb, _PinnedFakeRequest, _get_route, _make_endpoint

OLLAMA_V1 = "http://host.docker.internal:11434/v1"
OLLAMA_NATIVE = "http://host.docker.internal:11434"
LLAMACPP = "http://127.0.0.1:8080/v1"

# Opt out of the conftest metadata-network guard; probes here are faked.
CONTEXT_PROBE_NETWORK = True


class _Resp:
    def __init__(self, status=200, payload=None, text=""):
        self.status_code = status
        self.is_success = 200 <= status < 300
        self._payload = payload
        self.text = text

    def json(self):
        return self._payload


@pytest.fixture(autouse=True)
def _isolated(monkeypatch):
    monkeypatch.setattr(mc, "_configured_endpoint_kind", lambda url: None)
    monkeypatch.setattr(mc, "_ollama_variant_failures", {})
    monkeypatch.setattr(mc, "_ollama_last_loaded", {})
    monkeypatch.setattr(mc, "_context_cache", {})
    monkeypatch.setattr(variants, "_ensured", set())
    cr.clear_probe_cache()


@pytest.fixture
def setting(monkeypatch):
    """Configure {(url, model): tokens} as the Settings value."""
    def install(windows):
        monkeypatch.setattr(
            mc, "configured_context_window", lambda url, model: windows.get((url, model), 0),
        )
    return install


@pytest.fixture
def ollama(monkeypatch):
    """Fake Ollama /api/create and /api/delete; records request bodies."""
    calls = []

    def post(url, json=None, **kwargs):
        calls.append(("POST", url, json))
        return _Resp(200, {"status": "success"})

    def request(method, url, json=None, **kwargs):
        calls.append((method, url, json))
        return _Resp(200)

    monkeypatch.setattr(variants.httpx, "post", post)
    monkeypatch.setattr(variants.httpx, "request", request)
    return calls


# ---------------------------------------------------------------------------
# Parsing and naming
# ---------------------------------------------------------------------------

def test_parse_context_windows_keeps_valid_entries_only():
    raw = json.dumps({"a": 32768, "b": "8192", "c": 100, "d": True, "e": "x", "": 4096, "f": 3_000_000})
    assert mc.parse_context_windows(raw) == {"a": 32768, "b": 8192}
    assert mc.parse_context_windows(None) == {}
    assert mc.parse_context_windows("not json") == {}


@pytest.mark.parametrize("model,name", [
    ("qwen3.5:9b", "odysseus/qwen3.5:9b-ctx32768"),
    ("llama3.2", "odysseus/llama3.2:latest-ctx32768"),
    ("hf.co/user/Model-GGUF:Q4_K_M", "odysseus/Model-GGUF:Q4_K_M-ctx32768"),
])
def test_variant_names(model, name):
    assert mc.ollama_context_variant(model, 32768) == name
    assert mc.is_ollama_context_variant(name)
    assert not mc.is_ollama_context_variant(model)


# ---------------------------------------------------------------------------
# Budget (legacy lookup: chat meter, trimming, native num_ctx)
# ---------------------------------------------------------------------------

def test_setting_wins_on_a_local_ollama(setting, monkeypatch):
    setting({(OLLAMA_V1, "qwen3.5:9b"): 32768})
    monkeypatch.setattr(mc, "_discover_context_length", lambda *a: pytest.fail("no discovery needed"))
    assert mc.get_context_length_known(OLLAMA_V1, "qwen3.5:9b") == (32768, True)


def test_setting_caps_a_server_odysseus_cannot_change(setting, monkeypatch):
    monkeypatch.setattr(mc, "_discover_context_length", lambda url, model: (8192, True))
    setting({(LLAMACPP, "m"): 16384})
    assert mc.get_context_length(LLAMACPP, "m") == 8192
    setting({(LLAMACPP, "m"): 4096})
    assert mc.get_context_length(LLAMACPP, "m") == 4096


def test_setting_replaces_an_unknown_window(setting, monkeypatch):
    monkeypatch.setattr(mc, "_discover_context_length", lambda url, model: (mc.DEFAULT_CONTEXT, False))
    setting({(LLAMACPP, "m"): 16384})
    assert mc.get_context_length_known(LLAMACPP, "m") == (16384, True)


def test_failed_variant_falls_back_to_a_cap_until_retry(setting, monkeypatch):
    setting({(OLLAMA_V1, "qwen3.5:9b"): 32768})
    monkeypatch.setattr(mc, "_discover_context_length", lambda url, model: (16384, True))
    variant = mc.ollama_context_variant("qwen3.5:9b", 32768)
    mc._ollama_variant_failures[(mc.ollama_api_root(OLLAMA_V1), variant)] = mc.time.monotonic()
    assert mc.get_context_length(OLLAMA_V1, "qwen3.5:9b") == 16384
    mc._ollama_variant_failures[(mc.ollama_api_root(OLLAMA_V1), variant)] -= mc.OLLAMA_VARIANT_RETRY_SECONDS + 1
    assert mc.get_context_length(OLLAMA_V1, "qwen3.5:9b") == 32768


# ---------------------------------------------------------------------------
# Served model (what goes on the wire)
# ---------------------------------------------------------------------------

def test_served_model_creates_the_variant_once(setting, ollama):
    setting({(OLLAMA_V1, "qwen3.5:9b"): 32768})
    assert variants.served_model(OLLAMA_V1, "qwen3.5:9b") == "odysseus/qwen3.5:9b-ctx32768"
    assert variants.served_model(OLLAMA_V1, "qwen3.5:9b") == "odysseus/qwen3.5:9b-ctx32768"
    assert ollama == [("POST", "http://host.docker.internal:11434/api/create", {
        "model": "odysseus/qwen3.5:9b-ctx32768", "from": "qwen3.5:9b",
        "parameters": {"num_ctx": 32768}, "stream": False,
    })]


def test_served_model_async_matches(setting, ollama):
    setting({(OLLAMA_V1, "qwen3.5:9b"): 8192})
    served = asyncio.run(variants.served_model_async(OLLAMA_V1, "qwen3.5:9b"))
    assert served == "odysseus/qwen3.5:9b-ctx8192"
    assert len(ollama) == 1


def test_served_model_leaves_other_requests_alone(setting, ollama):
    setting({
        (OLLAMA_NATIVE, "qwen3.5:9b"): 32768,   # native API sends options.num_ctx itself
        (LLAMACPP, "m"): 8192,                   # not Ollama
    })
    assert variants.served_model(OLLAMA_NATIVE, "qwen3.5:9b") == "qwen3.5:9b"
    assert variants.served_model(LLAMACPP, "m") == "m"
    assert variants.served_model(OLLAMA_V1, "no-setting") == "no-setting"
    assert ollama == []


def test_failed_creation_sends_the_original_and_is_remembered(setting, monkeypatch):
    setting({(OLLAMA_V1, "qwen3.5:9b"): 32768})
    monkeypatch.setattr(variants.httpx, "post", lambda *a, **k: _Resp(500, text="boom"))
    assert variants.served_model(OLLAMA_V1, "qwen3.5:9b") == "qwen3.5:9b"
    assert not mc.controls_ollama_window(OLLAMA_V1, "qwen3.5:9b")


def test_llm_stream_sends_the_variant_but_keeps_name_based_behaviour(setting, ollama, monkeypatch):
    from src import llm_core

    setting({(OLLAMA_V1 + "/chat/completions", "qwen3.5:9b"): 32768})
    captured = {}

    class _Stream:
        status_code = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def aiter_lines(self):
            yield 'data: {"choices":[{"delta":{"content":"ok"}}]}'
            yield "data: [DONE]"

    class _Client:
        def stream(self, method, url, headers=None, json=None, **kwargs):
            captured.update(json or {})
            return _Stream()

    monkeypatch.setattr(llm_core, "_get_http_client", lambda: _Client())
    monkeypatch.setattr(llm_core, "_is_host_dead", lambda u: False)
    monkeypatch.setattr(llm_core, "note_model_activity", lambda *a, **k: None)
    monkeypatch.setattr(llm_core, "_clear_host_dead", lambda *a, **k: None)

    async def run():
        return [c async for c in llm_core.stream_llm(
            OLLAMA_V1 + "/chat/completions", "qwen3.5:9b", [{"role": "user", "content": "hi"}],
        )]

    asyncio.run(run())
    assert captured["model"] == "odysseus/qwen3.5:9b-ctx32768"
    assert captured.get("think") is False  # still recognised as a thinking model


# ---------------------------------------------------------------------------
# Turn resolver
# ---------------------------------------------------------------------------

def test_resolver_uses_the_setting_without_probing_a_controlled_ollama(setting, monkeypatch):
    setting({(OLLAMA_V1, "qwen3.5:9b"): 32768})

    async def no_probe(*args, **kwargs):
        raise AssertionError("a controlled Ollama window needs no probe")

    monkeypatch.setattr(cr, "_cached_probe", no_probe)
    resolution = asyncio.run(resolve_effective_context(OLLAMA_V1, "qwen3.5:9b"))
    assert (resolution.effective, resolution.evidence, resolution.source) == (
        32768, ContextEvidence.OPERATOR_DECLARED, "model_setting",
    )


def test_resolver_setting_cannot_exceed_a_measured_window(setting, monkeypatch):
    setting({(LLAMACPP, "m"): 16384})
    monkeypatch.setattr(mc, "is_local_endpoint", lambda url: True)

    async def probe(endpoint_url, model, headers, deadline_seconds, clock):
        return cr._ProbeResult((cr.ContextObservation(
            ContextEvidence.RUNTIME_CONFIRMED, 8192, "llamacpp_slots"),), (), True), False

    monkeypatch.setattr(cr, "_cached_probe", probe)
    resolution = asyncio.run(resolve_effective_context(LLAMACPP, "m"))
    assert (resolution.effective, resolution.source) == (8192, "llamacpp_slots")


# ---------------------------------------------------------------------------
# Settings API
# ---------------------------------------------------------------------------

def _patch(monkeypatch, ep, body):
    db = _PinnedFakeDb([ep])
    monkeypatch.setattr(model_routes, "SessionLocal", lambda: db)
    monkeypatch.setattr(model_routes, "require_admin", lambda request: None)
    endpoint = _get_route("/api/model-endpoints/{ep_id}/models", "PATCH")
    return asyncio.run(endpoint("ep1", _PinnedFakeRequest(body=body)))


def test_patch_saves_and_clears_context_windows(monkeypatch, ollama):
    ep = _make_endpoint(base_url=OLLAMA_V1, model_context_windows=None, model_tool_modes=None)
    result = _patch(monkeypatch, ep, {"model_context_windows": {"qwen3.5:9b": 32768, "gemma3:4b": "8192"}})
    assert json.loads(ep.model_context_windows) == {"qwen3.5:9b": 32768, "gemma3:4b": 8192}
    assert result["model_context_windows"] == {"qwen3.5:9b": 32768, "gemma3:4b": 8192}

    _patch(monkeypatch, ep, {"model_context_windows": {"qwen3.5:9b": None}})
    assert json.loads(ep.model_context_windows) == {"gemma3:4b": 8192}
    # The derived model of the cleared window is removed from Ollama.
    assert ("DELETE", "http://host.docker.internal:11434/api/delete",
            {"model": "odysseus/qwen3.5:9b-ctx32768"}) in ollama


@pytest.mark.parametrize("value", [100, 5_000_000, "lots", True])
def test_patch_rejects_invalid_context_windows(monkeypatch, value):
    ep = _make_endpoint(base_url=OLLAMA_V1, model_context_windows=None, model_tool_modes=None)
    with pytest.raises(HTTPException) as exc:
        _patch(monkeypatch, ep, {"model_context_windows": {"qwen3.5:9b": value}})
    assert exc.value.status_code == 400
    assert ep.model_context_windows is None


def test_ollama_model_lists_hide_derived_models(monkeypatch):
    monkeypatch.setattr(
        model_routes, "_probe_endpoint_models",
        lambda base, key=None, timeout=5: ["qwen3.5:9b", "odysseus/qwen3.5:9b-ctx32768"],
    )
    assert model_routes._probe_endpoint(OLLAMA_V1) == ["qwen3.5:9b"]
    assert model_routes._probe_endpoint(LLAMACPP) == ["qwen3.5:9b", "odysseus/qwen3.5:9b-ctx32768"]
