"""Ollama context window detection in the legacy model_context lookup.

Ollama's /v1/models lists no window, so the lookup used to fall back to the
name table: qwen3.5:9b matched "qwen3" (131072) while Ollama served 4096,
which truncated prompts from the front and cut replies off mid-sentence.
"""
import pytest

import src.model_context as mc

OLLAMA = "http://host.docker.internal:11434/v1"
LLAMACPP = "http://127.0.0.1:8080/v1"


class _Resp:
    def __init__(self, status=200, payload=None):
        self.status_code = status
        self.is_success = 200 <= status < 300
        self._payload = payload

    def json(self):
        return self._payload


class _FakeHttp:
    def __init__(self, routes):
        self.routes = routes
        self.calls = []

    def _answer(self, method, url):
        self.calls.append((method, url))
        for suffix, response in self.routes.items():
            if url.endswith(suffix):
                return response
        return _Resp(404)

    def get(self, url, **kwargs):
        return self._answer("GET", url)

    def post(self, url, **kwargs):
        return self._answer("POST", url)


@pytest.fixture
def http(monkeypatch):
    def install(routes):
        fake = _FakeHttp(routes)
        monkeypatch.setattr(mc.httpx, "get", fake.get)
        monkeypatch.setattr(mc.httpx, "post", fake.post)
        return fake

    monkeypatch.setattr(mc, "_configured_endpoint_kind", lambda url: None)
    monkeypatch.setattr(mc, "_ollama_last_loaded", {})
    monkeypatch.setattr("src.endpoint_resolver.build_models_url", lambda base: base.rstrip("/") + "/models")
    return install


def _ps(*models):
    return _Resp(200, {"models": [
        {"name": name, "model": name, "context_length": ctx} for name, ctx in models
    ]})


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("url,root", [
    ("http://host.docker.internal:11434/v1", "http://host.docker.internal:11434"),
    ("http://127.0.0.1:11434/v1/chat/completions", "http://127.0.0.1:11434"),
    ("http://127.0.0.1:11434/api/chat", "http://127.0.0.1:11434"),
    ("http://127.0.0.1:11434", "http://127.0.0.1:11434"),
    ("http://gpu-box/ollama/v1", "http://gpu-box/ollama"),
    ("not a url", ""),
])
def test_ollama_api_root(url, root):
    assert mc.ollama_api_root(url) == root


def test_looks_like_ollama():
    assert mc.looks_like_ollama("http://127.0.0.1:11434/v1")
    assert mc.looks_like_ollama("http://ollama.lan/v1")
    assert not mc.looks_like_ollama("http://127.0.0.1:8080/v1")


def test_ps_context_matches_names_and_latest_tag():
    payload = {"models": [
        {"name": "qwen3.5:9b", "model": "qwen3.5:9b", "context_length": 16384},
        {"name": "llama3.2:latest", "model": "llama3.2:latest", "context_length": 4096},
    ]}
    assert mc.ollama_ps_context(payload, "qwen3.5:9b") == (True, 16384)
    assert mc.ollama_ps_context(payload, "llama3.2") == (True, 4096)
    assert mc.ollama_ps_context(payload, "gemma3:4b") == (True, 0)
    assert mc.ollama_ps_context({"models": []}, "qwen3.5:9b") == (True, 0)
    assert mc.ollama_ps_context({"data": []}, "qwen3.5:9b") == (False, 0)
    assert mc.ollama_ps_context(None, "qwen3.5:9b") == (False, 0)


def test_modelfile_num_ctx_ignores_the_trained_maximum():
    assert mc.ollama_modelfile_num_ctx({"parameters": "temperature 1\nnum_ctx 32768"}) == 32768
    assert mc.ollama_modelfile_num_ctx({
        "parameters": "temperature 1",
        "model_info": {"qwen35.context_length": 262144},
    }) == 0
    assert mc.ollama_modelfile_num_ctx(None) == 0


# ---------------------------------------------------------------------------
# Legacy lookup (chat meter, trimming, native Ollama num_ctx)
# ---------------------------------------------------------------------------

def test_loaded_ollama_model_reports_its_serving_window(http):
    fake = http({"/api/ps": _ps(("qwen3.5:9b", 16384))})
    assert mc._query_context_length(OLLAMA, "qwen3.5:9b") == (16384, True)
    # Ollama answered, so no llama.cpp probes were spent on it.
    assert fake.calls == [("GET", "http://host.docker.internal:11434/api/ps")]


def test_unloaded_ollama_model_uses_modelfile_num_ctx(http):
    http({
        "/api/ps": _ps(),
        "/api/show": _Resp(200, {"parameters": "num_ctx 8192"}),
    })
    assert mc._query_context_length(OLLAMA, "qwen3.5:9b") == (8192, True)


def test_unloaded_ollama_model_reuses_the_window_it_was_last_loaded_with(http):
    fake = http({"/api/ps": _ps(("qwen3.5:9b", 16384))})
    assert mc._query_context_length(OLLAMA, "qwen3.5:9b") == (16384, True)
    fake.routes["/api/ps"] = _ps()  # unloaded after OLLAMA_KEEP_ALIVE
    fake.routes["/api/show"] = _Resp(200, {"parameters": "temperature 1"})
    assert mc._query_context_length(OLLAMA, "qwen3.5:9b") == (16384, True)
    # Another server serving the same model name has its own window.
    other = "http://127.0.0.1:11434/v1"
    assert mc._query_context_length(other, "qwen3.5:9b")[0] == mc._lookup_known("qwen3.5:9b")


def test_unloaded_ollama_model_without_num_ctx_falls_back_as_before(http):
    fake = http({
        "/api/ps": _ps(),
        "/api/show": _Resp(200, {"parameters": "", "model_info": {"qwen35.context_length": 262144}}),
    })
    ctx, known = mc._query_context_length(OLLAMA, "qwen3.5:9b")
    assert ctx == mc._lookup_known("qwen3.5:9b") and known
    assert not any(url.endswith(("/slots", "/props")) for _, url in fake.calls)


def test_llamacpp_still_wins_on_its_own_server(http):
    fake = http({"/slots": _Resp(200, [{"n_ctx": 12288}])})
    assert mc._query_context_length(LLAMACPP, "local-model") == (12288, True)
    assert not any("/api/" in url for _, url in fake.calls)


def test_non_ollama_answer_on_the_ollama_port_falls_back_to_llamacpp(http):
    # Something else on :11434 (not an Ollama process list) is not trusted.
    fake = http({
        "/api/ps": _Resp(200, {"data": []}),
        "/props": _Resp(200, {"default_generation_settings": {"n_ctx": 6144}}),
    })
    assert mc._query_context_length("http://127.0.0.1:11434/v1", "local-model") == (6144, True)
    assert not any(method == "POST" for method, _ in fake.calls)


def test_ollama_down_does_not_break_the_lookup(http):
    http({})
    ctx, _ = mc._query_context_length(OLLAMA, "qwen3.5:9b")
    assert ctx == mc._lookup_known("qwen3.5:9b")


# ---------------------------------------------------------------------------
# Which endpoints are asked (issue #5193: manually added = endpoint_kind "api")
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("url,kind,expected", [
    ("http://127.0.0.1:11434/v1", None, True),
    ("http://127.0.0.1:11434/v1", "api", True),
    ("http://192.168.1.20:11434/v1", "api", True),
    ("http://100.101.102.103:11434/v1", None, True),   # Tailscale
    ("http://ollama:11434/v1", None, True),            # Docker service name
    ("http://gpu.example.com:11434/v1", "local", True),
    ("http://127.0.0.1:11434/v1", "proxy", False),
    ("https://ollama.com/v1", "api", False),           # Ollama Cloud
    ("http://gpu.example.com:11434/v1", None, False),
    ("http://127.0.0.1:8080/v1", "api", False),        # not Ollama
])
def test_is_local_ollama_endpoint(url, kind, expected):
    assert mc.is_local_ollama_endpoint(url, kind) is expected


def test_api_kind_local_ollama_is_asked_before_the_api_shortcut(http, monkeypatch):
    monkeypatch.setattr(mc, "_configured_endpoint_kind", lambda url: "api")
    fake = http({"/api/ps": _ps(("qwen3.5:9b", 16384))})
    assert mc._query_context_length(OLLAMA, "qwen3.5:9b") == (16384, True)
    assert fake.calls == [("GET", "http://host.docker.internal:11434/api/ps")]


def test_api_kind_local_ollama_window_is_not_pinned_in_the_cache(http, monkeypatch):
    monkeypatch.setattr(mc, "_configured_endpoint_kind", lambda url: "api")
    monkeypatch.setattr(mc, "_context_cache", {})
    fake = http({"/api/ps": _ps(("qwen3.5:9b", 16384))})
    assert mc.get_context_length(OLLAMA, "qwen3.5:9b") == 16384
    fake.routes["/api/ps"] = _ps(("qwen3.5:9b", 32768))  # reloaded with a new window
    assert mc.get_context_length(OLLAMA, "qwen3.5:9b") == 32768


def test_docker_service_name_ollama_is_asked(http):
    http({"/api/ps": _ps(("qwen3.5:9b", 8192))})
    assert mc._query_context_length("http://ollama:11434/v1", "qwen3.5:9b") == (8192, True)


def test_proxy_kind_is_never_probed(http, monkeypatch):
    monkeypatch.setattr(mc, "_configured_endpoint_kind", lambda url: "proxy")
    fake = http({"/api/ps": _ps(("qwen3.5:9b", 16384))})
    mc._query_context_length(OLLAMA, "qwen3.5:9b")
    assert not any("/api/" in url for _, url in fake.calls)
