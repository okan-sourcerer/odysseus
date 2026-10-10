"""diffusion_server.py on-demand GPU use: --lazy-load, --idle-unload, --unload-ollama.

The helpers are compiled out of the script via AST (no torch / diffusers
needed), with fakes for Ollama, the model loader and torch.
"""
import ast
import asyncio
import functools
import inspect
import logging
import sys
import threading
import time
import types
from pathlib import Path
from types import SimpleNamespace

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "diffusion_server.py"
_NAMES = ("_unload_ollama", "_release_gpu_users", "_unload_model", "_enter_model_job", "_exit_model_job",
          "_uses_model")


class _Resp:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


class _FakeOllama:
    """/api/ps lists models until each was posted keep_alive 0."""

    def __init__(self, models):
        self.loaded = list(models)
        self.calls = []

    def get(self, url, timeout=None):
        self.calls.append(("GET", url))
        return _Resp({"models": [{"name": m, "model": m} for m in self.loaded]})

    def post(self, url, json=None, timeout=None):
        self.calls.append(("POST", url, json))
        self.loaded.remove(json["model"])
        return _Resp({})


def _helpers(monkeypatch, *, lazy=True, ollama_url="", ollama=None, release=()):
    tree = ast.parse(_SCRIPT.read_text(encoding="utf-8"))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in _NAMES]
    assert sorted(n.name for n in nodes) == sorted(_NAMES)
    if ollama is not None:
        monkeypatch.setitem(sys.modules, "httpx", types.SimpleNamespace(get=ollama.get, post=ollama.post))
    ns = {
        "_args": SimpleNamespace(lazy_load=lazy, unload_ollama=ollama_url, idle_unload=30,
                                 release_gpu=list(release)),
        "_pipe": None, "_inpaint_pipe": None, "_img2img_pipe": None,
        "_model_lock": threading.RLock(), "_active_jobs": 0, "_last_used": 0.0,
        "asyncio": asyncio, "functools": functools, "time": time,
        "logger": logging.getLogger("test-diffusion"),
        "torch": SimpleNamespace(cuda=SimpleNamespace(empty_cache=lambda: None)),
    }
    loads = []

    def load_model():
        loads.append(1)
        ns["_pipe"] = object()

    ns["load_model"] = load_model
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(_SCRIPT), "exec"), ns)
    return ns, loads


def test_lazy_load_loads_on_first_job_only(monkeypatch):
    ns, loads = _helpers(monkeypatch)
    ns["_enter_model_job"]()
    ns["_exit_model_job"]()
    ns["_enter_model_job"]()
    ns["_exit_model_job"]()
    assert loads == [1]
    assert ns["_active_jobs"] == 0 and ns["_last_used"] > 0


def test_without_lazy_load_a_job_does_not_load(monkeypatch):
    ns, loads = _helpers(monkeypatch, lazy=False)
    ns["_enter_model_job"]()
    assert loads == [] and ns["_pipe"] is None


def test_ollama_models_are_unloaded_before_a_job(monkeypatch):
    ollama = _FakeOllama(["qwen3.5:9b", "llama3.2:latest"])
    ns, _ = _helpers(monkeypatch, ollama_url="http://ollama:11434/", ollama=ollama)
    ns["_enter_model_job"]()
    posts = [c for c in ollama.calls if c[0] == "POST"]
    assert posts == [
        ("POST", "http://ollama:11434/api/generate", {"model": "qwen3.5:9b", "keep_alive": 0}),
        ("POST", "http://ollama:11434/api/generate", {"model": "llama3.2:latest", "keep_alive": 0}),
    ]
    assert ollama.loaded == []


def test_no_ollama_url_means_no_ollama_calls(monkeypatch):
    ollama = _FakeOllama(["qwen3.5:9b"])
    ns, _ = _helpers(monkeypatch, ollama_url="", ollama=ollama)
    ns["_enter_model_job"]()
    assert ollama.calls == []


def test_other_gpu_services_are_asked_to_release_before_a_job(monkeypatch):
    posted = []
    fake_httpx = types.SimpleNamespace(post=lambda url, timeout=None: posted.append(url))
    monkeypatch.setitem(sys.modules, "httpx", fake_httpx)
    ns, _ = _helpers(monkeypatch, release=["http://tts:8200/v1/unload"])
    ns["_enter_model_job"]()
    assert posted == ["http://tts:8200/v1/unload"]


def test_a_failed_release_request_does_not_block_the_job(monkeypatch):
    def boom(url, timeout=None):
        raise ConnectionError("tts is down")

    monkeypatch.setitem(sys.modules, "httpx", types.SimpleNamespace(post=boom))
    ns, loads = _helpers(monkeypatch, release=["http://tts:8200/v1/unload"])
    ns["_enter_model_job"]()
    assert loads == [1]


def test_unload_model_releases_every_pipeline(monkeypatch):
    ns, _ = _helpers(monkeypatch)
    ns["_pipe"] = ns["_inpaint_pipe"] = ns["_img2img_pipe"] = object()
    ns["_unload_model"]()
    assert ns["_pipe"] is ns["_inpaint_pipe"] is ns["_img2img_pipe"] is None


def test_uses_model_keeps_the_endpoint_signature_and_counts_jobs(monkeypatch):
    ns, loads = _helpers(monkeypatch)
    seen = []

    def generate(req: str, n: int = 1):
        seen.append(ns["_active_jobs"])
        return req * n

    async def edit(prompt: str):
        seen.append(ns["_active_jobs"])
        return prompt.upper()

    sync_wrapped = ns["_uses_model"](generate)
    async_wrapped = ns["_uses_model"](edit)

    assert inspect.signature(sync_wrapped) == inspect.signature(generate)  # FastAPI reads it
    assert inspect.iscoroutinefunction(async_wrapped)
    assert sync_wrapped("a", n=2) == "aa"
    assert asyncio.run(async_wrapped("b")) == "B"
    assert seen == [1, 1] and ns["_active_jobs"] == 0 and loads == [1]


def test_server_wires_the_options():
    source = _SCRIPT.read_text(encoding="utf-8")
    assert source.count("@_uses_model\n") == 4  # generations, edits, inpaint, harmonize
    for flag in ('"--lazy-load"', '"--idle-unload"', '"--unload-ollama"', '"--release-gpu"'):
        assert flag in source


def _unload_endpoint(pipe, active_jobs):
    tree = ast.parse(_SCRIPT.read_text(encoding="utf-8"))
    [node] = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "unload"]
    node.decorator_list = []
    released = []
    ns = {
        "_model_lock": threading.RLock(), "_pipe": pipe, "_active_jobs": active_jobs,
        "_model_id": "FLUX", "logger": logging.getLogger("test-diffusion"),
        "_unload_model": lambda: released.append(1),
    }
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(_SCRIPT), "exec"), ns)
    return ns["unload"], released


def test_unload_endpoint_frees_an_idle_model_only():
    unload, released = _unload_endpoint(object(), 0)
    assert unload() == {"released": True} and released == [1]
    unload, released = _unload_endpoint(object(), 1)    # a job is running
    assert unload()["released"] is False and released == []
    unload, released = _unload_endpoint(None, 0)        # nothing loaded
    assert unload()["released"] is False and released == []
