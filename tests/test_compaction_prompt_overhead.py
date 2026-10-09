"""Auto-compaction counts the prompt overhead each request carries.

maybe_compact sees the conversation, but the request also carries the agent
system prompt, tool schemas and injected context (~7k tokens on a local
model). Judged on the conversation alone, compaction fired too late and the
final request trim dropped the oldest messages instead of summarizing them.
"""
import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

import src.context_compactor as cc

ROOT = Path(__file__).resolve().parents[1]
WINDOW = 32_768


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    monkeypatch.setattr(cc, "_prompt_overhead", cc.OrderedDict())
    monkeypatch.setattr(cc, "_last_prompt_overhead", 0)


def _compact(monkeypatch, session, conversation_tokens):
    calls = []

    async def summarize(*args, **kwargs):
        calls.append(1)
        return "Earlier: the user and assistant discussed the plan."

    monkeypatch.setattr(cc, "get_context_length", lambda url, model: WINDOW)
    monkeypatch.setattr(cc, "estimate_tokens", lambda msgs: conversation_tokens)
    monkeypatch.setattr(cc, "llm_call_async", summarize)
    monkeypatch.setattr(cc, "resolve_endpoint", lambda *a, **k: (None, None, None))
    monkeypatch.setattr(cc, "_update_session_history", lambda *a, **k: None)
    monkeypatch.setattr(cc, "get_setting", lambda key, default=None: default)
    messages = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i}"} for i in range(8)]
    _, _, compacted = asyncio.run(cc.maybe_compact(
        session, "http://ollama:11434/v1", "qwen3.5:9b", messages, {},
    ))
    return compacted, calls


def test_overhead_pushes_a_long_chat_over_the_threshold(monkeypatch):
    session = SimpleNamespace(id="chat-1")
    # 22k conversation tokens is 67% of 32k; with the measured 7k overhead
    # the real request is 89%, past the default 85% threshold.
    cc.record_prompt_overhead("chat-1", request_tokens=7_230, conversation_tokens=230)
    compacted, calls = _compact(monkeypatch, session, 22_000)
    assert compacted and calls == [1]


def test_without_overhead_the_same_chat_is_not_compacted(monkeypatch):
    compacted, calls = _compact(monkeypatch, SimpleNamespace(id="chat-1"), 22_000)
    assert not compacted and calls == []


def test_overhead_uses_the_smallest_recent_sample_per_session():
    cc.record_prompt_overhead("chat-1", 7_500, 500)      # plain turn: 7000
    cc.record_prompt_overhead("chat-1", 19_500, 500)     # carried tool output: 19000
    assert cc.prompt_overhead_tokens("chat-1") == 7_000


def test_unmeasured_session_falls_back_to_the_last_overhead_seen():
    cc.record_prompt_overhead("chat-1", 7_300, 300)
    assert cc.prompt_overhead_tokens("chat-2") == 7_000
    assert cc.prompt_overhead_tokens(None) == 7_000


@pytest.mark.parametrize("request_tokens,conversation_tokens", [(0, 0), (300, 500), ("x", 1)])
def test_unusable_measurements_are_ignored(request_tokens, conversation_tokens):
    cc.record_prompt_overhead("chat-1", request_tokens, conversation_tokens)
    assert cc.prompt_overhead_tokens("chat-1") == 0


def test_session_samples_are_bounded(monkeypatch):
    monkeypatch.setattr(cc, "_OVERHEAD_SESSIONS", 3)
    for i in range(5):
        cc.record_prompt_overhead(f"chat-{i}", 1_000 + i, 0)
    assert list(cc._prompt_overhead) == ["chat-2", "chat-3", "chat-4"]


def test_chat_route_records_overhead_with_every_metrics_event():
    source = (ROOT / "routes" / "chat_routes.py").read_text(encoding="utf-8")
    assert source.count("_note_prompt_overhead(last_metrics)") == 3
    assert 'metrics.get("request_context_tokens") or metrics.get("input_tokens")' in source
