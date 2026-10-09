"""Apply a per-model context window to a local Ollama's OpenAI-compatible API.

Ollama's /v1 endpoints ignore ``num_ctx``: the window a model loads with comes
from OLLAMA_CONTEXT_LENGTH or the model's Modelfile. When a context window is
set for a model in Settings, requests on /v1 are sent to a derived model
created FROM the original with only ``num_ctx`` changed (see
``model_context.ollama_context_variant``). It shares the original's weights,
so it costs no disk space, and it is created once, on first use.

The native /api/chat path needs none of this: it already sends
``options.num_ctx`` from ``get_context_length``, which returns the setting.
"""
import asyncio
import logging
import time
from typing import Optional, Tuple
from urllib.parse import urlparse

import httpx

from src import model_context as mc

logger = logging.getLogger(__name__)

CREATE_TIMEOUT_SECONDS = 60.0

# (server root, variant) pairs known to exist in this process.
_ensured: set = set()


def _target(endpoint_url: str, model: str) -> Optional[Tuple[str, str, int]]:
    """``(root, variant, num_ctx)`` when ``model`` is served through a derived
    model on this endpoint, else None."""
    try:
        path = urlparse(endpoint_url or "").path or ""
    except ValueError:
        return None
    if "/v1" not in path or mc.is_ollama_context_variant(model) or not mc.looks_like_ollama(endpoint_url):
        return None
    configured = mc.configured_context_window(endpoint_url, model)
    if not configured or not mc.is_local_ollama_endpoint(endpoint_url):
        return None
    variant = mc.ollama_context_variant(model, configured)
    if mc.ollama_variant_failed(endpoint_url, variant):
        return None
    return mc.ollama_api_root(endpoint_url), variant, configured


def _create(root: str, variant: str, model: str, num_ctx: int) -> bool:
    try:
        response = httpx.post(
            f"{root}/api/create",
            json={"model": variant, "from": model, "parameters": {"num_ctx": num_ctx}, "stream": False},
            timeout=CREATE_TIMEOUT_SECONDS,
        )
        ok = response.is_success
        detail = "" if ok else f"HTTP {response.status_code}: {response.text[:200]}"
    except Exception as exc:
        ok, detail = False, type(exc).__name__
    if ok:
        _ensured.add((root, variant))
        mc._ollama_variant_failures.pop((root, variant), None)
        logger.info("Created Ollama model %s (%s with num_ctx=%s)", variant, model, num_ctx)
    else:
        mc._ollama_variant_failures[(root, variant)] = time.monotonic()
        logger.warning(
            "Could not create Ollama model %s for the %s-token window of %s (%s); "
            "sending %s unchanged", variant, num_ctx, model, detail, model,
        )
    return ok


def served_model(endpoint_url: str, model: str) -> str:
    """Model name to send to ``endpoint_url`` for ``model``."""
    target = _target(endpoint_url, model)
    if not target:
        return model
    root, variant, num_ctx = target
    if (root, variant) in _ensured or _create(root, variant, model, num_ctx):
        return variant
    return model


async def served_model_async(endpoint_url: str, model: str) -> str:
    """``served_model`` for async callers; creation runs off the event loop."""
    target = _target(endpoint_url, model)
    if not target:
        return model
    root, variant, num_ctx = target
    if (root, variant) in _ensured:
        return variant
    if await asyncio.to_thread(_create, root, variant, model, num_ctx):
        return variant
    return model


def delete_variant(endpoint_url: str, model: str, num_ctx: int) -> None:
    """Best-effort removal of a derived model that a setting no longer uses."""
    if not num_ctx or not mc.is_local_ollama_endpoint(endpoint_url):
        return
    root = mc.ollama_api_root(endpoint_url)
    variant = mc.ollama_context_variant(model, num_ctx)
    _ensured.discard((root, variant))
    try:
        httpx.request("DELETE", f"{root}/api/delete", json={"model": variant}, timeout=10.0)
    except Exception as exc:
        logger.debug("Could not delete Ollama model %s: %s", variant, type(exc).__name__)
