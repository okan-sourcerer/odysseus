"""Regression tests for local image generation (SDXL & co. on a self-hosted
diffusion endpoint).

- Direct image-model chats passed the raw message through the line protocol
  ("prompt\\nmodel\\nsize"), so a multi-line prompt kept only its first line and
  the second line was read as the model name; the size was pinned to 512x512.
- The chat tool forced every non-OpenAI model onto DALL-E 3's size list and
  never sent ``quality`` to self-hosted backends.
- ``generate_image`` had no native function schema, so models using native
  tool calls never received the tool.
"""
import ast
import json
import os

from src import ai_interaction

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _patch_capture(monkeypatch):
    """Capture the images API payload, then fail the request so the call stops
    before touching the filesystem or the gallery DB."""
    captured = {}

    class _Resp:
        status_code = 500
        text = "stop after capture"

        def json(self):
            return {"error": "stop after capture"}

    class _AsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def post(self, url, json=None, headers=None, **kwargs):
            captured["url"] = url
            captured["payload"] = json
            return _Resp()

    import httpx
    import src.settings as settings

    monkeypatch.setattr(settings, "load_settings", lambda: {})
    monkeypatch.setattr(httpx, "AsyncClient", _AsyncClient)
    monkeypatch.setattr(
        ai_interaction,
        "_resolve_model",
        lambda model_spec, owner=None, model_type=None: (
            "http://diffusion.local:8101/v1/chat/completions",
            model_spec,
            {},
        ),
    )
    return captured


async def test_json_prompt_keeps_every_line(monkeypatch):
    captured = _patch_capture(monkeypatch)
    prompt = "a beach with mild waves\ndock on the left side\nrain clouds in the distance"

    await ai_interaction.do_generate_image(json.dumps({
        "prompt": prompt, "model": "Juggernaut-XL_v9.safetensors", "size": "1024x1024",
    }))

    payload = captured["payload"]
    assert payload["prompt"] == prompt
    assert payload["model"] == "Juggernaut-XL_v9.safetensors"
    assert payload["size"] == "1024x1024"
    assert captured["url"] == "http://diffusion.local:8101/v1/images/generations"


async def test_line_format_still_works_for_existing_callers(monkeypatch):
    captured = _patch_capture(monkeypatch)

    await ai_interaction.do_generate_image("a red fox\nsdxl-local\n832x1216\nhigh")

    payload = captured["payload"]
    assert payload["prompt"] == "a red fox"
    assert payload["model"] == "sdxl-local"
    assert payload["size"] == "832x1216"
    assert payload["quality"] == "high"


def test_direct_image_chat_passes_prompt_as_json_at_full_size():
    src = open(os.path.join(ROOT, "routes", "chat_routes.py"), encoding="utf-8").read()
    assert "512x512" not in src
    assert '"prompt": _user_msg' in src


def test_local_image_size_accepts_model_native_buckets():
    from mcp_servers.image_gen_server import _local_image_size

    assert _local_image_size("1216x832") == "1216x832"
    assert _local_image_size("832x1216") == "832x1216"
    assert _local_image_size("1024x1024") == "1024x1024"
    for bad in ("", "auto", "big", "1024", "100x100", "4096x4096", "1024x1024x3"):
        assert _local_image_size(bad) == "1024x1024", bad


def test_generate_image_has_a_native_schema():
    src = open(os.path.join(ROOT, "src", "tool_schemas.py"), encoding="utf-8").read()
    tree = ast.parse(src)
    value = next(
        node.value for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "FUNCTION_TOOL_SCHEMAS" for t in node.targets)
    )
    schemas = {s["function"]["name"]: s["function"] for s in ast.literal_eval(value)}

    assert "generate_image" in schemas
    params = schemas["generate_image"]["parameters"]
    assert params["required"] == ["prompt"]
    assert {"prompt", "size", "quality"} <= set(params["properties"])
