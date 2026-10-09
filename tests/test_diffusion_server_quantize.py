"""diffusion_server.py --quantize: 4-bit loading for small GPUs.

Like test_diffusion_server_security.py, the helper is compiled out of the
script via AST so torch / diffusers need not be installed.
"""
import ast
import sys
import types
from pathlib import Path
from types import SimpleNamespace

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "diffusion_server.py"
_SOURCE = _SCRIPT.read_text(encoding="utf-8")


def _quantization_kwargs(args):
    tree = ast.parse(_SOURCE)
    [node] = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_quantization_kwargs"]
    module = ast.Module(body=[node], type_ignores=[])
    ns = {"_args": args}
    exec(compile(module, str(_SCRIPT), "exec"), ns)
    return ns["_quantization_kwargs"]


class _FakeConfig:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def test_quantize_off_adds_nothing():
    fn = _quantization_kwargs(SimpleNamespace(quantize="none", quantize_components="transformer"))
    assert fn("bf16") == {}


def test_nf4_quantizes_the_requested_components(monkeypatch):
    fake = types.ModuleType("diffusers")
    fake.PipelineQuantizationConfig = _FakeConfig
    monkeypatch.setitem(sys.modules, "diffusers", fake)
    fn = _quantization_kwargs(SimpleNamespace(quantize="nf4", quantize_components="transformer, text_encoder,"))

    config = fn("bf16")["quantization_config"]

    assert config.kwargs["quant_backend"] == "bitsandbytes_4bit"
    assert config.kwargs["quant_kwargs"] == {
        "load_in_4bit": True, "bnb_4bit_quant_type": "nf4", "bnb_4bit_compute_dtype": "bf16",
    }
    assert config.kwargs["components_to_quantize"] == ["transformer", "text_encoder"]


def test_every_pretrained_load_receives_the_quantization_kwargs():
    load_model = _SOURCE.split("def load_model():", 1)[1].split("\ndef ", 1)[0]
    assert load_model.count('kwargs = {"torch_dtype": torch_dtype, **quant_kwargs}') == 3
    assert "**quant_kwargs," in load_model  # the custom_pipeline retry
    assert '"--quantize", default="none", choices=["none", "nf4"]' in _SOURCE
