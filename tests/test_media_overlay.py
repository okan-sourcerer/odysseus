"""docker/media.nvidia.yml runs scripts/diffusion_server.py as the `media` service."""
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
OVERLAY = yaml.safe_load((ROOT / "docker" / "media.nvidia.yml").read_text(encoding="utf-8"))
SERVER = (ROOT / "scripts" / "diffusion_server.py").read_text(encoding="utf-8")
DOCKERFILE = (ROOT / "docker" / "media" / "Dockerfile").read_text(encoding="utf-8")


def test_every_flag_the_overlay_passes_exists_on_the_server():
    command = OVERLAY["services"]["media"]["command"]
    flags = {re.match(r"(--[a-z-]+)", arg).group(1) for arg in command}
    missing = sorted(flag for flag in flags if f'"{flag}"' not in SERVER)
    assert not missing, f"overlay passes flags diffusion_server.py does not define: {missing}"


def test_media_service_shares_the_gpu_on_demand():
    media = OVERLAY["services"]["media"]
    command = " ".join(media["command"])
    assert "--lazy-load" in command and "--idle-unload=" in command and "--unload-ollama=" in command
    assert "--allowed-host=media" in command  # Odysseus calls it by service name
    [device] = media["deploy"]["resources"]["reservations"]["devices"]
    assert device["driver"] == "nvidia" and "gpu" in device["capabilities"]
    assert "ports" not in media  # reachable on the compose network only


def test_dockerfile_runs_the_repo_server_script():
    assert "COPY --chown=media scripts/diffusion_server.py /app/diffusion_server.py" in DOCKERFILE
    assert 'ENTRYPOINT ["python", "/app/diffusion_server.py"]' in DOCKERFILE
