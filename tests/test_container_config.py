import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
PYTHON_BASE_DIGEST = "sha256:b18992999dbe963a45a8a4da40ac2b1975be1a776d939d098c647482bcad5cba"
UV_DIGEST = "sha256:eb2843a1e56fd9e30c7276ce1a52cba86e64c7b385f5e3279a0e08e02dd058fc"


def test_docker_build_and_runtime_are_hardened():
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    compose = (ROOT / "compose.yaml").read_text(encoding="utf-8")
    dockerignore = (ROOT / ".dockerignore").read_text(encoding="utf-8")

    assert f"@{UV_DIGEST}" in dockerfile
    assert f"@{PYTHON_BASE_DIGEST}" in dockerfile
    assert "uv 0.11.29" in dockerfile
    assert "USER forge" in dockerfile
    assert "network_mode: none" in compose
    assert "cap_drop:" in compose and "- ALL" in compose
    assert "read_only: true" in compose
    assert "/tmp:size=16m" in compose
    assert "no-new-privileges:true" in compose
    assert "volumes:" not in compose
    for excluded in (".git", ".venv", ".env*", "outputs", "**/__pycache__", ".pytest_cache"):
        assert excluded in dockerignore


@pytest.mark.skipif(shutil.which("docker") is None, reason="Docker CLI unavailable")
def test_effective_compose_config_is_hardened():
    result = subprocess.run(
        ["docker", "compose", "config", "--format", "json"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    service = json.loads(result.stdout)["services"]["smoke"]

    assert service["network_mode"] == "none"
    assert service["read_only"] is True
    assert service["cap_drop"] == ["ALL"]
    assert service["security_opt"] == ["no-new-privileges:true"]
    assert service["tmpfs"] == ["/tmp:size=16m"]
    assert not service.get("volumes")


@pytest.mark.skipif(
    os.environ.get("AGENTENV_DOCKER_TESTS") != "1",
    reason="set AGENTENV_DOCKER_TESTS=1 for the real container gate",
)
def test_real_container_runtime_is_hardened():
    probe = (
        'test "$(id -u)" = 10001 && '
        'test "$(awk "/^CapEff:/ {print \\$2}" /proc/self/status)" = 0000000000000000 && '
        'test "$(awk "/^NoNewPrivs:/ {print \\$2}" /proc/self/status)" = 1 && '
        'if touch /root/forbidden 2>/dev/null; then exit 1; fi && '
        'touch /tmp/allowed'
    )
    subprocess.run(
        [
            "docker",
            "compose",
            "run",
            "--rm",
            "--build",
            "--entrypoint",
            "sh",
            "smoke",
            "-c",
            probe,
        ],
        cwd=ROOT,
        check=True,
    )
