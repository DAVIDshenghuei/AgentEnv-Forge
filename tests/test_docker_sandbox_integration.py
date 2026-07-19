import os

import pytest

from agentenv_forge.sandbox import (
    BoundedProcessExecutor,
    DockerSandbox,
    DockerSandboxError,
)


@pytest.mark.skipif(
    os.environ.get("AGENTENV_FORGE_DOCKER_INTEGRATION") != "1",
    reason="requires the explicitly built M2 sandbox image and Docker daemon",
)
def test_real_docker_sandbox_is_persistent_restricted_and_cleaned_up(tmp_path) -> None:
    executor = BoundedProcessExecutor(max_output_bytes=65_536)
    container_name = "agentenv-forge-m2-integration"
    sandbox = DockerSandbox(
        workspace=tmp_path,
        image="agentenv-forge-sandbox:test",
        command_executor=executor,
        container_name=container_name,
        command_timeout_seconds=5.0,
    )

    with sandbox:
        identity = sandbox.execute(
            ("python", "-c", "import os; print(os.getuid())")
        )
        assert identity.exit_code == 0
        assert identity.stdout.strip() == "10001"

        write = sandbox.execute(
            (
                "python",
                "-c",
                "from pathlib import Path; Path('result.txt').write_text('persistent')",
            )
        )
        assert write.exit_code == 0
        read = sandbox.execute(
            (
                "python",
                "-c",
                "from pathlib import Path; print(Path('result.txt').read_text())",
            )
        )
        assert read.exit_code == 0
        assert read.stdout.strip() == "persistent"

        rootfs_write = sandbox.execute(
            (
                "python",
                "-c",
                "from pathlib import Path; Path('/forbidden').write_text('x')",
            )
        )
        assert rootfs_write.exit_code != 0

        network = sandbox.execute(
            (
                "python",
                "-c",
                "import socket; "
                "s=socket.socket(); s.settimeout(0.5); "
                "\ntry: s.connect(('1.1.1.1', 53))\n"
                "except OSError: raise SystemExit(0)\n"
                "raise SystemExit(1)",
            )
        )
        assert network.exit_code == 0

    inspection = executor(("docker", "inspect", container_name), 5.0)
    assert inspection.exit_code != 0
    assert (tmp_path / "result.txt").read_text(encoding="utf-8") == "persistent"


@pytest.mark.skipif(
    os.environ.get("AGENTENV_FORGE_DOCKER_INTEGRATION") != "1",
    reason="requires the explicitly built M2 sandbox image and Docker daemon",
)
def test_real_docker_timeout_force_removes_the_episode_container(tmp_path) -> None:
    executor = BoundedProcessExecutor(max_output_bytes=65_536)
    container_name = "agentenv-forge-m2-timeout-integration"
    sandbox = DockerSandbox(
        workspace=tmp_path,
        image="agentenv-forge-sandbox:test",
        command_executor=executor,
        container_name=container_name,
        command_timeout_seconds=1.0,
    )
    sandbox.start()

    with pytest.raises(DockerSandboxError, match="^sandbox command failed$"):
        sandbox.execute(("python", "-c", "import time; time.sleep(10)"))

    inspection = executor(("docker", "inspect", container_name), 5.0)
    assert inspection.exit_code != 0
    sandbox.close()
