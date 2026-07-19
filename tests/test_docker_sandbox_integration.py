import os

import pytest

import agentenv_forge.sandbox.process as process_module
from agentenv_forge.runner import _prepare_docker_workspace
from agentenv_forge.sandbox import (
    BoundedProcessExecutor,
    DockerSandbox,
    DockerSandboxError,
)


_INTEGRATION_CONTAINER_NAMES = (
    "agentenv-forge-m2-integration",
    "agentenv-forge-m2-timeout-integration",
    "agentenv-forge-m2-interrupted-start",
    "agentenv-forge-m2-interrupted-exec",
    "agentenv-forge-m2-workspace-quota",
    "agentenv-forge-m2-inode-quota",
)


@pytest.fixture(autouse=True)
def _remove_exact_integration_containers():
    yield
    if os.environ.get("AGENTENV_FORGE_DOCKER_INTEGRATION") != "1":
        return
    executor = BoundedProcessExecutor(max_output_bytes=65_536)
    failures = []
    for name in _INTEGRATION_CONTAINER_NAMES:
        result = executor(("docker", "rm", "--force", name), 5.0)
        if result.exit_code not in {0, 1}:
            failures.append(name)
    assert failures == []


@pytest.mark.skipif(
    os.environ.get("AGENTENV_FORGE_DOCKER_INTEGRATION") != "1",
    reason="requires the explicitly built M2 sandbox image and Docker daemon",
)
def test_real_docker_sandbox_is_persistent_restricted_and_cleaned_up(tmp_path) -> None:
    workspace = tmp_path / "forge,readonly"
    workspace.mkdir()
    executor = BoundedProcessExecutor(max_output_bytes=65_536)
    container_name = "agentenv-forge-m2-integration"
    container_user = _prepare_docker_workspace(workspace)
    sandbox = DockerSandbox(
        workspace=workspace,
        image="agentenv-forge-sandbox:test",
        command_executor=executor,
        container_name=container_name,
        command_timeout_seconds=5.0,
        container_user=container_user,
    )

    with sandbox:
        identity = sandbox.execute(
            ("python", "-c", "import os; print(f'{os.getuid()}:{os.getgid()}')")
        )
        assert identity.exit_code == 0
        assert identity.stdout.strip() == container_user

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

        host_write = sandbox.execute(
            (
                "python",
                "-c",
                "from pathlib import Path; Path('/host-workspace/escape').write_text('x')",
            )
        )
        assert host_write.exit_code != 0
        assert not (workspace / "escape").exists()

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
    assert (workspace / "result.txt").read_text(encoding="utf-8") == "persistent"


@pytest.mark.skipif(
    os.environ.get("AGENTENV_FORGE_DOCKER_INTEGRATION") != "1",
    reason="requires the explicitly built M2 sandbox image and Docker daemon",
)
def test_real_docker_timeout_force_removes_the_episode_container(tmp_path) -> None:
    executor = BoundedProcessExecutor(max_output_bytes=65_536)
    container_name = "agentenv-forge-m2-timeout-integration"
    container_user = _prepare_docker_workspace(tmp_path)
    sandbox = DockerSandbox(
        workspace=tmp_path,
        image="agentenv-forge-sandbox:test",
        command_executor=executor,
        container_name=container_name,
        command_timeout_seconds=1.0,
        container_user=container_user,
    )
    sandbox.start()

    with pytest.raises(DockerSandboxError, match="^sandbox command failed$"):
        sandbox.execute(("python", "-c", "import time; time.sleep(10)"))

    inspection = executor(("docker", "inspect", container_name), 5.0)
    assert inspection.exit_code != 0
    sandbox.close()


@pytest.mark.skipif(
    os.environ.get("AGENTENV_FORGE_DOCKER_INTEGRATION") != "1",
    reason="requires the explicitly built M2 sandbox image and Docker daemon",
)
def test_interrupted_real_docker_start_reaps_client_and_removes_exact_container(
    tmp_path, monkeypatch
) -> None:
    real_popen = process_module.subprocess.Popen
    container_name = "agentenv-forge-m2-interrupted-start"
    interrupted = False

    class InterruptingDockerRun:
        def __init__(self, process) -> None:
            self._process = process
            self.stdout = process.stdout
            self.stderr = process.stderr

        def kill(self) -> None:
            self._process.kill()

        def wait(self, timeout):
            nonlocal interrupted
            if not interrupted:
                interrupted = True
                try:
                    self._process.wait(timeout=0.05)
                except process_module.subprocess.TimeoutExpired:
                    pass
                raise KeyboardInterrupt("cancel real docker start")
            return self._process.wait(timeout=timeout)

    def popen(argv, *args, **kwargs):
        process = real_popen(argv, *args, **kwargs)
        if tuple(argv[:2]) == ("docker", "run"):
            return InterruptingDockerRun(process)
        return process

    monkeypatch.setattr(process_module.subprocess, "Popen", popen)
    executor = BoundedProcessExecutor(max_output_bytes=65_536)
    executor(("docker", "rm", "--force", container_name), 5.0)
    sandbox = DockerSandbox(
        workspace=tmp_path,
        image="agentenv-forge-sandbox:test",
        command_executor=executor,
        container_name=container_name,
        command_timeout_seconds=5.0,
        container_user=_prepare_docker_workspace(tmp_path),
    )

    with pytest.raises(KeyboardInterrupt, match="^cancel real docker start$"):
        sandbox.start()
    sandbox.close()

    inspection = executor(("docker", "inspect", container_name), 5.0)
    assert inspection.exit_code != 0


@pytest.mark.skipif(
    os.environ.get("AGENTENV_FORGE_DOCKER_INTEGRATION") != "1",
    reason="requires the explicitly built M2 sandbox image and Docker daemon",
)
def test_interrupted_real_docker_exec_removes_exact_container(
    tmp_path, monkeypatch
) -> None:
    executor = BoundedProcessExecutor(max_output_bytes=65_536)
    container_name = "agentenv-forge-m2-interrupted-exec"
    sandbox = DockerSandbox(
        workspace=tmp_path,
        image="agentenv-forge-sandbox:test",
        command_executor=executor,
        container_name=container_name,
        command_timeout_seconds=5.0,
        container_user=_prepare_docker_workspace(tmp_path),
    )
    sandbox.start()
    real_popen = process_module.subprocess.Popen
    interrupted = False

    class InterruptingDockerExec:
        def __init__(self, process) -> None:
            self._process = process
            self.stdout = process.stdout
            self.stderr = process.stderr

        def kill(self) -> None:
            self._process.kill()

        def wait(self, timeout):
            nonlocal interrupted
            if not interrupted:
                interrupted = True
                try:
                    self._process.wait(timeout=0.05)
                except process_module.subprocess.TimeoutExpired:
                    pass
                raise KeyboardInterrupt("cancel real docker exec")
            return self._process.wait(timeout=timeout)

    def popen(argv, *args, **kwargs):
        process = real_popen(argv, *args, **kwargs)
        if tuple(argv[:2]) == ("docker", "exec"):
            return InterruptingDockerExec(process)
        return process

    monkeypatch.setattr(process_module.subprocess, "Popen", popen)

    with pytest.raises(KeyboardInterrupt, match="^cancel real docker exec$"):
        sandbox.execute(("python", "-c", "import time; time.sleep(30)"))
    sandbox.close()

    inspection = executor(("docker", "inspect", container_name), 5.0)
    assert inspection.exit_code != 0


@pytest.mark.skipif(
    os.environ.get("AGENTENV_FORGE_DOCKER_INTEGRATION") != "1",
    reason="requires the explicitly built M2 sandbox image and Docker daemon",
)
def test_real_workspace_quota_prevents_large_host_write(tmp_path) -> None:
    executor = BoundedProcessExecutor(max_output_bytes=65_536)
    sandbox = DockerSandbox(
        workspace=tmp_path,
        image="agentenv-forge-sandbox:test",
        command_executor=executor,
        container_name="agentenv-forge-m2-workspace-quota",
        command_timeout_seconds=5.0,
        container_user=_prepare_docker_workspace(tmp_path),
    )
    sandbox.start()
    try:
        try:
            result = sandbox.execute(
                (
                    "python",
                    "-c",
                    "from pathlib import Path; Path('fill.bin').write_bytes(b'x' * 8_388_608)",
                )
            )
        except DockerSandboxError:
            pass
        else:
            assert result.exit_code != 0
        entries = tuple(tmp_path.rglob("*"))
        files = tuple(path for path in entries if path.is_file())
        assert len(entries) <= 128
        assert all(path.stat().st_size <= 1_048_576 for path in files)
        assert sum(path.stat().st_size for path in files) <= 4_194_304
    finally:
        try:
            sandbox.close()
        except DockerSandboxError:
            pass


@pytest.mark.skipif(
    os.environ.get("AGENTENV_FORGE_DOCKER_INTEGRATION") != "1",
    reason="requires the explicitly built M2 sandbox image and Docker daemon",
)
def test_real_workspace_inode_quota_prevents_host_file_flood(tmp_path) -> None:
    executor = BoundedProcessExecutor(max_output_bytes=65_536)
    sandbox = DockerSandbox(
        workspace=tmp_path,
        image="agentenv-forge-sandbox:test",
        command_executor=executor,
        container_name="agentenv-forge-m2-inode-quota",
        command_timeout_seconds=5.0,
        container_user=_prepare_docker_workspace(tmp_path),
    )
    sandbox.start()
    try:
        try:
            result = sandbox.execute(
                (
                    "python",
                    "-c",
                    "from pathlib import Path; "
                    "[(Path(f'f-{index}').touch()) for index in range(1000)]",
                )
            )
        except DockerSandboxError:
            pass
        else:
            assert result.exit_code != 0
        assert len(tuple(tmp_path.rglob("*"))) <= 128
    finally:
        try:
            sandbox.close()
        except DockerSandboxError:
            pass
