from pathlib import Path

import pytest

from agentenv_forge.sandbox import (
    DockerCommandResult,
    DockerSandbox,
    DockerSandboxError,
)
from agentenv_forge.tools import TerminalResult


class RecordingDockerExecutor:
    def __init__(self, results: list[DockerCommandResult]) -> None:
        self.results = list(results)
        self.calls: list[tuple[tuple[str, ...], float]] = []

    def __call__(
        self, argv: tuple[str, ...], timeout_seconds: float
    ) -> DockerCommandResult:
        self.calls.append((argv, timeout_seconds))
        if not self.results:
            raise AssertionError("unexpected Docker command")
        return self.results.pop(0)


def _sandbox(
    tmp_path: Path,
    executor: RecordingDockerExecutor,
) -> DockerSandbox:
    return DockerSandbox(
        workspace=tmp_path,
        image="agentenv-forge-sandbox:test",
        command_executor=executor,
        container_name="agentenv-forge-episode-test",
        command_timeout_seconds=2.5,
    )


def _expected_run_command(tmp_path: Path) -> tuple[str, ...]:
    return (
        "docker",
        "run",
        "--detach",
        "--name",
        "agentenv-forge-episode-test",
        "--network",
        "none",
        "--read-only",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--pids-limit",
        "64",
        "--memory",
        "256m",
        "--cpus",
        "1",
        "--tmpfs",
        "/tmp:rw,noexec,nosuid,size=16m",
        "--mount",
        f"type=bind,src={tmp_path.resolve()},dst=/workspace",
        "--workdir",
        "/workspace",
        "--user",
        "10001:10001",
        "agentenv-forge-sandbox:test",
        "sleep",
        "infinity",
    )


def test_docker_sandbox_uses_hardened_exact_argv_and_persists_across_execs(
    tmp_path,
) -> None:
    executor = RecordingDockerExecutor(
        [
            DockerCommandResult(0, "container-id\n", ""),
            DockerCommandResult(0, "one\n", ""),
            DockerCommandResult(7, "", "two\n"),
            DockerCommandResult(0, "", ""),
        ]
    )
    sandbox = _sandbox(tmp_path, executor)

    sandbox.start()
    first = sandbox.execute(("python", "-c", "print('one')"))
    second = sandbox.execute(("python", "-c", "raise SystemExit(7)"))
    sandbox.close()

    assert first == TerminalResult(0, "one\n", "")
    assert second == TerminalResult(7, "", "two\n")
    assert executor.calls == [
        (_expected_run_command(tmp_path), 2.5),
        (
            (
                "docker",
                "exec",
                "--workdir",
                "/workspace",
                "agentenv-forge-episode-test",
                "python",
                "-c",
                "print('one')",
            ),
            2.5,
        ),
        (
            (
                "docker",
                "exec",
                "--workdir",
                "/workspace",
                "agentenv-forge-episode-test",
                "python",
                "-c",
                "raise SystemExit(7)",
            ),
            2.5,
        ),
        (
            (
                "docker",
                "rm",
                "--force",
                "agentenv-forge-episode-test",
            ),
            2.5,
        ),
    ]


def test_execute_before_start_fails_without_docker_command(tmp_path) -> None:
    executor = RecordingDockerExecutor([])
    sandbox = _sandbox(tmp_path, executor)

    with pytest.raises(DockerSandboxError, match="^sandbox is not running$"):
        sandbox.execute(("python", "-V"))

    assert executor.calls == []


def test_start_failure_is_sanitized_and_cleans_exact_container_name(tmp_path) -> None:
    secret = "SECRET DOCKER START"
    executor = RecordingDockerExecutor(
        [
            DockerCommandResult(125, "", f"{tmp_path} {secret}"),
            DockerCommandResult(0, "", ""),
        ]
    )
    sandbox = _sandbox(tmp_path, executor)

    with pytest.raises(DockerSandboxError, match="^sandbox startup failed$") as failure:
        sandbox.start()

    message = str(failure.value)
    assert str(tmp_path) not in message
    assert secret not in message
    assert executor.calls == [
        (_expected_run_command(tmp_path), 2.5),
        (
            (
                "docker",
                "rm",
                "--force",
                "agentenv-forge-episode-test",
            ),
            2.5,
        ),
    ]


def test_exec_failure_is_sanitized_but_nonzero_exit_is_not_failure(tmp_path) -> None:
    secret = "SECRET DOCKER EXEC"

    class RaisingExecutor(RecordingDockerExecutor):
        def __call__(self, argv, timeout_seconds):
            self.calls.append((argv, timeout_seconds))
            if argv[1] == "run":
                return DockerCommandResult(0, "container-id\n", "")
            if argv[1] == "exec":
                raise RuntimeError(f"{tmp_path} {secret}")
            return DockerCommandResult(0, "", "")

    executor = RaisingExecutor([])
    sandbox = _sandbox(tmp_path, executor)
    sandbox.start()

    with pytest.raises(DockerSandboxError, match="^sandbox command failed$") as failure:
        sandbox.execute(("python", "-V"))

    with pytest.raises(DockerSandboxError, match="^sandbox is not running$"):
        sandbox.execute(("python", "-V"))
    sandbox.close()

    message = str(failure.value)
    assert str(tmp_path) not in message
    assert secret not in message
    assert executor.calls[-1][0] == (
        "docker",
        "rm",
        "--force",
        "agentenv-forge-episode-test",
    )


def test_close_is_idempotent_and_close_failure_is_sanitized(tmp_path) -> None:
    executor = RecordingDockerExecutor(
        [
            DockerCommandResult(0, "container-id\n", ""),
            DockerCommandResult(1, "", "sensitive cleanup detail"),
        ]
    )
    sandbox = _sandbox(tmp_path, executor)
    sandbox.start()

    with pytest.raises(DockerSandboxError, match="^sandbox cleanup failed$"):
        sandbox.close()
    sandbox.close()

    assert len(executor.calls) == 2


def test_context_manager_cleans_sandbox_when_body_raises_base_exception(
    tmp_path,
) -> None:
    executor = RecordingDockerExecutor(
        [
            DockerCommandResult(0, "container-id\n", ""),
            DockerCommandResult(0, "", ""),
        ]
    )
    sandbox = _sandbox(tmp_path, executor)

    with pytest.raises(KeyboardInterrupt):
        with sandbox:
            raise KeyboardInterrupt

    assert executor.calls[-1][0] == (
        "docker",
        "rm",
        "--force",
        "agentenv-forge-episode-test",
    )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("image", ""),
        ("image", "image\x00name"),
        ("container_name", "unsafe name"),
        ("container_name", "--option"),
        ("command_timeout_seconds", True),
        ("command_timeout_seconds", 0.0),
    ),
)
def test_constructor_rejects_invalid_trusted_configuration_without_side_effects(
    tmp_path, field, value
) -> None:
    executor = RecordingDockerExecutor([])
    kwargs = {
        "workspace": tmp_path,
        "image": "agentenv-forge-sandbox:test",
        "command_executor": executor,
        "container_name": "agentenv-forge-episode-test",
        "command_timeout_seconds": 2.5,
    }
    kwargs[field] = value

    with pytest.raises(ValueError, match="^invalid sandbox configuration$"):
        DockerSandbox(**kwargs)

    assert executor.calls == []
