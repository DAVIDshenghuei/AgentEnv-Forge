import csv
from pathlib import Path

import pytest

import agentenv_forge.sandbox.docker as docker_module
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
        "--pull",
        "never",
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
        "--tmpfs",
        "/workspace:rw,noexec,nosuid,size=4194304,nr_inodes=256,uid=10001,gid=10001,mode=0700",
        "--mount",
        f"type=bind,src={tmp_path.resolve()},dst=/host-workspace,readonly",
        "--workdir",
        "/workspace",
        "--user",
        "10001:10001",
        "agentenv-forge-sandbox:test",
        "sleep",
        "infinity",
    )


def test_bind_mount_source_is_csv_encoded_for_comma_bearing_workspace(tmp_path) -> None:
    workspace = tmp_path / "forge,readonly"
    workspace.mkdir()
    executor = RecordingDockerExecutor(
        [DockerCommandResult(0, "container-id\n", ""), DockerCommandResult(0, "", "")]
    )
    sandbox = _sandbox(workspace, executor)

    sandbox.start()
    sandbox.close()

    run_argv = executor.calls[0][0]
    mount_argument = run_argv[run_argv.index("--mount") + 1]
    assert next(csv.reader([mount_argument])) == [
        "type=bind",
        f"src={workspace.resolve()}",
        "dst=/host-workspace",
        "readonly",
    ]


def test_docker_sandbox_uses_hardened_exact_argv_and_persists_across_execs(
    tmp_path,
) -> None:
    executor = RecordingDockerExecutor(
        [
            DockerCommandResult(0, "container-id\n", ""),
            DockerCommandResult(0, "", ""),
            DockerCommandResult(0, "one\n", ""),
            DockerCommandResult(0, '{"directories":[],"files":[]}', ""),
            DockerCommandResult(0, "", ""),
            DockerCommandResult(7, "", "two\n"),
            DockerCommandResult(0, '{"directories":[],"files":[]}', ""),
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
    calls = [call[0] for call in executor.calls]
    assert calls[0] == _expected_run_command(tmp_path)
    for sync_call in (calls[1], calls[4]):
        assert sync_call[:9] == (
            "docker",
            "exec",
            "--user",
            "10001:10001",
            "--workdir",
            "/",
            "agentenv-forge-episode-test",
            "python",
            "-c",
        )
        assert 'source = "/host-workspace"' in sync_call[9]
        assert 'target = "/workspace"' in sync_call[9]
    assert calls[2] == (
        "docker",
        "exec",
        "--workdir",
        "/workspace",
        "agentenv-forge-episode-test",
        "python",
        "-c",
        "print('one')",
    )
    assert calls[5] == (
        "docker",
        "exec",
        "--workdir",
        "/workspace",
        "agentenv-forge-episode-test",
        "python",
        "-c",
        "raise SystemExit(7)",
    )
    for manifest_call in (calls[3], calls[6]):
        assert manifest_call[:9] == (
            "docker",
            "exec",
            "--user",
            "10001:10001",
            "--workdir",
            "/",
            "agentenv-forge-episode-test",
            "python",
            "-c",
        )
        assert 'root = "/workspace"' in manifest_call[9]
        assert manifest_call[10:] == ("128", "1048576", "4194304")
    assert calls[7] == (
        "docker",
        "rm",
        "--force",
        "agentenv-forge-episode-test",
    )
    assert all(executor.calls[index][1] == 2.5 for index in (0, 1, 2, 4, 5, 7))
    assert all(0 < executor.calls[index][1] <= 2.5 for index in (3, 6))


def test_valid_manifest_chunk_sync_rebuilds_host_workspace(tmp_path) -> None:
    (tmp_path / "before.txt").write_text("before", encoding="utf-8")
    manifest = (
        '{"directories":[],"files":['
        '{"path":"result.bin","size":3,'
        '"sha256":"ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"}'
        "]}"
    )
    executor = RecordingDockerExecutor(
        [
            DockerCommandResult(0, "container-id\n", ""),
            DockerCommandResult(0, "", ""),
            DockerCommandResult(0, "", ""),
            DockerCommandResult(0, manifest, ""),
            DockerCommandResult(0, "YWJj", ""),
            DockerCommandResult(0, "", ""),
        ]
    )
    sandbox = _sandbox(tmp_path, executor)

    sandbox.start()
    result = sandbox.execute(("python", "-V"))
    sandbox.close()

    assert result.exit_code == 0
    assert not (tmp_path / "before.txt").exists()
    assert (tmp_path / "result.bin").read_bytes() == b"abc"
    chunk_call = executor.calls[4][0]
    assert chunk_call[:9] == (
        "docker",
        "exec",
        "--user",
        "10001:10001",
        "--workdir",
        "/",
        "agentenv-forge-episode-test",
        "python",
        "-c",
    )
    assert "os.O_NOFOLLOW" in chunk_call[9]
    assert chunk_call[10:] == ("result.bin", "3", "0", "3")


def test_manifest_hash_mismatch_aborts_before_host_replacement(tmp_path) -> None:
    original = tmp_path / "before.txt"
    original.write_text("before", encoding="utf-8")
    manifest = (
        '{"directories":[],"files":['
        '{"path":"result.bin","size":3,'
        '"sha256":"0000000000000000000000000000000000000000000000000000000000000000"}'
        "]}"
    )
    executor = RecordingDockerExecutor(
        [
            DockerCommandResult(0, "container-id\n", ""),
            DockerCommandResult(0, "", ""),
            DockerCommandResult(0, "", ""),
            DockerCommandResult(0, manifest, ""),
            DockerCommandResult(0, "YWJj", ""),
            DockerCommandResult(0, "", ""),
        ]
    )
    sandbox = _sandbox(tmp_path, executor)
    sandbox.start()

    with pytest.raises(DockerSandboxError, match="^sandbox command failed$"):
        sandbox.execute(("python", "-V"))

    assert original.read_text(encoding="utf-8") == "before"
    assert executor.calls[-1][0] == (
        "docker",
        "rm",
        "--force",
        "agentenv-forge-episode-test",
    )


def test_manifest_and_chunks_share_one_deadline(tmp_path, monkeypatch) -> None:
    original = tmp_path / "before.txt"
    original.write_text("before", encoding="utf-8")
    manifest = (
        '{"directories":[],"files":['
        '{"path":"result.bin","size":3,'
        '"sha256":"ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"}'
        "]}"
    )
    executor = RecordingDockerExecutor(
        [
            DockerCommandResult(0, "container-id\n", ""),
            DockerCommandResult(0, "", ""),
            DockerCommandResult(0, "", ""),
            DockerCommandResult(0, manifest, ""),
            DockerCommandResult(0, "", ""),
        ]
    )
    times = iter((10.0, 10.0, 13.0))
    monkeypatch.setattr(docker_module, "monotonic", lambda: next(times))
    sandbox = _sandbox(tmp_path, executor)
    sandbox.start()

    with pytest.raises(DockerSandboxError, match="^sandbox command failed$"):
        sandbox.execute(("python", "-V"))

    assert original.read_text(encoding="utf-8") == "before"
    assert len(executor.calls) == 5
    assert executor.calls[3][1] == 2.5
    assert executor.calls[-1][0] == (
        "docker",
        "rm",
        "--force",
        "agentenv-forge-episode-test",
    )


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


def test_interrupted_exec_retries_cleanup_and_preserves_primary(tmp_path) -> None:
    calls: list[tuple[str, ...]] = []
    removal_attempts = 0

    def interrupting_executor(argv, timeout_seconds):
        nonlocal removal_attempts
        calls.append(argv)
        if argv[1] == "run":
            return DockerCommandResult(0, "container-id\n", "")
        if argv[1] == "exec":
            raise KeyboardInterrupt("primary exec cancellation")
        removal_attempts += 1
        if removal_attempts == 1:
            raise SystemExit("secondary cleanup interruption")
        return DockerCommandResult(0, "", "")

    sandbox = DockerSandbox(
        workspace=tmp_path,
        image="agentenv-forge-sandbox:test",
        command_executor=interrupting_executor,
        container_name="agentenv-forge-episode-test",
        command_timeout_seconds=2.5,
    )
    sandbox.start()

    with pytest.raises(KeyboardInterrupt, match="^primary exec cancellation$"):
        sandbox.execute(("python", "-V"))
    sandbox.close()

    assert [call[1] for call in calls] == ["run", "exec", "rm", "rm"]


def test_close_failure_remains_retryable_until_exact_container_is_removed(
    tmp_path,
) -> None:
    executor = RecordingDockerExecutor(
        [
            DockerCommandResult(0, "container-id\n", ""),
            DockerCommandResult(1, "", "sensitive cleanup detail"),
            DockerCommandResult(0, "", ""),
        ]
    )
    sandbox = _sandbox(tmp_path, executor)
    sandbox.start()

    with pytest.raises(DockerSandboxError, match="^sandbox cleanup failed$"):
        sandbox.close()
    sandbox.close()
    sandbox.close()

    assert [call[0][1] for call in executor.calls] == ["run", "rm", "rm"]


def test_interrupted_start_attempts_cleanup_and_preserves_base_exception(
    tmp_path,
) -> None:
    calls: list[tuple[str, ...]] = []

    def interrupting_executor(argv, timeout_seconds):
        calls.append(argv)
        if argv[1] == "run":
            raise KeyboardInterrupt("original cancellation")
        return DockerCommandResult(0, "", "")

    sandbox = DockerSandbox(
        workspace=tmp_path,
        image="agentenv-forge-sandbox:test",
        command_executor=interrupting_executor,
        container_name="agentenv-forge-episode-test",
        command_timeout_seconds=2.5,
    )

    with pytest.raises(KeyboardInterrupt, match="^original cancellation$"):
        sandbox.start()
    sandbox.close()

    assert [call[1] for call in calls] == ["run", "rm"]


def test_failed_start_cleanup_failure_remains_retryable(tmp_path) -> None:
    executor = RecordingDockerExecutor(
        [
            DockerCommandResult(125, "", "start failed"),
            DockerCommandResult(1, "", "first cleanup failed"),
            DockerCommandResult(0, "", ""),
        ]
    )
    sandbox = _sandbox(tmp_path, executor)

    with pytest.raises(DockerSandboxError, match="^sandbox startup failed$"):
        sandbox.start()
    sandbox.close()

    assert [call[0][1] for call in executor.calls] == ["run", "rm", "rm"]


def test_exec_abort_cleanup_failure_remains_retryable(tmp_path) -> None:
    calls: list[tuple[str, ...]] = []
    removal_attempts = 0

    def failing_executor(argv, timeout_seconds):
        nonlocal removal_attempts
        calls.append(argv)
        if argv[1] == "run":
            return DockerCommandResult(0, "container-id\n", "")
        if argv[1] == "exec":
            raise RuntimeError("exec failed")
        removal_attempts += 1
        return DockerCommandResult(0 if removal_attempts == 2 else 1, "", "")

    sandbox = DockerSandbox(
        workspace=tmp_path,
        image="agentenv-forge-sandbox:test",
        command_executor=failing_executor,
        container_name="agentenv-forge-episode-test",
        command_timeout_seconds=2.5,
    )
    sandbox.start()

    with pytest.raises(DockerSandboxError, match="^sandbox command failed$"):
        sandbox.execute(("python", "-V"))
    sandbox.close()

    assert [call[1] for call in calls] == ["run", "exec", "rm", "rm"]


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


def test_context_manager_retries_transient_cleanup_failure(tmp_path) -> None:
    executor = RecordingDockerExecutor(
        [
            DockerCommandResult(0, "container-id\n", ""),
            DockerCommandResult(1, "", "temporary failure"),
            DockerCommandResult(0, "", ""),
        ]
    )

    with _sandbox(tmp_path, executor):
        pass

    assert [call[0][1] for call in executor.calls] == ["run", "rm", "rm"]


def test_context_manager_retries_cleanup_base_exception_before_reraising(
    tmp_path,
) -> None:
    calls: list[tuple[str, ...]] = []
    removal_attempts = 0

    def interrupting_executor(argv, timeout_seconds):
        nonlocal removal_attempts
        calls.append(argv)
        if argv[1] == "run":
            return DockerCommandResult(0, "container-id\n", "")
        removal_attempts += 1
        if removal_attempts == 1:
            raise KeyboardInterrupt("cleanup cancellation")
        return DockerCommandResult(0, "", "")

    sandbox = DockerSandbox(
        workspace=tmp_path,
        image="agentenv-forge-sandbox:test",
        command_executor=interrupting_executor,
        container_name="agentenv-forge-episode-test",
        command_timeout_seconds=2.5,
    )

    with pytest.raises(KeyboardInterrupt, match="^cleanup cancellation$"):
        with sandbox:
            pass

    assert [call[1] for call in calls] == ["run", "rm", "rm"]


def test_context_manager_preserves_primary_while_retrying_cleanup(tmp_path) -> None:
    executor = RecordingDockerExecutor(
        [
            DockerCommandResult(0, "container-id\n", ""),
            DockerCommandResult(1, "", "temporary failure"),
            DockerCommandResult(0, "", ""),
        ]
    )

    with pytest.raises(KeyboardInterrupt, match="^primary$"):
        with _sandbox(tmp_path, executor):
            raise KeyboardInterrupt("primary")

    assert [call[0][1] for call in executor.calls] == ["run", "rm", "rm"]


def test_docker_sandbox_accepts_valid_custom_non_root_identity(tmp_path) -> None:
    executor = RecordingDockerExecutor(
        [
            DockerCommandResult(0, "container-id\n", ""),
            DockerCommandResult(0, "", ""),
        ]
    )
    sandbox = DockerSandbox(
        workspace=tmp_path,
        image="agentenv-forge-sandbox:test",
        command_executor=executor,
        container_name="agentenv-forge-episode-test",
        command_timeout_seconds=2.5,
        container_user="1234:5678",
    )

    sandbox.start()
    sandbox.close()

    run_argv = executor.calls[0][0]
    assert run_argv[run_argv.index("--user") + 1] == "1234:5678"


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("image", ""),
        ("image", "image\x00name"),
        ("container_name", "unsafe name"),
        ("container_name", "--option"),
        ("container_user", "0:0"),
        ("container_user", "10001:0"),
        ("container_user", "10001:group"),
        ("container_user", True),
        ("container_user", "2147483648:10001"),
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
