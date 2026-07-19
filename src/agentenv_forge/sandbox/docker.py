import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from ..tools import TerminalResult

_IMAGE_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/@:-]{0,255}$")
_CONTAINER_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,62}$")
_MAX_OUTPUT_BYTES = 65_536


class DockerSandboxError(RuntimeError):
    """A sanitized trusted Docker sandbox lifecycle failure."""


@dataclass(frozen=True, slots=True)
class DockerCommandResult:
    exit_code: int
    stdout: str
    stderr: str


DockerCommandExecutor = Callable[[tuple[str, ...], float], DockerCommandResult]


class DockerSandbox:
    """Persistent, hardened per-episode container managed by trusted host code."""

    __slots__ = (
        "_command_executor",
        "_container_name",
        "_image",
        "_running",
        "_timeout_seconds",
        "_workspace",
    )

    def __init__(
        self,
        workspace: Path,
        image: str,
        command_executor: DockerCommandExecutor,
        container_name: str,
        command_timeout_seconds: float,
    ) -> None:
        if (
            not isinstance(workspace, Path)
            or type(image) is not str
            or _IMAGE_PATTERN.fullmatch(image) is None
            or type(container_name) is not str
            or _CONTAINER_NAME_PATTERN.fullmatch(container_name) is None
            or type(command_timeout_seconds) not in {int, float}
            or not 0 < command_timeout_seconds <= 60
            or not callable(command_executor)
        ):
            raise ValueError("invalid sandbox configuration")
        try:
            resolved_workspace = workspace.resolve(strict=True)
            if not resolved_workspace.is_dir():
                raise ValueError("invalid sandbox configuration")
        except OSError:
            raise ValueError("invalid sandbox configuration") from None
        self._workspace = resolved_workspace
        self._image = image
        self._command_executor = command_executor
        self._container_name = container_name
        self._timeout_seconds = float(command_timeout_seconds)
        self._running = False

    def _remove_command(self) -> tuple[str, ...]:
        return ("docker", "rm", "--force", self._container_name)

    def _execute_docker(self, argv: tuple[str, ...]) -> DockerCommandResult:
        try:
            result = self._command_executor(argv, self._timeout_seconds)
        except Exception:
            raise DockerSandboxError("sandbox command failed") from None
        if (
            type(result) is not DockerCommandResult
            or type(result.exit_code) is not int
            or not 0 <= result.exit_code <= 255
            or type(result.stdout) is not str
            or type(result.stderr) is not str
        ):
            raise DockerSandboxError("sandbox command failed")
        try:
            stdout_size = len(result.stdout.encode("utf-8"))
            stderr_size = len(result.stderr.encode("utf-8"))
        except UnicodeError:
            raise DockerSandboxError("sandbox command failed") from None
        if stdout_size > _MAX_OUTPUT_BYTES or stderr_size > _MAX_OUTPUT_BYTES:
            raise DockerSandboxError("sandbox command failed")
        return result

    def _cleanup_after_failed_start(self) -> None:
        try:
            self._execute_docker(self._remove_command())
        except DockerSandboxError:
            pass

    def start(self) -> None:
        if self._running:
            raise DockerSandboxError("sandbox is already running")
        run_command = (
            "docker",
            "run",
            "--detach",
            "--name",
            self._container_name,
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
            f"type=bind,src={self._workspace},dst=/workspace",
            "--workdir",
            "/workspace",
            "--user",
            "10001:10001",
            self._image,
            "sleep",
            "infinity",
        )
        try:
            result = self._execute_docker(run_command)
        except DockerSandboxError:
            self._cleanup_after_failed_start()
            raise DockerSandboxError("sandbox startup failed") from None
        if result.exit_code != 0:
            self._cleanup_after_failed_start()
            raise DockerSandboxError("sandbox startup failed")
        self._running = True

    @staticmethod
    def _validate_exec_argv(argv: tuple[str, ...]) -> None:
        if type(argv) is not tuple or not argv:
            raise DockerSandboxError("sandbox command failed")
        for argument in argv:
            if type(argument) is not str or not argument or "\x00" in argument:
                raise DockerSandboxError("sandbox command failed")

    def _abort_running_sandbox(self) -> None:
        if not self._running:
            return
        self._running = False
        try:
            self._execute_docker(self._remove_command())
        except DockerSandboxError:
            pass

    def execute(self, argv: tuple[str, ...]) -> TerminalResult:
        if not self._running:
            raise DockerSandboxError("sandbox is not running")
        self._validate_exec_argv(argv)
        try:
            result = self._execute_docker(
                (
                    "docker",
                    "exec",
                    "--workdir",
                    "/workspace",
                    self._container_name,
                    *argv,
                )
            )
        except DockerSandboxError:
            self._abort_running_sandbox()
            raise DockerSandboxError("sandbox command failed") from None
        return TerminalResult(result.exit_code, result.stdout, result.stderr)

    def close(self) -> None:
        if not self._running:
            return
        self._running = False
        try:
            result = self._execute_docker(self._remove_command())
        except DockerSandboxError:
            raise DockerSandboxError("sandbox cleanup failed") from None
        if result.exit_code != 0:
            raise DockerSandboxError("sandbox cleanup failed")

    def __enter__(self) -> "DockerSandbox":
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        try:
            self.close()
        except DockerSandboxError:
            if exc_type is None:
                raise
        return False
