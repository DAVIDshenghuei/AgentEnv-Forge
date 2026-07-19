import base64
import csv
import hashlib
import json
import re
import shutil
from dataclasses import dataclass
from io import StringIO
from pathlib import Path, PurePosixPath
from time import monotonic, sleep
from typing import Callable

from ..schemas import (
    MAX_DECLARED_ENTRIES,
    MAX_DECLARED_FILE_BYTES,
    MAX_DECLARED_TOTAL_BYTES,
    validate_relative_artifact_path,
)
from ..tools import TerminalResult

_IMAGE_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/@:-]{0,255}$")
_CONTAINER_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,62}$")
_CONTAINER_USER_PATTERN = re.compile(r"^[1-9][0-9]{0,9}:[1-9][0-9]{0,9}$")
_MAX_OUTPUT_BYTES = 65_536
_FAILED_START_CLEANUP_ATTEMPTS = 20
_FAILED_START_CLEANUP_RETRY_SECONDS = 0.05
_WORKSPACE_TMPFS_INODES = 256
_SYNC_IN_SCRIPT = """
import os
import shutil
import stat

source = "/host-workspace"
target = "/workspace"


def remove_entry(path):
    mode = os.lstat(path).st_mode
    if stat.S_ISDIR(mode):
        os.chmod(path, 0o700)
        for name in os.listdir(path):
            remove_entry(os.path.join(path, name))
        os.rmdir(path)
    else:
        os.unlink(path)


def copy_entry(source_path, target_path):
    mode = os.lstat(source_path).st_mode
    if stat.S_ISDIR(mode):
        os.mkdir(target_path, 0o700)
        for name in os.listdir(source_path):
            copy_entry(os.path.join(source_path, name), os.path.join(target_path, name))
    elif stat.S_ISREG(mode):
        shutil.copyfile(source_path, target_path)
    else:
        raise RuntimeError("unsupported workspace entry")


for entry in os.listdir(target):
    remove_entry(os.path.join(target, entry))
for entry in os.listdir(source):
    copy_entry(os.path.join(source, entry), os.path.join(target, entry))
"""
_SYNC_CHUNK_BYTES = 32_768
_MANIFEST_SCRIPT = """
import hashlib
import json
import os
import stat
import sys

root = "/workspace"
max_entries = int(sys.argv[1])
max_file_bytes = int(sys.argv[2])
max_total_bytes = int(sys.argv[3])
entries = 0
total_bytes = 0
directories = []
files = []

for current, directory_names, file_names in os.walk(root, topdown=True, followlinks=False):
    directory_names.sort()
    file_names.sort()
    for name in directory_names:
        path = os.path.join(current, name)
        mode = os.lstat(path).st_mode
        if not stat.S_ISDIR(mode) or stat.S_ISLNK(mode):
            raise RuntimeError("unsupported workspace entry")
        entries += 1
        if entries > max_entries:
            raise RuntimeError("workspace entry limit exceeded")
        directories.append(os.path.relpath(path, root).replace(os.sep, "/"))
    for name in file_names:
        path = os.path.join(current, name)
        mode = os.lstat(path).st_mode
        if not stat.S_ISREG(mode) or stat.S_ISLNK(mode):
            raise RuntimeError("unsupported workspace entry")
        entries += 1
        if entries > max_entries:
            raise RuntimeError("workspace entry limit exceeded")
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        try:
            file_stat = os.fstat(descriptor)
            if not stat.S_ISREG(file_stat.st_mode) or file_stat.st_size > max_file_bytes:
                raise RuntimeError("workspace file limit exceeded")
            digest = hashlib.sha256()
            observed = 0
            while True:
                block = os.read(descriptor, 65536)
                if not block:
                    break
                observed += len(block)
                digest.update(block)
            if observed != file_stat.st_size:
                raise RuntimeError("workspace changed during snapshot")
        finally:
            os.close(descriptor)
        total_bytes += observed
        if total_bytes > max_total_bytes:
            raise RuntimeError("workspace total limit exceeded")
        files.append({
            "path": os.path.relpath(path, root).replace(os.sep, "/"),
            "size": observed,
            "sha256": digest.hexdigest(),
        })

print(json.dumps({"directories": directories, "files": files}, separators=(",", ":")))
"""
_CHUNK_SCRIPT = """
import base64
import os
import stat
import sys

relative = sys.argv[1]
expected_size = int(sys.argv[2])
offset = int(sys.argv[3])
length = int(sys.argv[4])
parts = relative.split("/")
directory_fd = os.open("/workspace", os.O_RDONLY | os.O_DIRECTORY)
file_fd = None
try:
    for part in parts[:-1]:
        next_fd = os.open(
            part,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
            dir_fd=directory_fd,
        )
        os.close(directory_fd)
        directory_fd = next_fd
    file_fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory_fd)
    file_stat = os.fstat(file_fd)
    if not stat.S_ISREG(file_stat.st_mode) or file_stat.st_size != expected_size:
        raise RuntimeError("workspace changed during transfer")
    os.lseek(file_fd, offset, os.SEEK_SET)
    data = os.read(file_fd, length)
    if len(data) != length:
        raise RuntimeError("workspace changed during transfer")
    print(base64.b64encode(data).decode("ascii"), end="")
finally:
    if file_fd is not None:
        os.close(file_fd)
    os.close(directory_fd)
"""


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
        "_container_user",
        "_cleanup_required",
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
        container_user: str = "10001:10001",
    ) -> None:
        if (
            not isinstance(workspace, Path)
            or type(image) is not str
            or _IMAGE_PATTERN.fullmatch(image) is None
            or type(container_name) is not str
            or _CONTAINER_NAME_PATTERN.fullmatch(container_name) is None
            or type(container_user) is not str
            or _CONTAINER_USER_PATTERN.fullmatch(container_user) is None
            or any(int(part) > 2_147_483_647 for part in container_user.split(":"))
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
        self._container_user = container_user
        self._timeout_seconds = float(command_timeout_seconds)
        self._running = False
        self._cleanup_required = False

    def _mount_argument(self) -> str:
        output = StringIO(newline="")
        csv.writer(output, lineterminator="").writerow(
            (
                "type=bind",
                f"src={self._workspace}",
                "dst=/host-workspace",
                "readonly",
            )
        )
        return output.getvalue()

    def _workspace_tmpfs_argument(self) -> str:
        uid, gid = self._container_user.split(":")
        return (
            "/workspace:rw,noexec,nosuid,"
            f"size={MAX_DECLARED_TOTAL_BYTES},nr_inodes={_WORKSPACE_TMPFS_INODES},"
            f"uid={uid},gid={gid},mode=0700"
        )

    def _sync_into_container(self) -> None:
        result = self._execute_docker(
            (
                "docker",
                "exec",
                "--user",
                self._container_user,
                "--workdir",
                "/",
                self._container_name,
                "python",
                "-c",
                _SYNC_IN_SCRIPT,
            )
        )
        if result.exit_code != 0:
            raise DockerSandboxError("sandbox command failed")

    def _internal_exec(
        self,
        script: str,
        *arguments: str,
        timeout_seconds: float | None = None,
    ) -> DockerCommandResult:
        result = self._execute_docker(
            (
                "docker",
                "exec",
                "--user",
                self._container_user,
                "--workdir",
                "/",
                self._container_name,
                "python",
                "-c",
                script,
                *arguments,
            ),
            timeout_seconds=timeout_seconds,
        )
        if result.exit_code != 0:
            raise DockerSandboxError("sandbox command failed")
        return result

    @staticmethod
    def _validated_manifest_path(value: object) -> tuple[Path, str]:
        if type(value) is not str:
            raise DockerSandboxError("sandbox command failed")
        try:
            validate_relative_artifact_path(value)
        except (ValueError, UnicodeError):
            raise DockerSandboxError("sandbox command failed") from None
        pure_path = PurePosixPath(value)
        canonical = "/".join(part.casefold() for part in pure_path.parts)
        return Path(*pure_path.parts), canonical

    def _read_workspace_manifest(
        self,
    ) -> tuple[tuple[Path, ...], tuple[tuple[Path, bytes], ...]]:
        deadline = monotonic() + self._timeout_seconds

        def remaining_timeout() -> float:
            remaining = deadline - monotonic()
            if remaining <= 0:
                raise DockerSandboxError("sandbox command failed")
            return remaining

        result = self._internal_exec(
            _MANIFEST_SCRIPT,
            str(MAX_DECLARED_ENTRIES),
            str(MAX_DECLARED_FILE_BYTES),
            str(MAX_DECLARED_TOTAL_BYTES),
            timeout_seconds=remaining_timeout(),
        )
        try:
            manifest = json.loads(result.stdout)
        except (json.JSONDecodeError, UnicodeError):
            raise DockerSandboxError("sandbox command failed") from None
        if type(manifest) is not dict or set(manifest) != {"directories", "files"}:
            raise DockerSandboxError("sandbox command failed")
        raw_directories = manifest["directories"]
        raw_files = manifest["files"]
        if type(raw_directories) is not list or type(raw_files) is not list:
            raise DockerSandboxError("sandbox command failed")
        if len(raw_directories) + len(raw_files) > MAX_DECLARED_ENTRIES:
            raise DockerSandboxError("sandbox command failed")

        directories: list[Path] = []
        canonical_paths: set[str] = set()
        for raw_directory in raw_directories:
            directory, canonical = self._validated_manifest_path(raw_directory)
            if canonical in canonical_paths:
                raise DockerSandboxError("sandbox command failed")
            canonical_paths.add(canonical)
            directories.append(directory)

        directory_paths = {directory.as_posix() for directory in directories}
        files: list[tuple[Path, bytes]] = []
        total_bytes = 0
        for raw_file in raw_files:
            if type(raw_file) is not dict or set(raw_file) != {
                "path",
                "size",
                "sha256",
            }:
                raise DockerSandboxError("sandbox command failed")
            relative, canonical = self._validated_manifest_path(raw_file["path"])
            size = raw_file["size"]
            expected_hash = raw_file["sha256"]
            if (
                canonical in canonical_paths
                or type(size) is not int
                or not 0 <= size <= MAX_DECLARED_FILE_BYTES
                or type(expected_hash) is not str
                or re.fullmatch(r"[0-9a-f]{64}", expected_hash) is None
            ):
                raise DockerSandboxError("sandbox command failed")
            canonical_paths.add(canonical)
            parent = PurePosixPath(raw_file["path"]).parent.as_posix()
            if parent != "." and parent not in directory_paths:
                raise DockerSandboxError("sandbox command failed")
            total_bytes += size
            if total_bytes > MAX_DECLARED_TOTAL_BYTES:
                raise DockerSandboxError("sandbox command failed")

            content = bytearray()
            for offset in range(0, size, _SYNC_CHUNK_BYTES):
                length = min(_SYNC_CHUNK_BYTES, size - offset)
                chunk_result = self._internal_exec(
                    _CHUNK_SCRIPT,
                    raw_file["path"],
                    str(size),
                    str(offset),
                    str(length),
                    timeout_seconds=remaining_timeout(),
                )
                try:
                    chunk = base64.b64decode(chunk_result.stdout, validate=True)
                except (ValueError, UnicodeError):
                    raise DockerSandboxError("sandbox command failed") from None
                if len(chunk) != length:
                    raise DockerSandboxError("sandbox command failed")
                content.extend(chunk)
            content_bytes = bytes(content)
            if hashlib.sha256(content_bytes).hexdigest() != expected_hash:
                raise DockerSandboxError("sandbox command failed")
            files.append((relative, content_bytes))
        return tuple(directories), tuple(files)

    def _replace_host_workspace(
        self,
        directories: tuple[Path, ...],
        files: tuple[tuple[Path, bytes], ...],
    ) -> None:
        try:
            for path in self._workspace.iterdir():
                if path.is_symlink() or not path.is_dir():
                    path.unlink()
                else:
                    shutil.rmtree(path)
            for relative in sorted(directories, key=lambda item: len(item.parts)):
                (self._workspace / relative).mkdir(parents=True, exist_ok=True)
            for relative, content in files:
                target = self._workspace / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)
        except OSError:
            raise DockerSandboxError("sandbox command failed") from None

    def _sync_from_container(self) -> None:
        directories, files = self._read_workspace_manifest()
        self._replace_host_workspace(directories, files)

    def _remove_command(self) -> tuple[str, ...]:
        return ("docker", "rm", "--force", self._container_name)

    def _execute_docker(
        self,
        argv: tuple[str, ...],
        *,
        timeout_seconds: float | None = None,
    ) -> DockerCommandResult:
        effective_timeout = (
            self._timeout_seconds if timeout_seconds is None else timeout_seconds
        )
        if type(effective_timeout) not in {int, float} or effective_timeout <= 0:
            raise DockerSandboxError("sandbox command failed")
        try:
            result = self._command_executor(argv, float(effective_timeout))
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
        for attempt in range(_FAILED_START_CLEANUP_ATTEMPTS):
            try:
                result = self._execute_docker(self._remove_command())
            except BaseException:
                return
            if result.exit_code == 0:
                self._cleanup_required = False
                return
            if attempt + 1 < _FAILED_START_CLEANUP_ATTEMPTS:
                try:
                    sleep(_FAILED_START_CLEANUP_RETRY_SECONDS)
                except BaseException:
                    return

    def start(self) -> None:
        if self._running or self._cleanup_required:
            raise DockerSandboxError("sandbox is already running")
        run_command = (
            "docker",
            "run",
            "--detach",
            "--pull",
            "never",
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
            "--tmpfs",
            self._workspace_tmpfs_argument(),
            "--mount",
            self._mount_argument(),
            "--workdir",
            "/workspace",
            "--user",
            self._container_user,
            self._image,
            "sleep",
            "infinity",
        )
        self._cleanup_required = True
        try:
            result = self._execute_docker(run_command)
        except DockerSandboxError:
            self._cleanup_after_failed_start()
            raise DockerSandboxError("sandbox startup failed") from None
        except BaseException:
            self._cleanup_after_failed_start()
            raise
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
        for _ in range(2):
            try:
                self.close()
            except BaseException:
                continue
            return

    def execute(self, argv: tuple[str, ...]) -> TerminalResult:
        if not self._running:
            raise DockerSandboxError("sandbox is not running")
        self._validate_exec_argv(argv)
        try:
            self._sync_into_container()
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
            self._sync_from_container()
        except DockerSandboxError:
            self._abort_running_sandbox()
            raise DockerSandboxError("sandbox command failed") from None
        except BaseException:
            self._abort_running_sandbox()
            raise
        return TerminalResult(result.exit_code, result.stdout, result.stderr)

    def close(self) -> None:
        if not self._cleanup_required:
            self._running = False
            return
        self._running = False
        try:
            result = self._execute_docker(self._remove_command())
        except DockerSandboxError:
            raise DockerSandboxError("sandbox cleanup failed") from None
        if result.exit_code != 0:
            raise DockerSandboxError("sandbox cleanup failed")
        self._cleanup_required = False

    def __enter__(self) -> "DockerSandbox":
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        cleanup_error: DockerSandboxError | None = None
        cleanup_base_exception: BaseException | None = None
        for _ in range(2):
            try:
                self.close()
            except DockerSandboxError as error:
                cleanup_error = error
                continue
            except BaseException as error:
                if exc_type is None and cleanup_base_exception is None:
                    cleanup_base_exception = error
                continue
            if exc_type is None and cleanup_base_exception is not None:
                raise cleanup_base_exception
            return False
        if exc_type is None:
            if cleanup_base_exception is not None:
                raise cleanup_base_exception
            if cleanup_error is not None:
                raise cleanup_error
        return False
