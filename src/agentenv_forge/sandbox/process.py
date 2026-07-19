import subprocess
from dataclasses import dataclass
from threading import Event, Thread
from time import monotonic

from .docker import DockerCommandResult

_MAX_ARGUMENTS = 256
_MAX_ARGUMENT_BYTES = 4_096
_MAX_TOTAL_ARGUMENT_BYTES = 65_536
_MAX_CONFIGURED_OUTPUT_BYTES = 1_048_576
_READ_CHUNK_BYTES = 8_192
_POLL_SECONDS = 0.01
_KILL_WAIT_SECONDS = 2.0


class ProcessExecutionError(RuntimeError):
    """A sanitized trusted process execution failure."""


class ProcessTimeoutError(ProcessExecutionError):
    """The trusted process exceeded its wall-clock deadline."""


class ProcessOutputLimitError(ProcessExecutionError):
    """The trusted process exceeded a bounded output stream."""


@dataclass(slots=True)
class _Capture:
    data: bytearray
    overflow: Event
    failure: Event


def _drain_stream(stream, capture: _Capture, limit: int) -> None:
    try:
        while chunk := stream.read(_READ_CHUNK_BYTES):
            remaining = limit - len(capture.data)
            if remaining > 0:
                capture.data.extend(chunk[:remaining])
            if len(chunk) > remaining:
                capture.overflow.set()
    except Exception:
        capture.failure.set()
    finally:
        try:
            stream.close()
        except Exception:
            capture.failure.set()


class BoundedProcessExecutor:
    """Trusted exact-argv subprocess executor with bounded binary output."""

    __slots__ = ("_max_output_bytes",)

    def __init__(self, max_output_bytes: int = 65_536) -> None:
        if (
            type(max_output_bytes) is not int
            or not 1 <= max_output_bytes <= _MAX_CONFIGURED_OUTPUT_BYTES
        ):
            raise ValueError("invalid process output limit")
        self._max_output_bytes = max_output_bytes

    @staticmethod
    def _validate_invocation(
        argv: tuple[str, ...], timeout_seconds: float
    ) -> None:
        if (
            type(argv) is not tuple
            or not 1 <= len(argv) <= _MAX_ARGUMENTS
            or type(timeout_seconds) not in {int, float}
            or not 0 < timeout_seconds <= 60
        ):
            raise ValueError("invalid process invocation")
        total_bytes = 0
        for argument in argv:
            if type(argument) is not str or not argument or "\x00" in argument:
                raise ValueError("invalid process invocation")
            try:
                argument_bytes = len(argument.encode("utf-8"))
            except UnicodeError:
                raise ValueError("invalid process invocation") from None
            if argument_bytes > _MAX_ARGUMENT_BYTES:
                raise ValueError("invalid process invocation")
            total_bytes += argument_bytes
            if total_bytes > _MAX_TOTAL_ARGUMENT_BYTES:
                raise ValueError("invalid process invocation")

    @staticmethod
    def _kill_and_wait(process: subprocess.Popen[bytes]) -> None:
        try:
            process.kill()
        except OSError:
            pass
        try:
            process.wait(timeout=_KILL_WAIT_SECONDS)
        except (OSError, subprocess.TimeoutExpired):
            pass

    @classmethod
    def _settle_or_kill_and_wait(cls, process: subprocess.Popen[bytes]) -> None:
        try:
            process.wait(timeout=_KILL_WAIT_SECONDS)
        except subprocess.TimeoutExpired:
            cls._kill_and_wait(process)
        except OSError:
            cls._kill_and_wait(process)

    def __call__(
        self, argv: tuple[str, ...], timeout_seconds: float
    ) -> DockerCommandResult:
        self._validate_invocation(argv, timeout_seconds)
        try:
            process = subprocess.Popen(
                argv,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=False,
                text=False,
            )
        except (OSError, ValueError):
            raise ProcessExecutionError("process launch failed") from None
        if process.stdout is None or process.stderr is None:
            self._kill_and_wait(process)
            raise ProcessExecutionError("process execution failed")

        stdout_capture = _Capture(bytearray(), Event(), Event())
        stderr_capture = _Capture(bytearray(), Event(), Event())
        stdout_thread = Thread(
            target=_drain_stream,
            args=(process.stdout, stdout_capture, self._max_output_bytes),
            daemon=True,
        )
        stderr_thread = Thread(
            target=_drain_stream,
            args=(process.stderr, stderr_capture, self._max_output_bytes),
            daemon=True,
        )
        stdout_thread.start()
        stderr_thread.start()

        deadline = monotonic() + float(timeout_seconds)
        failure: ProcessExecutionError | None = None
        exit_code: int | None = None
        try:
            while exit_code is None:
                if stdout_capture.overflow.is_set() or stderr_capture.overflow.is_set():
                    failure = ProcessOutputLimitError("process output limit exceeded")
                    self._kill_and_wait(process)
                    break
                if stdout_capture.failure.is_set() or stderr_capture.failure.is_set():
                    failure = ProcessExecutionError("process execution failed")
                    self._kill_and_wait(process)
                    break
                remaining = deadline - monotonic()
                if remaining <= 0:
                    failure = ProcessTimeoutError("process timed out")
                    self._kill_and_wait(process)
                    break
                try:
                    exit_code = process.wait(timeout=min(_POLL_SECONDS, remaining))
                except subprocess.TimeoutExpired:
                    continue
                except OSError:
                    failure = ProcessExecutionError("process execution failed")
                    self._kill_and_wait(process)
                    break
        except BaseException:
            try:
                self._settle_or_kill_and_wait(process)
            except BaseException:
                try:
                    self._kill_and_wait(process)
                except BaseException:
                    pass
            for thread in (stdout_thread, stderr_thread):
                try:
                    thread.join(timeout=_KILL_WAIT_SECONDS)
                except BaseException:
                    pass
            raise

        stdout_thread.join(timeout=_KILL_WAIT_SECONDS)
        stderr_thread.join(timeout=_KILL_WAIT_SECONDS)
        if stdout_thread.is_alive() or stderr_thread.is_alive():
            raise ProcessExecutionError("process execution failed")
        if failure is not None:
            raise failure
        if stdout_capture.failure.is_set() or stderr_capture.failure.is_set():
            raise ProcessExecutionError("process execution failed")
        if stdout_capture.overflow.is_set() or stderr_capture.overflow.is_set():
            raise ProcessOutputLimitError("process output limit exceeded")
        if type(exit_code) is not int or not 0 <= exit_code <= 255:
            raise ProcessExecutionError("process execution failed")
        try:
            stdout = bytes(stdout_capture.data).decode("utf-8")
            stderr = bytes(stderr_capture.data).decode("utf-8")
        except UnicodeError:
            raise ProcessExecutionError("process output decoding failed") from None
        return DockerCommandResult(exit_code, stdout, stderr)
