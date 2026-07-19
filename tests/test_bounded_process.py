import os
import sys
import time

import pytest

from agentenv_forge.sandbox import (
    BoundedProcessExecutor,
    DockerCommandResult,
    ProcessOutputLimitError,
    ProcessTimeoutError,
)


def test_bounded_executor_preserves_exact_argv_without_shell_interpretation() -> None:
    executor = BoundedProcessExecutor(max_output_bytes=4_096)
    literal = "value; echo must-not-run"

    result = executor(
        (
            sys.executable,
            "-c",
            "import sys; print(sys.argv[1])",
            literal,
        ),
        5.0,
    )

    assert result == DockerCommandResult(0, literal + os.linesep, "")


def test_bounded_executor_returns_nonzero_stdout_and_stderr() -> None:
    executor = BoundedProcessExecutor(max_output_bytes=4_096)

    result = executor(
        (
            sys.executable,
            "-c",
            "import sys; print('out'); print('err', file=sys.stderr); raise SystemExit(7)",
        ),
        5.0,
    )

    assert result == DockerCommandResult(
        7,
        "out" + os.linesep,
        "err" + os.linesep,
    )


@pytest.mark.parametrize("stream", ("stdout", "stderr"))
def test_bounded_executor_terminates_on_output_overflow_without_leaking_output(
    stream,
) -> None:
    executor = BoundedProcessExecutor(max_output_bytes=1_024)
    code = (
        "import sys; "
        + ("sys.stdout" if stream == "stdout" else "sys.stderr")
        + ".write('SECRET OUTPUT' * 10000); "
        + ("sys.stdout" if stream == "stdout" else "sys.stderr")
        + ".flush()"
    )

    with pytest.raises(
        ProcessOutputLimitError, match="^process output limit exceeded$"
    ) as failure:
        executor((sys.executable, "-c", code), 5.0)

    assert "SECRET OUTPUT" not in str(failure.value)


def test_bounded_executor_terminates_on_timeout() -> None:
    executor = BoundedProcessExecutor(max_output_bytes=4_096)
    started = time.monotonic()

    with pytest.raises(ProcessTimeoutError, match="^process timed out$"):
        executor((sys.executable, "-c", "import time; time.sleep(10)"), 0.1)

    assert time.monotonic() - started < 3.0


@pytest.mark.parametrize(
    ("argv", "timeout"),
    (
        ([], 1.0),
        ((), 1.0),
        (("",), 1.0),
        ((sys.executable, 3), 1.0),
        ((sys.executable,), True),
        ((sys.executable,), 0.0),
    ),
)
def test_bounded_executor_rejects_invalid_inputs_without_launch(argv, timeout) -> None:
    executor = BoundedProcessExecutor(max_output_bytes=4_096)

    with pytest.raises(ValueError, match="^invalid process invocation$"):
        executor(argv, timeout)


def test_bounded_executor_rejects_hostile_argv_without_magic_calls() -> None:
    calls: list[str] = []

    class HostileTuple(tuple):
        def __iter__(self):
            calls.append("iter")
            raise AssertionError("must not iterate hostile tuple")

        def __len__(self):
            calls.append("len")
            raise AssertionError("must not size hostile tuple")

    executor = BoundedProcessExecutor(max_output_bytes=4_096)

    with pytest.raises(ValueError, match="^invalid process invocation$"):
        executor(HostileTuple((sys.executable,)), 1.0)

    assert calls == []


@pytest.mark.parametrize("limit", (True, 0, -1, 1.5, "1024", 1_048_577))
def test_bounded_executor_rejects_invalid_output_limit(limit) -> None:
    with pytest.raises(ValueError, match="^invalid process output limit$"):
        BoundedProcessExecutor(max_output_bytes=limit)
