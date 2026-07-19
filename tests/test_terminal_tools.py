from threading import Event, Thread

import pytest

from agentenv_forge.schemas import PublicTask
from agentenv_forge.tools import (
    ActionBudget,
    TerminalActionLimitError,
    TerminalProtocol,
    TerminalResult,
    TerminalTools,
    WorkspaceTools,
)


def _public_task(max_actions: int = 3) -> PublicTask:
    return PublicTask.model_validate(
        {
            "task_id": "terminal-contract",
            "version": "1",
            "instruction": "Inspect the input and produce the output.",
            "input_artifacts": ("source.txt",),
            "allowed_artifacts": ("result.txt",),
            "max_actions": max_actions,
        }
    )


class RecordingRunner:
    def __init__(self, result: TerminalResult) -> None:
        self.result = result
        self.calls: list[tuple[str, ...]] = []

    def __call__(self, argv: tuple[str, ...]) -> TerminalResult:
        self.calls.append(argv)
        return self.result


def test_terminal_executes_exact_argv_and_returns_bounded_result() -> None:
    task = _public_task()
    budget = ActionBudget(task.max_actions)
    expected = TerminalResult(exit_code=0, stdout="Python 3.11\n", stderr="")
    runner = RecordingRunner(expected)
    tools = TerminalTools(task=task, budget=budget, command_runner=runner)

    assert isinstance(tools, TerminalProtocol)
    assert tools.execute(("python", "--version")) == expected
    assert runner.calls == [("python", "--version")]
    assert budget.used == 1


def test_nonzero_exit_is_a_normal_terminal_result() -> None:
    task = _public_task()
    budget = ActionBudget(task.max_actions)
    expected = TerminalResult(exit_code=7, stdout="", stderr="not found\n")
    tools = TerminalTools(
        task=task,
        budget=budget,
        command_runner=RecordingRunner(expected),
    )

    assert tools.execute(("missing-command",)) == expected
    assert budget.used == 1


def test_workspace_and_terminal_consume_one_shared_budget(tmp_path) -> None:
    task = _public_task(max_actions=2)
    (tmp_path / "source.txt").write_text("source", encoding="utf-8")
    budget = ActionBudget(task.max_actions)
    workspace = WorkspaceTools(workspace=tmp_path, task=task, budget=budget)
    runner = RecordingRunner(TerminalResult(0, "ok\n", ""))
    terminal = TerminalTools(task=task, budget=budget, command_runner=runner)

    assert workspace.read_text("source.txt") == "source"
    assert terminal.execute(("python", "-V")).exit_code == 0
    with pytest.raises(
        TerminalActionLimitError, match="^terminal action budget exhausted$"
    ):
        terminal.execute(("python", "-V"))
    assert runner.calls == [("python", "-V")]
    assert budget.used == 2


@pytest.mark.parametrize(
    "argv",
    (
        [],
        (),
        ("",),
        ("python\x00",),
        ("python", 3),
        tuple("x" * 4097 for _ in range(1)),
    ),
)
def test_invalid_terminal_call_is_rejected_after_consuming_one_action(argv) -> None:
    task = _public_task(max_actions=1)
    budget = ActionBudget(task.max_actions)
    runner = RecordingRunner(TerminalResult(0, "", ""))
    terminal = TerminalTools(task=task, budget=budget, command_runner=runner)

    with pytest.raises(ValueError, match="^invalid terminal tool call$"):
        terminal.execute(argv)

    assert runner.calls == []
    assert budget.used == 1


def test_hostile_argv_subclasses_are_rejected_without_magic_calls() -> None:
    calls: list[str] = []

    class HostileTuple(tuple):
        def __iter__(self):
            calls.append("iter")
            raise AssertionError("must not iterate hostile tuple")

        def __len__(self):
            calls.append("len")
            raise AssertionError("must not size hostile tuple")

    class HostileString(str):
        def encode(self, *args, **kwargs):
            calls.append("encode")
            raise AssertionError("must not encode hostile string")

    task = _public_task(max_actions=2)
    budget = ActionBudget(task.max_actions)
    runner = RecordingRunner(TerminalResult(0, "", ""))
    terminal = TerminalTools(task=task, budget=budget, command_runner=runner)

    with pytest.raises(ValueError, match="^invalid terminal tool call$"):
        terminal.execute(HostileTuple(("python",)))
    with pytest.raises(ValueError, match="^invalid terminal tool call$"):
        terminal.execute((HostileString("python"),))

    assert calls == []
    assert runner.calls == []
    assert budget.used == 2


def test_runner_exception_is_sanitized_and_charged(tmp_path) -> None:
    secret = "SECRET RUNNER FAILURE"

    def fail(_argv: tuple[str, ...]) -> TerminalResult:
        raise RuntimeError(f"{tmp_path} {secret}")

    task = _public_task(max_actions=1)
    budget = ActionBudget(task.max_actions)
    terminal = TerminalTools(task=task, budget=budget, command_runner=fail)

    with pytest.raises(ValueError, match="^terminal execution failed$") as failure:
        terminal.execute(("python", "-V"))

    message = str(failure.value)
    assert str(tmp_path) not in message
    assert secret not in message
    assert budget.used == 1


@pytest.mark.parametrize(
    "result",
    (
        object(),
        TerminalResult(exit_code=True, stdout="", stderr=""),
        TerminalResult(exit_code=256, stdout="", stderr=""),
        TerminalResult(exit_code=0, stdout="x" * 65_537, stderr=""),
        TerminalResult(exit_code=0, stdout="", stderr="x" * 65_537),
    ),
)
def test_invalid_or_oversized_runner_result_fails_closed(result) -> None:
    task = _public_task(max_actions=1)
    budget = ActionBudget(task.max_actions)
    terminal = TerminalTools(
        task=task,
        budget=budget,
        command_runner=lambda _argv: result,
    )

    with pytest.raises(ValueError, match="^terminal execution failed$"):
        terminal.execute(("python", "-V"))
    assert budget.used == 1


def test_revoked_terminal_rejects_without_consuming_budget() -> None:
    task = _public_task(max_actions=1)
    budget = ActionBudget(task.max_actions)
    runner = RecordingRunner(TerminalResult(0, "", ""))
    terminal = TerminalTools(task=task, budget=budget, command_runner=runner)
    terminal.revoke()

    with pytest.raises(ValueError, match="^terminal tools revoked$"):
        terminal.execute(("python", "-V"))

    assert budget.used == 0
    assert runner.calls == []


def test_terminal_revoke_waits_for_admitted_execution_to_finish() -> None:
    task = _public_task(max_actions=1)
    budget = ActionBudget(task.max_actions)
    entered = Event()
    release = Event()
    revoke_finished = Event()
    outcomes: list[object] = []

    def block(_argv: tuple[str, ...]) -> TerminalResult:
        entered.set()
        assert release.wait(timeout=5)
        return TerminalResult(0, "done\n", "")

    terminal = TerminalTools(task=task, budget=budget, command_runner=block)

    def execute() -> None:
        try:
            outcomes.append(terminal.execute(("python", "-V")))
        except BaseException as error:
            outcomes.append(error)

    execution_thread = Thread(target=execute)
    execution_thread.start()
    assert entered.wait(timeout=5)

    revoke_thread = Thread(
        target=lambda: (terminal.revoke(), revoke_finished.set())
    )
    revoke_thread.start()
    assert not revoke_finished.wait(timeout=0.1)

    release.set()
    execution_thread.join(timeout=5)
    revoke_thread.join(timeout=5)

    assert not execution_thread.is_alive()
    assert not revoke_thread.is_alive()
    assert revoke_finished.is_set()
    assert outcomes == [TerminalResult(0, "done\n", "")]
    assert budget.used == 1
