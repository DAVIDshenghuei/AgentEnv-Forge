from agentenv_forge.adapters import AgentRunResult
from agentenv_forge.runner import run_agent_episode
from agentenv_forge.tools import (
    TerminalActionLimitError,
    TerminalResult,
    WorkspaceActionLimitError,
)


class TerminalThenWriteAdapter:
    def __init__(self, argv_marker: str) -> None:
        self.closed = False
        self.argv_marker = argv_marker

    def run(self, task, tools, event_sink):
        argv = ("normalize", task.input_artifacts[0], self.argv_marker)
        event_sink("tool_call", "terminal_execute")
        result = tools.execute(argv)
        event_sink("tool_result", "terminal_execute")
        event_sink("tool_call", f"write_text:{task.allowed_artifacts[0]}")
        tools.write_text(task.allowed_artifacts[0], result.stdout)
        event_sink("tool_result", f"write_text:{task.allowed_artifacts[0]}")
        return AgentRunResult(termination_reason="finished", agent_failure=None)

    def close(self) -> None:
        self.closed = True


def test_agent_runner_exposes_terminal_without_recording_raw_argv(tmp_path) -> None:
    calls: list[tuple[str, ...]] = []
    secret_argv = "SECRET MODEL ARGUMENT"

    def terminal_runner(argv: tuple[str, ...]) -> TerminalResult:
        calls.append(argv)
        return TerminalResult(0, "agentenv forge\ncausal evaluation\n", "")

    adapter = TerminalThenWriteAdapter(secret_argv)
    trajectory = run_agent_episode(
        task_id="text-normalization-001",
        adapter=adapter,
        seed=42,
        workspace_root=tmp_path,
        terminal_command_runner=terminal_runner,
    )

    assert adapter.closed is True
    assert calls == [("normalize", "input.txt", secret_argv)]
    assert trajectory.reward.total == 1.0
    assert [(event.kind, event.detail) for event in trajectory.events] == [
        ("reset", "initial state restored"),
        ("adapter_start", "adapter started"),
        ("tool_call", "terminal_execute"),
        ("tool_result", "terminal_execute"),
        ("tool_call", "write_text:result.txt"),
        ("tool_result", "write_text:result.txt"),
        ("adapter_stop", "adapter stopped"),
        ("tools_revoked", "workspace tools revoked"),
        ("verify", "deterministic verifier completed"),
    ]
    assert secret_argv not in trajectory.canonical_content()


class SharedBudgetAdapter:
    def __init__(self) -> None:
        self.closed = False

    def run(self, task, tools, event_sink):
        for index in range(task.max_actions):
            if index % 2 == 0:
                detail = "terminal_execute"
                event_sink("tool_call", detail)
                tools.execute(("true",))
                event_sink("tool_result", detail)
            else:
                detail = "list_files"
                event_sink("tool_call", detail)
                tools.list_files()
                event_sink("tool_result", detail)
        event_sink("tool_call", "terminal_execute")
        try:
            tools.execute(("true",))
        except (TerminalActionLimitError, WorkspaceActionLimitError):
            return AgentRunResult(termination_reason="action_limit", agent_failure=None)
        raise AssertionError("shared action budget did not terminate")

    def close(self) -> None:
        self.closed = True


def test_agent_runner_shares_one_budget_across_terminal_and_workspace(tmp_path) -> None:
    calls: list[tuple[str, ...]] = []

    def terminal_runner(argv: tuple[str, ...]) -> TerminalResult:
        calls.append(argv)
        return TerminalResult(0, "", "")

    adapter = SharedBudgetAdapter()
    trajectory = run_agent_episode(
        task_id="text-normalization-001",
        adapter=adapter,
        seed=42,
        workspace_root=tmp_path,
        terminal_command_runner=terminal_runner,
    )

    assert adapter.closed is True
    assert calls == [("true",)] * 4
    assert trajectory.termination_reason == "action_limit"
    assert trajectory.agent_failure is None
    assert trajectory.events[-4].model_dump() == {
        "sequence": 19,
        "kind": "adapter_termination",
        "detail": "action_limit",
    }
    assert trajectory.events[-3].detail == "adapter stopped"
    assert trajectory.events[-2].detail == "workspace tools revoked"
    assert trajectory.events[-1].detail == "deterministic verifier completed"


def test_terminal_capability_setup_failure_closes_started_environment(tmp_path) -> None:
    lifecycle: list[str] = []

    class MissingExecuteEnvironment:
        def start(self) -> None:
            lifecycle.append("start")

        def close(self) -> None:
            lifecycle.append("environment_close")

    class NoRunAdapter:
        def run(self, task, tools, event_sink):
            raise AssertionError("adapter must not run")

        def close(self) -> None:
            lifecycle.append("adapter_close")

    def factory(workspace):
        lifecycle.append("factory")
        return MissingExecuteEnvironment()

    trajectory = run_agent_episode(
        task_id="text-normalization-001",
        adapter=NoRunAdapter(),
        seed=42,
        workspace_root=tmp_path,
        terminal_environment_factory=factory,
    )

    assert lifecycle == ["factory", "start", "environment_close", "adapter_close"]
    assert trajectory.termination_reason == "environment_error"
    assert trajectory.environment_failure == "terminal environment startup failed"
    assert trajectory.reward.total == 0.0
    assert [(event.kind, event.detail) for event in trajectory.events] == [
        ("reset", "initial state restored"),
        ("environment_failure", "terminal environment startup failed"),
    ]
