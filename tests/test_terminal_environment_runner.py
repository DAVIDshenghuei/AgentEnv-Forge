from pathlib import Path

import pytest

from agentenv_forge.adapters import AgentRunResult
from agentenv_forge.runner import run_agent_episode
from agentenv_forge.tools import TerminalResult


class LifecycleAdapter:
    def __init__(self, lifecycle: list[str]) -> None:
        self.lifecycle = lifecycle

    def run(self, task, tools, event_sink):
        self.lifecycle.append("adapter_run")
        event_sink("tool_call", "terminal_execute")
        result = tools.execute(("normalize",))
        event_sink("tool_result", "terminal_execute")
        event_sink("tool_call", f"write_text:{task.allowed_artifacts[0]}")
        tools.write_text(task.allowed_artifacts[0], result.stdout)
        event_sink("tool_result", f"write_text:{task.allowed_artifacts[0]}")
        return AgentRunResult(termination_reason="finished", agent_failure=None)

    def close(self) -> None:
        self.lifecycle.append("adapter_close")


class MutatingTerminalEnvironment:
    def __init__(self, workspace: Path, lifecycle: list[str]) -> None:
        self.workspace = workspace
        self.lifecycle = lifecycle
        self.started = False

    def start(self) -> None:
        self.lifecycle.append("environment_start")
        self.started = True

    def execute(self, argv: tuple[str, ...]) -> TerminalResult:
        assert self.started is True
        self.lifecycle.append("environment_execute")
        return TerminalResult(0, "agentenv forge\ncausal evaluation\n", "")

    def close(self) -> None:
        self.lifecycle.append("environment_close")
        (self.workspace / "result.txt").write_text("mutated during cleanup\n", encoding="utf-8")
        self.started = False


def test_agent_runner_closes_environment_before_hidden_verification(tmp_path) -> None:
    lifecycle: list[str] = []
    workspaces: list[Path] = []

    def environment_factory(workspace: Path) -> MutatingTerminalEnvironment:
        lifecycle.append("environment_create")
        workspaces.append(workspace)
        return MutatingTerminalEnvironment(workspace, lifecycle)

    trajectory = run_agent_episode(
        task_id="text-normalization-001",
        adapter=LifecycleAdapter(lifecycle),
        seed=42,
        workspace_root=tmp_path,
        terminal_environment_factory=environment_factory,
    )

    assert lifecycle == [
        "environment_create",
        "environment_start",
        "adapter_run",
        "environment_execute",
        "adapter_close",
        "environment_close",
    ]
    assert len(workspaces) == 1
    assert not workspaces[0].exists()
    assert trajectory.reward.model_dump() == {
        "artifact_exists": 1.0,
        "exact_content": 0.0,
        "policy_compliance": 1.0,
        "total": 0.3,
    }
    assert trajectory.termination_reason == "finished"
    assert [(event.kind, event.detail) for event in trajectory.events][-4:] == [
        ("adapter_stop", "adapter stopped"),
        ("environment_stop", "terminal environment stopped"),
        ("tools_revoked", "workspace tools revoked"),
        ("verify", "deterministic verifier completed"),
    ]


class FailingEnvironment:
    def __init__(self, lifecycle: list[str], failure_point: str, marker: str) -> None:
        self.lifecycle = lifecycle
        self.failure_point = failure_point
        self.marker = marker

    def start(self) -> None:
        self.lifecycle.append("environment_start")
        if self.failure_point == "start":
            raise RuntimeError(self.marker)

    def execute(self, argv: tuple[str, ...]) -> TerminalResult:
        return TerminalResult(0, "agentenv forge\ncausal evaluation\n", "")

    def close(self) -> None:
        self.lifecycle.append("environment_close")
        if self.failure_point == "close":
            raise RuntimeError(self.marker)


class TransientCloseEnvironment(FailingEnvironment):
    def __init__(self, lifecycle: list[str]) -> None:
        super().__init__(lifecycle, "none", "unused")
        self.close_attempts = 0

    def close(self) -> None:
        self.lifecycle.append("environment_close")
        self.close_attempts += 1
        if self.close_attempts == 1:
            raise RuntimeError("transient cleanup failure")


def test_agent_runner_retries_transient_environment_cleanup_before_verification(
    tmp_path,
) -> None:
    lifecycle: list[str] = []

    trajectory = run_agent_episode(
        task_id="text-normalization-001",
        adapter=LifecycleAdapter(lifecycle),
        seed=42,
        workspace_root=tmp_path,
        terminal_environment_factory=lambda workspace: TransientCloseEnvironment(
            lifecycle
        ),
    )

    assert lifecycle[-3:] == [
        "adapter_close",
        "environment_close",
        "environment_close",
    ]
    assert trajectory.termination_reason == "finished"
    assert trajectory.environment_failure is None
    assert trajectory.reward.total == 1.0


def test_agent_runner_sanitizes_environment_start_failure_and_closes_adapter(
    tmp_path,
) -> None:
    lifecycle: list[str] = []
    marker = "PRIVATE START FAILURE"

    trajectory = run_agent_episode(
        task_id="text-normalization-001",
        adapter=LifecycleAdapter(lifecycle),
        seed=42,
        workspace_root=tmp_path,
        terminal_environment_factory=lambda workspace: FailingEnvironment(
            lifecycle, "start", marker
        ),
    )

    assert lifecycle == ["environment_start", "environment_close", "adapter_close"]
    assert trajectory.termination_reason == "environment_error"
    assert trajectory.environment_failure == "terminal environment startup failed"
    assert trajectory.reward.total == 0.0
    assert marker not in trajectory.canonical_content()
    assert not list(tmp_path.iterdir())


def test_agent_runner_blocks_verification_when_environment_cleanup_fails(
    tmp_path,
) -> None:
    lifecycle: list[str] = []
    marker = "PRIVATE CLEANUP FAILURE"

    trajectory = run_agent_episode(
        task_id="text-normalization-001",
        adapter=LifecycleAdapter(lifecycle),
        seed=42,
        workspace_root=tmp_path,
        terminal_environment_factory=lambda workspace: FailingEnvironment(
            lifecycle, "close", marker
        ),
    )

    assert lifecycle == [
        "environment_start",
        "adapter_run",
        "adapter_close",
        "environment_close",
        "environment_close",
    ]
    assert trajectory.termination_reason == "environment_error"
    assert trajectory.environment_failure == "terminal environment cleanup failed"
    assert trajectory.reward.total == 0.0
    assert trajectory.events[-2].detail == "terminal environment cleanup failed"
    assert trajectory.events[-1].detail == "workspace tools revoked"
    assert all(event.kind != "verify" for event in trajectory.events)
    assert marker not in trajectory.canonical_content()
    assert not list(tmp_path.iterdir())


class InterruptingStartEnvironment(FailingEnvironment):
    def start(self) -> None:
        self.lifecycle.append("environment_start")
        raise KeyboardInterrupt


class InterruptingCloseAdapter(LifecycleAdapter):
    def close(self) -> None:
        self.lifecycle.append("adapter_close")
        raise KeyboardInterrupt


def test_environment_start_base_exception_closes_environment_and_adapter(tmp_path) -> None:
    lifecycle: list[str] = []

    with pytest.raises(KeyboardInterrupt):
        run_agent_episode(
            task_id="text-normalization-001",
            adapter=LifecycleAdapter(lifecycle),
            seed=42,
            workspace_root=tmp_path,
            terminal_environment_factory=lambda workspace: InterruptingStartEnvironment(
                lifecycle, "none", "unused"
            ),
        )

    assert lifecycle == ["environment_start", "environment_close", "adapter_close"]
    assert not list(tmp_path.iterdir())


def test_adapter_close_base_exception_still_closes_environment(tmp_path) -> None:
    lifecycle: list[str] = []

    with pytest.raises(KeyboardInterrupt):
        run_agent_episode(
            task_id="text-normalization-001",
            adapter=InterruptingCloseAdapter(lifecycle),
            seed=42,
            workspace_root=tmp_path,
            terminal_environment_factory=lambda workspace: FailingEnvironment(
                lifecycle, "none", "unused"
            ),
        )

    assert lifecycle == [
        "environment_start",
        "adapter_run",
        "adapter_close",
        "environment_close",
    ]
    assert not list(tmp_path.iterdir())


class MaskingStartEnvironment(FailingEnvironment):
    def start(self) -> None:
        self.lifecycle.append("environment_start")
        raise KeyboardInterrupt("original start cancellation")

    def close(self) -> None:
        self.lifecycle.append("environment_close")
        raise SystemExit("secondary environment cleanup")


class CleanupFailingEnvironment(FailingEnvironment):
    def close(self) -> None:
        self.lifecycle.append("environment_close")
        raise SystemExit("secondary environment cleanup")


class TransientBaseCloseEnvironment(FailingEnvironment):
    def __init__(self, lifecycle: list[str]) -> None:
        super().__init__(lifecycle, "none", "unused")
        self.close_attempts = 0

    def close(self) -> None:
        self.lifecycle.append("environment_close")
        self.close_attempts += 1
        if self.close_attempts == 1:
            raise SystemExit("transient cleanup cancellation")


class PrimaryRunCancellationAdapter(LifecycleAdapter):
    def run(self, task, tools, event_sink):
        self.lifecycle.append("adapter_run")
        raise KeyboardInterrupt("primary adapter cancellation")


class MaskingRunAdapter(LifecycleAdapter):
    def run(self, task, tools, event_sink):
        self.lifecycle.append("adapter_run")
        raise KeyboardInterrupt("original adapter cancellation")

    def close(self) -> None:
        self.lifecycle.append("adapter_close")
        raise SystemExit("secondary adapter cleanup")


def test_start_cleanup_base_exception_does_not_mask_original(tmp_path) -> None:
    lifecycle: list[str] = []

    with pytest.raises(KeyboardInterrupt, match="^original start cancellation$"):
        run_agent_episode(
            task_id="text-normalization-001",
            adapter=LifecycleAdapter(lifecycle),
            seed=42,
            workspace_root=tmp_path,
            terminal_environment_factory=lambda workspace: MaskingStartEnvironment(
                lifecycle, "none", "unused"
            ),
        )

    assert lifecycle == [
        "environment_start",
        "environment_close",
        "environment_close",
        "adapter_close",
    ]


def test_adapter_cancellation_retries_environment_base_cleanup(tmp_path) -> None:
    lifecycle: list[str] = []

    with pytest.raises(KeyboardInterrupt, match="^primary adapter cancellation$"):
        run_agent_episode(
            task_id="text-normalization-001",
            adapter=PrimaryRunCancellationAdapter(lifecycle),
            seed=42,
            workspace_root=tmp_path,
            terminal_environment_factory=lambda workspace: TransientBaseCloseEnvironment(
                lifecycle
            ),
        )

    assert lifecycle == [
        "environment_start",
        "adapter_run",
        "adapter_close",
        "environment_close",
        "environment_close",
    ]


def test_adapter_cleanup_base_exceptions_do_not_mask_original(tmp_path) -> None:
    lifecycle: list[str] = []

    with pytest.raises(KeyboardInterrupt, match="^original adapter cancellation$"):
        run_agent_episode(
            task_id="text-normalization-001",
            adapter=MaskingRunAdapter(lifecycle),
            seed=42,
            workspace_root=tmp_path,
            terminal_environment_factory=lambda workspace: CleanupFailingEnvironment(
                lifecycle, "none", "unused"
            ),
        )

    assert lifecycle == [
        "environment_start",
        "adapter_run",
        "adapter_close",
        "environment_close",
        "environment_close",
    ]