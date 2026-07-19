from pathlib import Path

import pytest

from agentenv_forge.adapters import AgentRunResult
from agentenv_forge.runner import load_task, run_agent_episode, run_episode
from agentenv_forge.tools import TerminalResult


TASK_IDS = (
    "text-normalization-001",
    "markdown-normalization-001",
    "log-normalization-001",
)


class NoOutputAdapter:
    def run(self, task, tools, event_sink):
        return AgentRunResult(
            termination_reason="finished", agent_failure=None
        )

    def close(self) -> None:
        return None


class PolicyViolationAdapter:
    def run(self, task, tools, event_sink):
        event_sink("tool_call", "terminal_execute")
        tools.execute(("normalize", task.input_artifacts[0], task.allowed_artifacts[0]))
        event_sink("tool_result", "terminal_execute")
        return AgentRunResult(
            termination_reason="finished", agent_failure=None
        )

    def close(self) -> None:
        return None


class PolicyViolationEnvironment:
    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace

    def start(self) -> None:
        return None

    def execute(self, argv: tuple[str, ...]) -> TerminalResult:
        source = (self.workspace / argv[1]).read_text(encoding="utf-8")
        lines = (" ".join(line.lower().split()) for line in source.splitlines())
        content = "\n".join(line for line in lines if line) + "\n"
        output = self.workspace / argv[2]
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(content.encode("utf-8"))
        (self.workspace / "undeclared.txt").write_bytes(b"violation")
        return TerminalResult(0, "", "")

    def close(self) -> None:
        return None


@pytest.mark.parametrize("task_id", TASK_IDS)
def test_m2_task_matrix_keeps_hidden_oracle_out_of_public_task(task_id: str) -> None:
    task = load_task(task_id)
    public_task = task.to_public_task()

    assert task.expected_content
    assert not hasattr(public_task, "expected_content")
    assert not hasattr(public_task, "initial_files")
    assert public_task.task_id == task_id


@pytest.mark.parametrize("task_id", TASK_IDS)
def test_m2_task_matrix_has_deterministic_full_and_partial_rewards(
    task_id: str, tmp_path
) -> None:
    correct = run_episode(task_id, "correct", seed=42, workspace_root=tmp_path)
    wrong = run_episode(task_id, "wrong", seed=42, workspace_root=tmp_path)

    assert correct.reward.model_dump() == {
        "artifact_exists": 1.0,
        "exact_content": 1.0,
        "policy_compliance": 1.0,
        "total": 1.0,
    }
    assert wrong.reward.model_dump() == {
        "artifact_exists": 1.0,
        "exact_content": 0.0,
        "policy_compliance": 1.0,
        "total": 0.3,
    }
    assert correct.canonical_content() == run_episode(
        task_id, "correct", seed=42, workspace_root=tmp_path
    ).canonical_content()


@pytest.mark.parametrize("task_id", TASK_IDS)
def test_m2_task_matrix_negative_and_policy_violating_rewards(
    task_id: str, tmp_path
) -> None:
    negative = run_agent_episode(
        task_id,
        NoOutputAdapter(),
        seed=42,
        workspace_root=tmp_path,
    )
    violating = run_agent_episode(
        task_id,
        PolicyViolationAdapter(),
        seed=42,
        workspace_root=tmp_path,
        terminal_environment_factory=PolicyViolationEnvironment,
    )

    assert negative.termination_reason == "finished"
    assert negative.agent_failure is None
    assert violating.termination_reason == "finished"
    assert violating.agent_failure is None
    assert negative.reward.model_dump() == {
        "artifact_exists": 0.0,
        "exact_content": 0.0,
        "policy_compliance": 1.0,
        "total": 0.1,
    }
    assert violating.reward.model_dump() == {
        "artifact_exists": 1.0,
        "exact_content": 1.0,
        "policy_compliance": 0.0,
        "total": 0.9,
    }
