import os

import pytest

from agentenv_forge.adapters import AgentRunResult
from agentenv_forge.runner import run_docker_agent_episode
from agentenv_forge.sandbox import BoundedProcessExecutor


class DockerNormalizationAdapter:
    def __init__(self) -> None:
        self.public_task = None
        self.closed = False

    def run(self, task, tools, event_sink):
        self.public_task = task
        script = (
            "import sys; from pathlib import Path; "
            "s=Path(sys.argv[1]).read_text(encoding='utf-8'); "
            "lines=(' '.join(x.lower().split()) for x in s.splitlines()); "
            "Path(sys.argv[2]).write_text('\\n'.join(x for x in lines if x)+'\\n', encoding='utf-8')"
        )
        event_sink("tool_call", "terminal_execute")
        result = tools.execute(
            (
                "python",
                "-c",
                script,
                task.input_artifacts[0],
                task.allowed_artifacts[0],
            )
        )
        event_sink("tool_result", "terminal_execute")
        assert result.exit_code == 0
        return AgentRunResult(termination_reason="finished", agent_failure=None)

    def close(self) -> None:
        self.closed = True


@pytest.mark.skipif(
    os.environ.get("AGENTENV_FORGE_DOCKER_INTEGRATION") != "1",
    reason="requires the explicitly built sandbox image and Docker daemon",
)
@pytest.mark.parametrize(
    "task_id",
    (
        "text-normalization-001",
        "markdown-normalization-001",
        "log-normalization-001",
    ),
)
def test_docker_agent_episode_runs_terminal_then_cleans_before_oracle(
    tmp_path, task_id: str
) -> None:
    adapter = DockerNormalizationAdapter()

    trajectory = run_docker_agent_episode(
        task_id=task_id,
        adapter=adapter,
        seed=42,
        workspace_root=tmp_path,
        image="agentenv-forge-sandbox:test",
    )

    assert adapter.closed is True
    assert adapter.public_task is not None
    assert not hasattr(adapter.public_task, "expected_content")
    assert trajectory.reward.total == 1.0
    assert trajectory.termination_reason == "finished"
    assert trajectory.environment_failure is None
    assert [event.detail for event in trajectory.events] == [
        "initial state restored",
        "terminal environment started",
        "adapter started",
        "terminal_execute",
        "terminal_execute",
        "adapter stopped",
        "terminal environment stopped",
        "workspace tools revoked",
        "deterministic verifier completed",
    ]
    executor = BoundedProcessExecutor(max_output_bytes=65_536)
    containers = executor(
        (
            "docker",
            "ps",
            "--all",
            "--quiet",
            "--filter",
            "ancestor=agentenv-forge-sandbox:test",
        ),
        5.0,
    )
    assert containers.exit_code == 0
    assert containers.stdout.strip() == ""
