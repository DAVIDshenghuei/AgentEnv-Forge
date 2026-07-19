import os

import pytest

from agentenv_forge.adapters import AgentRunResult
from agentenv_forge.mcp.client import StdioResearchClient
from agentenv_forge.mcp.research import RESEARCH_CORPUS_VERSION
from agentenv_forge.runner import load_task, run_agent_episode, run_docker_agent_episode
from agentenv_forge.sandbox import BoundedProcessExecutor
from agentenv_forge.tools import TerminalResult


QUERY = "Offline Tool Evaluation"
TITLE = "Offline Tool Evaluation"
YEAR = 2023
ABSTRACT = "An AGENT ENVIRONMENT corpus for offline research."
BODY = "second body"
EXPECTED_OUTPUT = (
    "Offline Tool Evaluation (2023)\n"
    "An AGENT ENVIRONMENT corpus for offline research.\n"
)
SCRIPT = (
    "import sys; print(f'{sys.argv[1]} ({sys.argv[2]})'); print(sys.argv[3])"
)
EXPECTED_ARGV = ("python", "-c", SCRIPT, TITLE, str(YEAR), ABSTRACT)


def _paired(event_sink, detail, call):
    event_sink("tool_call", detail)
    result = call()
    event_sink("tool_result", detail)
    return result


class ResearchSynthesisAdapter:
    def __init__(self) -> None:
        self.closed = False

    def run(self, task, tools, event_sink):
        _paired(event_sink, "list_files", tools.list_files)
        summaries = _paired(
            event_sink,
            "research_search_papers",
            lambda: tools.search_papers(QUERY, 1),
        )
        paper = _paired(
            event_sink,
            "research_get_paper",
            lambda: tools.get_paper(summaries[0].paper_id),
        )
        terminal = _paired(
            event_sink,
            "terminal_execute",
            lambda: tools.execute(
                (
                    "python",
                    "-c",
                    SCRIPT,
                    paper.title,
                    str(paper.year),
                    paper.abstract,
                )
            ),
        )
        assert terminal.exit_code == 0
        _paired(
            event_sink,
            "write_text:result.txt",
            lambda: tools.write_text("result.txt", terminal.stdout),
        )
        return AgentRunResult("finished", None)

    def close(self) -> None:
        self.closed = True


def _assert_success(trajectory, adapter, client) -> None:
    assert trajectory.reward.total == 1.0
    assert trajectory.termination_reason == "finished"
    assert adapter.closed is True
    assert client.has_active_session is False
    tool_events = [
        (event.kind, event.detail)
        for event in trajectory.events
        if event.kind in {"tool_call", "tool_result"}
    ]
    assert tool_events == [
        (kind, detail)
        for detail in (
            "list_files",
            "research_search_papers",
            "research_get_paper",
            "terminal_execute",
            "write_text:result.txt",
        )
        for kind in ("tool_call", "tool_result")
    ]
    canonical = trajectory.canonical_content()
    for private_value in (QUERY, "paper-002", TITLE, ABSTRACT, BODY):
        assert private_value not in canonical


def test_research_task_public_contract_excludes_hidden_oracle() -> None:
    task = load_task("research-synthesis-001")
    public_task = task.to_public_task()

    assert task.split == "train"
    assert task.version == RESEARCH_CORPUS_VERSION
    assert "expected_content" not in type(public_task).model_fields
    assert "initial_files" not in type(public_task).model_fields
    assert task.expected_content not in public_task.model_dump_json()


def test_host_research_mcp_episode_acceptance(tmp_path) -> None:
    terminal_calls: list[tuple[str, ...]] = []

    def terminal_runner(argv: tuple[str, ...]) -> TerminalResult:
        assert argv == EXPECTED_ARGV
        terminal_calls.append(argv)
        return TerminalResult(0, EXPECTED_OUTPUT, "")

    adapter = ResearchSynthesisAdapter()
    client = StdioResearchClient()
    trajectory = run_agent_episode(
        task_id="research-synthesis-001",
        adapter=adapter,
        seed=42,
        workspace_root=tmp_path,
        terminal_command_runner=terminal_runner,
        research_client=client,
    )

    _assert_success(trajectory, adapter, client)
    assert terminal_calls == [EXPECTED_ARGV]


@pytest.mark.skipif(
    os.environ.get("AGENTENV_FORGE_DOCKER_INTEGRATION") != "1",
    reason="requires the explicitly built sandbox image and Docker daemon",
)
def test_docker_research_mcp_episode_acceptance(tmp_path) -> None:
    adapter = ResearchSynthesisAdapter()
    client = StdioResearchClient()
    trajectory = run_docker_agent_episode(
        task_id="research-synthesis-001",
        adapter=adapter,
        seed=42,
        workspace_root=tmp_path,
        image="agentenv-forge-sandbox:test",
        research_client=client,
    )

    _assert_success(trajectory, adapter, client)
    details = [event.detail for event in trajectory.events]
    assert details.index("terminal environment started") < details.index(
        "adapter started"
    )
    assert details.index("adapter stopped") < details.index(
        "terminal environment stopped"
    )
    assert details.index("terminal environment stopped") < details.index(
        "deterministic verifier completed"
    )

    containers = BoundedProcessExecutor(max_output_bytes=65_536)(
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
