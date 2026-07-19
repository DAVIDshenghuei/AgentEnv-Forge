import pytest

from agentenv_forge.adapters import AgentRunResult
from agentenv_forge.mcp.research import PaperRecord, PaperSummary
from agentenv_forge.runner import run_agent_episode
from agentenv_forge.tools import (
    ResearchActionLimitError,
    ResearchClientProtocol,
    TerminalResult,
)


SUMMARY = PaperSummary("paper-001", "Deterministic Agent Environments", 2024)
PAPER = PaperRecord(
    paper_id="paper-002",
    title="Offline Tool Evaluation",
    year=2023,
    abstract="An AGENT ENVIRONMENT corpus for offline research.",
    body="second body",
)


class FakeResearchClient(ResearchClientProtocol):
    def __init__(self) -> None:
        self.calls: list[tuple[object, ...]] = []

    def search_papers(self, query: str, limit: int):
        self.calls.append(("search_papers", query, limit))
        return (SUMMARY,)

    def get_paper(self, paper_id: str):
        self.calls.append(("get_paper", paper_id))
        return PAPER


def _paired(event_sink, detail, call):
    event_sink("tool_call", detail)
    result = call()
    event_sink("tool_result", detail)
    return result


class ResearchNormalizeAdapter:
    def __init__(self) -> None:
        self.closed = False

    def run(self, task, tools, event_sink):
        _paired(
            event_sink,
            "research_search_papers",
            lambda: tools.search_papers("agent environment", 2),
        )
        _paired(
            event_sink,
            "research_get_paper",
            lambda: tools.get_paper("paper-002"),
        )
        source = _paired(
            event_sink,
            "read_text:input.txt",
            lambda: tools.read_text("input.txt"),
        )
        normalized = "\n".join(
            " ".join(line.lower().split())
            for line in source.splitlines()
            if line.strip()
        ) + "\n"
        _paired(
            event_sink,
            "write_text:result.txt",
            lambda: tools.write_text("result.txt", normalized),
        )
        return AgentRunResult("finished", None)

    def close(self) -> None:
        self.closed = True


def test_runner_exposes_research_with_generic_events_and_full_reward(tmp_path) -> None:
    client = FakeResearchClient()
    adapter = ResearchNormalizeAdapter()

    trajectory = run_agent_episode(
        task_id="text-normalization-001",
        adapter=adapter,
        seed=42,
        workspace_root=tmp_path,
        research_client=client,
    )

    assert adapter.closed is True
    assert trajectory.reward.total == 1.0
    assert client.calls == [
        ("search_papers", "agent environment", 2),
        ("get_paper", "paper-002"),
    ]
    event_pairs = [(event.kind, event.detail) for event in trajectory.events]
    assert ("tool_call", "research_search_papers") in event_pairs
    assert ("tool_result", "research_search_papers") in event_pairs
    assert ("tool_call", "research_get_paper") in event_pairs
    assert ("tool_result", "research_get_paper") in event_pairs
    serialized = trajectory.canonical_content()
    for secret in (
        "agent environment",
        "paper-002",
        PAPER.title,
        PAPER.abstract,
        PAPER.body,
    ):
        assert secret not in serialized


def test_research_terminal_and_workspace_share_runner_budget(tmp_path) -> None:
    client = FakeResearchClient()
    terminal_calls: list[tuple[str, ...]] = []

    class ExhaustingAdapter:
        def __init__(self) -> None:
            self.closed = False

        def run(self, task, tools, event_sink):
            _paired(
                event_sink,
                "research_search_papers",
                lambda: tools.search_papers("agent", 1),
            )
            for _ in range(3):
                _paired(
                    event_sink,
                    "terminal_execute",
                    lambda: tools.execute(("true",)),
                )
            for _ in range(4):
                _paired(event_sink, "list_files", tools.list_files)
            event_sink("tool_call", "research_search_papers")
            try:
                tools.search_papers("agent", 1)
            except ResearchActionLimitError:
                return AgentRunResult("action_limit", None)
            raise AssertionError("shared budget did not terminate")

        def close(self) -> None:
            self.closed = True

    adapter = ExhaustingAdapter()
    trajectory = run_agent_episode(
        task_id="text-normalization-001",
        adapter=adapter,
        seed=42,
        workspace_root=tmp_path,
        terminal_command_runner=lambda argv: (
            terminal_calls.append(argv) or TerminalResult(0, "", "")
        ),
        research_client=client,
    )

    assert adapter.closed is True
    assert trajectory.termination_reason == "action_limit"
    assert trajectory.agent_failure is None
    assert client.calls == [("search_papers", "agent", 1)]
    assert terminal_calls == [("true",)] * 3


def test_missing_research_client_is_sanitized_agent_failure(tmp_path) -> None:
    class MissingClientAdapter:
        def __init__(self) -> None:
            self.closed = False

        def run(self, task, tools, event_sink):
            event_sink("tool_call", "research_search_papers")
            tools.search_papers("agent", 1)
            raise AssertionError("research unexpectedly succeeded")

        def close(self) -> None:
            self.closed = True

    adapter = MissingClientAdapter()
    trajectory = run_agent_episode(
        task_id="text-normalization-001",
        adapter=adapter,
        seed=42,
        workspace_root=tmp_path,
    )

    assert adapter.closed is True
    assert trajectory.termination_reason == "agent_error"
    assert trajectory.agent_failure == "adapter execution failed"
    assert trajectory.environment_failure is None
    details = [event.detail for event in trajectory.events]
    assert "research_search_papers" in details
    assert details.index("workspace tools revoked") < details.index(
        "deterministic verifier completed"
    )


def test_research_facade_is_revoked_before_hidden_verification(tmp_path) -> None:
    client = FakeResearchClient()

    class RetainingAdapter(ResearchNormalizeAdapter):
        retained_tools = None

        def run(self, task, tools, event_sink):
            self.retained_tools = tools
            return super().run(task, tools, event_sink)

    adapter = RetainingAdapter()
    trajectory = run_agent_episode(
        task_id="text-normalization-001",
        adapter=adapter,
        seed=42,
        workspace_root=tmp_path,
        research_client=client,
    )
    calls_after_verification = tuple(client.calls)

    assert trajectory.reward.total == 1.0
    assert calls_after_verification == (
        ("search_papers", "agent environment", 2),
        ("get_paper", "paper-002"),
    )
    with pytest.raises(ValueError, match="^(workspace|research) tools revoked$"):
        adapter.retained_tools.search_papers("agent", 1)
    with pytest.raises(ValueError, match="^(workspace|research) tools revoked$"):
        adapter.retained_tools.list_files()
    assert tuple(client.calls) == calls_after_verification


def test_invalid_research_client_is_rejected_before_environment_start(tmp_path) -> None:
    lifecycle: list[str] = []

    class RecordingEnvironment:
        def start(self) -> None:
            lifecycle.append("start")

        def execute(self, argv):
            raise AssertionError("terminal must not execute")

        def close(self) -> None:
            lifecycle.append("close")

    class RecordingAdapter:
        def __init__(self) -> None:
            self.closed = False

        def run(self, task, tools, event_sink):
            raise AssertionError("adapter must not run")

        def close(self) -> None:
            self.closed = True

    def factory(workspace):
        lifecycle.append("factory")
        return RecordingEnvironment()

    adapter = RecordingAdapter()
    with pytest.raises(ValueError, match="^invalid research client$"):
        run_agent_episode(
            task_id="text-normalization-001",
            adapter=adapter,
            seed=42,
            workspace_root=tmp_path,
            terminal_environment_factory=factory,
            research_client=object(),
        )

    assert lifecycle == []
    assert adapter.closed is False
