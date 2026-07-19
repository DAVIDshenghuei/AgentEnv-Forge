from threading import Event, Thread

import pytest

from agentenv_forge.mcp.research import PaperRecord, PaperSummary
from agentenv_forge.schemas import PublicTask
from agentenv_forge.tools import (
    ActionBudget,
    ResearchActionLimitError,
    ResearchClientProtocol,
    ResearchProtocol,
    ResearchTools,
    WorkspaceTools,
)


SUMMARY = PaperSummary("paper-001", "Deterministic Agent Environments", 2024)
PAPER = PaperRecord(
    paper_id="paper-002",
    title="Offline Tool Evaluation",
    year=2023,
    abstract="An AGENT ENVIRONMENT corpus for offline research.",
    body="second body",
)


def _task(max_actions: int = 3) -> PublicTask:
    return PublicTask.model_validate(
        {
            "task_id": "research-tools",
            "version": "1",
            "instruction": "Research before writing the declared output.",
            "input_artifacts": ("source.txt",),
            "allowed_artifacts": ("result.txt",),
            "max_actions": max_actions,
        }
    )


class RecordingClient(ResearchClientProtocol):
    def __init__(self, search_result=(SUMMARY,), paper_result=PAPER) -> None:
        self.search_result = search_result
        self.paper_result = paper_result
        self.calls: list[tuple[object, ...]] = []

    def search_papers(self, query: str, limit: int) -> tuple[PaperSummary, ...]:
        self.calls.append(("search_papers", query, limit))
        return self.search_result

    def get_paper(self, paper_id: str) -> PaperRecord:
        self.calls.append(("get_paper", paper_id))
        return self.paper_result


def test_research_and_workspace_share_one_action_budget(tmp_path) -> None:
    task = _task()
    (tmp_path / "source.txt").write_text("source", encoding="utf-8")
    budget = ActionBudget(task.max_actions)
    client = RecordingClient()
    research = ResearchTools(task=task, budget=budget, client=client)
    workspace = WorkspaceTools(workspace=tmp_path, task=task, budget=budget)

    assert isinstance(research, ResearchProtocol)
    assert research.search_papers("agent environment", 2) == (SUMMARY,)
    assert workspace.list_files() == ("source.txt",)
    returned = research.get_paper("paper-002")
    assert returned == PAPER
    assert returned is not PAPER
    with pytest.raises(
        ResearchActionLimitError, match="^research action budget exhausted$"
    ):
        research.search_papers("agent", 1)

    assert budget.used == 3
    assert client.calls == [
        ("search_papers", "agent environment", 2),
        ("get_paper", "paper-002"),
    ]


def test_malformed_calls_are_charged_before_validation_without_client_use() -> None:
    calls: list[str] = []

    class HostileStr(str):
        def casefold(self):
            calls.append("casefold")
            raise AssertionError("must not inspect hostile string")

    task = _task(max_actions=2)
    budget = ActionBudget(task.max_actions)
    client = RecordingClient()
    research = ResearchTools(task=task, budget=budget, client=client)

    with pytest.raises(ValueError, match="^invalid research tool call$"):
        research.search_papers(HostileStr("agent"), 1)
    with pytest.raises(ValueError, match="^invalid research tool call$"):
        research.search_papers("agent", True)

    assert budget.used == 2
    assert client.calls == []
    assert calls == []


def test_untrusted_client_results_fail_closed_with_sanitized_errors() -> None:
    class SummarySubclass(PaperSummary):
        pass

    class RecordSubclass(PaperRecord):
        pass

    invalid_search_results = (
        [SUMMARY],
        (SummarySubclass(SUMMARY.paper_id, SUMMARY.title, SUMMARY.year),),
        (PAPER,),
    )
    for result in invalid_search_results:
        task = _task(max_actions=1)
        research = ResearchTools(
            task=task,
            budget=ActionBudget(1),
            client=RecordingClient(search_result=result),
        )
        with pytest.raises(ValueError, match="^research client failed$"):
            research.search_papers("agent", 1)

    invalid_records = (
        RecordSubclass(
            PAPER.paper_id, PAPER.title, PAPER.year, PAPER.abstract, PAPER.body
        ),
        PaperRecord(
            "paper-001", "Wrong paper", 2024, "Wrong abstract.", "wrong body"
        ),
    )
    for result in invalid_records:
        task = _task(max_actions=1)
        research = ResearchTools(
            task=task,
            budget=ActionBudget(1),
            client=RecordingClient(paper_result=result),
        )
        with pytest.raises(ValueError, match="^research client failed$"):
            research.get_paper("paper-002")


def test_forged_exact_domain_results_are_revalidated_before_return() -> None:
    forged_summary = PaperSummary(
        "paper-001", "Deterministic Agent Environments", 2024
    )
    object.__setattr__(forged_summary, "title", "x" * 257)
    search = ResearchTools(
        task=_task(max_actions=1),
        budget=ActionBudget(1),
        client=RecordingClient(search_result=(forged_summary,)),
    )
    with pytest.raises(ValueError, match="^research client failed$"):
        search.search_papers("agent", 1)

    forged_record = PaperRecord(
        PAPER.paper_id, PAPER.title, PAPER.year, PAPER.abstract, PAPER.body
    )
    object.__setattr__(forged_record, "body", "x" * 1_048_577)
    get = ResearchTools(
        task=_task(max_actions=1),
        budget=ActionBudget(1),
        client=RecordingClient(paper_result=forged_record),
    )
    with pytest.raises(ValueError, match="^research client failed$"):
        get.get_paper("paper-002")


def test_revoke_drains_in_flight_call_then_rejects_without_recharging() -> None:
    entered = Event()
    release = Event()
    revoke_finished = Event()
    outcomes: list[object] = []

    class BlockingClient(RecordingClient):
        def search_papers(self, query: str, limit: int):
            self.calls.append(("search_papers", query, limit))
            entered.set()
            assert release.wait(timeout=5)
            return (SUMMARY,)

    task = _task(max_actions=2)
    budget = ActionBudget(task.max_actions)
    client = BlockingClient()
    research = ResearchTools(task=task, budget=budget, client=client)

    call_thread = Thread(
        target=lambda: outcomes.append(research.search_papers("agent", 1))
    )
    call_thread.start()
    assert entered.wait(timeout=5)
    revoke_thread = Thread(
        target=lambda: (research.revoke(), revoke_finished.set())
    )
    revoke_thread.start()
    assert not revoke_finished.wait(timeout=0.1)

    release.set()
    call_thread.join(timeout=5)
    revoke_thread.join(timeout=5)
    assert not call_thread.is_alive()
    assert not revoke_thread.is_alive()
    assert outcomes == [(SUMMARY,)]
    assert revoke_finished.is_set()

    with pytest.raises(ValueError, match="^research tools revoked$"):
        research.get_paper("paper-002")
    assert budget.used == 1
    assert client.calls == [("search_papers", "agent", 1)]
