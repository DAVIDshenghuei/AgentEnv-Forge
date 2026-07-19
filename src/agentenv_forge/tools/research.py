from threading import Condition
from typing import Protocol, runtime_checkable

from ..mcp.research import PaperRecord, PaperSummary
from ..schemas import PublicTask
from .budget import ActionBudget, ActionBudgetExhaustedError


class ResearchActionLimitError(ValueError):
    """The shared episode action budget has been exhausted by research use."""


@runtime_checkable
class ResearchProtocol(Protocol):
    def search_papers(
        self, query: str, limit: int
    ) -> tuple[PaperSummary, ...]: ...

    def get_paper(self, paper_id: str) -> PaperRecord: ...


@runtime_checkable
class ResearchClientProtocol(Protocol):
    def search_papers(
        self, query: str, limit: int
    ) -> tuple[PaperSummary, ...]: ...

    def get_paper(self, paper_id: str) -> PaperRecord: ...


class ResearchTools:
    __slots__ = ("_budget", "_client", "_condition", "_in_flight", "_revoked")

    def __init__(
        self,
        task: PublicTask,
        budget: ActionBudget,
        client: ResearchClientProtocol,
    ) -> None:
        if type(task) is not PublicTask or type(task.max_actions) is not int:
            raise ValueError("invalid research action budget")
        if type(budget) is not ActionBudget or budget.limit != task.max_actions:
            raise ValueError("invalid research action budget")
        if not isinstance(client, ResearchClientProtocol):
            raise ValueError("invalid research client")
        self._budget = budget
        self._client = client
        self._condition = Condition()
        self._in_flight = 0
        self._revoked = False

    def revoke(self) -> None:
        with self._condition:
            self._revoked = True
            while self._in_flight:
                self._condition.wait()

    def _begin_action(self) -> None:
        with self._condition:
            if self._revoked:
                raise ValueError("research tools revoked")
            try:
                self._budget.charge()
            except ActionBudgetExhaustedError:
                raise ResearchActionLimitError(
                    "research action budget exhausted"
                ) from None
            self._in_flight += 1

    def _end_action(self) -> None:
        with self._condition:
            self._in_flight -= 1
            if self._in_flight == 0:
                self._condition.notify_all()

    @staticmethod
    def _valid_text(value: str, maximum_bytes: int) -> bool:
        if not value or value.isspace():
            return False
        if any(ord(character) < 32 or ord(character) == 127 for character in value):
            return False
        try:
            return len(value.encode("utf-8")) <= maximum_bytes
        except UnicodeError:
            return False

    @classmethod
    def _validate_search(cls, query: str, limit: int) -> None:
        if type(query) is not str or type(limit) is not int:
            raise ValueError("invalid research tool call")
        if not cls._valid_text(query, 256) or not 1 <= limit <= 32:
            raise ValueError("invalid research tool call")

    @classmethod
    def _validate_paper_id(cls, paper_id: str) -> None:
        if type(paper_id) is not str:
            raise ValueError("invalid research tool call")
        if not cls._valid_text(paper_id, 64):
            raise ValueError("invalid research tool call")
        if (
            not paper_id[0].isalnum()
            or not paper_id[-1].isalnum()
            or "--" in paper_id
            or any(
                not (
                    "a" <= character <= "z"
                    or "0" <= character <= "9"
                    or character == "-"
                )
                for character in paper_id
            )
        ):
            raise ValueError("invalid research tool call")

    @staticmethod
    def _validate_search_result(
        result: object, limit: int
    ) -> tuple[PaperSummary, ...]:
        if type(result) is not tuple or len(result) > limit:
            raise ValueError("research client failed")
        if any(type(summary) is not PaperSummary for summary in result):
            raise ValueError("research client failed")
        try:
            validated = tuple(
                PaperSummary(summary.paper_id, summary.title, summary.year)
                for summary in result
            )
        except Exception:
            raise ValueError("research client failed") from None
        paper_ids = tuple(summary.paper_id for summary in validated)
        if paper_ids != tuple(sorted(paper_ids)) or len(set(paper_ids)) != len(
            paper_ids
        ):
            raise ValueError("research client failed")
        return validated

    def search_papers(
        self, query: str, limit: int
    ) -> tuple[PaperSummary, ...]:
        self._begin_action()
        try:
            self._validate_search(query, limit)
            try:
                result = self._client.search_papers(query, limit)
            except Exception:
                raise ValueError("research client failed") from None
            return self._validate_search_result(result, limit)
        finally:
            self._end_action()

    def get_paper(self, paper_id: str) -> PaperRecord:
        self._begin_action()
        try:
            self._validate_paper_id(paper_id)
            try:
                result = self._client.get_paper(paper_id)
            except Exception:
                raise ValueError("research client failed") from None
            if type(result) is not PaperRecord:
                raise ValueError("research client failed")
            try:
                validated = PaperRecord(
                    result.paper_id,
                    result.title,
                    result.year,
                    result.abstract,
                    result.body,
                )
            except Exception:
                raise ValueError("research client failed") from None
            if validated.paper_id != paper_id:
                raise ValueError("research client failed")
            return validated
        finally:
            self._end_action()
