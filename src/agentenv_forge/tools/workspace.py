import stat
from pathlib import Path
from threading import Condition, RLock
from typing import Callable, Protocol, runtime_checkable
from weakref import WeakKeyDictionary

from ..runner import _read_bounded_text, _workspace_target, _write_workspace_text
from ..schemas import PublicTask
from .budget import ActionBudget, ActionBudgetExhaustedError
from .research import ResearchProtocol, ResearchTools
from .terminal import TerminalProtocol, TerminalResult, TerminalTools


class WorkspaceActionLimitError(ValueError):
    """The public workspace action budget has been exhausted."""


@runtime_checkable
class WorkspaceProtocol(Protocol):
    def list_files(self) -> tuple[str, ...]: ...

    def read_text(self, relative: str) -> str: ...

    def write_text(self, relative: str, content: str) -> None: ...


@runtime_checkable
class AgentToolsProtocol(
    WorkspaceProtocol, TerminalProtocol, ResearchProtocol, Protocol
):
    """Opaque model-facing workspace, terminal, and research capabilities."""


class WorkspaceTools:
    __slots__ = (
        "_workspace",
        "_task",
        "_budget",
        "_revoked",
        "_condition",
        "_in_flight",
    )

    def __init__(
        self,
        workspace: Path,
        task: PublicTask,
        budget: ActionBudget | None = None,
    ) -> None:
        task_limit = task.max_actions
        if type(task_limit) is not int:
            raise ValueError("invalid workspace action budget")
        if budget is not None:
            if type(budget) is not ActionBudget or budget.limit != task_limit:
                raise ValueError("invalid workspace action budget")
        else:
            budget = ActionBudget(task_limit)
        try:
            root = workspace.resolve(strict=True)
            if not root.is_dir():
                raise ValueError("workspace is unavailable")
        except OSError:
            raise ValueError("workspace is unavailable") from None
        self._workspace = root
        self._task = task
        self._budget = budget
        self._revoked = False
        self._condition = Condition()
        self._in_flight = 0

    def revoke(self) -> None:
        with self._condition:
            self._revoked = True
            while self._in_flight:
                self._condition.wait()

    def _begin_action(self) -> None:
        with self._condition:
            if self._revoked:
                raise ValueError("workspace tools revoked")
            try:
                self._budget.charge()
            except ActionBudgetExhaustedError:
                raise WorkspaceActionLimitError(
                    "workspace action budget exhausted"
                ) from None
            self._in_flight += 1

    def _end_action(self) -> None:
        with self._condition:
            self._in_flight -= 1
            if self._in_flight == 0:
                self._condition.notify_all()

    def _check_declared_target(self, relative: str) -> None:
        target = self._workspace
        parts = Path(relative).parts
        for index, part in enumerate(parts):
            target = target / part
            try:
                entry_stat = target.lstat()
            except FileNotFoundError:
                return
            except OSError:
                raise ValueError(
                    "workspace contains unsupported filesystem entry"
                ) from None
            mode = entry_stat.st_mode
            file_attributes = getattr(entry_stat, "st_file_attributes", 0)
            reparse_point = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
            if stat.S_ISLNK(mode) or file_attributes & reparse_point:
                raise ValueError(
                    "workspace contains unsupported filesystem entry"
                ) from None
            is_leaf = index == len(parts) - 1
            if (not is_leaf and not stat.S_ISDIR(mode)) or (
                is_leaf and not stat.S_ISREG(mode)
            ):
                raise ValueError(
                    "workspace contains unsupported filesystem entry"
                ) from None

    def list_files(self) -> tuple[str, ...]:
        self._begin_action()
        try:
            declared_paths = self._task.input_artifacts + self._task.allowed_artifacts
            existing_files = []
            try:
                for relative in sorted(declared_paths):
                    self._check_declared_target(relative)
                    if _workspace_target(self._workspace, relative).is_file():
                        existing_files.append(relative)
            except OSError:
                raise ValueError("workspace listing failed") from None
            return tuple(existing_files)
        finally:
            self._end_action()

    def read_text(self, relative: str) -> str:
        self._begin_action()
        try:
            if relative not in self._task.input_artifacts:
                raise ValueError("artifact is not a declared public input")
            self._check_declared_target(relative)
            try:
                return _read_bounded_text(_workspace_target(self._workspace, relative))
            except (OSError, UnicodeError):
                raise ValueError("public input is not readable text") from None
        finally:
            self._end_action()

    def write_text(self, relative: str, content: str) -> None:
        self._begin_action()
        try:
            if relative not in self._task.allowed_artifacts:
                raise ValueError("artifact is not a declared allowed output")
            self._check_declared_target(relative)
            try:
                _write_workspace_text(self._workspace, relative, content)
            except (OSError, UnicodeError):
                raise ValueError("output could not be written") from None
        finally:
            self._end_action()


class _AdapterWorkspaceFacade:
    """Opaque model-facing capability for trusted host adapter integrations.

    It exposes the narrow protocol, never a raw workspace path. This is not a
    sandbox for malicious same-process Python plugins; those require process
    isolation outside M1.
    """

    __slots__ = ("__weakref__",)

    def list_files(self) -> tuple[str, ...]:
        state = _facade_state(self)
        detail = "list_files"
        state.before_call(detail)
        try:
            result = state.tools.list_files()
        except BaseException:
            state.after_call(detail, False)
            raise
        state.after_call(detail, True)
        return result

    def read_text(self, relative: str) -> str:
        if type(relative) is not str:
            raise ValueError("invalid workspace tool call")
        state = _facade_state(self)
        detail = f"read_text:{relative}"
        state.before_call(detail)
        try:
            result = state.tools.read_text(relative)
        except BaseException:
            state.after_call(detail, False)
            raise
        state.after_call(detail, True)
        return result

    def write_text(self, relative: str, content: str) -> None:
        if type(relative) is not str or type(content) is not str:
            raise ValueError("invalid workspace tool call")
        state = _facade_state(self)
        detail = f"write_text:{relative}"
        state.before_call(detail)
        try:
            state.tools.write_text(relative, content)
        except BaseException:
            state.after_call(detail, False)
            raise
        state.after_call(detail, True)

    def execute(self, argv: tuple[str, ...]) -> TerminalResult:
        state = _facade_state(self)
        if state.terminal_tools is None:
            raise ValueError("terminal tools unavailable")
        detail = "terminal_execute"
        state.before_call(detail)
        try:
            result = state.terminal_tools.execute(argv)
        except BaseException:
            state.after_call(detail, False)
            raise
        state.after_call(detail, True)
        return result

    def search_papers(self, query: str, limit: int):
        if type(query) is not str or type(limit) is not int:
            raise ValueError("invalid research tool call")
        state = _facade_state(self)
        detail = "research_search_papers"
        state.before_call(detail)
        try:
            result = state.research_tools.search_papers(query, limit)
        except BaseException:
            state.after_call(detail, False)
            raise
        state.after_call(detail, True)
        return result

    def get_paper(self, paper_id: str):
        if type(paper_id) is not str:
            raise ValueError("invalid research tool call")
        state = _facade_state(self)
        detail = "research_get_paper"
        state.before_call(detail)
        try:
            result = state.research_tools.get_paper(paper_id)
        except BaseException:
            state.after_call(detail, False)
            raise
        state.after_call(detail, True)
        return result


class _FacadeState:
    __slots__ = (
        "tools",
        "terminal_tools",
        "research_tools",
        "before_call",
        "after_call",
    )

    def __init__(
        self,
        tools: WorkspaceTools,
        before_call: Callable[[str], None],
        after_call: Callable[[str, bool], None],
        terminal_tools: TerminalTools | None,
        research_tools: ResearchTools,
    ) -> None:
        self.tools = tools
        self.terminal_tools = terminal_tools
        self.research_tools = research_tools
        self.before_call = before_call
        self.after_call = after_call


_FACADE_TOOLS: WeakKeyDictionary[
    _AdapterWorkspaceFacade, _FacadeState | None
] = WeakKeyDictionary()
_FACADE_TOOLS_LOCK = RLock()


def _facade_state(facade: _AdapterWorkspaceFacade) -> _FacadeState:
    with _FACADE_TOOLS_LOCK:
        state = _FACADE_TOOLS.get(facade)
    if state is None:
        raise ValueError("workspace tools revoked")
    return state


def _create_workspace_facade(
    tools: WorkspaceTools,
    before_call: Callable[[str], None],
    after_call: Callable[[str, bool], None],
    terminal_tools: TerminalTools | None = None,
    research_tools: ResearchTools | None = None,
) -> AgentToolsProtocol:
    if research_tools is None:
        raise ValueError("research tools unavailable")
    facade = _AdapterWorkspaceFacade()
    with _FACADE_TOOLS_LOCK:
        _FACADE_TOOLS[facade] = _FacadeState(
            tools, before_call, after_call, terminal_tools, research_tools
        )
    return facade


def _revoke_workspace_facade(facade: WorkspaceProtocol) -> None:
    with _FACADE_TOOLS_LOCK:
        state = _FACADE_TOOLS.get(facade)
        _FACADE_TOOLS[facade] = None
    if state is not None:
        state.tools.revoke()
        if state.terminal_tools is not None:
            state.terminal_tools.revoke()
        state.research_tools.revoke()
