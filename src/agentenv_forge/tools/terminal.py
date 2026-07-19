from dataclasses import dataclass
from threading import Condition
from typing import Callable, Protocol, runtime_checkable

from ..schemas import PublicTask
from .budget import ActionBudget, ActionBudgetExhaustedError

_MAX_ARGUMENTS = 64
_MAX_ARGUMENT_BYTES = 4_096
_MAX_TOTAL_ARGUMENT_BYTES = 16_384
_MAX_OUTPUT_BYTES = 65_536


class TerminalActionLimitError(ValueError):
    """The shared episode action budget has been exhausted by terminal use."""


@dataclass(frozen=True, slots=True)
class TerminalResult:
    exit_code: int
    stdout: str
    stderr: str


@runtime_checkable
class TerminalProtocol(Protocol):
    def execute(self, argv: tuple[str, ...]) -> TerminalResult: ...


CommandRunner = Callable[[tuple[str, ...]], TerminalResult]


class TerminalTools:
    """Revocable terminal capability around a trusted injected command runner.

    This class never executes host commands itself. M2's Docker sandbox runner is
    a separate trusted lifecycle component injected through ``command_runner``.
    """

    __slots__ = (
        "_budget",
        "_command_runner",
        "_condition",
        "_in_flight",
        "_revoked",
    )

    def __init__(
        self,
        task: PublicTask,
        budget: ActionBudget,
        command_runner: CommandRunner,
    ) -> None:
        if type(task) is not PublicTask or type(task.max_actions) is not int:
            raise ValueError("invalid terminal action budget")
        if type(budget) is not ActionBudget or budget.limit != task.max_actions:
            raise ValueError("invalid terminal action budget")
        if not callable(command_runner):
            raise ValueError("invalid terminal command runner")
        self._budget = budget
        self._command_runner = command_runner
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
                raise ValueError("terminal tools revoked")
            try:
                self._budget.charge()
            except ActionBudgetExhaustedError:
                raise TerminalActionLimitError(
                    "terminal action budget exhausted"
                ) from None
            self._in_flight += 1

    def _end_action(self) -> None:
        with self._condition:
            self._in_flight -= 1
            if self._in_flight == 0:
                self._condition.notify_all()

    @staticmethod
    def _validate_argv(argv: tuple[str, ...]) -> None:
        if type(argv) is not tuple or not 1 <= len(argv) <= _MAX_ARGUMENTS:
            raise ValueError("invalid terminal tool call")
        total_bytes = 0
        for argument in argv:
            if type(argument) is not str or not argument or "\x00" in argument:
                raise ValueError("invalid terminal tool call")
            try:
                argument_bytes = len(argument.encode("utf-8"))
            except UnicodeError:
                raise ValueError("invalid terminal tool call") from None
            if argument_bytes > _MAX_ARGUMENT_BYTES:
                raise ValueError("invalid terminal tool call")
            total_bytes += argument_bytes
            if total_bytes > _MAX_TOTAL_ARGUMENT_BYTES:
                raise ValueError("invalid terminal tool call")

    @staticmethod
    def _validate_result(result: object) -> TerminalResult:
        if type(result) is not TerminalResult:
            raise ValueError("terminal execution failed")
        if type(result.exit_code) is not int or not 0 <= result.exit_code <= 255:
            raise ValueError("terminal execution failed")
        if type(result.stdout) is not str or type(result.stderr) is not str:
            raise ValueError("terminal execution failed")
        try:
            stdout_bytes = len(result.stdout.encode("utf-8"))
            stderr_bytes = len(result.stderr.encode("utf-8"))
        except UnicodeError:
            raise ValueError("terminal execution failed") from None
        if stdout_bytes > _MAX_OUTPUT_BYTES or stderr_bytes > _MAX_OUTPUT_BYTES:
            raise ValueError("terminal execution failed")
        return result

    def execute(self, argv: tuple[str, ...]) -> TerminalResult:
        self._begin_action()
        try:
            self._validate_argv(argv)
            try:
                result = self._command_runner(argv)
            except Exception:
                raise ValueError("terminal execution failed") from None
            return self._validate_result(result)
        finally:
            self._end_action()
