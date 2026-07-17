from dataclasses import dataclass
from typing import Callable, Literal, Protocol, runtime_checkable

from ..schemas import PublicTask
from ..tools import WorkspaceProtocol


@dataclass(frozen=True)
class AgentRunResult:
    termination_reason: Literal[
        "finished", "action_limit", "timeout", "agent_error"
    ]
    agent_failure: str | None


@runtime_checkable
class AgentAdapter(Protocol):
    def run(
        self,
        task: PublicTask,
        tools: WorkspaceProtocol,
        event_sink: Callable[[str, str], None],
    ) -> AgentRunResult: ...

    def close(self) -> None: ...
