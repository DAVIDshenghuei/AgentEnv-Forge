from dataclasses import dataclass
from typing import Callable, Literal, Protocol, runtime_checkable

from ..schemas import PublicTask
from ..tools import AgentToolsProtocol


@dataclass(frozen=True)
class AgentRunResult:
    termination_reason: Literal[
        "finished", "action_limit", "timeout", "agent_error"
    ]
    agent_failure: str | None


@runtime_checkable
class AgentAdapter(Protocol):
    """Trusted host integration around an untrusted model-facing capability.

    Implementations receive only the narrow workspace protocol. Malicious
    same-process Python plugins require process isolation outside M1.
    """

    def run(
        self,
        task: PublicTask,
        tools: AgentToolsProtocol,
        event_sink: Callable[[str, str], None],
    ) -> AgentRunResult: ...

    def close(self) -> None: ...
