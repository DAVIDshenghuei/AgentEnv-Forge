from typing import Callable

from ..schemas import PublicTask
from ..tools import WorkspaceActionLimitError, WorkspaceProtocol
from .base import AgentRunResult


class ScriptedAdapter:
    __slots__ = ("_input_artifact", "_output_artifact", "_transform", "_closed")

    def __init__(
        self,
        input_artifact: str,
        output_artifact: str,
        transform: Callable[[str], str],
    ) -> None:
        self._input_artifact = input_artifact
        self._output_artifact = output_artifact
        self._transform = transform
        self._closed = False

    def close(self) -> None:
        self._closed = True

    def run(
        self,
        task: PublicTask,
        tools: WorkspaceProtocol,
        event_sink: Callable[[str, str], None],
    ) -> AgentRunResult:
        if self._closed:
            return AgentRunResult(
                termination_reason="agent_error",
                agent_failure="scripted adapter is closed",
            )
        if (
            type(self._input_artifact) is not str
            or type(self._output_artifact) is not str
            or self._input_artifact not in task.input_artifacts
            or self._output_artifact not in task.allowed_artifacts
        ):
            return AgentRunResult(
                termination_reason="agent_error",
                agent_failure="scripted adapter configuration is invalid",
            )
        try:
            event_sink("tool_call", f"read_text:{self._input_artifact}")
        except Exception:
            return AgentRunResult(
                termination_reason="agent_error",
                agent_failure="scripted adapter failed",
            )
        try:
            source = tools.read_text(self._input_artifact)
        except WorkspaceActionLimitError:
            return AgentRunResult(termination_reason="action_limit", agent_failure=None)
        except Exception:
            return AgentRunResult(
                termination_reason="agent_error",
                agent_failure="scripted adapter failed",
            )
        try:
            event_sink("tool_result", f"read_text:{self._input_artifact}")
        except Exception:
            return AgentRunResult(
                termination_reason="agent_error",
                agent_failure="scripted adapter failed",
            )
        try:
            result = self._transform(source)
        except Exception:
            return AgentRunResult(
                termination_reason="agent_error",
                agent_failure="scripted adapter failed",
            )
        try:
            event_sink("tool_call", f"write_text:{self._output_artifact}")
        except Exception:
            return AgentRunResult(
                termination_reason="agent_error",
                agent_failure="scripted adapter failed",
            )
        try:
            tools.write_text(self._output_artifact, result)
        except WorkspaceActionLimitError:
            return AgentRunResult(termination_reason="action_limit", agent_failure=None)
        except Exception:
            return AgentRunResult(
                termination_reason="agent_error",
                agent_failure="scripted adapter failed",
            )
        try:
            event_sink("tool_result", f"write_text:{self._output_artifact}")
        except Exception:
            return AgentRunResult(
                termination_reason="agent_error",
                agent_failure="scripted adapter failed",
            )
        return AgentRunResult(termination_reason="finished", agent_failure=None)
