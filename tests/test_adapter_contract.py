import inspect
from dataclasses import FrozenInstanceError

import pytest

from agentenv_forge.adapters import AgentAdapter, AgentRunResult


def test_agent_run_result_is_frozen_and_preserves_outcome_fields():
    result = AgentRunResult(termination_reason="finished", agent_failure=None)

    assert result.termination_reason == "finished"
    assert result.agent_failure is None
    with pytest.raises(FrozenInstanceError):
        result.termination_reason = "changed"


def test_agent_adapter_is_a_narrow_runtime_structural_protocol():
    class LocalAdapter:
        def run(self, task, tools, event_sink):
            return AgentRunResult("finished", None)

        def close(self):
            return None

    adapter = LocalAdapter()

    assert isinstance(adapter, AgentAdapter)
    public_methods = {
        name
        for name, method in inspect.getmembers(
            AgentAdapter, predicate=inspect.isfunction
        )
        if not name.startswith("_")
    }
    assert public_methods == {"run", "close"}
    assert tuple(inspect.signature(AgentAdapter.run).parameters) == (
        "self",
        "task",
        "tools",
        "event_sink",
    )
