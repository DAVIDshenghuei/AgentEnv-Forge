import pytest

from agentenv_forge.adapters import AgentAdapter, ScriptedAdapter
from agentenv_forge.schemas import PublicTask
from agentenv_forge.tools import WorkspaceActionLimitError, WorkspaceTools


def test_scripted_adapter_normalizes_text_and_emits_sanitized_ordered_events(tmp_path):
    public_task = PublicTask.model_validate(
        {
            "task_id": "scripted-normalization",
            "version": "1",
            "instruction": "Normalize the input text into the declared result.",
            "input_artifacts": ("input.txt",),
            "allowed_artifacts": ("result.txt",),
            "max_actions": 4,
        }
    )
    input_text = "  HéLLo   WORLD  \r\n\r\n  Causal   Trace \n"
    expected = "héllo world\ncausal trace\n"

    def normalize(value: str) -> str:
        lines = (" ".join(line.lower().split()) for line in value.splitlines())
        return "\n".join(line for line in lines if line) + "\n"

    (tmp_path / "input.txt").write_bytes(input_text.encode("utf-8"))
    tools = WorkspaceTools(workspace=tmp_path, task=public_task)
    adapter = ScriptedAdapter(
        input_artifact="input.txt",
        output_artifact="result.txt",
        transform=normalize,
    )
    events = []

    result = adapter.run(
        public_task,
        tools,
        lambda kind, detail: events.append((kind, detail)),
    )

    assert result.termination_reason == "finished"
    assert result.agent_failure is None
    assert (tmp_path / "input.txt").read_bytes() == input_text.encode("utf-8")
    assert (tmp_path / "result.txt").read_bytes() == expected.encode("utf-8")
    assert events == [
        ("tool_call", "read_text:input.txt"),
        ("tool_result", "read_text:input.txt"),
        ("tool_call", "write_text:result.txt"),
        ("tool_result", "write_text:result.txt"),
    ]
    event_text = repr(events)
    assert input_text not in event_text
    assert expected not in event_text
    assert str(tmp_path) not in event_text


def test_scripted_adapter_returns_action_limit_with_partial_deterministic_trace(
    tmp_path,
):
    public_task = PublicTask.model_validate(
        {
            "task_id": "scripted-action-limit",
            "version": "1",
            "instruction": "Copy the input into the declared result.",
            "input_artifacts": ("input.txt",),
            "allowed_artifacts": ("result.txt",),
            "max_actions": 1,
        }
    )
    input_text = "unchanged input\n"
    (tmp_path / "input.txt").write_text(input_text, encoding="utf-8")
    tools = WorkspaceTools(workspace=tmp_path, task=public_task)
    adapter = ScriptedAdapter(
        input_artifact="input.txt",
        output_artifact="result.txt",
        transform=lambda value: value.upper(),
    )
    events = []

    result = adapter.run(
        public_task,
        tools,
        lambda kind, detail: events.append((kind, detail)),
    )

    assert result.termination_reason == "action_limit"
    assert result.agent_failure is None
    assert (tmp_path / "input.txt").read_text(encoding="utf-8") == input_text
    assert not (tmp_path / "result.txt").exists()
    assert events == [
        ("tool_call", "read_text:input.txt"),
        ("tool_result", "read_text:input.txt"),
        ("tool_call", "write_text:result.txt"),
    ]


def test_scripted_adapter_maps_transform_failure_to_sanitized_agent_error(tmp_path):
    public_task = PublicTask.model_validate(
        {
            "task_id": "scripted-agent-error",
            "version": "1",
            "instruction": "Transform the input into the declared result.",
            "input_artifacts": ("input.txt",),
            "allowed_artifacts": ("result.txt",),
            "max_actions": 4,
        }
    )
    input_text = "unchanged public input\n"
    secret_marker = "SECRET TRANSFORM DETAIL"
    (tmp_path / "input.txt").write_text(input_text, encoding="utf-8")
    tools = WorkspaceTools(workspace=tmp_path, task=public_task)

    def fail_transform(_value: str) -> str:
        raise RuntimeError(f"{tmp_path} input.txt {secret_marker}")

    adapter = ScriptedAdapter(
        input_artifact="input.txt",
        output_artifact="result.txt",
        transform=fail_transform,
    )
    events = []

    result = adapter.run(
        public_task,
        tools,
        lambda kind, detail: events.append((kind, detail)),
    )

    assert result.termination_reason == "agent_error"
    assert result.agent_failure == "scripted adapter failed"
    assert (tmp_path / "input.txt").read_text(encoding="utf-8") == input_text
    assert not (tmp_path / "result.txt").exists()
    assert events == [
        ("tool_call", "read_text:input.txt"),
        ("tool_result", "read_text:input.txt"),
    ]
    public_record = repr(result) + repr(events)
    assert str(tmp_path) not in public_record
    assert secret_marker not in public_record


def test_closed_scripted_adapter_releases_run_capability_without_touching_tools(
    tmp_path,
):
    public_task = PublicTask.model_validate(
        {
            "task_id": "scripted-close",
            "version": "1",
            "instruction": "Run only while the adapter is open.",
            "input_artifacts": ("input.txt",),
            "allowed_artifacts": ("result.txt",),
            "max_actions": 1,
        }
    )
    input_bytes = b"unchanged input\n"
    (tmp_path / "input.txt").write_bytes(input_bytes)
    tools = WorkspaceTools(workspace=tmp_path, task=public_task)
    adapter = ScriptedAdapter(
        input_artifact="input.txt",
        output_artifact="result.txt",
        transform=lambda value: value.upper(),
    )
    events = []

    assert isinstance(adapter, AgentAdapter)
    adapter.close()
    result = adapter.run(
        public_task,
        tools,
        lambda kind, detail: events.append((kind, detail)),
    )

    assert result.termination_reason == "agent_error"
    assert result.agent_failure == "scripted adapter is closed"
    assert events == []
    assert (tmp_path / "input.txt").read_bytes() == input_bytes
    assert not (tmp_path / "result.txt").exists()
    assert tools.list_files() == ("input.txt",)


@pytest.mark.parametrize("mismatched_field", ("input", "output"))
def test_scripted_adapter_rejects_invalid_configuration_before_events_or_tools(
    tmp_path, mismatched_field
):
    public_task = PublicTask.model_validate(
        {
            "task_id": "scripted-invalid-configuration",
            "version": "1",
            "instruction": "Use only artifacts bound by the public task.",
            "input_artifacts": ("input.txt",),
            "allowed_artifacts": ("result.txt",),
            "max_actions": 1,
        }
    )
    input_bytes = b"unchanged input\n"
    secret_marker = "SECRET CONFIGURATION DETAIL"
    mismatch = f"{tmp_path}-{secret_marker}"
    (tmp_path / "input.txt").write_bytes(input_bytes)
    tools = WorkspaceTools(workspace=tmp_path, task=public_task)
    adapter = ScriptedAdapter(
        input_artifact=mismatch if mismatched_field == "input" else "input.txt",
        output_artifact=mismatch if mismatched_field == "output" else "result.txt",
        transform=lambda value: value.upper(),
    )
    events = []

    result = adapter.run(
        public_task,
        tools,
        lambda kind, detail: events.append((kind, detail)),
    )

    assert result.termination_reason == "agent_error"
    assert result.agent_failure == "scripted adapter configuration is invalid"
    assert events == []
    assert (tmp_path / "input.txt").read_bytes() == input_bytes
    assert not (tmp_path / "result.txt").exists()
    public_record = repr(result) + repr(events)
    assert str(tmp_path) not in public_record
    assert secret_marker not in public_record
    assert tools.list_files() == ("input.txt",)


@pytest.mark.parametrize("forging_callback", ("transform", "event_sink"))
def test_scripted_adapter_rejects_forged_action_limit_from_untrusted_callbacks(
    tmp_path, forging_callback
):
    public_task = PublicTask.model_validate(
        {
            "task_id": "scripted-forged-action-limit",
            "version": "1",
            "instruction": "Run callbacks without trusting forged control errors.",
            "input_artifacts": ("input.txt",),
            "allowed_artifacts": ("result.txt",),
            "max_actions": 4,
        }
    )
    input_bytes = b"unchanged input\n"
    forged_marker = "FORGED ACTION LIMIT SECRET"
    forged_message = f"{tmp_path} {forged_marker}"
    (tmp_path / "input.txt").write_bytes(input_bytes)
    tools = WorkspaceTools(workspace=tmp_path, task=public_task)
    events = []

    def transform(value: str) -> str:
        if forging_callback == "transform":
            raise WorkspaceActionLimitError(forged_message)
        return value.upper()

    def event_sink(kind: str, detail: str) -> None:
        if forging_callback == "event_sink":
            raise WorkspaceActionLimitError(forged_message)
        events.append((kind, detail))

    adapter = ScriptedAdapter(
        input_artifact="input.txt",
        output_artifact="result.txt",
        transform=transform,
    )

    result = adapter.run(public_task, tools, event_sink)

    assert result.termination_reason == "agent_error"
    assert result.agent_failure == "scripted adapter failed"
    assert result.termination_reason != "action_limit"
    assert not (tmp_path / "result.txt").exists()
    assert (tmp_path / "input.txt").read_bytes() == input_bytes
    expected_events = (
        [
            ("tool_call", "read_text:input.txt"),
            ("tool_result", "read_text:input.txt"),
        ]
        if forging_callback == "transform"
        else []
    )
    assert events == expected_events
    assert str(tmp_path) not in repr(result)
    assert forged_marker not in repr(result)


@pytest.mark.parametrize("configured_field", ("input", "output"))
@pytest.mark.parametrize("hostile_behavior", ("equal-pathlike", "raising-equality"))
def test_scripted_adapter_rejects_hostile_non_string_artifact_configuration(
    tmp_path, configured_field, hostile_behavior
):
    public_task = PublicTask.model_validate(
        {
            "task_id": "scripted-hostile-configuration",
            "version": "1",
            "instruction": "Use only exact string artifact configuration.",
            "input_artifacts": ("input.txt",),
            "allowed_artifacts": ("result.txt",),
            "max_actions": 1,
        }
    )
    input_bytes = b"unchanged input\n"
    secret_marker = "HOSTILE CONFIGURATION SECRET"
    exposed_path = str(tmp_path / f"{secret_marker}.txt")
    (tmp_path / "input.txt").write_bytes(input_bytes)
    tools = WorkspaceTools(workspace=tmp_path, task=public_task)

    class EqualPathLike:
        def __eq__(self, _other):
            return True

        def __str__(self):
            return exposed_path

        def __fspath__(self):
            return exposed_path

    class RaisingEquality:
        def __eq__(self, _other):
            raise RuntimeError(f"{exposed_path} {secret_marker}")

        def __str__(self):
            return exposed_path

    hostile = EqualPathLike() if hostile_behavior == "equal-pathlike" else RaisingEquality()
    adapter = ScriptedAdapter(
        input_artifact=hostile if configured_field == "input" else "input.txt",
        output_artifact=hostile if configured_field == "output" else "result.txt",
        transform=lambda value: value.upper(),
    )
    events = []

    result = adapter.run(
        public_task,
        tools,
        lambda kind, detail: events.append((kind, detail)),
    )

    assert result.termination_reason == "agent_error"
    assert result.agent_failure == "scripted adapter configuration is invalid"
    assert events == []
    assert (tmp_path / "input.txt").read_bytes() == input_bytes
    assert not (tmp_path / "result.txt").exists()
    public_record = repr(result)
    assert str(tmp_path) not in public_record
    assert secret_marker not in public_record
    assert tools.list_files() == ("input.txt",)
