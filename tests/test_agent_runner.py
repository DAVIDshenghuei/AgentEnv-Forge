import pytest
from pathlib import Path
from threading import Barrier, BrokenBarrierError, Event, Lock, Thread

import agentenv_forge.runner as runner_module
import agentenv_forge.tools.workspace as workspace_module
from agentenv_forge.adapters import AgentRunResult, ScriptedAdapter
from agentenv_forge.runner import ResourceLimitError, run_agent_episode


def test_run_agent_episode_executes_complete_happy_lifecycle(tmp_path):
    def normalize(value: str) -> str:
        lines = (" ".join(line.lower().split()) for line in value.splitlines())
        return "\n".join(line for line in lines if line) + "\n"

    adapter = ScriptedAdapter(
        input_artifact="input.txt",
        output_artifact="result.txt",
        transform=normalize,
    )

    trajectory = run_agent_episode(
        task_id="text-normalization-001",
        adapter=adapter,
        seed=42,
        workspace_root=tmp_path,
    )

    assert trajectory.reward.total == 1.0
    assert trajectory.reward.exact_content == 1.0
    assert trajectory.reward.policy_compliance == 1.0
    assert trajectory.termination_reason == "finished"
    assert trajectory.environment_failure is None
    assert trajectory.agent_failure is None
    assert [artifact.path for artifact in trajectory.artifacts] == ["result.txt"]
    assert [event.sequence for event in trajectory.events] == list(
        range(len(trajectory.events))
    )
    assert [(event.kind, event.detail) for event in trajectory.events] == [
        ("reset", "initial state restored"),
        ("adapter_start", "adapter started"),
        ("tool_call", "read_text:input.txt"),
        ("tool_result", "read_text:input.txt"),
        ("tool_call", "write_text:result.txt"),
        ("tool_result", "write_text:result.txt"),
        ("adapter_stop", "adapter stopped"),
        ("tools_revoked", "workspace tools revoked"),
        ("verify", "deterministic verifier completed"),
    ]


def test_run_agent_episode_sanitizes_adapter_exception_and_completes_lifecycle(
    tmp_path,
):
    secret_marker = "SECRET ADAPTER FAILURE"

    class RaisingAdapter:
        def __init__(self):
            self.close_called = False

        def run(self, task, tools, event_sink):
            raise RuntimeError(f"{tmp_path} {secret_marker}")

        def close(self):
            self.close_called = True

    adapter = RaisingAdapter()

    trajectory = run_agent_episode(
        task_id="text-normalization-001",
        adapter=adapter,
        seed=42,
        workspace_root=tmp_path,
    )

    assert adapter.close_called
    assert trajectory.termination_reason == "agent_error"
    assert trajectory.agent_failure == "adapter execution failed"
    assert trajectory.environment_failure is None
    assert trajectory.reward.artifact_exists == 0.0
    assert trajectory.reward.exact_content == 0.0
    assert trajectory.reward.policy_compliance == 1.0
    assert trajectory.reward.total == 0.1
    assert trajectory.artifacts == []
    assert [event.sequence for event in trajectory.events] == list(
        range(len(trajectory.events))
    )
    assert [(event.kind, event.detail) for event in trajectory.events] == [
        ("reset", "initial state restored"),
        ("adapter_start", "adapter started"),
        ("adapter_failure", "adapter execution failed"),
        ("adapter_stop", "adapter stopped"),
        ("tools_revoked", "workspace tools revoked"),
        ("verify", "deterministic verifier completed"),
    ]
    public_record = trajectory.canonical_content() + repr(trajectory)
    assert str(tmp_path) not in public_record
    assert secret_marker not in public_record


def test_run_agent_episode_sanitizes_adapter_close_exception_and_still_verifies(
    tmp_path,
):
    secret_marker = "SECRET CLOSE FAILURE"

    def normalize(value: str) -> str:
        lines = (" ".join(line.lower().split()) for line in value.splitlines())
        return "\n".join(line for line in lines if line) + "\n"

    class CloseRaisingAdapter(ScriptedAdapter):
        def close(self):
            raise RuntimeError(f"{tmp_path} {secret_marker}")

    adapter = CloseRaisingAdapter(
        input_artifact="input.txt",
        output_artifact="result.txt",
        transform=normalize,
    )

    trajectory = run_agent_episode(
        task_id="text-normalization-001",
        adapter=adapter,
        seed=42,
        workspace_root=tmp_path,
    )

    assert trajectory.reward.total == 1.0
    assert trajectory.reward.exact_content == 1.0
    assert trajectory.reward.policy_compliance == 1.0
    assert [artifact.path for artifact in trajectory.artifacts] == ["result.txt"]
    assert trajectory.termination_reason == "agent_error"
    assert trajectory.agent_failure == "adapter close failed"
    assert trajectory.environment_failure is None
    assert [event.sequence for event in trajectory.events] == list(
        range(len(trajectory.events))
    )
    assert [(event.kind, event.detail) for event in trajectory.events] == [
        ("reset", "initial state restored"),
        ("adapter_start", "adapter started"),
        ("tool_call", "read_text:input.txt"),
        ("tool_result", "read_text:input.txt"),
        ("tool_call", "write_text:result.txt"),
        ("tool_result", "write_text:result.txt"),
        ("adapter_failure", "adapter close failed"),
        ("adapter_stop", "adapter stop failed"),
        ("tools_revoked", "workspace tools revoked"),
        ("verify", "deterministic verifier completed"),
    ]
    public_record = trajectory.canonical_content() + repr(trajectory)
    assert str(tmp_path) not in public_record
    assert secret_marker not in public_record


@pytest.mark.parametrize("reason", ("action_limit", "timeout"))
def test_run_agent_episode_records_returned_nonfinished_adapter_termination(
    tmp_path, reason
):
    class TerminatingAdapter:
        def __init__(self):
            self.close_called = False

        def run(self, task, tools, event_sink):
            return AgentRunResult(reason, None)

        def close(self):
            self.close_called = True

    adapter = TerminatingAdapter()

    trajectory = run_agent_episode(
        task_id="text-normalization-001",
        adapter=adapter,
        seed=42,
        workspace_root=tmp_path,
    )

    assert adapter.close_called
    assert trajectory.termination_reason == reason
    assert trajectory.agent_failure is None
    assert trajectory.environment_failure is None
    assert trajectory.reward.total == 0.1
    assert trajectory.reward.policy_compliance == 1.0
    assert trajectory.artifacts == []
    assert [event.sequence for event in trajectory.events] == list(
        range(len(trajectory.events))
    )
    assert [(event.kind, event.detail) for event in trajectory.events] == [
        ("reset", "initial state restored"),
        ("adapter_start", "adapter started"),
        ("adapter_termination", reason),
        ("adapter_stop", "adapter stopped"),
        ("tools_revoked", "workspace tools revoked"),
        ("verify", "deterministic verifier completed"),
    ]


def test_run_agent_episode_sanitizes_normally_returned_agent_error(tmp_path):
    secret_marker = "SECRET REPORTED FAILURE"

    class ReportingAdapter:
        def __init__(self):
            self.close_called = False

        def run(self, task, tools, event_sink):
            return AgentRunResult("agent_error", f"{tmp_path} {secret_marker}")

        def close(self):
            self.close_called = True

    adapter = ReportingAdapter()

    trajectory = run_agent_episode(
        task_id="text-normalization-001",
        adapter=adapter,
        seed=42,
        workspace_root=tmp_path,
    )

    assert adapter.close_called
    assert trajectory.termination_reason == "agent_error"
    assert trajectory.agent_failure == "adapter reported failure"
    assert trajectory.environment_failure is None
    assert trajectory.reward.total == 0.1
    assert trajectory.reward.policy_compliance == 1.0
    assert trajectory.artifacts == []
    assert [event.sequence for event in trajectory.events] == list(
        range(len(trajectory.events))
    )
    assert [(event.kind, event.detail) for event in trajectory.events] == [
        ("reset", "initial state restored"),
        ("adapter_start", "adapter started"),
        ("adapter_failure", "adapter reported failure"),
        ("adapter_stop", "adapter stopped"),
        ("tools_revoked", "workspace tools revoked"),
        ("verify", "deterministic verifier completed"),
    ]
    public_record = trajectory.canonical_content() + repr(trajectory)
    assert str(tmp_path) not in public_record
    assert secret_marker not in public_record


@pytest.mark.parametrize(
    "invalid_case",
    ("plain-object", "invalid-reason", "finished-with-failure", "limit-with-failure"),
)
def test_run_agent_episode_sanitizes_invalid_adapter_return_records(
    tmp_path, invalid_case
):
    secret_marker = "SECRET INVALID RESULT"
    secret_failure = f"{tmp_path} {secret_marker}"
    if invalid_case == "plain-object":
        returned_result = object()
    elif invalid_case == "invalid-reason":
        returned_result = AgentRunResult(secret_failure, None)
    elif invalid_case == "finished-with-failure":
        returned_result = AgentRunResult("finished", secret_failure)
    else:
        returned_result = AgentRunResult("action_limit", secret_failure)

    class InvalidResultAdapter:
        def __init__(self):
            self.close_called = False

        def run(self, task, tools, event_sink):
            return returned_result

        def close(self):
            self.close_called = True

    adapter = InvalidResultAdapter()

    trajectory = run_agent_episode(
        task_id="text-normalization-001",
        adapter=adapter,
        seed=42,
        workspace_root=tmp_path,
    )

    assert adapter.close_called
    assert trajectory.termination_reason == "agent_error"
    assert trajectory.agent_failure == "adapter returned invalid result"
    assert trajectory.environment_failure is None
    assert trajectory.reward.total == 0.1
    assert trajectory.reward.policy_compliance == 1.0
    assert trajectory.artifacts == []
    assert [event.sequence for event in trajectory.events] == list(
        range(len(trajectory.events))
    )
    assert [(event.kind, event.detail) for event in trajectory.events] == [
        ("reset", "initial state restored"),
        ("adapter_start", "adapter started"),
        ("adapter_failure", "adapter returned invalid result"),
        ("adapter_stop", "adapter stopped"),
        ("tools_revoked", "workspace tools revoked"),
        ("verify", "deterministic verifier completed"),
    ]
    public_record = trajectory.canonical_content() + repr(trajectory)
    assert str(tmp_path) not in public_record
    assert secret_marker not in public_record


@pytest.mark.parametrize(
    "payload_kind",
    ("none", "integer", "empty", "string-subclass", "oversized-string"),
)
def test_run_agent_episode_rejects_malformed_agent_error_failure_payloads(
    tmp_path, payload_kind
):
    secret_marker = "SECRET MALFORMED RESULT"

    class FailureString(str):
        pass

    if payload_kind == "none":
        failure_payload = None
    elif payload_kind == "integer":
        failure_payload = 7
    elif payload_kind == "empty":
        failure_payload = ""
    elif payload_kind == "string-subclass":
        failure_payload = FailureString(f"{tmp_path} {secret_marker}")
    else:
        prefix = f"{tmp_path} {secret_marker}"
        failure_payload = prefix + "x" * (257 - len(prefix.encode("utf-8")))
        assert type(failure_payload) is str
        assert len(failure_payload.encode("utf-8")) == 257

    class MalformedAgentErrorAdapter:
        def __init__(self):
            self.close_called = False

        def run(self, task, tools, event_sink):
            return AgentRunResult("agent_error", failure_payload)

        def close(self):
            self.close_called = True

    adapter = MalformedAgentErrorAdapter()

    trajectory = run_agent_episode(
        task_id="text-normalization-001",
        adapter=adapter,
        seed=42,
        workspace_root=tmp_path,
    )

    assert adapter.close_called
    assert trajectory.termination_reason == "agent_error"
    assert trajectory.agent_failure == "adapter returned invalid result"
    assert trajectory.environment_failure is None
    assert trajectory.reward.total == 0.1
    assert trajectory.reward.policy_compliance == 1.0
    assert trajectory.artifacts == []
    assert [event.sequence for event in trajectory.events] == list(
        range(len(trajectory.events))
    )
    assert [(event.kind, event.detail) for event in trajectory.events] == [
        ("reset", "initial state restored"),
        ("adapter_start", "adapter started"),
        ("adapter_failure", "adapter returned invalid result"),
        ("adapter_stop", "adapter stopped"),
        ("tools_revoked", "workspace tools revoked"),
        ("verify", "deterministic verifier completed"),
    ]
    public_record = trajectory.canonical_content() + repr(trajectory)
    assert str(tmp_path) not in public_record
    assert secret_marker not in public_record


def test_run_agent_episode_separates_sanitized_verifier_resource_failure(
    tmp_path, monkeypatch
):
    secret_marker = "SECRET VERIFY FAILURE"

    def normalize(value: str) -> str:
        lines = (" ".join(line.lower().split()) for line in value.splitlines())
        return "\n".join(line for line in lines if line) + "\n"

    def fail_verification(*_args, **_kwargs):
        raise ResourceLimitError(f"{tmp_path} {secret_marker}")

    monkeypatch.setattr(runner_module, "_verify", fail_verification)
    adapter = ScriptedAdapter(
        input_artifact="input.txt",
        output_artifact="result.txt",
        transform=normalize,
    )

    trajectory = runner_module.run_agent_episode(
        task_id="text-normalization-001",
        adapter=adapter,
        seed=42,
        workspace_root=tmp_path,
    )

    assert trajectory.termination_reason == "resource_limit"
    assert trajectory.environment_failure == "verification resource limit exceeded"
    assert trajectory.agent_failure is None
    assert trajectory.reward.artifact_exists == 0.0
    assert trajectory.reward.exact_content == 0.0
    assert trajectory.reward.policy_compliance == 0.0
    assert trajectory.reward.total == 0.0
    assert trajectory.artifacts == []
    assert [event.sequence for event in trajectory.events] == list(
        range(len(trajectory.events))
    )
    assert [(event.kind, event.detail) for event in trajectory.events] == [
        ("reset", "initial state restored"),
        ("adapter_start", "adapter started"),
        ("tool_call", "read_text:input.txt"),
        ("tool_result", "read_text:input.txt"),
        ("tool_call", "write_text:result.txt"),
        ("tool_result", "write_text:result.txt"),
        ("adapter_stop", "adapter stopped"),
        ("tools_revoked", "workspace tools revoked"),
        ("verify_failed", "resource_limit"),
    ]
    public_record = trajectory.canonical_content() + repr(trajectory)
    assert str(tmp_path) not in public_record
    assert secret_marker not in public_record


def test_run_agent_episode_separates_sanitized_generic_verifier_failure(
    tmp_path, monkeypatch
):
    secret_marker = "SECRET GENERIC VERIFY"

    def normalize(value: str) -> str:
        lines = (" ".join(line.lower().split()) for line in value.splitlines())
        return "\n".join(line for line in lines if line) + "\n"

    def fail_verification(*_args, **_kwargs):
        raise RuntimeError(f"{tmp_path} {secret_marker}")

    monkeypatch.setattr(runner_module, "_verify", fail_verification)
    adapter = ScriptedAdapter(
        input_artifact="input.txt",
        output_artifact="result.txt",
        transform=normalize,
    )

    trajectory = runner_module.run_agent_episode(
        task_id="text-normalization-001",
        adapter=adapter,
        seed=42,
        workspace_root=tmp_path,
    )

    assert trajectory.termination_reason == "environment_error"
    assert trajectory.environment_failure == "verification failed"
    assert trajectory.agent_failure is None
    assert trajectory.reward.artifact_exists == 0.0
    assert trajectory.reward.exact_content == 0.0
    assert trajectory.reward.policy_compliance == 0.0
    assert trajectory.reward.total == 0.0
    assert trajectory.artifacts == []
    assert [event.sequence for event in trajectory.events] == list(
        range(len(trajectory.events))
    )
    assert [(event.kind, event.detail) for event in trajectory.events] == [
        ("reset", "initial state restored"),
        ("adapter_start", "adapter started"),
        ("tool_call", "read_text:input.txt"),
        ("tool_result", "read_text:input.txt"),
        ("tool_call", "write_text:result.txt"),
        ("tool_result", "write_text:result.txt"),
        ("adapter_stop", "adapter stopped"),
        ("tools_revoked", "workspace tools revoked"),
        ("verify_failed", "environment_error"),
    ]
    public_record = trajectory.canonical_content() + repr(trajectory)
    assert str(tmp_path) not in public_record
    assert secret_marker not in public_record


def test_run_agent_episode_rejects_unsafe_adapter_event_metadata(tmp_path):
    secret_marker = "SECRET EVENT PAYLOAD"

    class UnsafeEventAdapter:
        def __init__(self):
            self.close_called = False

        def run(self, task, tools, event_sink):
            event_sink("tool_call", f"read_text:{tmp_path} {secret_marker}")
            return AgentRunResult("finished", None)

        def close(self):
            self.close_called = True

    adapter = UnsafeEventAdapter()

    trajectory = run_agent_episode(
        task_id="text-normalization-001",
        adapter=adapter,
        seed=42,
        workspace_root=tmp_path,
    )

    assert adapter.close_called
    assert trajectory.termination_reason == "agent_error"
    assert trajectory.agent_failure == "adapter execution failed"
    assert trajectory.environment_failure is None
    assert trajectory.reward.total == 0.1
    assert trajectory.reward.policy_compliance == 1.0
    assert trajectory.artifacts == []
    assert [event.sequence for event in trajectory.events] == list(
        range(len(trajectory.events))
    )
    assert [(event.kind, event.detail) for event in trajectory.events] == [
        ("reset", "initial state restored"),
        ("adapter_start", "adapter started"),
        ("adapter_failure", "adapter execution failed"),
        ("adapter_stop", "adapter stopped"),
        ("tools_revoked", "workspace tools revoked"),
        ("verify", "deterministic verifier completed"),
    ]
    public_record = trajectory.canonical_content() + repr(trajectory)
    assert str(tmp_path) not in public_record
    assert secret_marker not in public_record


def test_run_agent_episode_revokes_run_capabilities_before_adapter_close(tmp_path):
    class RetainingAdapter:
        def __init__(self):
            self.retained_tools = None
            self.retained_event_sink = None
            self.event_denied = False
            self.tool_denied = False

        def run(self, task, tools, event_sink):
            self.retained_tools = tools
            self.retained_event_sink = event_sink
            return AgentRunResult("finished", None)

        def close(self):
            try:
                self.retained_event_sink("tool_call", "list_files")
            except ValueError:
                self.event_denied = True
            try:
                self.retained_tools.write_text(
                    "result.txt", "post-close mutation\n"
                )
            except ValueError:
                self.tool_denied = True

    adapter = RetainingAdapter()

    trajectory = run_agent_episode(
        task_id="text-normalization-001",
        adapter=adapter,
        seed=42,
        workspace_root=tmp_path,
    )

    assert adapter.event_denied
    assert adapter.tool_denied
    assert trajectory.termination_reason == "finished"
    assert trajectory.agent_failure is None
    assert trajectory.environment_failure is None
    assert trajectory.artifacts == []
    assert trajectory.reward.total == 0.1
    assert trajectory.reward.policy_compliance == 1.0
    assert [(event.kind, event.detail) for event in trajectory.events] == [
        ("reset", "initial state restored"),
        ("adapter_start", "adapter started"),
        ("adapter_stop", "adapter stopped"),
        ("tools_revoked", "workspace tools revoked"),
        ("verify", "deterministic verifier completed"),
    ]
    with pytest.raises(ValueError, match="^workspace tools revoked$"):
        adapter.retained_tools.list_files()


def test_run_agent_episode_integrates_workspace_action_limit_lifecycle(
    tmp_path, monkeypatch
):
    task = runner_module.load_task("text-normalization-001").model_copy(
        update={"max_actions": 1}
    )
    monkeypatch.setattr(runner_module, "load_task", lambda _task_id: task)

    def normalize(value: str) -> str:
        lines = (" ".join(line.lower().split()) for line in value.splitlines())
        return "\n".join(line for line in lines if line) + "\n"

    adapter = ScriptedAdapter(
        input_artifact="input.txt",
        output_artifact="result.txt",
        transform=normalize,
    )

    trajectory = runner_module.run_agent_episode(
        task_id="text-normalization-001",
        adapter=adapter,
        seed=42,
        workspace_root=tmp_path,
    )

    assert trajectory.termination_reason == "action_limit"
    assert trajectory.agent_failure is None
    assert trajectory.environment_failure is None
    assert trajectory.artifacts == []
    assert trajectory.reward.total == 0.1
    assert trajectory.reward.policy_compliance == 1.0
    assert [(event.kind, event.detail) for event in trajectory.events] == [
        ("reset", "initial state restored"),
        ("adapter_start", "adapter started"),
        ("tool_call", "read_text:input.txt"),
        ("tool_result", "read_text:input.txt"),
        ("tool_call", "write_text:result.txt"),
        ("adapter_termination", "action_limit"),
        ("adapter_stop", "adapter stopped"),
        ("tools_revoked", "workspace tools revoked"),
        ("verify", "deterministic verifier completed"),
    ]


def test_run_agent_episode_canonical_replay_is_deterministic(tmp_path):
    def normalize(value: str) -> str:
        lines = (" ".join(line.lower().split()) for line in value.splitlines())
        return "\n".join(line for line in lines if line) + "\n"

    def new_adapter():
        return ScriptedAdapter(
            input_artifact="input.txt",
            output_artifact="result.txt",
            transform=normalize,
        )

    first = run_agent_episode(
        task_id="text-normalization-001",
        adapter=new_adapter(),
        seed=42,
        workspace_root=tmp_path / "first",
    )
    second = run_agent_episode(
        task_id="text-normalization-001",
        adapter=new_adapter(),
        seed=42,
        workspace_root=tmp_path / "second",
    )

    assert first.reward.total == 1.0
    assert second.reward.total == 1.0
    assert first.canonical_content() == second.canonical_content()


def test_run_agent_episode_cleans_up_capabilities_before_reraising_base_exception(
    tmp_path,
):
    class CancellingAdapter:
        def __init__(self):
            self.retained_tools = None
            self.retained_event_sink = None
            self.close_called = False
            self.event_denied = False
            self.tool_denied = False

        def run(self, task, tools, event_sink):
            self.retained_tools = tools
            self.retained_event_sink = event_sink
            raise KeyboardInterrupt("cancel")

        def close(self):
            self.close_called = True
            try:
                self.retained_event_sink("tool_call", "list_files")
            except ValueError:
                self.event_denied = True
            try:
                self.retained_tools.write_text("result.txt", "cancel mutation\n")
            except ValueError:
                self.tool_denied = True

    adapter = CancellingAdapter()

    with pytest.raises(KeyboardInterrupt, match="^cancel$"):
        run_agent_episode(
            task_id="text-normalization-001",
            adapter=adapter,
            seed=42,
            workspace_root=tmp_path,
        )

    assert adapter.close_called
    assert adapter.event_denied
    assert adapter.tool_denied
    with pytest.raises(ValueError, match="^workspace tools revoked$"):
        adapter.retained_tools.list_files()
    with pytest.raises(ValueError, match="^adapter events are inactive$"):
        adapter.retained_event_sink("tool_call", "list_files")


def test_run_agent_episode_does_not_expose_raw_workspace_path_to_adapter(tmp_path):
    class WorkspaceProbingAdapter:
        def __init__(self):
            self.hidden = False
            self.exposed_path = None
            self.received_tools = None
            self.close_called = False

        def run(self, task, tools, event_sink):
            self.received_tools = tools
            try:
                self.exposed_path = tools._workspace
            except AttributeError:
                self.hidden = True
            else:
                (Path(self.exposed_path) / "rogue.txt").write_text(
                    "undeclared mutation", encoding="utf-8"
                )
            return AgentRunResult("finished", None)

        def close(self):
            self.close_called = True

    adapter = WorkspaceProbingAdapter()

    trajectory = run_agent_episode(
        task_id="text-normalization-001",
        adapter=adapter,
        seed=42,
        workspace_root=tmp_path,
    )

    assert adapter.hidden
    assert adapter.exposed_path is None
    assert adapter.close_called
    assert not hasattr(adapter.received_tools, "__dict__")
    assert trajectory.termination_reason == "finished"
    assert trajectory.agent_failure is None
    assert trajectory.environment_failure is None
    assert trajectory.artifacts == []
    assert trajectory.reward.policy_compliance == 1.0
    assert trajectory.reward.total == 0.1
    assert [(event.kind, event.detail) for event in trajectory.events] == [
        ("reset", "initial state restored"),
        ("adapter_start", "adapter started"),
        ("adapter_stop", "adapter stopped"),
        ("tools_revoked", "workspace tools revoked"),
        ("verify", "deterministic verifier completed"),
    ]
    public_record = trajectory.canonical_content() + repr(trajectory)
    assert str(tmp_path) not in public_record


def test_run_agent_episode_rejects_fabricated_unbounded_tool_events(tmp_path):
    class FabricatingAdapter:
        def __init__(self):
            self.close_called = False

        def run(self, task, tools, event_sink):
            for _ in range(1_000):
                event_sink("tool_call", "list_files")
                event_sink("tool_result", "list_files")
            return AgentRunResult("finished", None)

        def close(self):
            self.close_called = True

    adapter = FabricatingAdapter()

    trajectory = run_agent_episode(
        task_id="text-normalization-001",
        adapter=adapter,
        seed=42,
        workspace_root=tmp_path,
    )

    assert adapter.close_called
    assert trajectory.termination_reason == "agent_error"
    assert trajectory.agent_failure == "adapter execution failed"
    assert trajectory.environment_failure is None
    assert trajectory.artifacts == []
    assert trajectory.reward.total == 0.1
    assert trajectory.reward.policy_compliance == 1.0
    assert len(trajectory.events) == 6
    assert [(event.kind, event.detail) for event in trajectory.events] == [
        ("reset", "initial state restored"),
        ("adapter_start", "adapter started"),
        ("adapter_failure", "adapter execution failed"),
        ("adapter_stop", "adapter stopped"),
        ("tools_revoked", "workspace tools revoked"),
        ("verify", "deterministic verifier completed"),
    ]


def test_run_agent_episode_rejects_return_while_tool_operation_is_in_flight(
    tmp_path, monkeypatch
):
    write_entered = Event()
    release_write = Event()
    revoke_entered = Event()
    controller_errors = []
    original_write = workspace_module._write_workspace_text
    original_revoke = workspace_module.WorkspaceTools.revoke

    def blocking_write(workspace, relative, content):
        if relative == "result.txt":
            write_entered.set()
            if not release_write.wait(5):
                raise TimeoutError("test write release timed out")
        original_write(workspace, relative, content)

    def observing_revoke(tools):
        revoke_entered.set()
        return original_revoke(tools)

    monkeypatch.setattr(workspace_module, "_write_workspace_text", blocking_write)
    monkeypatch.setattr(workspace_module.WorkspaceTools, "revoke", observing_revoke)

    def release_on_revoke():
        try:
            if not revoke_entered.wait(5):
                raise TimeoutError("test revoke entry timed out")
        except BaseException as error:
            controller_errors.append(error)
        finally:
            release_write.set()

    controller = Thread(target=release_on_revoke, daemon=True)
    controller.start()

    class EarlyReturningAdapter:
        def __init__(self):
            self.retained_tools = None
            self.writer = None
            self.write_errors = []
            self.result_errors = []
            self.close_called = False
            self.writer_alive_at_close = None

        def run(self, task, tools, event_sink):
            self.retained_tools = tools
            event_sink("tool_call", "write_text:result.txt")

            def write_target():
                try:
                    tools.write_text(
                        "result.txt", "agentenv forge\ncausal evaluation\n"
                    )
                except BaseException as error:
                    self.write_errors.append(error)
                    return
                try:
                    event_sink("tool_result", "write_text:result.txt")
                except ValueError as error:
                    self.result_errors.append(error)

            self.writer = Thread(target=write_target, daemon=True)
            self.writer.start()
            if not write_entered.wait(2):
                raise TimeoutError("test write entry timed out")
            return AgentRunResult("finished", None)

        def close(self):
            self.close_called = True
            self.writer_alive_at_close = self.writer.is_alive()

    adapter = EarlyReturningAdapter()
    trajectories = []
    episode_errors = []
    episode_done = Event()

    def run_episode_target():
        try:
            trajectories.append(
                run_agent_episode(
                    task_id="text-normalization-001",
                    adapter=adapter,
                    seed=42,
                    workspace_root=tmp_path,
                )
            )
        except BaseException as error:
            episode_errors.append(error)
        finally:
            episode_done.set()

    episode = Thread(target=run_episode_target, daemon=True)
    episode.start()
    episode_finished = False
    try:
        episode_finished = episode_done.wait(8)
    finally:
        release_write.set()
        episode.join(2)
        controller.join(2)
        if adapter.writer is not None:
            adapter.writer.join(2)

    assert episode_finished
    assert not episode.is_alive()
    assert not controller.is_alive()
    assert adapter.writer is not None and not adapter.writer.is_alive()
    assert episode_errors == []
    assert len(trajectories) == 1
    trajectory = trajectories[0]
    assert controller_errors == []
    assert trajectory.termination_reason == "agent_error"
    assert trajectory.agent_failure == "adapter returned invalid result"
    assert trajectory.environment_failure is None
    assert adapter.write_errors == []
    assert len(adapter.result_errors) == 1
    assert str(adapter.result_errors[0]) == "adapter events are inactive"
    assert adapter.close_called
    assert adapter.writer_alive_at_close is False
    assert trajectory.reward.total == 1.0
    assert [artifact.path for artifact in trajectory.artifacts] == ["result.txt"]
    assert [(event.kind, event.detail) for event in trajectory.events] == [
        ("reset", "initial state restored"),
        ("adapter_start", "adapter started"),
        ("tool_call", "write_text:result.txt"),
        ("adapter_failure", "adapter returned invalid result"),
        ("adapter_stop", "adapter stopped"),
        ("tools_revoked", "workspace tools revoked"),
        ("verify", "deterministic verifier completed"),
    ]
    with pytest.raises(ValueError, match="^workspace tools revoked$"):
        adapter.retained_tools.list_files()


def test_run_agent_episode_consumes_one_pending_intent_only_once(
    tmp_path, monkeypatch
):
    tool_call_barrier = Barrier(2)
    original_event = runner_module.Event

    def synchronized_event(*args, **kwargs):
        kind = kwargs.get("kind")
        if kind == "tool_call":
            try:
                tool_call_barrier.wait(timeout=0.25)
            except BrokenBarrierError:
                pass
        return original_event(*args, **kwargs)

    write_count = 0
    write_count_lock = Lock()
    original_write = workspace_module._write_workspace_text

    def counting_write(workspace, relative, content):
        nonlocal write_count
        if relative == "result.txt":
            with write_count_lock:
                write_count += 1
        return original_write(workspace, relative, content)

    monkeypatch.setattr(runner_module, "Event", synchronized_event)
    monkeypatch.setattr(workspace_module, "_write_workspace_text", counting_write)

    class ConcurrentWriterAdapter:
        def __init__(self):
            self.threads = []
            self.errors = []
            self.close_called = False

        def run(self, task, tools, event_sink):
            event_sink("tool_call", "write_text:result.txt")
            start = Barrier(3)

            def write_target():
                try:
                    start.wait(timeout=2)
                    tools.write_text(
                        "result.txt", "agentenv forge\ncausal evaluation\n"
                    )
                    event_sink("tool_result", "write_text:result.txt")
                except BaseException as error:
                    self.errors.append(error)

            self.threads = [
                Thread(target=write_target, daemon=True),
                Thread(target=write_target, daemon=True),
            ]
            for thread in self.threads:
                thread.start()
            try:
                start.wait(timeout=2)
                for thread in self.threads:
                    thread.join(3)
                if any(thread.is_alive() for thread in self.threads):
                    raise RuntimeError("concurrent writer test thread remained alive")
                return AgentRunResult("finished", None)
            finally:
                for thread in self.threads:
                    thread.join(3)

        def close(self):
            self.close_called = True

    adapter = ConcurrentWriterAdapter()
    try:
        trajectory = run_agent_episode(
            task_id="text-normalization-001",
            adapter=adapter,
            seed=42,
            workspace_root=tmp_path,
        )
    finally:
        for thread in adapter.threads:
            thread.join(3)

    assert all(not thread.is_alive() for thread in adapter.threads)
    assert write_count == 1
    assert len(adapter.errors) == 1
    assert type(adapter.errors[0]) is ValueError
    assert str(adapter.errors[0]) == "invalid adapter event"
    assert adapter.close_called
    assert trajectory.termination_reason == "finished"
    assert trajectory.agent_failure is None
    assert trajectory.environment_failure is None
    assert trajectory.reward.total == 1.0
    assert [(event.kind, event.detail) for event in trajectory.events] == [
        ("reset", "initial state restored"),
        ("adapter_start", "adapter started"),
        ("tool_call", "write_text:result.txt"),
        ("tool_result", "write_text:result.txt"),
        ("adapter_stop", "adapter stopped"),
        ("tools_revoked", "workspace tools revoked"),
        ("verify", "deterministic verifier completed"),
    ]
