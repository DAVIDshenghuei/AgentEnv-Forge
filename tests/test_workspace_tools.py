import inspect
import os
import subprocess
from threading import Event, Thread

import pytest

import agentenv_forge.tools.workspace as workspace_module
from agentenv_forge.runner import MAX_FILE_BYTES, ResourceLimitError
from agentenv_forge.schemas import PublicTask, TaskSpec
from agentenv_forge.tools import WorkspaceProtocol, WorkspaceTools


def test_workspace_tools_implements_narrow_public_workspace_protocol(tmp_path):
    public_task = PublicTask.model_validate(
        {
            "task_id": "workspace-protocol",
            "version": "1",
            "instruction": "Use the narrow public workspace interface.",
            "input_artifacts": ("source.txt",),
            "allowed_artifacts": ("result.txt",),
            "max_actions": 8,
        }
    )
    tools = WorkspaceTools(workspace=tmp_path, task=public_task)

    assert isinstance(tools, WorkspaceProtocol)
    public_methods = {
        name
        for name, method in inspect.getmembers(
            WorkspaceProtocol, predicate=inspect.isfunction
        )
        if not name.startswith("_")
    }
    assert public_methods == {"list_files", "read_text", "write_text"}


@pytest.mark.parametrize("workspace_kind", ("missing", "regular-file"))
def test_workspace_tools_constructor_reports_sanitized_unavailable_workspace(
    tmp_path, workspace_kind
):
    public_task = PublicTask.model_validate(
        {
            "task_id": "workspace-constructor",
            "version": "1",
            "instruction": "Use the available workspace.",
            "input_artifacts": ("source.txt",),
            "allowed_artifacts": ("result.txt",),
            "max_actions": 8,
        }
    )
    workspace = tmp_path / f"{workspace_kind}-workspace"
    if workspace_kind == "regular-file":
        workspace.write_text("not a directory", encoding="utf-8")

    with pytest.raises(ValueError, match="^workspace is unavailable$") as failure:
        WorkspaceTools(workspace=workspace, task=public_task)

    message = str(failure.value)
    assert str(workspace) not in message
    assert str(tmp_path) not in message


def test_list_files_reports_sanitized_failure_and_consumes_action_when_root_vanishes(
    tmp_path,
):
    public_task = PublicTask.model_validate(
        {
            "task_id": "workspace-listing-failure",
            "version": "1",
            "instruction": "List the declared workspace files.",
            "input_artifacts": ("source.txt",),
            "allowed_artifacts": ("result.txt",),
            "max_actions": 1,
        }
    )
    tools = WorkspaceTools(workspace=tmp_path, task=public_task)
    tmp_path.rmdir()

    with pytest.raises(ValueError, match="^workspace listing failed$") as failure:
        tools.list_files()

    message = str(failure.value)
    assert str(tmp_path) not in message
    assert "source.txt" not in message
    with pytest.raises(ValueError, match="^workspace action budget exhausted$"):
        tools.list_files()


@pytest.mark.parametrize(
    ("failure_kind", "content"),
    (
        pytest.param("invalid-utf8", "\ud800", id="unpaired-surrogate"),
        pytest.param(
            "missing-root",
            "ATTEMPTED OUTPUT CONTENT",
            id="workspace-removed",
        ),
    ),
)
def test_write_text_reports_sanitized_filesystem_failures_and_consumes_action(
    tmp_path, failure_kind, content
):
    relative = "result.txt"
    public_task = PublicTask.model_validate(
        {
            "task_id": "workspace-write-failure",
            "version": "1",
            "instruction": "Write the declared output.",
            "input_artifacts": ("source.txt",),
            "allowed_artifacts": (relative,),
            "max_actions": 1,
        }
    )
    tools = WorkspaceTools(workspace=tmp_path, task=public_task)
    if failure_kind == "missing-root":
        tmp_path.rmdir()

    with pytest.raises(ValueError, match="^output could not be written$") as failure:
        tools.write_text(relative, content)

    message = str(failure.value)
    assert str(tmp_path) not in message
    assert relative not in message
    assert content not in message
    assert not (tmp_path / relative).exists()
    with pytest.raises(ValueError, match="^workspace action budget exhausted$"):
        tools.list_files()


def test_list_files_returns_only_existing_declared_public_paths(tmp_path):
    task = TaskSpec.model_validate(
        {
            "task_id": "workspace-listing",
            "version": "1",
            "instruction": "Inspect the declared workspace artifacts.",
            "max_actions": 8,
            "split": "train",
            "initial_files": [
                {"path": "input.txt", "content": "primary input"},
                {"path": "context.txt", "content": "public context"},
            ],
            "input_artifact": "input.txt",
            "expected_artifact": "result.txt",
            "expected_content": "result",
            "allowed_artifacts": ["result.txt"],
        }
    )
    public_task = task.to_public_task()

    for initial_file in task.initial_files:
        (tmp_path / initial_file.path).write_text(
            initial_file.content, encoding="utf-8"
        )
    (tmp_path / "result.txt").write_text("result", encoding="utf-8")
    (tmp_path / "undeclared-secret.txt").write_text("secret", encoding="utf-8")

    tools = WorkspaceTools(workspace=tmp_path, task=public_task)

    listed_files = tools.list_files()

    assert listed_files == ("context.txt", "input.txt", "result.txt")
    assert "undeclared-secret.txt" not in listed_files
    assert all(str(tmp_path) not in path for path in listed_files)


def test_read_text_preserves_exact_utf8_text(tmp_path):
    public_task = PublicTask.model_validate(
        {
            "task_id": "workspace-reading",
            "version": "1",
            "instruction": "Read the declared source text.",
            "input_artifacts": ("nested/source.txt",),
            "allowed_artifacts": ("result.txt",),
            "max_actions": 8,
        }
    )
    expected = "café\r\n東京\n"
    source = tmp_path / "nested" / "source.txt"
    source.parent.mkdir()
    source.write_bytes(expected.encode("utf-8"))
    tools = WorkspaceTools(workspace=tmp_path, task=public_task)

    content = tools.read_text("nested/source.txt")

    assert content == expected


@pytest.mark.parametrize(
    ("path", "content"),
    (
        pytest.param("result.txt", "ALLOWED OUTPUT SECRET", id="allowed-output"),
        pytest.param("private.txt", "UNDECLARED SECRET", id="undeclared"),
    ),
)
def test_read_text_rejects_paths_not_declared_as_public_inputs(
    tmp_path, path, content
):
    public_task = PublicTask.model_validate(
        {
            "task_id": "workspace-read-authorization",
            "version": "1",
            "instruction": "Read only declared public inputs.",
            "input_artifacts": ("source.txt",),
            "allowed_artifacts": ("result.txt",),
            "max_actions": 8,
        }
    )
    (tmp_path / path).write_text(content, encoding="utf-8")
    tools = WorkspaceTools(workspace=tmp_path, task=public_task)

    with pytest.raises(ValueError) as failure:
        tools.read_text(path)

    message = str(failure.value)
    assert content not in message
    assert str(tmp_path) not in message


def test_write_text_creates_declared_nested_output_with_exact_utf8_bytes(tmp_path):
    public_task = PublicTask.model_validate(
        {
            "task_id": "workspace-writing",
            "version": "1",
            "instruction": "Write the result to the declared output.",
            "input_artifacts": ("source.txt",),
            "allowed_artifacts": ("nested/result.txt",),
            "max_actions": 8,
        }
    )
    content = "café\r\n東京\n"
    tools = WorkspaceTools(workspace=tmp_path, task=public_task)

    tools.write_text("nested/result.txt", content)

    result = tmp_path / "nested" / "result.txt"
    assert result.is_file()
    assert result.read_bytes() == content.encode("utf-8")


@pytest.mark.parametrize(
    ("path", "existing_content"),
    (
        pytest.param("source.txt", b"ORIGINAL INPUT", id="declared-input"),
        pytest.param("private.txt", None, id="undeclared"),
    ),
)
def test_write_text_rejects_paths_not_declared_as_allowed_outputs(
    tmp_path, path, existing_content
):
    public_task = PublicTask.model_validate(
        {
            "task_id": "workspace-write-authorization",
            "version": "1",
            "instruction": "Write only declared output artifacts.",
            "input_artifacts": ("source.txt",),
            "allowed_artifacts": ("result.txt",),
            "max_actions": 8,
        }
    )
    target = tmp_path / path
    if existing_content is not None:
        target.write_bytes(existing_content)
    attempted_content = "ATTEMPTED SECRET CONTENT"
    tools = WorkspaceTools(workspace=tmp_path, task=public_task)

    with pytest.raises(ValueError) as failure:
        tools.write_text(path, attempted_content)

    if existing_content is None:
        assert not target.exists()
    else:
        assert target.read_bytes() == existing_content
    message = str(failure.value)
    assert attempted_content not in message
    assert str(tmp_path) not in message


def test_workspace_tools_share_one_action_budget_across_operations(tmp_path):
    public_task = PublicTask.model_validate(
        {
            "task_id": "workspace-action-budget",
            "version": "1",
            "instruction": "Use the declared workspace within the action budget.",
            "input_artifacts": ("source.txt",),
            "allowed_artifacts": ("result.txt",),
            "max_actions": 3,
        }
    )
    source_content = "declared input"
    result_content = "written output"
    (tmp_path / "source.txt").write_text(source_content, encoding="utf-8")
    tools = WorkspaceTools(workspace=tmp_path, task=public_task)

    assert tools.list_files() == ("source.txt",)
    assert tools.read_text("source.txt") == source_content
    tools.write_text("result.txt", result_content)
    assert (tmp_path / "result.txt").read_text(encoding="utf-8") == result_content

    with pytest.raises(ValueError, match="^workspace action budget exhausted$") as failure:
        tools.list_files()

    message = str(failure.value)
    assert str(tmp_path) not in message
    assert source_content not in message
    assert result_content not in message


@pytest.mark.parametrize("operation", ("list_files", "read_text", "write_text"))
def test_revoked_workspace_tools_reject_every_operation(tmp_path, operation):
    public_task = PublicTask.model_validate(
        {
            "task_id": "workspace-revocation",
            "version": "1",
            "instruction": "Use the workspace only while access is active.",
            "input_artifacts": ("source.txt",),
            "allowed_artifacts": ("result.txt",),
            "max_actions": 8,
        }
    )
    source_content = "declared input content"
    attempted_content = "attempted output content"
    (tmp_path / "source.txt").write_text(source_content, encoding="utf-8")
    tools = WorkspaceTools(workspace=tmp_path, task=public_task)

    tools.revoke()
    tools.revoke()

    with pytest.raises(ValueError, match="^workspace tools revoked$") as failure:
        if operation == "list_files":
            tools.list_files()
        elif operation == "read_text":
            tools.read_text("source.txt")
        else:
            tools.write_text("result.txt", attempted_content)

    assert not (tmp_path / "result.txt").exists()
    message = str(failure.value)
    assert str(tmp_path) not in message
    assert source_content not in message
    assert attempted_content not in message


@pytest.mark.parametrize(
    "file_bytes",
    (
        pytest.param(None, id="missing"),
        pytest.param(b"\xff\xfe", id="invalid-utf8"),
    ),
)
def test_read_text_reports_safe_deterministic_failures_and_consumes_action(
    tmp_path, file_bytes
):
    relative = "declared-source.txt"
    public_task = PublicTask.model_validate(
        {
            "task_id": "workspace-safe-read-failure",
            "version": "1",
            "instruction": "Read the declared public input.",
            "input_artifacts": (relative,),
            "allowed_artifacts": ("result.txt",),
            "max_actions": 1,
        }
    )
    if file_bytes is not None:
        (tmp_path / relative).write_bytes(file_bytes)
    tools = WorkspaceTools(workspace=tmp_path, task=public_task)

    with pytest.raises(
        ValueError, match="^public input is not readable text$"
    ) as failure:
        tools.read_text(relative)

    message = str(failure.value)
    assert str(tmp_path) not in message
    assert relative not in message
    if file_bytes is not None:
        assert repr(file_bytes) not in message

    with pytest.raises(ValueError, match="^workspace action budget exhausted$"):
        tools.list_files()


def _filesystem_entry_task(*, input_path="source.txt", output_path="result.txt"):
    return PublicTask.model_validate(
        {
            "task_id": "workspace-entry-safety",
            "version": "1",
            "instruction": "Use only supported declared filesystem entries.",
            "input_artifacts": (input_path,),
            "allowed_artifacts": (output_path,),
            "max_actions": 8,
        }
    )


def _create_symlink_or_skip(link, referent):
    try:
        link.symlink_to(referent.name)
    except OSError as error:
        pytest.skip(f"symlinks unavailable: {error}")


def _assert_unsupported_entry_error_is_sanitized(failure, tmp_path, content):
    message = str(failure.value)
    assert message == "workspace contains unsupported filesystem entry"
    assert str(tmp_path) not in message
    assert content not in message


def test_list_files_rejects_declared_input_symlink(tmp_path):
    content = "SYMLINK REFERENT CONTENT"
    referent = tmp_path / "referent.txt"
    referent.write_text(content, encoding="utf-8")
    _create_symlink_or_skip(tmp_path / "source.txt", referent)
    tools = WorkspaceTools(workspace=tmp_path, task=_filesystem_entry_task())

    with pytest.raises(
        ValueError, match="^workspace contains unsupported filesystem entry$"
    ) as failure:
        tools.list_files()

    _assert_unsupported_entry_error_is_sanitized(failure, tmp_path, content)


def test_read_text_rejects_declared_input_symlink(tmp_path):
    content = "SYMLINK INPUT CONTENT"
    referent = tmp_path / "referent.txt"
    referent.write_text(content, encoding="utf-8")
    _create_symlink_or_skip(tmp_path / "source.txt", referent)
    tools = WorkspaceTools(workspace=tmp_path, task=_filesystem_entry_task())

    with pytest.raises(
        ValueError, match="^workspace contains unsupported filesystem entry$"
    ) as failure:
        tools.read_text("source.txt")

    _assert_unsupported_entry_error_is_sanitized(failure, tmp_path, content)


def test_write_text_rejects_declared_output_symlink_without_mutating_referent(tmp_path):
    content = "ORIGINAL OUTPUT REFERENT"
    attempted_content = "ATTEMPTED REPLACEMENT"
    referent = tmp_path / "referent.txt"
    referent.write_text(content, encoding="utf-8")
    _create_symlink_or_skip(tmp_path / "result.txt", referent)
    tools = WorkspaceTools(workspace=tmp_path, task=_filesystem_entry_task())

    with pytest.raises(
        ValueError, match="^workspace contains unsupported filesystem entry$"
    ) as failure:
        tools.write_text("result.txt", attempted_content)

    assert referent.read_text(encoding="utf-8") == content
    _assert_unsupported_entry_error_is_sanitized(
        failure, tmp_path, attempted_content
    )


def test_list_files_rejects_declared_fifo(tmp_path):
    if not hasattr(os, "mkfifo"):
        pytest.skip("FIFO creation unavailable")
    fifo = tmp_path / "source.pipe"
    try:
        os.mkfifo(fifo)
    except OSError as error:
        pytest.skip(f"FIFO creation unavailable: {error}")
    tools = WorkspaceTools(
        workspace=tmp_path,
        task=_filesystem_entry_task(input_path="source.pipe"),
    )

    with pytest.raises(
        ValueError, match="^workspace contains unsupported filesystem entry$"
    ) as failure:
        tools.list_files()

    _assert_unsupported_entry_error_is_sanitized(failure, tmp_path, "FIFO CONTENT")


def test_read_text_inherits_file_size_limit_and_consumes_action(tmp_path):
    relative = "oversized-input.bin"
    public_task = PublicTask.model_validate(
        {
            "task_id": "workspace-oversized-read",
            "version": "1",
            "instruction": "Read the declared public input within limits.",
            "input_artifacts": (relative,),
            "allowed_artifacts": ("result.txt",),
            "max_actions": 1,
        }
    )
    (tmp_path / relative).write_bytes(b"\xff" * (MAX_FILE_BYTES + 1))
    tools = WorkspaceTools(workspace=tmp_path, task=public_task)

    with pytest.raises(ResourceLimitError) as failure:
        tools.read_text(relative)

    message = str(failure.value)
    assert message == f"file size exceeds limit: {relative}"
    assert str(tmp_path) not in message
    assert repr(b"\xff") not in message
    with pytest.raises(ValueError, match="^workspace action budget exhausted$"):
        tools.list_files()


def test_write_text_inherits_utf8_size_limit_without_residue_and_consumes_action(
    tmp_path,
):
    relative = "nested/result.txt"
    public_task = PublicTask.model_validate(
        {
            "task_id": "workspace-oversized-write",
            "version": "1",
            "instruction": "Write the declared output within limits.",
            "input_artifacts": ("source.txt",),
            "allowed_artifacts": (relative,),
            "max_actions": 1,
        }
    )
    content = "x" * (MAX_FILE_BYTES + 1)
    tools = WorkspaceTools(workspace=tmp_path, task=public_task)

    with pytest.raises(ResourceLimitError) as failure:
        tools.write_text(relative, content)

    message = str(failure.value)
    assert message == f"file size exceeds limit: {relative}"
    assert str(tmp_path) not in message
    assert content not in message
    assert not (tmp_path / relative).exists()
    assert not (tmp_path / "nested").exists()
    with pytest.raises(ValueError, match="^workspace action budget exhausted$"):
        tools.list_files()


@pytest.mark.parametrize(
    ("operation", "expected_error"),
    (
        pytest.param(
            "read_text",
            "artifact is not a declared public input",
            id="read",
        ),
        pytest.param(
            "write_text",
            "artifact is not a declared allowed output",
            id="write",
        ),
    ),
)
@pytest.mark.parametrize(
    "attempted_path",
    (
        pytest.param("../escape.txt", id="parent-traversal"),
        pytest.param("/absolute.txt", id="absolute"),
        pytest.param("nested/../../escape.txt", id="nested-traversal"),
        pytest.param("C:/escape.txt", id="windows-absolute"),
        pytest.param(r"..\escape.txt", id="windows-traversal"),
    ),
)
def test_workspace_tools_reject_path_boundary_attempts_before_filesystem_access(
    tmp_path, operation, expected_error, attempted_path
):
    public_task = PublicTask.model_validate(
        {
            "task_id": "workspace-path-boundary",
            "version": "1",
            "instruction": "Use only the exactly declared artifact paths.",
            "input_artifacts": ("source.txt",),
            "allowed_artifacts": ("result.txt",),
            "max_actions": 1,
        }
    )
    source_content = "UNCHANGED DECLARED INPUT"
    attempted_content = "ATTEMPTED OUTSIDE CONTENT"
    source = tmp_path / "source.txt"
    source.write_text(source_content, encoding="utf-8")
    tools = WorkspaceTools(workspace=tmp_path, task=public_task)

    with pytest.raises(ValueError, match=f"^{expected_error}$") as failure:
        if operation == "read_text":
            tools.read_text(attempted_path)
        else:
            tools.write_text(attempted_path, attempted_content)

    assert list(tmp_path.iterdir()) == [source]
    assert source.read_text(encoding="utf-8") == source_content
    message = str(failure.value)
    assert str(tmp_path) not in message
    assert attempted_content not in message
    with pytest.raises(ValueError, match="^workspace action budget exhausted$"):
        tools.list_files()


def test_write_text_rejects_windows_junction_parent_without_mutating_input(tmp_path):
    if os.name != "nt":
        pytest.skip("Windows junctions are unavailable on this platform")

    original_content = b"ORIGINAL PUBLIC INPUT"
    attempted_content = "ATTEMPTED JUNCTION WRITE"
    (tmp_path / "result.txt").write_bytes(original_content)
    junction = tmp_path / "alias"
    creation = subprocess.run(
        [
            "cmd.exe",
            "/d",
            "/c",
            "mklink",
            "/J",
            str(junction),
            str(tmp_path),
        ],
        check=False,
        capture_output=True,
        text=True,
        shell=False,
    )
    if creation.returncode != 0:
        if junction.exists():
            os.rmdir(junction)
        pytest.skip(f"junction creation failed with exit code {creation.returncode}")

    public_task = PublicTask.model_validate(
        {
            "task_id": "workspace-junction-parent",
            "version": "1",
            "instruction": "Write only through supported workspace paths.",
            "input_artifacts": ("result.txt",),
            "allowed_artifacts": ("alias/result.txt",),
            "max_actions": 8,
        }
    )
    tools = WorkspaceTools(workspace=tmp_path, task=public_task)

    try:
        with pytest.raises(
            ValueError,
            match="^workspace contains unsupported filesystem entry$",
        ) as failure:
            tools.write_text("alias/result.txt", attempted_content)

        assert (tmp_path / "result.txt").read_bytes() == original_content
        message = str(failure.value)
        assert str(tmp_path) not in message
        assert attempted_content not in message
        assert original_content.decode("utf-8") not in message
    finally:
        os.rmdir(junction)


def test_revoke_waits_for_in_flight_write_to_finish(tmp_path, monkeypatch):
    public_task = _filesystem_entry_task()
    tools = WorkspaceTools(workspace=tmp_path, task=public_task)
    write_entered = Event()
    release_write = Event()
    revoke_entered = Event()
    revoke_done = Event()
    thread_errors = []
    original_write = workspace_module._write_workspace_text

    def blocking_write(workspace, relative, content):
        if relative == "result.txt":
            write_entered.set()
            if not release_write.wait(5):
                raise TimeoutError("test write release timed out")
        original_write(workspace, relative, content)

    monkeypatch.setattr(workspace_module, "_write_workspace_text", blocking_write)

    def write_target():
        try:
            tools.write_text("result.txt", "complete output\n")
        except BaseException as error:
            thread_errors.append(error)

    def revoke_target():
        revoke_entered.set()
        try:
            tools.revoke()
        except BaseException as error:
            thread_errors.append(error)
        finally:
            revoke_done.set()

    writer = Thread(target=write_target, daemon=True)
    revoker = Thread(target=revoke_target, daemon=True)
    writer.start()
    assert write_entered.wait(2)
    revoker.start()
    try:
        assert revoke_entered.wait(2)
        assert revoke_done.wait(0.1) is False
    finally:
        release_write.set()
        writer.join(2)
        revoker.join(2)

    assert not writer.is_alive()
    assert not revoker.is_alive()
    assert thread_errors == []
    assert revoke_done.is_set()
    assert (tmp_path / "result.txt").read_text(encoding="utf-8") == "complete output\n"
    with pytest.raises(ValueError, match="^workspace tools revoked$"):
        tools.list_files()
