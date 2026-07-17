import hashlib
from pathlib import Path

import pytest

from agentenv_forge.runner import (
    ResourceLimitError,
    _apply_action,
    _bounded_workspace_entries,
    _read_bounded_text,
    _verify,
    _write_workspace_text,
)
from agentenv_forge.schemas import TaskSpec


def _task(expected_artifact: str = "result.txt") -> TaskSpec:
    return TaskSpec.model_validate(
        {
            "task_id": "test-task",
            "version": "1",
            "instruction": "Process the declared input artifact.",
            "max_actions": 8,
            "split": "train",
            "initial_files": [{"path": "input.txt", "content": "input"}],
            "input_artifact": "input.txt",
            "expected_artifact": expected_artifact,
            "expected_content": "result",
            "allowed_artifacts": [expected_artifact],
        }
    )


def test_contained_nested_artifact_write_creates_parent(tmp_path):
    _write_workspace_text(tmp_path, "input.txt", "input")
    _write_workspace_text(tmp_path, "nested/output/result.txt", "result")

    assert (tmp_path / "nested/output/result.txt").read_text(encoding="utf-8") == "result"
    reward, artifacts = _verify(_task("nested/output/result.txt"), tmp_path)
    assert reward.total == 1.0
    assert [artifact.path for artifact in artifacts] == ["nested/output/result.txt"]


def test_action_reads_schema_declared_input_artifact(tmp_path):
    task = TaskSpec.model_validate(
        {
            "task_id": "custom-input",
            "version": "1",
            "instruction": "Normalize the declared source text.",
            "max_actions": 8,
            "split": "train",
            "initial_files": [{"path": "source.txt", "content": "  CUSTOM   INPUT  "}],
            "input_artifact": "source.txt",
            "expected_artifact": "result.txt",
            "expected_content": "custom input\n",
            "allowed_artifacts": ["result.txt"],
        }
    )
    _write_workspace_text(tmp_path, "source.txt", "  CUSTOM   INPUT  ")

    _apply_action(task, tmp_path, "correct")

    assert (tmp_path / "result.txt").read_text(encoding="utf-8") == "custom input\n"


def test_workspace_text_io_preserves_exact_utf8_newline_bytes(tmp_path):
    content = "alpha\r\nbeta\n"
    task = TaskSpec.model_validate(
        {
            "task_id": "newline-bytes",
            "version": "1",
            "instruction": "Preserve the declared text exactly.",
            "max_actions": 8,
            "split": "train",
            "initial_files": [{"path": "input.txt", "content": content}],
            "input_artifact": "input.txt",
            "expected_artifact": "result.txt",
            "expected_content": content,
            "allowed_artifacts": ["result.txt"],
        }
    )
    _write_workspace_text(tmp_path, "input.txt", content)
    _write_workspace_text(tmp_path, "result.txt", content)

    assert (tmp_path / "input.txt").read_bytes() == content.encode("utf-8")
    assert _read_bounded_text(tmp_path / "input.txt") == content
    reward, artifacts = _verify(task, tmp_path)
    assert reward.total == 1.0
    assert artifacts[0].sha256 == hashlib.sha256(content.encode("utf-8")).hexdigest()


def test_verifier_rejects_symlink_escape(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("result", encoding="utf-8")
    try:
        (workspace / "result.txt").symlink_to(outside)
    except OSError as error:
        pytest.skip(f"symlinks unavailable: {error}")

    with pytest.raises(ValueError, match="symlinks are forbidden"):
        _verify(_task(), workspace)


def test_verifier_penalizes_initial_state_mutation(tmp_path):
    (tmp_path / "input.txt").write_text("tampered", encoding="utf-8")
    (tmp_path / "result.txt").write_text("result", encoding="utf-8")

    reward, _ = _verify(_task(), tmp_path)

    assert reward.artifact_exists == 1.0
    assert reward.exact_content == 1.0
    assert reward.policy_compliance == 0.0
    assert reward.total == 0.9


def test_verifier_penalizes_undeclared_directory(tmp_path):
    (tmp_path / "input.txt").write_text("input", encoding="utf-8")
    (tmp_path / "result.txt").write_text("result", encoding="utf-8")
    (tmp_path / "undeclared").mkdir()

    reward, _ = _verify(_task(), tmp_path)

    assert reward.policy_compliance == 0.0
    assert reward.total == 0.9


def test_verifier_penalizes_special_filesystem_entry(tmp_path):
    import os

    if not hasattr(os, "mkfifo"):
        pytest.skip("FIFO creation unavailable")
    (tmp_path / "input.txt").write_text("input", encoding="utf-8")
    (tmp_path / "result.txt").write_text("result", encoding="utf-8")
    os.mkfifo(tmp_path / "undeclared-fifo")

    reward, _ = _verify(_task(), tmp_path)

    assert reward.policy_compliance == 0.0
    assert reward.total == 0.9


def test_verifier_rejects_too_many_artifacts(tmp_path):
    (tmp_path / "input.txt").write_text("input", encoding="utf-8")
    (tmp_path / "result.txt").write_text("result", encoding="utf-8")
    for index in range(64):
        (tmp_path / f"extra-{index}.txt").write_text("x", encoding="utf-8")

    with pytest.raises(ValueError, match="file count"):
        _verify(_task(), tmp_path)


def test_verifier_rejects_oversized_artifact_before_decoding(tmp_path):
    (tmp_path / "input.txt").write_text("input", encoding="utf-8")
    (tmp_path / "result.txt").write_bytes(b"x" * 1_048_577)

    with pytest.raises(ValueError, match="file size"):
        _verify(_task(), tmp_path)


def test_resource_limit_error_is_independent_of_enumeration_order(tmp_path, monkeypatch):
    first = tmp_path / "a.bin"
    second = tmp_path / "z.bin"
    first.write_bytes(b"a" * (1_048_576 + 1))
    second.write_bytes(b"z" * (1_048_576 + 1))
    original_iterdir = Path.iterdir
    messages = []

    for order in ([second, first], [first, second]):
        monkeypatch.setattr(Path, "iterdir", lambda _self, order=order: iter(order))
        with pytest.raises(ResourceLimitError) as failure:
            _bounded_workspace_entries(tmp_path)
        messages.append(str(failure.value))

    monkeypatch.setattr(Path, "iterdir", original_iterdir)
    assert messages == [
        "file size exceeds limit: a.bin",
        "file size exceeds limit: a.bin",
    ]
