import json

import pytest
from pydantic import ValidationError

from agentenv_forge.runner import load_task, parse_task
from agentenv_forge.schemas import TaskSpec


def _task_data() -> dict:
    return {
        "task_id": "test-task",
        "version": "1",
        "split": "train",
        "initial_files": [{"path": "input.txt", "content": "input"}],
        "input_artifact": "input.txt",
        "expected_artifact": "result.txt",
        "expected_content": "result",
        "allowed_artifacts": ["result.txt"],
    }


def test_task_initial_files_are_deeply_immutable():
    task = load_task("text-normalization-001")

    with pytest.raises((TypeError, ValidationError)):
        task.initial_files[0].content = "mutated"

    with pytest.raises(TypeError):
        task.initial_files[0] = task.initial_files[0]


@pytest.mark.parametrize(
    ("field", "invalid_path"),
    [
        ("initial", "/absolute.txt"),
        ("initial", "../escape.txt"),
        ("expected", "nested/../../escape.txt"),
        ("expected", r"..\escape.txt"),
        ("allowed", r"C:\escape.txt"),
        ("allowed", ""),
        ("allowed", "."),
        ("initial", "file.txt:stream"),
        ("initial", "CON"),
        ("initial", "aux.txt"),
        ("expected", "name."),
        ("expected", "name "),
        ("allowed", "a/b/c/d/e/f/g/h/i.txt"),
        ("allowed", "a" * 241),
        ("initial", "bad<name.txt"),
        ("initial", "bad>name.txt"),
        ("initial", 'bad"name.txt'),
        ("initial", "bad|name.txt"),
        ("initial", "bad?name.txt"),
        ("initial", "bad*name.txt"),
        ("initial", "bad\x00name.txt"),
        ("initial", "bad\x1fname.txt"),
    ],
)
def test_task_rejects_unsafe_artifact_paths(field, invalid_path):
    data = _task_data()
    if field == "initial":
        data["initial_files"][0]["path"] = invalid_path
        data["input_artifact"] = invalid_path
    elif field == "expected":
        data["expected_artifact"] = invalid_path
        data["allowed_artifacts"] = [invalid_path]
    else:
        data["allowed_artifacts"] = [invalid_path]
        data["expected_artifact"] = invalid_path

    with pytest.raises(ValidationError):
        TaskSpec.model_validate(data)


def test_parse_rejects_internal_task_id_mismatch():
    data = _task_data()
    data["task_id"] = "different-task"

    with pytest.raises(ValueError, match="task ID mismatch"):
        parse_task("requested-task", json.dumps(data))


@pytest.mark.parametrize(
    "task_id", ("../escape", "nested/task", r"..\escape", "", ".", "con")
)
def test_load_rejects_unsafe_requested_task_id_before_resource_lookup(task_id):
    with pytest.raises(ValueError, match="task ID"):
        load_task(task_id)


def test_task_requires_expected_artifact_to_be_allowed():
    data = _task_data()
    data["allowed_artifacts"] = ["other.txt"]

    with pytest.raises(ValidationError, match="expected_artifact"):
        TaskSpec.model_validate(data)


def test_task_requires_declared_input_to_exist_in_initial_files():
    data = _task_data()
    data["input_artifact"] = "missing.txt"

    with pytest.raises(ValidationError, match="input_artifact"):
        TaskSpec.model_validate(data)


def test_task_rejects_duplicate_initial_paths():
    data = _task_data()
    data["initial_files"].append({"path": "input.txt", "content": "different"})

    with pytest.raises(ValidationError, match="unique"):
        TaskSpec.model_validate(data)


def test_task_rejects_output_overlap_with_initial_state():
    data = _task_data()
    data["expected_artifact"] = "input.txt"
    data["allowed_artifacts"] = ["input.txt"]

    with pytest.raises(ValidationError, match="overlap"):
        TaskSpec.model_validate(data)


def test_task_rejects_windows_case_insensitive_collision():
    data = _task_data()
    data["initial_files"][0]["path"] = "Input.txt"
    data["input_artifact"] = "Input.txt"
    data["expected_artifact"] = "input.txt"
    data["allowed_artifacts"] = ["input.txt"]

    with pytest.raises(ValidationError, match="Windows-equivalent"):
        TaskSpec.model_validate(data)


def test_task_rejects_non_nfc_unicode_path_alias():
    data = _task_data()
    decomposed = "cafe\u0301.txt"
    data["expected_artifact"] = decomposed
    data["allowed_artifacts"] = [decomposed]

    with pytest.raises(ValidationError, match="NFC"):
        TaskSpec.model_validate(data)


@pytest.mark.parametrize(
    ("initial_files", "allowed_artifacts", "expected_artifact"),
    [
        (
            [
                {"path": "a", "content": "file"},
                {"path": "a/input.txt", "content": "nested"},
            ],
            ["result.txt"],
            "result.txt",
        ),
        (
            [{"path": "out", "content": "file"}],
            ["out/result.txt"],
            "out/result.txt",
        ),
    ],
)
def test_task_rejects_ancestor_file_path_collisions(
    initial_files, allowed_artifacts, expected_artifact
):
    data = _task_data()
    data["initial_files"] = initial_files
    data["input_artifact"] = initial_files[0]["path"]
    data["allowed_artifacts"] = allowed_artifacts
    data["expected_artifact"] = expected_artifact

    with pytest.raises(ValidationError, match="ancestor"):
        TaskSpec.model_validate(data)


def test_task_rejects_utf8_content_over_byte_limit():
    data = _task_data()
    data["initial_files"][0]["content"] = "界" * 400_000

    with pytest.raises(ValidationError, match="UTF-8 byte"):
        TaskSpec.model_validate(data)


def test_task_reserves_file_capacity_for_allowed_outputs():
    data = _task_data()
    data["initial_files"] = [
        {"path": f"input-{index}.txt", "content": "x"} for index in range(64)
    ]
    data["input_artifact"] = "input-0.txt"

    with pytest.raises(ValidationError, match="final workspace file count"):
        TaskSpec.model_validate(data)


def test_task_reserves_byte_capacity_for_expected_output():
    data = _task_data()
    data["initial_files"] = [
        {"path": f"input-{index}.txt", "content": "x" * 1_000_000}
        for index in range(4)
    ]
    data["input_artifact"] = "input-0.txt"
    data["expected_content"] = "x" * 1_000_000

    with pytest.raises(ValidationError, match="final workspace byte"):
        TaskSpec.model_validate(data)


def test_task_reserves_entry_capacity_for_parent_directories():
    data = _task_data()
    data["initial_files"] = [
        {"path": f"root-{index}/a/b/c/d/e/f/file.txt", "content": "x"}
        for index in range(16)
    ]
    data["input_artifact"] = data["initial_files"][0]["path"]

    with pytest.raises(ValidationError, match="final workspace entry count"):
        TaskSpec.model_validate(data)
