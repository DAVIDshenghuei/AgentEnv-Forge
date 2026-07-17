import json

import pytest
from pydantic import ValidationError

from agentenv_forge.runner import load_task, parse_task
from agentenv_forge.schemas import PublicTask, TaskSpec


def _task_data() -> dict:
    data = {
        "task_id": "test-task",
        "version": "1",
        "instruction": "Normalize the input text.",
        "max_actions": 4,
        "split": "train",
        "initial_files": [{"path": "input.txt", "content": "input"}],
        "input_artifact": "input.txt",
        "expected_artifact": "result.txt",
        "expected_content": "result",
        "allowed_artifacts": ["result.txt"],
    }
    return data


def test_task_requires_non_empty_instruction():
    data = _task_data()

    assert TaskSpec.model_validate(data).instruction == data["instruction"]

    del data["instruction"]
    with pytest.raises(ValidationError, match="instruction"):
        TaskSpec.model_validate(data)

    for instruction in ("", " \t\n"):
        data["instruction"] = instruction
        with pytest.raises(ValidationError, match="instruction"):
            TaskSpec.model_validate(data)


def test_task_constrains_max_actions_to_public_limit():
    for max_actions in (1, 32):
        data = _task_data()
        data["max_actions"] = max_actions
        assert TaskSpec.model_validate(data).max_actions == max_actions

    for max_actions in (0, 33):
        data = _task_data()
        data["max_actions"] = max_actions
        with pytest.raises(ValidationError, match="max_actions"):
            TaskSpec.model_validate(data)


def test_to_public_task_is_deeply_frozen_and_forbids_extra_fields():
    data = _task_data()
    data["initial_files"] = [
        {"path": "input.txt", "content": "PRIMARY SECRET CONTENT"},
        {"path": "context.txt", "content": "SECONDARY SECRET CONTENT"},
    ]
    task = TaskSpec.model_validate(data)

    public_task = task.to_public_task()

    assert set(type(public_task).model_fields) == {
        "task_id",
        "version",
        "instruction",
        "input_artifacts",
        "allowed_artifacts",
        "max_actions",
    }
    assert public_task.input_artifacts == tuple(
        initial_file.path for initial_file in task.initial_files
    )

    with pytest.raises((TypeError, ValidationError)):
        public_task.instruction = "Changed"
    with pytest.raises(TypeError):
        public_task.input_artifacts[0] = "changed.txt"
    with pytest.raises(ValidationError, match="extra"):
        type(public_task).model_validate(
            {**public_task.model_dump(mode="json"), "expected_content": "secret"}
        )


def test_serialized_public_task_excludes_hidden_verifier_data():
    data = _task_data()
    data["initial_files"] = [
        {"path": "input.txt", "content": "PRIMARY SECRET CONTENT"},
        {"path": "context.txt", "content": "SECONDARY SECRET CONTENT"},
    ]
    task = TaskSpec.model_validate(data)

    serialized = json.loads(task.to_public_task().model_dump_json())

    assert set(serialized) == {
        "task_id",
        "version",
        "instruction",
        "input_artifacts",
        "allowed_artifacts",
        "max_actions",
    }
    assert serialized["input_artifacts"] == ["input.txt", "context.txt"]
    assert "expected_content" not in serialized
    assert "expected_artifact" not in serialized
    serialized_json = json.dumps(serialized)
    assert "PRIMARY SECRET CONTENT" not in serialized_json
    assert "SECONDARY SECRET CONTENT" not in serialized_json


def _public_task_data() -> dict:
    return {
        "task_id": "test-task",
        "version": "1",
        "instruction": "Normalize the input text.",
        "input_artifacts": ("input.txt",),
        "allowed_artifacts": ("result.txt",),
        "max_actions": 8,
    }


@pytest.mark.parametrize("max_actions", (1, 32))
def test_public_task_direct_validation_accepts_max_action_boundaries(max_actions):
    public_task = PublicTask.model_validate(
        {**_public_task_data(), "max_actions": max_actions}
    )

    assert public_task.max_actions == max_actions


@pytest.mark.parametrize(
    "override",
    (
        {"instruction": " \t\n"},
        {"max_actions": 0},
        {"max_actions": 33},
    ),
)
def test_public_task_direct_validation_rejects_invalid_content_limits(override):
    with pytest.raises(ValidationError):
        PublicTask.model_validate({**_public_task_data(), **override})


def test_public_task_direct_validation_accepts_valid_minimal_security_contract():
    public_task = PublicTask.model_validate(_public_task_data())

    assert public_task.model_dump() == _public_task_data()


@pytest.mark.parametrize(
    "override",
    (
        pytest.param({"task_id": "../escape"}, id="unsafe-task-id"),
        pytest.param(
            {"input_artifacts": ("../secret",)}, id="unsafe-input-path"
        ),
        pytest.param(
            {"allowed_artifacts": ("C:/escape.txt",)}, id="unsafe-allowed-path"
        ),
        pytest.param({"input_artifacts": ()}, id="empty-input-artifacts"),
        pytest.param({"allowed_artifacts": ()}, id="empty-allowed-artifacts"),
        pytest.param(
            {"input_artifacts": ("input.txt", "input.txt")},
            id="duplicate-input-artifacts",
        ),
        pytest.param(
            {"allowed_artifacts": ("result.txt", "result.txt")},
            id="duplicate-allowed-artifacts",
        ),
        pytest.param(
            {
                "input_artifacts": ("shared.txt",),
                "allowed_artifacts": ("shared.txt",),
            },
            id="input-allowed-overlap",
        ),
        pytest.param(
            {
                "input_artifacts": ("Artifact.txt",),
                "allowed_artifacts": ("artifact.txt",),
            },
            id="windows-equivalent-cross-collection-collision",
        ),
        pytest.param(
            {
                "input_artifacts": ("a",),
                "allowed_artifacts": ("a/b",),
            },
            id="ancestor-cross-collection-collision",
        ),
        pytest.param(
            {"input_artifacts": ("a", "a/b")},
            id="ancestor-input-collection-collision",
        ),
        pytest.param(
            {"allowed_artifacts": ("a", "a/b")},
            id="ancestor-allowed-collection-collision",
        ),
    ),
)
def test_public_task_direct_validation_rejects_unsafe_contract(override):
    with pytest.raises(ValidationError):
        PublicTask.model_validate({**_public_task_data(), **override})


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
