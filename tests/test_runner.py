import pytest
from pydantic import ValidationError

import agentenv_forge.runner as runner
from agentenv_forge.runner import ResourceLimitError, run_episode
from agentenv_forge.schemas import RewardBreakdown, TaskSpec


def test_correct_action_gets_full_reward(tmp_path):
    trajectory = run_episode(
        task_id="text-normalization-001",
        action="correct",
        seed=42,
        workspace_root=tmp_path,
    )

    assert trajectory.reward.total == 1.0
    assert trajectory.reward.artifact_exists == 1.0
    assert trajectory.reward.exact_content == 1.0
    assert trajectory.reward.policy_compliance == 1.0
    assert trajectory.environment_failure is None
    assert trajectory.agent_failure is None


def test_wrong_action_scores_lower_than_correct(tmp_path):
    correct = run_episode("text-normalization-001", "correct", 42, tmp_path)
    wrong = run_episode("text-normalization-001", "wrong", 42, tmp_path)

    assert wrong.reward.total < correct.reward.total
    assert wrong.reward.artifact_exists == 1.0
    assert wrong.reward.exact_content == 0.0


def test_canonical_rerun_is_deterministic(tmp_path):
    first = run_episode("text-normalization-001", "correct", 42, tmp_path)
    second = run_episode("text-normalization-001", "correct", 42, tmp_path)

    assert first.runtime_metadata != second.runtime_metadata
    assert first.canonical_content() == second.canonical_content()


def test_resource_limit_becomes_fail_closed_trajectory(tmp_path, monkeypatch):
    def reject_verification(*_args, **_kwargs):
        raise ResourceLimitError("file size exceeds limit: result.txt")

    monkeypatch.setattr(runner, "_verify", reject_verification)
    trajectory = run_episode("text-normalization-001", "correct", 42, tmp_path)

    assert trajectory.termination_reason == "resource_limit"
    assert trajectory.environment_failure == "file size exceeds limit: result.txt"
    assert trajectory.agent_failure is None
    assert trajectory.reward.total == 0.0
    assert trajectory.artifacts == []
    assert trajectory.events[-1].kind == "verify_failed"


def test_action_resource_limit_becomes_fail_closed_trajectory(tmp_path, monkeypatch):
    expanding_input = "İ" * 400_000
    task = TaskSpec.model_validate(
        {
            "task_id": "expanding-input",
            "version": "1",
            "split": "train",
            "initial_files": [{"path": "source.txt", "content": expanding_input}],
            "input_artifact": "source.txt",
            "expected_artifact": "result.txt",
            "expected_content": "unused",
            "allowed_artifacts": ["result.txt"],
        }
    )
    monkeypatch.setattr(runner, "load_task", lambda _task_id: task)

    trajectory = run_episode("expanding-input", "correct", 42, tmp_path)

    assert trajectory.termination_reason == "resource_limit"
    assert "file size exceeds limit" in trajectory.environment_failure
    assert trajectory.reward.total == 0.0
    assert trajectory.events[-1].kind == "action_failed"


def test_reset_workspace_is_isolated_and_removed(tmp_path):
    sentinel = tmp_path / "result.txt"
    sentinel.write_text("contamination", encoding="utf-8")

    trajectory = run_episode("text-normalization-001", "correct", 42, tmp_path)

    assert trajectory.reward.total == 1.0
    assert sentinel.read_text(encoding="utf-8") == "contamination"
    assert list(tmp_path.iterdir()) == [sentinel]


def test_invalid_deep_path_is_rejected_before_reset_materialization(tmp_path, monkeypatch):
    invalid_data = {
        "task_id": "deep-path",
        "version": "1",
        "split": "train",
        "initial_files": [
            {"path": "a/b/c/d/e/f/g/h/i.txt", "content": "input"}
        ],
        "input_artifact": "a/b/c/d/e/f/g/h/i.txt",
        "expected_artifact": "result.txt",
        "expected_content": "result",
        "allowed_artifacts": ["result.txt"],
    }

    def load_invalid(_task_id):
        return TaskSpec.model_validate(invalid_data)

    monkeypatch.setattr(runner, "load_task", load_invalid)
    with pytest.raises(ValidationError, match="depth"):
        run_episode("deep-path", "correct", 42, tmp_path)

    assert list(tmp_path.iterdir()) == []


def test_maximum_valid_file_count_executes_correct_action(tmp_path, monkeypatch):
    initial_files = [
        {"path": f"input-{index}.txt", "content": "input"}
        for index in range(63)
    ]
    task = TaskSpec.model_validate(
        {
            "task_id": "max-files",
            "version": "1",
            "split": "train",
            "initial_files": initial_files,
            "input_artifact": "input-0.txt",
            "expected_artifact": "result.txt",
            "expected_content": "input\n",
            "allowed_artifacts": ["result.txt"],
        }
    )
    monkeypatch.setattr(runner, "load_task", lambda _task_id: task)

    trajectory = run_episode("max-files", "correct", 42, tmp_path)

    assert trajectory.termination_reason == "completed"
    assert trajectory.reward.total == 1.0


def test_reward_schema_rejects_inconsistent_total():
    with pytest.raises(ValidationError):
        RewardBreakdown(
            artifact_exists=1.0,
            exact_content=1.0,
            policy_compliance=1.0,
            total=0.5,
        )

    with pytest.raises(ValidationError):
        RewardBreakdown(
            artifact_exists=1.1,
            exact_content=1.0,
            policy_compliance=1.0,
            total=1.0,
        )
