import pytest

from agentenv_forge.runner import load_task, run_episode


TASK_IDS = (
    "text-normalization-001",
    "markdown-normalization-001",
    "log-normalization-001",
)


@pytest.mark.parametrize("task_id", TASK_IDS)
def test_m2_task_matrix_keeps_hidden_oracle_out_of_public_task(task_id: str) -> None:
    task = load_task(task_id)
    public_task = task.to_public_task()

    assert task.expected_content
    assert not hasattr(public_task, "expected_content")
    assert not hasattr(public_task, "initial_files")
    assert public_task.task_id == task_id


@pytest.mark.parametrize("task_id", TASK_IDS)
def test_m2_task_matrix_has_deterministic_full_and_partial_rewards(
    task_id: str, tmp_path
) -> None:
    correct = run_episode(task_id, "correct", seed=42, workspace_root=tmp_path)
    wrong = run_episode(task_id, "wrong", seed=42, workspace_root=tmp_path)

    assert correct.reward.model_dump() == {
        "artifact_exists": 1.0,
        "exact_content": 1.0,
        "policy_compliance": 1.0,
        "total": 1.0,
    }
    assert wrong.reward.model_dump() == {
        "artifact_exists": 1.0,
        "exact_content": 0.0,
        "policy_compliance": 1.0,
        "total": 0.3,
    }
    assert correct.canonical_content() == run_episode(
        task_id, "correct", seed=42, workspace_root=tmp_path
    ).canonical_content()
