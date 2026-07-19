from threading import Barrier, Lock, Thread

import pytest

from agentenv_forge.schemas import PublicTask
from agentenv_forge.tools import WorkspaceActionLimitError, WorkspaceTools
from agentenv_forge.tools.budget import ActionBudget, ActionBudgetExhaustedError


def test_action_budget_charges_exact_limit_and_exhaustion_is_stable() -> None:
    budget = ActionBudget(3)

    for _ in range(3):
        budget.charge()

    assert budget.limit == 3
    assert budget.used == 3
    assert budget.remaining == 0
    for _ in range(2):
        with pytest.raises(
            ActionBudgetExhaustedError, match="^episode action budget exhausted$"
        ):
            budget.charge()
    assert budget.used == 3


@pytest.mark.parametrize("invalid_limit", (True, 0, 33, 1.0, "3"))
def test_action_budget_rejects_non_exact_or_out_of_range_limits(invalid_limit) -> None:
    with pytest.raises(ValueError, match="^invalid action budget$"):
        ActionBudget(invalid_limit)


def test_action_budget_rejects_hostile_integer_subclass_without_magic_calls() -> None:
    calls: list[str] = []

    class HostileInt(int):
        def __eq__(self, other):
            calls.append("eq")
            raise AssertionError("must not compare hostile integer")

        def __index__(self):
            calls.append("index")
            raise AssertionError("must not coerce hostile integer")

    with pytest.raises(ValueError, match="^invalid action budget$"):
        ActionBudget(HostileInt(3))

    assert calls == []


def test_action_budget_is_atomic_under_contention() -> None:
    limit = 17
    workers = 64
    budget = ActionBudget(limit)
    barrier = Barrier(workers)
    result_lock = Lock()
    results: list[str] = []

    def charge_once() -> None:
        barrier.wait()
        try:
            budget.charge()
        except ActionBudgetExhaustedError:
            result = "exhausted"
        else:
            result = "charged"
        with result_lock:
            results.append(result)

    threads = [Thread(target=charge_once) for _ in range(workers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    assert all(not thread.is_alive() for thread in threads)
    assert results.count("charged") == limit
    assert results.count("exhausted") == workers - limit
    assert budget.used == limit
    assert budget.remaining == 0


def test_action_budget_exposes_no_reset_or_refund_api() -> None:
    budget = ActionBudget(1)

    assert not hasattr(budget, "reset")
    assert not hasattr(budget, "refund")
    assert not hasattr(budget, "decrement")


def test_workspace_capabilities_can_share_one_episode_budget(tmp_path) -> None:
    public_task = PublicTask.model_validate(
        {
            "task_id": "shared-budget",
            "version": "1",
            "instruction": "Inspect the declared input.",
            "input_artifacts": ("source.txt",),
            "allowed_artifacts": ("result.txt",),
            "max_actions": 1,
        }
    )
    (tmp_path / "source.txt").write_text("source", encoding="utf-8")
    budget = ActionBudget(public_task.max_actions)
    first = WorkspaceTools(workspace=tmp_path, task=public_task, budget=budget)
    second = WorkspaceTools(workspace=tmp_path, task=public_task, budget=budget)

    assert first.list_files() == ("source.txt",)
    with pytest.raises(
        WorkspaceActionLimitError, match="^workspace action budget exhausted$"
    ):
        second.read_text("source.txt")
    assert budget.used == 1


def test_workspace_rejects_mismatched_or_subclassed_injected_budget(tmp_path) -> None:
    public_task = PublicTask.model_validate(
        {
            "task_id": "budget-injection",
            "version": "1",
            "instruction": "Inspect the declared input.",
            "input_artifacts": ("source.txt",),
            "allowed_artifacts": ("result.txt",),
            "max_actions": 2,
        }
    )

    with pytest.raises(ValueError, match="^invalid workspace action budget$"):
        WorkspaceTools(workspace=tmp_path, task=public_task, budget=ActionBudget(1))

    class BudgetSubclass(ActionBudget):
        pass

    with pytest.raises(ValueError, match="^invalid workspace action budget$"):
        WorkspaceTools(workspace=tmp_path, task=public_task, budget=BudgetSubclass(2))


def test_workspace_rejects_hostile_model_construct_limit_before_comparison(
    tmp_path,
) -> None:
    calls: list[str] = []

    class HostileInt(int):
        def __eq__(self, other):
            calls.append("eq")
            raise AssertionError("must not compare hostile task limit")

        def __ne__(self, other):
            calls.append("ne")
            raise AssertionError("must not compare hostile task limit")

    public_task = PublicTask.model_construct(
        task_id="hostile-budget",
        version="1",
        instruction="Inspect the declared input.",
        input_artifacts=("source.txt",),
        allowed_artifacts=("result.txt",),
        max_actions=HostileInt(2),
    )

    with pytest.raises(ValueError, match="^invalid workspace action budget$"):
        WorkspaceTools(workspace=tmp_path, task=public_task, budget=ActionBudget(2))

    assert calls == []


def test_revoked_workspace_consumes_no_shared_budget_and_peer_remains_usable(
    tmp_path,
) -> None:
    public_task = PublicTask.model_validate(
        {
            "task_id": "revoked-shared-budget",
            "version": "1",
            "instruction": "Inspect the declared input and write the output.",
            "input_artifacts": ("source.txt",),
            "allowed_artifacts": ("result.txt",),
            "max_actions": 2,
        }
    )
    (tmp_path / "source.txt").write_text("source", encoding="utf-8")
    budget = ActionBudget(public_task.max_actions)
    revoked = WorkspaceTools(workspace=tmp_path, task=public_task, budget=budget)
    active = WorkspaceTools(workspace=tmp_path, task=public_task, budget=budget)
    revoked.revoke()

    with pytest.raises(ValueError, match="^workspace tools revoked$"):
        revoked.list_files()
    assert budget.used == 0

    assert active.read_text("source.txt") == "source"
    active.write_text("result.txt", "result")
    assert budget.used == 2


def test_rejected_workspace_action_consumes_shared_budget_for_peer(tmp_path) -> None:
    public_task = PublicTask.model_validate(
        {
            "task_id": "rejected-shared-budget",
            "version": "1",
            "instruction": "Inspect the declared input.",
            "input_artifacts": ("source.txt",),
            "allowed_artifacts": ("result.txt",),
            "max_actions": 1,
        }
    )
    budget = ActionBudget(public_task.max_actions)
    first = WorkspaceTools(workspace=tmp_path, task=public_task, budget=budget)
    second = WorkspaceTools(workspace=tmp_path, task=public_task, budget=budget)

    with pytest.raises(ValueError, match="^artifact is not a declared public input$"):
        first.read_text("undeclared.txt")
    with pytest.raises(
        WorkspaceActionLimitError, match="^workspace action budget exhausted$"
    ):
        second.list_files()
    assert budget.used == 1


def test_workspace_capabilities_share_budget_atomically_under_contention(
    tmp_path,
) -> None:
    limit = 16
    workers = 48
    public_task = PublicTask.model_validate(
        {
            "task_id": "concurrent-shared-budget",
            "version": "1",
            "instruction": "Inspect the declared input.",
            "input_artifacts": ("source.txt",),
            "allowed_artifacts": ("result.txt",),
            "max_actions": limit,
        }
    )
    (tmp_path / "source.txt").write_text("source", encoding="utf-8")
    budget = ActionBudget(limit)
    capabilities = (
        WorkspaceTools(workspace=tmp_path, task=public_task, budget=budget),
        WorkspaceTools(workspace=tmp_path, task=public_task, budget=budget),
    )
    barrier = Barrier(workers)
    result_lock = Lock()
    results: list[str] = []

    def list_once(index: int) -> None:
        barrier.wait()
        try:
            capabilities[index % 2].list_files()
        except WorkspaceActionLimitError:
            result = "exhausted"
        else:
            result = "charged"
        with result_lock:
            results.append(result)

    threads = [Thread(target=list_once, args=(index,)) for index in range(workers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    assert all(not thread.is_alive() for thread in threads)
    assert results.count("charged") == limit
    assert results.count("exhausted") == workers - limit
    assert budget.used == limit
