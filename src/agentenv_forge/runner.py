import hashlib
import json
import tempfile
from datetime import datetime, timezone
from importlib.resources import files
from pathlib import Path, PurePosixPath

from .schemas import (
    ArtifactRecord,
    ConditionLabels,
    Event,
    RewardBreakdown,
    TaskSpec,
    Trajectory,
    validate_task_id,
)


MAX_WORKSPACE_ENTRIES = 128
MAX_WORKSPACE_FILES = 64
MAX_FILE_BYTES = 1_048_576
MAX_TOTAL_FILE_BYTES = 4_194_304
MAX_PATH_DEPTH = 8
HASH_CHUNK_BYTES = 65_536


class ResourceLimitError(ValueError):
    """Deterministic fail-closed termination caused by bounded resources."""


def parse_task(requested_task_id: str, raw_spec: str) -> TaskSpec:
    task = TaskSpec.model_validate_json(raw_spec)
    if task.task_id != requested_task_id:
        raise ValueError(
            f"task ID mismatch: requested {requested_task_id!r}, parsed {task.task_id!r}"
        )
    return task


def load_task(task_id: str) -> TaskSpec:
    validate_task_id(task_id)
    resource = files("agentenv_forge.tasks").joinpath(f"{task_id}.json")
    if not resource.is_file():
        raise ValueError(f"unknown task: {task_id}")
    task = parse_task(task_id, resource.read_text(encoding="utf-8"))
    if task.split == "holdout":
        raise ValueError("holdout tasks cannot be executed by the local runner")
    return task


def _normalize(value: str) -> str:
    lines = (" ".join(line.lower().split()) for line in value.splitlines())
    return "\n".join(line for line in lines if line) + "\n"


def _apply_action(task: TaskSpec, workspace: Path, action: str) -> None:
    if action not in {"correct", "wrong"}:
        raise ValueError(f"unsupported action: {action}")
    # This adapter is deliberately synchronous; no background process is
    # allowed to retain workspace access when control returns.
    source = _read_bounded_text(_workspace_target(workspace, task.input_artifact))
    result = _normalize(source) if action == "correct" else source
    _write_workspace_text(workspace, task.expected_artifact, result)


def _workspace_target(workspace: Path, relative: str) -> Path:
    """Resolve an existing or prospective target and prove workspace containment."""
    root = workspace.resolve(strict=True)
    target = (root / relative).resolve(strict=False)
    try:
        target.relative_to(root)
    except ValueError as error:
        raise ValueError(f"workspace path escapes containment: {relative!r}") from error
    return target


def _write_workspace_text(workspace: Path, relative: str, content: str) -> None:
    if len(content.encode("utf-8")) > MAX_FILE_BYTES:
        raise ResourceLimitError(f"file size exceeds limit: {relative}")
    target = _workspace_target(workspace, relative)
    target.parent.mkdir(parents=True, exist_ok=True)
    # Controlled actions are synchronous and must release workspace access before
    # verification. Re-resolve after parent creation to reject static symlinks.
    target = _workspace_target(workspace, relative)
    target.write_bytes(content.encode("utf-8"))


def _read_bounded_text(target: Path) -> str:
    size = target.stat().st_size
    if size > MAX_FILE_BYTES:
        raise ResourceLimitError(f"file size exceeds limit: {target.name}")
    return target.read_bytes().decode("utf-8")


def _hash_bounded_file(target: Path) -> str:
    digest = hashlib.sha256()
    with target.open("rb") as stream:
        while chunk := stream.read(HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _bounded_sorted_workspace_paths(workspace: Path):
    discovered = 0

    def walk(directory: Path):
        nonlocal discovered
        children: list[Path] = []
        for child in directory.iterdir():
            discovered += 1
            if discovered > MAX_WORKSPACE_ENTRIES:
                raise ResourceLimitError("workspace entry count exceeds limit")
            children.append(child)
        for child in sorted(children, key=lambda path: path.name):
            yield child
            if not child.is_symlink() and child.is_dir():
                yield from walk(child)

    yield from walk(workspace)


def _bounded_workspace_entries(
    workspace: Path,
) -> tuple[dict[str, Path], set[str], set[str]]:
    total_bytes = 0
    workspace_files: dict[str, Path] = {}
    workspace_directories: set[str] = set()
    special_entries: set[str] = set()
    for path in _bounded_sorted_workspace_paths(workspace):
        relative_path = path.relative_to(workspace)
        if len(relative_path.parts) > MAX_PATH_DEPTH:
            raise ResourceLimitError("workspace path depth exceeds limit")
        relative = relative_path.as_posix()
        if path.is_symlink():
            raise ValueError(f"workspace symlinks are forbidden: {relative}")
        if path.is_dir():
            workspace_directories.add(relative)
            continue
        if not path.is_file():
            special_entries.add(relative)
            continue
        if len(workspace_files) >= MAX_WORKSPACE_FILES:
            raise ResourceLimitError("workspace file count exceeds limit")
        safe_target = _workspace_target(workspace, relative)
        size = safe_target.stat().st_size
        if size > MAX_FILE_BYTES:
            raise ResourceLimitError(f"file size exceeds limit: {relative}")
        total_bytes += size
        if total_bytes > MAX_TOTAL_FILE_BYTES:
            raise ResourceLimitError("workspace total file size exceeds limit")
        workspace_files[relative] = safe_target
    return workspace_files, workspace_directories, special_entries


def _verify(task: TaskSpec, workspace: Path) -> tuple[RewardBreakdown, list[ArtifactRecord]]:
    # The action phase must be quiescent here: no agent/background process may
    # retain workspace access while this deterministic snapshot is verified.
    workspace_files, workspace_directories, special_entries = _bounded_workspace_entries(workspace)
    artifact = workspace_files.get(task.expected_artifact)
    exists = artifact is not None
    content = _read_bounded_text(artifact) if artifact is not None else None
    initial_paths = {item.path for item in task.initial_files}
    initial_state_intact = all(
        (target := workspace_files.get(item.path)) is not None
        and _read_bounded_text(target) == item.content
        for item in task.initial_files
    )
    declared_paths = initial_paths.union(task.allowed_artifacts)
    declared_directories = {
        PurePosixPath(*PurePosixPath(path).parts[:depth]).as_posix()
        for path in declared_paths
        for depth in range(1, len(PurePosixPath(path).parts))
    }
    non_file_entries_compliant = (
        not special_entries and workspace_directories.issubset(declared_directories)
    )
    produced_targets = sorted(
        (
            (relative, target)
            for relative, target in workspace_files.items()
            if relative not in initial_paths
        ),
        key=lambda item: item[0],
    )
    produced = [relative for relative, _ in produced_targets]
    compliant = (
        initial_state_intact
        and non_file_entries_compliant
        and all(path in task.allowed_artifacts for path in produced)
    )
    exact = exists and content == task.expected_content
    reward = RewardBreakdown(
        artifact_exists=float(exists),
        exact_content=float(exact),
        policy_compliance=float(compliant),
        total=round(0.2 * exists + 0.7 * exact + 0.1 * compliant, 10),
    )
    artifacts = [
        ArtifactRecord(path=path, sha256=_hash_bounded_file(target))
        for path, target in produced_targets
    ]
    return reward, artifacts


def _build_trajectory(
    task: TaskSpec,
    action: str,
    seed: int,
    events: list[Event],
    reward: RewardBreakdown,
    artifacts: list[ArtifactRecord],
    termination_reason: str,
    environment_failure: str | None,
) -> Trajectory:
    return Trajectory(
        task_id=task.task_id,
        task_version=task.version,
        split=task.split,
        conditions=ConditionLabels(baseline="reset-v1", action_variant=action),
        seed=seed,
        events=events,
        artifacts=artifacts,
        reward=reward,
        termination_reason=termination_reason,
        environment_failure=environment_failure,
        agent_failure=None,
        runtime_metadata={"completed_at": datetime.now(timezone.utc).isoformat()},
    )


def _resource_failure(
    task: TaskSpec, action: str, seed: int, events: list[Event], error: ResourceLimitError
) -> Trajectory:
    return _build_trajectory(
        task=task,
        action=action,
        seed=seed,
        events=events,
        reward=RewardBreakdown(
            artifact_exists=0.0,
            exact_content=0.0,
            policy_compliance=0.0,
            total=0.0,
        ),
        artifacts=[],
        termination_reason="resource_limit",
        environment_failure=str(error),
    )


def run_episode(task_id: str, action: str, seed: int, workspace_root: Path | None = None) -> Trajectory:
    task = load_task(task_id)
    root = Path(workspace_root) if workspace_root is not None else None
    if root is not None:
        root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="agentenv-forge-", dir=root) as directory:
        workspace = Path(directory)
        for initial_file in task.initial_files:
            _write_workspace_text(workspace, initial_file.path, initial_file.content)
        events = [Event(sequence=0, kind="reset", detail="initial state restored")]
        try:
            _apply_action(task, workspace, action)
        except ResourceLimitError as error:
            events.append(Event(sequence=1, kind="action_failed", detail="resource_limit"))
            return _resource_failure(task, action, seed, events, error)
        events.append(Event(sequence=1, kind="action", detail=action))
        try:
            reward, artifacts = _verify(task, workspace)
        except ResourceLimitError as error:
            events.append(Event(sequence=2, kind="verify_failed", detail="resource_limit"))
            return _resource_failure(task, action, seed, events, error)
        events.append(Event(sequence=2, kind="verify", detail="deterministic verifier completed"))
        return _build_trajectory(
            task=task,
            action=action,
            seed=seed,
            events=events,
            reward=reward,
            artifacts=artifacts,
            termination_reason="completed",
            environment_failure=None,
        )
