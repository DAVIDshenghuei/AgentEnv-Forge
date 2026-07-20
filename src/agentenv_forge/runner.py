import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from importlib.resources import files
from pathlib import Path, PurePosixPath
from threading import Condition
from typing import TYPE_CHECKING, Callable, Protocol

from .schemas import (
    ArtifactRecord,
    ConditionLabels,
    Event,
    RewardBreakdown,
    TaskSpec,
    Trajectory,
    validate_task_id,
)

if TYPE_CHECKING:
    from .adapters import AgentAdapter
    from .tools import BrowserProtocol, ResearchClientProtocol, TerminalResult


class TerminalEnvironment(Protocol):
    def start(self) -> None: ...

    def execute(self, argv: tuple[str, ...]) -> "TerminalResult": ...

    def close(self) -> None: ...


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
    agent_failure: str | None = None,
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
        agent_failure=agent_failure,
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


def _close_terminal_environment(
    environment: TerminalEnvironment,
) -> tuple[bool, BaseException | None]:
    cleanup_base_exception: BaseException | None = None
    for _ in range(2):
        try:
            environment.close()
        except BaseException as error:
            if not isinstance(error, Exception) and cleanup_base_exception is None:
                cleanup_base_exception = error
            continue
        return True, cleanup_base_exception
    return False, cleanup_base_exception


def run_agent_episode(
    task_id: str,
    adapter: "AgentAdapter",
    seed: int,
    workspace_root: Path | None = None,
    terminal_command_runner: "Callable[[tuple[str, ...]], TerminalResult] | None" = None,
    terminal_environment_factory: "Callable[[Path], TerminalEnvironment] | None" = None,
    research_client: "ResearchClientProtocol | None" = None,
    browser_client: "BrowserProtocol | None" = None,
) -> Trajectory:
    if terminal_command_runner is not None and terminal_environment_factory is not None:
        raise ValueError("terminal runner and environment are mutually exclusive")
    from .adapters import AgentRunResult
    from .tools import (
        ActionBudget,
        BrowserTools,
        ResearchTools,
        TerminalResult,
        TerminalTools,
    )
    from .tools.workspace import (
        WorkspaceTools,
        _create_workspace_facade,
        _revoke_workspace_facade,
    )

    task = load_task(task_id)
    root = Path(workspace_root) if workspace_root is not None else None
    if root is not None:
        root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="agentenv-forge-", dir=root) as directory:
        workspace = Path(directory)
        for initial_file in task.initial_files:
            _write_workspace_text(workspace, initial_file.path, initial_file.content)

        events = [Event(sequence=0, kind="reset", detail="initial state restored")]

        def emit(kind: str, detail: str) -> None:
            events.append(Event(sequence=len(events), kind=kind, detail=detail))

        public_task = task.to_public_task()
        budget = ActionBudget(public_task.max_actions)
        workspace_tools = WorkspaceTools(
            workspace=workspace,
            task=public_task,
            budget=budget,
        )
        if research_client is None:

            class _UnavailableResearchClient:
                def search_papers(self, query: str, limit: int):
                    raise ValueError("research client unavailable")

                def get_paper(self, paper_id: str):
                    raise ValueError("research client unavailable")

            research_client = _UnavailableResearchClient()
        research_tools = ResearchTools(
            task=public_task,
            budget=budget,
            client=research_client,
        )
        if browser_client is None:

            class _UnavailableBrowserClient:
                def open_page(self, path: str):
                    raise ValueError("browser client unavailable")

                def click_link(self, current_path: str, link_id: str):
                    raise ValueError("browser client unavailable")

            browser_client = _UnavailableBrowserClient()
        browser_tools = BrowserTools(
            task=public_task,
            budget=budget,
            client=browser_client,
        )
        terminal_environment: TerminalEnvironment | None = None
        if terminal_environment_factory is not None:
            try:
                terminal_environment = terminal_environment_factory(workspace)
                terminal_environment.start()
            except Exception:
                if terminal_environment is not None:
                    _close_terminal_environment(terminal_environment)
                try:
                    adapter.close()
                except BaseException:
                    pass
                emit("environment_failure", "terminal environment startup failed")
                return _build_trajectory(
                    task=task,
                    action="agent",
                    seed=seed,
                    events=events,
                    reward=RewardBreakdown(
                        artifact_exists=0.0,
                        exact_content=0.0,
                        policy_compliance=0.0,
                        total=0.0,
                    ),
                    artifacts=[],
                    termination_reason="environment_error",
                    environment_failure="terminal environment startup failed",
                    agent_failure=None,
                )
            except BaseException:
                if terminal_environment is not None:
                    _close_terminal_environment(terminal_environment)
                try:
                    adapter.close()
                except BaseException:
                    pass
                raise
        allowed_adapter_event_details = {
            "list_files",
            "terminal_execute",
            "research_search_papers",
            "research_get_paper",
            "browser_open_page",
            "browser_click_link",
            *(f"read_text:{path}" for path in public_task.input_artifacts),
            *(f"write_text:{path}" for path in public_task.allowed_artifacts),
        }
        adapter_events_active = True
        pending_adapter_call: str | None = None
        in_flight_tool_call: str | None = None
        completed_tool_call: str | None = None
        adapter_tool_event_count = 0
        facade_calls_in_flight = 0
        adapter_event_condition = Condition()

        def emit_adapter_event(kind: str, detail: str) -> None:
            nonlocal pending_adapter_call
            nonlocal in_flight_tool_call
            nonlocal completed_tool_call
            nonlocal adapter_tool_event_count
            with adapter_event_condition:
                if not adapter_events_active:
                    raise ValueError("adapter events are inactive")
                if (
                    type(kind) is not str
                    or type(detail) is not str
                    or kind not in {"tool_call", "tool_result"}
                    or detail not in allowed_adapter_event_details
                ):
                    raise ValueError("invalid adapter event")
                if kind == "tool_call":
                    if (
                        pending_adapter_call is not None
                        or in_flight_tool_call is not None
                        or completed_tool_call is not None
                    ):
                        raise ValueError("invalid adapter event")
                    pending_adapter_call = detail
                    return
                if (
                    pending_adapter_call is not None
                    or in_flight_tool_call is not None
                    or completed_tool_call != detail
                ):
                    raise ValueError("invalid adapter event")
                emit(kind, detail)
                adapter_tool_event_count += 1
                completed_tool_call = None

        def before_tool_call(detail: str) -> None:
            nonlocal pending_adapter_call
            nonlocal in_flight_tool_call
            nonlocal adapter_tool_event_count
            nonlocal facade_calls_in_flight
            with adapter_event_condition:
                if (
                    not adapter_events_active
                    or in_flight_tool_call is not None
                    or completed_tool_call is not None
                    or pending_adapter_call != detail
                    or adapter_tool_event_count >= 2 * public_task.max_actions + 1
                ):
                    raise ValueError("invalid adapter event")
                emit("tool_call", detail)
                adapter_tool_event_count += 1
                pending_adapter_call = None
                in_flight_tool_call = detail
                facade_calls_in_flight += 1

        def after_tool_call(detail: str, succeeded: bool) -> None:
            nonlocal in_flight_tool_call
            nonlocal completed_tool_call
            nonlocal facade_calls_in_flight
            with adapter_event_condition:
                if in_flight_tool_call != detail:
                    raise ValueError("invalid adapter event")
                in_flight_tool_call = None
                try:
                    if not adapter_events_active or not succeeded:
                        completed_tool_call = None
                    else:
                        if completed_tool_call is not None:
                            raise ValueError("invalid adapter event")
                        completed_tool_call = detail
                finally:
                    facade_calls_in_flight -= 1
                    if facade_calls_in_flight == 0:
                        adapter_event_condition.notify_all()

        try:
            if terminal_environment is not None:
                command_runner = terminal_environment.execute
            elif terminal_command_runner is None:

                def unavailable_terminal_runner(
                    argv: tuple[str, ...],
                ) -> TerminalResult:
                    raise ValueError("terminal is unavailable")

                command_runner = unavailable_terminal_runner
            else:
                command_runner = terminal_command_runner
            terminal_tools = TerminalTools(
                task=public_task,
                budget=budget,
                command_runner=command_runner,
            )
            adapter_tools = _create_workspace_facade(
                workspace_tools,
                before_tool_call,
                after_tool_call,
                terminal_tools,
                research_tools,
                browser_tools,
            )
        except Exception:
            if terminal_environment is None:
                raise
            _close_terminal_environment(terminal_environment)
            try:
                adapter.close()
            except BaseException:
                pass
            emit("environment_failure", "terminal environment startup failed")
            return _build_trajectory(
                task=task,
                action="agent",
                seed=seed,
                events=events,
                reward=RewardBreakdown(
                    artifact_exists=0.0,
                    exact_content=0.0,
                    policy_compliance=0.0,
                    total=0.0,
                ),
                artifacts=[],
                termination_reason="environment_error",
                environment_failure="terminal environment startup failed",
                agent_failure=None,
            )
        except BaseException:
            if terminal_environment is not None:
                _close_terminal_environment(terminal_environment)
                try:
                    adapter.close()
                except BaseException:
                    pass
            raise
        if terminal_environment is not None:
            emit("environment_start", "terminal environment started")

        def deactivate_revoke_and_drain(
            snapshot_handshake: bool = False,
        ) -> tuple[str | None, str | None, str | None] | None:
            nonlocal adapter_events_active
            with adapter_event_condition:
                adapter_events_active = False
                snapshot = (
                    pending_adapter_call,
                    in_flight_tool_call,
                    completed_tool_call,
                )
            _revoke_workspace_facade(adapter_tools)
            with adapter_event_condition:
                while facade_calls_in_flight > 0:
                    adapter_event_condition.wait()
            return snapshot if snapshot_handshake else None

        emit("adapter_start", "adapter started")
        try:
            result = adapter.run(public_task, adapter_tools, emit_adapter_event)
        except Exception:
            deactivate_revoke_and_drain()
            termination_reason = "agent_error"
            agent_failure = "adapter execution failed"
            emit("adapter_failure", agent_failure)
        except BaseException:
            deactivate_revoke_and_drain()
            try:
                adapter.close()
            except BaseException:
                pass
            if terminal_environment is not None:
                _close_terminal_environment(terminal_environment)
            raise
        else:
            handshake_snapshot = deactivate_revoke_and_drain(True)
            valid_result = False
            if type(result) is AgentRunResult and type(result.termination_reason) is str:
                if result.termination_reason == "agent_error":
                    if type(result.agent_failure) is str:
                        try:
                            failure_size = len(result.agent_failure.encode("utf-8"))
                        except UnicodeError:
                            pass
                        else:
                            valid_result = 1 <= failure_size <= 256
                elif result.termination_reason in {"finished", "action_limit", "timeout"}:
                    valid_result = result.agent_failure is None
            valid_result = (
                valid_result
                and handshake_snapshot == (None, None, None)
            )
            if not valid_result:
                termination_reason = "agent_error"
                agent_failure = "adapter returned invalid result"
                emit("adapter_failure", agent_failure)
            elif result.termination_reason == "agent_error":
                termination_reason = "agent_error"
                agent_failure = "adapter reported failure"
                emit("adapter_failure", agent_failure)
            else:
                termination_reason = result.termination_reason
                agent_failure = result.agent_failure
                if termination_reason in {"action_limit", "timeout"}:
                    emit("adapter_termination", termination_reason)
        try:
            adapter.close()
        except Exception:
            termination_reason = "agent_error"
            agent_failure = "adapter close failed"
            emit("adapter_failure", agent_failure)
            emit("adapter_stop", "adapter stop failed")
        except BaseException:
            if terminal_environment is not None:
                _close_terminal_environment(terminal_environment)
            raise
        else:
            emit("adapter_stop", "adapter stopped")
        environment_cleanup_failed = False
        if terminal_environment is not None:
            environment_closed, cleanup_base_exception = _close_terminal_environment(
                terminal_environment
            )
            if cleanup_base_exception is not None:
                raise cleanup_base_exception
            if environment_closed:
                emit("environment_stop", "terminal environment stopped")
            else:
                environment_cleanup_failed = True
                emit("environment_failure", "terminal environment cleanup failed")
        emit("tools_revoked", "workspace tools revoked")
        if environment_cleanup_failed:
            return _build_trajectory(
                task=task,
                action="agent",
                seed=seed,
                events=events,
                reward=RewardBreakdown(
                    artifact_exists=0.0,
                    exact_content=0.0,
                    policy_compliance=0.0,
                    total=0.0,
                ),
                artifacts=[],
                termination_reason="environment_error",
                environment_failure="terminal environment cleanup failed",
                agent_failure=agent_failure,
            )
        try:
            reward, artifacts = _verify(task, workspace)
        except ResourceLimitError:
            emit("verify_failed", "resource_limit")
            return _build_trajectory(
                task=task,
                action="agent",
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
                environment_failure="verification resource limit exceeded",
                agent_failure=agent_failure,
            )
        except Exception:
            emit("verify_failed", "environment_error")
            return _build_trajectory(
                task=task,
                action="agent",
                seed=seed,
                events=events,
                reward=RewardBreakdown(
                    artifact_exists=0.0,
                    exact_content=0.0,
                    policy_compliance=0.0,
                    total=0.0,
                ),
                artifacts=[],
                termination_reason="environment_error",
                environment_failure="verification failed",
                agent_failure=agent_failure,
            )
        emit("verify", "deterministic verifier completed")
        return _build_trajectory(
            task=task,
            action="agent",
            seed=seed,
            events=events,
            reward=reward,
            artifacts=artifacts,
            termination_reason=termination_reason,
            environment_failure=None,
            agent_failure=agent_failure,
        )


def _prepare_docker_workspace(workspace: Path) -> str:
    getuid = getattr(os, "getuid", None)
    getgid = getattr(os, "getgid", None)
    if getuid is None and getgid is None:
        return "10001:10001"
    if not callable(getuid) or not callable(getgid):
        raise ValueError("sandbox identity is unavailable")
    uid = getuid()
    gid = getgid()
    if (
        type(uid) is not int
        or type(gid) is not int
        or not 0 <= uid <= 2_147_483_647
        or not 0 <= gid <= 2_147_483_647
        or (uid != 0 and gid == 0)
    ):
        raise ValueError("sandbox identity is unavailable")
    if uid != 0:
        return f"{uid}:{gid}"

    entries = sorted(workspace.rglob("*"), key=lambda path: len(path.parts), reverse=True)
    for entry in (*entries, workspace):
        os.chown(entry, 10001, 10001, follow_symlinks=False)
    return "10001:10001"


def run_docker_agent_episode(
    task_id: str,
    adapter: "AgentAdapter",
    seed: int,
    workspace_root: Path | None = None,
    image: str = "agentenv-forge-sandbox:test",
    command_timeout_seconds: float = 10.0,
    research_client: "ResearchClientProtocol | None" = None,
    browser_client: "BrowserProtocol | None" = None,
) -> Trajectory:
    from uuid import uuid4

    from .sandbox import BoundedProcessExecutor, DockerSandbox

    executor = BoundedProcessExecutor(max_output_bytes=65_536)

    def create_environment(workspace: Path) -> DockerSandbox:
        container_user = _prepare_docker_workspace(workspace)
        return DockerSandbox(
            workspace=workspace,
            image=image,
            command_executor=executor,
            container_name=f"agentenv-forge-episode-{uuid4().hex}",
            command_timeout_seconds=command_timeout_seconds,
            container_user=container_user,
        )

    return run_agent_episode(
        task_id=task_id,
        adapter=adapter,
        seed=seed,
        workspace_root=workspace_root,
        terminal_environment_factory=create_environment,
        research_client=research_client,
        browser_client=browser_client,
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
