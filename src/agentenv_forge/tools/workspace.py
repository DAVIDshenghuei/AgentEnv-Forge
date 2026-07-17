import stat
from pathlib import Path
from typing import Protocol, runtime_checkable

from ..runner import _read_bounded_text, _workspace_target, _write_workspace_text
from ..schemas import PublicTask


@runtime_checkable
class WorkspaceProtocol(Protocol):
    def list_files(self) -> tuple[str, ...]: ...

    def read_text(self, relative: str) -> str: ...

    def write_text(self, relative: str, content: str) -> None: ...


class WorkspaceTools:
    __slots__ = ("_workspace", "_task", "_used_actions", "_revoked")

    def __init__(self, workspace: Path, task: PublicTask) -> None:
        try:
            root = workspace.resolve(strict=True)
            if not root.is_dir():
                raise ValueError("workspace is unavailable")
        except OSError:
            raise ValueError("workspace is unavailable") from None
        self._workspace = root
        self._task = task
        self._used_actions = 0
        self._revoked = False

    def revoke(self) -> None:
        self._revoked = True

    def _consume_action(self) -> None:
        if self._revoked:
            raise ValueError("workspace tools revoked")
        if self._used_actions >= self._task.max_actions:
            raise ValueError("workspace action budget exhausted")
        self._used_actions += 1

    def _check_declared_target(self, relative: str) -> None:
        target = self._workspace
        parts = Path(relative).parts
        for index, part in enumerate(parts):
            target = target / part
            try:
                entry_stat = target.lstat()
            except FileNotFoundError:
                return
            except OSError:
                raise ValueError(
                    "workspace contains unsupported filesystem entry"
                ) from None
            mode = entry_stat.st_mode
            file_attributes = getattr(entry_stat, "st_file_attributes", 0)
            reparse_point = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
            if stat.S_ISLNK(mode) or file_attributes & reparse_point:
                raise ValueError(
                    "workspace contains unsupported filesystem entry"
                ) from None
            is_leaf = index == len(parts) - 1
            if (not is_leaf and not stat.S_ISDIR(mode)) or (
                is_leaf and not stat.S_ISREG(mode)
            ):
                raise ValueError(
                    "workspace contains unsupported filesystem entry"
                ) from None

    def list_files(self) -> tuple[str, ...]:
        self._consume_action()
        declared_paths = self._task.input_artifacts + self._task.allowed_artifacts
        existing_files = []
        try:
            for relative in sorted(declared_paths):
                self._check_declared_target(relative)
                if _workspace_target(self._workspace, relative).is_file():
                    existing_files.append(relative)
        except OSError:
            raise ValueError("workspace listing failed") from None
        return tuple(existing_files)

    def read_text(self, relative: str) -> str:
        self._consume_action()
        if relative not in self._task.input_artifacts:
            raise ValueError("artifact is not a declared public input")
        self._check_declared_target(relative)
        try:
            return _read_bounded_text(_workspace_target(self._workspace, relative))
        except (OSError, UnicodeError):
            raise ValueError("public input is not readable text") from None

    def write_text(self, relative: str, content: str) -> None:
        self._consume_action()
        if relative not in self._task.allowed_artifacts:
            raise ValueError("artifact is not a declared allowed output")
        self._check_declared_target(relative)
        try:
            _write_workspace_text(self._workspace, relative, content)
        except (OSError, UnicodeError):
            raise ValueError("output could not be written") from None
