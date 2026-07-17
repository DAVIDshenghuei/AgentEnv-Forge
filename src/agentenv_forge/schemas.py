import json
import re
import unicodedata
from pathlib import PurePosixPath, PureWindowsPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


TASK_ID_PATTERN = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,62}[a-z0-9])?$")
MAX_DECLARED_FILE_BYTES = 1_048_576
MAX_DECLARED_TOTAL_BYTES = 4_194_304
MAX_DECLARED_FILES = 64
MAX_DECLARED_ENTRIES = 128
MAX_ARTIFACT_PATH_BYTES = 240
MAX_ARTIFACT_PATH_DEPTH = 8
WINDOWS_INVALID_PATH_CHARS = frozenset('<>:"|?*')
WINDOWS_RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    "CLOCK$",
    "CONIN$",
    "CONOUT$",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
}


def validate_utf8_size(value: str) -> str:
    if len(value.encode("utf-8")) > MAX_DECLARED_FILE_BYTES:
        raise ValueError("content exceeds UTF-8 byte limit")
    return value


def validate_task_id(value: str) -> str:
    if not TASK_ID_PATTERN.fullmatch(value) or value.upper() in WINDOWS_RESERVED_NAMES:
        raise ValueError("task ID must be a safe 1-64 lowercase alphanumeric or hyphen name")
    return value


def validate_instruction(value: str) -> str:
    if not value.strip():
        raise ValueError("instruction must contain non-whitespace content")
    return value


def validate_relative_artifact_path(value: str) -> str:
    """Accept only cross-platform-safe normalized relative file paths."""
    if not value or "\\" in value:
        raise ValueError("artifact path must be a non-empty POSIX-style relative path")
    if unicodedata.normalize("NFC", value) != value:
        raise ValueError("artifact path must use NFC Unicode normalization")
    if any(ord(character) < 32 for character in value):
        raise ValueError("artifact path cannot contain control characters")
    if any(character in WINDOWS_INVALID_PATH_CHARS for character in value):
        raise ValueError("artifact path contains a Windows-invalid character")
    if PurePosixPath(value).is_absolute() or PureWindowsPath(value).is_absolute():
        raise ValueError("artifact path must be relative")
    parts = value.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError("artifact path cannot contain empty, dot, or traversal components")
    if len(parts) > MAX_ARTIFACT_PATH_DEPTH:
        raise ValueError("artifact path depth exceeds limit")
    if len(value.encode("utf-8")) > MAX_ARTIFACT_PATH_BYTES:
        raise ValueError("artifact path byte length exceeds limit")
    for part in parts:
        if ":" in part:
            raise ValueError("artifact path cannot use Windows alternate data streams")
        if part.endswith((".", " ")):
            raise ValueError("artifact path cannot end a component with dot or space")
        if part.split(".", 1)[0].upper() in WINDOWS_RESERVED_NAMES:
            raise ValueError("artifact path cannot use a Windows reserved device name")
    return value


def validate_declared_file_count(
    input_paths: tuple[str, ...], allowed_paths: tuple[str, ...]
) -> None:
    if len(input_paths) + len(allowed_paths) > MAX_DECLARED_FILES:
        raise ValueError("final workspace file count exceeds limit")


def validate_unique_paths(paths: tuple[str, ...], *, label: str) -> None:
    if len(paths) != len(set(paths)):
        raise ValueError(f"{label} paths must be unique")


def validate_artifact_collections_do_not_overlap(
    input_paths: tuple[str, ...], allowed_paths: tuple[str, ...]
) -> None:
    if set(allowed_paths).intersection(input_paths):
        raise ValueError("allowed output artifacts cannot overlap initial state")


def validate_declared_path_collisions(declared_paths: tuple[str, ...]) -> None:
    parent_directories = {
        PurePosixPath(*PurePosixPath(path).parts[:depth]).as_posix()
        for path in declared_paths
        for depth in range(1, len(PurePosixPath(path).parts))
    }
    if len(declared_paths) + len(parent_directories) > MAX_DECLARED_ENTRIES:
        raise ValueError("final workspace entry count exceeds limit")
    canonical_paths = tuple(
        "/".join(part.casefold() for part in PurePosixPath(path).parts)
        for path in declared_paths
    )
    if len(canonical_paths) != len(set(canonical_paths)):
        raise ValueError("declared paths have a Windows-equivalent collision")
    path_parts = [(path, PurePosixPath(path).parts) for path in canonical_paths]
    for left_path, left_parts in path_parts:
        for right_path, right_parts in path_parts:
            if left_path != right_path and len(left_parts) < len(right_parts):
                if right_parts[: len(left_parts)] == left_parts:
                    raise ValueError("declared file path cannot be an ancestor of another path")


class InitialFile(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    path: str
    content: str = Field(max_length=1_048_576)

    @field_validator("path")
    @classmethod
    def path_is_safe(cls, value: str) -> str:
        return validate_relative_artifact_path(value)

    @field_validator("content")
    @classmethod
    def content_fits_byte_limit(cls, value: str) -> str:
        return validate_utf8_size(value)


class PublicTask(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    task_id: str
    version: str
    instruction: str
    input_artifacts: tuple[str, ...] = Field(
        min_length=1, max_length=MAX_DECLARED_FILES
    )
    allowed_artifacts: tuple[str, ...] = Field(
        min_length=1, max_length=MAX_DECLARED_FILES
    )
    max_actions: int = Field(ge=1, le=32)

    @field_validator("instruction")
    @classmethod
    def instruction_has_usable_content(cls, value: str) -> str:
        return validate_instruction(value)

    @field_validator("task_id")
    @classmethod
    def task_id_is_safe(cls, value: str) -> str:
        return validate_task_id(value)

    @field_validator("input_artifacts", "allowed_artifacts")
    @classmethod
    def artifact_paths_are_safe(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(validate_relative_artifact_path(value) for value in values)

    @model_validator(mode="after")
    def artifact_collections_form_a_safe_contract(self) -> "PublicTask":
        validate_declared_file_count(self.input_artifacts, self.allowed_artifacts)
        validate_unique_paths(self.input_artifacts, label="input artifact")
        validate_unique_paths(self.allowed_artifacts, label="allowed artifact")
        validate_artifact_collections_do_not_overlap(
            self.input_artifacts, self.allowed_artifacts
        )
        validate_declared_path_collisions(
            self.input_artifacts + self.allowed_artifacts
        )
        return self


class TaskSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    task_id: str
    version: str
    instruction: str
    max_actions: int = Field(ge=1, le=32)
    split: Literal["train", "validation", "holdout"]
    initial_files: tuple[InitialFile, ...]
    input_artifact: str
    expected_artifact: str
    expected_content: str = Field(max_length=1_048_576)
    allowed_artifacts: tuple[str, ...]

    @field_validator("instruction")
    @classmethod
    def instruction_has_usable_content(cls, value: str) -> str:
        return validate_instruction(value)

    @field_validator("task_id")
    @classmethod
    def task_id_is_safe(cls, value: str) -> str:
        return validate_task_id(value)

    @field_validator("input_artifact", "expected_artifact")
    @classmethod
    def artifact_path_is_safe(cls, value: str) -> str:
        return validate_relative_artifact_path(value)

    @field_validator("expected_content")
    @classmethod
    def expected_content_fits_byte_limit(cls, value: str) -> str:
        return validate_utf8_size(value)

    @field_validator("allowed_artifacts")
    @classmethod
    def allowed_artifacts_are_safe(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(validate_relative_artifact_path(value) for value in values)

    @model_validator(mode="after")
    def paths_form_a_valid_causal_contract(self) -> "TaskSpec":
        initial_paths = tuple(item.path for item in self.initial_files)
        validate_declared_file_count(initial_paths, self.allowed_artifacts)
        initial_bytes = sum(len(item.content.encode("utf-8")) for item in self.initial_files)
        expected_bytes = len(self.expected_content.encode("utf-8"))
        if initial_bytes + expected_bytes > MAX_DECLARED_TOTAL_BYTES:
            raise ValueError("final workspace byte count exceeds limit")
        validate_unique_paths(initial_paths, label="initial file")
        if self.input_artifact not in initial_paths:
            raise ValueError("input_artifact must identify an initial file")
        validate_unique_paths(self.allowed_artifacts, label="allowed artifact")
        if self.expected_artifact not in self.allowed_artifacts:
            raise ValueError("expected_artifact must be listed in allowed_artifacts")
        validate_artifact_collections_do_not_overlap(
            initial_paths, self.allowed_artifacts
        )
        declared_paths = initial_paths + self.allowed_artifacts
        validate_declared_path_collisions(declared_paths)
        return self

    def to_public_task(self) -> PublicTask:
        return PublicTask(
            task_id=self.task_id,
            version=self.version,
            instruction=self.instruction,
            input_artifacts=tuple(item.path for item in self.initial_files),
            allowed_artifacts=self.allowed_artifacts,
            max_actions=self.max_actions,
        )


class ConditionLabels(BaseModel):
    model_config = ConfigDict(frozen=True)

    baseline: str
    action_variant: str


class Event(BaseModel):
    sequence: int
    kind: str
    detail: str


class ArtifactRecord(BaseModel):
    path: str
    sha256: str


class RewardBreakdown(BaseModel):
    artifact_exists: float = Field(ge=0, le=1)
    exact_content: float = Field(ge=0, le=1)
    policy_compliance: float = Field(ge=0, le=1)
    total: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def total_matches_components(self) -> "RewardBreakdown":
        expected = round(
            0.2 * self.artifact_exists
            + 0.7 * self.exact_content
            + 0.1 * self.policy_compliance,
            10,
        )
        if self.total != expected:
            raise ValueError("total must equal the deterministic weighted component score")
        return self


class Trajectory(BaseModel):
    task_id: str
    task_version: str
    split: Literal["train", "validation"]
    conditions: ConditionLabels
    seed: int
    events: list[Event]
    artifacts: list[ArtifactRecord]
    reward: RewardBreakdown
    termination_reason: str
    environment_failure: str | None
    agent_failure: str | None
    runtime_metadata: dict[str, str]

    def canonical_content(self) -> str:
        """Stable causal record, deliberately excluding runtime-only metadata."""
        return json.dumps(
            self.model_dump(exclude={"runtime_metadata"}, mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        )
