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


class TaskSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    task_id: str
    version: str
    split: Literal["train", "validation", "holdout"]
    initial_files: tuple[InitialFile, ...]
    input_artifact: str
    expected_artifact: str
    expected_content: str = Field(max_length=1_048_576)
    allowed_artifacts: tuple[str, ...]

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
        if len(initial_paths) + len(self.allowed_artifacts) > MAX_DECLARED_FILES:
            raise ValueError("final workspace file count exceeds limit")
        initial_bytes = sum(len(item.content.encode("utf-8")) for item in self.initial_files)
        expected_bytes = len(self.expected_content.encode("utf-8"))
        if initial_bytes + expected_bytes > MAX_DECLARED_TOTAL_BYTES:
            raise ValueError("final workspace byte count exceeds limit")
        if len(initial_paths) != len(set(initial_paths)):
            raise ValueError("initial file paths must be unique")
        if self.input_artifact not in initial_paths:
            raise ValueError("input_artifact must identify an initial file")
        if len(self.allowed_artifacts) != len(set(self.allowed_artifacts)):
            raise ValueError("allowed artifact paths must be unique")
        if self.expected_artifact not in self.allowed_artifacts:
            raise ValueError("expected_artifact must be listed in allowed_artifacts")
        if set(self.allowed_artifacts).intersection(initial_paths):
            raise ValueError("allowed output artifacts cannot overlap initial state")
        declared_paths = initial_paths + self.allowed_artifacts
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
        return self


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
