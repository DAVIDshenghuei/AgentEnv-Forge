# AgentEnv Forge

This repository contains a deliberately small, deterministic causal evaluation slice. It has no model or API dependency: a checked-in task is reset into a fresh temporary workspace, a controlled action writes `result.txt`, and a deterministic verifier emits reward plus a JSONL trajectory.

## Setup

Requirements: [uv](https://docs.astral.sh/uv/) and a platform supported by uv. The project pins Python 3.11 through `.python-version`; uv installs it when needed.

```sh
uv sync --extra dev
uv run pytest
```

## Local smoke

Correct intervention:

```sh
uv run python -m agentenv_forge run --task text-normalization-001 --action correct --seed 42 --output outputs/trajectory.jsonl
```

Negative control:

```sh
uv run python -m agentenv_forge run --task text-normalization-001 --action wrong --seed 42 --output outputs/wrong-trajectory.jsonl
```

Each invocation appends exactly one JSON object as one line. The output path is a **single-writer interface**: do not point concurrent CLI processes at the same JSONL file; use one file per worker and merge after completion. The correct action scores `1.0`; the wrong action scores lower. Repeating an identical task/action/seed produces identical `canonical_content()` after excluding `runtime_metadata` timestamps.

## Container smoke

The configuration does not mount the Docker socket or any host directory. Its image is digest-pinned; the smoke service runs non-root with no network, no Linux capabilities, a read-only root filesystem, and only `/tmp` writable through tmpfs.

```sh
docker compose build
docker compose run --rm smoke
```

The seed task is explicitly `train`. The local runner rejects holdout tasks before reset and never places expected content in trajectory output. Task-controlled paths must be NFC-normalized relative POSIX paths; traversal, absolute paths, backslashes, control characters, Windows-invalid characters/device names/ADS aliases, case-insensitive aliases, symlinks, excessive depth/length, and declared file/parent collisions are rejected. Every runtime file target is resolved beneath the episode workspace before access. Initial-state mutation and undeclared files, directories, or special filesystem entries reduce policy compliance. Verification enforces deterministic limits: 128 entries, 64 files, 1 MiB per file, 4 MiB total, and path depth 8. Task validation reserves capacity for the expected final workspace. Resource-limit termination is recorded as a zero-reward failure trajectory. Controlled actions are synchronous; future Agent adapters must stop and release the workspace before verification.
