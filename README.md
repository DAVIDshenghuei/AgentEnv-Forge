# AgentEnv Forge

This repository contains a deliberately small, deterministic causal evaluation slice. It has no model or remote API dependency: a checked-in task is reset into a fresh temporary workspace, a controlled action writes `result.txt`, and a deterministic verifier emits reward plus a JSONL trajectory.

## Project records

- [M3B release notes](docs/releases/2026-07-20-m3b.md)
- [ADR 0001: Use a Bounded Offline Browser Capability](docs/adr/0001-bounded-offline-browser.md)

## Setup

Requirements: [uv](https://docs.astral.sh/uv/) and a platform supported by uv. The project pins Python 3.11 through `.python-version`; uv installs it when needed.

```sh
uv sync --extra dev
uv run --no-sync playwright install chromium
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
docker compose run --rm browser-smoke
```

The seed task is explicitly `train`. The local runner rejects holdout tasks before reset and never places expected content in trajectory output. Task-controlled paths must be NFC-normalized relative POSIX paths; traversal, absolute paths, backslashes, control characters, Windows-invalid characters/device names/ADS aliases, case-insensitive aliases, symlinks, excessive depth/length, and declared file/parent collisions are rejected. Every runtime file target is resolved beneath the episode workspace before access. Initial-state mutation and undeclared files, directories, or special filesystem entries reduce policy compliance. Verification enforces deterministic limits: 128 entries, 64 files, 1 MiB per file, 4 MiB total, and path depth 8. Task validation reserves capacity for the expected final workspace. Resource-limit termination is recorded as a zero-reward failure trajectory. Controlled actions are synchronous; terminal-agent episodes revoke all capabilities and close their Docker environment before verification.

## Terminal-agent environment

Status: **implemented**

Agent episodes can run commands in a dedicated, non-root Docker sandbox with no network, no added capabilities, and a read-only root filesystem. The episode directory is exposed to the container only as read-only `/host-workspace`; agent changes are synchronized through a separate writable `/workspace` tmpfs limited to 4 MiB and 256-inode capacity. The host validates the synchronized snapshot before applying declared outputs.

Workspace and terminal capabilities consume one shared action budget, so either kind of admitted call counts against the task's single `max_actions` limit. At the end of an agent run, the runner revokes and drains both capabilities, closes the adapter, and closes the terminal environment. This environment cleanup completes before hidden verification. Agent failures and environment failures remain separate trajectory fields; cleanup or verification failures are environment failures and cannot be mistaken for agent failures.

## Offline Research MCP

Status: **implemented**

Research uses a fixed, versioned corpus bundled with the project and the official MCP stdio transport. Corpus version `1.0.0` is bound to every ordered record field by a canonical SHA-256 manifest. The server exposes exactly `search_papers` and `get_paper`; records and summary projections use strict immutable validation. It performs no network access or model inference.

The synchronous production client starts only the bundled Python module and does not accept an arbitrary MCP command. It opens one stdio process and session per call and validates the exact server identity and tool inventory, including schemas and metadata, then applies a default 10-second operation deadline. The official transport closes stdin and uses a two-second graceful shutdown window before terminating and, if necessary, killing the process tree. Each is closed before return and before hidden verification. Research, workspace, and terminal capabilities consume one shared episode action budget. Trajectories record generic, payload-free trajectory events, so queries, paper identifiers, titles, abstracts, and bodies are not copied into event details.

The train task `research-synthesis-001` is the M3A acceptance path: it searches the offline corpus, reads one paper, formats the permitted fields through the terminal capability, writes `result.txt`, and is scored by the existing hidden verifier.

## Offline Browser MCP

Status: **implemented**

Browser uses Playwright `1.61.0` with its matching Chromium revision against a fixed synthetic HTTPS origin. Site version `1.0.0` is bound to the ordered HTML fixture by a canonical SHA-256 manifest. The server exposes exactly `open_page` and `click_link`; paths and link identifiers are canonical, bounded values. The browser context fulfills only exact bundled pages, blocks service workers, and aborts every other request. Arbitrary URLs, selectors, forms, downloads, screenshots, cookies, storage, and page evaluation are outside this slice.

The synchronous production client starts only the bundled Browser MCP module. Each admitted transport call owns one MCP stdio process and one real Chromium lifecycle, validates the exact server name, tools capability, tool inventory, schemas, metadata, and bounded result, then closes before returning. A default 10-second deadline covers the call; the official transport terminates and reaps the process tree on timeout. Success-path tests continuously observe the real MCP worker, Playwright driver, and Chromium descendants and require all of them to disappear. The timeout-path test substitutes a fixed hanging child that launches real Playwright and Chromium, continuously observes its descendants, and requires the complete observed tree to be reaped. Failures are sanitized and browser payloads never enter trajectory details.

Browser, research, workspace, and terminal calls share one episode action budget and are revoked and drained before adapter close and hidden verification. The train task `browser-evaluation-001` opens the offline index, performs a real DOM link click, passes page evidence through the Docker terminal, writes `browser-report.txt`, and is scored by the hidden verifier. Its host, real-Docker, production-image, and network-disabled Compose acceptance paths are exercised in CI. The production image installs the Playwright-managed browser in the non-root runtime user's standard cache.

The browser worker deadline and process-tree reaping bound duration, but this slice does not claim hard host memory or PID isolation for Chromium. CAMEL, model inference, RL training, and arbitrary web browsing remain unimplemented. Hostile same-process Python plugin isolation is also outside the milestone.
