# Causal design

## Vertical-slice DAG

```text
task specification ──> reset initial state ──> controlled action ──> result.txt
        │                       │                    │                  │
        └───────────────────────┴────────────────────┴──> verifier ──> reward
                                                                    └─> trajectory
```

The task specification fixes the initial file, expected artifact, exact expected content, allowed artifacts, version, and split. Reset materializes that initial state in a new per-episode temporary directory. The action is the intervention. The verifier observes only the resulting workspace and deterministically calculates the outcome. The trajectory records the causal inputs and observations.

## Treatment and outcome

The treatment is the immutable `conditions.action_variant`: `correct` normalizes case and whitespace, while the `wrong` negative control copies the noisy input. The baseline condition is `reset-v1`. The primary outcome is scalar reward in `[0, 1]`; its components are `artifact_exists` (weight 0.2), `exact_content` (0.7), and `policy_compliance` (0.1). Component values and the weighted total are schema-validated.

The intervention comparison holds task, version, baseline, and seed fixed while changing only the action variant. It must satisfy `reward(correct) > reward(wrong)`.

## Controlled confounders

- Initial state is reset from the deeply immutable task specification for every episode; initial files are a tuple of frozen entries.
- Each episode gets a distinct temporary workspace that is removed afterward.
- Action implementations and verifier logic are local, versioned code with no model, API, clock, network, or randomness inputs.
- The seed is recorded and held fixed even though this first task consumes no randomness.
- Artifact paths and hashes are sorted; event sequence numbers define order.
- Task paths reject absolute, empty, dot, traversal, and backslash-ambiguous forms. Runtime targets are independently resolved beneath the workspace, so symlink escapes fail closed.
- Timestamps live only in `runtime_metadata` and cannot affect verification or canonical comparisons.

## Invariants and failure policy

- Task ID, task version, split, baseline, action variant, and seed are present in every successful trajectory. Parsed task identity must match the requested resource identity.
- Condition labels and task specifications are frozen Pydantic models.
- Rewards are bounded and their total must exactly match the declared deterministic weights.
- Initial paths are excluded from produced artifacts only after their exact contents are revalidated; any mutation zeros `policy_compliance`. Produced artifacts are hashed incrementally with SHA-256.
- Verification rejects symlinks and is bounded to 128 workspace entries, 64 files, 1 MiB per file, 4 MiB total file bytes, and path depth 8. Resource-limit failures produce a deterministic zero-reward trajectory with `termination_reason=resource_limit`.
- Agent execution and deterministic controlled actions use the same verification boundary. Agent episodes revoke and drain capabilities before verification begins; concurrent agent/verifier access is outside this slice's security contract.
- Environment and agent failure fields are separate. A successful episode sets both to null.
- Unknown tasks/actions, invalid schemas, holdout tasks, and non-resource verifier/schema errors fail closed: no successful trajectory is returned or written.
- Runtime output contains the split label but never the hidden expected content. Holdout task specifications are rejected before workspace creation, preventing holdout leakage.

## Implemented terminal-agent lifecycle

The agent runner creates one episode-scoped action budget and shares it between the declared workspace tools and terminal tools. Terminal commands execute in a non-root Docker environment whose host workspace bind is read-only; a bounded writable tmpfs holds the container-side working copy. Each capability performs its own revocation while budget ownership remains episode-wide.

Each completed terminal command synchronizes and validates the container working copy back to the host before `execute()` returns. Closing the Docker environment removes the container; it does not perform a final workspace synchronization.

The terminal lifecycle is ordered: run the adapter, deactivate event admission, revoke and drain capabilities, close the adapter, close the Docker environment, and only then run hidden verification. If cleanup fails, verification does not run and the episode returns an environment error.

Failure precedence preserves the primary failure instead of collapsing unrelated causes. Agent execution, result, or adapter-close failures populate `agent_failure`; environment startup, cleanup, resource-limit verification, and verifier failures populate `environment_failure`. A later successful cleanup does not erase an agent failure, while a cleanup failure prevents hidden verification and becomes the environment outcome.

## Implemented offline Research MCP lifecycle

The offline server owns a fixed, versioned corpus with strict immutable validation for full records and summary projections. Corpus version `1.0.0` is bound to every ordered record field by a canonical SHA-256 manifest. It uses the official MCP stdio transport and exposes exactly `search_papers` and `get_paper`; it has no network, clock, randomness, remote API, browser, or model-inference dependency.

The synchronous production wrapper starts only the bundled Python module and does not accept an arbitrary MCP command. It creates one stdio process and session per call, validates the exact server identity and tool inventory, including schemas and metadata, and applies a default 10-second operation deadline. The official transport closes stdin and uses a two-second graceful shutdown window before terminating and, if necessary, killing the process tree. Each is closed before return and before hidden verification. Malformed identity, inventory, results, timeouts, or tool failures collapse to a sanitized local failure without copying remote payloads into the trajectory.

Research tools share the shared episode action budget with workspace and terminal tools. The opaque adapter facade emits generic, payload-free trajectory events for research calls, then revokes and drains the research capability alongside the other capabilities before adapter close and verification. The M3A train task `research-synthesis-001` exercises this complete lifecycle against deterministic hidden verification.

## Implemented offline Browser MCP lifecycle

The browser server owns a fixed synthetic HTTPS site. Site version `1.0.0` is bound to the ordered HTML fixture by a canonical SHA-256 manifest. It uses Playwright `1.61.0` and its matching Chromium revision, exposes exactly `open_page` and `click_link`, fulfills only exact manifest URLs, blocks service workers, and aborts every other request. The public capability accepts only bounded canonical paths and opaque link identifiers; it does not expose arbitrary URLs, selectors, forms, downloads, screenshots, cookies, storage, HTML, or `evaluate()`.

The synchronous production wrapper starts only the bundled Browser MCP module and accepts no command, executable, browser path, launch argument, or site directory from the adapter. Every admitted transport call creates one stdio process and one real browser lifecycle, validates the exact server name, tools capability, tool inventory, schemas, metadata, and bounded response, and closes before returning. A default 10-second deadline bounds the operation. On timeout, the official transport terminates the process tree. Success-path tests continuously observe the real MCP worker, Playwright driver, and Chromium descendants and require all of them to be reaped. The timeout-path test substitutes a fixed hanging child that launches real Playwright and Chromium, continuously observes its descendants, and requires the complete observed tree to be reaped. Errors collapse to a fixed local failure without copying paths, page content, or Playwright diagnostics into trajectories.

Browser tools use the same episode action budget and facade handshake as workspace, terminal, and research tools. The facade emits only `browser_open_page` and `browser_click_link` call/result details. Runner revocation stops new browser admission and drains any in-flight call; because each call closes its browser before returning, the drain completes before adapter close and hidden verification. The M3B train task `browser-evaluation-001` proves the causal path through a real DOM click, real Docker terminal transformation, workspace output, and deterministic hidden verification. Production and Compose smokes run non-root with no network, no Linux capabilities, a read-only root filesystem, and writable tmpfs only.

The deadline and process-tree reap bound duration but do not provide hard host memory or PID quotas. A compromised local administrator, Docker daemon, Playwright/Chromium supply chain, or malicious same-process Python adapter is outside this slice. CAMEL, model inference, RL training, and arbitrary web browsing remain unimplemented.
