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

Browser, MCP, CAMEL, model inference, and RL training are not implemented. Any later integrations must remain adapters behind the task-safe capability boundary and must preserve lifecycle ordering, hidden-oracle isolation, and deterministic verification.
