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
- The controlled action is synchronous. A future Agent adapter must terminate and revoke all workspace access before verification begins; concurrent Agent/verifier access is outside this slice's security contract.
- Environment and agent failure fields are separate. A successful episode sets both to null.
- Unknown tasks/actions, invalid schemas, holdout tasks, and non-resource verifier/schema errors fail closed: no successful trajectory is returned or written.
- Runtime output contains the split label but never the hidden expected content. Holdout task specifications are rejected before workspace creation, preventing holdout leakage.

## Future adapter seam

The boundary to extend is the controlled action step: a future action adapter can receive a task-safe workspace and return an action result while reset, event capture, verifier, reward schemas, and trajectory serialization remain unchanged. CAMEL may later orchestrate agents, SGLang may supply inference, and AReaL may consume trajectories for training. Those integrations must remain adapters behind this boundary and are intentionally not implemented here.
