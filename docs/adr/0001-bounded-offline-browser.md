# ADR 0001: Use a Bounded Offline Browser Capability

- **Status:** Accepted
- **Date:** 2026-07-20
- **Decision owners:** AgentEnv Forge maintainers
- **Milestone:** M3B

## Context

AgentEnv Forge needed a Browser capability that could exercise real DOM behavior while preserving deterministic evaluation, hidden-oracle integrity, shared action accounting, and a fail-closed security boundary.

A conventional browser automation interface would introduce uncontrolled URLs, live network state, selectors, page scripts, downloads, storage, and large untrusted outputs. Those features would add nondeterminism and create paths around the task-safe capability boundary. A fake parser would be deterministic but would not prove that the Browser MCP, Playwright driver, Chromium process tree, timeout behavior, and Docker runtime work together.

The existing episode runner already defined the required lifecycle: capabilities share one action budget; admission is revoked and in-flight calls are drained before adapter close, terminal shutdown, and hidden verification. Browser integration had to preserve that lifecycle and the existing reward algebra.

## Decision

Implement Browser as a bounded offline MCP capability backed by real Playwright and Chromium.

### Public capability

Expose exactly two operations:

1. `open_page(path)` opens a canonical path on the bundled synthetic origin.
2. `click_link(link_id)` follows an opaque link ID from the current page.

Do not expose arbitrary URLs, selectors, forms, downloads, screenshots, cookies, storage, HTML, JavaScript evaluation, launch arguments, executable paths, or site directories.

### Deterministic site

Use a fixed synthetic HTTPS origin and an ordered set of canonical HTML fixtures. Bind the site version and fixture order to a canonical SHA-256 manifest. Fulfill only exact manifest routes, block service workers, and abort every other request.

### Transport and validation

Start only the bundled Browser MCP module. Reject invalid public inputs before transport admission. For each admitted transport call:

- create one stdio MCP process and one Chromium lifecycle;
- validate the exact server name, presence of the tools capability, tool inventory, input schemas, output schemas, annotations, metadata, structured result shape, and bounded field values;
- sanitize failures to a fixed local error;
- close the browser lifecycle before returning.

### Episode integration

Use the existing shared action budget and facade handshake. Emit only generic `browser_open_page` and `browser_click_link` call/result details; do not copy page content or hidden values into the trajectory.

At episode shutdown, revoke new Browser admission and drain in-flight Browser calls alongside Research, workspace, and terminal capabilities before adapter close and hidden verification. Do not introduce Browser-specific reward gates or modify reward weights.

### Runtime and verification

Use the Playwright-managed Chromium revision matching the pinned Python package. The production image stores the browser in the non-root runtime user's standard Playwright cache so MCP child processes can resolve it without inheriting an arbitrary browser-path environment variable.

Prove the implementation through:

- exact-type and bounded-schema unit tests;
- real MCP stdio calls with continuous observation and reaping checks for the MCP worker, Playwright driver, and Chromium descendants on success;
- real DOM navigation and link clicking;
- timeout reaping with a fixed hanging child that launches real Playwright and Chromium, while the parent continuously observes and requires the complete descendant tree to disappear;
- host and Docker Browser → terminal → workspace → hidden-verifier acceptance;
- a network-disabled, read-only, capability-dropped Compose smoke;
- pull-request and post-merge hosted CI.

## Invariants

Future changes must preserve all of the following unless a superseding ADR explicitly changes the boundary:

1. Browser cannot select an arbitrary origin or external URL.
2. The public inventory remains explicit and bounded.
3. Hostile values such as `bool` where an integer is expected are rejected by exact type.
4. All non-manifest browser requests are aborted.
5. Browser payloads and oracle values do not enter trajectory details.
6. Browser actions consume the shared episode budget.
7. Revocation and drain complete before adapter close and hidden verification.
8. Cleanup attempts every initialized resource and does not swallow cancellation, interrupts, or system-exit signals.
9. Browser failures do not alter reward algebra.
10. Real Docker and real process-tree behavior remain distinguished from fake or host-only tests.

## Consequences

### Positive

- Evaluation remains deterministic and reproducible while exercising a real browser engine.
- The capability is materially smaller than general browser automation and can be validated exactly.
- Per-call ownership makes timeout cleanup and process-tree assertions straightforward.
- The existing runner lifecycle and reward model remain intact.
- Production and CI exercise the same Playwright/Chromium pairing.

### Costs

- Starting MCP and Chromium per admitted call is slower than a persistent browser session.
- The production image is substantially larger and its build installs browser OS dependencies.
- The offline fixture must be versioned and rehashed when content changes.
- Hard host memory and PID isolation are not provided by this milestone.

## Alternatives considered

### General-purpose browser automation

Rejected because arbitrary URLs, selectors, scripts, downloads, and storage would violate the bounded deterministic capability model and substantially increase the attack surface.

### HTTP client or HTML parser only

Rejected because it would not exercise real DOM behavior, Playwright transport, Chromium process ownership, or browser cleanup.

### Persistent episode-wide Browser MCP and Chromium

Deferred. It could reduce latency but complicates state isolation, timeout ownership, revocation, and proof that every episode reaps its process tree.

### Pass `PLAYWRIGHT_BROWSERS_PATH` through the MCP child environment

Rejected. The MCP transport uses a restricted default environment, and forwarding a caller-controlled path would weaken the fixed-executable boundary. Installing the matching browser in the runtime user's standard cache preserves both operability and the security model.

## Scope not decided here

This ADR does not authorize arbitrary web browsing, CAMEL integration, model inference, RL training, or claims of hard host resource isolation. Each requires a separate bounded contract and, where it changes these invariants, a superseding ADR.
