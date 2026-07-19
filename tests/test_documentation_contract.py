from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
DESIGN = ROOT / "DESIGN.md"
CI_WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"
_UNIMPLEMENTED = "Browser, MCP, CAMEL, model inference, and RL training are not implemented"
_EXPECTED_CI_WORKFLOW = """name: CI

on:
  push:
  pull_request:

permissions:
  contents: read

concurrency:
  group: ci-${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true

jobs:
  test:
    runs-on: ubuntu-latest
    timeout-minutes: 20
    steps:
      - name: Check out repository
        uses: actions/checkout@34e114876b0b11c390a56381ad16ebd13914f8d5 # v4
        with:
          persist-credentials: false

      - name: Set up uv
        uses: astral-sh/setup-uv@d0cc045d04ccac9d8b7881df0226f9e82c39688e # v6
        with:
          version: "0.8.11"

      - name: Install frozen development environment
        run: uv sync --extra dev --frozen

      - name: Build dedicated sandbox image
        run: docker build --no-cache -f sandbox/Dockerfile -t agentenv-forge-sandbox:test .

      - name: Run full test suite with Docker integration
        run: AGENTENV_FORGE_DOCKER_INTEGRATION=1 uv run --no-sync pytest -q

      - name: Build Compose services
        run: docker compose build

      - name: Run Compose smoke test
        run: docker compose run --rm smoke
"""


def test_readme_documents_the_implemented_terminal_environment() -> None:
    readme = README.read_text(encoding="utf-8")

    assert "## Terminal-agent environment" in readme
    assert "Status: **implemented**" in readme
    assert "read-only `/host-workspace`" in readme
    assert "4 MiB" in readme
    assert "256-inode" in readme
    assert "environment cleanup completes before hidden verification" in readme


def test_design_documents_the_implemented_terminal_lifecycle() -> None:
    design = DESIGN.read_text(encoding="utf-8")

    assert "## Implemented terminal-agent lifecycle" in design
    assert "revoke and drain capabilities" in design
    assert "close the Docker environment" in design
    assert "run hidden verification" in design
    assert "primary failure" in design


def test_docs_fail_closed_about_unimplemented_next_milestone() -> None:
    for document in (README, DESIGN):
        contents = document.read_text(encoding="utf-8")
        assert _UNIMPLEMENTED in contents, document.name
        assert "a future Agent adapter" not in contents, document.name
        assert "future Agent adapters" not in contents, document.name


def test_design_distinguishes_command_sync_from_environment_close() -> None:
    design = DESIGN.read_text(encoding="utf-8")

    assert "Each completed terminal command synchronizes and validates" in design
    assert "Closing the Docker environment removes the container" in design
    assert "Closing synchronizes" not in design


def test_ci_matches_the_reviewed_fail_closed_workflow() -> None:
    assert CI_WORKFLOW.read_bytes() == _EXPECTED_CI_WORKFLOW.encode("utf-8")
