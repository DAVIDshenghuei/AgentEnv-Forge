from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
DESIGN = ROOT / "DESIGN.md"
CI_WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"
GIT_ATTRIBUTES = ROOT / ".gitattributes"
_UNIMPLEMENTED = (
    "CAMEL, model inference, RL training, and arbitrary web browsing remain unimplemented"
)
_EXPECTED_CI_WORKFLOW = """name: CI

on:
  push:
    branches:
      - main
  pull_request:

permissions:
  contents: read

concurrency:
  group: ci-${{ github.workflow }}-${{ github.event.pull_request.head.repo.full_name || github.repository }}-${{ github.ref_type }}-${{ github.head_ref || github.ref_name }}
  cancel-in-progress: true

jobs:
  test:
    runs-on: ubuntu-latest
    timeout-minutes: 20
    steps:
      - name: Check out repository
        uses: actions/checkout@df4cb1c069e1874edd31b4311f1884172cec0e10 # v6
        with:
          persist-credentials: false

      - name: Set up uv
        uses: astral-sh/setup-uv@37802adc94f370d6bfd71619e3f0bf239e1f3b78 # v7
        with:
          version: "0.8.11"

      - name: Install frozen development environment
        run: uv sync --extra dev --frozen

      - name: Install Playwright Chromium
        run: uv run --no-sync playwright install --with-deps chromium

      - name: Run Research MCP stdio contract
        run: uv run --no-sync pytest tests/test_research_mcp_stdio.py -q

      - name: Run Browser MCP stdio contract
        run: uv run --no-sync pytest tests/test_browser_mcp_stdio.py -q

      - name: Build dedicated sandbox image
        run: docker build --no-cache -f sandbox/Dockerfile -t agentenv-forge-sandbox:test .

      - name: Run full test suite with Docker integration
        run: AGENTENV_FORGE_DOCKER_INTEGRATION=1 uv run --no-sync pytest -q

      - name: Build Compose services
        run: docker compose build

      - name: Run Compose smoke tests
        run: |
          docker compose run --rm smoke
          docker compose run --rm browser-smoke
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


def test_readme_documents_the_implemented_offline_research_mcp() -> None:
    readme = README.read_text(encoding="utf-8")

    assert "## Offline Research MCP" in readme
    assert "Status: **implemented**" in readme
    assert "fixed, versioned corpus" in readme
    assert "canonical SHA-256 manifest" in readme
    assert "does not accept an arbitrary MCP command" in readme
    assert "10-second operation deadline" in readme
    assert "two-second graceful shutdown window" in readme
    assert "exact server identity and tool inventory" in readme
    assert "official MCP stdio transport" in readme
    assert "`search_papers` and `get_paper`" in readme
    assert "shared episode action budget" in readme
    assert "generic, payload-free trajectory events" in readme
    assert "one stdio process and session per call" in readme
    assert "closed before return and before hidden verification" in readme
    assert "`research-synthesis-001`" in readme


def test_design_documents_the_implemented_offline_research_lifecycle() -> None:
    design = DESIGN.read_text(encoding="utf-8")

    assert "## Implemented offline Research MCP lifecycle" in design
    assert "fixed, versioned corpus" in design
    assert "canonical SHA-256 manifest" in design
    assert "does not accept an arbitrary MCP command" in design
    assert "10-second operation deadline" in design
    assert "two-second graceful shutdown window" in design
    assert "exact server identity and tool inventory" in design
    assert "official MCP stdio transport" in design
    assert "`search_papers` and `get_paper`" in design
    assert "shared episode action budget" in design
    assert "generic, payload-free trajectory events" in design
    assert "one stdio process and session per call" in design
    assert "closed before return and before hidden verification" in design
    assert "`research-synthesis-001`" in design


def test_readme_documents_the_implemented_offline_browser_mcp() -> None:
    readme = README.read_text(encoding="utf-8")

    assert "## Offline Browser MCP" in readme
    assert "Playwright `1.61.0`" in readme
    assert "fixed synthetic HTTPS origin" in readme
    assert "`open_page` and `click_link`" in readme
    assert "aborts every other request" in readme
    assert "Each admitted transport call owns one MCP stdio process" in readme
    assert "process tree" in readme
    assert "one episode action budget" in readme
    assert "`browser-evaluation-001`" in readme
    assert "does not claim hard host memory or PID isolation" in readme


def test_design_documents_the_implemented_offline_browser_lifecycle() -> None:
    design = DESIGN.read_text(encoding="utf-8")

    assert "## Implemented offline Browser MCP lifecycle" in design
    assert "Playwright `1.61.0`" in design
    assert "fixed synthetic HTTPS site" in design
    assert "`open_page` and `click_link`" in design
    assert "aborts every other request" in design
    assert "Every admitted transport call creates one stdio process" in design
    assert "process tree" in design
    assert "same episode action budget" in design
    assert "`browser-evaluation-001`" in design
    assert "do not provide hard host memory or PID quotas" in design


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


def test_github_workflows_are_checked_out_with_lf_bytes() -> None:
    assert "/.github/workflows/*.yml text eol=lf" in GIT_ATTRIBUTES.read_text(
        encoding="utf-8"
    ).splitlines()
