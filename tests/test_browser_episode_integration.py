import os
import sys

import pytest

from agentenv_forge.adapters import AgentRunResult
from agentenv_forge.mcp.browser_client import StdioBrowserClient
from agentenv_forge.mcp.browser_site import BROWSER_SITE_VERSION
from agentenv_forge.runner import load_task, run_agent_episode, run_docker_agent_episode
from agentenv_forge.sandbox import BoundedProcessExecutor
from agentenv_forge.tools import BrowserPage, BrowserProtocol, TerminalResult


EXPECTED_OUTPUT = "Offline Browser Evaluation | 2024 | external requests aborted\n"
DETAIL_PATH = "/guides/browser-isolation"
DETAIL_TITLE = "Offline Browser Evaluation"
DETAIL_CONTENT = (
    "Published: 2024\nBrowser requests are fulfilled only from the bundled "
    "origin; every external request is aborted."
)
SCRIPT = (
    "import re,sys;"
    "sys.stdout.reconfigure(newline='');"
    "year=re.search(r'Published: ([0-9]{4})',sys.argv[2]).group(1);"
    "status='external requests aborted' "
    "if 'every external request is aborted' in sys.argv[2] else 'invalid';"
    "print(f'{sys.argv[1]} | {year} | {status}')"
)
EXPECTED_ARGV = (sys.executable, "-c", SCRIPT, DETAIL_TITLE, DETAIL_CONTENT)


def _paired(event_sink, detail, call):
    event_sink("tool_call", detail)
    result = call()
    event_sink("tool_result", detail)
    return result


def _string_values(value):
    if type(value) is str:
        yield value
    elif type(value) is dict:
        for key, item in value.items():
            yield from _string_values(key)
            yield from _string_values(item)
    elif type(value) in {list, tuple}:
        for item in value:
            yield from _string_values(item)


def _assert_private_text_absent(model, private_values) -> None:
    strings = tuple(_string_values(model.model_dump(mode="python")))
    for private_value in private_values:
        assert all(private_value not in value for value in strings)


class BrowserEvaluationAdapter:
    def __init__(self, python_executable: str = sys.executable) -> None:
        self.closed = False
        self.python_executable = python_executable

    def run(self, task, tools, event_sink):
        _paired(event_sink, "browser_open_page", lambda: tools.open_page("/"))
        page = _paired(
            event_sink,
            "browser_click_link",
            lambda: tools.click_link("browser-isolation"),
        )
        terminal = _paired(
            event_sink,
            "terminal_execute",
            lambda: tools.execute(
                (
                    self.python_executable,
                    "-c",
                    SCRIPT,
                    page.title,
                    page.content,
                )
            ),
        )
        assert terminal.exit_code == 0
        _paired(
            event_sink,
            "write_text:browser-report.txt",
            lambda: tools.write_text("browser-report.txt", terminal.stdout),
        )
        return AgentRunResult("finished", None)

    def close(self) -> None:
        self.closed = True


def _assert_success(trajectory, adapter, client) -> None:
    assert trajectory.reward.total == 1.0
    assert trajectory.termination_reason == "finished"
    assert adapter.closed is True
    assert client.has_active_session is False
    assert [
        (event.kind, event.detail)
        for event in trajectory.events
        if event.kind in {"tool_call", "tool_result"}
    ] == [
        (kind, detail)
        for detail in (
            "browser_open_page",
            "browser_click_link",
            "terminal_execute",
            "write_text:browser-report.txt",
        )
        for kind in ("tool_call", "tool_result")
    ]
    _assert_private_text_absent(
        trajectory,
        (
            DETAIL_PATH,
            DETAIL_TITLE,
            DETAIL_CONTENT,
            "Published: 2024",
            EXPECTED_OUTPUT,
        ),
    )


def test_browser_evaluation_task_public_contract_hides_oracle() -> None:
    task = load_task("browser-evaluation-001")
    public_task = task.to_public_task()

    assert task.split == "train"
    assert task.version == BROWSER_SITE_VERSION
    assert task.max_actions == 4
    assert public_task.input_artifacts == ("browser-instructions.txt",)
    assert public_task.allowed_artifacts == ("browser-report.txt",)
    assert "/" in public_task.instruction
    assert "browser-isolation" in public_task.instruction
    assert "terminal" in public_task.instruction.lower()
    assert "browser-report.txt" in public_task.instruction
    assert "expected_content" not in type(public_task).model_fields
    assert "initial_files" not in type(public_task).model_fields
    _assert_private_text_absent(public_task, (task.expected_content,))


def test_host_browser_episode_acceptance_uses_real_bounded_terminal(tmp_path) -> None:
    executor = BoundedProcessExecutor(max_output_bytes=65_536)
    terminal_calls: list[tuple[str, ...]] = []

    def terminal_runner(argv: tuple[str, ...]) -> TerminalResult:
        assert argv == EXPECTED_ARGV
        terminal_calls.append(argv)
        result = executor(argv, 5.0)
        return TerminalResult(result.exit_code, result.stdout, result.stderr)

    adapter = BrowserEvaluationAdapter()
    client = StdioBrowserClient()
    trajectory = run_agent_episode(
        task_id="browser-evaluation-001",
        adapter=adapter,
        seed=42,
        workspace_root=tmp_path,
        terminal_command_runner=terminal_runner,
        browser_client=client,
    )

    _assert_success(trajectory, adapter, client)
    assert terminal_calls == [EXPECTED_ARGV]


def test_browser_payload_alone_cannot_satisfy_hidden_verifier(tmp_path) -> None:
    class OracleLikeClient(BrowserProtocol):
        def open_page(self, path: str) -> BrowserPage:
            return BrowserPage(
                path="/",
                title="Untrusted browser output",
                content=EXPECTED_OUTPUT,
                links=(),
            )

        def click_link(self, current_path: str, link_id: str) -> BrowserPage:
            raise AssertionError("click must not be called")

    class BrowserOnlyAdapter:
        closed = False

        def run(self, task, tools, event_sink):
            _paired(event_sink, "browser_open_page", lambda: tools.open_page("/"))
            return AgentRunResult("finished", None)

        def close(self) -> None:
            self.closed = True

    adapter = BrowserOnlyAdapter()
    trajectory = run_agent_episode(
        task_id="browser-evaluation-001",
        adapter=adapter,
        seed=42,
        workspace_root=tmp_path,
        browser_client=OracleLikeClient(),
    )

    assert adapter.closed is True
    assert trajectory.reward.total < 1.0
    assert trajectory.artifacts == []
    _assert_private_text_absent(trajectory, (EXPECTED_OUTPUT,))


@pytest.mark.skipif(
    os.environ.get("AGENTENV_FORGE_DOCKER_INTEGRATION") != "1",
    reason="requires the explicitly built sandbox image and Docker daemon",
)
def test_docker_browser_episode_acceptance_and_cleanup(tmp_path) -> None:
    adapter = BrowserEvaluationAdapter(python_executable="python")
    client = StdioBrowserClient()
    trajectory = run_docker_agent_episode(
        task_id="browser-evaluation-001",
        adapter=adapter,
        seed=42,
        workspace_root=tmp_path,
        image="agentenv-forge-sandbox:test",
        browser_client=client,
    )

    _assert_success(trajectory, adapter, client)
    details = [event.detail for event in trajectory.events]
    assert details.index("terminal environment started") < details.index(
        "adapter started"
    )
    assert details.index("adapter stopped") < details.index(
        "terminal environment stopped"
    )
    assert details.index("terminal environment stopped") < details.index(
        "deterministic verifier completed"
    )

    containers = BoundedProcessExecutor(max_output_bytes=65_536)(
        (
            "docker",
            "ps",
            "--all",
            "--quiet",
            "--filter",
            "ancestor=agentenv-forge-sandbox:test",
        ),
        5.0,
    )
    assert containers.exit_code == 0
    assert containers.stdout.strip() == ""
