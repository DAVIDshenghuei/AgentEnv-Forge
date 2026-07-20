from threading import Event, Thread

import pytest

from agentenv_forge.adapters import AgentRunResult
from agentenv_forge.runner import run_agent_episode
from agentenv_forge.tools import TerminalResult
from agentenv_forge.tools.browser import (
    BrowserActionLimitError,
    BrowserLink,
    BrowserPage,
    BrowserProtocol,
)


INDEX = BrowserPage(
    path="/index",
    title="Private Index Title",
    content="Private index content.",
    links=(
        BrowserLink(
            link_id="private-details",
            label="Private link label",
            path="/details",
        ),
    ),
)
DETAIL = BrowserPage(
    path="/details",
    title="Private Detail Title",
    content="Private detail content.",
    links=(),
)


class FakeBrowserClient(BrowserProtocol):
    def __init__(self) -> None:
        self.calls: list[tuple[object, ...]] = []

    def open_page(self, path: str) -> BrowserPage:
        self.calls.append(("open_page", path))
        return INDEX

    def click_link(self, current_path: str, link_id: str) -> BrowserPage:
        self.calls.append(("click_link", current_path, link_id))
        return DETAIL


def _paired(event_sink, detail, call):
    event_sink("tool_call", detail)
    result = call()
    event_sink("tool_result", detail)
    return result


def test_runner_browser_handshake_reward_and_payload_privacy(tmp_path) -> None:
    client = FakeBrowserClient()

    class Adapter:
        closed = False

        def run(self, task, tools, event_sink):
            _paired(event_sink, "browser_open_page", lambda: tools.open_page("/index"))
            _paired(
                event_sink,
                "browser_click_link",
                lambda: tools.click_link("private-details"),
            )
            source = _paired(
                event_sink,
                "read_text:input.txt",
                lambda: tools.read_text("input.txt"),
            )
            normalized = "\n".join(
                " ".join(line.lower().split())
                for line in source.splitlines()
                if line.strip()
            ) + "\n"
            _paired(
                event_sink,
                "write_text:result.txt",
                lambda: tools.write_text("result.txt", normalized),
            )
            return AgentRunResult("finished", None)

        def close(self):
            self.closed = True

    adapter = Adapter()
    trajectory = run_agent_episode(
        "text-normalization-001",
        adapter,
        42,
        workspace_root=tmp_path,
        browser_client=client,
    )

    assert trajectory.reward.total == 1.0
    assert adapter.closed is True
    assert client.calls == [
        ("open_page", "/index"),
        ("click_link", "/index", "private-details"),
    ]
    browser_events = [
        (event.kind, event.detail)
        for event in trajectory.events
        if event.detail.startswith("browser_")
    ]
    assert browser_events == [
        ("tool_call", "browser_open_page"),
        ("tool_result", "browser_open_page"),
        ("tool_call", "browser_click_link"),
        ("tool_result", "browser_click_link"),
    ]
    canonical = trajectory.canonical_content()
    for private_value in (
        "/index",
        "/details",
        "private-details",
        INDEX.title,
        INDEX.content,
        INDEX.links[0].label,
        DETAIL.title,
        DETAIL.content,
    ):
        assert private_value not in canonical


def test_browser_workspace_and_terminal_share_one_runner_budget(tmp_path) -> None:
    client = FakeBrowserClient()
    terminal_calls: list[tuple[str, ...]] = []

    class Adapter:
        closed = False

        def run(self, task, tools, event_sink):
            _paired(event_sink, "browser_open_page", lambda: tools.open_page("/index"))
            _paired(
                event_sink,
                "browser_click_link",
                lambda: tools.click_link("private-details"),
            )
            for _ in range(2):
                _paired(
                    event_sink,
                    "terminal_execute",
                    lambda: tools.execute(("true",)),
                )
            for _ in range(task.max_actions - 4):
                _paired(event_sink, "list_files", tools.list_files)
            event_sink("tool_call", "browser_open_page")
            with pytest.raises(
                BrowserActionLimitError,
                match="^browser action budget exhausted$",
            ):
                tools.open_page("/index")
            return AgentRunResult("action_limit", None)

        def close(self):
            self.closed = True

    adapter = Adapter()
    trajectory = run_agent_episode(
        "text-normalization-001",
        adapter,
        42,
        workspace_root=tmp_path,
        terminal_command_runner=lambda argv: (
            terminal_calls.append(argv) or TerminalResult(0, "", "")
        ),
        browser_client=client,
    )

    assert trajectory.termination_reason == "action_limit"
    assert adapter.closed is True
    assert terminal_calls == [("true",), ("true",)]
    assert client.calls == [
        ("open_page", "/index"),
        ("click_link", "/index", "private-details"),
    ]


def test_invalid_browser_client_precedes_terminal_environment_start(tmp_path) -> None:
    lifecycle: list[str] = []

    class Adapter:
        closed = False

        def run(self, task, tools, event_sink):
            raise AssertionError("adapter must not run")

        def close(self):
            self.closed = True

    def factory(workspace):
        lifecycle.append("factory")
        raise AssertionError("environment factory must not run")

    adapter = Adapter()
    with pytest.raises(ValueError, match="^invalid browser client$"):
        run_agent_episode(
            "text-normalization-001",
            adapter,
            42,
            workspace_root=tmp_path,
            terminal_environment_factory=factory,
            browser_client=object(),
        )
    assert lifecycle == []
    assert adapter.closed is False


def test_browser_revoke_drains_before_adapter_close_and_hidden_verifier(tmp_path) -> None:
    entered = Event()
    release = Event()
    call_finished = Event()
    episode_finished = Event()
    lifecycle: list[str] = []
    trajectories = []
    episode_errors: list[BaseException] = []

    class BlockingClient(FakeBrowserClient):
        def open_page(self, path: str) -> BrowserPage:
            self.calls.append(("open_page", path))
            entered.set()
            assert release.wait(timeout=5)
            lifecycle.append("browser_call_finished")
            return INDEX

    client = BlockingClient()

    class Adapter:
        closed = False
        retained_tools = None

        def run(self, task, tools, event_sink):
            self.retained_tools = tools

            def call_browser():
                try:
                    _paired(
                        event_sink,
                        "browser_open_page",
                        lambda: tools.open_page("/index"),
                    )
                except ValueError:
                    pass
                finally:
                    call_finished.set()

            Thread(target=call_browser).start()
            assert entered.wait(timeout=5)
            return AgentRunResult("finished", None)

        def close(self):
            assert call_finished.is_set()
            lifecycle.append("adapter_close")
            self.closed = True

    adapter = Adapter()

    def run_episode():
        try:
            trajectories.append(
                run_agent_episode(
                    "text-normalization-001",
                    adapter,
                    42,
                    workspace_root=tmp_path,
                    browser_client=client,
                )
            )
        except BaseException as error:
            episode_errors.append(error)
        finally:
            episode_finished.set()

    episode_thread = Thread(target=run_episode)
    episode_thread.start()
    if not entered.wait(timeout=5):
        episode_thread.join(timeout=5)
        if episode_errors:
            raise episode_errors[0]
        pytest.fail("browser call did not start")
    assert not episode_finished.is_set()
    assert adapter.closed is False
    release.set()
    episode_thread.join(timeout=5)

    assert not episode_thread.is_alive()
    assert lifecycle[:2] == ["browser_call_finished", "adapter_close"]
    assert trajectories[0].reward.artifact_exists == 0.0
    assert trajectories[0].events[-1].detail == "deterministic verifier completed"
    assert adapter.closed is True
    calls_after_episode = tuple(client.calls)
    with pytest.raises(ValueError, match="^browser tools revoked$"):
        adapter.retained_tools.open_page("/index")
    assert tuple(client.calls) == calls_after_episode
