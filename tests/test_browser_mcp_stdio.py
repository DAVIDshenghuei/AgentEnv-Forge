import asyncio
import json
import os
import sys
import time
from threading import Thread

import psutil
import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

import agentenv_forge.mcp.browser_client as client_module
import agentenv_forge.mcp.browser_server as server_module
from agentenv_forge.mcp.browser_client import StdioBrowserClient
from agentenv_forge.tools.browser import BrowserLink, BrowserPage


DIGEST = "b80be2fdc2d6c318a40ffa297201132b928c0037cbefd5f225f01f335f553b05"
SERVER_NAME = "agentenv-forge-browser-1.0.0-sha256-" + DIGEST
INDEX_PAGE = BrowserPage(
    path="/",
    title="Offline Evaluation Library",
    content=(
        "Offline Evaluation Library\n"
        "Browse deterministic offline evaluation guides.\n"
        "Browser isolation"
    ),
    links=(
        BrowserLink(
            link_id="browser-isolation",
            label="Browser isolation",
            path="/guides/browser-isolation",
        ),
    ),
)
DETAIL_PAGE = BrowserPage(
    path="/guides/browser-isolation",
    title="Offline Browser Evaluation",
    content=(
        "Published: 2024\n"
        "Browser requests are fulfilled only from the bundled origin; "
        "every external request is aborted."
    ),
    links=(),
)
SCHEMAS = {
    "open_page": {
        "properties": {"path": {"title": "Path", "type": "string"}},
        "required": ["path"],
        "title": "open_pageArguments",
        "type": "object",
    },
    "click_link": {
        "properties": {
            "current_path": {"title": "Current Path", "type": "string"},
            "link_id": {"title": "Link Id", "type": "string"},
        },
        "required": ["current_path", "link_id"],
        "title": "click_linkArguments",
        "type": "object",
    },
}


def _payload(result):
    assert result.structuredContent is not None
    assert set(result.structuredContent) == {"result"}
    text = [block.text for block in result.content if hasattr(block, "text")]
    assert len(text) == 1
    parsed = json.loads(text[0])
    assert parsed == result.structuredContent["result"]
    return parsed


def _page_payload(page: BrowserPage) -> dict[str, object]:
    return {
        "path": page.path,
        "title": page.title,
        "content": page.content,
        "links": [
            {"link_id": link.link_id, "label": link.label, "path": link.path}
            for link in page.links
        ],
    }


def test_browser_mcp_real_playwright_stdio_roundtrip() -> None:
    async def roundtrip() -> None:
        parameters = StdioServerParameters(
            command=sys.executable,
            args=["-m", "agentenv_forge.mcp.browser_server"],
        )
        async with stdio_client(parameters) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                initialized = await session.initialize()
                assert initialized.serverInfo.name == SERVER_NAME
                assert initialized.capabilities.tools is not None
                listed = await session.list_tools()
                assert tuple(sorted(tool.name for tool in listed.tools)) == (
                    "click_link",
                    "open_page",
                )
                assert {tool.name: tool.inputSchema for tool in listed.tools} == SCHEMAS
                expected_metadata = {
                    name: {
                        "title": None,
                        "description": "",
                        "outputSchema": None,
                        "icons": None,
                        "annotations": None,
                        "execution": None,
                        "_meta": None,
                    }
                    for name in SCHEMAS
                }
                metadata = {
                    tool.name: {
                        key: value
                        for key, value in tool.model_dump(
                            mode="json", by_alias=True
                        ).items()
                        if key not in {"name", "inputSchema"}
                    }
                    for tool in listed.tools
                }
                assert metadata == expected_metadata
                StdioBrowserClient._validate_inventory(listed.tools)
                impostors = list(listed.tools)
                impostors[0] = impostors[0].model_copy(
                    update={"description": "impostor"}
                )
                with pytest.raises(ValueError, match="^browser MCP call failed$"):
                    StdioBrowserClient._validate_inventory(impostors)

                opened = await session.call_tool("open_page", {"path": "/"})
                assert opened.isError is False
                assert _payload(opened) == _page_payload(INDEX_PAGE)
                clicked = await session.call_tool(
                    "click_link",
                    {"current_path": "/", "link_id": "browser-isolation"},
                )
                assert clicked.isError is False
                assert _payload(clicked) == _page_payload(DETAIL_PAGE)

                secret = "/secret-do-not-echo"
                for tool, arguments in (
                    ("open_page", {"path": secret}),
                    ("open_page", {"path": "https://evil.invalid"}),
                    (
                        "click_link",
                        {"current_path": "/", "link_id": "secret-link"},
                    ),
                ):
                    failure = await session.call_tool(tool, arguments)
                    assert failure.isError is True
                    error_text = " ".join(
                        block.text
                        for block in failure.content
                        if hasattr(block, "text")
                    )
                    assert error_text == "browser MCP call failed"
                    assert secret not in error_text

    asyncio.run(roundtrip())


def test_sync_browser_client_returns_exact_pages_and_clears_active_state() -> None:
    with pytest.raises(TypeError):
        StdioBrowserClient(
            StdioServerParameters(
                command=sys.executable,
                args=["-m", "agentenv_forge.mcp.browser_server"],
            )
        )
    client = StdioBrowserClient()
    assert client.has_active_session is False
    assert client.open_page("/") == INDEX_PAGE
    assert client.has_active_session is False
    assert client.click_link("/", "browser-isolation") == DETAIL_PAGE
    assert client.has_active_session is False
    secret = "/secret-wrapper-id"
    with pytest.raises(ValueError, match="^browser MCP call failed$") as failure:
        client.open_page(secret)
    assert secret not in str(failure.value)
    assert client.has_active_session is False

    async def reject_running_loop() -> None:
        with pytest.raises(ValueError, match="^browser MCP call failed$"):
            client.open_page("/")

    asyncio.run(reject_running_loop())
    assert client.has_active_session is False


def test_successful_browser_call_reaps_observed_real_process_tree() -> None:
    parent = psutil.Process(os.getpid())
    baseline = {child.pid for child in parent.children(recursive=True)}
    client = StdioBrowserClient()
    pages: list[BrowserPage] = []
    failures: list[BaseException] = []

    def call_browser() -> None:
        try:
            pages.append(client.open_page("/"))
        except BaseException as error:
            failures.append(error)

    thread = Thread(target=call_browser)
    thread.start()
    observed: dict[int, str] = {}
    deadline = time.monotonic() + 10.0
    while thread.is_alive() and time.monotonic() < deadline:
        try:
            children = parent.children(recursive=True)
        except psutil.Error:
            children = []
        for child in children:
            if child.pid in baseline:
                continue
            try:
                command = " ".join(child.cmdline()).lower()
            except psutil.Error:
                command = ""
            observed[child.pid] = command
        time.sleep(0.005)
    thread.join(timeout=2.0)

    assert not thread.is_alive()
    assert failures == []
    assert pages == [INDEX_PAGE]
    assert len(observed) >= 3
    assert any(
        "chromium" in command
        or "chrome-headless-shell" in command
        or "ms-playwright" in command
        for command in observed.values()
    )
    reap_deadline = time.monotonic() + 5.0
    while time.monotonic() < reap_deadline:
        alive = [pid for pid in observed if psutil.pid_exists(pid)]
        if not alive:
            break
        time.sleep(0.02)
    else:
        pytest.fail(f"successful browser process tree was not reaped: {alive!r}")
    assert client.has_active_session is False


def test_sync_browser_client_timeout_and_malformed_payload_are_sanitized(
    monkeypatch,
) -> None:
    async def hang_forever(self, tool_name, arguments):
        await asyncio.Event().wait()

    with pytest.raises(ValueError, match="^browser MCP call failed$"):
        StdioBrowserClient(timeout_seconds=True)
    client = StdioBrowserClient(timeout_seconds=0.05)
    monkeypatch.setattr(StdioBrowserClient, "_call", hang_forever)
    started = time.monotonic()
    with pytest.raises(ValueError, match="^browser MCP call failed$"):
        client.open_page("/")
    assert time.monotonic() - started < 1.0
    assert client.has_active_session is False

    async def malformed(self, tool_name, arguments):
        return {
            "path": "/",
            "title": "Offline Evaluation Library",
            "content": "x" * 65_537,
            "links": [],
            "hidden": "must not cross",
        }

    monkeypatch.setattr(StdioBrowserClient, "_call", malformed)
    with pytest.raises(ValueError, match="^browser MCP call failed$"):
        client.open_page("/")
    assert client.has_active_session is False


def test_sync_browser_client_timeout_reaps_fixed_real_chromium_tree(
    monkeypatch,
) -> None:
    script = (
        "import time;"
        "from playwright.sync_api import sync_playwright;"
        "playwright=sync_playwright().start();"
        "browser=playwright.chromium.launch(headless=True);"
        "time.sleep(30)"
    )
    client = StdioBrowserClient(timeout_seconds=1.5)
    monkeypatch.setattr(
        client_module,
        "_fixed_server_parameters",
        lambda: StdioServerParameters(command=sys.executable, args=["-c", script]),
    )
    parent = psutil.Process(os.getpid())
    baseline = {child.pid for child in parent.children(recursive=True)}
    failures: list[BaseException] = []
    elapsed: list[float] = []

    def call_browser() -> None:
        started = time.monotonic()
        try:
            client.open_page("/")
        except BaseException as error:
            failures.append(error)
        finally:
            elapsed.append(time.monotonic() - started)

    thread = Thread(target=call_browser)
    thread.start()
    observed: dict[int, str] = {}
    deadline = time.monotonic() + 7.0
    while thread.is_alive() and time.monotonic() < deadline:
        try:
            children = parent.children(recursive=True)
        except psutil.Error:
            children = []
        for child in children:
            if child.pid in baseline:
                continue
            try:
                command = " ".join(child.cmdline()).lower()
            except psutil.Error:
                command = ""
            observed[child.pid] = command
        time.sleep(0.005)
    thread.join(timeout=2.0)

    assert not thread.is_alive()
    assert len(failures) == 1
    assert type(failures[0]) is ValueError
    assert str(failures[0]) == "browser MCP call failed"
    assert len(elapsed) == 1 and elapsed[0] < 6.0
    assert len(observed) >= 3
    assert any(
        "chromium" in command
        or "chrome-headless-shell" in command
        or "ms-playwright" in command
        for command in observed.values()
    )
    reap_deadline = time.monotonic() + 5.0
    while time.monotonic() < reap_deadline:
        alive = [pid for pid in observed if psutil.pid_exists(pid)]
        if not alive:
            break
        time.sleep(0.02)
    else:
        pytest.fail(f"timed-out browser process tree was not reaped: {alive!r}")
    assert client.has_active_session is False


def test_browser_render_does_not_swallow_cleanup_keyboard_interrupt(
    monkeypatch,
) -> None:
    lifecycle: list[str] = []

    class PrimaryFailure(Exception):
        pass

    class FakePage:
        async def goto(self, url, wait_until):
            lifecycle.append("page_goto")
            raise PrimaryFailure("primary browser failure")

        async def close(self) -> None:
            lifecycle.append("page_close")
            raise KeyboardInterrupt("cleanup interrupted")

    class FakeContext:
        async def route(self, pattern, handler) -> None:
            assert pattern == "**/*"

        async def new_page(self):
            return FakePage()

        async def close(self) -> None:
            lifecycle.append("context_close")

    class FakeBrowser:
        async def new_context(self, **kwargs):
            assert kwargs == {"service_workers": "block"}
            return FakeContext()

        async def close(self) -> None:
            lifecycle.append("browser_close")

    class FakeChromium:
        async def launch(self, **kwargs):
            assert kwargs == {"headless": True}
            return FakeBrowser()

    class FakePlaywright:
        chromium = FakeChromium()

        async def start(self):
            return self

        async def stop(self) -> None:
            lifecycle.append("playwright_stop")

    monkeypatch.setattr(server_module, "async_playwright", lambda: FakePlaywright())

    with pytest.raises(KeyboardInterrupt, match="^cleanup interrupted$"):
        asyncio.run(server_module._render("/"))

    assert lifecycle == [
        "page_goto",
        "page_close",
        "context_close",
        "browser_close",
        "playwright_stop",
    ]


def test_browser_render_preserves_primary_failure_and_attempts_all_cleanup(
    monkeypatch,
) -> None:
    lifecycle: list[str] = []

    class PrimaryFailure(Exception):
        pass

    class CleanupFailure(Exception):
        pass

    class FakeRoute:
        async def abort(self) -> None:
            lifecycle.append("route_abort")

        async def fulfill(self, **kwargs) -> None:
            raise AssertionError("external request must not be fulfilled")

    class FakeRequest:
        url = "https://external.invalid/tracker.png"

    class FakePage:
        async def goto(self, url, wait_until):
            lifecycle.append("page_goto")
            await context.handler(FakeRoute(), FakeRequest())
            raise PrimaryFailure("primary browser failure")

        async def close(self) -> None:
            lifecycle.append("page_close")
            raise CleanupFailure("page close failed")

    class FakeContext:
        handler = None

        async def route(self, pattern, handler) -> None:
            assert pattern == "**/*"
            self.handler = handler

        async def new_page(self):
            return FakePage()

        async def close(self) -> None:
            lifecycle.append("context_close")

    context = FakeContext()

    class FakeBrowser:
        async def new_context(self, **kwargs):
            assert kwargs == {"service_workers": "block"}
            return context

        async def close(self) -> None:
            lifecycle.append("browser_close")

    class FakeChromium:
        async def launch(self, **kwargs):
            assert kwargs == {"headless": True}
            return FakeBrowser()

    class FakePlaywright:
        chromium = FakeChromium()

        async def start(self):
            return self

        async def stop(self) -> None:
            lifecycle.append("playwright_stop")

    monkeypatch.setattr(server_module, "async_playwright", lambda: FakePlaywright())

    with pytest.raises(PrimaryFailure, match="^primary browser failure$"):
        asyncio.run(server_module._render("/"))

    assert lifecycle == [
        "page_goto",
        "route_abort",
        "page_close",
        "context_close",
        "browser_close",
        "playwright_stop",
    ]
