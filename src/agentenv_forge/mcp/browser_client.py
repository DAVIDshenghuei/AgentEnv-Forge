import asyncio
import json
import math
import os
import sys
from threading import Lock

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import TextContent, Tool

from ..tools.browser import BrowserLink, BrowserPage
from .browser_site import (
    BROWSER_SITE_SHA256,
    BROWSER_SITE_VERSION,
    browser_link_target,
    browser_site_page,
)


_FAILURE = "browser MCP call failed"
_TOOLS = ("click_link", "open_page")
_MAX_RESPONSE_BYTES = 70_000
_SERVER_NAME = (
    f"agentenv-forge-browser-{BROWSER_SITE_VERSION}-sha256-"
    f"{BROWSER_SITE_SHA256}"
)
_INPUT_SCHEMAS = {
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
_TOOL_METADATA = {
    name: {
        "title": None,
        "description": "",
        "outputSchema": None,
        "icons": None,
        "annotations": None,
        "execution": None,
        "_meta": None,
    }
    for name in _TOOLS
}


def _fixed_server_parameters() -> StdioServerParameters:
    return StdioServerParameters(
        command=sys.executable,
        args=["-m", "agentenv_forge.mcp.browser_server"],
    )


class StdioBrowserClient:
    __slots__ = ("_active", "_lock", "_timeout_seconds")

    def __init__(self, *, timeout_seconds: float = 10.0) -> None:
        if (
            type(timeout_seconds) is not float
            or not math.isfinite(timeout_seconds)
            or not 0 < timeout_seconds <= 60
        ):
            raise ValueError(_FAILURE)
        self._timeout_seconds = timeout_seconds
        self._lock = Lock()
        self._active = False

    @property
    def has_active_session(self) -> bool:
        with self._lock:
            return self._active

    @staticmethod
    def _validate_inventory(tools: object) -> None:
        if type(tools) is not list or len(tools) != len(_TOOLS):
            raise ValueError(_FAILURE)
        if any(type(tool) is not Tool for tool in tools):
            raise ValueError(_FAILURE)
        names = tuple(sorted(tool.name for tool in tools))
        schemas = {tool.name: tool.inputSchema for tool in tools}
        metadata = {
            tool.name: {
                key: value
                for key, value in tool.model_dump(mode="json", by_alias=True).items()
                if key not in {"name", "inputSchema"}
            }
            for tool in tools
        }
        if names != _TOOLS or schemas != _INPUT_SCHEMAS or metadata != _TOOL_METADATA:
            raise ValueError(_FAILURE)

    def _run(self, tool_name: str, arguments: dict[str, object]) -> object:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            raise ValueError(_FAILURE)
        with self._lock:
            if self._active:
                raise ValueError(_FAILURE)
            self._active = True
        try:
            try:
                return asyncio.run(
                    asyncio.wait_for(
                        self._call(tool_name, arguments),
                        timeout=self._timeout_seconds,
                    )
                )
            except Exception:
                raise ValueError(_FAILURE) from None
        finally:
            with self._lock:
                self._active = False

    async def _call(self, tool_name: str, arguments: dict[str, object]) -> object:
        with open(os.devnull, "w", encoding="utf-8") as error_log:
            async with stdio_client(
                _fixed_server_parameters(), errlog=error_log
            ) as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    initialized = await session.initialize()
                    if (
                        initialized.serverInfo.name != _SERVER_NAME
                        or initialized.capabilities.tools is None
                    ):
                        raise ValueError(_FAILURE)
                    inventory = await session.list_tools()
                    self._validate_inventory(inventory.tools)
                    result = await session.call_tool(tool_name, arguments)
                    if result.isError is not False:
                        raise ValueError(_FAILURE)
                    return self._decode_result(result)

    @staticmethod
    def _decode_result(result: object) -> object:
        structured = result.structuredContent
        if type(structured) is not dict or set(structured) != {"result"}:
            raise ValueError(_FAILURE)
        content = result.content
        if type(content) is not list or len(content) != 1:
            raise ValueError(_FAILURE)
        block = content[0]
        if type(block) is not TextContent or type(block.text) is not str:
            raise ValueError(_FAILURE)
        try:
            if len(block.text.encode("utf-8")) > _MAX_RESPONSE_BYTES:
                raise ValueError(_FAILURE)
            text_payload = json.loads(block.text)
        except (UnicodeError, TypeError, ValueError):
            raise ValueError(_FAILURE) from None
        if text_payload != structured["result"]:
            raise ValueError(_FAILURE)
        return text_payload

    @staticmethod
    def _page(payload: object) -> BrowserPage:
        if type(payload) is not dict or set(payload) != {
            "path",
            "title",
            "content",
            "links",
        }:
            raise ValueError(_FAILURE)
        links_payload = payload["links"]
        if type(links_payload) is not list or len(links_payload) > 32:
            raise ValueError(_FAILURE)
        try:
            links = []
            for item in links_payload:
                if type(item) is not dict or set(item) != {
                    "link_id",
                    "label",
                    "path",
                }:
                    raise ValueError(_FAILURE)
                links.append(BrowserLink(**item))
            return BrowserPage(
                path=payload["path"],
                title=payload["title"],
                content=payload["content"],
                links=tuple(links),
            )
        except Exception:
            raise ValueError(_FAILURE) from None

    def open_page(self, path: str) -> BrowserPage:
        try:
            browser_site_page(path)
        except Exception:
            raise ValueError(_FAILURE) from None
        page = self._page(self._run("open_page", {"path": path}))
        if page.path != path:
            raise ValueError(_FAILURE)
        return page

    def click_link(self, current_path: str, link_id: str) -> BrowserPage:
        try:
            target = browser_link_target(current_path, link_id)
        except Exception:
            raise ValueError(_FAILURE) from None
        page = self._page(
            self._run(
                "click_link",
                {"current_path": current_path, "link_id": link_id},
            )
        )
        if page.path != target:
            raise ValueError(_FAILURE)
        return page
