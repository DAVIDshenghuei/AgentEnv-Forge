import asyncio
import json
import math
import os
import sys
from threading import Lock

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import TextContent, Tool

from .research import (
    RESEARCH_CORPUS_SHA256,
    RESEARCH_CORPUS_VERSION,
    PaperRecord,
    PaperSummary,
)


_FAILURE = "research MCP call failed"
_TOOLS = ("get_paper", "search_papers")
_MAX_RESPONSE_BYTES = 1_100_000
_SERVER_NAME = (
    f"agentenv-forge-research-{RESEARCH_CORPUS_VERSION}-sha256-"
    f"{RESEARCH_CORPUS_SHA256}"
)
_INPUT_SCHEMAS = {
    "search_papers": {
        "properties": {
            "query": {"title": "Query", "type": "string"},
            "limit": {"title": "Limit", "type": "integer"},
        },
        "required": ["query", "limit"],
        "title": "search_papersArguments",
        "type": "object",
    },
    "get_paper": {
        "properties": {
            "paper_id": {"title": "Paper Id", "type": "string"},
        },
        "required": ["paper_id"],
        "title": "get_paperArguments",
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
        args=["-m", "agentenv_forge.mcp.server"],
    )


class StdioResearchClient:
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
    def _valid_text(value: str, maximum_bytes: int) -> bool:
        if not value or value.isspace():
            return False
        if any(ord(character) < 32 or ord(character) == 127 for character in value):
            return False
        try:
            return len(value.encode("utf-8")) <= maximum_bytes
        except UnicodeError:
            return False

    @classmethod
    def _validate_search_call(cls, query: str, limit: int) -> None:
        if type(query) is not str or type(limit) is not int:
            raise ValueError(_FAILURE)
        if not cls._valid_text(query, 256) or not 1 <= limit <= 32:
            raise ValueError(_FAILURE)

    @classmethod
    def _validate_paper_id(cls, paper_id: str) -> None:
        if type(paper_id) is not str or not cls._valid_text(paper_id, 64):
            raise ValueError(_FAILURE)
        if (
            not paper_id[0].isalnum()
            or not paper_id[-1].isalnum()
            or "--" in paper_id
            or any(
                not (
                    "a" <= character <= "z"
                    or "0" <= character <= "9"
                    or character == "-"
                )
                for character in paper_id
            )
        ):
            raise ValueError(_FAILURE)

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
                for key, value in tool.model_dump(
                    mode="json", by_alias=True
                ).items()
                if key not in {"name", "inputSchema"}
            }
            for tool in tools
        }
        if (
            names != _TOOLS
            or schemas != _INPUT_SCHEMAS
            or metadata != _TOOL_METADATA
        ):
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

    async def _call(
        self, tool_name: str, arguments: dict[str, object]
    ) -> object:
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
        structured_payload = None
        if structured is not None:
            if type(structured) is not dict or set(structured) != {"result"}:
                raise ValueError(_FAILURE)
            structured_payload = structured["result"]

        content = result.content
        if type(content) is not list or len(content) != 1:
            raise ValueError(_FAILURE)
        block = content[0]
        if type(block) is not TextContent or type(block.text) is not str:
            raise ValueError(_FAILURE)
        try:
            encoded_size = len(block.text.encode("utf-8"))
        except UnicodeError:
            raise ValueError(_FAILURE) from None
        if encoded_size > _MAX_RESPONSE_BYTES:
            raise ValueError(_FAILURE)
        try:
            text_payload = json.loads(block.text)
        except (TypeError, ValueError):
            raise ValueError(_FAILURE) from None
        if structured is not None and structured_payload != text_payload:
            raise ValueError(_FAILURE)
        return text_payload if structured is None else structured_payload

    def search_papers(
        self, query: str, limit: int
    ) -> tuple[PaperSummary, ...]:
        self._validate_search_call(query, limit)
        payload = self._run(
            "search_papers", {"query": query, "limit": limit}
        )
        if type(payload) is not list or len(payload) > limit:
            raise ValueError(_FAILURE)
        summaries: list[PaperSummary] = []
        try:
            for item in payload:
                if type(item) is not dict or set(item) != {
                    "paper_id",
                    "title",
                    "year",
                }:
                    raise ValueError(_FAILURE)
                summaries.append(
                    PaperSummary(
                        paper_id=item["paper_id"],
                        title=item["title"],
                        year=item["year"],
                    )
                )
        except (TypeError, ValueError):
            raise ValueError(_FAILURE) from None
        result = tuple(summaries)
        paper_ids = tuple(summary.paper_id for summary in result)
        if paper_ids != tuple(sorted(paper_ids)) or len(set(paper_ids)) != len(
            paper_ids
        ):
            raise ValueError(_FAILURE)
        return result

    def get_paper(self, paper_id: str) -> PaperRecord:
        self._validate_paper_id(paper_id)
        payload = self._run("get_paper", {"paper_id": paper_id})
        if type(payload) is not dict or set(payload) != {
            "paper_id",
            "title",
            "year",
            "abstract",
            "body",
        }:
            raise ValueError(_FAILURE)
        try:
            result = PaperRecord(
                paper_id=payload["paper_id"],
                title=payload["title"],
                year=payload["year"],
                abstract=payload["abstract"],
                body=payload["body"],
            )
        except (TypeError, ValueError):
            raise ValueError(_FAILURE) from None
        if result.paper_id != paper_id:
            raise ValueError(_FAILURE)
        return result
