import asyncio
import json
import os
import sys
import time

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

import agentenv_forge.mcp.client as client_module
from agentenv_forge.mcp.client import StdioResearchClient
from agentenv_forge.mcp.research import PaperRecord, PaperSummary


def _result_payload(result):
    if result.structuredContent is not None:
        structured = result.structuredContent
        return structured["result"] if set(structured) == {"result"} else structured
    text_blocks = [block.text for block in result.content if hasattr(block, "text")]
    assert len(text_blocks) == 1
    return json.loads(text_blocks[0])


def test_research_mcp_real_stdio_roundtrip() -> None:
    async def roundtrip() -> None:
        parameters = StdioServerParameters(
            command=sys.executable,
            args=["-m", "agentenv_forge.mcp.server"],
        )
        async with stdio_client(parameters) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                initialized = await session.initialize()
                digest = (
                    "6870b6d0751a8a6743d508cad7e97888f842d47f6e8a13ee8049c9c236b0579e"
                )
                assert initialized.serverInfo.name == (
                    "agentenv-forge-research-1.0.0-sha256-" + digest
                )
                listed = await session.list_tools()
                assert tuple(sorted(tool.name for tool in listed.tools)) == (
                    "get_paper",
                    "search_papers",
                )
                assert {tool.name: tool.inputSchema for tool in listed.tools} == {
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
                            "paper_id": {
                                "title": "Paper Id",
                                "type": "string",
                            }
                        },
                        "required": ["paper_id"],
                        "title": "get_paperArguments",
                        "type": "object",
                    },
                }
                assert {
                    tool.name: {
                        key: value
                        for key, value in tool.model_dump(
                            mode="json", by_alias=True
                        ).items()
                        if key not in {"name", "inputSchema"}
                    }
                    for tool in listed.tools
                } == {
                    name: {
                        "title": None,
                        "description": "",
                        "outputSchema": None,
                        "icons": None,
                        "annotations": None,
                        "execution": None,
                        "_meta": None,
                    }
                    for name in ("search_papers", "get_paper")
                }

                StdioResearchClient._validate_inventory(listed.tools)
                impostor_tools = list(listed.tools)
                impostor_tools[0] = impostor_tools[0].model_copy(
                    update={"description": "impostor"}
                )
                with pytest.raises(ValueError, match="^research MCP call failed$"):
                    StdioResearchClient._validate_inventory(impostor_tools)

                search = await session.call_tool(
                    "search_papers",
                    {"query": "agent environment", "limit": 2},
                )
                assert search.isError is False
                assert _result_payload(search) == [
                    {
                        "paper_id": "paper-001",
                        "title": "Deterministic Agent Environments",
                        "year": 2024,
                    },
                    {
                        "paper_id": "paper-002",
                        "title": "Offline Tool Evaluation",
                        "year": 2023,
                    },
                ]

                paper = await session.call_tool(
                    "get_paper", {"paper_id": "paper-002"}
                )
                assert paper.isError is False
                assert _result_payload(paper) == {
                    "paper_id": "paper-002",
                    "title": "Offline Tool Evaluation",
                    "year": 2023,
                    "abstract": (
                        "An AGENT ENVIRONMENT corpus for offline research."
                    ),
                    "body": "second body",
                }

                secret_id = "secret-paper-id"
                unknown = await session.call_tool(
                    "get_paper", {"paper_id": secret_id}
                )
                assert unknown.isError is True
                error_text = " ".join(
                    block.text for block in unknown.content if hasattr(block, "text")
                )
                assert error_text == "unknown paper id"
                assert secret_id not in error_text

    asyncio.run(roundtrip())


def test_sync_research_client_returns_domain_types_and_closes_each_session() -> None:
    with pytest.raises(TypeError):
        StdioResearchClient(
            StdioServerParameters(
                command=sys.executable,
                args=["-m", "agentenv_forge.mcp.server"],
            )
        )
    client = StdioResearchClient()
    assert client.has_active_session is False

    assert client.search_papers("agent environment", 2) == (
        PaperSummary(
            "paper-001", "Deterministic Agent Environments", 2024
        ),
        PaperSummary("paper-002", "Offline Tool Evaluation", 2023),
    )
    assert client.has_active_session is False

    assert client.get_paper("paper-002") == PaperRecord(
        paper_id="paper-002",
        title="Offline Tool Evaluation",
        year=2023,
        abstract="An AGENT ENVIRONMENT corpus for offline research.",
        body="second body",
    )
    assert client.has_active_session is False

    secret_id = "secret-wrapper-paper-id"
    with pytest.raises(ValueError, match="^research MCP call failed$") as failure:
        client.get_paper(secret_id)
    assert secret_id not in str(failure.value)
    assert client.has_active_session is False

    async def reject_running_loop() -> None:
        with pytest.raises(ValueError, match="^research MCP call failed$"):
            client.search_papers("agent environment", 1)

    asyncio.run(reject_running_loop())
    assert client.has_active_session is False


def test_sync_research_client_times_out_and_clears_active_state(monkeypatch) -> None:
    async def hang_forever(self, tool_name, arguments):
        await asyncio.Event().wait()

    with pytest.raises(ValueError, match="^research MCP call failed$"):
        StdioResearchClient(timeout_seconds=True)
    client = StdioResearchClient(timeout_seconds=0.05)
    monkeypatch.setattr(StdioResearchClient, "_call", hang_forever)

    started = time.monotonic()
    with pytest.raises(ValueError, match="^research MCP call failed$"):
        client.search_papers("agent", 1)

    assert time.monotonic() - started < 1.0
    assert client.has_active_session is False


def test_sync_research_client_timeout_reaps_real_child_process(
    tmp_path, monkeypatch
) -> None:
    pid_file = tmp_path / "mcp-child.pid"
    script = (
        "import os,time,pathlib;"
        f"pathlib.Path({str(pid_file)!r}).write_text(str(os.getpid()));"
        "time.sleep(30)"
    )
    client = StdioResearchClient(timeout_seconds=0.5)
    monkeypatch.setattr(
        client_module,
        "_fixed_server_parameters",
        lambda: StdioServerParameters(command=sys.executable, args=["-c", script]),
    )

    started = time.monotonic()
    with pytest.raises(ValueError, match="^research MCP call failed$"):
        client.search_papers("agent", 1)

    assert time.monotonic() - started < 4.0
    assert pid_file.is_file()
    pid = int(pid_file.read_text())
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except OSError:
            break
        time.sleep(0.01)
    else:
        pytest.fail("timed-out MCP child process was not reaped")
    assert client.has_active_session is False
