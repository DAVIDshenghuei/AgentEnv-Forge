import json

from mcp.server.fastmcp import FastMCP
from mcp.types import CallToolResult, TextContent

from .research import (
    RESEARCH_CORPUS_SHA256,
    RESEARCH_CORPUS_VERSION,
    RESEARCH_PAPERS,
    ResearchCorpus,
)


_CORPUS = ResearchCorpus(RESEARCH_PAPERS)
server = FastMCP(
    f"agentenv-forge-research-{RESEARCH_CORPUS_VERSION}-sha256-"
    f"{RESEARCH_CORPUS_SHA256}",
    log_level="ERROR",
)


def _tool_failure(message: str) -> CallToolResult:
    return CallToolResult(
        content=[TextContent(type="text", text=message)],
        isError=True,
    )


def _tool_success(payload: object) -> CallToolResult:
    return CallToolResult(
        content=[
            TextContent(
                type="text",
                text=json.dumps(payload, ensure_ascii=False, sort_keys=True),
            )
        ],
        structuredContent={"result": payload},
        isError=False,
    )


@server.tool(structured_output=False)
def search_papers(query: str, limit: int) -> CallToolResult:
    try:
        summaries = _CORPUS.search_papers(query=query, limit=limit)
    except ValueError as error:
        return _tool_failure(str(error))
    return _tool_success(
        [
            {
                "paper_id": summary.paper_id,
                "title": summary.title,
                "year": summary.year,
            }
            for summary in summaries
        ]
    )


@server.tool(structured_output=False)
def get_paper(paper_id: str) -> CallToolResult:
    try:
        paper = _CORPUS.get_paper(paper_id)
    except ValueError as error:
        return _tool_failure(str(error))
    return _tool_success(
        {
            "paper_id": paper.paper_id,
            "title": paper.title,
            "year": paper.year,
            "abstract": paper.abstract,
            "body": paper.body,
        }
    )


if __name__ == "__main__":
    server.run(transport="stdio")
