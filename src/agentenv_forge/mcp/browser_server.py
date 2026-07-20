import asyncio
import json
import sys
from urllib.parse import urlsplit

from mcp.server.fastmcp import FastMCP
from mcp.types import CallToolResult, TextContent
from playwright.async_api import async_playwright

from ..tools.browser import BrowserLink, BrowserPage
from .browser_site import (
    BROWSER_SITE_ORIGIN,
    BROWSER_SITE_PAGES,
    BROWSER_SITE_SHA256,
    BROWSER_SITE_VERSION,
    browser_link_target,
    browser_site_page,
)


_FAILURE = "browser MCP call failed"
_OPERATION_TIMEOUT_SECONDS = 10.0
_PAGE_URLS = {
    BROWSER_SITE_ORIGIN + page.path: page for page in BROWSER_SITE_PAGES
}

server = FastMCP(
    f"agentenv-forge-browser-{BROWSER_SITE_VERSION}-sha256-"
    f"{BROWSER_SITE_SHA256}",
    log_level="ERROR",
)


def _tool_failure() -> CallToolResult:
    return CallToolResult(
        content=[TextContent(type="text", text=_FAILURE)],
        isError=True,
    )


def _tool_success(page: BrowserPage) -> CallToolResult:
    payload = {
        "path": page.path,
        "title": page.title,
        "content": page.content,
        "links": [
            {"link_id": link.link_id, "label": link.label, "path": link.path}
            for link in page.links
        ],
    }
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


async def _render(path: str, link_id: str | None = None) -> BrowserPage:
    playwright = None
    browser = None
    context = None
    page = None
    try:
        playwright = await async_playwright().start()
        browser = await playwright.chromium.launch(headless=True)
        context = await browser.new_context(service_workers="block")

        async def route_request(route, request) -> None:
            bundled = _PAGE_URLS.get(request.url)
            if bundled is None:
                await route.abort()
            else:
                await route.fulfill(
                    status=200,
                    content_type="text/html; charset=utf-8",
                    body=bundled.html,
                )

        await context.route("**/*", route_request)
        page = await context.new_page()
        await page.goto(BROWSER_SITE_ORIGIN + path, wait_until="load")
        if link_id is not None:
            locator = page.locator(f'main a[data-link-id="{link_id}"]')
            if await locator.count() != 1:
                raise ValueError(_FAILURE)
            await locator.click()
            await page.wait_for_load_state("load")

        parsed = urlsplit(page.url)
        if (
            f"{parsed.scheme}://{parsed.netloc}" != BROWSER_SITE_ORIGIN
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError(_FAILURE)
        final_path = parsed.path
        browser_site_page(final_path)
        main = page.locator("main")
        if await main.count() != 1:
            raise ValueError(_FAILURE)
        anchors = main.locator("a[data-link-id]")
        anchor_count = await anchors.count()
        if anchor_count > 32:
            raise ValueError(_FAILURE)
        links = []
        for index in range(anchor_count):
            anchor = anchors.nth(index)
            link_id_value = await anchor.get_attribute("data-link-id")
            label = await anchor.inner_text()
            target = await anchor.get_attribute("href")
            links.append(
                BrowserLink(link_id=link_id_value, label=label, path=target)
            )
        return BrowserPage(
            path=final_path,
            title=await page.title(),
            content=await main.inner_text(),
            links=tuple(links),
        )
    finally:
        primary_failure = sys.exception()
        first_cleanup_failure = None
        first_control_flow_failure = None
        cleanup_calls = (
            (page, "close"),
            (context, "close"),
            (browser, "close"),
            (playwright, "stop"),
        )
        for resource, method_name in cleanup_calls:
            if resource is None:
                continue
            try:
                await getattr(resource, method_name)()
            except BaseException as error:
                if not isinstance(error, Exception):
                    if first_control_flow_failure is None:
                        first_control_flow_failure = error
                elif first_cleanup_failure is None:
                    first_cleanup_failure = error
        if first_control_flow_failure is not None:
            raise first_control_flow_failure
        if primary_failure is None and first_cleanup_failure is not None:
            raise first_cleanup_failure


@server.tool(structured_output=False)
async def open_page(path: str) -> CallToolResult:
    try:
        browser_site_page(path)
        page = await asyncio.wait_for(
            _render(path), timeout=_OPERATION_TIMEOUT_SECONDS
        )
    except Exception:
        return _tool_failure()
    return _tool_success(page)


@server.tool(structured_output=False)
async def click_link(current_path: str, link_id: str) -> CallToolResult:
    try:
        expected_path = browser_link_target(current_path, link_id)
        page = await asyncio.wait_for(
            _render(current_path, link_id), timeout=_OPERATION_TIMEOUT_SECONDS
        )
        if page.path != expected_path:
            raise ValueError(_FAILURE)
    except Exception:
        return _tool_failure()
    return _tool_success(page)


if __name__ == "__main__":
    server.run(transport="stdio")
