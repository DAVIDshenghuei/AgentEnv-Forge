import hashlib
import json

import pytest

import agentenv_forge.mcp.browser_site as site_module
from agentenv_forge.mcp.browser_site import BrowserSitePage


ORIGIN = "https://forge.invalid"
VERSION = "1.0.0"
DIGEST = "b80be2fdc2d6c318a40ffa297201132b928c0037cbefd5f225f01f335f553b05"
INDEX_HTML = (
    '<!doctype html><html><head><title>Offline Evaluation Library</title></head>'
    '<body><main>Offline Evaluation Library<br>Browse deterministic offline '
    'evaluation guides.<br><a data-link-id="browser-isolation" '
    'href="/guides/browser-isolation">Browser isolation</a>'
    '<img src="https://external.invalid/tracker.png" alt=""></main></body></html>'
)
DETAIL_HTML = (
    '<!doctype html><html><head><title>Offline Browser Evaluation</title></head>'
    '<body><main>Published: 2024<br>Browser requests are fulfilled only from '
    'the bundled origin; every external request is aborted.</main></body></html>'
)


def _canonical(version, origin, pages) -> bytes:
    return json.dumps(
        {
            "version": version,
            "origin": origin,
            "pages": [
                {"path": page.path, "html": page.html} for page in pages
            ],
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def test_browser_site_manifest_binds_exact_ordered_html_bytes() -> None:
    assert site_module.BROWSER_SITE_VERSION == VERSION
    assert site_module.BROWSER_SITE_ORIGIN == ORIGIN
    assert site_module.BROWSER_SITE_SHA256 == DIGEST
    assert type(site_module.BROWSER_SITE_PAGES) is tuple
    assert site_module.BROWSER_SITE_PAGES == (
        BrowserSitePage(path="/", html=INDEX_HTML),
        BrowserSitePage(path="/guides/browser-isolation", html=DETAIL_HTML),
    )
    assert hashlib.sha256(
        _canonical(VERSION, ORIGIN, site_module.BROWSER_SITE_PAGES)
    ).hexdigest() == DIGEST

    changed = (
        BrowserSitePage(path="/", html=INDEX_HTML + " "),
        site_module.BROWSER_SITE_PAGES[1],
    )
    assert hashlib.sha256(_canonical(VERSION, ORIGIN, changed)).hexdigest() != DIGEST
    assert hashlib.sha256(
        _canonical("1.0.1", ORIGIN, site_module.BROWSER_SITE_PAGES)
    ).hexdigest() != DIGEST


def test_browser_site_records_are_strict_immutable_and_canonical() -> None:
    page = site_module.BROWSER_SITE_PAGES[0]
    with pytest.raises((AttributeError, TypeError)):
        page.path = "/changed"
    for invalid_path in (
        "//evil.invalid",
        "/guides/../secret",
        "/guides/./secret",
        "/guides/browser-isolation/",
        "/guides?x=1",
        "/guides#fragment",
        "/guides\\escape",
        "/guidés",
    ):
        with pytest.raises(ValueError, match="^invalid browser site page$"):
            BrowserSitePage(path=invalid_path, html="<main>safe</main>")
    with pytest.raises(ValueError, match="^invalid browser site page$"):
        BrowserSitePage(path="/valid", html="bad\ud800html")
