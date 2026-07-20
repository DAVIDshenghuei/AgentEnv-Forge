import hashlib
import json
from dataclasses import dataclass


BROWSER_SITE_VERSION = "1.0.0"
BROWSER_SITE_ORIGIN = "https://forge.invalid"
BROWSER_SITE_SHA256 = (
    "b80be2fdc2d6c318a40ffa297201132b928c0037cbefd5f225f01f335f553b05"
)

_MAX_PATH_BYTES = 256
_MAX_HTML_BYTES = 65_536


def _valid_path(path: str) -> bool:
    if not path or len(path.encode("utf-8")) > _MAX_PATH_BYTES:
        return False
    if path == "/":
        return True
    if not path.startswith("/") or path.endswith("/"):
        return False
    segments = path[1:].split("/")
    return all(
        segment
        and all(
            "a" <= character <= "z"
            or "0" <= character <= "9"
            or character in {"-", "_"}
            for character in segment
        )
        for segment in segments
    )


@dataclass(frozen=True, slots=True)
class BrowserSitePage:
    path: str
    html: str

    def __post_init__(self) -> None:
        if type(self.path) is not str or type(self.html) is not str:
            raise ValueError("invalid browser site page")
        try:
            html_size = len(self.html.encode("utf-8"))
        except UnicodeError:
            raise ValueError("invalid browser site page") from None
        if not _valid_path(self.path) or not self.html or html_size > _MAX_HTML_BYTES:
            raise ValueError("invalid browser site page")


_INDEX_HTML = (
    '<!doctype html><html><head><title>Offline Evaluation Library</title></head>'
    '<body><main>Offline Evaluation Library<br>Browse deterministic offline '
    'evaluation guides.<br><a data-link-id="browser-isolation" '
    'href="/guides/browser-isolation">Browser isolation</a>'
    '<img src="https://external.invalid/tracker.png" alt=""></main></body></html>'
)
_DETAIL_HTML = (
    '<!doctype html><html><head><title>Offline Browser Evaluation</title></head>'
    '<body><main>Published: 2024<br>Browser requests are fulfilled only from '
    'the bundled origin; every external request is aborted.</main></body></html>'
)

BROWSER_SITE_PAGES = (
    BrowserSitePage(path="/", html=_INDEX_HTML),
    BrowserSitePage(path="/guides/browser-isolation", html=_DETAIL_HTML),
)


def _canonical_manifest() -> bytes:
    return json.dumps(
        {
            "version": BROWSER_SITE_VERSION,
            "origin": BROWSER_SITE_ORIGIN,
            "pages": [
                {"path": page.path, "html": page.html}
                for page in BROWSER_SITE_PAGES
            ],
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


if hashlib.sha256(_canonical_manifest()).hexdigest() != BROWSER_SITE_SHA256:
    raise RuntimeError("invalid browser site manifest")


_PAGES_BY_PATH = {page.path: page for page in BROWSER_SITE_PAGES}
_LINK_TARGETS = {("/", "browser-isolation"): "/guides/browser-isolation"}


def browser_site_page(path: str) -> BrowserSitePage:
    if type(path) is not str or not _valid_path(path):
        raise ValueError("browser MCP call failed")
    page = _PAGES_BY_PATH.get(path)
    if page is None:
        raise ValueError("browser MCP call failed")
    return page


def browser_link_target(current_path: str, link_id: str) -> str:
    if type(current_path) is not str or type(link_id) is not str:
        raise ValueError("browser MCP call failed")
    browser_site_page(current_path)
    try:
        link_id_size = len(link_id.encode("utf-8"))
    except UnicodeError:
        raise ValueError("browser MCP call failed") from None
    if (
        not link_id
        or link_id_size > 64
        or any(
            not ("a" <= character <= "z" or "0" <= character <= "9")
            and character not in "-_"
            for character in link_id
        )
    ):
        raise ValueError("browser MCP call failed")
    target = _LINK_TARGETS.get((current_path, link_id))
    if target is None:
        raise ValueError("browser MCP call failed")
    return target
