from threading import Event, Thread

import pytest
from pydantic import ValidationError

from agentenv_forge.schemas import PublicTask
from agentenv_forge.tools import ActionBudget, WorkspaceTools
from agentenv_forge.tools.browser import (
    BrowserActionLimitError,
    BrowserLink,
    BrowserPage,
    BrowserProtocol,
    BrowserTools,
)


INDEX_LINK = BrowserLink(link_id="details", label="Details", path="/details")
INDEX_PAGE = BrowserPage(
    path="/index",
    title="Index",
    content="Choose a page.",
    links=(INDEX_LINK,),
)
DETAIL_PAGE = BrowserPage(
    path="/details",
    title="Details",
    content="Offline deterministic detail.",
    links=(),
)


def _task(max_actions: int = 3) -> PublicTask:
    return PublicTask.model_validate(
        {
            "task_id": "browser-tools",
            "version": "1",
            "instruction": "Browse the offline pages before writing output.",
            "input_artifacts": ("source.txt",),
            "allowed_artifacts": ("result.txt",),
            "max_actions": max_actions,
        }
    )


class RecordingBrowserClient(BrowserProtocol):
    def __init__(self, open_result=INDEX_PAGE, click_result=DETAIL_PAGE) -> None:
        self.open_result = open_result
        self.click_result = click_result
        self.calls: list[tuple[object, ...]] = []

    def open_page(self, path: str) -> BrowserPage:
        self.calls.append(("open_page", path))
        return self.open_result

    def click_link(self, current_path: str, link_id: str) -> BrowserPage:
        self.calls.append(("click_link", current_path, link_id))
        return self.click_result


def test_open_then_click_uses_private_state_and_returns_immutable_clones() -> None:
    client = RecordingBrowserClient()
    budget = ActionBudget(3)
    browser = BrowserTools(task=_task(), budget=budget, client=client)

    with pytest.raises(ValueError, match="^browser page is not open$"):
        browser.click_link("details")
    opened = browser.open_page("/index")
    clicked = browser.click_link("details")

    assert opened == INDEX_PAGE and opened is not INDEX_PAGE
    assert opened.links == (INDEX_LINK,)
    assert opened.links[0] is not INDEX_LINK
    assert clicked == DETAIL_PAGE and clicked is not DETAIL_PAGE
    assert client.calls == [
        ("open_page", "/index"),
        ("click_link", "/index", "details"),
    ]
    assert budget.used == 3
    with pytest.raises((AttributeError, TypeError, ValidationError)):
        opened.title = "changed"


def test_browser_shares_budget_and_charges_before_validation_or_client_use(
    tmp_path,
) -> None:
    task = _task(max_actions=3)
    (tmp_path / "source.txt").write_text("source", encoding="utf-8")
    budget = ActionBudget(task.max_actions)
    client = RecordingBrowserClient()
    browser = BrowserTools(task=task, budget=budget, client=client)
    workspace = WorkspaceTools(workspace=tmp_path, task=task, budget=budget)

    calls: list[str] = []

    class HostileStr(str):
        def encode(self, *args, **kwargs):
            calls.append("encode")
            raise AssertionError("must not inspect hostile path")

    with pytest.raises(ValueError, match="^invalid browser tool call$"):
        browser.open_page(HostileStr("/index"))
    assert workspace.list_files() == ("source.txt",)
    assert browser.open_page("/index") == INDEX_PAGE
    with pytest.raises(
        BrowserActionLimitError, match="^browser action budget exhausted$"
    ):
        browser.click_link("details")

    assert calls == []
    assert client.calls == [("open_page", "/index")]
    assert budget.used == 3


def test_domain_bounds_and_untrusted_pages_fail_closed() -> None:
    multiline = BrowserPage(
        path="/guides/browser-isolation",
        title="Offline Browser Evaluation",
        content="Published: 2024\nDeterministic browsers block remote network.",
        links=(),
    )
    assert "\n" in multiline.content

    for invalid_path in (
        "//evil.invalid/path",
        "/../details",
        "/details?leak=1",
        "/details#fragment",
        "/details\\child",
        "/details/",
        "/détails",
    ):
        with pytest.raises(ValueError):
            BrowserLink(link_id="details", label="Details", path=invalid_path)

    with pytest.raises(ValueError):
        BrowserLink(link_id="details", label="x" * 257, path="/details")
    with pytest.raises(ValueError):
        BrowserPage(
            path="/index",
            title="Index",
            content="bad\ud800text",
            links=(),
        )
    with pytest.raises(ValueError):
        BrowserTools(task=_task(), budget=ActionBudget(3), client=object())

    forged_link = BrowserLink.model_construct(
        link_id="details",
        label="Details",
        path="/details",
        hidden="must not cross",
    )
    forged_page = BrowserPage.model_construct(
        path="/index",
        title="Index",
        content="content",
        links=(forged_link,),
        hidden="must not cross",
    )
    reconstructing_client = RecordingBrowserClient(open_result=forged_page)
    reconstructing = BrowserTools(
        task=_task(max_actions=2),
        budget=ActionBudget(2),
        client=reconstructing_client,
    )
    reconstructed = reconstructing.open_page("/index")
    assert not hasattr(reconstructed, "hidden")
    assert not hasattr(reconstructed.links[0], "hidden")

    oversized_page = BrowserPage.model_construct(
        path="/index", title="x" * 257, content="content", links=()
    )
    reconstructing_client.open_result = oversized_page
    with pytest.raises(ValueError, match="^browser client failed$"):
        reconstructing.open_page("/index")

    class PageSubclass(BrowserPage):
        pass

    subclassed = PageSubclass(
        path="/index", title="Index", content="content", links=()
    )
    subclass_client = BrowserTools(
        task=_task(max_actions=1),
        budget=ActionBudget(1),
        client=RecordingBrowserClient(open_result=subclassed),
    )
    with pytest.raises(ValueError, match="^browser client failed$"):
        subclass_client.open_page("/index")


def test_revoke_drains_in_flight_open_and_rejects_without_recharging() -> None:
    entered = Event()
    release = Event()
    revoke_finished = Event()
    outcomes: list[object] = []

    class BlockingClient(RecordingBrowserClient):
        def open_page(self, path: str) -> BrowserPage:
            self.calls.append(("open_page", path))
            entered.set()
            assert release.wait(timeout=5)
            return INDEX_PAGE

    budget = ActionBudget(2)
    client = BlockingClient()
    browser = BrowserTools(task=_task(max_actions=2), budget=budget, client=client)
    call_thread = Thread(
        target=lambda: outcomes.append(browser.open_page("/index"))
    )
    call_thread.start()
    assert entered.wait(timeout=5)
    revoke_thread = Thread(
        target=lambda: (browser.revoke(), revoke_finished.set())
    )
    revoke_thread.start()
    assert not revoke_finished.wait(timeout=0.1)

    release.set()
    call_thread.join(timeout=5)
    revoke_thread.join(timeout=5)
    assert not call_thread.is_alive()
    assert not revoke_thread.is_alive()
    assert outcomes == [INDEX_PAGE]
    assert revoke_finished.is_set()

    with pytest.raises(ValueError, match="^browser tools revoked$"):
        browser.open_page("/index")
    assert budget.used == 1
    assert client.calls == [("open_page", "/index")]
