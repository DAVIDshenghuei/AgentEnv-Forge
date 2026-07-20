from threading import Condition
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, model_validator

from ..schemas import PublicTask
from .budget import ActionBudget, ActionBudgetExhaustedError


_MAX_PATH_BYTES = 256
_MAX_LINK_ID_BYTES = 64
_MAX_LABEL_BYTES = 256
_MAX_TITLE_BYTES = 256
_MAX_CONTENT_BYTES = 65_536
_MAX_LINKS = 32


def _utf8_size(value: str) -> int | None:
    try:
        return len(value.encode("utf-8"))
    except UnicodeError:
        return None


def _valid_text(
    value: str,
    maximum_bytes: int,
    *,
    empty: bool = False,
    allow_lf: bool = False,
) -> bool:
    if not empty and (not value or value.isspace()):
        return False
    if any(
        (ord(character) < 32 and not (allow_lf and character == "\n"))
        or ord(character) == 127
        for character in value
    ):
        return False
    size = _utf8_size(value)
    return size is not None and size <= maximum_bytes


def _valid_path(value: str) -> bool:
    if not _valid_text(value, _MAX_PATH_BYTES):
        return False
    if value == "/":
        return True
    if not value.startswith("/") or value.endswith("/"):
        return False
    segments = value[1:].split("/")
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


def _valid_link_id(value: str) -> bool:
    return _valid_text(value, _MAX_LINK_ID_BYTES) and all(
        "a" <= character <= "z"
        or "0" <= character <= "9"
        or character in {"-", "_"}
        for character in value
    )


class BrowserLink(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    link_id: str
    label: str
    path: str

    @model_validator(mode="before")
    @classmethod
    def exact_bounded_fields(cls, data):
        if type(data) is not dict or set(data) != {"link_id", "label", "path"}:
            raise ValueError("invalid browser link")
        if any(type(data[field]) is not str for field in data):
            raise ValueError("invalid browser link")
        if (
            not _valid_link_id(data["link_id"])
            or not _valid_text(data["label"], _MAX_LABEL_BYTES)
            or not _valid_path(data["path"])
        ):
            raise ValueError("invalid browser link")
        return data


class BrowserPage(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    path: str
    title: str
    content: str
    links: tuple[BrowserLink, ...]

    @model_validator(mode="before")
    @classmethod
    def exact_bounded_fields(cls, data):
        if type(data) is not dict or set(data) != {
            "path",
            "title",
            "content",
            "links",
        }:
            raise ValueError("invalid browser page")
        if (
            type(data["path"]) is not str
            or type(data["title"]) is not str
            or type(data["content"]) is not str
            or type(data["links"]) is not tuple
        ):
            raise ValueError("invalid browser page")
        links = data["links"]
        if not 0 <= len(links) <= _MAX_LINKS:
            raise ValueError("invalid browser page")
        if any(type(link) is not BrowserLink for link in links):
            raise ValueError("invalid browser page")
        if (
            not _valid_path(data["path"])
            or not _valid_text(data["title"], _MAX_TITLE_BYTES)
            or not _valid_text(
                data["content"],
                _MAX_CONTENT_BYTES,
                empty=True,
                allow_lf=True,
            )
        ):
            raise ValueError("invalid browser page")
        link_ids = tuple(link.link_id for link in links)
        if len(link_ids) != len(set(link_ids)):
            raise ValueError("invalid browser page")
        return data


class BrowserActionLimitError(ValueError):
    """The shared episode action budget has been exhausted by browser use."""


@runtime_checkable
class BrowserProtocol(Protocol):
    def open_page(self, path: str) -> BrowserPage: ...

    def click_link(self, current_path: str, link_id: str) -> BrowserPage: ...


class BrowserTools:
    __slots__ = (
        "_budget",
        "_client",
        "_condition",
        "_current_page",
        "_in_flight",
        "_revoked",
    )

    def __init__(
        self,
        task: PublicTask,
        budget: ActionBudget,
        client: BrowserProtocol,
    ) -> None:
        if type(task) is not PublicTask or type(task.max_actions) is not int:
            raise ValueError("invalid browser action budget")
        if type(budget) is not ActionBudget or budget.limit != task.max_actions:
            raise ValueError("invalid browser action budget")
        if not isinstance(client, BrowserProtocol):
            raise ValueError("invalid browser client")
        self._budget = budget
        self._client = client
        self._condition = Condition()
        self._current_page: BrowserPage | None = None
        self._in_flight = 0
        self._revoked = False

    def revoke(self) -> None:
        with self._condition:
            self._revoked = True
            while self._in_flight:
                self._condition.wait()

    def _begin_action(self) -> None:
        with self._condition:
            if self._revoked:
                raise ValueError("browser tools revoked")
            try:
                self._budget.charge()
            except ActionBudgetExhaustedError:
                raise BrowserActionLimitError(
                    "browser action budget exhausted"
                ) from None
            self._in_flight += 1

    def _end_action(self) -> None:
        with self._condition:
            self._in_flight -= 1
            if self._in_flight == 0:
                self._condition.notify_all()

    @staticmethod
    def _validate_path(path: str) -> None:
        if type(path) is not str or not _valid_path(path):
            raise ValueError("invalid browser tool call")

    @staticmethod
    def _validate_link_id(link_id: str) -> None:
        if type(link_id) is not str or not _valid_link_id(link_id):
            raise ValueError("invalid browser tool call")

    @staticmethod
    def _reconstruct_link(link: object) -> BrowserLink:
        if type(link) is not BrowserLink:
            raise ValueError("browser client failed")
        try:
            return BrowserLink(
                link_id=link.link_id,
                label=link.label,
                path=link.path,
            )
        except Exception:
            raise ValueError("browser client failed") from None

    @classmethod
    def _reconstruct_page(cls, page: object) -> BrowserPage:
        if type(page) is not BrowserPage or type(page.links) is not tuple:
            raise ValueError("browser client failed")
        try:
            links = tuple(cls._reconstruct_link(link) for link in page.links)
            return BrowserPage(
                path=page.path,
                title=page.title,
                content=page.content,
                links=links,
            )
        except Exception:
            raise ValueError("browser client failed") from None

    def open_page(self, path: str) -> BrowserPage:
        self._begin_action()
        try:
            self._validate_path(path)
            try:
                raw_page = self._client.open_page(path)
            except Exception:
                raise ValueError("browser client failed") from None
            page = self._reconstruct_page(raw_page)
            if page.path != path:
                raise ValueError("browser client failed")
            self._current_page = page
            return page
        finally:
            self._end_action()

    def click_link(self, link_id: str) -> BrowserPage:
        self._begin_action()
        try:
            self._validate_link_id(link_id)
            current_page = self._current_page
            if current_page is None:
                raise ValueError("browser page is not open")
            matching = tuple(
                link for link in current_page.links if link.link_id == link_id
            )
            if len(matching) != 1:
                raise ValueError("invalid browser tool call")
            try:
                raw_page = self._client.click_link(current_page.path, link_id)
            except Exception:
                raise ValueError("browser client failed") from None
            page = self._reconstruct_page(raw_page)
            if page.path != matching[0].path:
                raise ValueError("browser client failed")
            self._current_page = page
            return page
        finally:
            self._end_action()
