import hashlib
import json
from dataclasses import dataclass


RESEARCH_CORPUS_VERSION = "1.0.0"
RESEARCH_CORPUS_SHA256 = (
    "6870b6d0751a8a6743d508cad7e97888f842d47f6e8a13ee8049c9c236b0579e"
)

_MAX_PAPER_ID_BYTES = 64
_MAX_TITLE_BYTES = 256
_MAX_ABSTRACT_BYTES = 4096
_MAX_BODY_BYTES = 1_048_576
_MAX_QUERY_BYTES = 256
_MAX_CORPUS_SIZE = 256


def _valid_text(value: str, maximum_bytes: int) -> bool:
    if not value or value.isspace():
        return False
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        return False
    try:
        size = len(value.encode("utf-8"))
    except UnicodeError:
        return False
    return size <= maximum_bytes


def _valid_paper_id(value: str) -> bool:
    if not _valid_text(value, _MAX_PAPER_ID_BYTES):
        return False
    if not value[0].isalnum() or not value[-1].isalnum() or "--" in value:
        return False
    return all(
        "a" <= character <= "z" or "0" <= character <= "9" or character == "-"
        for character in value
    )


@dataclass(frozen=True, slots=True)
class PaperRecord:
    paper_id: str
    title: str
    year: int
    abstract: str
    body: str

    def __post_init__(self) -> None:
        if (
            type(self.paper_id) is not str
            or type(self.title) is not str
            or type(self.year) is not int
            or type(self.abstract) is not str
            or type(self.body) is not str
        ):
            raise ValueError("invalid paper record")
        if (
            not _valid_paper_id(self.paper_id)
            or not _valid_text(self.title, _MAX_TITLE_BYTES)
            or not 1 <= self.year <= 9999
            or not _valid_text(self.abstract, _MAX_ABSTRACT_BYTES)
            or not _valid_text(self.body, _MAX_BODY_BYTES)
        ):
            raise ValueError("invalid paper record")


@dataclass(frozen=True, slots=True)
class PaperSummary:
    paper_id: str
    title: str
    year: int

    def __post_init__(self) -> None:
        if (
            type(self.paper_id) is not str
            or type(self.title) is not str
            or type(self.year) is not int
        ):
            raise ValueError("invalid paper summary")
        if (
            not _valid_paper_id(self.paper_id)
            or not _valid_text(self.title, _MAX_TITLE_BYTES)
            or not 1 <= self.year <= 9999
        ):
            raise ValueError("invalid paper summary")


RESEARCH_PAPERS = (
    PaperRecord(
        paper_id="paper-003",
        title="Evaluating Agent Environment Safety",
        year=2025,
        abstract="A benchmark for isolated agents.",
        body="third body",
    ),
    PaperRecord(
        paper_id="paper-001",
        title="Deterministic Agent Environments",
        year=2024,
        abstract="Reproducible evaluation methods.",
        body="first body",
    ),
    PaperRecord(
        paper_id="paper-002",
        title="Offline Tool Evaluation",
        year=2023,
        abstract="An AGENT ENVIRONMENT corpus for offline research.",
        body="second body",
    ),
    PaperRecord(
        paper_id="paper-004",
        title="Unrelated Systems",
        year=2022,
        abstract="A control paper.",
        body="fourth body",
    ),
)


def _canonical_manifest() -> bytes:
    payload = {
        "version": RESEARCH_CORPUS_VERSION,
        "papers": [
            {
                "paper_id": paper.paper_id,
                "title": paper.title,
                "year": paper.year,
                "abstract": paper.abstract,
                "body": paper.body,
            }
            for paper in RESEARCH_PAPERS
        ],
    }
    return json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


if hashlib.sha256(_canonical_manifest()).hexdigest() != RESEARCH_CORPUS_SHA256:
    raise RuntimeError("invalid research corpus manifest")


@dataclass(frozen=True, slots=True)
class ResearchCorpus:
    papers: tuple[PaperRecord, ...]

    def __post_init__(self) -> None:
        if type(self.papers) is not tuple:
            raise ValueError("invalid research corpus")
        if not 1 <= len(self.papers) <= _MAX_CORPUS_SIZE:
            raise ValueError("invalid research corpus")
        if any(type(paper) is not PaperRecord for paper in self.papers):
            raise ValueError("invalid research corpus")
        paper_ids = {paper.paper_id for paper in self.papers}
        if len(paper_ids) != len(self.papers):
            raise ValueError("invalid research corpus")

    def search_papers(self, query: str, limit: int) -> tuple[PaperSummary, ...]:
        if type(query) is not str or type(limit) is not int:
            raise ValueError("invalid paper search")
        if not _valid_text(query, _MAX_QUERY_BYTES) or not 1 <= limit <= 32:
            raise ValueError("invalid paper search")

        folded_query = query.casefold()
        matches = (
            paper
            for paper in self.papers
            if folded_query in paper.title.casefold()
            or folded_query in paper.abstract.casefold()
        )
        ordered = sorted(matches, key=lambda paper: paper.paper_id)
        return tuple(
            PaperSummary(paper.paper_id, paper.title, paper.year)
            for paper in ordered[:limit]
        )

    def get_paper(self, paper_id: str) -> PaperRecord:
        if type(paper_id) is str:
            for paper in self.papers:
                if paper.paper_id == paper_id:
                    return paper
        raise ValueError("unknown paper id")
