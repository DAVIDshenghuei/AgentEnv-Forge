import hashlib
import json

import pytest

import agentenv_forge.mcp.research as research_module

from agentenv_forge.mcp.research import (
    RESEARCH_CORPUS_VERSION,
    PaperRecord,
    ResearchCorpus,
)


def test_research_corpus_has_an_explicit_version_identity() -> None:
    assert RESEARCH_CORPUS_VERSION == "1.0.0"


PAPERS = (
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


def _canonical_manifest(version, papers) -> bytes:
    payload = {
        "version": version,
        "papers": [
            {
                "paper_id": paper.paper_id,
                "title": paper.title,
                "year": paper.year,
                "abstract": paper.abstract,
                "body": paper.body,
            }
            for paper in papers
        ],
    }
    return json.dumps(
        payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def test_research_corpus_manifest_digest_binds_version_and_every_record() -> None:
    expected = "6870b6d0751a8a6743d508cad7e97888f842d47f6e8a13ee8049c9c236b0579e"
    assert research_module.RESEARCH_CORPUS_SHA256 == expected
    assert hashlib.sha256(
        _canonical_manifest(RESEARCH_CORPUS_VERSION, PAPERS)
    ).hexdigest() == expected

    mutations = [("1.0.1", PAPERS)]
    for index, paper in enumerate(PAPERS):
        for field, value in (
            ("paper_id", paper.paper_id + "x"),
            ("title", paper.title + "x"),
            ("year", paper.year + 1),
            ("abstract", paper.abstract + "x"),
            ("body", paper.body + "x"),
        ):
            values = {
                "paper_id": paper.paper_id,
                "title": paper.title,
                "year": paper.year,
                "abstract": paper.abstract,
                "body": paper.body,
            }
            values[field] = value
            changed = PAPERS[:index] + (PaperRecord(**values),) + PAPERS[index + 1 :]
            mutations.append((RESEARCH_CORPUS_VERSION, changed))

    assert all(
        hashlib.sha256(_canonical_manifest(version, papers)).hexdigest()
        != expected
        for version, papers in mutations
    )


def test_search_is_deterministic_bounded_and_summary_only() -> None:
    corpus = ResearchCorpus(PAPERS)

    first = corpus.search_papers(query="agent environment", limit=2)
    second = corpus.search_papers(query="AGENT ENVIRONMENT", limit=2)

    assert type(first) is tuple
    assert first == second
    assert tuple(summary.paper_id for summary in first) == (
        "paper-001",
        "paper-002",
    )
    assert tuple((summary.title, summary.year) for summary in first) == (
        ("Deterministic Agent Environments", 2024),
        ("Offline Tool Evaluation", 2023),
    )
    assert all(not hasattr(summary, "abstract") for summary in first)
    assert all(not hasattr(summary, "body") for summary in first)
    with pytest.raises((AttributeError, TypeError)):
        first[0].title = "changed"


def test_get_paper_returns_exact_immutable_record_and_sanitizes_unknown_id() -> None:
    corpus = ResearchCorpus(PAPERS)

    paper = corpus.get_paper("paper-002")

    assert type(paper) is PaperRecord
    assert paper is PAPERS[2]
    assert paper == PaperRecord(
        paper_id="paper-002",
        title="Offline Tool Evaluation",
        year=2023,
        abstract="An AGENT ENVIRONMENT corpus for offline research.",
        body="second body",
    )
    with pytest.raises((AttributeError, TypeError)):
        paper.body = "changed"
    with pytest.raises(ValueError, match="^unknown paper id$"):
        corpus.get_paper("secret-not-in-corpus")


def test_search_rejects_invalid_exact_types_and_ranges() -> None:
    calls: list[str] = []

    class HostileStr(str):
        def casefold(self):
            calls.append("casefold")
            raise AssertionError("hostile query must not be inspected")

    corpus = ResearchCorpus(PAPERS)
    invalid_calls = (
        ("", 1),
        ("   ", 1),
        (HostileStr("agent"), 1),
        ("agent", True),
        ("agent", 1.0),
        ("agent", "2"),
        ("agent", 0),
        ("agent", 33),
    )
    for query, limit in invalid_calls:
        with pytest.raises(ValueError, match="^invalid paper search$"):
            corpus.search_papers(query=query, limit=limit)

    assert calls == []


def test_paper_record_rejects_unsafe_scalars_without_magic_calls() -> None:
    calls: list[str] = []

    class HostileStr(str):
        def __str__(self):
            calls.append("str")
            raise AssertionError("must not format hostile string")

        def __eq__(self, other):
            calls.append("eq")
            raise AssertionError("must not compare hostile string")

        def casefold(self):
            calls.append("casefold")
            raise AssertionError("must not normalize hostile string")

    valid = {
        "paper_id": "paper-005",
        "title": "Valid title",
        "year": 2025,
        "abstract": "Valid abstract.",
        "body": "Valid body.",
    }
    invalid_updates = (
        {"paper_id": HostileStr("paper-005")},
        {"title": HostileStr("title")},
        {"abstract": HostileStr("abstract")},
        {"body": HostileStr("body")},
        {"year": True},
        {"paper_id": ""},
        {"title": "bad\ncontrol"},
        {"abstract": "bad\ud800surrogate"},
        {"body": "bad\x00text"},
    )
    for update in invalid_updates:
        with pytest.raises(ValueError, match="^invalid paper record$"):
            PaperRecord(**(valid | update))

    assert calls == []


def test_corpus_requires_exact_tuple_of_exact_records() -> None:
    with pytest.raises(ValueError, match="^invalid research corpus$"):
        ResearchCorpus(list(PAPERS))

    class TupleSubclass(tuple):
        pass

    with pytest.raises(ValueError, match="^invalid research corpus$"):
        ResearchCorpus(TupleSubclass(PAPERS))

    class PaperRecordSubclass(PaperRecord):
        pass

    subclassed = PaperRecordSubclass(
        paper_id="paper-005",
        title="Subclass",
        year=2025,
        abstract="Not independently owned.",
        body="body",
    )
    with pytest.raises(ValueError, match="^invalid research corpus$"):
        ResearchCorpus((subclassed,))

    duplicate = PaperRecord(
        paper_id="paper-001",
        title="Duplicate identity",
        year=2025,
        abstract="A duplicate identifier.",
        body="duplicate body",
    )
    with pytest.raises(ValueError, match="^invalid research corpus$"):
        ResearchCorpus(PAPERS + (duplicate,))
