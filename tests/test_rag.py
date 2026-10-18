"""Tests for digital_twin.rag (retrieval only; no model, no network, no API key)."""

import json
import sys
from datetime import date
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from digital_twin import rag  # noqa: E402
from digital_twin.rag import (
    Hit,
    KnowledgeBase,
    parse_time_range,
    tokenize,
)  # noqa: E402

ACTIVITIES = [
    ("2026-08-10", "reviewed pull requests for the backend team"),
    ("2026-08-11", "practiced a technical presentation with the team"),
    ("2026-08-12", "created a presentation comparing cloud providers"),
    ("2026-08-13", "wrote unit tests for the payment module"),
    ("2026-08-14", "created slides on edge computing"),
    ("2026-08-17", "prepared talking points for a stakeholder demo"),
    ("2026-08-18", "debugged a memory leak in the worker service"),
    ("2026-08-19", "read a research paper on vector databases"),
    ("2026-08-25", "went for a 45 minute run before breakfast"),
]
TWIN = {
    "user_profile": {"name": "Davies", "skills": ["python", "sql"]},
    "activity_history": [{"time": d, "activity": a} for d, a in ACTIVITIES],
}


@pytest.fixture()
def kb():
    base = KnowledgeBase()
    base.build(rag.documents_from_twin_data(TWIN))
    return base


# --- tokenizing -------------------------------------------------------------
def test_tokenize_lowercases_strips_punctuation_and_stopwords():
    assert tokenize("The Morning, run!") == ["morning", "run"]


def test_tokenize_folds_plurals_but_not_status():
    assert tokenize("presentations status analysis") == [
        "presentation",
        "status",
        "analysis",
    ]


# --- search -----------------------------------------------------------------
def test_presentation_work_returns_presentation_entries_with_scores_and_sources(kb):
    hits = kb.search("presentation work", k=3)
    assert {h.text for h in hits[:2]} == {
        "practiced a technical presentation with the team",
        "created a presentation comparing cloud providers",
    }
    assert all(
        isinstance(h, Hit) and h.source == "activity_history" and 0.1 <= h.score <= 1
        for h in hits
    )
    assert hits == sorted(hits, key=lambda h: -h.score)


def test_synonyms_reach_slides_and_demo_entries(kb):
    texts = {h.text for h in kb.search("presentation work", k=10)}
    assert "created slides on edge computing" in texts
    assert "prepared talking points for a stakeholder demo" in texts


def test_synonyms_can_be_disabled():
    base = KnowledgeBase(synonyms={})
    base.build(rag.documents_from_twin_data(TWIN))
    texts = {h.text for h in base.search("presentation work", k=10)}
    assert "created slides on edge computing" not in texts


def test_zero_matches_returns_empty_list(kb):
    assert kb.search("quantum gardening") == []
    assert kb.search("") == []
    assert kb.search("the of and") == []


def test_empty_knowledge_base_returns_empty():
    assert KnowledgeBase().search("anything") == []


def test_min_score_drops_weak_hits(kb):
    assert kb.search("presentation work", min_score=0.99) == []
    assert len(kb.search("presentation work", k=10, min_score=0.0)) >= len(
        kb.search("presentation work", k=10)
    )


def test_k_limits_results(kb):
    assert len(kb.search("presentation", k=1)) == 1


@pytest.mark.parametrize("k", [0, -1, 51, True, 2.5])
def test_bad_k_raises(kb, k):
    with pytest.raises(ValueError):
        kb.search("x", k=k)


def test_bad_min_score_and_query_type(kb):
    with pytest.raises(ValueError):
        kb.search("x", min_score=2)
    with pytest.raises(TypeError):
        kb.search(None)


# --- filters ----------------------------------------------------------------
def test_date_filter_restricts_results(kb):
    hits = kb.search(
        "presentation",
        k=10,
        filters={"date_from": "2026-08-12", "date_to": "2026-08-12"},
    )
    assert [h.text for h in hits] == [
        "created a presentation comparing cloud providers"
    ]


def test_date_filter_excludes_undated_chunks(kb):
    assert kb.search("python skills", filters={"date_from": "2026-01-01"}) == []


def test_source_filter_and_prefix(kb):
    assert kb.search("python", filters={"source": "activity_history"}) == []
    assert kb.search("python", filters={"source": "user_profile*"})


def test_bad_filters_raise(kb):
    with pytest.raises(ValueError):
        kb.search("x", filters={"colour": "red"})
    with pytest.raises(ValueError):
        kb.search("x", filters={"date_from": "tomorrow"})
    with pytest.raises(ValueError):
        kb.search("x", filters={"date_from": "2026-09-01", "date_to": "2026-08-01"})


# --- build ------------------------------------------------------------------
def test_build_skips_invalid_documents_and_replaces_index():
    base = KnowledgeBase()
    base.build(
        [{"source": "a", "text": "alpha note"}, {"text": ""}, "junk", {"text": 5}]
    )
    assert base.size == 1 and base.skipped == 3
    base.build([{"source": "b", "text": "beta note"}])
    assert base.search("alpha") == [] and base.search("beta")


def test_build_rejects_a_string():
    with pytest.raises(TypeError):
        KnowledgeBase().build("text")


def test_document_matrix_is_built_once_not_per_search(kb, monkeypatch):
    calls = []
    monkeypatch.setattr(
        KnowledgeBase, "_index", staticmethod(lambda *a: calls.append(1))
    )
    kb.search("presentation")
    kb.search("tests")
    assert calls == []


def test_long_documents_are_chunked_and_metadata_extracted():
    long_note = "\n\n".join(
        [
            f"On 2026-09-0{i} I worked on the report for 2 hours at 9:30 am and discussed budgets."
            * 3
            for i in range(1, 6)
        ]
    )
    base = KnowledgeBase()
    base.build([{"source": "notes.md", "text": long_note}])
    assert base.size > 1
    hit = base.search("report budgets", k=1)[0]
    assert (
        hit.meta["dates"]
        and "2 hours" in hit.meta["durations"]
        and hit.meta["date"].startswith("2026-09")
    )
    assert hit.meta["chunks"] == base.size


def test_activity_entries_stay_single_chunks(kb):
    assert (
        kb.size == len(ACTIVITIES) + 2
    )  # + one chunk each for the profile's name and skills
    hit = kb.search("45 minute run", k=1)[0]
    assert hit.meta["durations"] == ["45 minute"] and hit.meta["date"] == "2026-08-25"


def test_broken_extractor_does_not_lose_documents():
    def boom(_):
        raise RuntimeError("x")

    base = KnowledgeBase(extractor=boom)
    base.build([{"source": "a", "text": "alpha note"}])
    assert base.search("alpha")


def test_custom_chunker_is_used():
    base = KnowledgeBase(chunker=lambda t: t.split("|"), single_chunk_chars=5)
    base.build([{"source": "a", "text": "alpha one|beta two|gamma three"}])
    assert base.size == 3


# --- date phrases -----------------------------------------------------------
TODAY = date(2026, 8, 20)  # a Thursday


@pytest.mark.parametrize(
    "query,start,end",
    [
        ("what did I do yesterday", date(2026, 8, 19), date(2026, 8, 19)),
        ("last Tuesday", date(2026, 8, 18), date(2026, 8, 18)),
        ("last Thursday", date(2026, 8, 13), date(2026, 8, 13)),
        ("this week", date(2026, 8, 17), TODAY),
        ("last week work", date(2026, 8, 10), date(2026, 8, 16)),
        ("in August", date(2026, 8, 1), date(2026, 8, 31)),
        ("in December", date(2025, 12, 1), date(2025, 12, 31)),
        ("past 7 days", date(2026, 8, 13), TODAY),
        ("since 2026-08-12", date(2026, 8, 12), None),
        ("before 2026-08-12", None, date(2026, 8, 11)),
        ("on 2026-08-12", date(2026, 8, 12), date(2026, 8, 12)),
        ("last month", date(2026, 7, 1), date(2026, 7, 31)),
    ],
)
def test_parse_time_range(query, start, end):
    parsed = parse_time_range(query, TODAY)
    assert (parsed.date_from, parsed.date_to) == (start, end)


def test_parse_time_range_removes_phrase_and_ignores_plain_text():
    assert (
        parse_time_range("presentation work last Tuesday", TODAY).text
        == "presentation work"
    )
    plain = parse_time_range("presentation work", TODAY)
    assert (
        plain.date_from is None
        and plain.text == "presentation work"
        and plain.phrase == ""
    )
    assert parse_time_range("on 2026-13-45", TODAY).date_from is None


# --- default kb / retrieve ----------------------------------------------------
@pytest.fixture()
def default(tmp_path, monkeypatch):
    (tmp_path / "2026-08-20-diary.md").write_text(
        "Spent the day on the quarterly budget review.", encoding="utf-8"
    )
    (tmp_path / "ignore.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(rag, "DATA_DIR", tmp_path)
    monkeypatch.setattr(rag, "_twin_data", lambda: TWIN)
    rag.default_kb.cache_clear()
    yield
    rag.default_kb.cache_clear()


def test_retrieve_presentation_work(default):
    hits = rag.retrieve("presentation work")
    assert hits and all(h.source == "activity_history" for h in hits)
    assert {h.text for h in hits[:2]} <= {
        "practiced a technical presentation with the team",
        "created a presentation comparing cloud providers",
        "created slides on edge computing",
        "prepared talking points for a stakeholder demo",
    }


def test_retrieve_with_date_phrase_uses_today_anchor(default):
    hits = rag.retrieve("what did I do last Tuesday", today=TODAY)
    assert [h.text for h in hits] == ["debugged a memory leak in the worker service"]
    assert hits[0].score == 1.0


def test_retrieve_date_plus_topic(default):
    hits = rag.retrieve("presentation in August", k=10)
    assert hits and all(h.meta["date"].startswith("2026-08") for h in hits)
    assert rag.retrieve("presentation", filters={"source": "diary*"}) == []


def test_default_kb_is_cached_and_clearable(default):
    first = rag.default_kb()
    assert rag.default_kb() is first
    rag.default_kb.cache_clear()
    assert rag.default_kb() is not first


def test_data_dir_notes_are_loaded_with_filename_date(default):
    hit = rag.retrieve("budget review", k=1)[0]
    assert hit.source == "2026-08-20-diary.md" and hit.meta["date"] == "2026-08-20"
    assert all(
        h.source != "ignore.json"
        for h in rag.default_kb().search("anything", min_score=0)
    )


def test_load_data_dir_skips_large_hidden_and_symlinked(tmp_path):
    (tmp_path / "ok.txt").write_text("fine note", encoding="utf-8")
    (tmp_path / ".hidden.md").write_text("secret", encoding="utf-8")
    (tmp_path / "big.md").write_text("x" * (rag.MAX_FILE_BYTES + 1), encoding="utf-8")
    outside = tmp_path.parent / "outside_rag_test.txt"
    outside.write_text("outside", encoding="utf-8")
    try:
        (tmp_path / "link.txt").symlink_to(outside)
    except OSError:
        pass
    assert [d["source"] for d in rag.load_data_dir(tmp_path)] == ["ok.txt"]
    assert rag.load_data_dir(tmp_path / "missing") == []


def test_twin_data_falls_back_to_json_when_config_missing(tmp_path, monkeypatch):
    (tmp_path / "twin_data.json").write_text(json.dumps(TWIN), encoding="utf-8")
    monkeypatch.setattr(rag, "DATA_DIR", tmp_path)
    assert rag._twin_data()["user_profile"]["name"] == "Davies"
