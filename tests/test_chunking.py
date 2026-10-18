import json

import pytest

from digital_twin.chunking import (
    ChunkingSystem,
    chunk_text,
    detect_format,
    extract_entities,
    normalize_text,
    wc,
)

PROSE = (
    "A digital twin is a virtual representation of a physical object or system. "
    "Digital twins are used across various industries. In healthcare, they model patient health. "
    "In manufacturing, they simulate production processes. Smart cities use digital twins for urban planning.\n\n"
    "The technology relies on real-time data from sensors. This data feeds into simulation models. "
    "Machine learning algorithms analyze patterns. Predictions help optimize operations.\n\n"
    "Benefits include improved decision-making. Companies reduce costs through predictive maintenance. "
    "Healthcare providers personalize treatments. City planners optimize resource allocation."
)

MARKDOWN = """---
title: KB Test
---

# KB Test

## Projects

Created on 2026-08-10. Contact davies@example.com or see https://example.com/kb for details.

```python
def add(a, b):
    return a + b
```

| Database | Cost |
|----------|-----:|
| Chroma | $0 |
| Qdrant | $25 |
| Pinecone | $70 |

## Schedule

Davies reviews pull requests every morning at 8:00 and practices talks for 45 min on Thursday evenings.
"""

PLAIN = """WEEKLY SYNC
===========

Date: 2026-09-14
Attendees: Davies, Amara
Duration: 55 min

SERVICE METRICS

Service        Latency   Errors   Owner
-------------  --------  -------  ------
auth-api       120ms     0.4%     Amara
billing        340ms     2.1%     Amara
ingest         95ms      0.2%     Ngozi

The billing failures traced back to an expired token in the payment client. Nothing else looked wrong.
"""

LOG = (
    "2026-09-01 08:00:01,100 INFO  [auth-api] Login ok user=davies@example.com ip=10.0.1.5 took=40ms\n"
    "2026-09-01 08:05:12,200 ERROR [billing] Unhandled exception in job job-123\n"
    "Traceback (most recent call last):\n"
    '  File "/app/worker.py", line 10, in run\n'
    "ValueError: bad record id\n"
    "2026-09-01 08:09:00,300 WARN  [ingest] Slow query detected took=900ms table=events\n"
)

REQUIRED_KEYS = {
    "chunk_id",
    "doc_id",
    "position",
    "total_chunks",
    "prev_id",
    "next_id",
    "source",
    "format",
    "strategy",
    "section",
    "text",
    "word_count",
    "char_count",
    "entities",
    "keywords",
    "issues",
    "notes",
    "valid",
    "timestamp" if False else "created_at",
    "content_hash",
}


# --------------------------------------------------
# construction and validation
# --------------------------------------------------


def test_unknown_strategy_raises_error():
    with pytest.raises(ValueError, match="Unknown strategy"):
        ChunkingSystem(strategy="nope")


def test_overlap_must_be_smaller_than_window():
    with pytest.raises(ValueError, match="overlap must be smaller"):
        ChunkingSystem(words_per_chunk=10, overlap=10)


def test_min_words_cannot_exceed_max_words():
    with pytest.raises(ValueError, match="min_words"):
        ChunkingSystem(min_words=50, max_words=10)


def test_non_string_input_raises_type_error():
    with pytest.raises(TypeError, match="text must be a string"):
        ChunkingSystem().chunk(None)


def test_empty_and_whitespace_text_returns_empty_list():
    cs = ChunkingSystem()
    assert cs.chunk("") == []
    assert cs.chunk("   \n\n  ") == []


# --------------------------------------------------
# every strategy: schema, ordering, content preservation
# --------------------------------------------------


@pytest.mark.parametrize("strategy", ChunkingSystem.STRATEGIES)
def test_every_strategy_returns_uniform_schema(strategy):
    cs = ChunkingSystem(
        strategy=strategy,
        words_per_chunk=30,
        overlap=5,
        min_words=5,
        max_words=60,
        max_chunk_size=50,
    )
    chunks = cs.chunk(PROSE, source="doc.txt")

    assert chunks
    for i, c in enumerate(chunks, 1):
        assert REQUIRED_KEYS <= set(c)
        assert c["position"] == i
        assert c["total_chunks"] == len(chunks)
        assert c["strategy"] == strategy
        assert c["source"] == "doc.txt"
        assert c["word_count"] == wc(c["text"])
        assert c["valid"] == (not c["issues"])
    assert chunks[0]["prev_id"] is None
    assert chunks[-1]["next_id"] is None
    assert chunks[0]["next_id"] == (chunks[1]["chunk_id"] if len(chunks) > 1 else None)


@pytest.mark.parametrize("strategy", ChunkingSystem.STRATEGIES)
def test_every_strategy_preserves_all_words(strategy):
    cs = ChunkingSystem(
        strategy=strategy,
        words_per_chunk=30,
        overlap=5,
        min_words=5,
        max_words=60,
        max_chunk_size=50,
    )
    chunks = cs.chunk(PROSE)
    joined = " ".join(c["text"] for c in chunks).lower()
    for word in ("virtual", "sensors", "maintenance", "allocation"):
        assert word in joined


def test_source_defaults_to_constructor_value():
    chunks = ChunkingSystem(source="default.txt").chunk(PROSE)
    assert all(c["source"] == "default.txt" for c in chunks)


# --------------------------------------------------
# simple strategies
# --------------------------------------------------


def test_sentence_strategy_groups_sentences():
    cs = ChunkingSystem(strategy="sentence", sentences_per_chunk=2, min_words=1)
    chunks = cs.chunk(PROSE)
    assert len(chunks) == 7  # 13 sentences in PROSE: six pairs and one left over


def test_word_strategy_respects_window_and_overlap():
    cs = ChunkingSystem(
        strategy="word", words_per_chunk=20, overlap=5, min_words=1, max_words=100
    )
    chunks = cs.chunk(PROSE)
    assert all(len(c["text"].split()) <= 20 for c in chunks)
    first_tail = chunks[0]["text"].split()[-5:]
    second_head = chunks[1]["text"].split()[:5]
    assert first_tail == second_head
    assert chunks[1]["overlap_words"] == 5


def test_paragraph_strategy_one_chunk_per_paragraph():
    cs = ChunkingSystem(strategy="paragraph", min_words=1)
    assert len(cs.chunk(PROSE)) == 3


def test_smart_strategy_adds_sentence_overlap():
    cs = ChunkingSystem(strategy="smart", max_chunk_size=40, min_words=1, max_words=100)
    chunks = cs.chunk(PROSE)
    assert len(chunks) >= 2
    assert chunks[1]["overlap_words"] > 0
    prev_last_sentence = chunks[0]["text"].rstrip().split(". ")[-1]
    assert prev_last_sentence.rstrip(".") in chunks[1]["text"]


def test_word_strategy_flags_mid_sentence_cuts():
    cs = ChunkingSystem(
        strategy="word", words_per_chunk=12, overlap=2, min_words=1, max_words=100
    )
    chunks = cs.chunk(PROSE)
    assert any("ends mid-sentence" in c["issues"] for c in chunks)


def test_oversize_chunk_is_flagged_for_simple_strategy():
    cs = ChunkingSystem(strategy="paragraph", min_words=1, max_words=10)
    chunks = cs.chunk(PROSE)
    assert any(any(i.startswith("too large") for i in c["issues"]) for c in chunks)


# --------------------------------------------------
# structured strategy: markdown
# --------------------------------------------------


def test_structured_markdown_keeps_code_and_table_intact():
    cs = ChunkingSystem(strategy="structured", max_words=40, min_words=5)
    chunks = cs.chunk(MARKDOWN, source="kb.md")

    code = [c for c in chunks if c["has_code"]]
    table = [c for c in chunks if c["has_table"]]
    assert code and table
    assert "def add(a, b):" in code[0]["text"]
    assert code[0]["code_languages"] == ["python"]
    assert "Pinecone" in table[0]["text"] and "Chroma" in table[0]["text"]
    assert all("unbalanced code fence" not in c["issues"] for c in chunks)


def test_structured_never_crosses_h2_boundaries():
    cs = ChunkingSystem(strategy="structured", max_words=500, min_words=1)
    chunks = cs.chunk(MARKDOWN)
    sections = {c["section"] for c in chunks}
    assert any("Projects" in s for s in sections)
    assert any("Schedule" in s for s in sections)
    for c in chunks:
        assert not ("Projects" in c["text"] and "reviews pull requests" in c["text"])


def test_structured_uses_front_matter_title_and_outline():
    cs = ChunkingSystem()
    doc = cs.add_document("kb.md", MARKDOWN)
    assert doc["title"] == "KB Test"
    assert doc["format"] == "markdown"
    assert (2, "Projects") in doc["outline"]


def test_structured_extracts_entities_into_metadata():
    chunks = ChunkingSystem(max_words=60, min_words=1).chunk(MARKDOWN)
    entities = {}
    for c in chunks:
        for k, v in c["entities"].items():
            entities.setdefault(k, []).extend(v)
    assert "davies@example.com" in entities["email"]
    assert "https://example.com/kb" in entities["url"]
    assert "2026-08-10" in entities["date"]


def test_oversize_table_is_split_with_header_repeated():
    rows = "\n".join(f"| item{i} | {i * 10} extra words here |" for i in range(40))
    md = f"# T\n\n## Data\n\n| Name | Value |\n|------|-------|\n{rows}\n"
    chunks = ChunkingSystem(max_words=40, min_words=1).chunk(md)
    tables = [c for c in chunks if c["has_table"]]
    assert len(tables) > 1
    assert all("| Name | Value |" in c["text"] for c in tables)
    assert all(c["split_atomic"] for c in tables)


# --------------------------------------------------
# structured strategy: plain text and logs
# --------------------------------------------------


def test_plain_text_ascii_table_detected():
    chunks = ChunkingSystem(max_words=80, min_words=1).chunk(PLAIN, source="sync.txt")
    table = next(c for c in chunks if c["has_table"])
    assert "auth-api" in table["text"] and "billing" in table["text"]
    assert table["n_table_rows"] == 3


def test_log_format_detected_and_entries_stay_whole():
    assert detect_format(LOG) == "log"
    chunks = ChunkingSystem(min_words=1, log_max_words=200).chunk(LOG, source="app.log")
    assert chunks[0]["format"] == "log"
    trace = next(c for c in chunks if c["has_stacktrace"])
    assert "Traceback" in trace["text"] and "ValueError: bad record id" in trace["text"]
    assert trace["log_levels"].get("ERROR") == 1
    assert chunks[0]["time_start"] == "2026-09-01 08:00:01.100"
    assert "billing" in chunks[0]["log_components"]


def test_log_chunks_split_on_size_not_on_entry():
    big = "\n".join(
        f"2026-09-01 08:{i // 60:02d}:{i % 60:02d},000 INFO  [svc] message number {i} with some extra words"
        for i in range(120)
    )
    chunks = ChunkingSystem(min_words=1, log_max_words=80).chunk(big)
    assert len(chunks) > 1
    for c in chunks:
        for line in c["text"].split("\n"):
            assert line.startswith("2026-09-01")  # no entry was cut in half


# --------------------------------------------------
# options
# --------------------------------------------------


def test_drop_duplicates_removes_repeated_chunks():
    text = "Same sentence repeated here for testing purposes only.\n\n" * 3
    kept = ChunkingSystem(
        strategy="paragraph", min_words=1, drop_duplicates=True
    ).chunk(text)
    all_ = ChunkingSystem(
        strategy="paragraph", min_words=1, drop_duplicates=False
    ).chunk(text)
    assert len(all_) == 3 and len(kept) == 1
    assert kept[0]["total_chunks"] == 1


def test_custom_entity_extractor_is_used():
    cs = ChunkingSystem(
        min_words=1, entity_extractor=lambda text, only: {"custom": ["x"]}
    )
    assert cs.chunk(PROSE)[0]["entities"] == {"custom": ["x"]}


def test_failing_entity_extractor_falls_back_to_builtin():
    def boom(text, only):
        raise RuntimeError("bad extractor")

    chunks = ChunkingSystem(min_words=1, entity_extractor=boom).chunk(
        "Mail me at a@b.com today please."
    )
    assert chunks[0]["entities"]["email"] == ["a@b.com"]


def test_chunk_text_convenience_function():
    chunks = chunk_text(PROSE, source="x.txt", strategy="paragraph", min_words=1)
    assert len(chunks) == 3 and chunks[0]["source"] == "x.txt"


# --------------------------------------------------
# normalization and helpers
# --------------------------------------------------


def test_normalize_text_fixes_smart_quotes_and_line_endings():
    assert normalize_text("\u201cHi\u201d\r\nthere  \r\n") == '"Hi"\nthere'


def test_extract_entities_basic():
    result = extract_entities("Mail a@b.com on 2026-01-02 at 6:30am #tag")
    assert result["email"] == ["a@b.com"]
    assert result["date"] == ["2026-01-02"]
    assert result["hashtag"] == ["#tag"]


# --------------------------------------------------
# corpus: index, search, reports, export
# --------------------------------------------------


@pytest.fixture
def corpus():
    cs = ChunkingSystem(max_words=60, min_words=5, log_max_words=100)
    return cs.run({"kb.md": MARKDOWN, "sync.txt": PLAIN, "app.log": LOG})


def test_run_ingests_all_documents(corpus):
    assert len(corpus.docs) == 3
    assert {d["format"] for d in corpus.docs} == {"markdown", "text", "log"}
    assert len({c["chunk_id"] for c in corpus.chunks}) == len(corpus.chunks)
    assert all(c["keywords"] for c in corpus.chunks)


def test_duplicate_document_is_skipped(corpus):
    n = len(corpus.chunks)
    result = corpus.add_document("kb.md", MARKDOWN)
    assert result["skipped"] == "duplicate"
    assert len(corpus.chunks) == n


def test_search_finds_relevant_chunk(corpus):
    hits = corpus.search("pinecone qdrant database cost", top_k=2)
    assert hits and hits[0]["chunk"]["has_table"]


def test_search_filters(corpus):
    hits = corpus.search("billing", filters={"level": "ERROR"})
    assert hits and all("ERROR" in h["chunk"]["log_levels"] for h in hits)
    with pytest.raises(ValueError, match="Unknown filter"):
        corpus.search("billing", filters={"bogus": 1})


def test_find_entity(corpus):
    ids = corpus.find_entity("email", "davies@example.com")
    assert ids
    assert any(c["chunk_id"] in ids for c in corpus.chunks)


def test_quality_report_has_expected_fields(corpus):
    r = corpus.quality_report()
    for key in (
        "documents",
        "chunks",
        "valid_ratio",
        "token_coverage",
        "self_retrieval_at_3",
        "overall_score",
        "grade",
    ):
        assert key in r
    assert r["documents"] == 3
    assert 0 <= r["overall_score"] <= 100
    assert r["token_coverage"] > 0.95


def test_quality_report_on_empty_corpus():
    assert ChunkingSystem().quality_report() == {"error": "no chunks"}


def test_jsonl_and_index_export(corpus, tmp_path):
    jl, ix = tmp_path / "chunks.jsonl", tmp_path / "index.json"
    corpus.to_jsonl(str(jl))
    corpus.save_index(str(ix))
    lines = jl.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == len(corpus.chunks)
    assert json.loads(lines[0])["chunk_id"] == corpus.chunks[0]["chunk_id"]
    assert "postings" in json.loads(ix.read_text())


def test_ingest_directory(tmp_path):
    (tmp_path / "a.md").write_text(MARKDOWN, encoding="utf-8")
    (tmp_path / "b.txt").write_text(PLAIN, encoding="utf-8")
    (tmp_path / "skip.bin").write_bytes(b"\x00\x01")
    cs = ChunkingSystem(min_words=1)
    docs = cs.ingest_directory(str(tmp_path))
    assert len(docs) == 2


# --------------------------------------------------
# process() pipeline
# --------------------------------------------------


def test_process_pipeline_steps_and_stats():
    cs = ChunkingSystem(min_words=1)
    result = cs.process(
        "Hi!  I'm Alex (alex@email.com).\n\nI run 5km at 6:30am. #running",
        source="p.txt",
    )

    assert result["pipeline_steps"] == ["cleaned", "extracted", "chunked"]
    assert result["extracted_data"]["email"] == ["alex@email.com"]
    assert result["chunks"] and result["chunks"][0]["source"] == "p.txt"
    assert result["original_stats"]["words"] == result["processed_stats"]["words"]
    assert len(cs.processing_history) == 1


def test_process_without_chunking():
    result = ChunkingSystem().process("Some text here.", chunk=False)
    assert result["chunks"] == [] and "chunked" not in result["pipeline_steps"]


def test_process_non_string_raises_type_error():
    with pytest.raises(TypeError, match="text must be a string"):
        ChunkingSystem().process(None)
