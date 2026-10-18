"""Retrieval for the digital twin: find the stored facts that answer a question.

Retrieval only. This file never calls a model; ``agent.py`` makes the only model call, so every
reply still passes through the safety layer. A caller (the ``search_history`` tool, or any future
pre-retrieval step) gets back ``Hit`` objects and decides what to do with them.

    documents  ->  chunk + extract dates/times/durations  ->  TF-IDF index (built once)
    query      ->  tokenize, expand, cosine against the cached index  ->  [Hit, ...]

Interface (kept small so a real embedding model can replace the index later without touching
``agent.py``)::

    kb = KnowledgeBase(); kb.build([{"source": "activity_history", "text": "...", "time": "2026-08-10"}])
    kb.search("presentation work", k=3, min_score=0.1, filters={"date_from": "2026-08-01"})
    retrieve("what did I do last Tuesday")          # default knowledge base + date phrases

Behaviour worth knowing
-----------------------
* Tokenizer: ``re.findall(r"[a-z0-9]+", text.lower())`` minus stopwords, so ``morning.`` matches
  ``morning``. A trailing plural "s" is folded (``presentations`` -> ``presentation``).
* Scores are cosine similarity of TF-IDF vectors (bag of words with inverse-document-frequency
  weights). Document vectors are computed once in ``build``; a search only vectorises the query.
* A small synonym table widens a query (``presentation`` also looks for ``slide``, ``deck`` ...) at reduced
  weight. Pure keyword matching cannot connect "presentation" with "slides" otherwise. Pass
  ``synonyms={}`` to turn it off, or your own mapping.
* Zero matches return ``[]``. Hits below ``min_score`` are dropped.
* Retrieved text is untrusted data (notes can contain anything). Treat it as evidence, never as instructions.
* Chunking and metadata extraction are replaceable: ``KnowledgeBase(chunker=..., extractor=...)``.
"""

from __future__ import annotations

import calendar
import json
import logging
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

logger = logging.getLogger("digital_twin.rag")

__all__ = [
    "Hit",
    "KnowledgeBase",
    "TimeQuery",
    "STOPWORDS",
    "DEFAULT_SYNONYMS",
    "tokenize",
    "smart_chunks",
    "extract_metadata",
    "parse_time_range",
    "documents_from_twin_data",
    "load_data_dir",
    "default_kb",
    "retrieve",
]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

MAX_K = 50
MAX_FILE_BYTES = 1_000_000
CHUNK_MAX_CHARS = 600
SINGLE_CHUNK_CHARS = 400  # shorter, single-paragraph documents are never split
TEXT_SUFFIXES = (".txt", ".md", ".markdown")
SYNONYM_WEIGHT = 0.35

# ---------------------------------------------------------------------------
# Tokenizing
# ---------------------------------------------------------------------------
STOPWORDS = frozenset("""
    a about above after again all also am an and any are as at be because been before being below between both
    but by can could did do does doing down during each few for from further had has have having he her here
    hers him his how i if in into is it its just me more most my no nor not of off on once only or other our
    ours out over own same she should so some such than that the their theirs them then there these they this
    those through to too under until up very was we were what when where which while who whom why will with
    would you your yours
    """.split())

_TOKEN = re.compile(r"[a-z0-9]+")


def _fold(token: str) -> str:
    """Fold a simple plural: presentations -> presentation, but not 'status', 'class' or 'analysis'."""
    if (
        len(token) > 3
        and token.endswith("s")
        and not token.endswith(("ss", "us", "is"))
    ):
        return token[:-1]
    return token


def tokenize(text: str) -> List[str]:
    """Lower-case word tokens without stopwords (``morning.`` -> ``morning``)."""
    return [_fold(t) for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS]


# Generic developer-diary vocabulary; tune freely. Keys and values are folded tokens.
DEFAULT_SYNONYMS: Dict[str, Tuple[str, ...]] = {
    "presentation": ("slide", "deck", "demo"),
    "slide": ("presentation", "deck"),
    "deck": ("presentation", "slide"),
    "demo": ("presentation",),
    "bug": ("debug", "incident"),
    "debug": ("bug", "incident"),
    "incident": ("bug", "debug"),
    "research": ("paper", "read"),
    "paper": ("research",),
    "test": ("unit",),
}


# ---------------------------------------------------------------------------
# Chunking and metadata extraction (replaceable)
# ---------------------------------------------------------------------------
_SENTENCE = re.compile(r"(?<=[.!?])\s+")


def smart_chunks(text: str, max_chars: int = CHUNK_MAX_CHARS) -> List[str]:
    """Split on paragraphs, packing neighbours up to ``max_chars``; oversized paragraphs split by sentence.

    A built-in stand-in for ``chunking.py``'s ``smart`` strategy: pass your own function as
    ``KnowledgeBase(chunker=...)`` to use that instead.
    """
    pieces: List[str] = []
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = " ".join(paragraph.split())
        if not paragraph:
            continue
        if len(paragraph) <= max_chars:
            pieces.append(paragraph)
            continue
        current = ""
        for sentence in _SENTENCE.split(paragraph):
            while (
                len(sentence) > max_chars
            ):  # a single enormous sentence: cut on a word boundary
                cut = sentence.rfind(" ", 0, max_chars) or max_chars
                cut = cut if cut > 0 else max_chars
                if current:
                    pieces.append(current)
                    current = ""
                pieces.append(sentence[:cut].strip())
                sentence = sentence[cut:].strip()
            if current and len(current) + 1 + len(sentence) > max_chars:
                pieces.append(current)
                current = sentence
            else:
                current = f"{current} {sentence}".strip()
        if current:
            pieces.append(current)

    chunks: List[str] = []
    buffer = ""
    for piece in pieces:  # pack short neighbours together
        if buffer and len(buffer) + 1 + len(piece) > max_chars:
            chunks.append(buffer)
            buffer = piece
        else:
            buffer = f"{buffer} {piece}".strip()
    if buffer:
        chunks.append(buffer)
    return chunks


_MONTHS = {name.lower(): i for i, name in enumerate(calendar.month_name) if name}
_MONTHS.update({name.lower(): i for i, name in enumerate(calendar.month_abbr) if name})
_MONTH_ALT = "|".join(sorted(_MONTHS, key=len, reverse=True))
_ISO_DATE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_DMY = re.compile(
    rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({_MONTH_ALT})\.?,?\s+(\d{{4}})\b", re.I
)
_MDY = re.compile(
    rf"\b({_MONTH_ALT})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(\d{{4}})\b", re.I
)
_TIME = re.compile(r"\b(?:\d{1,2}:\d{2}\s?(?:am|pm)?|\d{1,2}\s?(?:am|pm))\b", re.I)
_DURATION = re.compile(
    r"\b\d+(?:\.\d+)?\s?(?:hours?|hrs?|minutes?|mins?|days?|weeks?)\b", re.I
)


def _safe_date(year: int, month: int, day: int) -> Optional[date]:
    try:
        return date(year, month, day)
    except ValueError:
        return None


def extract_metadata(text: str) -> Dict[str, List[str]]:
    """Dates (ISO), clock times and durations mentioned in ``text``.

    A built-in stand-in for ``extract_profile_data``: pass your own as
    ``KnowledgeBase(extractor=...)``; it must return a dict (extra keys are kept as metadata).
    """
    dates: List[str] = []

    def add(d: Optional[date]) -> None:
        if d is not None and d.isoformat() not in dates:
            dates.append(d.isoformat())

    for m in _ISO_DATE.finditer(text):
        add(_safe_date(int(m[1]), int(m[2]), int(m[3])))
    for m in _DMY.finditer(text):
        add(_safe_date(int(m[3]), _MONTHS[m[2].lower()], int(m[1])))
    for m in _MDY.finditer(text):
        add(_safe_date(int(m[3]), _MONTHS[m[1].lower()], int(m[2])))
    return {
        "dates": dates,
        "times": [" ".join(t.split()) for t in _TIME.findall(text)],
        "durations": [" ".join(d.split()) for d in _DURATION.findall(text)],
    }


def _to_date(value: Any) -> Optional[date]:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        m = _ISO_DATE.search(value)
        if m:
            return _safe_date(int(m[1]), int(m[2]), int(m[3]))
    return None


def _strict_date(value: Any, name: str) -> Optional[date]:
    if value is None or value == "":
        return None
    parsed = _to_date(value)
    if parsed is None:
        raise ValueError(f"{name} must be a date or an ISO string (YYYY-MM-DD).")
    return parsed


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Hit:
    text: str
    source: str
    score: float
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "source": self.source,
            "score": self.score,
            "meta": dict(self.meta),
        }


@dataclass(frozen=True)
class _Chunk:
    text: str
    source: str
    meta: Dict[str, Any]
    day: Optional[date]
    norm: float


@dataclass(frozen=True)
class _State:
    """Everything a search needs, built once. Replaced as a whole by ``build`` (no half-built reads)."""

    chunks: Tuple[_Chunk, ...] = ()
    index: Dict[str, Tuple[Tuple[int, float], ...]] = field(
        default_factory=dict
    )  # term -> ((chunk, weight), ...)
    idf: Dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Knowledge base
# ---------------------------------------------------------------------------
Chunker = Callable[[str], Sequence[str]]
Extractor = Callable[[str], Mapping[str, Any]]
_FILTER_KEYS = frozenset({"date_from", "date_to", "source"})


class KnowledgeBase:
    def __init__(
        self,
        *,
        chunker: Optional[Chunker] = None,
        extractor: Optional[Extractor] = None,
        synonyms: Optional[Mapping[str, Sequence[str]]] = None,
        single_chunk_chars: int = SINGLE_CHUNK_CHARS,
    ) -> None:
        self._chunker = chunker or smart_chunks
        self._extractor = extractor or extract_metadata
        self._synonyms = dict(DEFAULT_SYNONYMS if synonyms is None else synonyms)
        self._single_chunk_chars = single_chunk_chars
        self._state = _State()
        self.skipped = 0

    # -- build ------------------------------------------------------------------------------
    def build(self, documents: Iterable[Mapping[str, Any]]) -> None:
        """(Re)build the index from ``{"source": str, "text": str, "time"?: date|str, "meta"?: dict, "chunk"?: bool}``.

        Short single-paragraph documents (activity entries) stay one chunk; longer ones go through the
        chunker. Invalid documents are skipped and counted in ``skipped``. Replaces any earlier index.
        """
        if isinstance(documents, (str, bytes, Mapping)):
            raise TypeError("documents must be a list of dicts.")
        chunks: List[_Chunk] = []
        term_counts: List[Counter] = []
        skipped = 0

        for doc in documents:
            if (
                not isinstance(doc, Mapping)
                or not isinstance(doc.get("text"), str)
                or not doc["text"].strip()
            ):
                skipped += 1
                continue
            source = str(doc.get("source") or "unknown")
            text = doc["text"].strip()
            doc_day = _to_date(doc.get("time"))
            extra = doc.get("meta") if isinstance(doc.get("meta"), Mapping) else {}
            whole = doc.get("chunk") is False or (
                len(text) <= self._single_chunk_chars and "\n" not in text
            )
            pieces = (
                [text]
                if whole
                else [
                    p.strip()
                    for p in self._chunker(text)
                    if isinstance(p, str) and p.strip()
                ]
            )
            for position, piece in enumerate(pieces):
                try:
                    found = dict(self._extractor(piece))
                except Exception:  # a broken extractor must not lose the document
                    logger.exception(
                        "metadata extraction failed for a chunk of %s", source
                    )
                    found = {}
                first_extracted = next(
                    (d for d in map(_to_date, found.get("dates", []) or []) if d), None
                )
                day = doc_day or first_extracted
                meta: Dict[str, Any] = {
                    **found,
                    "source": source,
                    "date": day.isoformat() if day else None,
                    "doc_time": (
                        str(doc.get("time")) if doc.get("time") is not None else None
                    ),
                    "chunk": position,
                    "chunks": len(pieces),
                }
                for key, value in extra.items():
                    meta.setdefault(str(key), value)
                counts = Counter(tokenize(piece))
                if not counts:
                    skipped += 1
                    continue
                chunks.append(_Chunk(piece, source, meta, day, 0.0))
                term_counts.append(counts)

        self._state = self._index(chunks, term_counts)
        self.skipped = skipped
        logger.info("knowledge base built: %d chunks, %d skipped", len(chunks), skipped)

    @staticmethod
    def _index(chunks: List[_Chunk], term_counts: List[Counter]) -> _State:
        n = len(chunks)
        df: Counter = Counter()
        for counts in term_counts:
            df.update(counts.keys())
        idf = {term: math.log((1 + n) / (1 + freq)) + 1.0 for term, freq in df.items()}

        postings: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
        finished: List[_Chunk] = []
        for i, (chunk, counts) in enumerate(zip(chunks, term_counts)):
            weights = {t: c * idf[t] for t, c in counts.items()}
            norm = math.sqrt(sum(w * w for w in weights.values()))
            finished.append(
                _Chunk(chunk.text, chunk.source, chunk.meta, chunk.day, norm)
            )
            for term, weight in weights.items():
                postings[term].append((i, weight))
        return _State(tuple(finished), {t: tuple(p) for t, p in postings.items()}, idf)

    # -- info -------------------------------------------------------------------------------
    @property
    def size(self) -> int:
        return len(self._state.chunks)

    @property
    def latest_date(self) -> Optional[date]:
        days = [c.day for c in self._state.chunks if c.day]
        return max(days) if days else None

    # -- search -----------------------------------------------------------------------------
    def _query_vector(self, query: str, idf: Mapping[str, float]) -> Dict[str, float]:
        weights: Dict[str, float] = {}
        for term, count in Counter(tokenize(query)).items():
            weights[term] = max(weights.get(term, 0.0), float(count))
        for term in list(weights):
            for alt in self._synonyms.get(term, ()):
                alt = _fold(alt.lower())
                weights.setdefault(alt, SYNONYM_WEIGHT)
        # Terms no document contains cannot match; they must not dilute the cosine either.
        return {t: w * idf[t] for t, w in weights.items() if t in idf}

    @staticmethod
    def _check_filters(
        filters: Optional[Mapping[str, Any]],
    ) -> Tuple[Optional[date], Optional[date], Tuple[str, ...]]:
        if not filters:
            return None, None, ()
        if not isinstance(filters, Mapping):
            raise TypeError("filters must be a dict.")
        unknown = set(filters) - _FILTER_KEYS
        if unknown:
            raise ValueError(
                f"Unknown filter(s): {', '.join(sorted(map(str, unknown)))}."
            )
        start = _strict_date(filters.get("date_from"), "date_from")
        end = _strict_date(filters.get("date_to"), "date_to")
        if start and end and start > end:
            raise ValueError("date_from is after date_to.")
        sources = filters.get("source")
        if sources is None:
            wanted: Tuple[str, ...] = ()
        elif isinstance(sources, str):
            wanted = (sources,)
        else:
            wanted = tuple(str(s) for s in sources)
        return start, end, wanted

    @staticmethod
    def _matches(
        chunk: _Chunk,
        start: Optional[date],
        end: Optional[date],
        sources: Tuple[str, ...],
    ) -> bool:
        if (start or end) and chunk.day is None:
            return False
        if start and chunk.day < start:  # type: ignore[operator]
            return False
        if end and chunk.day > end:  # type: ignore[operator]
            return False
        if sources and not any(
            chunk.source == s or (s.endswith("*") and chunk.source.startswith(s[:-1]))
            for s in sources
        ):
            return False
        return True

    def search(
        self,
        query: str,
        k: int = 3,
        min_score: float = 0.1,
        filters: Optional[Mapping[str, Any]] = None,
    ) -> List[Hit]:
        """The ``k`` best chunks for ``query``, best first. ``[]`` when nothing matches.

        filters   ``date_from`` / ``date_to`` (inclusive, date or ISO string; chunks without a date are
                  excluded when either is set) and ``source`` (a name, a list, or a prefix ending in ``*``).
        """
        if not isinstance(query, str):
            raise TypeError("query must be a string.")
        if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= MAX_K:
            raise ValueError(f"k must be a whole number from 1 to {MAX_K}.")
        if (
            isinstance(min_score, bool)
            or not isinstance(min_score, (int, float))
            or not 0 <= min_score <= 1
        ):
            raise ValueError("min_score must be between 0 and 1.")
        start, end, sources = self._check_filters(filters)

        state = self._state  # one consistent snapshot even if build() runs meanwhile
        if not state.chunks:
            return []
        qvec = self._query_vector(query, state.idf)
        if not qvec:
            return []
        qnorm = math.sqrt(sum(w * w for w in qvec.values()))

        dots: Dict[int, float] = defaultdict(float)
        for term, qweight in qvec.items():
            for i, weight in state.index.get(term, ()):
                dots[i] += qweight * weight

        scored: List[Tuple[float, int]] = []
        for i, dot in dots.items():
            chunk = state.chunks[i]
            if chunk.norm == 0 or not self._matches(chunk, start, end, sources):
                continue
            score = dot / (qnorm * chunk.norm)
            if score >= min_score:
                scored.append((score, i))
        scored.sort(
            key=lambda s: (
                -s[0],
                -(state.chunks[s[1]].day.toordinal() if state.chunks[s[1]].day else 0),
                s[1],
            )
        )
        return [
            Hit(
                state.chunks[i].text,
                state.chunks[i].source,
                round(score, 4),
                dict(state.chunks[i].meta),
            )
            for score, i in scored[:k]
        ]

    def in_range(
        self,
        k: int = 3,
        filters: Optional[Mapping[str, Any]] = None,
    ) -> List[Hit]:
        """Chunks matching only the filters, newest first (for questions that are only about *when*)."""
        start, end, sources = self._check_filters(filters)
        state = self._state
        picked = [c for c in state.chunks if self._matches(c, start, end, sources)]
        picked.sort(key=lambda c: (c.day.toordinal() if c.day else 0), reverse=True)
        return [Hit(c.text, c.source, 1.0, dict(c.meta)) for c in picked[:k]]


# ---------------------------------------------------------------------------
# Date phrases in a query ("last Tuesday", "in August", "past 7 days")
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TimeQuery:
    text: str  # the query with the date phrase removed
    date_from: Optional[date] = None
    date_to: Optional[date] = None
    phrase: str = ""


_WEEKDAYS = {name.lower(): i for i, name in enumerate(calendar.day_name)}
_DAY_ALT = "|".join(_WEEKDAYS)
_RX_ISO = re.compile(
    r"\b(on|since|from|after|before|until|till)?\s*(\d{4})-(\d{2})-(\d{2})\b", re.I
)
_RX_MONTH = re.compile(rf"\b(?:in|during)\s+({_MONTH_ALT})(?:\s+(\d{{4}}))?\b", re.I)
_RX_LAST_DAY = re.compile(rf"\blast\s+({_DAY_ALT})\b", re.I)
_RX_PAST_N = re.compile(
    r"\b(?:in\s+the\s+)?(?:past|last)\s+(\d{1,3})\s+(day|week|month)s?\b", re.I
)
_RX_WORDS = re.compile(
    r"\b(today|yesterday|this\s+week|last\s+week|this\s+month|last\s+month)\b", re.I
)


def _month_range(year: int, month: int) -> Tuple[date, date]:
    return date(year, month, 1), date(year, month, calendar.monthrange(year, month)[1])


def parse_time_range(query: str, today: Optional[date] = None) -> TimeQuery:
    """Find one date phrase in ``query`` and turn it into an inclusive date range.

    Understood: ``today``, ``yesterday``, ``this/last week`` (Monday to Sunday), ``this/last month``,
    ``last <weekday>`` (the most recent one before today), ``past/last N days|weeks|months``,
    ``in <month> [year]`` (a month that has not happened yet this year means last year), and ISO dates
    with ``on / since / from / after / before / until``. Anything else leaves the query unchanged.
    Relative phrases are relative to ``today`` (default: the real date).
    """
    today = today or date.today()
    text = query or ""

    def done(
        m: "re.Match[str]", start: Optional[date], end: Optional[date]
    ) -> TimeQuery:
        rest = " ".join((text[: m.start()] + " " + text[m.end() :]).split())
        return TimeQuery(rest, start, end, m.group(0).strip())

    m = _RX_ISO.search(text)
    if m:
        day = _safe_date(int(m[2]), int(m[3]), int(m[4]))
        if day:
            word = (m[1] or "on").lower()
            if word in ("since", "from"):
                return done(m, day, None)
            if word == "after":
                return done(m, day + timedelta(days=1), None)
            if word == "before":
                return done(m, None, day - timedelta(days=1))
            if word in ("until", "till"):
                return done(m, None, day)
            return done(m, day, day)

    m = _RX_MONTH.search(text)
    if m:
        month = _MONTHS[m[1].lower()]
        year = (
            int(m[2])
            if m[2]
            else (today.year if month <= today.month else today.year - 1)
        )
        start, end = _month_range(year, month)
        return done(m, start, end)

    m = _RX_LAST_DAY.search(text)
    if m:
        delta = (today.weekday() - _WEEKDAYS[m[1].lower()]) % 7 or 7
        day = today - timedelta(days=delta)
        return done(m, day, day)

    m = _RX_PAST_N.search(text)
    if m:
        span = int(m[1]) * {"day": 1, "week": 7, "month": 30}[m[2].lower()]
        return done(m, today - timedelta(days=span), today)

    m = _RX_WORDS.search(text)
    if m:
        word = " ".join(m[1].lower().split())
        monday = today - timedelta(days=today.weekday())
        if word == "today":
            return done(m, today, today)
        if word == "yesterday":
            return done(m, today - timedelta(days=1), today - timedelta(days=1))
        if word == "this week":
            return done(m, monday, today)
        if word == "last week":
            return done(m, monday - timedelta(days=7), monday - timedelta(days=1))
        if word == "this month":
            return done(m, today.replace(day=1), today)
        last_of_prev = today.replace(day=1) - timedelta(days=1)
        return done(m, *_month_range(last_of_prev.year, last_of_prev.month))

    return TimeQuery(" ".join(text.split()))


# ---------------------------------------------------------------------------
# Loading the twin's data
# ---------------------------------------------------------------------------
def _flatten(prefix: str, value: Any) -> Iterable[Tuple[str, str]]:
    """(path, text) pairs for a profile-like structure: one short fact per field."""
    if isinstance(value, Mapping):
        for key, inner in value.items():
            yield from _flatten(f"{prefix}.{key}" if prefix else str(key), inner)
    elif isinstance(value, (list, tuple)):
        items = [str(v) for v in value if isinstance(v, (str, int, float))]
        if items:
            yield prefix, "; ".join(items)
    elif (
        isinstance(value, (str, int, float))
        and not isinstance(value, bool)
        and str(value).strip()
    ):
        yield prefix, str(value)


def documents_from_twin_data(data: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Turn ``twin_data.json`` into documents: one chunk per activity entry and per profile field."""
    docs: List[Dict[str, Any]] = []
    for entry in data.get("activity_history") or []:
        if isinstance(entry, Mapping):
            text = entry.get("activity") or entry.get("text")
            if isinstance(text, str) and text.strip():
                docs.append(
                    {
                        "source": "activity_history",
                        "text": text,
                        "time": entry.get("time"),
                        "chunk": False,
                    }
                )
    for section in ("user_profile", "temporal_patterns"):
        for path, text in _flatten("", data.get(section) or {}):
            docs.append(
                {
                    "source": f"{section}.{path}",
                    "text": f"{path.replace('_', ' ').replace('.', ' ')}: {text}",
                    "chunk": False,
                }
            )
    return docs


_DATED_NAME = re.compile(r"^(\d{4}-\d{2}-\d{2})")


def load_data_dir(directory: Union[str, Path, None] = None) -> List[Dict[str, Any]]:
    """Read ``.txt`` / ``.md`` notes under the data folder (not ``twin_data.json``) as documents.

    Files are read as UTF-8, must resolve inside the folder (no symlinks out) and are capped at 1 MB.
    A file named ``2026-08-10-notes.md`` gets that date. Unreadable files are skipped.
    """
    root = Path(directory if directory is not None else DATA_DIR)
    if not root.is_dir():
        return []
    base = root.resolve()
    docs: List[Dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        try:
            resolved = path.resolve()
            if (
                not path.is_file()
                or path.suffix.lower() not in TEXT_SUFFIXES
                or path.name.startswith(".")
                or base not in resolved.parents
            ):
                continue
            if resolved.stat().st_size > MAX_FILE_BYTES:
                logger.warning(
                    "skipping %s: larger than %d bytes", path.name, MAX_FILE_BYTES
                )
                continue
            text = resolved.read_text(encoding="utf-8", errors="replace")
        except OSError:
            logger.warning("could not read %s", path.name)
            continue
        if text.strip():
            m = _DATED_NAME.match(path.name)
            docs.append(
                {
                    "source": path.relative_to(root).as_posix(),
                    "text": text,
                    "time": m.group(1) if m else None,
                }
            )
    return docs


def _twin_data() -> Mapping[str, Any]:
    """``config.load_twin_data()``; falls back to reading the JSON directly when the config module
    cannot be imported (it raises if GROQ_API_KEY is unset, which retrieval does not need).
    """
    try:
        from .config import load_twin_data

        return load_twin_data()
    except (RuntimeError, ImportError):
        with open(DATA_DIR / "twin_data.json", "r", encoding="utf-8") as handle:
            return json.load(handle)


@lru_cache(maxsize=1)
def default_kb() -> KnowledgeBase:
    """The shared knowledge base: twin data plus notes in ``data/``. Built once; clear with ``cache_clear()``."""
    documents: List[Dict[str, Any]] = []
    try:
        documents += documents_from_twin_data(_twin_data())
    except (OSError, ValueError, TypeError, KeyError, ImportError):
        logger.warning("could not load the twin data for retrieval", exc_info=True)
    documents += load_data_dir()
    kb = KnowledgeBase()
    kb.build(documents)
    return kb


def retrieve(
    query: str,
    k: int = 3,
    *,
    min_score: float = 0.1,
    filters: Optional[Mapping[str, Any]] = None,
    today: Optional[date] = None,
) -> List[Hit]:
    """Search the default knowledge base, understanding date phrases such as "last Tuesday".

    A date phrase becomes a ``date_from`` / ``date_to`` filter (explicit ``filters`` win) and is removed
    from the text that is scored. A question that is only about *when* ("what did I do yesterday") returns
    the entries in that range, newest first, each with score 1.0 (matched by date, not by wording).
    ``today`` anchors relative phrases; it defaults to the real date, so for old data pass the data's date.
    """
    if not isinstance(query, str):
        raise TypeError("query must be a string.")
    kb = default_kb()
    parsed = parse_time_range(query, today)
    merged: Dict[str, Any] = {}
    if parsed.date_from:
        merged["date_from"] = parsed.date_from
    if parsed.date_to:
        merged["date_to"] = parsed.date_to
    merged.update({k_: v for k_, v in (filters or {}).items() if v is not None})

    if (parsed.date_from or parsed.date_to) and not tokenize(parsed.text):
        return kb.in_range(k, merged)
    return kb.search(
        parsed.text if parsed.phrase else query,
        k=k,
        min_score=min_score,
        filters=merged or None,
    )
