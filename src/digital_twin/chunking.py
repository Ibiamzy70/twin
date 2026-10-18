from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import statistics
import textwrap
import unicodedata
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Deque, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)

__all__ = [
    "ChunkingSystem",
    "SearchIndex",
    "Block",
    "Draft",
    "chunk_text",
    "visualize_chunks",
    "extract_entities",
    "normalize_text",
    "detect_format",
    "wc",
]


ENTITY_PATTERNS: Dict[str, str] = {
    "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
    "url": r"https?://[^\s<>()\]\"']+",
    "phone": r"(?<!\d)(?:\+\d{1,3}[-.\s])?\d{3}[-.]\d{3}[-.]\d{4}(?!\d)",
    "handle": r"(?<![\w.])@\w[\w-]*",
    "date": r"\b\d{4}-\d{2}-\d{2}\b",
    "time": r"\b\d{1,2}:\d{2}(?::\d{2})?(?:\s?[ap]m)?\b",
    "duration": r"\b\d+\s?(?:hours?|hrs?|minutes?|mins?|seconds?|secs?|ms|h|m|s)\b",
    "hashtag": r"(?<!\w)#[A-Za-z]\w*",
    "money": r"[$€£₦]\s?\d[\d,]*(?:\.\d+)?",
    "ip": r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
}
COMPILED = {k: re.compile(v, re.IGNORECASE) for k, v in ENTITY_PATTERNS.items()}
CODE_ENTITY_TYPES = {
    "email",
    "url",
    "ip",
}  # noise control: only these are read from code

STOPWORDS = set(
    """a an and are as at be but by for from has have he her his i if in into is it its
of on or our she so that the their them then there these they this to was we were what when which
who will with you your not no can could should would may might also than too very just about over
under more most other some such only own same both each few any all been being do does did done""".split()
)

SMART_CHARS = {
    "\u2018": "'",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u2013": "-",
    "\u2014": "-",
    "\u2026": "...",
    "\u00a0": " ",
}

WORD_RE = re.compile(r"\w+(?:['\u2019.\-]\w+)*")
FENCE_RE = re.compile(r"^\s{0,3}(`{3,}|~{3,})\s*([\w+#.-]*)")
HEADING_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(.+?)\s*#*\s*$")
TABLE_SEP_RE = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$")
LIST_RE = re.compile(r"^(\s*)(?:[-*+]|\d+[.)])\s+(.*)")
LIST_FULL_RE = re.compile(r"^(\s*(?:[-*+]|\d+[.)])\s+)(.*)")
QUOTE_RE = re.compile(r"^\s{0,3}>\s?(.*)")
HR_RE = re.compile(r"^\s*([-*_])(?:\s*\1){2,}\s*$")
KV_RE = re.compile(r"^[A-Za-z][\w /()-]{0,28}:\s+\S")
CAPS_RE = re.compile(r"^[A-Z][A-Z0-9 &/,\-']{2,60}:?$")
LOG_RE = re.compile(
    r"^\[?(?P<ts>\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?Z?)\]?\s+"
    r"\[?(?P<level>TRACE|DEBUG|INFO|WARN(?:ING)?|ERROR|CRITICAL|FATAL)\]?\b\s*[:\-|]?\s*(?P<msg>.*)$"
)

PROSE = ("paragraph", "quote")
ATOMIC = ("table", "code", "log_entry")
LOG_EXTENSIONS = {".log"}
MARKDOWN_EXTENSIONS = {".md", ".markdown"}
TEXT_EXTENSIONS = {".txt", ".md", ".markdown", ".log"}
SENTENCE_END_CHARS = ".!?:;\"')"

EntityExtractor = Callable[[str, Optional[set]], Dict[str, List[str]]]


def wc(text: str) -> int:
    """Word count used everywhere (keeps 'well-known', '3.13', '2026-09-01' as one word)."""
    return len(WORD_RE.findall(text))


def _check_text(text: Any) -> None:
    if not isinstance(text, str):
        raise TypeError("text must be a string")


# =====================================================================
# 2. NORMALIZATION AND CLEANING
# =====================================================================
def normalize_text(text: str) -> str:
    """Document-level normalization. Safe for code: never touches inner indentation."""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    for a, b in SMART_CHARS.items():
        text = text.replace(a, b)
    text = "".join(
        ch for ch in text if ch in "\n\t" or unicodedata.category(ch)[0] != "C"
    )
    text = re.sub(r"[^\S\n]+$", "", text, flags=re.M)  # trailing whitespace per line
    return text.strip("\n")


def clean_inline(s: str) -> str:
    """Strip markdown/HTML decoration from prose; keep URLs as 'text (url)'."""
    s = re.sub(r"<!--.*?-->", "", s, flags=re.S)
    codes: List[str] = []

    def stash(m):
        codes.append(m.group(1))
        return f"\x00{len(codes) - 1}\x00"

    s = re.sub(r"`([^`\n]+)`", stash, s)
    s = re.sub(
        r"!\[([^\]]*)\]\(([^)]*)\)", lambda m: f"[image: {m.group(1) or m.group(2)}]", s
    )
    s = re.sub(
        r"\[([^\]]+)\]\(([^)\s]*)\)",
        lambda m: (
            f"{m.group(1)} ({m.group(2)})"
            if m.group(2).startswith("http")
            else m.group(1)
        ),
        s,
    )
    s = re.sub(r"<(https?://[^>\s]+|[^@\s<>]+@[^>\s]+)>", r"\1", s)
    s = re.sub(
        r"</?(?:b|i|u|em|strong|br|p|div|span|a|img|sup|sub|code|pre|details|summary|center|font|"
        r"h[1-6]|ul|ol|li|table|tr|td|th)\b[^>\n]*>",
        "",
        s,
        flags=re.I,
    )
    s = re.sub(r"(\*\*|__)(?=\S)(.+?)(?<=\S)\1", r"\2", s)
    s = re.sub(r"(?<![\w*])\*(?=\S)(.+?)(?<=\S)\*(?![\w*])", r"\1", s)
    s = re.sub(r"(?<![\w_])_(?=\S)(.+?)(?<=\S)_(?![\w_])", r"\1", s)
    s = re.sub(r"~~(.+?)~~", r"\1", s)
    s = re.sub(r"\\([\\`*_{}\[\]()#+\-.!|])", r"\1", s)
    s = re.sub(r"\x00(\d+)\x00", lambda m: codes[int(m.group(1))], s)
    return re.sub(r"[ \t]+", " ", s).strip()


def split_sentences(text: str) -> List[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])", text.strip()) if s]


def guess_language(code: str) -> str:
    c = code.strip()
    if re.match(r"^[\[{]", c) and re.search(r"[\]}]$", c):
        return "json"
    if re.search(r"^\s*(def |class |import |from \w+ import )", c, re.M):
        return "python"
    if re.search(r"^\s*(SELECT|INSERT|UPDATE|DELETE|CREATE|WITH)\b", c, re.I | re.M):
        return "sql"
    if re.search(r"^\s*(\$ |sudo |pip |docker |git |curl |cd |export |echo )", c, re.M):
        return "bash"
    if re.search(r"^\s*[\w-]+:\s*(\S.*)?$", c, re.M) and "{" not in c:
        return "yaml"
    return "text"


# =====================================================================
# 3. ENTITY EXTRACTION
# =====================================================================
def extract_entities(text: str, only: Optional[set] = None) -> Dict[str, List[str]]:
    """Return {entity_type: [unique values in order of appearance]}."""
    out: Dict[str, List[str]] = {}
    for name, rx in COMPILED.items():
        if only and name not in only:
            continue
        found = [m.strip() for m in rx.findall(text)]
        if name == "url":
            found = [u.rstrip(".,;:") for u in found]
        if name == "hashtag":
            found = [h.lower() for h in found]
        if found:
            out[name] = list(dict.fromkeys(found))
    return out


def merge_entities(dicts: Iterable[Dict[str, List[str]]]) -> Dict[str, List[str]]:
    merged: Dict[str, List[str]] = defaultdict(list)
    for d in dicts:
        for k, vals in d.items():
            merged[k].extend(vals)
    return {k: list(dict.fromkeys(v)) for k, v in merged.items()}


# =====================================================================
# 4. FORMAT DETECTION AND BLOCK PARSING
# =====================================================================
@dataclass
class Block:
    type: str  # heading paragraph quote list keyvalue table code log_entry
    text: str
    section: Tuple[str, ...] = ()
    meta: Dict[str, Any] = field(default_factory=dict)
    index_text: Optional[str] = None

    @property
    def itext(self) -> str:
        return self.index_text if self.index_text is not None else self.text


def detect_format(text: str, name: str = "") -> str:
    """Return 'markdown', 'log' or 'text'. File extension wins, then content sniffing."""
    ext = "." + name.lower().rsplit(".", 1)[-1] if "." in name else ""
    if ext in MARKDOWN_EXTENSIONS:
        return "markdown"
    if ext in LOG_EXTENSIONS:
        return "log"
    lines = [l for l in text.splitlines() if l.strip()][:200]
    # Stack-trace lines between entries do not match LOG_RE, so a log with one
    # short traceback can sit at exactly half. Accept half or more, with at least
    # two timestamped entries so a lone dated line never turns prose into a log.
    hits = sum(1 for l in lines if LOG_RE.match(l))
    if lines and hits >= 2 and hits / len(lines) >= 0.5:
        return "log"
    if (
        re.search(r"^#{1,6}\s+\S", text, re.M)
        or re.search(r"^\s{0,3}(```|~~~)", text, re.M)
        or re.search(r"^\s*\|?\s*:?-{2,}:?\s*\|", text, re.M)
        or text.startswith("---\n")
    ):
        return "markdown"
    return "text"


def parse_front_matter(text: str) -> Tuple[Dict[str, str], str]:
    if not text.startswith("---\n"):
        return {}, text
    end = text.find("\n---", 4)
    if end == -1:
        return {}, text
    meta = {}
    for line in text[4:end].split("\n"):
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip().strip("\"'")
    return meta, text[end + 4 :].lstrip("\n")


def render_table(header: List[str], rows: List[List[str]]) -> str:
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def make_table_block(header, rows, section, **extra) -> Block:
    header = [h or f"col{i + 1}" for i, h in enumerate(header)]
    width = len(header)
    rows = [(r + [""] * width)[:width] for r in rows]
    records = [dict(zip(header, r)) for r in rows]
    index_text = "\n".join(
        "; ".join(f"{h}: {v}" for h, v in zip(header, r) if v) for r in rows
    )
    meta = {
        "header": header,
        "rows": rows,
        "records": records,
        "n_rows": len(rows),
        **extra,
    }
    return Block(
        "table",
        render_table(header, rows),
        section,
        meta,
        f"table {' '.join(header)}\n{index_text}",
    )


def make_code_block(lang, code, section, **extra) -> Block:
    fence = "```"
    while fence in code:
        fence += "`"
    return Block(
        "code",
        f"{fence}{lang}\n{code}\n{fence}",
        section,
        {"lang": lang, "code": code, **extra},
        f"{lang} code\n{code}",
    )


def _split_row(line: str) -> List[str]:
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|") and not s.endswith("\\|"):
        s = s[:-1]
    return [clean_inline(c.replace("\\|", "|")) for c in re.split(r"(?<!\\)\|", s)]


def _is_table_start(lines: List[str], i: int) -> bool:
    return (
        i + 1 < len(lines)
        and "|" in lines[i]
        and "|" in lines[i + 1]
        and bool(TABLE_SEP_RE.match(lines[i + 1]))
    )


def _ascii_table(lines: List[str], i: int):
    """Whitespace-aligned table: >=3 consistent rows split by 2+ spaces."""
    j, ncols, rows = i, None, []
    while j < len(lines):
        l = lines[j]
        if not l.strip():
            break
        if re.fullmatch(r"[\s\-=+|]+", l):
            j += 1
            continue
        cells = re.split(r"\s{2,}", l.strip())
        if len(cells) < 2 or (ncols and len(cells) != ncols):
            break
        ncols = ncols or len(cells)
        rows.append([clean_inline(c) for c in cells])
        j += 1
    return (rows, j) if len(rows) >= 3 else None


def _starts_block(lines: List[str], j: int) -> bool:
    l = lines[j]
    return bool(
        FENCE_RE.match(l)
        or HEADING_RE.match(l)
        or HR_RE.match(l)
        or QUOTE_RE.match(l)
        or LIST_RE.match(l)
        or _is_table_start(lines, j)
    )


def parse_log(text: str) -> List[Block]:
    """One block per log entry; continuation lines (stack traces) stay with their entry."""
    blocks: List[Block] = []
    cur, pre = None, []

    def finish(c):
        ts = c["ts"].replace(",", ".").replace("T", " ")
        cont = "\n".join(c["cont"])
        msg = c["msg"]
        comp = re.match(r"^\[([\w.\-]+)\]\s*", msg)
        level = "WARN" if c["level"].upper() == "WARNING" else c["level"].upper()
        body = c["first"] + ("\n" + cont if cont else "")
        blocks.append(
            Block(
                "log_entry",
                body,
                (ts[:10],),
                {
                    "ts": ts,
                    "level": level,
                    "component": comp.group(1) if comp else "",
                    "has_trace": bool(cont),
                    "entity_text": msg + ("\n" + cont if cont else ""),
                },
                f"{level} {msg}" + ("\n" + cont if cont else ""),
            )
        )

    for line in text.split("\n"):
        m = LOG_RE.match(line)
        if m:
            if cur:
                finish(cur)
            cur = {
                "ts": m.group("ts"),
                "level": m.group("level"),
                "msg": m.group("msg"),
                "first": line.rstrip(),
                "cont": [],
            }
        elif cur is not None:
            if line.strip():
                cur["cont"].append(line.rstrip())
        elif line.strip():
            pre.append(line.strip())
    if cur:
        finish(cur)
    if pre:
        blocks.insert(0, Block("paragraph", clean_inline(" ".join(pre)), ("preamble",)))
    return blocks


def parse_blocks(text: str, fmt: str) -> List[Block]:
    """Parse normalized text into typed blocks, tracking the heading path as `section`."""
    if fmt == "log":
        return parse_log(text)
    plain = fmt == "text"
    lines = text.split("\n")
    n, i = len(lines), 0
    blocks: List[Block] = []
    stack: List[Tuple[int, str]] = []

    def section() -> Tuple[str, ...]:
        return tuple(t for _, t in stack)

    def heading(level: int, title: str):
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))
        blocks.append(Block("heading", title, section(), {"level": level}))

    while i < n:
        line = lines[i]
        if not line.strip():
            i += 1
            continue

        # fenced code -------------------------------------------------
        m = FENCE_RE.match(line)
        if m:
            fence, lang = m.group(1), m.group(2)
            closing = re.compile(
                r"^\s{0,3}" + re.escape(fence[0]) + "{%d,}\\s*$" % len(fence)
            )
            j, body, closed = i + 1, [], False
            while j < n:
                if closing.match(lines[j]):
                    closed = True
                    break
                body.append(lines[j])
                j += 1
            code = textwrap.dedent("\n".join(body)).rstrip()
            blocks.append(
                make_code_block(
                    lang or guess_language(code), code, section(), unclosed=not closed
                )
            )
            i = j + 1 if closed else j
            continue

        # headings ----------------------------------------------------
        m = HEADING_RE.match(line)
        if m:
            heading(len(m.group(1)), clean_inline(m.group(2)))
            i += 1
            continue
        if HR_RE.match(line):
            i += 1
            continue

        # tables ------------------------------------------------------
        if _is_table_start(lines, i):
            header = _split_row(line)
            j, rows = i + 2, []
            while j < n and lines[j].strip() and "|" in lines[j]:
                rows.append(_split_row(lines[j]))
                j += 1
            blocks.append(make_table_block(header, rows, section()))
            i = j
            continue

        # plain-text specials ----------------------------------------
        if plain:
            if line.startswith(("    ", "\t")):
                j, buf = i, []
                while j < n and (
                    lines[j].startswith(("    ", "\t"))
                    or (
                        not lines[j].strip()
                        and j + 1 < n
                        and lines[j + 1].startswith(("    ", "\t"))
                    )
                ):
                    buf.append(lines[j])
                    j += 1
                if sum(1 for b in buf if b.strip()) >= 2:
                    code = textwrap.dedent("\n".join(buf)).rstrip()
                    blocks.append(
                        make_code_block(guess_language(code), code, section())
                    )
                    i = j
                    continue
            at = _ascii_table(lines, i)
            if at:
                rows, j = at
                blocks.append(
                    make_table_block(rows[0], rows[1:], section(), ascii=True)
                )
                i = j
                continue
        if (
            i + 1 < n
            and re.match(r"^\s{0,3}(=+|-+)\s*$", lines[i + 1])
            and len(lines[i + 1].strip()) >= 3
            and not LIST_RE.match(line)
            and "|" not in line
        ):
            heading(1 if "=" in lines[i + 1] else 2, clean_inline(line))
            i += 2
            continue
        if (
            plain
            and CAPS_RE.match(line.strip())
            and len(re.findall(r"[A-Z]", line)) >= 3
        ):
            heading(2, line.strip().rstrip(":").title())
            i += 1
            continue

        # quotes ------------------------------------------------------
        if QUOTE_RE.match(line):
            j, buf = i, []
            while j < n and QUOTE_RE.match(lines[j]):
                buf.append(QUOTE_RE.match(lines[j]).group(1).strip())
                j += 1
            blocks.append(
                Block("quote", clean_inline(" ".join(b for b in buf if b)), section())
            )
            i = j
            continue

        # lists -------------------------------------------------------
        if LIST_RE.match(line):
            j, items = i, []
            while j < n:
                l = lines[j]
                if LIST_RE.match(l):
                    items.append(l.rstrip())
                    j += 1
                elif not l.strip() and j + 1 < n and LIST_RE.match(lines[j + 1]):
                    j += 1
                elif (
                    l.strip()
                    and l.startswith((" ", "\t"))
                    and items
                    and not FENCE_RE.match(l)
                ):
                    items[-1] += " " + l.strip()
                    j += 1
                else:
                    break
            rendered = []
            for it in items:
                mm = LIST_FULL_RE.match(it)
                rendered.append(
                    mm.group(1) + clean_inline(mm.group(2)) if mm else clean_inline(it)
                )
            blocks.append(
                Block(
                    "list", "\n".join(rendered), section(), {"n_items": len(rendered)}
                )
            )
            i = j
            continue

        # paragraph / key-value --------------------------------------
        j, buf = i, []
        while j < n and lines[j].strip() and (j == i or not _starts_block(lines, j)):
            buf.append(lines[j].strip())
            j += 1
        if len(buf) >= 2 and all(KV_RE.match(b) for b in buf):
            pairs = {
                b.split(":", 1)[0].strip(): clean_inline(b.split(":", 1)[1])
                for b in buf
            }
            blocks.append(
                Block(
                    "keyvalue",
                    "\n".join(f"{k}: {v}" for k, v in pairs.items()),
                    section(),
                    {"pairs": pairs},
                )
            )
        else:
            blocks.append(Block("paragraph", clean_inline(" ".join(buf)), section()))
        i = j
    return [b for b in blocks if b.type == "heading" or b.text.strip()]


# =====================================================================
# 5. DRAFTS AND THE STRUCTURE-AWARE PACKER
# =====================================================================
@dataclass
class Draft:
    """A chunk under construction: ordered pieces plus overlap bookkeeping."""

    key: tuple
    pieces: List[Block] = field(default_factory=list)
    words: int = 0  # includes overlap words
    overlap_text: str = ""  # text prepended from the previous chunk
    overlap_words: int = 0  # words that are repeats from the previous chunk
    closed_by: str = ""
    same_key_neighbor: bool = False


def _greedy(units: List[str], limit: int, joiner: str) -> List[str]:
    out, cur, w = [], [], 0
    for u in units:
        uw = wc(u)
        if cur and w + uw > limit:
            out.append(joiner.join(cur))
            cur, w = [], 0
        cur.append(u)
        w += uw
    if cur:
        out.append(joiner.join(cur))
    return out


def _mark_neighbors(drafts: List[Draft]) -> None:
    for k, d in enumerate(drafts):
        d.same_key_neighbor = (k > 0 and drafts[k - 1].key == d.key) or (
            k + 1 < len(drafts) and drafts[k + 1].key == d.key
        )


class _Packer:
    """Packs blocks into drafts. Tables/code/log entries are atomic; if one must be
    split, tables repeat their header row and code parts are re-fenced."""

    def __init__(
        self,
        max_words: int,
        min_words: int,
        log_max_words: int,
        overlap_sentences: int,
        split_level: int,
    ):
        self.max_words, self.min_words = max_words, min_words
        self.log_max_words = log_max_words
        self.overlap_sentences = overlap_sentences
        self.split_level = split_level  # H1/H2 = hard boundaries, H3+ may share a chunk

    # ---- splitting oversize blocks -----------------------------------
    def _split_table(self, b: Block, limit: int) -> List[Block]:
        header, rows = b.meta["header"], b.meta["rows"]
        hw = wc(" ".join(header))
        groups, cur, w = [], [], hw
        for r in rows:
            rw = wc(" ".join(r))
            if cur and w + rw > limit:
                groups.append(cur)
                cur, w = [], hw
            cur.append(r)
            w += rw
        if cur:
            groups.append(cur)
        return [
            make_table_block(header, g, b.section, split=True, part=k + 1)
            for k, g in enumerate(groups)
        ]

    def _split_code(self, b: Block, limit: int) -> List[Block]:
        lang, lines = b.meta["lang"], b.meta["code"].split("\n")
        groups, cur, w = [], [], 0
        for ln in lines:
            lw = wc(ln)
            if cur and w + lw > limit:
                blank = max((k for k, x in enumerate(cur) if not x.strip()), default=-1)
                if blank > len(cur) // 2:
                    groups.append(cur[:blank])
                    cur = cur[blank + 1 :]
                else:
                    groups.append(cur)
                    cur = []
                w = sum(wc(x) for x in cur)
            cur.append(ln)
            w += lw
        if cur:
            groups.append(cur)
        codes = [c for c in ("\n".join(g).strip("\n") for g in groups) if c.strip()]
        return [
            make_code_block(lang, c, b.section, split=True, part=k + 1)
            for k, c in enumerate(codes)
        ]

    def _explode(self, b: Block, limit: int) -> List[Block]:
        if b.type == "log_entry" or wc(b.text) <= limit:
            return [b]
        if b.type == "table":
            return self._split_table(b, limit)
        if b.type == "code":
            return self._split_code(b, limit)
        units, joiner = (
            (b.text.split("\n"), "\n")
            if b.type in ("list", "keyvalue")
            else (split_sentences(b.text), " ")
        )
        fixed = []
        for u in units:
            if wc(u) > limit:
                ws = u.split()
                fixed += [" ".join(ws[k : k + limit]) for k in range(0, len(ws), limit)]
            else:
                fixed.append(u)
        return [
            Block(b.type, g, b.section, {**b.meta, "split": True})
            for g in _greedy(fixed, limit, joiner)
        ]

    def _last_sentence(self, text: str, limit: int) -> str:
        s = split_sentences(text)
        return s[-1] if s and wc(s[-1]) <= max(1, limit // 4) else ""

    # ---- packing ------------------------------------------------------
    def build(
        self, blocks: List[Block], fmt: str
    ) -> Tuple[List[Draft], Dict[str, int], int]:
        is_log = fmt == "log"
        limit = self.log_max_words if is_log else self.max_words
        drafts: List[Draft] = []
        state: Dict[str, Any] = {"cur": None, "seed": "", "seed_key": None}
        stats = {"atomic_total": 0, "atomic_split": 0, "blocks": 0}

        def close(reason: str):
            cur = state["cur"]
            if cur is None:
                return
            cur.closed_by = reason
            drafts.append(cur)
            last = cur.pieces[-1]
            state["seed"] = (
                self._last_sentence(last.text, limit)
                if (not is_log and self.overlap_sentences and last.type in PROSE)
                else ""
            )
            state["seed_key"], state["cur"] = cur.key, None

        for b in blocks:
            if b.type == "heading":
                continue
            stats["blocks"] += 1
            parts = self._explode(b, limit)
            if b.type in ("table", "code"):
                stats["atomic_total"] += 1
                stats["atomic_split"] += int(len(parts) > 1)
            key = () if is_log else tuple(b.section[: self.split_level])
            for p in parts:
                w = wc(p.text)
                if state["cur"] is not None and state["cur"].key != key:
                    close("boundary")
                cur = state["cur"]
                if cur is not None and cur.words + w > limit:
                    # keep an intro line ending in ':' together with the table/code it introduces
                    keep = (
                        cur.pieces[-1].type in PROSE
                        and cur.pieces[-1].text.rstrip().endswith(":")
                        and p.type in ("table", "code")
                        and cur.words + w <= limit * 1.5
                    )
                    if not keep:
                        close("size")
                if state["cur"] is None:
                    ov = (
                        state["seed"]
                        if (
                            state["seed"]
                            and state["seed_key"] == key
                            and p.type in PROSE
                        )
                        else ""
                    )
                    state["cur"] = Draft(
                        key=key, overlap_text=ov, overlap_words=wc(ov), words=wc(ov)
                    )
                state["cur"].pieces.append(p)
                state["cur"].words += w
        close("end")

        merged: List[Draft] = []  # fold tiny tail chunks into their predecessor
        for d in drafts:
            own = d.words - d.overlap_words
            if (
                merged
                and merged[-1].key == d.key
                and own < self.min_words
                and merged[-1].words + own <= limit * 1.25
            ):
                prev = merged[-1]
                prev.pieces += d.pieces
                prev.words += own
                prev.closed_by = d.closed_by
            else:
                merged.append(d)
        _mark_neighbors(merged)
        return merged, stats, limit


# =====================================================================
# 6. SEARCH INDEX (BM25 + structured lookups)
# =====================================================================
def stem(t: str) -> str:
    if len(t) <= 3 or t.isdigit():
        return t
    for suf, rep in (
        ("ies", "y"),
        ("sses", "ss"),
        ("ches", "ch"),
        ("shes", "sh"),
        ("xes", "x"),
        ("ing", ""),
        ("ed", ""),
        ("s", ""),
    ):
        if t.endswith(suf) and len(t) - len(suf) >= 3:
            if suf == "s" and t.endswith("ss"):
                return t
            return t[: -len(suf)] + rep
    return t


def tokenize(text: str, keep_stop: bool = False) -> List[str]:
    toks = re.findall(r"[a-z0-9_]+", text.lower())
    if not keep_stop:
        toks = [t for t in toks if t not in STOPWORDS and len(t) > 1]
    return [stem(t) for t in toks]


class SearchIndex:
    """In-memory BM25 index with entity / section / block-type / log-level lookups."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.chunks: List[dict] = []
        self.postings: Dict[str, Dict[int, int]] = defaultdict(dict)
        self.doc_len: List[int] = []
        self.avg_len = 0.0
        self.by_entity: Dict[str, Dict[str, List[str]]] = defaultdict(
            lambda: defaultdict(list)
        )
        self.by_section: Dict[str, List[str]] = defaultdict(list)
        self.by_block_type: Dict[str, List[str]] = defaultdict(list)
        self.by_level: Dict[str, List[str]] = defaultdict(list)

    def _terms(self, c: dict) -> List[str]:
        terms = tokenize(c["context"]) * 2 + tokenize(c["index_text"])
        for vals in c["entities"].values():
            terms += [v.lower() for v in vals]
        return terms

    def build(self, chunks: List[dict]) -> "SearchIndex":
        self.chunks = chunks
        for i, c in enumerate(chunks):
            terms = self._terms(c)
            self.doc_len.append(len(terms))
            for t, tf in Counter(terms).items():
                self.postings[t][i] = tf
            for etype, vals in c["entities"].items():
                for v in vals:
                    self.by_entity[etype][v.lower()].append(c["chunk_id"])
            self.by_section[c["section"]].append(c["chunk_id"])
            for bt in c["block_types"]:
                self.by_block_type[bt].append(c["chunk_id"])
            for lv in c["log_levels"]:
                self.by_level[lv].append(c["chunk_id"])
        self.avg_len = sum(self.doc_len) / max(1, len(self.doc_len))
        return self

    def _query_terms(self, query: str) -> List[str]:
        terms = tokenize(query) or tokenize(query, keep_stop=True)
        for w in query.split():
            w = w.strip(".,;:!?()\"'").lower()
            if w and re.search(r"[@./:\-]", w):
                terms.append(w)  # exact entity-like token (email, ip, url, date)
        return terms

    @staticmethod
    def _match(c: dict, filters: dict) -> bool:
        for k, v in filters.items():
            if k == "format":
                ok = c["format"] == v
            elif k == "source":
                ok = c["source"] == v
            elif k == "block_type":
                ok = v in c["block_types"]
            elif k in ("has_code", "has_table"):
                ok = c[k] == v
            elif k == "level":
                ok = v.upper() in c["log_levels"]
            elif k == "section":
                ok = v.lower() in c["section"].lower()
            elif k == "entity_type":
                ok = bool(c["entities"].get(v))
            elif k == "language":
                ok = v in c["code_languages"]
            else:
                raise ValueError(f"Unknown filter '{k}'")
            if not ok:
                return False
        return True

    def search(
        self, query: str, top_k: int = 5, filters: Optional[dict] = None
    ) -> List[dict]:
        N, scores = len(self.chunks), defaultdict(float)
        avg = max(self.avg_len, 1e-9)
        for t in set(self._query_terms(query)):
            plist = self.postings.get(t)
            if not plist:
                continue
            idf = math.log(1 + (N - len(plist) + 0.5) / (len(plist) + 0.5))
            for i, tf in plist.items():
                norm = tf + self.k1 * (1 - self.b + self.b * self.doc_len[i] / avg)
                scores[i] += idf * tf * (self.k1 + 1) / norm
        ranked = sorted(
            (
                (s, i)
                for i, s in scores.items()
                if not filters or self._match(self.chunks[i], filters)
            ),
            reverse=True,
        )[:top_k]
        qtoks = [
            w.lower() for w in re.findall(r"\w+", query) if w.lower() not in STOPWORDS
        ]
        out = []
        for s, i in ranked:
            c = self.chunks[i]
            snippet = next(
                (
                    ln
                    for ln in c["text"].split("\n")
                    if any(q in ln.lower() for q in qtoks)
                ),
                c["text"].split("\n")[0],
            )
            out.append(
                {
                    "score": round(s, 3),
                    "chunk_id": c["chunk_id"],
                    "section": c["section"],
                    "source": c["source"],
                    "snippet": snippet[:160],
                    "chunk": c,
                }
            )
        return out

    def find_entity(self, etype: str, value: str) -> List[str]:
        return self.by_entity.get(etype, {}).get(value.lower(), [])

    def stats(self) -> dict:
        return {
            "chunks": len(self.chunks),
            "vocabulary": len(self.postings),
            "postings": sum(len(p) for p in self.postings.values()),
            "avg_terms_per_chunk": round(self.avg_len, 1),
            "entity_values": sum(len(v) for v in self.by_entity.values()),
            "sections": len(self.by_section),
        }

    def to_dict(self) -> dict:
        return {
            "params": {"k1": self.k1, "b": self.b},
            "doc_len": self.doc_len,
            "postings": {t: p for t, p in self.postings.items()},
            "by_entity": {k: dict(v) for k, v in self.by_entity.items()},
            "by_section": dict(self.by_section),
            "by_block_type": dict(self.by_block_type),
            "by_level": dict(self.by_level),
        }


def _assign_keywords(chunks: List[dict], k: int) -> None:
    """TF-IDF keywords per chunk (IDF computed over the chunks passed in)."""
    N = len(chunks)
    surfaces: List[Dict[str, Counter]] = []
    df: Counter = Counter()
    for c in chunks:
        surface: Dict[str, Counter] = defaultdict(Counter)
        for raw in re.findall(r"[a-z][a-z0-9_]+", c["index_text"].lower()):
            if raw in STOPWORDS:
                continue
            surface[stem(raw)][raw] += 1
        surfaces.append(surface)
        df.update(surface.keys())
    for c, surface in zip(chunks, surfaces):
        scored = []
        for st, forms in surface.items():
            tf = sum(forms.values())
            scored.append(((1 + math.log(tf)) * math.log(1 + N / df[st]), st))
        scored.sort(key=lambda x: (-x[0], x[1]))
        c["keywords"] = [surface[st].most_common(1)[0][0] for _, st in scored[:k]]


# =====================================================================
# 7. CHUNKINGSYSTEM
# =====================================================================
class ChunkingSystem:
    """Validated, metadata-rich chunking for the digital twin.

    Parameters
    ----------
    strategy            one of STRATEGIES (default "structured")
    source              default source label for chunks
    sentences_per_chunk "sentence" strategy: sentences per chunk
    words_per_chunk     "word" strategy: window size in words
    overlap             "word" strategy: overlapping words (must be < words_per_chunk)
    max_chunk_size      "smart" strategy: word limit per chunk (defaults to max_words)
    min_words           chunks with fewer new words are flagged "too small" (or noted if unavoidable)
    max_words           target size for "structured"; hard ceiling for the other strategies
    log_max_words       size target for log files in "structured"
    overlap_sentences   0 or 1: repeat the previous chunk's last sentence ("smart"/"structured")
    split_level         "structured": heading depth that is a hard boundary (2 = H1/H2)
    keywords_per_chunk  TF-IDF keywords stored on each chunk
    normalize           apply Unicode / whitespace normalization before chunking
    drop_duplicates     drop chunks whose normalized text duplicates an earlier chunk in the same document
    entity_extractor    optional callable(text, only: Optional[set]) -> {type: [values]}
    history_size        number of process() results kept in processing_history
    """

    STRATEGIES = ("sentence", "word", "paragraph", "smart", "structured")

    def __init__(
        self,
        strategy: str = "structured",
        source: str = "unknown",
        sentences_per_chunk: int = 3,
        words_per_chunk: int = 50,
        overlap: int = 10,
        max_chunk_size: Optional[int] = None,
        min_words: int = 15,
        max_words: int = 120,
        log_max_words: int = 150,
        overlap_sentences: int = 1,
        split_level: int = 2,
        keywords_per_chunk: int = 6,
        normalize: bool = True,
        drop_duplicates: bool = False,
        entity_extractor: Optional[EntityExtractor] = None,
        history_size: int = 50,
    ):
        if strategy not in self.STRATEGIES:
            raise ValueError(
                f"Unknown strategy '{strategy}'. Choose from {self.STRATEGIES}"
            )
        for name, val, low in (
            ("sentences_per_chunk", sentences_per_chunk, 1),
            ("words_per_chunk", words_per_chunk, 1),
            ("min_words", min_words, 0),
            ("max_words", max_words, 1),
            ("log_max_words", log_max_words, 1),
            ("split_level", split_level, 1),
            ("keywords_per_chunk", keywords_per_chunk, 0),
            ("overlap", overlap, 0),
            ("overlap_sentences", overlap_sentences, 0),
            ("history_size", history_size, 1),
        ):
            if not isinstance(val, int) or isinstance(val, bool) or val < low:
                raise ValueError(f"{name} must be an integer >= {low}")
        if overlap >= words_per_chunk:
            raise ValueError("overlap must be smaller than words_per_chunk")
        if min_words > max_words:
            raise ValueError("min_words must not exceed max_words")
        if max_chunk_size is not None and (
            not isinstance(max_chunk_size, int) or max_chunk_size < 1
        ):
            raise ValueError("max_chunk_size must be a positive integer")

        self.strategy = strategy
        self.source = source
        self.sentences_per_chunk = sentences_per_chunk
        self.words_per_chunk = words_per_chunk
        self.overlap = overlap
        self.max_chunk_size = max_chunk_size or max_words
        self.min_words = min_words
        self.max_words = max_words
        self.log_max_words = log_max_words
        self.overlap_sentences = overlap_sentences
        self.split_level = split_level
        self.keywords_per_chunk = keywords_per_chunk
        self.normalize = normalize
        self.drop_duplicates = drop_duplicates
        self.entity_extractor: EntityExtractor = entity_extractor or extract_entities

        self._packer = _Packer(
            max_words, min_words, log_max_words, overlap_sentences, split_level
        )

        # corpus state (used by add_document / run / search / quality_report)
        self.docs: List[dict] = []
        self.chunks: List[dict] = []
        self.index: Optional[SearchIndex] = None
        self._coverage: Dict[str, set] = {}
        self.processing_history: Deque[dict] = deque(maxlen=history_size)

    # ------------------------------------------------------------------
    # Splitting helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _paragraphs(text: str) -> List[str]:
        return [p.strip() for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]

    @staticmethod
    def _sentences(text: str) -> List[str]:
        return split_sentences(text)

    @staticmethod
    def _simple_draft(
        parts: List[str],
        overlap_text: str = "",
        overlap_words: Optional[int] = None,
        split: bool = False,
        closed_by: str = "strategy",
    ) -> Draft:
        pieces = [
            Block("paragraph", p, (), {"check_boundaries": True, "split": split})
            for p in parts
        ]
        ow = wc(overlap_text) if overlap_words is None else overlap_words
        words = (
            wc(overlap_text) + sum(wc(p) for p in parts)
            if overlap_text
            else sum(wc(p) for p in parts)
        )
        return Draft(
            key=(),
            pieces=pieces,
            words=words,
            overlap_text=overlap_text,
            overlap_words=ow,
            closed_by=closed_by,
        )

    # ------------------------------------------------------------------
    # Strategies (each returns drafts)
    # ------------------------------------------------------------------
    def _by_sentence(self, text: str) -> List[Draft]:
        sents = self._sentences(text)
        n = self.sentences_per_chunk
        return [
            self._simple_draft([" ".join(sents[i : i + n])])
            for i in range(0, len(sents), n)
        ]

    def _by_word(self, text: str) -> List[Draft]:
        words = text.split()
        step = self.words_per_chunk - self.overlap
        drafts: List[Draft] = []
        for i in range(0, len(words), step):
            window = words[i : i + self.words_per_chunk]
            ov = min(self.overlap, len(window)) if i > 0 else 0
            drafts.append(
                self._simple_draft([" ".join(window)], overlap_words=ov, split=True)
            )
            if i + self.words_per_chunk >= len(words):
                break
        return drafts

    def _by_paragraph(self, text: str) -> List[Draft]:
        return [
            self._simple_draft([" ".join(p.split())]) for p in self._paragraphs(text)
        ]

    def _by_smart(self, text: str) -> List[Draft]:
        """Pack whole paragraphs up to max_chunk_size words; overlap = last sentence of previous chunk."""
        limit = self.max_chunk_size
        drafts: List[Draft] = []
        cur: List[str] = []
        cur_w, seed = 0, ""

        def close():
            nonlocal cur, cur_w, seed
            drafts.append(self._simple_draft(cur, overlap_text=seed, closed_by="size"))
            body = " ".join(cur)
            seed = (
                self._packer._last_sentence(body, limit)
                if self.overlap_sentences
                else ""
            )
            cur, cur_w = [], 0

        for para in (" ".join(p.split()) for p in self._paragraphs(text)):
            pw = wc(para)
            if cur and wc(seed) + cur_w + pw > limit:
                close()
            cur.append(para)
            cur_w += pw
        if cur:
            close()
            drafts[-1].closed_by = "end"
        return drafts

    # ------------------------------------------------------------------
    # Chunk construction and validation
    # ------------------------------------------------------------------
    def _extract(self, text: str, only: Optional[set] = None) -> Dict[str, List[str]]:
        try:
            return self.entity_extractor(text, only)
        except Exception:  # a bad custom extractor must not kill chunking
            logger.exception(
                "entity_extractor failed; falling back to built-in extractor"
            )
            return extract_entities(text, only)

    def _validate(
        self, text: str, d: Draft, ceiling: float
    ) -> Tuple[List[str], List[str]]:
        if not text.strip():
            return ["empty"], []
        issues: List[str] = []
        notes: List[str] = []
        n, own = wc(text), d.words - d.overlap_words
        single_atomic = len(d.pieces) == 1 and d.pieces[0].type in ATOMIC

        if n > ceiling:
            if single_atomic:
                notes.append(f"oversize atomic block ({n} words)")
            else:
                issues.append(f"too large ({n} words)")
        if own < self.min_words:
            if single_atomic or not d.same_key_neighbor:
                notes.append(f"small ({own} words)")
            else:
                issues.append(f"too small ({own} words)")

        first, last = d.pieces[0], d.pieces[-1]
        first_checked = first.meta.get("split") or first.meta.get("check_boundaries")
        last_checked = last.meta.get("split") or last.meta.get("check_boundaries")
        if (
            not d.overlap_text
            and first.type in PROSE
            and first_checked
            and first.text[:1].islower()
        ):
            issues.append("starts mid-sentence")
        if (
            last.type in PROSE
            and last_checked
            and last.text.rstrip()[-1:] not in SENTENCE_END_CHARS
        ):
            issues.append("ends mid-sentence")
        if len(re.findall(r"^\s*(?:`{3,}|~{3,})", text, re.M)) % 2:
            issues.append("unbalanced code fence")
        return issues, notes

    def _make_chunk(
        self,
        d: Draft,
        doc_id: str,
        source: str,
        fmt: str,
        title: str,
        ceiling: float,
        sep: str,
    ) -> dict:
        body = sep.join(p.text for p in d.pieces)
        text = f"{d.overlap_text}\n\n{body}" if d.overlap_text else body
        itext = sep.join(p.itext for p in d.pieces)
        if d.overlap_text:
            itext = d.overlap_text + "\n\n" + itext

        ents = []
        for p in d.pieces:
            only = CODE_ENTITY_TYPES if p.type == "code" else None
            ents.append(self._extract(p.meta.get("entity_text", p.text), only))
        if d.overlap_text:
            ents.append(self._extract(d.overlap_text))
        entities = merge_entities(ents)

        sections = list(
            dict.fromkeys(" > ".join(p.section) for p in d.pieces if p.section)
        )
        dominant = max(d.pieces, key=lambda p: wc(p.text))
        context = " > ".join(dominant.section) if dominant.section else title
        types = Counter(p.type for p in d.pieces)
        langs = sorted({p.meta["lang"] for p in d.pieces if p.type == "code"})
        log_pieces = [p for p in d.pieces if p.type == "log_entry"]
        issues, notes = self._validate(text, d, ceiling)

        return {
            "chunk_id": "",
            "doc_id": doc_id,
            "position": 0,
            "total_chunks": 0,
            "prev_id": None,
            "next_id": None,
            "source": source,
            "format": fmt,
            "strategy": self.strategy,
            "doc_title": title,
            "section": context,
            "sections_covered": sections,
            "context": context,
            "text": text,
            "index_text": itext,
            "word_count": wc(text),
            "char_count": len(text),
            "token_estimate": math.ceil(len(text) / 4),
            "overlap_words": d.overlap_words,
            "closed_by": d.closed_by,
            "block_types": dict(types),
            "has_table": "table" in types,
            "has_code": "code" in types,
            "code_languages": langs,
            "n_table_rows": sum(p.meta.get("n_rows", 0) for p in d.pieces),
            "split_atomic": any(
                p.meta.get("split") and p.type in ("table", "code") for p in d.pieces
            ),
            "entities": entities,
            "keywords": [],
            "time_start": min((p.meta["ts"] for p in log_pieces), default=None),
            "time_end": max((p.meta["ts"] for p in log_pieces), default=None),
            "log_levels": dict(Counter(p.meta["level"] for p in log_pieces)),
            "log_components": sorted(
                {p.meta["component"] for p in log_pieces if p.meta["component"]}
            ),
            "has_stacktrace": any(p.meta.get("has_trace") for p in log_pieces),
            "content_hash": hashlib.md5(
                re.sub(r"\s+", " ", text.lower()).encode()
            ).hexdigest()[:12],
            "issues": issues,
            "notes": notes,
            "valid": not issues,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }

    def _finalize(self, chunks: List[dict], doc_id: str) -> List[dict]:
        if self.drop_duplicates:
            seen, unique = set(), []
            for c in chunks:
                if c["content_hash"] in seen:
                    continue
                seen.add(c["content_hash"])
                unique.append(c)
            chunks = unique
        total = len(chunks)
        for k, c in enumerate(chunks):
            c["position"] = k + 1
            c["total_chunks"] = total
            c["chunk_id"] = f"{doc_id}-{k + 1:04d}"
            c["prev_id"] = f"{doc_id}-{k:04d}" if k else None
            c["next_id"] = f"{doc_id}-{k + 2:04d}" if k + 1 < total else None
        return chunks

    # ------------------------------------------------------------------
    # Core: one document -> (doc record, chunks, coverage tokens)
    # ------------------------------------------------------------------
    def _process_document(
        self, source: str, text: str, fmt: Optional[str]
    ) -> Tuple[dict, List[dict], set]:
        norm = normalize_text(text) if self.normalize else text.strip("\n")
        fmt = fmt or detect_format(norm, source)
        doc_id = hashlib.sha1((source + norm).encode("utf-8")).hexdigest()[:8]

        front: Dict[str, str] = {}
        headings: List[Block] = []
        stats = {"atomic_total": 0, "atomic_split": 0}
        block_counts: Dict[str, int] = {}

        if self.strategy == "structured":
            front, body = parse_front_matter(norm) if fmt == "markdown" else ({}, norm)
            blocks = parse_blocks(body, fmt)
            drafts, stats, limit = self._packer.build(blocks, fmt)
            ceiling = limit * 1.5
            sep = "\n" if fmt == "log" else "\n\n"
            headings = [b for b in blocks if b.type == "heading"]
            block_counts = dict(Counter(b.type for b in blocks))
            coverage = {
                t
                for b in blocks
                if b.type != "heading"
                for t in tokenize(b.text, keep_stop=True)
            }
        else:
            dispatch = {
                "sentence": self._by_sentence,
                "word": self._by_word,
                "paragraph": self._by_paragraph,
                "smart": self._by_smart,
            }
            drafts = [
                d
                for d in dispatch[self.strategy](norm)
                if any(p.text.strip() for p in d.pieces)
            ]
            _mark_neighbors(drafts)
            ceiling = self.max_words
            sep = "\n\n"
            coverage = set(tokenize(norm, keep_stop=True))

        if fmt == "log" and self.strategy == "structured":
            title = source
        else:
            first_line = norm.split("\n", 1)[0][:60] if norm else ""
            title = (
                front.get("title")
                or next((h.text for h in headings if h.meta["level"] == 1), "")
                or first_line
                or source
            )

        chunks = [
            self._make_chunk(d, doc_id, source, fmt, title, ceiling, sep)
            for d in drafts
        ]
        chunks = self._finalize(chunks, doc_id)
        _assign_keywords(chunks, self.keywords_per_chunk)

        doc = {
            "doc_id": doc_id,
            "source": source,
            "format": fmt,
            "title": title,
            "front_matter": front,
            "n_chars": len(norm),
            "n_words": wc(norm),
            "n_chunks": len(chunks),
            "block_counts": block_counts,
            "outline": [(h.meta["level"], h.text) for h in headings],
            "entities": merge_entities([c["entities"] for c in chunks]),
            "atomic_total": stats["atomic_total"],
            "atomic_split": stats["atomic_split"],
        }
        return doc, chunks, coverage

    # ------------------------------------------------------------------
    # Public API: stateless chunking
    # ------------------------------------------------------------------
    def chunk(
        self, text: str, source: Optional[str] = None, fmt: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Chunk one document and return the chunk dicts. Does not touch corpus state."""
        _check_text(text)
        if not text.strip():
            return []
        _, chunks, _ = self._process_document(source or self.source, text, fmt)
        return chunks

    # ------------------------------------------------------------------
    # Public API: corpus (stateful) operations
    # ------------------------------------------------------------------
    def add_document(self, source: str, text: str, fmt: Optional[str] = None) -> dict:
        """Chunk a document and add it to the corpus. Returns the document record."""
        _check_text(text)
        if not text.strip():
            logger.warning("add_document: '%s' is empty, skipped", source)
            return {"doc_id": None, "source": source, "n_chunks": 0, "skipped": "empty"}
        doc, chunks, coverage = self._process_document(source, text, fmt)
        if any(d["doc_id"] == doc["doc_id"] for d in self.docs):
            logger.info(
                "add_document: '%s' already ingested (same content), skipped", source
            )
            return {**doc, "skipped": "duplicate"}
        self.docs.append(doc)
        self.chunks.extend(chunks)
        self._coverage[doc["doc_id"]] = coverage
        self.index = None
        return doc

    def ingest_file(self, path: str, fmt: Optional[str] = None) -> dict:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return self.add_document(str(path), f.read(), fmt)

    def ingest_directory(self, directory: str, recursive: bool = True) -> List[dict]:
        root = Path(directory)
        if not root.is_dir():
            raise NotADirectoryError(directory)
        pattern = root.rglob("*") if recursive else root.glob("*")
        files = sorted(
            p for p in pattern if p.is_file() and p.suffix.lower() in TEXT_EXTENSIONS
        )
        return [self.ingest_file(str(p)) for p in files]

    def run(self, documents: Dict[str, str]) -> "ChunkingSystem":
        for name, text in documents.items():
            self.add_document(name, text)
        self.build_index()
        return self

    def reset(self) -> None:
        self.docs, self.chunks, self.index = [], [], None
        self._coverage = {}

    # ------------------------------------------------------------------
    # Indexing and search
    # ------------------------------------------------------------------
    def build_index(self) -> SearchIndex:
        _assign_keywords(self.chunks, self.keywords_per_chunk)  # corpus-wide IDF
        self.index = SearchIndex().build(self.chunks)
        return self.index

    def search(
        self, query: str, top_k: int = 5, filters: Optional[dict] = None
    ) -> List[dict]:
        if self.index is None:
            self.build_index()
        return self.index.search(query, top_k, filters)

    def find_entity(self, etype: str, value: str) -> List[str]:
        if self.index is None:
            self.build_index()
        return self.index.find_entity(etype, value)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    def to_jsonl(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for c in self.chunks:
                f.write(json.dumps(c, ensure_ascii=False, default=str) + "\n")

    def save_index(self, path: str) -> None:
        if self.index is None:
            self.build_index()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.index.to_dict(), f)

    # ------------------------------------------------------------------
    # Text-processing pipeline (clean -> extract -> chunk)
    # ------------------------------------------------------------------
    @staticmethod
    def _get_stats(text: str) -> Dict[str, int]:
        return {"chars": len(text), "words": wc(text), "lines": len(text.split("\n"))}

    def process(
        self,
        text: str,
        source: Optional[str] = None,
        clean: bool = True,
        extract_data: bool = True,
        chunk: bool = True,
        profile: bool = False,
    ) -> Dict[str, Any]:
        """Run the full pipeline on one text and return everything in one dict.

        clean         structure-safe normalization (does not collapse newlines, so tables/code survive)
        extract_data  document-level entities (emails, urls, dates, ...)
        chunk         chunk with the configured strategy
        profile       also run digital_twin.extraction.extract_profile_data if that module is importable
        """
        _check_text(text)
        result: Dict[str, Any] = {
            "original": text,
            "original_stats": self._get_stats(text),
            "processed": text,
            "extracted_data": {},
            "profile": None,
            "chunks": [],
            "pipeline_steps": [],
        }
        if clean:
            result["processed"] = normalize_text(text)
            result["pipeline_steps"].append("cleaned")
        if extract_data:
            result["extracted_data"] = self._extract(result["processed"])
            result["pipeline_steps"].append("extracted")
        if profile:
            try:
                from digital_twin.extraction import extract_profile_data

                result["profile"] = extract_profile_data(result["processed"])
                result["pipeline_steps"].append("profiled")
            except ImportError:
                logger.warning(
                    "process(profile=True): digital_twin.extraction not importable; skipped"
                )
        if chunk:
            result["chunks"] = self.chunk(result["processed"], source=source)
            result["pipeline_steps"].append("chunked")
        result["processed_stats"] = self._get_stats(result["processed"])
        self.processing_history.append(result)
        return result

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------
    def report(self, chunks: List[Dict[str, Any]]) -> None:
        """Short console summary for one chunk list."""
        print(f"Strategy: {self.strategy} | chunks: {len(chunks)}")
        if not chunks:
            return
        wcs = [c["word_count"] for c in chunks]
        bad = [c for c in chunks if not c["valid"]]
        print(f"  words min/avg/max: {min(wcs)}/{sum(wcs) / len(wcs):.1f}/{max(wcs)}")
        print(f"  valid chunks: {len(chunks) - len(bad)}/{len(chunks)}")
        for c in bad:
            print(f"  [!] chunk {c['position']}: {'; '.join(c['issues'])}")
        print()

    def quality_report(self) -> dict:
        """Corpus-level quality metrics and a 0-100 score."""
        if not self.chunks:
            return {"error": "no chunks"}
        if self.index is None:
            self.build_index()
        chunks, N = self.chunks, len(self.chunks)
        words = [c["word_count"] for c in chunks]
        valid_ratio = sum(c["valid"] for c in chunks) / N
        issue_counts = Counter(i.split(" (")[0] for c in chunks for i in c["issues"])
        note_counts = Counter(i.split(" (")[0] for c in chunks for i in c["notes"])

        src = {(d, t) for d, toks in self._coverage.items() for t in toks}
        got = {
            (c["doc_id"], t)
            for c in chunks
            for t in tokenize(c["text"], keep_stop=True)
        }
        coverage = len(src & got) / len(src) if src else 1.0

        atomic_total = sum(d["atomic_total"] for d in self.docs)
        atomic_split = sum(d["atomic_split"] for d in self.docs)
        integrity = 1 - atomic_split / atomic_total if atomic_total else 1.0

        dup_ratio = 1 - len({c["content_hash"] for c in chunks}) / N
        required = [
            "chunk_id",
            "doc_id",
            "source",
            "format",
            "section",
            "text",
            "word_count",
            "keywords",
            "content_hash",
        ]
        completeness = (
            sum(sum(bool(c[r]) for r in required) / len(required) for c in chunks) / N
        )

        ent_totals: Counter = Counter()
        for c in chunks:
            for k, v in c["entities"].items():
                ent_totals[k] += len(v)

        sample = chunks[:: max(1, N // 150)]
        hits = tried = 0
        for c in sample:
            if not c["keywords"]:
                continue
            tried += 1
            res = self.index.search(" ".join(c["keywords"][:4]), top_k=3)
            hits += any(r["chunk_id"] == c["chunk_id"] for r in res)
        self_retrieval = hits / tried if tried else 0.0

        score = 100 * (
            0.30 * valid_ratio
            + 0.20 * coverage
            + 0.15 * integrity
            + 0.20 * self_retrieval
            + 0.10 * completeness
            + 0.05 * (1 - dup_ratio)
        )
        grade = (
            "A"
            if score >= 90
            else (
                "B"
                if score >= 80
                else "C" if score >= 70 else "D" if score >= 60 else "F"
            )
        )
        return {
            "documents": len(self.docs),
            "chunks": N,
            "strategy": self.strategy,
            "by_format": dict(Counter(c["format"] for c in chunks)),
            "chunk_words": {
                "min": min(words),
                "avg": round(sum(words) / N, 1),
                "median": statistics.median(words),
                "max": max(words),
                "stdev": round(statistics.pstdev(words), 1),
            },
            "valid_ratio": round(valid_ratio, 3),
            "issues": dict(issue_counts),
            "notes": dict(note_counts),
            "token_coverage": round(coverage, 4),
            "overlap_share": round(
                sum(c["overlap_words"] for c in chunks) / max(1, sum(words)), 3
            ),
            "atomic_blocks": {
                "total": atomic_total,
                "split": atomic_split,
                "integrity": round(integrity, 3),
            },
            "duplicate_ratio": round(dup_ratio, 3),
            "metadata_completeness": round(completeness, 3),
            "chunks_with_entities": round(
                sum(bool(c["entities"]) for c in chunks) / N, 3
            ),
            "entity_totals": dict(ent_totals),
            "chunks_with_tables": sum(c["has_table"] for c in chunks),
            "chunks_with_code": sum(c["has_code"] for c in chunks),
            "self_retrieval_at_3": round(self_retrieval, 3),
            "index": self.index.stats(),
            "overall_score": round(score, 1),
            "grade": grade,
        }

    @staticmethod
    def print_quality_report(r: dict) -> None:
        if "error" in r:
            print(f"QUALITY REPORT: {r['error']}")
            return
        line = "=" * 64
        print(line + "\nQUALITY REPORT\n" + line)
        print(
            f"Strategy: {r['strategy']}   Documents: {r['documents']}   Chunks: {r['chunks']}   "
            f"Formats: {r['by_format']}"
        )
        print(f"Chunk words: {r['chunk_words']}")
        print(f"Valid chunks: {r['valid_ratio']:.1%}   Issues: {r['issues'] or 'none'}")
        print(f"Notes (non-fatal): {r['notes'] or 'none'}")
        print(
            f"Token coverage: {r['token_coverage']:.2%}   Overlap share: {r['overlap_share']:.1%}"
        )
        a = r["atomic_blocks"]
        print(
            f"Atomic blocks (tables+code): {a['total']}, split: {a['split']}, intact: {a['integrity']:.1%}"
        )
        print(
            f"Duplicate chunks: {r['duplicate_ratio']:.1%}   Metadata completeness: {r['metadata_completeness']:.1%}"
        )
        print(
            f"Chunks with entities: {r['chunks_with_entities']:.1%}   Entities: {r['entity_totals']}"
        )
        print(
            f"Chunks with tables: {r['chunks_with_tables']}   with code: {r['chunks_with_code']}"
        )
        print(f"Self-retrieval@3: {r['self_retrieval_at_3']:.1%}   Index: {r['index']}")
        print(
            f"\nOVERALL SCORE: {r['overall_score']}/100   GRADE: {r['grade']}\n" + line
        )


# =====================================================================
# 8. CONVENIENCE AND VISUALIZATION
# =====================================================================
def chunk_text(
    text: str, source: str = "unknown", **kwargs: Any
) -> List[Dict[str, Any]]:
    """One-liner: chunk_text(text, strategy="smart", max_words=80)."""
    return ChunkingSystem(source=source, **kwargs).chunk(text)


def visualize_chunks(
    results_by_strategy: Dict[str, List[Dict[str, Any]]],
    min_words: int,
    max_words: int,
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """Bar chart of chunk sizes per strategy (green = valid, red = has issues)."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "visualize_chunks requires matplotlib (pip install matplotlib)"
        ) from exc

    items = [(s, c) for s, c in results_by_strategy.items() if c]
    if not items:
        return
    cols = 2 if len(items) > 1 else 1
    rows = math.ceil(len(items) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows), squeeze=False)
    flat = list(axes.flatten())
    for ax, (strategy, chunks) in zip(flat, items):
        positions = [c["position"] for c in chunks]
        counts = [c["word_count"] for c in chunks]
        colors = ["seagreen" if c["valid"] else "indianred" for c in chunks]
        ax.bar(positions, counts, color=colors)
        ax.axhline(min_words, color="gray", linestyle="--", linewidth=1, label="min")
        ax.axhline(max_words, color="black", linestyle="--", linewidth=1, label="max")
        ax.set_title(f"{strategy} ({len(chunks)} chunks)")
        ax.set_xlabel("Chunk position")
        ax.set_ylabel("Word count")
        if len(positions) <= 40:
            ax.set_xticks(positions)
        ax.legend()
    for ax in flat[len(items) :]:
        ax.axis("off")
    fig.suptitle("Chunk size distribution (green = valid, red = has issues)")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


# =====================================================================
# 9. SMOKE DEMO  (python -m digital_twin.chunking)
# =====================================================================
_DEMO_TEXT = """# Digital Twin Notes

## Overview

A digital twin is a virtual representation of a physical object or system. Digital twins are used across
various industries. In healthcare, they model patient health. In manufacturing, they simulate production.

The technology relies on real-time data from sensors. This data feeds into simulation models. Machine
learning algorithms analyze patterns. Predictions help optimize operations.

## Benefits

| Industry | Benefit |
|----------|---------|
| Healthcare | Personalized treatment |
| Manufacturing | Predictive maintenance |
| Cities | Resource allocation |

Contact davies@example.com or visit https://example.com/twins for details. #digitaltwin
"""


def _demo() -> None:  # pragma: no cover
    results = {}
    for strategy in ChunkingSystem.STRATEGIES:
        cs = ChunkingSystem(
            strategy=strategy,
            words_per_chunk=30,
            overlap=5,
            min_words=8,
            max_words=60,
            max_chunk_size=50,
        )
        results[strategy] = cs.chunk(_DEMO_TEXT, source="demo.md")
        cs.report(results[strategy])
    cs = ChunkingSystem(strategy="structured", min_words=8, max_words=60)
    cs.run({"demo.md": _DEMO_TEXT})
    cs.print_quality_report(cs.quality_report())


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    _demo()
