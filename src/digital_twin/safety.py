from __future__ import annotations

import hashlib
import itertools
import json
import logging
import os
import re
import secrets
import threading
import time
import unicodedata
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Deque, Dict, Iterable, Iterator, List, Optional, Tuple

safety_logger = logging.getLogger("digital_twin.safety")


# ===========================================================================
# CONFIG
# ===========================================================================
@dataclass(frozen=True)
class SafetyConfig:
    max_input_chars: int = 2000
    max_output_chars: int = 3000  # OutputValidator.max_length
    rate_limit_calls: int = 10
    rate_limit_window_seconds: float = 60.0
    injection_threshold: float = 0.7  # score >= this blocks the request
    injection_flag_threshold: float = (
        0.4  # score >= this (but below block) is logged only
    )
    profanity_input_action: str = "flag"  # "flag" (log, allow) or "block"
    domain_input_action: str = "flag"  # salary/password/... topics: "flag" or "block"
    redact_output_pii: bool = True
    leak_min_matches: int = 2
    user_name: str = "Davies"  # used in the fallback JSON
    violation_log_path: Optional[str] = None  # JSONL file; None = memory + logger only
    max_logged_violations: int = 1000

    def __post_init__(self):
        if self.max_input_chars <= 0 or self.max_output_chars <= 0:
            raise ValueError("max_input_chars and max_output_chars must be positive.")
        if self.rate_limit_calls <= 0 or self.rate_limit_window_seconds <= 0:
            raise ValueError(
                "rate_limit_calls and rate_limit_window_seconds must be positive."
            )
        if not 0 < self.injection_threshold <= 1:
            raise ValueError("injection_threshold must be in (0, 1].")
        if not 0 <= self.injection_flag_threshold <= self.injection_threshold:
            raise ValueError(
                "injection_flag_threshold must be between 0 and injection_threshold."
            )
        for name in ("profanity_input_action", "domain_input_action"):
            if getattr(self, name) not in ("flag", "block"):
                raise ValueError(f"{name} must be 'flag' or 'block'.")
        if self.leak_min_matches <= 0:
            raise ValueError("leak_min_matches must be positive.")
        if self.max_logged_violations <= 0:
            raise ValueError("max_logged_violations must be positive.")


# ===========================================================================
# TEXT UTILITIES
# ===========================================================================
# Look-alike letters (Cyrillic / Greek) commonly used to dodge keyword filters.
# Applied only to a COPY used for detection - the text sent to the model is untouched.
_HOMOGLYPHS = str.maketrans(
    {
        "а": "a",
        "е": "e",
        "о": "o",
        "р": "p",
        "с": "c",
        "х": "x",
        "у": "y",
        "і": "i",
        "ј": "j",
        "ѕ": "s",
        "ԁ": "d",
        "ɡ": "g",
        "ο": "o",
        "α": "a",
        "ε": "e",
        "ν": "v",
        "ρ": "p",
        "τ": "t",
        "ι": "i",
        "κ": "k",
    }
)
_LEET_BASE = str.maketrans(
    {"0": "o", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s"}
)
# '1' is ambiguous (i or l), so it is resolved per token against this vocabulary.
_LEET_KNOWN_WORDS = frozenset("""
ignore all previous prior instructions instruction rules guidelines your system prompt reveal
disregard forget bypass pretend restrictions unrestricted override mode developer anything now
jailbreak display repeat show print secret directives safety training filters tell above
""".split())


def _strip_invisible(text: str) -> str:
    """Drop control / zero-width / format characters (keeps newline and tab as spaces)."""
    out = []
    for ch in text:
        if ch in "\n\t\r":
            out.append(" ")
        elif unicodedata.category(ch) in ("Cc", "Cf"):
            continue
        else:
            out.append(ch)
    return "".join(out)


def _basic_normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = _strip_invisible(text).lower()
    return re.sub(r"\s+", " ", text).strip()


def _deleet_token(token: str) -> str:
    base = token.translate(_LEET_BASE)
    if "1" not in base:
        return base
    count = base.count("1")
    if count <= 3:
        for combo in itertools.product("il", repeat=count):
            it = iter(combo)
            candidate = "".join(next(it) if c == "1" else c for c in base)
            if candidate in _LEET_KNOWN_WORDS:
                return candidate
    return base.replace("1", "i")


def _deleet(text: str) -> str:
    """Undo digit/symbol substitutions, only inside tokens that mix letters and digits,
    so plain numbers and dates ("2026-08-10") are never touched."""

    def fix(match: "re.Match[str]") -> str:
        tok = match.group(0)
        has_alpha = any(c.isalpha() for c in tok)
        has_leet = any(c in "013457@$" for c in tok)
        return _deleet_token(tok) if has_alpha and has_leet else tok

    return re.sub(r"[\w@$]+", fix, text)


def iter_strings(obj: Any) -> Iterator[str]:
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for value in obj.values():
            yield from iter_strings(value)
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            yield from iter_strings(value)


def map_strings(obj: Any, fn: Callable[[str], str]) -> Any:
    if isinstance(obj, str):
        return fn(obj)
    if isinstance(obj, dict):
        return {k: map_strings(v, fn) for k, v in obj.items()}
    if isinstance(obj, list):
        return [map_strings(v, fn) for v in obj]
    return obj


def _luhn_ok(number: str) -> bool:
    digits = [int(c) for c in re.sub(r"\D", "", number)]
    if not 13 <= len(digits) <= 19:
        return False
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def _valid_ipv4(value: str) -> bool:
    return all(0 <= int(part) <= 255 for part in value.split("."))


# ===========================================================================
# PII + SECRETS
# ===========================================================================
@dataclass
class PIIFindings:
    pii_types: List[str] = field(default_factory=list)
    secret_types: List[str] = field(default_factory=list)

    @property
    def has_pii(self) -> bool:
        return bool(self.pii_types)

    @property
    def has_secrets(self) -> bool:
        return bool(self.secret_types)

    @property
    def found(self) -> bool:
        return self.has_pii or self.has_secrets


class PIIDetector:
    """Detects and redacts personal data and credentials.

    Patterns are deliberately tight (checksums, case-sensitive street names,
    octet ranges) so normal diary text such as dates, durations, versions and
    "2 steps along the way" is not redacted.
    """

    # (type, pattern, validator, flags) - order matters for redaction.
    _SECRETS: List[Tuple[str, str, Optional[Callable[[str], bool]], int]] = [
        (
            "private_key",
            r"-----BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----",
            None,
            0,
        ),
        (
            "api_key",
            r"\b(?:gsk_[A-Za-z0-9]{20,}|sk-(?:proj-)?[A-Za-z0-9_\-]{20,}|AKIA[0-9A-Z]{16}"
            r"|gh[pousr]_[A-Za-z0-9]{36,}|xox[baprs]-[A-Za-z0-9\-]{10,})\b",
            None,
            0,
        ),
        ("bearer_token", r"\bBearer\s+[A-Za-z0-9._\-]{20,}", None, re.I),
        (
            "credential_assignment",
            r"\b(?:password|passwd|pwd|secret|api[_\- ]?key|access[_\- ]?token|token)\s*[:=]\s*\S{6,}",
            None,
            re.I,
        ),
    ]
    _PII: List[Tuple[str, str, Optional[Callable[[str], bool]], int]] = [
        ("email", r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", None, 0),
        (
            "credit_card",
            r"(?<![\d-])\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{1,7}(?![\d-])",
            _luhn_ok,
            0,
        ),
        ("ssn", r"(?<![\d-])\d{3}-\d{2}-\d{4}(?![\d-])", None, 0),
        (
            "phone",
            r"(?<![\w.+-])(?:\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}(?![\w-])",
            None,
            0,
        ),
        # local-format mobile numbers such as 0803 555 0142 (070/080/081/090/091 prefixes)
        (
            "phone_local",
            r"(?<![\w.+-])0[7-9][01]\d[-.\s]?\d{3}[-.\s]?\d{4}(?![\w-])",
            None,
            0,
        ),
        ("ip_address", r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])", _valid_ipv4, 0),
        (
            "date_of_birth",
            r"\b(?:DOB|D\.O\.B\.?|date of birth|born(?: on)?)[:\s]*"
            r"(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2})\b",
            None,
            re.I,
        ),
        # capitalised street names only, so "2 steps along the way" is not an address
        (
            "address",
            r"\b\d{1,5}\s+(?:[A-Z][A-Za-z0-9'.\-]*\s+){1,4}"
            r"(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct|Place|Pl"
            r"|Way|Close|Crescent|Terrace|Ter)\b\.?",
            None,
            0,
        ),
        ("passport", r"\b[A-Z]{1,2}\d{6,9}\b", None, 0),
    ]

    def __init__(self):
        self._rules = [
            (kind, True, re.compile(p, f), v) for kind, p, v, f in self._SECRETS
        ] + [(kind, False, re.compile(p, f), v) for kind, p, v, f in self._PII]

    def detect(self, text: str) -> PIIFindings:
        findings = PIIFindings()
        for kind, is_secret, rx, validator in self._rules:
            if any(
                validator is None or validator(m.group(0)) for m in rx.finditer(text)
            ):
                (findings.secret_types if is_secret else findings.pii_types).append(
                    kind
                )
        return findings

    def redact(self, text: str) -> Tuple[str, PIIFindings]:
        findings = PIIFindings()
        for kind, is_secret, rx, validator in self._rules:
            hit = False

            def repl(m: "re.Match[str]", kind=kind) -> str:
                nonlocal hit
                if validator is None or validator(m.group(0)):
                    hit = True
                    return f"[REDACTED_{kind.upper()}]"
                return m.group(0)

            text = rx.sub(repl, text)
            if hit:
                (findings.secret_types if is_secret else findings.pii_types).append(
                    kind
                )
        return text, findings


# ===========================================================================
# CONTENT FILTER
# ===========================================================================
@dataclass
class ContentResult:
    blocked: List[str] = field(default_factory=list)  # category names that block
    flagged: List[str] = field(default_factory=list)  # category names logged only


class ContentFilter:
    """Category-based content checks with word-boundary matching.

    Each category has an action per stage ("block" or "flag"). Defaults:
      profanity   - input flag (frustration is not an attack), output block
      self_harm   - block both ways (input gets a supportive message)
      harm        - block both ways
      ambiguous   - abuse / exploitation / illegal / drugs: flag only, because these
                    are everyday words in security and debugging work
      domain      - salary / password / api key / ...: flag only. The twin holds no such
                    data, and blocking would also block its honest "not in my data" answer.
    Only category names are ever returned or logged - never the matched words.
    """

    _PROFANITY = [
        "fuck",
        "fucking",
        "fucked",
        "motherfucker",
        "shit",
        "shitty",
        "bullshit",
        "ass",
        "asshole",
        "bitch",
        "bastard",
        "damn",
        "dammit",
        "goddamn",
        "crap",
        "piss",
        "pissed",
        "dick",
        "dickhead",
        "cock",
        "cunt",
        "slut",
        "whore",
        "douche",
        "douchebag",
        "prick",
        "twat",
        "wanker",
        "bollocks",
        "bugger",
        "bloody hell",
        "jackass",
        "dumbass",
        "retard",
        "retarded",
        "idiot",
        "moron",
        "scumbag",
    ]
    _SELF_HARM = [
        r"suicid(?:e|al)",
        r"self[\s-]?harm",
        r"kill\s+myself",
        r"end\s+my\s+life",
        r"hurt\s+myself",
        r"want\s+to\s+die",
    ]
    _HARM = [r"violen(?:ce|t)", r"weapons?", r"terroris(?:m|ts?)", r"mass\s+shootings?"]
    _AMBIGUOUS = [r"abus(?:e|ed|es|ing|ive)", r"exploitation", r"illegal", r"drugs?"]
    _DOMAIN = [
        "salary",
        "compensation",
        "performance review",
        "layoffs?",
        "termination",
        "trade secrets?",
        "proprietary",
        "confidential",
        "passwords?",
        "api keys?",
        "secret keys?",
        "credentials",
    ]

    def __init__(
        self, profanity_input_action: str = "flag", domain_input_action: str = "flag"
    ):
        def compile_words(words: Iterable[str]) -> "re.Pattern[str]":
            return re.compile(r"\b(?:" + "|".join(words) + r")\b", re.I)

        # name -> (regex, input action, output action)
        self._categories: Dict[str, Tuple["re.Pattern[str]", str, str]] = {
            "profanity": (
                compile_words(re.escape(w) for w in self._PROFANITY),
                profanity_input_action,
                "block",
            ),
            "self_harm": (compile_words(self._SELF_HARM), "block", "block"),
            "harm": (compile_words(self._HARM), "block", "block"),
            "ambiguous_topic": (compile_words(self._AMBIGUOUS), "flag", "flag"),
            "domain_topic": (compile_words(self._DOMAIN), domain_input_action, "flag"),
        }

    def check(self, text: str, stage: str) -> ContentResult:
        if stage not in ("input", "output"):
            raise ValueError("stage must be 'input' or 'output'.")
        result = ContentResult()
        for name, (rx, in_action, out_action) in self._categories.items():
            if rx.search(text):
                action = in_action if stage == "input" else out_action
                (result.blocked if action == "block" else result.flagged).append(name)
        return result


# ===========================================================================
# PROMPT INJECTION DETECTION
# ===========================================================================
@dataclass(frozen=True)
class Rule:
    id: str
    pattern: str
    weight: float
    flags: int = 0


@dataclass
class InjectionResult:
    score: float
    is_injection: bool
    matched: List[str]


# Each matching rule adds its weight; the sum (capped at 1.0) is compared with the
# threshold (default 0.7). Weights >= 0.7 block on their own: unmistakable commands
# aimed at the assistant. Lower weights are supporting evidence that only block when
# combined with another signal.
_ADJ = r"(?:(?:previous|prior|earlier|above|preceding|original|initial|system|existing|safety)\s+)"
_NOUN = r"(?:instructions?|rules|guidelines|directives|prompts?|programming|restrictions|safeguards)"
_VERB = r"(?:ignore|disregard|forget|override|bypass|discard)"
# "ignore the previous instructions in the README" is about a document, not about me.
_DOC_CONTEXT = (
    r"(?!\s+(?:in|from|on|for|of|about|within)\s+(?:the|this|that|my|our)?\s*"
    r"(?:readme|docs?|documentation|documents?|files?|tickets?|issues?|pr|pull\s+request|"
    r"emails?|threads?|specs?|code|repo|repository|config|comments?|messages?|notes?|page|wiki)\b)"
)
_REVEAL = r"(?:reveal|show|print|display|repeat|output|leak|expose|dump|disclose|share|recite|paste|tell\s+me|give\s+me|send\s+me)"

INJECTION_RULES: Tuple[Rule, ...] = (
    Rule(
        "override_previous",
        rf"\b{_VERB}\s+(?:all\s+|any\s+|every\s+|the\s+|these\s+|those\s+|your\s+|my\s+)?"
        rf"(?:of\s+(?:the\s+|your\s+)?)?{_ADJ}+{_NOUN}\b{_DOC_CONTEXT}",
        0.8,
    ),
    Rule(
        "override_your",
        rf"\b{_VERB}\s+your\s+(?:{_ADJ}*{_NOUN}|training|safety|ethics|limits|boundaries)\b",
        0.8,
    ),
    Rule(
        "override_generic",
        rf"\b{_VERB}\s+(?:all|any|every)\s+(?:of\s+)?(?:the\s+)?{_NOUN}\b",
        0.45,
    ),
    Rule(
        "forget_everything",
        r"\bforget\s+(?:everything|all\s+(?:of\s+)?that|what\s+you\s+(?:know|were\s+told))\b",
        0.45,
    ),
    Rule(
        "reveal_prompt",
        rf"\b{_REVEAL}\s+(?:me\s+)?(?:all\s+of\s+)?(?:your\s+(?:(?:full|entire|complete|hidden|secret|original|initial|exact|internal|system|developer)\s+)*"
        rf"(?:prompt|instructions|rules|guidelines|configuration|programming|message)"
        rf"|the\s+(?:(?:full|entire|complete|hidden|secret|original|initial|exact|internal|system|developer)\s+)+"
        rf"(?:prompt|instructions|rules|guidelines|message))\b",
        0.8,
    ),
    Rule(
        "extract_above",
        r"\b(?:repeat|print|output|copy|recite|reproduce)\s+(?:all\s+of\s+|everything\s+|"
        r"the\s+(?:text|words|content|message)\s+)(?:above|before\s+this|so\s+far)\b",
        0.7,
    ),
    Rule(
        "secret_request",
        rf"\b{_REVEAL}\s+(?:me\s+)?your\s+(?:(?:groq|openai|api|secret|private|access)\s+)*"
        r"(?:api\s+)?(?:keys?|tokens?|passwords?|credentials)\b",
        0.7,
    ),
    Rule(
        "probe_instructions",
        r"\bwhat\s+(?:are|were|is)\s+your\s+(?:(?:system|initial|original|hidden|secret)\s+)?"
        r"(?:instructions|rules|prompt|guidelines)\b|\bwhat\s+were\s+you\s+told\b",
        0.45,
    ),
    Rule(
        "role_reassign",
        r"\byou\s+are\s+(?:now|no\s+longer)\s+(?:a|an|the|my|in|dan|free|unrestricted)\b",
        0.7,
    ),
    Rule(
        "from_now_on",
        r"\bfrom\s+now\s+on,?\s+you(?:'ll|\s+will|\s+must|\s+are|\s+should)\b",
        0.45,
    ),
    Rule(
        "act_unrestricted",
        r"\bact\s+as\s+(?:if\s+you\s+(?:are|were|have)\s+)?(?:an?\s+)?"
        r"(?:unrestricted|uncensored|unfiltered|jailbroken|evil|rogue)\b",
        0.8,
    ),
    Rule(
        "pretend_no_rules",
        r"\bpretend\s+(?:that\s+)?you\s+(?:have\s+no|are\s+not|don'?t\s+have|aren'?t|are\s+free\s+from)\b",
        0.7,
    ),
    Rule(
        "persona_jailbreak",
        r"\bdo\s+anything\s+now\b|\bdan\s+mode\b|\byou\s+are\s+(?:now\s+)?dan\b"
        r"|\b(?:unrestricted|unfiltered|uncensored|jailbreak(?:ed)?)\s+mode\b"
        r"|\bdeveloper\s+mode\s+(?:enabled|activated|output)\b",
        0.8,
    ),
    Rule("jailbreak_word", r"\bjailbreak(?:s|ed|ing)?\b", 0.3),
    Rule(
        "no_restrictions",
        r"\b(?:no|without(?:\s+any)?)\s+(?:restrictions|limits|limitations|filters|censorship|guardrails|rules|boundaries)\b",
        0.35,
    ),
    Rule(
        "new_directive",
        r"\bnew\s+(?:system\s+)?(?:directives?|instructions?|orders?)\b",
        0.4,
    ),
    Rule(
        "chat_template_tokens",
        r"<\|[a-z_]{2,20}\|>|\[/?(?:inst|sys)\]|<</?sys>>|</?(?:system|assistant|instructions?)>",
        0.8,
    ),
    Rule(
        "fake_system_marker",
        r"\[\s*system(?:\s+(?:override|message|prompt|note))?\s*\]|(?:#{2,}|-{3,}|={3,})\s*system\b",
        0.8,
    ),
    Rule("system_override_phrase", r"\bsystem\s+override\b", 0.5),
    Rule(
        "end_of_input",
        r"\bend\s+of\s+(?:the\s+)?(?:user\s+)?(?:input|prompt|message|conversation)\b",
        0.4,
    ),
    Rule(
        "decode_and_follow",
        r"\b(?:decode|decrypt|deobfuscate)\b[^.\n]{0,60}\b(?:and|then)\s+(?:follow|execute|obey|run|apply|do)\b",
        0.7,
    ),
)

# Same ideas written without spaces/punctuation, to catch "i g n o r e  a l l ..." style spacing.
_SQUASHED_PHRASES: Dict[str, float] = {
    "ignoreallpreviousinstructions": 0.8,
    "ignorepreviousinstructions": 0.8,
    "ignoreyourinstructions": 0.8,
    "disregardyourinstructions": 0.8,
    "disregardallpreviousinstructions": 0.8,
    "forgetallpreviousinstructions": 0.8,
    "revealyoursystemprompt": 0.8,
    "showyoursystemprompt": 0.8,
    "printyoursystemprompt": 0.8,
    "tellmeyourinstructions": 0.8,
    "youarenowdan": 0.8,
    "doanythingnow": 0.8,
    "danmodeenabled": 0.8,
}


class PromptInjectionDetector:
    """Scores a message against INJECTION_RULES using light normalisation
    (Unicode NFKC, invisible characters removed, lower-cased, whitespace collapsed)."""

    def __init__(
        self, threshold: float = 0.7, rules: Tuple[Rule, ...] = INJECTION_RULES
    ):
        if not 0 < threshold <= 1:
            raise ValueError("threshold must be in (0, 1].")
        self.threshold = threshold
        self._rules = [(r, re.compile(r.pattern, r.flags | re.I)) for r in rules]

    def _variants(self, text: str) -> List[Tuple[str, bool]]:
        """(text variant, is_squashed) pairs to score; the best score wins."""
        return [(_basic_normalize(text), False)]

    def _score(self, variant: str, squashed: bool) -> Tuple[float, List[str]]:
        if squashed:
            hits = [
                (phrase, w)
                for phrase, w in _SQUASHED_PHRASES.items()
                if phrase in variant
            ]
            return (
                min(1.0, max((w for _, w in hits), default=0.0)),
                [f"squashed:{p}" for p, _ in hits],
            )
        total, matched = 0.0, []
        for rule, rx in self._rules:
            if rx.search(variant):
                total += rule.weight
                matched.append(rule.id)
        if (
            variant.count("---") > 3
            or variant.count("###") > 3
            or variant.count("===") > 3
        ):
            total += 0.3
            matched.append("delimiter_abuse")
        return min(1.0, total), matched

    def detect(self, text: str) -> InjectionResult:
        best_score, best_matched = 0.0, []
        for variant, squashed in self._variants(text):
            score, matched = self._score(variant, squashed)
            if score > best_score:
                best_score, best_matched = score, matched
        return InjectionResult(
            round(best_score, 3), best_score >= self.threshold, best_matched
        )


class HardenedInjectionDetector(PromptInjectionDetector):
    """Adds the obfuscation defences found during red-teaming on top of the base
    normalisation (which already strips zero-width and control characters): look-alike
    letters, leetspeak (both readings of '1'), and letter-spacing."""

    def _variants(self, text: str) -> List[Tuple[str, bool]]:
        base = _basic_normalize(text).translate(_HOMOGLYPHS)
        deleeted = _deleet(base)
        squashed = re.sub(r"[^a-z0-9]", "", deleeted)
        return [(base, False), (deleeted, False), (squashed, True)]


# ===========================================================================
# SECURITY DEFENSE: sanitising + prompt-leak detection
# ===========================================================================
@dataclass
class LeakResult:
    leaked: bool
    matches: int


class SecurityDefense:
    """sanitize_input() strips chat-template tokens and invisible characters.
    detect_leak() checks whether an output reproduces the twin's INSTRUCTION text.

    protected_text must be the prompt with the personal data left out (instructions
    only). Comparing against the full prompt would flag every honest answer, because
    answers legitimately quote the diary entries that live in the prompt.
    Detection uses distinctive runs of words from the instruction prose, not a
    shared-vocabulary count.
    """

    _TEMPLATE_TOKENS = re.compile(
        r"<\|[^|>\n]{1,30}\|>|\[/?(?:INST|SYS)\]|<</?SYS>>", re.I
    )

    def __init__(
        self, protected_text: str = "", min_matches: int = 2, shingle_size: int = 6
    ):
        if min_matches <= 0 or shingle_size <= 1:
            raise ValueError(
                "min_matches must be positive and shingle_size must be > 1."
            )
        self.min_matches = min_matches
        self.shingle_size = shingle_size
        self._shingles = self._build_shingles(protected_text)

    @staticmethod
    def _tokens(text: str) -> List[str]:
        return re.findall(r"[a-z0-9']+", _basic_normalize(text))

    def _build_shingles(self, protected_text: str) -> frozenset:
        shingles = set()
        for line in protected_text.splitlines():
            if any(ch in line for ch in "{}[]"):  # skip the JSON-schema block
                continue
            tokens = self._tokens(line)
            if len(tokens) < self.shingle_size + 1:
                continue
            for i in range(len(tokens) - self.shingle_size + 1):
                shingles.add(tuple(tokens[i : i + self.shingle_size]))
        return frozenset(shingles)

    def sanitize_input(self, text: str) -> str:
        text = (
            unicodedata.normalize("NFKC", text)
            .replace("\r\n", "\n")
            .replace("\r", "\n")
        )
        text = "".join(
            ch
            for ch in text
            if ch in "\n\t" or unicodedata.category(ch) not in ("Cc", "Cf")
        )
        text = self._TEMPLATE_TOKENS.sub(" ", text)
        text = re.sub(r"[ \t]+", " ", text)
        return re.sub(r"\n{3,}", "\n\n", text).strip()

    def detect_leak(self, text: str) -> LeakResult:
        if not self._shingles:
            return LeakResult(False, 0)
        tokens = self._tokens(text)
        n = self.shingle_size
        found = {
            tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)
        } & self._shingles
        return LeakResult(len(found) >= self.min_matches, len(found))


# ===========================================================================
# RATE LIMITER
# ===========================================================================
@dataclass
class RateStatus:
    allowed: bool
    remaining: int
    retry_after: float


class RateLimiter:
    """Thread-safe sliding-window limiter, one window per identifier. In-memory and
    per-process: several server instances would need a shared store (e.g. Redis)."""

    def __init__(
        self,
        max_calls: int = 10,
        window_seconds: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
    ):
        if max_calls <= 0:
            raise ValueError(f"max_calls must be positive, got {max_calls}.")
        if window_seconds <= 0:
            raise ValueError(f"window_seconds must be positive, got {window_seconds}.")
        self.max_calls, self.window_seconds, self._clock = (
            max_calls,
            window_seconds,
            clock,
        )
        self._calls: Dict[str, Deque[float]] = {}
        self._lock = threading.Lock()

    def _prune(self, q: Deque[float], now: float) -> None:
        while q and now - q[0] >= self.window_seconds:
            q.popleft()

    def acquire(self, identifier: str = "default") -> RateStatus:
        """Check-and-record in one atomic step. A rejected call is not recorded."""
        with self._lock:
            now = self._clock()
            q = self._calls.setdefault(identifier, deque())
            self._prune(q, now)
            if len(q) >= self.max_calls:
                return RateStatus(
                    False, 0, round(self.window_seconds - (now - q[0]), 1)
                )
            q.append(now)
            if len(self._calls) > 10_000:  # drop idle identifiers
                for key in [k for k, v in self._calls.items() if not v]:
                    del self._calls[key]
            return RateStatus(True, self.max_calls - len(q), 0.0)

    def check(self, identifier: str = "default") -> RateStatus:
        """Look without consuming a slot."""
        with self._lock:
            now = self._clock()
            q = self._calls.get(identifier, deque())
            self._prune(q, now)
            remaining = self.max_calls - len(q)
            retry = (
                round(self.window_seconds - (now - q[0]), 1)
                if q and remaining <= 0
                else 0.0
            )
            return RateStatus(remaining > 0, max(0, remaining), retry)

    def reset(self, identifier: Optional[str] = None) -> None:
        with self._lock:
            if identifier is None:
                self._calls.clear()
            else:
                self._calls.pop(identifier, None)


# ===========================================================================
# VIOLATION LOG (metadata only, never raw content)
# ===========================================================================
_SAFE_KEY = re.compile(r"^[a-z0-9_]{1,40}$")
_SAFE_STR = re.compile(
    r"^[A-Za-z0-9_:.\-\[\]]{1,60}$"
)  # no spaces, '@', quotes: cannot hold prose or an email


def _scrub(value: Any, depth: int = 0) -> Any:
    """Keeps only metadata-looking values (numbers, bools, identifier-like strings)."""
    if isinstance(value, (bool, int, float)) or value is None:
        return value
    if isinstance(value, str):
        return value if _SAFE_STR.match(value) else "<omitted>"
    if depth < 2 and isinstance(value, (list, tuple)):
        return [_scrub(v, depth + 1) for v in list(value)[:20]]
    if depth < 2 and isinstance(value, dict):
        return {
            k: _scrub(v, depth + 1)
            for k, v in list(value.items())[:20]
            if _SAFE_KEY.match(str(k))
        }
    return "<omitted>"


class ViolationLog:
    """Structured, thread-safe record of safety events.

    Stores: UTC time, stage, code, severity, a salted hash of the identifier, a
    salted hash and length of the offending text, and scrubbed metadata. It never
    stores the text itself, matched values, or diary content. The salt is random
    per process, so hashes correlate repeats within a run but cannot be looked up
    in a precomputed table.
    """

    def __init__(self, path: Optional[str] = None, max_entries: int = 1000):
        self.path = path
        self._entries: Deque[Dict[str, Any]] = deque(maxlen=max_entries)
        self._lock = threading.Lock()
        self._salt = secrets.token_bytes(16)
        if path:
            try:
                os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            except OSError as e:
                safety_logger.warning(
                    "could not create violation log folder: %s", type(e).__name__
                )

    def _digest(self, value: str) -> str:
        return hashlib.sha256(
            self._salt + value.encode("utf-8", "replace")
        ).hexdigest()[:12]

    def record(
        self,
        stage: str,
        code: str,
        severity: str,
        identifier: str = "default",
        text: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "stage": stage,
            "code": code,
            "severity": severity,
            "identifier": self._digest(str(identifier)),
            "text_hash": self._digest(text) if text is not None else None,
            "text_len": len(text) if text is not None else None,
            "details": _scrub(details or {}),
        }
        with self._lock:
            self._entries.append(entry)
            if self.path:
                try:
                    with open(self.path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(entry) + "\n")
                except OSError as e:
                    safety_logger.warning(
                        "could not write violation log file: %s", type(e).__name__
                    )
        safety_logger.warning(
            "safety event stage=%s code=%s severity=%s id=%s text_hash=%s len=%s",
            stage,
            code,
            severity,
            entry["identifier"],
            entry["text_hash"],
            entry["text_len"],
        )
        return entry

    def entries(self, identifier: Optional[str] = None) -> List[Dict[str, Any]]:
        with self._lock:
            rows = list(self._entries)
        if identifier is not None:
            wanted = self._digest(str(identifier))
            rows = [r for r in rows if r["identifier"] == wanted]
        return rows

    def report(self, identifier: Optional[str] = None) -> Dict[str, Any]:
        rows = self.entries(identifier)
        by_code: Dict[str, int] = {}
        by_severity: Dict[str, int] = {}
        for r in rows:
            by_code[r["code"]] = by_code.get(r["code"], 0) + 1
            by_severity[r["severity"]] = by_severity.get(r["severity"], 0) + 1
        return {"total": len(rows), "by_code": by_code, "by_severity": by_severity}


# ===========================================================================
# OUTPUT VALIDATOR (JSON contract + safety)
# ===========================================================================
# Reply modes. "analysis" is the original structured-JSON contract (used by the eval suite).
# "conversation" is a natural-language answer for people; it gets the same safety checks,
# with a text validator instead of the JSON contract.
MODE_ANALYSIS = "analysis"
MODE_CONVERSATION = "conversation"
VALID_MODES = (MODE_ANALYSIS, MODE_CONVERSATION)

CONFIDENCE_LEVELS = ("high", "medium", "low")
CONTRACT_KEYS = (
    "user",
    "current_context",
    "predicted_next_action",
    "missing_information",
    "response_quality",
)


@dataclass
class ValidationResult:
    is_valid: bool
    data: Optional[Dict[str, Any]] = None  # normalised contract JSON (only when valid)
    text: Optional[str] = None  # the same, serialised
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    redactions: List[str] = field(default_factory=list)


class OutputValidator:
    """Enforces the twin's tested JSON contract and checks the content.

    Expected shape (unchanged from the Phase 4 prompt):
      {"user": str,
       "current_context": {"summary": str, "observed_signals": [str]},
       "predicted_next_action": [{"action": str, "confidence": "high|medium|low",
                                  "why": str, "supporting_data": [str]}],
       "missing_information": [str],
       "response_quality": {"grounded_in_data": bool, "contains_guessing": bool}}

    Malformed JSON, missing keys, wrong types, bad confidence values, oversized
    output, blocked content, secrets, unsafe markup and prompt leaks are rejected.
    Extra keys are dropped (reported as a warning) so nothing unvalidated reaches the
    UI. PII is redacted rather than rejected when redact_pii is on. ```json fences are
    tolerated.
    """

    MAX_ITEMS, MAX_STR = 20, 800
    MAX_ACTIONS = 10
    # Markdown images and active HTML can make the viewer's browser contact a third party
    # (a classic way to smuggle data out of a model's answer), so they are never allowed.
    _UNSAFE_MARKUP = re.compile(
        r"!\[[^\]]*\]\([^)]*\)|<\s*/?\s*(?:img|script|iframe|svg|object|embed|link|style|form|input|a|meta|base)\b",
        re.I,
    )

    def __init__(
        self,
        max_length: int = 3000,
        content_filter: Optional[ContentFilter] = None,
        pii_detector: Optional[PIIDetector] = None,
        defense: Optional[SecurityDefense] = None,
        redact_pii: bool = True,
    ):
        if max_length <= 0:
            raise ValueError("max_length must be positive.")
        self.max_length = max_length
        self.content = content_filter or ContentFilter()
        self.pii = pii_detector or PIIDetector()
        self.defense = defense or SecurityDefense()
        self.redact_pii = redact_pii

    # -- schema -------------------------------------------------------------
    def _str(
        self, value: Any, path: str, issues: List[str], max_len: Optional[int] = None
    ) -> str:
        if not isinstance(value, str):
            issues.append(f"schema:bad_type:{path}")
            return ""
        if len(value) > (max_len or self.MAX_STR):
            issues.append(f"schema:too_long:{path}")
        return value.strip()

    def _str_list(self, value: Any, path: str, issues: List[str]) -> List[str]:
        if not isinstance(value, list):
            issues.append(f"schema:bad_type:{path}")
            return []
        if len(value) > self.MAX_ITEMS:
            issues.append(f"schema:too_many_items:{path}")
        return [
            self._str(v, f"{path}[{i}]", issues)
            for i, v in enumerate(value[: self.MAX_ITEMS])
        ]

    def _normalise(
        self, obj: Any, issues: List[str], warnings: List[str]
    ) -> Dict[str, Any]:
        if not isinstance(obj, dict):
            issues.append("schema:not_an_object")
            return {}
        for key in CONTRACT_KEYS:
            if key not in obj:
                issues.append(f"schema:missing_key:{key}")
        extras = sorted(set(obj) - set(CONTRACT_KEYS))
        if extras:
            warnings.append("dropped_extra_keys")

        out: Dict[str, Any] = {}
        if "user" in obj:
            out["user"] = self._str(obj["user"], "user", issues, 100)
        ctx = obj.get("current_context")
        if "current_context" in obj:
            if isinstance(ctx, dict):
                out["current_context"] = {
                    "summary": self._str(
                        ctx.get("summary"), "current_context.summary", issues
                    ),
                    "observed_signals": self._str_list(
                        ctx.get("observed_signals"),
                        "current_context.observed_signals",
                        issues,
                    ),
                }
            else:
                issues.append("schema:bad_type:current_context")
        if "predicted_next_action" in obj:
            actions = obj["predicted_next_action"]
            if not isinstance(actions, list):
                issues.append("schema:bad_type:predicted_next_action")
            else:
                if len(actions) > self.MAX_ACTIONS:
                    issues.append("schema:too_many_items:predicted_next_action")
                clean = []
                for i, a in enumerate(actions[: self.MAX_ACTIONS]):
                    p = f"predicted_next_action[{i}]"
                    if not isinstance(a, dict):
                        issues.append(f"schema:bad_type:{p}")
                        continue
                    conf = a.get("confidence")
                    conf_norm = conf.strip().lower() if isinstance(conf, str) else None
                    if conf_norm not in CONFIDENCE_LEVELS:
                        issues.append(f"schema:bad_value:{p}.confidence")
                    clean.append(
                        {
                            "action": self._str(a.get("action"), f"{p}.action", issues),
                            "confidence": conf_norm or "",
                            "why": self._str(a.get("why"), f"{p}.why", issues),
                            "supporting_data": self._str_list(
                                a.get("supporting_data"), f"{p}.supporting_data", issues
                            ),
                        }
                    )
                out["predicted_next_action"] = clean
        if "missing_information" in obj:
            out["missing_information"] = self._str_list(
                obj["missing_information"], "missing_information", issues
            )
        rq = obj.get("response_quality")
        if "response_quality" in obj:
            if (
                isinstance(rq, dict)
                and isinstance(rq.get("grounded_in_data"), bool)
                and isinstance(rq.get("contains_guessing"), bool)
            ):
                out["response_quality"] = {
                    "grounded_in_data": rq["grounded_in_data"],
                    "contains_guessing": rq["contains_guessing"],
                }
            else:
                issues.append("schema:bad_type:response_quality")
        return out

    # -- main entry ---------------------------------------------------------
    def validate(self, output: Any) -> ValidationResult:
        issues: List[str] = []
        warnings: List[str] = []
        if not isinstance(output, str):
            return ValidationResult(False, issues=["not_a_string"])
        if not output.strip():
            return ValidationResult(False, issues=["empty_output"])
        if len(output) > self.max_length:
            return ValidationResult(False, issues=["too_long"])

        text = output.strip()
        if text.startswith("```"):
            text = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", text).strip()
        try:
            parsed = json.loads(text)
        except (json.JSONDecodeError, RecursionError):
            return ValidationResult(False, issues=["malformed_json"])

        data = self._normalise(parsed, issues, warnings)
        if issues:
            return ValidationResult(False, issues=issues[:10], warnings=warnings)

        joined = "\n".join(iter_strings(data))

        if self._UNSAFE_MARKUP.search(joined):
            issues.append("unsafe_markup")

        content = self.content.check(joined, "output")
        issues += [f"content:{name}" for name in content.blocked]
        warnings += [f"flagged:{name}" for name in content.flagged]

        findings = self.pii.detect(joined)
        if findings.has_secrets:
            issues.append("secret_in_output")
        redactions: List[str] = []
        if findings.has_pii:
            if self.redact_pii:
                data = map_strings(data, lambda s: self.pii.redact(s)[0])
                redactions = list(findings.pii_types)
                warnings.append("pii_redacted")
            else:
                issues.append("pii_in_output")

        if self.defense.detect_leak(joined).leaked:
            issues.append("prompt_leak")

        rq = data["response_quality"]
        if rq["contains_guessing"]:
            warnings.append("self_reported_guessing")
        if not rq["grounded_in_data"]:
            warnings.append("self_reported_ungrounded")

        if issues:
            return ValidationResult(
                False, issues=issues, warnings=warnings, redactions=redactions
            )
        return ValidationResult(
            True,
            data=data,
            text=json.dumps(data, ensure_ascii=False),
            warnings=warnings,
            redactions=redactions,
        )

    # -- conversation mode ------------------------------------------------------
    # Links are blocked as well as images: the twin has no web data, so any address in an
    # answer is made up or planted, and a clickable link can carry data out.
    _UNSAFE_LINK = re.compile(
        r"https?://|\bwww\.[a-z0-9-]+\.|\]\(\s*(?:[a-z][a-z0-9+.\-]*:|//)"
        r"|!\[[^\]]*\]\s*\[|^\s*\[[^\]]+\]:\s*\S+",
        re.I | re.M,
    )

    def validate_text(self, output: Any) -> ValidationResult:
        """Check a natural-language answer: same safety rules, no JSON contract.

        Rejected: empty or oversized text, HTML and markdown images, links and web addresses,
        blocked content, secrets, and prompt leaks. PII is redacted (or rejected when
        redact_pii is off). If the model answers with the old JSON contract anyway, it is
        validated as that contract, so ``data`` is set and the caller can still render it.
        """
        if not isinstance(output, str):
            return ValidationResult(False, issues=["not_a_string"])
        text = output.strip()
        if not text:
            return ValidationResult(False, issues=["empty_output"])
        if len(output) > self.max_length:
            return ValidationResult(False, issues=["too_long"])

        candidate = text
        if candidate.startswith("```"):
            candidate = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", candidate).strip()
        if candidate.startswith("{"):
            try:
                parsed = json.loads(candidate)
            except (json.JSONDecodeError, RecursionError):
                parsed = None
            if isinstance(parsed, dict) and set(CONTRACT_KEYS) <= set(parsed):
                return self.validate(output)

        text = re.sub(r"\r\n?", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = "".join(
            ch
            for ch in text
            if ch in "\n\t" or unicodedata.category(ch) not in ("Cc", "Cf")
        )

        issues: List[str] = []
        warnings: List[str] = []
        if self._UNSAFE_MARKUP.search(text):
            issues.append("unsafe_markup")
        if self._UNSAFE_LINK.search(text):
            issues.append("link_in_output")

        content = self.content.check(text, "output")
        issues += [f"content:{name}" for name in content.blocked]
        warnings += [f"flagged:{name}" for name in content.flagged]

        findings = self.pii.detect(text)
        if findings.has_secrets:
            issues.append("secret_in_output")
        redactions: List[str] = []
        if findings.has_pii:
            if self.redact_pii:
                text = self.pii.redact(text)[0]
                redactions = list(findings.pii_types)
                warnings.append("pii_redacted")
            else:
                issues.append("pii_in_output")

        if self.defense.detect_leak(text).leaked:
            issues.append("prompt_leak")

        if issues:
            return ValidationResult(
                False, issues=issues, warnings=warnings, redactions=redactions
            )
        return ValidationResult(
            True, data=None, text=text, warnings=warnings, redactions=redactions
        )


# ===========================================================================
# PIPELINE
# ===========================================================================
MESSAGES = {
    "invalid_input": "Please type a message.",
    "too_long": "That message is too long. Please keep it under {limit} characters.",
    "rate_limited": "You're sending messages too quickly. Please try again in {retry:g} seconds.",
    "blocked_injection": "I can't act on that request. Please rephrase it as a question about your activity, priorities or goals.",
    "blocked_content": "I can't help with that topic here. Please rephrase your question.",
    "blocked_crisis": (
        "This sounds like it may be a hard moment, and I'm not the right place for it. "
        "Please reach out to someone you trust, or to local emergency services if you are in danger."
    ),
    "blocked_output": "I couldn't produce a safe, valid answer to that. Please try rephrasing your question.",
    "error": "The twin is temporarily unavailable. Please try again shortly.",
    "upstream_busy": "The model is busy right now. Please wait a minute and try again.",
}


@dataclass
class InputCheck:
    allowed: bool
    status: str  # ok | invalid_input | too_long | rate_limited | blocked_injection | blocked_content | blocked_crisis
    message: str = ""  # user-facing text when not allowed
    sanitized: str = ""  # what to send to the model (redacted); "" when blocked
    notices: List[str] = field(default_factory=list)
    violations: List[str] = field(default_factory=list)
    retry_after: float = 0.0
    injection_score: float = 0.0


@dataclass
class OutputCheck:
    ok: bool
    data: Optional[Dict[str, Any]] = None
    text: Optional[str] = None
    notices: List[str] = field(default_factory=list)
    violations: List[str] = field(default_factory=list)
    message: str = ""


class SafetyPipeline:
    """Wires every check together. The caller makes the model call between
    check_input() and check_output() (see agent.safe_chat)."""

    def __init__(
        self,
        config: Optional[SafetyConfig] = None,
        protected_text: str = "",
        violation_log: Optional[ViolationLog] = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.config = config or SafetyConfig()
        c = self.config
        self.pii = PIIDetector()
        self.content = ContentFilter(c.profanity_input_action, c.domain_input_action)
        self.injection = HardenedInjectionDetector(threshold=c.injection_threshold)
        self.defense = SecurityDefense(protected_text, min_matches=c.leak_min_matches)
        self.limiter = RateLimiter(
            c.rate_limit_calls, c.rate_limit_window_seconds, clock=clock
        )
        self.violations = violation_log or ViolationLog(
            c.violation_log_path, c.max_logged_violations
        )
        self.validator = OutputValidator(
            c.max_output_chars,
            self.content,
            self.pii,
            self.defense,
            redact_pii=c.redact_output_pii,
        )

    def _log(
        self,
        stage: str,
        code: str,
        severity: str,
        identifier: str,
        text: Optional[str],
        **details: Any,
    ) -> None:
        self.violations.record(stage, code, severity, identifier, text, details)

    # -- input ----------------------------------------------------------------
    def check_input(self, text: Any, identifier: str = "default") -> InputCheck:
        c = self.config

        def blocked(status: str, message: str, **kw: Any) -> InputCheck:
            return InputCheck(False, status, message, **kw)

        # 0. shape checks
        if not isinstance(text, str) or not text.strip():
            return blocked("invalid_input", MESSAGES["invalid_input"])
        if len(text) > c.max_input_chars:
            self._log(
                "input",
                "input_too_long",
                "low",
                identifier,
                text,
                limit=c.max_input_chars,
            )
            return blocked(
                "too_long",
                MESSAGES["too_long"].format(limit=c.max_input_chars),
                violations=["input_too_long"],
            )

        # 1. rate limit
        rate = self.limiter.acquire(identifier)
        if not rate.allowed:
            self._log(
                "input",
                "rate_limited",
                "low",
                identifier,
                text,
                retry_after=rate.retry_after,
            )
            return blocked(
                "rate_limited",
                MESSAGES["rate_limited"].format(retry=rate.retry_after),
                violations=["rate_limited"],
                retry_after=rate.retry_after,
            )

        violations: List[str] = []

        # 2. injection
        inj = self.injection.detect(text)
        if inj.is_injection:
            self._log(
                "input",
                "injection_blocked",
                "high",
                identifier,
                text,
                score=inj.score,
                rules=inj.matched,
            )
            return blocked(
                "blocked_injection",
                MESSAGES["blocked_injection"],
                violations=["injection_blocked"],
                injection_score=inj.score,
            )
        if inj.score >= c.injection_flag_threshold:
            self._log(
                "input",
                "injection_suspected",
                "low",
                identifier,
                text,
                score=inj.score,
                rules=inj.matched,
            )
            violations.append("injection_suspected")

        # 3. content
        content = self.content.check(text, "input")
        for name in content.flagged:
            self._log("input", f"content_flagged:{name}", "low", identifier, text)
            violations.append(f"content_flagged:{name}")
        if content.blocked:
            for name in content.blocked:
                self._log(
                    "input", f"content_blocked:{name}", "medium", identifier, text
                )
                violations.append(f"content_blocked:{name}")
            status = (
                "blocked_crisis"
                if "self_harm" in content.blocked
                else "blocked_content"
            )
            return blocked(
                status,
                MESSAGES[status],
                violations=violations,
                injection_score=inj.score,
            )

        # 4. sanitise, then redact PII and secrets
        cleaned = self.defense.sanitize_input(text)
        cleaned, findings = self.pii.redact(cleaned)
        notices: List[str] = []
        if findings.has_secrets:
            self._log(
                "input",
                "secret_redacted_input",
                "high",
                identifier,
                text,
                types=findings.secret_types,
            )
            violations.append("secret_redacted_input")
            notices.append(
                "A possible password or key was removed from your message before it was sent."
            )
        if findings.has_pii:
            self._log(
                "input",
                "pii_redacted_input",
                "low",
                identifier,
                text,
                types=findings.pii_types,
            )
            violations.append("pii_redacted_input")
            notices.append(
                "Personal details were redacted from your message before it was sent."
            )
        return InputCheck(
            True,
            "ok",
            sanitized=cleaned,
            notices=notices,
            violations=violations,
            injection_score=inj.score,
        )

    # -- output ---------------------------------------------------------------
    def check_output(
        self, raw: Any, identifier: str = "default", mode: str = MODE_ANALYSIS
    ) -> OutputCheck:
        """Validate the model's reply. ``mode`` picks the contract: the structured JSON
        ("analysis", the default) or a natural-language answer ("conversation")."""
        if mode not in VALID_MODES:
            raise ValueError(f"mode must be one of: {', '.join(VALID_MODES)}.")
        result = (
            self.validator.validate_text(raw)
            if mode == MODE_CONVERSATION
            else self.validator.validate(raw)
        )
        # raw model text is passed only so it can be hashed, never stored
        text_for_hash = raw if isinstance(raw, str) else None
        violations: List[str] = []
        notices: List[str] = []

        if result.redactions:
            self._log(
                "output",
                "output_pii_redacted",
                "medium",
                identifier,
                text_for_hash,
                types=result.redactions,
            )
            violations.append("output_pii_redacted")
            notices.append("Some personal details in the answer were redacted.")
        if (
            "self_reported_guessing" in result.warnings
            or "self_reported_ungrounded" in result.warnings
        ):
            notices.append(
                "The twin flagged this answer as partly uncertain. Check it against your own records."
            )
        for w in result.warnings:
            if w.startswith("flagged:"):
                self._log("output", f"content_{w}", "low", identifier, text_for_hash)

        if result.is_valid:
            return OutputCheck(True, result.data, result.text, notices, violations)

        severity = (
            "high"
            if ("prompt_leak" in result.issues or "secret_in_output" in result.issues)
            else "medium"
        )
        self._log(
            "output",
            "output_invalid",
            severity,
            identifier,
            text_for_hash,
            issues=result.issues,
        )
        violations.append("output_invalid")
        return OutputCheck(
            False,
            notices=notices,
            violations=violations,
            message=MESSAGES["blocked_output"],
        )

    # -- fallback ---------------------------------------------------------------
    def fallback_response(self, message: str) -> Dict[str, Any]:
        """A contract-shaped reply, so anything that renders the normal JSON can render this too."""
        return {
            "user": self.config.user_name,
            "current_context": {"summary": message, "observed_signals": []},
            "predicted_next_action": [],
            "missing_information": ["no_answer_generated"],
            "response_quality": {"grounded_in_data": True, "contains_guessing": False},
        }

    def report(self, identifier: Optional[str] = None) -> Dict[str, Any]:
        return self.violations.report(identifier)
