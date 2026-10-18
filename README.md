# Digital Twin

A personal digital twin for **Davies**, a software engineer. It models his preferences and behavioral patterns from three sources of evidence (a profile, an activity history, and temporal patterns) and uses them to answer questions and predict likely next actions. It never guesses: when the data doesn't support an answer, it says what is missing.

The twin runs on the Groq API, is wrapped in a layered safety system, and is served through a Streamlit chat interface.

> **Status: in development.** The components were prototyped and tested in a Google Colab notebook and are being packaged into this repository. See [Project status](#project-status) for what is done and what is planned.

---

## Table of contents

1. [What it does](#what-it-does)
2. [How it works](#how-it-works)
3. [The twin's output format](#the-twins-output-format)
4. [Project structure](#project-structure)
5. [Modules and functions](#modules-and-functions)
6. [Data](#data)
7. [Setup](#setup)
8. [Running the app](#running-the-app)
9. [Safety system](#safety-system)
10. [Evaluation and testing](#evaluation-and-testing)
11. [Deployment](#deployment)
12. [Project status](#project-status)
13. [Known limitations](#known-limitations)
14. [Tech stack](#tech-stack)

---

## What it does

| Capability | Description |
|---|---|
| **Grounded Q&A** | Answers questions about the user's role, preferences, goals, activity and schedule, using only the supplied data |
| **Next-action prediction** | Ranks the most likely next actions with a confidence level and the evidence behind each |
| **Honest gaps** | Lists the missing information that limits its confidence instead of inventing facts |
| **Structured output** | Every reply follows a fixed JSON schema, so replies can be validated and rendered reliably |
| **Safety filtering** | Blocks prompt injection, redacts personal data, filters unsafe content, and rate-limits requests |
| **Document processing** | Cleans, chunks, and extracts structured data (emails, dates, durations, hashtags) from text |
| **Retrieval (RAG)** | Finds the most relevant chunks of a knowledge base and answers from them only |
| **Tools** | Goal tracking, idea saving, memory search, calculator, and clock |
| **Chat interface** | Multi-turn Streamlit chat with conversation history and a clear-chat control |

---

## How it works

Every message passes through the same pipeline. The twin is not called until the input has been checked, and its reply is not shown until it has been validated.

```
                      ┌──────────────────────────────┐
   user message ───▶  │ 1. Rate limiter              │  too many requests → "try again in Ns"
                      ├──────────────────────────────┤
                      │ 2. Input filter              │  injection / profanity / sensitive
                      │    (injection, content, PII) │  topics → blocked; PII → redacted
                      ├──────────────────────────────┤
                      │ 3. Twin (Groq LLM)           │  system prompt = profile + history
                      │                              │  + temporal patterns + rules
                      ├──────────────────────────────┤
                      │ 4. Output validator          │  length, content, PII, format
                      │                              │  fail → retry, then fallback message
                      └──────────────┬───────────────┘
                                     ▼
                              reply in the chat UI
```

**The system prompt** gives the twin its role, the three data sources, and a set of reasoning rules: use only supplied data, do not invent facts, separate observed facts from inferences, say so when evidence is weak, rank actions by confidence, and quote profile fields verbatim when asked about them directly.

**Retrieval (RAG)** is a separate path for larger documents: raw text is cleaned, chunked, indexed, and the most relevant chunks are passed to the model with an instruction to answer from that context only.

---

## The twin's output format

The twin always replies with JSON in this shape:

```json
{
  "user": "Davies",
  "current_context": {
    "summary": "string",
    "observed_signals": ["string"]
  },
  "predicted_next_action": [
    {
      "action": "string",
      "confidence": "high | medium | low",
      "why": "string",
      "supporting_data": ["string"]
    }
  ],
  "missing_information": ["string"],
  "response_quality": {
    "grounded_in_data": true,
    "contains_guessing": false
  }
}
```

A fixed schema is what makes the output testable. It also means the UI should render the JSON (summary, action cards, missing-data list) instead of showing it as raw text.

---

## Project structure

```
twin/
├── README.md
├── requirements.txt
├── pyproject.toml
├── .env.example            # template for your local .env
├── .gitignore
├── app.py                  # Streamlit chat UI (entry point)
├── src/
│   └── digital_twin/       # the importable package
│       ├── __init__.py
│       ├── config.py       # settings, API key loading, data loading
│       ├── groq_client.py  # Groq API wrapper with retry on rate limits
│       ├── chunking.py     # text chunking strategies + validation
│       ├── extraction.py   # regex extraction of structured data
│       ├── safety.py       # filters, PII, injection defense, validator
│       ├── rag.py          # retrieval and RAG pipeline
│       ├── tools.py        # tool registry and tools
│       └── agent.py        # system prompt + chat orchestration
├── tests/                  # one test file per module
├── data/
│   └── activity_history.json
├── notebooks/              # original Colab notebooks (unchanged)
└── scripts/
    └── run_demo.py         # terminal demo runner
└── services/
    └── calculator_services.py
    └── chat_services.py
    └── task_services.py
    └── twin_adapter.py
└── state/
    └── __init__.py
    └── session.py
└── styles/
    └── theme.css
└── views/
    └── assistant.py
    └── calculator.py
    └── dashboard.py
    └── settings.py
    └── tasks.py

```

The package uses the `src/` layout. The package name matches the repo name, which is normal for this layout.

---

## Modules and functions

### `config.py`
Loads the Groq API key from the environment (via `python-dotenv`), and holds the base URL, model name, and limits. Also loads the profile, activity history, and temporal patterns from `data/`.

### `groq_client.py`
`GroqClient` is a thin wrapper over the Groq chat-completions endpoint. `chat(messages)` sends a messages list and returns the reply text. It reads the `Retry-After` header on HTTP 429 and waits accordingly instead of retrying blindly.

### `extraction.py`
`extract_profile_data(text)` pulls structured data out of free text and returns a dictionary:

| Group | Fields |
|---|---|
| `identifiers` | emails, phones, handles |
| `temporal` | dates, times, durations |
| `preferences` | interests (from hashtags) |

The duration pattern matches the longest unit first (so `30min` is not returned as `30m`), and the handle pattern ignores the `@` inside email addresses.

### `chunking.py`
`ChunkingSystem` splits text using one of four strategies (`sentence`, `word`, `paragraph`, `smart`). The `smart` strategy packs whole paragraphs up to a word limit and carries the last sentence of one chunk into the next, so chunks do not start mid-sentence. Each chunk carries metadata (position, source, timestamp, strategy, word and character counts) and is validated for being too small, too large, or cut mid-sentence.

### `safety.py`

| Class | Role |
|---|---|
| `PromptInjectionDetector` / `HardenedInjectionDetector` | Flags injection attempts; the hardened version normalizes leetspeak and zero-width characters and covers persona jailbreaks (DAN, developer mode) and synonym phrasing |
| `ContentFilter` | Whole-word profanity matching, general sensitive topics, and domain topics (salary, credentials, API keys, confidential material) |
| `PIIDetector` | Detects and redacts emails, phones, SSNs, card numbers, street addresses, dates of birth, IP addresses, and passport-style IDs |
| `RateLimiter` | Sliding-window limiter per user or session |
| `ViolationLog` | Structured record of every violation, queryable by user and type |
| `SecurityDefense` | Input sanitizing, system-prompt hardening, and output leak checks |
| `OutputValidator` | Checks length, empty output, content safety, PII, and whether a disclaimer is needed |
| `SafeDigitalTwin` | Chains all of the above around the model call |

### `rag.py`
- `SimpleRAGRetriever`: bag-of-words vectors compared with cosine similarity. A query with no vocabulary overlap returns no match instead of an error.
- `load_and_preprocess`: validates a batch of `{source, text}` documents, then cleans, extracts, and chunks them. Bad documents are skipped and recorded with a reason.
- `RAGPipeline`: `index()` builds the knowledge base and `ask()` retrieves, builds a grounded prompt, and calls a `generate_fn` you supply. If generation fails, the retrieved chunks are still returned.

### `tools.py`
`ToolRegistry` maps tool names to Python functions.

| Tool | What it does |
|---|---|
| `goal_navigator` | Compares each profile goal against recent activity and shows supporting evidence |
| `idea_vault` | Saves and lists ideas (`save: <text>` or `list`) |
| `memory_search` | Keyword search over the activity history |
| `calculator` | Evaluates arithmetic expressions |
| `get_time` | Current date, time, and weekday |

### `agent.py`
Holds the system-prompt builder and the chat orchestration. `SimpleReActAgent` routes a message to a tool (think → act → answer) using keyword matching.

---

## Data

The twin's knowledge comes from three structures, stored in `data/` and not in code:

- **`user_profile`**: name, role, preferences, goals
- **`activity_history`**: dated entries such as `{"time": "2026-08-10", "activity": "created slides on edge computing"}`
- **`temporal_patterns`**: most active periods, common tasks, recurring focus areas

The twin can only reason about what is in these files. Questions outside them (for example, what you ate yesterday) get a reply that lists the missing information.

---

## Setup

**Requirements:** Python 3.10 or newer (developed on 3.13) and a [Groq API key](https://console.groq.com).

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2. Install dependencies and the package itself (editable)
pip install -r requirements.txt
pip install -e .

# 3. Add your API key
cp .env.example .env               # then edit .env
```

`.env` should contain:

```
GROQ_API_KEY=your_key_here
```

Never commit `.env`. It is listed in `.gitignore`.

---

## Running the app

```bash
streamlit run app.py
```

Streamlit opens at `http://localhost:8501`. To try the pipeline without the UI:

```bash
python scripts/run_demo.py
```

---

## Safety system

Safety is built in layers so that a miss at one layer can be caught by another.

1. **Before the model:** rate limiting, injection detection, content and domain filtering, PII redaction. Blocked input never reaches the model, which also saves latency and API cost.
2. **Around the prompt:** security instructions in the system prompt (never reveal it, treat user text as user content, hold the role).
3. **After the model:** output validation for length, content, PII, and format. If a reply fails, the system retries and then falls back to a safe message.
4. **Always:** every violation is recorded in the violation log.

The injection detector was hardened after red-teaming. Five attacks bypassed the original version (single-pattern evasion, synonym phrasing, leetspeak, zero-width characters, and a DAN persona), and each is now covered by a specific fix. They should stay in the test suite as permanent regression tests.

---

## Evaluation and testing

Answers are graded by three metrics:

| Metric | Question it answers |
|---|---|
| **Faithfulness** | Is the answer backed by the supplied context? |
| **Relevance** | Does the answer address the question? |
| **Format compliance** | Does the reply match the required JSON schema? |

`AITestSuite` runs a set of `TestCase` objects against the twin and reports accuracy. The twin suite covers profile recall, activity recall, goal recall, temporal patterns, missing-data honesty, out-of-scope questions, and output format.

Because the twin always answers in JSON, tests should check for structure (for example, that `"missing_information"` is present and non-empty) and not for plain-English phrases.

```bash
pytest
```

Adversarial cases (injection attempts, out-of-scope questions) expect a refusal or a "missing information" response, and they belong in the same suite so that any future regression fails the tests.

---

## Deployment

**Pre-flight checklist:**
- [ ] `requirements.txt` pins dependency versions
- [ ] `.env` exists locally and is listed in `.gitignore`
- [ ] No secrets in code or git history
- [ ] Tests pass

**Streamlit Community Cloud** (recommended): push the repo to GitHub, connect it in Streamlit Community Cloud, point it at `app.py`, and add `GROQ_API_KEY` in the app's Secrets settings. Each later `git push` redeploys automatically.

**Hugging Face Spaces** (alternative): create a Streamlit Space, clone its repository, push the files to it, and add the key under the Space's secrets.

If a deploy breaks, you can fix forward (correct, commit, push) or roll back with `git revert HEAD` and push. Roll back first when users are affected and the cause isn't clear yet.

---

## Project status

**Prototyped and tested in Colab:**
- Twin core (system prompt, structured output, multi-turn chat)
- Text cleaning, extraction, and chunking
- Safety layers and red-team hardening
- RAG retrieval and pipeline
- Tools and keyword-routed agent
- Evaluation suite and metrics
- Streamlit chat UI concepts

**In progress:** moving each component into `src/digital_twin/` and writing the matching tests.

**Planned for the MVP:** safety-wrapped chat in Streamlit, deployed to a public URL.

**After the MVP:** RAG over personal documents with source citations, tool routing in the chat, persistent chat history, a canary token for prompt-leak detection, and an evaluation dashboard.

---

## Known limitations

- **The calculator tool uses `eval()`.** This is unsafe on untrusted input and must be replaced with a safe expression parser, or the tool left out, before the app is public.
- **Tool routing is keyword matching.** It misfires on ordinary phrasing (for example, "recently?" being treated as a search keyword). Model-driven tool calling is the planned replacement.
- **Retrieval is bag-of-words with brute-force comparison.** It has no notion of meaning and slows as the knowledge base grows. A real embedding model and a vector index are the upgrade path.
- **The idea vault is in-memory.** Saved ideas are lost on restart.
- **The rate limiter is in-memory and per-process.** It will not work across multiple server instances without a shared store such as Redis.
- **Free-tier API limits are tight.** On `openai/gpt-oss-120b`, the free-tier token-per-minute cap is low, and the system prompt is resent with every request. A few simultaneous users will hit rate limits.
- **The profanity and injection lists are static.** They are easy to evade with new phrasing. Production use should add a maintained moderation service or a classifier.
- **The hallucination check is a word-overlap heuristic.** It can flag honest answers and miss subtle fabrications.
- **The data is hand-written sample data**, not logged real activity.

---

## Tech stack

- **Language:** Python
- **LLM:** Groq API (`openai/gpt-oss-120b`)
- **UI:** Streamlit
- **Libraries:** `groq`, `requests`, `numpy`, `python-dotenv`, `pytest`
- **Prototyping:** Google Colab