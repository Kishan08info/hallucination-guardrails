# Hallucination Guardrails

A drop-in Python guardrail library that verifies an LLM/RAG answer against
its retrieved source passages **before** it reaches the user — and rewrites
the answer to keep only the claims it can cite.

**All data in this repo is SYNTHETIC** (fictional company "Northwind
Analytics"). This is a portfolio skill demo, not work experience.

## Why this exists

RAG answers sound confident even when they are wrong. The failure modes
hiring teams keep asking for in AI Engineer postings — evaluation,
guardrails, hallucination checks (e.g. BPD's AI Engineer role, Tenor's
evaluation/promotion gates, Stacksync's guardrails and evaluation suites)
— are exactly what this library implements: check every claim against the
sources, decide PASS / FLAG / BLOCK, and never show a number you cannot cite.

## What it does

1. **Claim extraction** — splits an answer into sentence-level claims
   (`guardrails/claims.py`).
2. **Citation verification** — each claim is matched to its best source
   passage by TF-IDF cosine similarity, then checked with two hard factual
   tests (`guardrails/verify.py`):
   - **Numbers**: every number in the claim ($49, 2024, 15%) must appear
     in the passage. Similarity alone cannot catch a wrong price —
     "$149/mo" and "$199/mo" share almost every token.
   - **Entities**: every capitalized name must appear in the passage.
   - **Coverage**: the fraction of the claim's content tokens present in
     the passage — catches invented features that share generic words
     ("offers", "real-time") with a passage but add new content.
3. **Confidence scoring** — an explainable blend: 0.50 × similarity +
   0.30 × number agreement + 0.20 × entity agreement
   (`guardrails/confidence.py`).
4. **Policy** — PASS (all supported), FLAG (partial claims marked
   `[unverified]`), BLOCK (unsupported claims dropped; full abstention if
   nothing is supported) (`guardrails/policy.py`).
5. **Freshness cutoff** — passages carry an `as_of` date; a cutoff
   excludes stale docs so a 2024 pricing page cannot "support" a wrong
   2026 price.

Standard library only — no dependencies, no network, no LLM calls.

## Quick start

```bash
python demo.py            # runs the 8-answer demo set, writes outputs/
pytest tests/             # 18 tests

# Check one answer from the command line (exit 0 PASS / 1 FLAG / 2 BLOCK):
python -m guardrails.cli check \
  --answer "The Pro plan costs $199 per month." \
  --sources data/corpus.json --freshness-cutoff 2025-01-01

python -m guardrails.cli check \
  --answer "The Growth plan costs $79 per month." \
  --sources data/corpus.json --json
```

```python
from guardrails import Verifier, Policy
from guardrails.cli import load_corpus

passages = load_corpus("data/corpus.json")
verifier = Verifier(passages, freshness_cutoff="2025-01-01")
verdicts = verifier.verify_answer(answer)
decision = Policy().decide(verdicts, answer)
print(decision.decision)      # PASS / FLAG / BLOCK
print(decision.safe_answer)   # answer with unsupported claims removed
```

## Results (real runs — `python demo.py`)

Demo set: 8 answers, 11 claims, 5 planted hallucinations, 5 supported
claims, 1 unverifiable claim. Corpus: 8 passages, one deliberately stale
(2024 pricing). All numbers below come from `outputs/summary.json`;
re-running `demo.py` reproduces them byte-identically.

| Run | Precision | Recall | Misses |
| --- | --- | --- | --- |
| Stale 2024 doc included | 1.0 | 0.60 | stale price passes (false negative); attribution swap passes |
| Freshness cutoff 2025-01-01 | 1.0 | 0.80 | attribution swap passes |

- **0 false positives in both runs**: no supported claim was ever flagged
  UNSUPPORTED.
- **Caught**: wrong price ($79 vs $99), invented feature (AI fraud
  detection), wrong API limit (5,000 vs 1,000 requests/day), and — with
  the cutoff — the stale price ($29 vs $49).
- **The stale-doc run is the point**: without a freshness cutoff, the
  stale 2024 pricing doc "supports" the wrong $29 price. Retrieval
  staleness is a real guardrail failure mode, and the cutoff fixes it.
- **Known limitation, disclosed**: the attribution swap ("Meridian Health
  uses Northwind for weekly sales reporting" — the passage attributes that
  use to Beta Ltd) passes in both runs. Bag-of-words checks cannot catch
  who-did-what swaps; that needs NLI/entailment, which is the documented
  next step, not a hidden gap.
- The vague claim ("a popular analytics tool with many happy customers")
  is PARTIAL in both runs — flagged `[unverified]`, never presented as
  verified.

Tests: **18/18 pass** (`pytest tests/`), including CLI end-to-end tests
that assert the stale-price behaviour flips with the cutoff.

## Project layout

```
guardrails/
  text.py        tokenize / sentence split / TF-IDF cosine similarity
  claims.py      sentence-level claim extraction
  verify.py      citation verifier (similarity + numbers + entities + coverage)
  confidence.py  explainable per-claim confidence score
  policy.py      PASS / FLAG / BLOCK + safe rewrite
  report.py      JSON report builder
  cli.py         `python -m guardrails.cli check ...`
data/
  corpus.json        8 synthetic source passages (1 stale)
  demo_answers.json  8 demo answers with claim-level ground truth
demo.py            runs the demo set twice, writes outputs/
tests/             18 pytest tests (unit + CLI end-to-end)
outputs/
  report.json      per-claim verdicts for both runs
  summary.json     precision/recall per run
```

## How to explain it in an interview (60 seconds)

> "RAG answers hallucinate most often on numbers — a wrong price shares
> nearly every token with the right one, so similarity alone can't catch
> it. I built a guardrail that splits an answer into claims and checks
> each one against its sources three ways: TF-IDF similarity, exact
> number agreement, and entity agreement, plus token coverage. A policy
> layer then passes, flags, or blocks the answer and rewrites it to keep
> only citable claims. On a synthetic demo set it catches 4 of 5 planted
> hallucinations with zero false positives — and it showed me that stale
> retrieved docs are a failure mode of their own: a 2024 pricing page
> 'supported' a wrong price until I added a freshness cutoff. The one
> thing it still misses is attribution swaps, which need an entailment
> model — I documented that instead of hiding it."
