"""JSON report: one record per answer, with per-claim verdicts."""

from __future__ import annotations

import json

from .policy import Decision
from .verify import ClaimVerdict


def verdict_to_dict(v: ClaimVerdict) -> dict:
    return {
        "claim": v.claim.text,
        "verdict": v.verdict,
        "best_passage_id": v.best_passage_id,
        "best_score": v.best_score,
        "coverage": v.coverage,
        "confidence": v.confidence,
        "numbers_in_claim": v.numbers_in_claim,
        "numbers_missing": v.numbers_missing,
        "entities_in_claim": v.entities_in_claim,
        "entities_missing": v.entities_missing,
    }


def answer_report(
    answer_id: str,
    answer: str,
    verdicts: list[ClaimVerdict],
    decision: Decision,
) -> dict:
    return {
        "answer_id": answer_id,
        "original_answer": answer,
        "decision": decision.decision,
        "safe_answer": decision.safe_answer,
        "dropped_claims": decision.dropped_claims,
        "flagged_claims": decision.flagged_claims,
        "counts": {
            "supported": decision.n_supported,
            "partial": decision.n_partial,
            "unsupported": decision.n_unsupported,
        },
        "claims": [verdict_to_dict(v) for v in verdicts],
    }


def write_report(records: list[dict], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
        f.write("\n")
