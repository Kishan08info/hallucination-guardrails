"""Demo runner: check the 8 demo answers against the corpus, twice —
once with the stale 2024 pricing doc included, once with a freshness
cutoff of 2025-01-01 that excludes it — and score hallucination
detection against the claim-level ground truth.

Writes outputs/report.json (per-answer records for both runs) and
outputs/summary.json (detection precision/recall per run), and prints
a human-readable summary. Deterministic: no wall-clock values in the
outputs, so a from-scratch re-run must reproduce them byte-identically.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from guardrails.cli import load_corpus  # noqa: E402
from guardrails.policy import Policy  # noqa: E402
from guardrails.report import answer_report, write_report  # noqa: E402
from guardrails.verify import UNSUPPORTED, Verifier  # noqa: E402


def load_demo_answers(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def score_detection(records: list[dict], demos: list[dict]) -> dict:
    """Precision/recall of hallucination detection over labelled claims.

    A claim labelled 'hallucinated' counts as detected iff its verdict
    is UNSUPPORTED. 'unverifiable' claims are excluded from the score.
    """
    tp = fp = fn = 0
    misses: list[str] = []
    for record, demo in zip(records, demos):
        for verdict, expected in zip(record["claims"], demo["claims_expected"]):
            detected = verdict["verdict"] == UNSUPPORTED
            if expected == "hallucinated" and detected:
                tp += 1
            elif expected == "hallucinated":
                fn += 1
                misses.append(
                    f"{demo['answer_id']}: {verdict['claim']!r} "
                    f"-> {verdict['verdict']}"
                )
            elif expected == "supported" and detected:
                fp += 1
                misses.append(
                    f"{demo['answer_id']}: supported claim flagged "
                    f"UNSUPPORTED: {verdict['claim']!r}"
                )
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    return {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "misses": misses,
    }


def run_once(
    demos: list[dict],
    passages,
    label: str,
    policy: Policy,
) -> tuple[list[dict], dict]:
    verifier = Verifier(passages)
    records = []
    for demo in demos:
        verdicts = verifier.verify_answer(demo["answer"])
        decision = policy.decide(verdicts, demo["answer"])
        records.append(
            answer_report(demo["answer_id"], demo["answer"], verdicts, decision)
        )
    stats = score_detection(records, demos)
    decisions = {
        d: sum(1 for r in records if r["decision"] == d)
        for d in ("PASS", "FLAG", "BLOCK")
    }
    summary = {
        "run": label,
        "n_answers": len(records),
        "n_claims": sum(len(r["claims"]) for r in records),
        "decisions": decisions,
        "detection": stats,
    }
    return records, summary


def main() -> None:
    corpus_path = ROOT / "data" / "corpus.json"
    demos = load_demo_answers(ROOT / "data" / "demo_answers.json")
    policy = Policy()
    out_dir = ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)

    # Run 1: stale 2024 pricing doc included (the naive corpus).
    passages_all = load_corpus(str(corpus_path))
    records_all, summary_all = run_once(
        demos, passages_all, "stale-doc-included", policy
    )

    # Run 2: freshness cutoff excludes anything before 2025-01-01.
    passages_fresh = [
        p for p in passages_all if p.as_of is None or p.as_of >= "2025-01-01"
    ]
    records_fresh, summary_fresh = run_once(
        demos, passages_fresh, "freshness-cutoff-2025-01-01", policy
    )

    write_report(records_all + records_fresh, str(out_dir / "report.json"))
    summary = {
        "runs": [summary_all, summary_fresh],
        "corpus_passages": len(passages_all),
        "fresh_passages": len(passages_fresh),
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
        f.write("\n")

    for s in summary["runs"]:
        d = s["detection"]
        print(f"== {s['run']} ==")
        print(
            f"  answers: {s['n_answers']}, claims: {s['n_claims']}, "
            f"decisions: {s['decisions']}"
        )
        print(
            f"  hallucination detection: precision {d['precision']}, "
            f"recall {d['recall']} "
            f"(TP {d['true_positives']}, FP {d['false_positives']}, "
            f"FN {d['false_negatives']})"
        )
        for miss in d["misses"]:
            print(f"    miss: {miss}")
    print("Wrote outputs/report.json and outputs/summary.json")


if __name__ == "__main__":
    main()
