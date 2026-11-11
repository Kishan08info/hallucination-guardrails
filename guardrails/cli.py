"""CLI: check one answer against a corpus of source passages.

Usage:
    python -m guardrails.cli check --answer "Pro costs $199/mo." \\
        --sources data/corpus.json
    python -m guardrails.cli check --answer-file answer.txt \\
        --sources data/corpus.json --freshness-cutoff 2025-01-01 --json

Exit codes: 0 = PASS, 1 = FLAG, 2 = BLOCK, 3 = usage error.
Corpus format: JSON list of {"passage_id", "text", "as_of"} objects.
"""

from __future__ import annotations

import argparse
import json
import sys

from .policy import Policy
from .report import answer_report
from .verify import SourcePassage, Verifier


def load_corpus(path: str) -> list[SourcePassage]:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return [
        SourcePassage(
            passage_id=item["passage_id"],
            text=item["text"],
            as_of=item.get("as_of"),
        )
        for item in raw
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="guardrails",
        description="Verify an LLM answer against source passages.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    check = sub.add_parser("check", help="Check one answer.")
    group = check.add_mutually_exclusive_group(required=True)
    group.add_argument("--answer", help="Answer text to check.")
    group.add_argument("--answer-file", help="File containing the answer.")
    check.add_argument("--sources", required=True, help="Corpus JSON file.")
    check.add_argument(
        "--freshness-cutoff",
        default=None,
        help="ISO date; passages older than this are excluded.",
    )
    check.add_argument(
        "--support-threshold", type=float, default=0.30,
        help="Similarity needed for SUPPORTED (default 0.30).",
    )
    check.add_argument(
        "--partial-threshold", type=float, default=0.15,
        help="Similarity needed for PARTIAL (default 0.15).",
    )
    check.add_argument("--json", action="store_true", help="Print JSON report.")

    args = parser.parse_args(argv)

    if args.command == "check":
        if args.answer_file:
            with open(args.answer_file, encoding="utf-8") as f:
                answer = f.read().strip()
        else:
            answer = args.answer
        passages = load_corpus(args.sources)
        verifier = Verifier(
            passages,
            support_threshold=args.support_threshold,
            partial_threshold=args.partial_threshold,
            freshness_cutoff=args.freshness_cutoff,
        )
        verdicts = verifier.verify_answer(answer)
        decision = Policy().decide(verdicts, answer)
        record = answer_report("cli", answer, verdicts, decision)
        if args.json:
            print(json.dumps(record, indent=2, ensure_ascii=False))
        else:
            print(f"Decision: {decision.decision}")
            for v in verdicts:
                print(
                    f"  [{v.verdict}] ({v.best_score:.2f}, "
                    f"conf {v.confidence:.2f}) {v.claim.text}"
                )
                if v.numbers_missing:
                    print(f"      numbers not in source: {v.numbers_missing}")
                if v.entities_missing:
                    print(f"      entities not in source: {v.entities_missing}")
            print(f"Safe answer: {decision.safe_answer}")
        return {"PASS": 0, "FLAG": 1, "BLOCK": 2}[decision.decision]
    return 3


if __name__ == "__main__":
    sys.exit(main())
