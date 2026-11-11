"""Policy: turn claim verdicts into a decision and a safe answer.

Decisions:
    PASS  — every claim SUPPORTED. Answer goes out unchanged.
    FLAG  — no UNSUPPORTED claims, but at least one PARTIAL. Answer goes
            out with PARTIAL claims marked "[unverified]".
    BLOCK — at least one UNSUPPORTED claim. The safe rewrite drops the
            unsupported claims and keeps the supported ones; if nothing
            is supported, the whole answer is replaced with an abstention.

The safe rewrite is the point of the library: a RAG app should never
show a number it cannot cite. Dropping a claim is always safer than
guessing, and the report says exactly what was dropped and why.
"""

from __future__ import annotations

from dataclasses import dataclass

from .verify import PARTIAL, SUPPORTED, UNSUPPORTED, ClaimVerdict

PASS = "PASS"
FLAG = "FLAG"
BLOCK = "BLOCK"

ABSTENTION = "I don't have enough information in the retrieved sources to answer that."


@dataclass
class Decision:
    decision: str
    safe_answer: str
    dropped_claims: list[str]
    flagged_claims: list[str]
    n_supported: int
    n_partial: int
    n_unsupported: int


class Policy:
    def __init__(self, abstention: str = ABSTENTION) -> None:
        self.abstention = abstention

    def decide(
        self, verdicts: list[ClaimVerdict], original_answer: str
    ) -> Decision:
        supported = [v for v in verdicts if v.verdict == SUPPORTED]
        partial = [v for v in verdicts if v.verdict == PARTIAL]
        unsupported = [v for v in verdicts if v.verdict == UNSUPPORTED]

        if unsupported:
            decision = BLOCK
        elif partial:
            decision = FLAG
        else:
            decision = PASS

        if decision == PASS:
            safe_answer = original_answer
            dropped: list[str] = []
            flagged: list[str] = []
        elif decision == FLAG:
            safe_answer = original_answer
            flagged = [v.claim.text for v in partial]
            for v in partial:
                safe_answer = safe_answer.replace(
                    v.claim.text, f"{v.claim.text} [unverified]"
                )
            dropped = []
        else:  # BLOCK
            dropped = [v.claim.text for v in unsupported]
            flagged = [v.claim.text for v in partial]
            kept = [v.claim.text for v in supported]
            if kept:
                safe_answer = " ".join(kept)
                if flagged:
                    safe_answer += (
                        " " + " ".join(f"{t} [unverified]" for t in flagged)
                    )
            else:
                safe_answer = self.abstention

        return Decision(
            decision=decision,
            safe_answer=safe_answer,
            dropped_claims=dropped,
            flagged_claims=flagged,
            n_supported=len(supported),
            n_partial=len(partial),
            n_unsupported=len(unsupported),
        )


def apply_policy(
    verdicts: list[ClaimVerdict], original_answer: str
) -> Decision:
    return Policy().decide(verdicts, original_answer)
