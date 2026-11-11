"""Citation verification: does each claim trace to a retrieved passage?

A claim is checked against every source passage and matched to the best
one by TF-IDF cosine similarity. The verdict then combines that score
with two hard factual checks:

- numbers: every number in the claim ($49, 2024, 15%) must appear in the
  best passage (normalized: $, commas, % stripped). A wrong price is the
  classic RAG hallucination and similarity alone will not catch it —
  "$149/mo" and "$199/mo" share almost every token.
- entities: every capitalized name in the claim must appear in the
  passage. An invented customer or feature name fails here.

Verdict thresholds (defaults, tunable on the Verifier):
    SUPPORTED    similarity >= 0.30, coverage >= 0.60, and all numbers
                 + entities agree
    PARTIAL      similarity >= 0.15, coverage >= 0.40, and no hard
                 number/entity disagreement (claim is on-topic but not
                 fully backed)
    UNSUPPORTED  otherwise — including any claim whose numbers or
                 entities disagree with its best passage, however
                 similar the wording is. A wrong price is a hallucination,
                 not partial support.

Coverage is the fraction of the claim's content tokens that appear in
the best passage at all. It catches invented-feature claims that share
generic words ("offers", "real-time") with a passage but add new
content words the passage never mentions.

Known limit, stated honestly: bag-of-words checks cannot catch
attribution swaps — "Meridian Health uses Northwind for weekly sales
reporting" scores as SUPPORTED because every word and entity appears
in the passage, even though the passage attributes that use to Beta Ltd.
The demo includes exactly this case and reports it as a miss.

Freshness: passages carry an ``as_of`` date (ISO, e.g. "2026-01-15").
Passing ``freshness_cutoff`` drops older passages from the corpus, so a
stale pricing doc cannot "support" a wrong price. The demo shows both
modes: with the stale doc included, the stale-price claim passes (a real
failure mode); with a cutoff, it is caught.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .claims import Claim, split_claims
from .text import TfIdf, extract_entities, extract_numbers, tokenize

SUPPORTED = "SUPPORTED"
PARTIAL = "PARTIAL"
UNSUPPORTED = "UNSUPPORTED"


@dataclass
class ClaimVerdict:
    claim: Claim
    verdict: str
    best_passage_id: str | None
    best_score: float
    coverage: float = 0.0
    numbers_in_claim: list[str] = field(default_factory=list)
    numbers_missing: list[str] = field(default_factory=list)
    entities_in_claim: list[str] = field(default_factory=list)
    entities_missing: list[str] = field(default_factory=list)
    confidence: float = 0.0

    @property
    def numbers_ok(self) -> bool:
        return not self.numbers_missing

    @property
    def entities_ok(self) -> bool:
        return not self.entities_missing


@dataclass
class SourcePassage:
    passage_id: str
    text: str
    as_of: str | None = None  # ISO date, e.g. "2026-01-15"


class Verifier:
    def __init__(
        self,
        passages: list[SourcePassage],
        support_threshold: float = 0.30,
        partial_threshold: float = 0.15,
        support_coverage: float = 0.60,
        partial_coverage: float = 0.40,
        freshness_cutoff: str | None = None,
    ) -> None:
        """freshness_cutoff: ISO date; passages with as_of before it are
        excluded from the corpus (stale docs cannot support claims)."""
        if freshness_cutoff:
            self.passages = [
                p
                for p in passages
                if p.as_of is None or p.as_of >= freshness_cutoff
            ]
        else:
            self.passages = list(passages)
        self.support_threshold = support_threshold
        self.partial_threshold = partial_threshold
        self.support_coverage = support_coverage
        self.partial_coverage = partial_coverage
        self._tfidf = TfIdf([p.text for p in self.passages])

    def verify_claim(self, claim: Claim) -> ClaimVerdict:
        best_score = 0.0
        best_idx: int | None = None
        for i in range(len(self.passages)):
            score = self._tfidf.similarity(claim.text, i)
            if score > best_score:
                best_score = score
                best_idx = i
        best = self.passages[best_idx] if best_idx is not None else None

        numbers = extract_numbers(claim.text)
        entities = extract_entities(claim.text)
        if best is not None:
            passage_numbers = set(extract_numbers(best.text))
            passage_text = best.text
            missing_numbers = [n for n in numbers if n not in passage_numbers]
            missing_entities = [e for e in entities if e not in passage_text]
            claim_tokens = set(tokenize(claim.text))
            passage_tokens = set(tokenize(best.text))
            coverage = (
                len(claim_tokens & passage_tokens) / len(claim_tokens)
                if claim_tokens
                else 0.0
            )
        else:
            missing_numbers, missing_entities = list(numbers), list(entities)
            coverage = 0.0

        has_numbers = bool(numbers)
        has_entities = bool(entities)
        hard_disagreement = (has_numbers and missing_numbers) or (
            has_entities and missing_entities
        )

        if (
            best is not None
            and best_score >= self.support_threshold
            and coverage >= self.support_coverage
            and not hard_disagreement
        ):
            verdict = SUPPORTED
        elif (
            best is not None
            and best_score >= self.partial_threshold
            and coverage >= self.partial_coverage
            and not hard_disagreement
        ):
            verdict = PARTIAL
        else:
            verdict = UNSUPPORTED

        from .confidence import claim_confidence

        confidence = claim_confidence(
            best_score,
            numbers_ok=not missing_numbers,
            entities_ok=not missing_entities,
            has_numbers=bool(numbers),
            has_entities=bool(entities),
        )
        return ClaimVerdict(
            claim=claim,
            verdict=verdict,
            best_passage_id=best.passage_id if best else None,
            best_score=round(best_score, 4),
            coverage=round(coverage, 4),
            numbers_in_claim=numbers,
            numbers_missing=missing_numbers,
            entities_in_claim=entities,
            entities_missing=missing_entities,
            confidence=round(confidence, 4),
        )

    def verify_answer(self, answer: str) -> list[ClaimVerdict]:
        return [self.verify_claim(c) for c in split_claims(answer)]


def verify_answer(
    answer: str,
    passages: list[SourcePassage],
    **kwargs,
) -> list[ClaimVerdict]:
    """Convenience wrapper: build a Verifier and check a whole answer."""
    return Verifier(passages, **kwargs).verify_answer(answer)
