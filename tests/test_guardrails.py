"""Unit tests for the hallucination guardrails package."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from guardrails.claims import split_claims  # noqa: E402
from guardrails.confidence import claim_confidence  # noqa: E402
from guardrails.policy import Policy  # noqa: E402
from guardrails.text import (  # noqa: E402
    TfIdf,
    extract_entities,
    extract_numbers,
    split_sentences,
    tokenize,
)
from guardrails.verify import (  # noqa: E402
    PARTIAL,
    SUPPORTED,
    UNSUPPORTED,
    SourcePassage,
    Verifier,
)


def test_tokenize_drops_stopwords_and_lowercases():
    assert tokenize("The Pro plan costs $199 per month") == [
        "pro",
        "plan",
        "costs",
        "199",
        "month",
    ]


def test_split_sentences_keeps_punctuation():
    assert split_sentences("First claim here. Second claim there!") == [
        "First claim here.",
        "Second claim there!",
    ]


def test_split_claims_drops_short_fragments():
    claims = split_claims("Sure! The Starter plan costs $49 per month.")
    assert [c.text for c in claims] == ["The Starter plan costs $49 per month."]


def test_extract_numbers_normalizes():
    assert extract_numbers("Costs $1,250.50 and 15% off in 2026.") == [
        "1250.50",
        "15",
        "2026",
    ]


def test_extract_entities_drops_leading_the():
    assert extract_entities("The Starter plan is cheap.") == ["Starter"]
    assert extract_entities("Beta Ltd uses Northwind daily.") == [
        "Beta Ltd",
        "Northwind",
    ]


def test_tfidf_prefers_the_matching_doc():
    tfidf = TfIdf(["cats and dogs are pets", "quantum computing qubits"])
    assert tfidf.similarity("cats are pets", 0) > tfidf.similarity(
        "cats are pets", 1
    )
    assert tfidf.similarity("quantum qubits", 1) > 0.3


def _verifier():
    passages = [
        SourcePassage(
            "pricing",
            "The Starter plan costs $49 per month and the Pro plan costs "
            "$199 per month.",
            as_of="2026-01-15",
        ),
        SourcePassage(
            "stale",
            "The Starter plan costs $29 per month.",
            as_of="2024-03-01",
        ),
    ]
    return Verifier(passages)


def test_supported_clean_claim():
    v = _verifier().verify_claim(split_claims("The Pro plan costs $199 per month.")[0])
    assert v.verdict == SUPPORTED
    assert v.best_passage_id == "pricing"
    assert v.numbers_missing == [] and v.entities_missing == []


def test_wrong_number_is_unsupported_despite_high_similarity():
    # Similarity is high (same words as the true price) — only the number
    # check can catch this. It must be UNSUPPORTED, not PARTIAL.
    v = _verifier().verify_claim(split_claims("The Pro plan costs $149 per month.")[0])
    assert v.verdict == UNSUPPORTED
    assert v.numbers_missing == ["149"]
    assert v.best_score > 0.5


def test_invented_feature_is_unsupported():
    v = _verifier().verify_claim(
        split_claims("The platform offers AI-powered fraud detection.")[0]
    )
    assert v.verdict == UNSUPPORTED


def test_stale_doc_supports_stale_price_without_cutoff():
    v = _verifier().verify_claim(split_claims("The Starter plan costs $29 per month.")[0])
    assert v.verdict == SUPPORTED
    assert v.best_passage_id == "stale"


def test_freshness_cutoff_catches_stale_price():
    passages = _verifier().passages
    fresh = Verifier(passages, freshness_cutoff="2025-01-01")
    v = fresh.verify_claim(split_claims("The Starter plan costs $29 per month.")[0])
    assert v.verdict == UNSUPPORTED
    assert v.numbers_missing == ["29"]


def test_vague_claim_is_at_most_partial():
    v = _verifier().verify_claim(
        split_claims("This is a popular tool with many happy customers.")[0]
    )
    assert v.verdict in (PARTIAL, UNSUPPORTED)


def test_confidence_penalizes_wrong_numbers():
    assert claim_confidence(0.9, True, True, True, True) > 0.9
    assert claim_confidence(0.9, False, True, True, True) < 0.75
    assert claim_confidence(0.0, True, True, False, False) == 0.5


def test_policy_pass_flag_block():
    policy = Policy()
    clean = _verifier().verify_answer("The Pro plan costs $199 per month.")
    d = policy.decide(clean, "The Pro plan costs $199 per month.")
    assert d.decision == "PASS"
    assert d.safe_answer == "The Pro plan costs $199 per month."

    mixed = _verifier().verify_answer(
        "The Pro plan costs $199 per month. The Pro plan costs $149 per month."
    )
    d = policy.decide(
        mixed,
        "The Pro plan costs $199 per month. The Pro plan costs $149 per month.",
    )
    assert d.decision == "BLOCK"
    assert "$149" not in d.safe_answer
    assert "$199" in d.safe_answer
    assert d.dropped_claims == ["The Pro plan costs $149 per month."]


def test_policy_block_all_unsupported_abstains():
    policy = Policy()
    verdicts = _verifier().verify_answer("The platform offers AI-powered fraud detection.")
    d = policy.decide(verdicts, "The platform offers AI-powered fraud detection.")
    assert d.decision == "BLOCK"
    assert d.safe_answer.startswith("I don't have enough information")
