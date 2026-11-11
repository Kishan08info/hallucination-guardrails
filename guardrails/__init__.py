"""Hallucination guardrails for RAG / LLM answers.

Drop-in, standard-library-only checks that verify an LLM answer against
the retrieved source passages *before* it reaches the user:

- claim extraction (sentence-level claims)
- citation verification (TF-IDF cosine similarity + number/entity agreement)
- confidence scoring per claim
- policy decision (PASS / FLAG / BLOCK) + safe rewrite that keeps only
  supported claims

Nothing here calls an LLM or the network. All data in the demo is
synthetic (fictional company "Northwind Analytics").
"""

from .claims import split_claims
from .verify import ClaimVerdict, Verifier, verify_answer
from .confidence import claim_confidence
from .policy import Decision, Policy, apply_policy

__all__ = [
    "split_claims",
    "ClaimVerdict",
    "Verifier",
    "verify_answer",
    "claim_confidence",
    "Decision",
    "Policy",
    "apply_policy",
]

__version__ = "0.1.0"
