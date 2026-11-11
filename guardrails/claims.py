"""Claim extraction: split an answer into checkable sentence-level claims."""

from __future__ import annotations

from dataclasses import dataclass

from .text import split_sentences


@dataclass(frozen=True)
class Claim:
    text: str
    index: int  # position within the answer (0-based)


def split_claims(answer: str) -> list[Claim]:
    """One claim per sentence. Short fragments (< 3 words) are dropped:
    greetings and filler ("Sure!", "Here you go:") are not factual claims.
    """
    claims: list[Claim] = []
    for i, sentence in enumerate(split_sentences(answer)):
        if len(sentence.split()) < 3:
            continue
        claims.append(Claim(text=sentence, index=i))
    return claims
