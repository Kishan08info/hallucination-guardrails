"""Text helpers: tokenization, sentence splitting, TF-IDF cosine similarity.

All standard library. Tokenization is deliberately simple and documented:
lowercased alphanumeric words, English stopwords removed for similarity,
numbers kept as tokens so prices/dates participate in matching.
"""

from __future__ import annotations

import math
import re
from collections import Counter

STOPWORDS = frozenset(
    """
    a an the and or but if then else when while of to in on at for with by from
    as is are was were be been being has have had do does did will would shall
    should can could may might must i you he she it we they this that these
    those there here what which who whom whose how why not no yes so very just
    also than too more most other some any each every all both into over under
    between through during before after above below up down out off again once
    per via its it's their his her our your my me him us them
    """.split()
)

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")
_WORD_RE = re.compile(r"[a-z0-9]+(?:[.,][0-9]+)?")
_NUMBER_RE = re.compile(r"\$?\d[\d,]*(?:\.\d+)?%?")
_ENTITY_RE = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b|\b[A-Z][a-z]{2,}\b")


def tokenize(text: str, keep_stopwords: bool = False) -> list[str]:
    """Lowercased word tokens; stopwords dropped unless asked for."""
    words = _WORD_RE.findall(text.lower())
    if keep_stopwords:
        return words
    return [w for w in words if w not in STOPWORDS]


def split_sentences(text: str) -> list[str]:
    """Split on sentence-ending punctuation. Keeps the punctuation."""
    text = " ".join(text.split())
    if not text:
        return []
    parts = _SENTENCE_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


def extract_numbers(text: str) -> list[str]:
    """Numbers as they appear, normalized ($, commas and % stripped)."""
    out = []
    for m in _NUMBER_RE.findall(text):
        norm = m.replace("$", "").replace(",", "").rstrip("%")
        # Drop bare years from the *number* check only if they look like years?
        # No: years are facts too (e.g. "launched in 2024"). Keep them.
        out.append(norm)
    return out


def extract_entities(text: str) -> list[str]:
    """Capitalized words/phrases (proper names, products, companies).

    The article "The" is never part of a name: "The Starter plan" yields
    entity "Starter", not "The Starter". A single capitalized word at
    position 0 is skipped (capitalized by grammar, not necessarily a name).
    """
    entities: list[str] = []
    for m in _ENTITY_RE.finditer(text):
        phrase = m.group(0)
        was_multiword = " " in phrase
        if phrase.startswith("The "):
            phrase = phrase[4:]
            if not phrase:
                continue
        if m.start() == 0 and not was_multiword:
            continue  # sentence-initial word, not necessarily a name
        entities.append(phrase)
    return entities


class TfIdf:
    """Tiny TF-IDF over a fixed corpus of passages.

    IDF is computed over the corpus passages only. A query (claim) is
    scored against a passage by cosine similarity of TF-IDF vectors.
    """

    def __init__(self, documents: list[str]) -> None:
        self.n_docs = len(documents)
        df: Counter[str] = Counter()
        self.doc_vectors: list[dict[str, float]] = []
        self.doc_norms: list[float] = []
        for doc in documents:
            tokens = tokenize(doc)
            counts = Counter(tokens)
            for term in counts:
                df[term] += 1
            self.doc_vectors.append(dict(counts))
        self.idf = {
            term: math.log((1 + self.n_docs) / (1 + freq)) + 1.0
            for term, freq in df.items()
        }
        # Weight each document vector by IDF and precompute its norm.
        self.doc_vectors = [
            {t: c * self.idf.get(t, 1.0) for t, c in vec.items()}
            for vec in self.doc_vectors
        ]
        self.doc_norms = [
            math.sqrt(sum(v * v for v in vec.values())) or 1.0
            for vec in self.doc_vectors
        ]

    def similarity(self, query: str, doc_index: int) -> float:
        """Cosine similarity between a query string and one corpus doc."""
        q_tokens = tokenize(query)
        if not q_tokens:
            return 0.0
        q_counts = Counter(q_tokens)
        q_vec = {t: c * self.idf.get(t, 1.0) for t, c in q_counts.items()}
        q_norm = math.sqrt(sum(v * v for v in q_vec.values())) or 1.0
        doc_vec = self.doc_vectors[doc_index]
        dot = sum(q_vec.get(t, 0.0) * w for t, w in doc_vec.items())
        return dot / (q_norm * self.doc_norms[doc_index])
