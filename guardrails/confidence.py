"""Confidence scoring for a single claim verdict.

Confidence is a weighted blend of the three signals the verifier already
measured, so it is explainable rather than a black-box number:

    0.50 * retrieval similarity (0-1)
    0.30 * number agreement (1 if all numbers match or claim has none)
    0.20 * entity agreement (1 if all entities match or claim has none)

A claim with no numbers and no entities scores purely on similarity;
a claim whose price is wrong can never score above 0.70.
"""


def claim_confidence(
    similarity: float,
    numbers_ok: bool,
    entities_ok: bool,
    has_numbers: bool,
    has_entities: bool,
) -> float:
    number_score = 1.0 if (not has_numbers or numbers_ok) else 0.0
    entity_score = 1.0 if (not has_entities or entities_ok) else 0.0
    score = 0.50 * similarity + 0.30 * number_score + 0.20 * entity_score
    return max(0.0, min(1.0, score))
