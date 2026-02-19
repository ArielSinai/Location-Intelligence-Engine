"""
Opportunity Analyzer.

Scores each H3 area cell on how good an opportunity it represents
for opening a specific type of business.

Scoring factors (all normalized to 0-1):
  - competition_density    : competitors per km²  → LOWER is better
  - diversity_score        : unique service types → HIGHER is better
  - similarity_saturation  : fraction of businesses semantically similar
                             to the query        → LOWER is better

Final opportunity_score = weighted combination of the three signals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import h3
import numpy as np


# ---------------------------------------------------------------------------
# Output shape
# ---------------------------------------------------------------------------

@dataclass
class AreaScore:
    h3_cell: str
    competition_count: int
    competition_density: float       # competitors per km²
    diversity_score: float           # 0-1
    similarity_saturation: float     # 0-1
    opportunity_score: float         # 0-1  (higher = better opportunity)


# ---------------------------------------------------------------------------
# Individual scoring functions
# ---------------------------------------------------------------------------

def compute_competition_density(count: int, cell_area_km2: float) -> float:
    """
    Competitors per km².

    Returns 0.0 when there are no competitors or the area is zero.
    """
    if count == 0 or cell_area_km2 == 0:
        return 0.0
    return float(count) / cell_area_km2


def compute_diversity_score(categories: list[str]) -> float:
    """
    Measure how diverse the services in an area are.

    Score = (unique categories - 1) / (total businesses - 1)
      → 0 when all businesses share the same category
      → 1 when every business has a unique category
      → 0 for empty or single-business areas
    """
    n = len(categories)
    if n <= 1:
        return 0.0
    unique = len(set(categories))
    return float(unique - 1) / float(n - 1)


def compute_similarity_saturation(
    query_emb: np.ndarray,
    embeddings: list[np.ndarray],
    threshold: float = 0.8,
) -> float:
    """
    Fraction of area embeddings whose cosine similarity to the query
    exceeds *threshold*.

    Returns 0.0 for an empty area.
    """
    if not embeddings:
        return 0.0

    # Normalise query
    q = query_emb.astype(np.float32)
    norm = np.linalg.norm(q)
    if norm > 0:
        q = q / norm

    above = 0
    for emb in embeddings:
        e = emb.astype(np.float32)
        e_norm = np.linalg.norm(e)
        if e_norm > 0:
            e = e / e_norm
        sim = float(np.dot(q, e))
        if sim >= threshold:
            above += 1

    return above / len(embeddings)


def compute_opportunity_score(
    competition_density: float,
    diversity_score: float,
    similarity_saturation: float,
    *,
    max_density: float = 10.0,
    w_competition: float = 0.5,
    w_diversity: float = 0.3,
    w_saturation: float = 0.2,
) -> float:
    """
    Weighted combination of the three signals → final score in [0, 1].

    competition_density  → inverted and normalised against max_density
    diversity_score      → used directly (already 0-1)
    similarity_saturation → inverted (high saturation = bad)
    """
    # Invert and clamp competition signal
    comp_score = max(0.0, 1.0 - min(competition_density, max_density) / max_density)

    # Invert saturation signal
    sat_score = 1.0 - similarity_saturation

    raw = (
        w_competition * comp_score
        + w_diversity  * diversity_score
        + w_saturation * sat_score
    )
    total_weight = w_competition + w_diversity + w_saturation
    return float(raw / total_weight)


# ---------------------------------------------------------------------------
# Per-cell scoring
# ---------------------------------------------------------------------------

def score_area(
    h3_cell: str,
    query_embedding: np.ndarray,
    area_embeddings: dict[str, np.ndarray],   # osm_id → embedding
    area_businesses: list[dict],
    query_category: str,
) -> AreaScore:
    """
    Compute a full AreaScore for a single H3 cell.

    Parameters
    ----------
    h3_cell          : H3 index of the cell being scored
    query_embedding  : embedding of the user's query text
    area_embeddings  : all embeddings keyed by osm_id
    area_businesses  : list of business dicts (must contain 'osm_id', 'category')
    query_category   : normalised category the user is looking for
    """
    # Competition: count businesses in the same category
    competition_count = sum(
        1 for b in area_businesses if b.get("category") == query_category
    )

    cell_area_km2 = h3.average_hexagon_area(
        h3.get_resolution(h3_cell), unit="km^2"
    )
    competition_density = compute_competition_density(competition_count, cell_area_km2)

    # Diversity: all categories present in the cell
    categories = [b.get("category", "") for b in area_businesses]
    diversity_score = compute_diversity_score(categories)

    # Similarity saturation: embeddings of all businesses in the cell
    emb_list = [
        area_embeddings[b["osm_id"]]
        for b in area_businesses
        if b.get("osm_id") in area_embeddings
    ]
    similarity_saturation = compute_similarity_saturation(query_embedding, emb_list)

    opportunity_score = compute_opportunity_score(
        competition_density,
        diversity_score,
        similarity_saturation,
    )

    return AreaScore(
        h3_cell=h3_cell,
        competition_count=competition_count,
        competition_density=competition_density,
        diversity_score=diversity_score,
        similarity_saturation=similarity_saturation,
        opportunity_score=opportunity_score,
    )


# ---------------------------------------------------------------------------
# Multi-cell ranking
# ---------------------------------------------------------------------------

def rank_areas(
    cells: list[str],
    query_embedding: np.ndarray,
    all_embeddings: dict[str, np.ndarray],
    get_businesses_fn: Callable[[str], list[dict]],
    query_category: str,
    top_k: int = 5,
) -> list[AreaScore]:
    """
    Score every cell in *cells* and return the top-k by opportunity_score.

    Parameters
    ----------
    cells             : H3 cell indices to evaluate
    query_embedding   : embedding of the user's query
    all_embeddings    : full embedding store (osm_id → array)
    get_businesses_fn : callable(h3_cell) → list[dict]
                        (wraps either graph queries or cached data)
    query_category    : normalised category string
    top_k             : how many results to return
    """
    if not cells:
        return []

    scores = [
        score_area(
            cell,
            query_embedding,
            all_embeddings,
            get_businesses_fn(cell),
            query_category,
        )
        for cell in cells
    ]

    scores.sort(key=lambda s: s.opportunity_score, reverse=True)
    return scores[: min(top_k, len(scores))]
