"""
Opportunity Analyzer.

Scores each H3 area cell on how good an opportunity it represents
for opening a specific type of business.

Scoring factors (all normalized to 0-1):
  - competition_density    : competitors per km²  → LOWER is better
  - demand_proxy           : total businesses / max as foot-traffic signal
                             → HIGHER is better (busier area = more customers)
  - diversity_score        : unique service types → HIGHER is better
  - similarity_saturation  : fraction of businesses semantically similar
                             to the query        → LOWER is better

Final opportunity_score = weighted combination of the four signals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import h3
import numpy as np

from data_collector.normalizer import SUPPORT_CATEGORIES


# ---------------------------------------------------------------------------
# Output shape
# ---------------------------------------------------------------------------

@dataclass
class AreaScore:
    h3_cell: str
    competition_count: int
    competition_density: float       # competitors per km²
    demand_proxy: float              # 0-1, foot-traffic / activity signal
    diversity_score: float           # 0-1
    similarity_saturation: float     # 0-1
    opportunity_score: float         # 0-1  (higher = better opportunity)
    support_poi_summary: dict = field(default_factory=dict)  # category → count


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


def compute_demand_proxy(total_business_count: int, max_businesses: int = 50) -> float:
    """
    Estimate demand potential from the overall activity level of an area.

    A cell with many businesses of any type has higher foot traffic, which
    translates to more potential customers for a new service.

    Score = min(total_businesses, max_businesses) / max_businesses
      → 0.0 for an empty cell (no foot traffic)
      → 1.0 when the cell meets or exceeds max_businesses

    Parameters
    ----------
    total_business_count : total number of businesses in the cell (all categories)
    max_businesses       : upper bound used for normalisation (default 50)
    """
    if total_business_count <= 0 or max_businesses <= 0:
        return 0.0
    return float(min(total_business_count, max_businesses)) / float(max_businesses)


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
    demand_proxy: float,
    diversity_score: float,
    similarity_saturation: float,
    *,
    max_density: float = 10.0,
    w_competition: float = 0.40,
    w_demand: float = 0.20,
    w_diversity: float = 0.25,
    w_saturation: float = 0.15,
) -> float:
    """
    Weighted combination of the four signals → final score in [0, 1].

    competition_density   → inverted and normalised against max_density
    demand_proxy          → used directly (already 0-1, higher = better)
    diversity_score       → used directly (already 0-1)
    similarity_saturation → inverted (high saturation = bad)
    """
    # Invert and clamp competition signal
    comp_score = max(0.0, 1.0 - min(competition_density, max_density) / max_density)

    # Invert saturation signal
    sat_score = 1.0 - similarity_saturation

    raw = (
        w_competition * comp_score
        + w_demand     * demand_proxy
        + w_diversity  * diversity_score
        + w_saturation * sat_score
    )
    total_weight = w_competition + w_demand + w_diversity + w_saturation
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
    query_keywords: frozenset[str] = frozenset(),
    category_confidence: float = 0.0,
) -> AreaScore:
    """
    Compute a full AreaScore for a single H3 cell.

    Parameters
    ----------
    h3_cell             : H3 index of the cell being scored
    query_embedding     : embedding of the user's query text
    area_embeddings     : all embeddings keyed by osm_id
    area_businesses     : list of business dicts (must contain 'osm_id', 'category')
    query_category      : normalised category the user is looking for
    query_keywords      : core keywords from the stripped query (e.g. {"pilates"}).
                          A business whose name contains any of these is always
                          counted as a competitor, regardless of embedding score.
    category_confidence : how confidently the query mapped to query_category (0–1).
                          High confidence (≥ 0.70) → "exact mode": exact category
                          match is the primary competition signal and the embedding
                          threshold is raised to 0.75 to catch only near-identical
                          businesses that may have a different OSM tag.
                          Low confidence → "fuzzy mode": embedding similarity ≥ 0.45
                          is the primary signal (previous behaviour).
    """
    _SATURATION_THRESHOLD = 0.80  # floor for market-saturation signal
    _HIGH_CONFIDENCE      = 0.70  # above this: switch to exact-match competition mode

    is_exact_mode         = category_confidence >= _HIGH_CONFIDENCE
    _COMPETITION_THRESHOLD = 0.75 if is_exact_mode else 0.45

    # Normalise the query embedding once; reuse for both signals.
    q = query_embedding.astype(np.float32)
    q_norm = np.linalg.norm(q)
    if q_norm > 0:
        q = q / q_norm

    competition_count = 0
    saturation_above  = 0
    scored_count      = 0

    for b in area_businesses:
        name      = (b.get("name") or "").lower()
        osm_id    = b.get("osm_id")
        exact_cat = b.get("category") == query_category

        # Keyword match: business name contains a core query word → definite competitor.
        keyword_hit = any(kw in name for kw in query_keywords)

        if osm_id in area_embeddings:
            e = area_embeddings[osm_id].astype(np.float32)
            e_norm = np.linalg.norm(e)
            if e_norm > 0:
                e = e / e_norm
            sim = float(np.dot(q, e))
            scored_count += 1

            if is_exact_mode:
                # Well-defined category: exact OSM category match is the primary
                # signal.  High-sim embedding catches businesses tagged differently
                # but operating in the same space (e.g. "juice bar" vs "cafe").
                if keyword_hit or exact_cat or sim >= _COMPETITION_THRESHOLD:
                    competition_count += 1
            else:
                # Fuzzy mode: rely on embedding similarity (original behaviour).
                if keyword_hit or sim >= _COMPETITION_THRESHOLD:
                    competition_count += 1

            if sim >= _SATURATION_THRESHOLD:
                saturation_above += 1
        else:
            # Fallback for businesses without embeddings (both modes).
            if keyword_hit or exact_cat:
                competition_count += 1

    similarity_saturation = saturation_above / scored_count if scored_count > 0 else 0.0

    cell_area_km2 = h3.average_hexagon_area(
        h3.get_resolution(h3_cell), unit="km^2"
    )
    competition_density = compute_competition_density(competition_count, cell_area_km2)

    # Demand proxy: total business count as foot-traffic signal
    demand_proxy = compute_demand_proxy(len(area_businesses))

    # Diversity: all categories present in the cell
    categories = [b.get("category", "") for b in area_businesses]
    diversity_score = compute_diversity_score(categories)

    opportunity_score = compute_opportunity_score(
        competition_density,
        demand_proxy,
        diversity_score,
        similarity_saturation,
    )

    # Support POI summary: demand anchors present in the cell
    support_poi_summary: dict[str, int] = {}
    for b in area_businesses:
        cat = b.get("category", "")
        if cat in SUPPORT_CATEGORIES:
            support_poi_summary[cat] = support_poi_summary.get(cat, 0) + 1

    return AreaScore(
        h3_cell=h3_cell,
        competition_count=competition_count,
        competition_density=competition_density,
        demand_proxy=demand_proxy,
        diversity_score=diversity_score,
        similarity_saturation=similarity_saturation,
        opportunity_score=opportunity_score,
        support_poi_summary=support_poi_summary,
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
    query_keywords: frozenset[str] = frozenset(),
    category_confidence: float = 0.0,
) -> list[AreaScore]:
    """
    Score every cell in *cells* and return the top-k by opportunity_score.

    Parameters
    ----------
    cells               : H3 cell indices to evaluate
    query_embedding     : embedding of the user's query
    all_embeddings      : full embedding store (osm_id → array)
    get_businesses_fn   : callable(h3_cell) → list[dict]
                          (wraps either graph queries or cached data)
    query_category      : normalised category string
    top_k               : how many results to return
    query_keywords      : core keywords from the stripped query used to guarantee
                          name-matched businesses are always counted as competitors
    category_confidence : forwarded to score_area — see its docstring for details
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
            query_keywords,
            category_confidence,
        )
        for cell in cells
    ]

    scores.sort(key=lambda s: s.opportunity_score, reverse=True)
    return scores[: min(top_k, len(scores))]
