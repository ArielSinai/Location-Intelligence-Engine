"""
Recommendations router.

POST /recommendations
  → embeds the query, finds H3 cells in the requested bbox,
    scores each cell, and returns ranked recommendations.
"""

from __future__ import annotations

import logging
import re

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field, field_validator
import h3
import numpy as np

from analyzer.opportunity import rank_areas, AreaScore
from embeddings.generator import load_embeddings, EmbeddingGenerator, match_category_by_embedding
from graph.queries import get_area_businesses
from api.dependencies import get_embedding_generator, get_driver
from config import settings

logger = logging.getLogger(__name__)

router = APIRouter()

# Generic venue-type words that describe *where* a business operates but carry
# little signal about *what* it does.  Stripping them before category matching
# reduces the bias they introduce (e.g. "pilates studio" vs "dance studio"
# both share "studio", making the match drift toward dance over fitness).
_VENUE_WORDS = re.compile(
    r"\b(studio|center|centre|school|shop|store|salon|hall|club|bar|court|space|room|place|house)\b",
    re.IGNORECASE,
)


def _strip_venue_words(text: str) -> str:
    """Remove generic venue-type suffixes; fall back to original if fully stripped."""
    result = _VENUE_WORDS.sub("", text).strip()
    return result if result else text


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class BBoxModel(BaseModel):
    """Geographic bounding box defined by its four edges (decimal degrees)."""
    south: float = Field(..., description="Southern latitude boundary (e.g. 32.03)")
    west: float  = Field(..., description="Western longitude boundary (e.g. 34.74)")
    north: float = Field(..., description="Northern latitude boundary (e.g. 32.13)")
    east: float  = Field(..., description="Eastern longitude boundary (e.g. 34.84)")

    @field_validator("south", "north")
    @classmethod
    def valid_latitude(cls, v):
        if not -90 <= v <= 90:
            raise ValueError("latitude must be between -90 and 90")
        return v

    @field_validator("west", "east")
    @classmethod
    def valid_longitude(cls, v):
        if not -180 <= v <= 180:
            raise ValueError("longitude must be between -180 and 180")
        return v

    @field_validator("north")
    @classmethod
    def north_must_exceed_south(cls, v, info):
        south = info.data.get("south")
        if south is not None and v <= south:
            raise ValueError("north must be greater than south")
        return v


class RecommendationRequest(BaseModel):
    """Request body for the /recommendations endpoint."""
    bbox: BBoxModel = Field(..., description="The geographic area to analyse.")
    query: str = Field(
        ...,
        min_length=1,
        description=(
            "The type of business you want to open. "
            "Can be a specific type (e.g. 'sushi restaurant', 'pilates studio') "
            "or a general category (e.g. 'cafe', 'pharmacy'). "
            "The engine matches this to the closest known category using AI embeddings."
        ),
    )
    top_k: int = Field(
        default=5,
        ge=1,
        description="Number of top-ranked locations to return. Defaults to 5.",
    )


class RecommendationItem(BaseModel):
    h3_cell: str = Field(..., description="H3 hexagonal cell index identifying the recommended location.")
    rank: int = Field(..., description="Rank of this location (1 = best opportunity).")
    opportunity_score: float = Field(..., description="Overall opportunity score from 0 to 1. Higher is better.")
    competition_count: int = Field(..., description="Number of similar businesses already operating in this cell.")
    competition_density: float = Field(..., description="Competitors per km² in this cell.")
    demand_proxy: float = Field(..., description="Estimated foot traffic / activity level (0–1). Higher means more potential customers.")
    diversity_score: float = Field(..., description="How varied the mix of businesses in the area is (0–1). Higher means more complementary services nearby.")
    similarity_saturation: float = Field(..., description="Fraction of businesses semantically similar to your query (0–1). Lower is better — less market saturation.")
    support_poi_summary: dict[str, int] = Field(..., description="Nearby demand anchors (e.g. schools, offices, malls) and their counts.")
    explanation: str = Field(..., description="Human-readable summary of why this location was recommended.")


class RecommendationResponse(BaseModel):
    query: str = Field(..., description="The original query as submitted.")
    matched_category: str = Field(..., description="The business category the query was matched to in the dataset.")
    recommendations: list[RecommendationItem] = Field(..., description="Ranked list of recommended locations.")


class HeatmapRequest(BaseModel):
    """Request body for the /heatmap endpoint."""
    bbox: BBoxModel = Field(..., description="The geographic area to analyse.")
    query: str = Field(
        ...,
        min_length=1,
        description="The type of business you want to open (e.g. 'cafe', 'gym', 'pharmacy').",
    )


class HeatmapCell(BaseModel):
    h3_cell: str = Field(..., description="H3 hexagonal cell index.")
    lat: float = Field(..., description="Latitude of the cell centre.")
    lng: float = Field(..., description="Longitude of the cell centre.")
    opportunity_score: float = Field(..., description="Opportunity score for this cell (0–1).")
    competition_density: float = Field(..., description="Competitors per km² in this cell.")


class HeatmapResponse(BaseModel):
    query: str = Field(..., description="The original query as submitted.")
    matched_category: str = Field(..., description="The business category the query was matched to.")
    cells: list[HeatmapCell] = Field(..., description="All scored cells in the bbox, suitable for map rendering.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cells_in_bbox(south: float, west: float, north: float, east: float) -> list[str]:
    """Return all H3 cells at the configured resolution that cover the bbox."""
    poly = h3.LatLngPoly([
        (south, west),
        (north, west),
        (north, east),
        (south, east),
    ])
    return list(h3.h3shape_to_cells(poly, settings.h3_resolution))


def _resolve_category(
    query: str,
    cells: list[str],
    driver,
    generator: EmbeddingGenerator,
) -> tuple[str, float]:
    """
    Resolve the user query to the best matching stored category and a
    confidence score (0–1) indicating how specific the match is.

    Strategy
    --------
    1. Collect every unique category present in the requested bbox cells.
    2. If the lowercased query exactly matches a stored category name →
       return it immediately with confidence 1.0 (well-defined query).
    3. Otherwise strip generic venue-type words and use embedding similarity
       to find the closest category; the cosine score becomes the confidence.
    4. Fall back to the lowercased query string with confidence 0.0 when no
       categories exist in the graph yet.

    Returns
    -------
    (matched_category, confidence)
    """
    seen: set[str] = set()
    for cell in cells:
        for biz in get_area_businesses(driver, cell):
            cat = biz.get("category", "")
            if cat:
                seen.add(cat)

    category_names = list(seen)
    query_lower = query.strip().lower()

    if not category_names:
        return query_lower, 0.0

    # --- Step 1: exact text match (highest confidence) ---
    for cat in category_names:
        if cat.lower() == query_lower:
            return cat, 1.0

    # --- Step 2: embedding similarity match ---
    # Strip venue-type words before comparing so the meaningful business-type
    # word drives the match (e.g. "pilates" rather than "studio").
    stripped_query = _strip_venue_words(query)
    stripped_query_emb = generator.embed_query(stripped_query)
    stripped_categories = [_strip_venue_words(c) for c in category_names]

    matched, confidence = match_category_by_embedding(
        stripped_query_emb,
        category_names,
        generator,
        encoding_texts=stripped_categories,
    )
    return (matched if matched else query_lower), confidence


def _generate_explanation(
    score: AreaScore,
    query_category: str,
    avg_competition: float,
) -> str:
    """Build a human-readable explanation for why an area is recommended."""
    parts: list[str] = []

    if score.competition_count == 0:
        parts.append(f"No direct {query_category} competitors found in this area.")
    elif avg_competition > 0 and score.competition_count < avg_competition:
        pct_fewer = (avg_competition - score.competition_count) / avg_competition * 100
        parts.append(
            f"Found {score.competition_count} similar business(es) "
            f"({score.competition_density:.1f} per km²) — "
            f"{pct_fewer:.0f}% fewer than the area average."
        )
    else:
        parts.append(
            f"Found {score.competition_count} similar business(es) "
            f"({score.competition_density:.1f} per km²)."
        )

    if score.demand_proxy > 0.6:
        parts.append("High activity area — strong foot traffic and demand potential.")
    elif score.demand_proxy < 0.2:
        parts.append("Low activity area — limited existing foot traffic.")

    if score.diversity_score > 0.6:
        parts.append("The area has a diverse mix of complementary services nearby.")
    elif score.diversity_score < 0.3:
        parts.append("Limited service variety in the immediate area.")

    if score.similarity_saturation < 0.2:
        parts.append("Low semantic overlap — few conceptually similar businesses present.")
    elif score.similarity_saturation > 0.7:
        parts.append("High semantic saturation — many similar concepts already operating here.")

    if score.support_poi_summary:
        anchors = [
            f"{count} {cat}{'s' if count > 1 else ''}"
            for cat, count in sorted(score.support_poi_summary.items())
        ]
        parts.append(
            f"Located near {', '.join(anchors)}, "
            "which increases potential customer traffic."
        )

    parts.append(f"Overall opportunity score: {score.opportunity_score:.0%}.")
    return " ".join(parts)


# Loaded from disk once and reused for every subsequent request.
_embeddings_cache: tuple[list[str], np.ndarray] | None = None


def _load_embeddings_safe() -> tuple[list[str], np.ndarray]:
    """
    Load pre-computed business embeddings, caching the result after the
    first successful load. Returns empty structures if the file does not
    exist yet.
    """
    global _embeddings_cache
    if _embeddings_cache is not None:
        return _embeddings_cache

    path = settings.embeddings_dir / "embeddings.npz"
    try:
        _embeddings_cache = load_embeddings(path)
        return _embeddings_cache
    except Exception as e:
        logger.warning("Could not load embeddings from %s: %s", path, e)
        return [], np.empty((0, 384), dtype=np.float32)


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@router.post(
    "/recommendations",
    response_model=RecommendationResponse,
    summary="Get top location recommendations",
    description=(
        "Given a geographic bounding box and a business type, returns the top-k H3 cells "
        "ranked by opportunity score — a weighted combination of low competition, high foot "
        "traffic, area diversity, and semantic market saturation.\n\n"
        "**Example:** find the best 5 areas to open a bakery in Tel Aviv:\n"
        "```json\n"
        '{"bbox": {"south": 32.03, "west": 34.74, "north": 32.13, "east": 34.84}, '
        '"query": "bakery", "top_k": 5}\n'
        "```"
    ),
)
def get_recommendations(
    request: RecommendationRequest,
    generator: EmbeddingGenerator = Depends(get_embedding_generator),
    driver=Depends(get_driver),
) -> RecommendationResponse:
    bbox = request.bbox

    # 1. Embed the user's query
    query_emb = generator.embed_query(request.query)

    # 2. Find all H3 cells inside the requested area
    cells = _cells_in_bbox(bbox.south, bbox.west, bbox.north, bbox.east)

    # 3. Resolve matched category and how confidently it was matched
    matched_category, category_confidence = _resolve_category(request.query, cells, driver, generator)

    # 4. Load pre-computed business embeddings
    osm_ids, emb_matrix = _load_embeddings_safe()
    all_embeddings: dict[str, np.ndarray] = {
        oid: emb_matrix[i] for i, oid in enumerate(osm_ids)
    }

    # 5. Score all cells, then compute area-wide average for explanation context
    query_keywords = frozenset(_strip_venue_words(request.query).lower().split())

    def get_businesses(cell: str) -> list[dict]:
        return get_area_businesses(driver, cell)

    all_scores = rank_areas(
        cells,
        query_emb,
        all_embeddings,
        get_businesses,
        matched_category,
        top_k=len(cells),
        query_keywords=query_keywords,
        category_confidence=category_confidence,
    )

    avg_competition = (
        sum(s.competition_count for s in all_scores) / len(all_scores)
        if all_scores else 0.0
    )

    ranked = all_scores[: request.top_k]

    # 6. Build response items
    items = [
        RecommendationItem(
            h3_cell=score.h3_cell,
            rank=rank,
            opportunity_score=score.opportunity_score,
            competition_count=score.competition_count,
            competition_density=score.competition_density,
            demand_proxy=score.demand_proxy,
            diversity_score=score.diversity_score,
            similarity_saturation=score.similarity_saturation,
            support_poi_summary=score.support_poi_summary,
            explanation=_generate_explanation(score, matched_category, avg_competition),
        )
        for rank, score in enumerate(ranked, start=1)
    ]

    return RecommendationResponse(
        query=request.query,
        matched_category=matched_category,
        recommendations=items,
    )


@router.post(
    "/heatmap",
    response_model=HeatmapResponse,
    summary="Get opportunity heatmap",
    description=(
        "Returns an opportunity score for every H3 cell in the bounding box — "
        "useful for rendering a colour-coded map of where to open a business.\n\n"
        "Unlike /recommendations, this endpoint returns **all** cells (not just the top-k), "
        "so it is suited for map visualisation rather than a ranked list."
    ),
)
def get_heatmap(
    request: HeatmapRequest,
    generator: EmbeddingGenerator = Depends(get_embedding_generator),
    driver=Depends(get_driver),
) -> HeatmapResponse:
    bbox = request.bbox

    # 1. Embed the user's query
    query_emb = generator.embed_query(request.query)

    # 2. Find all H3 cells inside the requested area
    cells = _cells_in_bbox(bbox.south, bbox.west, bbox.north, bbox.east)

    # 3. Resolve matched category and how confidently it was matched
    matched_category, category_confidence = _resolve_category(request.query, cells, driver, generator)

    # 4. Load pre-computed business embeddings
    osm_ids, emb_matrix = _load_embeddings_safe()
    all_embeddings: dict[str, np.ndarray] = {
        oid: emb_matrix[i] for i, oid in enumerate(osm_ids)
    }

    # 5. Score every cell (no top-k cutoff)
    query_keywords = frozenset(_strip_venue_words(request.query).lower().split())

    def get_businesses(cell: str) -> list[dict]:
        return get_area_businesses(driver, cell)

    all_scores = rank_areas(
        cells,
        query_emb,
        all_embeddings,
        get_businesses,
        matched_category,
        top_k=len(cells),
        query_keywords=query_keywords,
        category_confidence=category_confidence,
    )

    # 6. Attach lat/lng centre of each H3 cell for map rendering
    heatmap_cells = []
    for score in all_scores:
        lat, lng = h3.cell_to_latlng(score.h3_cell)
        heatmap_cells.append(
            HeatmapCell(
                h3_cell=score.h3_cell,
                lat=lat,
                lng=lng,
                opportunity_score=score.opportunity_score,
                competition_density=score.competition_density,
            )
        )

    return HeatmapResponse(
        query=request.query,
        matched_category=matched_category,
        cells=heatmap_cells,
    )
