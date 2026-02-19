"""
Recommendations router.

POST /recommendations
  → embeds the query, finds H3 cells in the requested bbox,
    scores each cell, and returns ranked recommendations.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field, field_validator
import h3
import numpy as np

from analyzer.opportunity import rank_areas, AreaScore
from embeddings.generator import load_embeddings, EmbeddingGenerator, match_category_by_embedding
from graph.queries import get_area_businesses
from api.dependencies import get_embedding_generator, get_driver
from config import settings

router = APIRouter()


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class BBoxModel(BaseModel):
    south: float
    west: float
    north: float
    east: float

    @field_validator("north")
    @classmethod
    def north_must_exceed_south(cls, v, info):
        south = info.data.get("south")
        if south is not None and v <= south:
            raise ValueError("north must be greater than south")
        return v


class RecommendationRequest(BaseModel):
    bbox: BBoxModel
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1)


class RecommendationItem(BaseModel):
    h3_cell: str
    rank: int
    opportunity_score: float
    competition_count: int
    competition_density: float
    demand_proxy: float
    diversity_score: float
    similarity_saturation: float
    support_poi_summary: dict[str, int]
    explanation: str


class RecommendationResponse(BaseModel):
    query: str
    matched_category: str
    recommendations: list[RecommendationItem]


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
    query_emb: np.ndarray,
    cells: list[str],
    driver,
    generator: EmbeddingGenerator,
) -> str:
    """
    Use embedding similarity to pick the closest stored category to the query.

    Collects all unique category labels from businesses that exist in the
    requested bbox cells, encodes them, and returns the one most similar
    to the query embedding.  Falls back to the lowercased query string
    when no categories are found in the graph.
    """
    seen: set[str] = set()
    for cell in cells:
        for biz in get_area_businesses(driver, cell):
            cat = biz.get("category", "")
            if cat:
                seen.add(cat)

    category_names = list(seen)
    if not category_names:
        return query.strip().lower()

    matched = match_category_by_embedding(query_emb, category_names, generator)
    return matched if matched else query.strip().lower()


def _generate_explanation(score: AreaScore, query_category: str) -> str:
    """Build a human-readable explanation for why an area is recommended."""
    parts: list[str] = []

    if score.competition_count == 0:
        parts.append(f"No direct {query_category} competitors found in this area.")
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


def _load_embeddings_safe() -> tuple[list[str], np.ndarray]:
    """
    Load pre-computed business embeddings.
    Returns empty structures if the file does not exist yet.
    """
    path = settings.embeddings_dir / "embeddings.npz"
    try:
        return load_embeddings(path)
    except (FileNotFoundError, Exception):
        return [], np.empty((0, 384), dtype=np.float32)


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@router.post("/recommendations", response_model=RecommendationResponse)
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

    # 3. Resolve matched category via embedding similarity against stored categories
    matched_category = _resolve_category(request.query, query_emb, cells, driver, generator)

    # 4. Load pre-computed business embeddings
    osm_ids, emb_matrix = _load_embeddings_safe()
    all_embeddings: dict[str, np.ndarray] = {
        oid: emb_matrix[i] for i, oid in enumerate(osm_ids)
    }

    # 5. Score and rank cells
    def get_businesses(cell: str) -> list[dict]:
        return get_area_businesses(driver, cell)

    ranked = rank_areas(
        cells,
        query_emb,
        all_embeddings,
        get_businesses,
        matched_category,
        top_k=request.top_k,
    )

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
            explanation=_generate_explanation(score, matched_category),
        )
        for rank, score in enumerate(ranked, start=1)
    ]

    return RecommendationResponse(
        query=request.query,
        matched_category=matched_category,
        recommendations=items,
    )
