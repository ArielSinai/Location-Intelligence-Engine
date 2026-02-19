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
from embeddings.generator import load_embeddings, EmbeddingGenerator
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
    diversity_score: float
    similarity_saturation: float
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


def _matched_category(query: str) -> str:
    """
    Return a normalised category label for the user's query.
    Uses simple string normalisation for the MVP.
    """
    return query.strip().lower()


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

    if score.diversity_score > 0.6:
        parts.append("The area has a diverse mix of complementary services nearby.")
    elif score.diversity_score < 0.3:
        parts.append("Limited service variety in the immediate area.")

    if score.similarity_saturation < 0.2:
        parts.append("Low semantic overlap — few conceptually similar businesses present.")
    elif score.similarity_saturation > 0.7:
        parts.append("High semantic saturation — many similar concepts already operating here.")

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

    # 2. Determine matched category
    matched_category = _matched_category(request.query)

    # 3. Find all H3 cells inside the requested area
    cells = _cells_in_bbox(bbox.south, bbox.west, bbox.north, bbox.east)

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
            diversity_score=score.diversity_score,
            similarity_saturation=score.similarity_saturation,
            explanation=_generate_explanation(score, matched_category),
        )
        for rank, score in enumerate(ranked, start=1)
    ]

    return RecommendationResponse(
        query=request.query,
        matched_category=matched_category,
        recommendations=items,
    )
