
import numpy as np
import pytest
from fastapi.testclient import TestClient

from analyzer.opportunity import AreaScore

# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

CELL_A = "882db0cc39fffff"
CELL_B = "882db0cc07fffff"
DIM = 384

VALID_BBOX_PAYLOAD = {
    "bbox": {"south": 32.05, "west": 34.76, "north": 32.10, "east": 34.82},
    "query": "pilates studio",
    "top_k": 3,
}

FAKE_SCORES = [
    AreaScore(CELL_A, 1, 1.35, 0.6, 0.1, 0.82),
    AreaScore(CELL_B, 5, 6.76, 0.3, 0.7, 0.31),
]


# ---------------------------------------------------------------------------
# Fixture: app with all heavy deps mocked out
# ---------------------------------------------------------------------------

@pytest.fixture
def client(mocker):
    """
    Import the app AFTER patching the slow/external dependencies so the
    module-level setup (model loading, DB connection) never runs in tests.
    """
    # Patch the EmbeddingGenerator so no model weights are loaded
    fake_emb = np.ones(DIM, dtype=np.float32) / np.sqrt(DIM)
    mock_gen = mocker.MagicMock()
    mock_gen.embed_query.return_value = fake_emb
    mocker.patch("api.dependencies.EmbeddingGenerator", return_value=mock_gen)

    # Patch GraphDriver so no Neo4j connection is attempted
    mocker.patch("api.dependencies.GraphDatabase")

    # Patch get_area_businesses so no real Neo4j calls happen in _resolve_category
    mocker.patch(
        "api.routers.recommendations.get_area_businesses",
        return_value=[
            {"osm_id": "1", "category": "pilates studio"},
            {"osm_id": "2", "category": "fitness center"},
        ],
    )

    # Make the mock generator encode() return unit vectors (used by match_category_by_embedding)
    mock_gen.encode.return_value = np.stack([fake_emb, fake_emb])

    # Patch rank_areas so no real scoring happens
    mocker.patch("api.routers.recommendations.rank_areas", return_value=FAKE_SCORES)

    # Patch load_embeddings so no file I/O happens
    osm_ids = ["1", "2"]
    embeddings = np.stack([fake_emb, fake_emb])
    mocker.patch(
        "api.routers.recommendations.load_embeddings",
        return_value=(osm_ids, embeddings),
    )

    from api.main import app
    return TestClient(app)


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

class TestHealthCheck:
    def test_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_returns_json(self, client):
        response = client.get("/health")
        assert response.headers["content-type"].startswith("application/json")

    def test_body_has_status_ok(self, client):
        response = client.get("/health")
        assert response.json()["status"] == "ok"


# ---------------------------------------------------------------------------
# POST /recommendations — happy path
# ---------------------------------------------------------------------------

class TestRecommendationsHappyPath:
    def test_returns_200(self, client):
        response = client.post("/recommendations", json=VALID_BBOX_PAYLOAD)
        assert response.status_code == 200

    def test_response_has_query_field(self, client):
        response = client.post("/recommendations", json=VALID_BBOX_PAYLOAD)
        assert "query" in response.json()

    def test_response_query_matches_input(self, client):
        response = client.post("/recommendations", json=VALID_BBOX_PAYLOAD)
        assert response.json()["query"] == "pilates studio"

    def test_response_has_recommendations_list(self, client):
        response = client.post("/recommendations", json=VALID_BBOX_PAYLOAD)
        assert "recommendations" in response.json()
        assert isinstance(response.json()["recommendations"], list)

    def test_response_has_matched_category(self, client):
        response = client.post("/recommendations", json=VALID_BBOX_PAYLOAD)
        assert "matched_category" in response.json()
        assert isinstance(response.json()["matched_category"], str)

    def test_recommendations_count_respects_top_k(self, client):
        response = client.post("/recommendations", json=VALID_BBOX_PAYLOAD)
        recs = response.json()["recommendations"]
        assert len(recs) <= VALID_BBOX_PAYLOAD["top_k"]

    def test_recommendations_count_matches_fake_scores(self, client):
        response = client.post("/recommendations", json=VALID_BBOX_PAYLOAD)
        recs = response.json()["recommendations"]
        assert len(recs) == len(FAKE_SCORES)


# ---------------------------------------------------------------------------
# POST /recommendations — recommendation item shape
# ---------------------------------------------------------------------------

class TestRecommendationItemShape:
    def _first(self, client):
        response = client.post("/recommendations", json=VALID_BBOX_PAYLOAD)
        return response.json()["recommendations"][0]

    def test_item_has_h3_cell(self, client):
        item = self._first(client)
        assert "h3_cell" in item

    def test_item_has_rank(self, client):
        item = self._first(client)
        assert "rank" in item

    def test_first_item_has_rank_one(self, client):
        item = self._first(client)
        assert item["rank"] == 1

    def test_item_has_opportunity_score(self, client):
        item = self._first(client)
        assert "opportunity_score" in item

    def test_opportunity_score_is_float(self, client):
        item = self._first(client)
        assert isinstance(item["opportunity_score"], float)

    def test_item_has_competition_count(self, client):
        item = self._first(client)
        assert "competition_count" in item

    def test_item_has_explanation(self, client):
        item = self._first(client)
        assert "explanation" in item

    def test_explanation_is_non_empty_string(self, client):
        item = self._first(client)
        assert isinstance(item["explanation"], str)
        assert len(item["explanation"]) > 0

    def test_ranks_are_sequential(self, client):
        response = client.post("/recommendations", json=VALID_BBOX_PAYLOAD)
        ranks = [r["rank"] for r in response.json()["recommendations"]]
        assert ranks == list(range(1, len(ranks) + 1))

    def test_h3_cell_matches_fake_scores(self, client):
        response = client.post("/recommendations", json=VALID_BBOX_PAYLOAD)
        cells = [r["h3_cell"] for r in response.json()["recommendations"]]
        assert cells[0] == CELL_A


# ---------------------------------------------------------------------------
# POST /recommendations — input validation
# ---------------------------------------------------------------------------

class TestRecommendationsValidation:
    def test_missing_query_returns_422(self, client):
        payload = {
            "bbox": {"south": 32.05, "west": 34.76, "north": 32.10, "east": 34.82}
        }
        response = client.post("/recommendations", json=payload)
        assert response.status_code == 422

    def test_missing_bbox_returns_422(self, client):
        payload = {"query": "cafe"}
        response = client.post("/recommendations", json=payload)
        assert response.status_code == 422

    def test_empty_query_returns_422(self, client):
        payload = {
            "bbox": {"south": 32.05, "west": 34.76, "north": 32.10, "east": 34.82},
            "query": "",
        }
        response = client.post("/recommendations", json=payload)
        assert response.status_code == 422

    def test_inverted_bbox_returns_422(self, client):
        # north < south is invalid
        payload = {
            "bbox": {"south": 32.10, "west": 34.76, "north": 32.05, "east": 34.82},
            "query": "cafe",
        }
        response = client.post("/recommendations", json=payload)
        assert response.status_code == 422

    def test_top_k_defaults_to_five(self, client):
        payload = {
            "bbox": {"south": 32.05, "west": 34.76, "north": 32.10, "east": 34.82},
            "query": "cafe",
        }
        # Should not error — top_k has a default
        response = client.post("/recommendations", json=payload)
        assert response.status_code == 200

    def test_top_k_zero_returns_422(self, client):
        payload = {**VALID_BBOX_PAYLOAD, "top_k": 0}
        response = client.post("/recommendations", json=payload)
        assert response.status_code == 422

    def test_top_k_negative_returns_422(self, client):
        payload = {**VALID_BBOX_PAYLOAD, "top_k": -1}
        response = client.post("/recommendations", json=payload)
        assert response.status_code == 422
