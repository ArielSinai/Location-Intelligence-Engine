
import numpy as np
import pytest

CELL_A = "882db0cc39fffff"
CELL_B = "882db0cc07fffff"
CELL_C = "882db0cc3dfffff"

DIM = 8   # small dimension for fast test vectors


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def unit(v: list) -> np.ndarray:
    a = np.array(v, dtype=np.float32)
    return a / np.linalg.norm(a)


# ---------------------------------------------------------------------------
# AreaScore dataclass
# ---------------------------------------------------------------------------

class TestAreaScore:
    def test_can_be_instantiated(self):
        from analyzer.opportunity import AreaScore
        score = AreaScore(
            h3_cell=CELL_A,
            competition_count=3,
            competition_density=4.1,
            diversity_score=0.6,
            similarity_saturation=0.2,
            opportunity_score=0.75,
        )
        assert score.h3_cell == CELL_A

    def test_has_all_expected_fields(self):
        from analyzer.opportunity import AreaScore
        score = AreaScore(
            h3_cell=CELL_A,
            competition_count=0,
            competition_density=0.0,
            diversity_score=0.0,
            similarity_saturation=0.0,
            opportunity_score=0.0,
        )
        assert hasattr(score, "competition_count")
        assert hasattr(score, "competition_density")
        assert hasattr(score, "diversity_score")
        assert hasattr(score, "similarity_saturation")
        assert hasattr(score, "opportunity_score")


# ---------------------------------------------------------------------------
# compute_competition_density
# ---------------------------------------------------------------------------

class TestComputeCompetitionDensity:
    def test_zero_count_returns_zero(self):
        from analyzer.opportunity import compute_competition_density
        assert compute_competition_density(0, 0.74) == 0.0

    def test_correct_density_calculation(self):
        from analyzer.opportunity import compute_competition_density
        result = compute_competition_density(10, 2.0)
        assert result == pytest.approx(5.0)

    def test_returns_float(self):
        from analyzer.opportunity import compute_competition_density
        result = compute_competition_density(3, 0.74)
        assert isinstance(result, float)

    def test_larger_area_gives_lower_density(self):
        from analyzer.opportunity import compute_competition_density
        density_small = compute_competition_density(5, 1.0)
        density_large = compute_competition_density(5, 5.0)
        assert density_small > density_large

    def test_more_competitors_gives_higher_density(self):
        from analyzer.opportunity import compute_competition_density
        low  = compute_competition_density(2, 1.0)
        high = compute_competition_density(10, 1.0)
        assert high > low


# ---------------------------------------------------------------------------
# compute_diversity_score
# ---------------------------------------------------------------------------

class TestComputeDiversityScore:
    def test_all_same_category_returns_zero(self):
        from analyzer.opportunity import compute_diversity_score
        categories = ["cafe", "cafe", "cafe", "cafe"]
        assert compute_diversity_score(categories) == pytest.approx(0.0)

    def test_all_unique_categories_returns_one(self):
        from analyzer.opportunity import compute_diversity_score
        categories = ["cafe", "gym", "pharmacy", "bakery"]
        assert compute_diversity_score(categories) == pytest.approx(1.0)

    def test_partial_diversity_is_between_zero_and_one(self):
        from analyzer.opportunity import compute_diversity_score
        categories = ["cafe", "cafe", "gym", "pharmacy"]
        score = compute_diversity_score(categories)
        assert 0.0 < score < 1.0

    def test_single_business_returns_zero(self):
        from analyzer.opportunity import compute_diversity_score
        assert compute_diversity_score(["cafe"]) == pytest.approx(0.0)

    def test_empty_list_returns_zero(self):
        from analyzer.opportunity import compute_diversity_score
        assert compute_diversity_score([]) == pytest.approx(0.0)

    def test_returns_float(self):
        from analyzer.opportunity import compute_diversity_score
        result = compute_diversity_score(["cafe", "gym"])
        assert isinstance(result, float)

    def test_more_unique_categories_gives_higher_score(self):
        from analyzer.opportunity import compute_diversity_score
        low  = compute_diversity_score(["cafe", "cafe", "cafe", "gym"])
        high = compute_diversity_score(["cafe", "gym", "pharmacy", "bakery"])
        assert high > low


# ---------------------------------------------------------------------------
# compute_similarity_saturation
# ---------------------------------------------------------------------------

class TestComputeSimilaritySaturation:
    def test_empty_embeddings_returns_zero(self):
        from analyzer.opportunity import compute_similarity_saturation
        query = unit([1, 0, 0, 0, 0, 0, 0, 0])
        assert compute_similarity_saturation(query, []) == pytest.approx(0.0)

    def test_identical_embeddings_return_one(self):
        from analyzer.opportunity import compute_similarity_saturation
        query = unit([1, 0, 0, 0, 0, 0, 0, 0])
        embeddings = [query.copy(), query.copy(), query.copy()]
        result = compute_similarity_saturation(query, embeddings, threshold=0.8)
        assert result == pytest.approx(1.0)

    def test_orthogonal_embeddings_return_zero(self):
        from analyzer.opportunity import compute_similarity_saturation
        query = unit([1, 0, 0, 0, 0, 0, 0, 0])
        others = [
            unit([0, 1, 0, 0, 0, 0, 0, 0]),
            unit([0, 0, 1, 0, 0, 0, 0, 0]),
        ]
        result = compute_similarity_saturation(query, others, threshold=0.8)
        assert result == pytest.approx(0.0)

    def test_partial_saturation(self):
        from analyzer.opportunity import compute_similarity_saturation
        query = unit([1, 0, 0, 0, 0, 0, 0, 0])
        similar   = query.copy()                        # sim = 1.0 → above threshold
        dissimilar = unit([0, 1, 0, 0, 0, 0, 0, 0])   # sim = 0.0 → below threshold
        result = compute_similarity_saturation(query, [similar, dissimilar], threshold=0.8)
        assert result == pytest.approx(0.5)

    def test_result_is_between_zero_and_one(self):
        from analyzer.opportunity import compute_similarity_saturation
        rng = np.random.default_rng(42)
        query = rng.standard_normal(DIM).astype(np.float32)
        query /= np.linalg.norm(query)
        embeddings = [rng.standard_normal(DIM).astype(np.float32) for _ in range(10)]
        result = compute_similarity_saturation(query, embeddings)
        assert 0.0 <= result <= 1.0

    def test_lower_threshold_gives_higher_or_equal_saturation(self):
        from analyzer.opportunity import compute_similarity_saturation
        rng = np.random.default_rng(0)
        query = unit(rng.standard_normal(DIM).tolist())
        embeddings = [unit(rng.standard_normal(DIM).tolist()) for _ in range(20)]
        low_thresh  = compute_similarity_saturation(query, embeddings, threshold=0.1)
        high_thresh = compute_similarity_saturation(query, embeddings, threshold=0.9)
        assert low_thresh >= high_thresh


# ---------------------------------------------------------------------------
# compute_opportunity_score
# ---------------------------------------------------------------------------

class TestComputeOpportunityScore:
    def test_returns_float(self):
        from analyzer.opportunity import compute_opportunity_score
        result = compute_opportunity_score(
            competition_density=2.0,
            diversity_score=0.5,
            similarity_saturation=0.3,
        )
        assert isinstance(result, float)

    def test_score_is_between_zero_and_one(self):
        from analyzer.opportunity import compute_opportunity_score
        result = compute_opportunity_score(
            competition_density=5.0,
            diversity_score=0.6,
            similarity_saturation=0.4,
        )
        assert 0.0 <= result <= 1.0

    def test_zero_competition_zero_saturation_max_diversity_is_high(self):
        from analyzer.opportunity import compute_opportunity_score
        result = compute_opportunity_score(
            competition_density=0.0,
            diversity_score=1.0,
            similarity_saturation=0.0,
        )
        assert result > 0.7

    def test_max_competition_zero_diversity_max_saturation_is_low(self):
        from analyzer.opportunity import compute_opportunity_score
        result = compute_opportunity_score(
            competition_density=100.0,
            diversity_score=0.0,
            similarity_saturation=1.0,
        )
        assert result < 0.3

    def test_higher_competition_lowers_score(self):
        from analyzer.opportunity import compute_opportunity_score
        low_comp  = compute_opportunity_score(1.0, 0.5, 0.3)
        high_comp = compute_opportunity_score(9.0, 0.5, 0.3)
        assert low_comp > high_comp

    def test_higher_diversity_raises_score(self):
        from analyzer.opportunity import compute_opportunity_score
        low_div  = compute_opportunity_score(2.0, 0.1, 0.3)
        high_div = compute_opportunity_score(2.0, 0.9, 0.3)
        assert high_div > low_div

    def test_higher_saturation_lowers_score(self):
        from analyzer.opportunity import compute_opportunity_score
        low_sat  = compute_opportunity_score(2.0, 0.5, 0.1)
        high_sat = compute_opportunity_score(2.0, 0.5, 0.9)
        assert low_sat > high_sat


# ---------------------------------------------------------------------------
# score_area
# ---------------------------------------------------------------------------

class TestScoreArea:
    def _make_businesses(self, categories: list[str]) -> list[dict]:
        return [{"osm_id": str(i), "category": c} for i, c in enumerate(categories)]

    def _make_embeddings(self, osm_ids: list[str], query: np.ndarray, similar: bool) -> dict:
        rng = np.random.default_rng(0)
        result = {}
        for oid in osm_ids:
            if similar:
                result[oid] = query.copy()
            else:
                v = rng.standard_normal(DIM).astype(np.float32)
                result[oid] = v / np.linalg.norm(v)
        return result

    def test_returns_area_score(self):
        from analyzer.opportunity import score_area, AreaScore
        query = unit([1, 0, 0, 0, 0, 0, 0, 0])
        businesses = self._make_businesses(["cafe", "gym"])
        embeddings = self._make_embeddings(["0", "1"], query, similar=False)
        result = score_area(CELL_A, query, embeddings, businesses, "cafe")
        assert isinstance(result, AreaScore)

    def test_h3_cell_is_preserved(self):
        from analyzer.opportunity import score_area
        query = unit([1, 0, 0, 0, 0, 0, 0, 0])
        businesses = self._make_businesses(["cafe"])
        embeddings = self._make_embeddings(["0"], query, similar=False)
        result = score_area(CELL_A, query, embeddings, businesses, "cafe")
        assert result.h3_cell == CELL_A

    def test_competition_count_matches_category(self):
        from analyzer.opportunity import score_area
        query = unit([1, 0, 0, 0, 0, 0, 0, 0])
        businesses = self._make_businesses(["cafe", "cafe", "gym"])
        embeddings = self._make_embeddings(["0", "1", "2"], query, similar=False)
        result = score_area(CELL_A, query, embeddings, businesses, "cafe")
        assert result.competition_count == 2

    def test_empty_area_has_high_opportunity(self):
        from analyzer.opportunity import score_area
        query = unit([1, 0, 0, 0, 0, 0, 0, 0])
        result = score_area(CELL_A, query, {}, [], "cafe")
        assert result.opportunity_score > 0.5

    def test_saturated_area_has_low_opportunity(self):
        from analyzer.opportunity import score_area
        query = unit([1, 0, 0, 0, 0, 0, 0, 0])
        # Many identical businesses, same category as query
        businesses = self._make_businesses(["cafe"] * 20)
        osm_ids = [str(i) for i in range(20)]
        embeddings = self._make_embeddings(osm_ids, query, similar=True)
        result = score_area(CELL_A, query, embeddings, businesses, "cafe")
        assert result.opportunity_score < 0.5

    def test_opportunity_score_is_between_zero_and_one(self):
        from analyzer.opportunity import score_area
        query = unit([1, 0, 0, 0, 0, 0, 0, 0])
        businesses = self._make_businesses(["cafe", "gym", "pharmacy"])
        embeddings = self._make_embeddings(["0", "1", "2"], query, similar=False)
        result = score_area(CELL_A, query, embeddings, businesses, "cafe")
        assert 0.0 <= result.opportunity_score <= 1.0


# ---------------------------------------------------------------------------
# rank_areas
# ---------------------------------------------------------------------------

class TestRankAreas:
    def _get_businesses_fn(self, data: dict):
        """Factory for a fake get_area_businesses function."""
        def fn(cell):
            return data.get(cell, [])
        return fn

    def test_returns_list(self):
        from analyzer.opportunity import rank_areas
        query = unit([1, 0, 0, 0, 0, 0, 0, 0])
        get_fn = self._get_businesses_fn({CELL_A: [], CELL_B: [], CELL_C: []})
        result = rank_areas([CELL_A, CELL_B, CELL_C], query, {}, get_fn, "cafe")
        assert isinstance(result, list)

    def test_result_contains_area_scores(self):
        from analyzer.opportunity import rank_areas, AreaScore
        query = unit([1, 0, 0, 0, 0, 0, 0, 0])
        get_fn = self._get_businesses_fn({CELL_A: [], CELL_B: []})
        result = rank_areas([CELL_A, CELL_B], query, {}, get_fn, "cafe")
        assert all(isinstance(r, AreaScore) for r in result)

    def test_top_k_limits_results(self):
        from analyzer.opportunity import rank_areas
        query = unit([1, 0, 0, 0, 0, 0, 0, 0])
        cells = [CELL_A, CELL_B, CELL_C]
        get_fn = self._get_businesses_fn({c: [] for c in cells})
        result = rank_areas(cells, query, {}, get_fn, "cafe", top_k=2)
        assert len(result) == 2

    def test_results_sorted_by_opportunity_score_descending(self):
        from analyzer.opportunity import rank_areas
        query = unit([1, 0, 0, 0, 0, 0, 0, 0])
        # CELL_A: empty (high opportunity)
        # CELL_B: many same-category businesses (low opportunity)
        saturated = [{"osm_id": str(i), "category": "cafe"} for i in range(15)]
        get_fn = self._get_businesses_fn({CELL_A: [], CELL_B: saturated})
        emb_b = {str(i): query.copy() for i in range(15)}
        result = rank_areas([CELL_A, CELL_B], query, emb_b, get_fn, "cafe", top_k=2)
        scores = [r.opportunity_score for r in result]
        assert scores == sorted(scores, reverse=True)

    def test_top_k_larger_than_cells_returns_all(self):
        from analyzer.opportunity import rank_areas
        query = unit([1, 0, 0, 0, 0, 0, 0, 0])
        cells = [CELL_A, CELL_B]
        get_fn = self._get_businesses_fn({c: [] for c in cells})
        result = rank_areas(cells, query, {}, get_fn, "cafe", top_k=100)
        assert len(result) == 2

    def test_empty_cell_list_returns_empty(self):
        from analyzer.opportunity import rank_areas
        query = unit([1, 0, 0, 0, 0, 0, 0, 0])
        result = rank_areas([], query, {}, lambda c: [], "cafe")
        assert result == []
