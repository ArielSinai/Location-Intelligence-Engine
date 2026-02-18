
import numpy as np
import pytest

from data_collector.normalizer import BusinessPOI


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

EMBED_DIM = 384     # all-MiniLM-L6-v2 output dimension

def make_poi(osm_id: str, description: str = "some place") -> BusinessPOI:
    return BusinessPOI(
        osm_id=osm_id,
        name="Test",
        lat=32.07,
        lon=34.78,
        category="cafe",
        category_group="Food & Drink",
        description=description,
        h3_cell="882db0cc39fffff",
        tags={},
    )

def random_unit_vector(dim: int = EMBED_DIM, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


# ---------------------------------------------------------------------------
# Fixture: patches SentenceTransformer so no model download happens
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_model(mocker):
    """
    Returns a mock whose .encode() produces deterministic unit vectors.
    Each unique text gets a reproducible random vector seeded by its index.
    """
    def fake_encode(texts, **kwargs):
        vecs = np.stack([random_unit_vector(EMBED_DIM, seed=i) for i, _ in enumerate(texts)])
        return vecs

    mock = mocker.MagicMock()
    mock.encode.side_effect = fake_encode
    mocker.patch(
        "embeddings.generator.SentenceTransformer",
        return_value=mock,
    )
    return mock


# ---------------------------------------------------------------------------
# EmbeddingGenerator.encode
# ---------------------------------------------------------------------------

class TestEncode:
    def test_returns_ndarray(self, mock_model):
        from embeddings.generator import EmbeddingGenerator
        gen = EmbeddingGenerator()
        result = gen.encode(["hello", "world"])
        assert isinstance(result, np.ndarray)

    def test_shape_matches_input_length(self, mock_model):
        from embeddings.generator import EmbeddingGenerator
        gen = EmbeddingGenerator()
        texts = ["one", "two", "three"]
        result = gen.encode(texts)
        assert result.shape[0] == len(texts)

    def test_output_dimension_is_embed_dim(self, mock_model):
        from embeddings.generator import EmbeddingGenerator
        gen = EmbeddingGenerator()
        result = gen.encode(["test"])
        assert result.shape[1] == EMBED_DIM

    def test_single_text_works(self, mock_model):
        from embeddings.generator import EmbeddingGenerator
        gen = EmbeddingGenerator()
        result = gen.encode(["just one"])
        assert result.shape == (1, EMBED_DIM)

    def test_vectors_are_float32(self, mock_model):
        from embeddings.generator import EmbeddingGenerator
        gen = EmbeddingGenerator()
        result = gen.encode(["test"])
        assert result.dtype == np.float32

    def test_model_encode_called_once(self, mock_model):
        from embeddings.generator import EmbeddingGenerator
        gen = EmbeddingGenerator()
        gen.encode(["a", "b"])
        mock_model.encode.assert_called_once()


# ---------------------------------------------------------------------------
# EmbeddingGenerator.embed_query
# ---------------------------------------------------------------------------

class TestEmbedQuery:
    def test_returns_1d_array(self, mock_model):
        from embeddings.generator import EmbeddingGenerator
        gen = EmbeddingGenerator()
        result = gen.embed_query("pilates studio")
        assert result.ndim == 1

    def test_length_is_embed_dim(self, mock_model):
        from embeddings.generator import EmbeddingGenerator
        gen = EmbeddingGenerator()
        result = gen.embed_query("pilates studio")
        assert len(result) == EMBED_DIM

    def test_is_float32(self, mock_model):
        from embeddings.generator import EmbeddingGenerator
        gen = EmbeddingGenerator()
        result = gen.embed_query("test")
        assert result.dtype == np.float32


# ---------------------------------------------------------------------------
# generate_embeddings
# ---------------------------------------------------------------------------

class TestGenerateEmbeddings:
    def test_returns_dict(self, mock_model):
        from embeddings.generator import EmbeddingGenerator, generate_embeddings
        pois = [make_poi("1"), make_poi("2")]
        gen = EmbeddingGenerator()
        result = generate_embeddings(pois, gen)
        assert isinstance(result, dict)

    def test_keys_are_osm_ids(self, mock_model):
        from embeddings.generator import EmbeddingGenerator, generate_embeddings
        pois = [make_poi("A"), make_poi("B"), make_poi("C")]
        gen = EmbeddingGenerator()
        result = generate_embeddings(pois, gen)
        assert set(result.keys()) == {"A", "B", "C"}

    def test_values_are_1d_arrays(self, mock_model):
        from embeddings.generator import EmbeddingGenerator, generate_embeddings
        pois = [make_poi("1")]
        gen = EmbeddingGenerator()
        result = generate_embeddings(pois, gen)
        assert result["1"].ndim == 1

    def test_embedding_length_is_embed_dim(self, mock_model):
        from embeddings.generator import EmbeddingGenerator, generate_embeddings
        pois = [make_poi("1")]
        gen = EmbeddingGenerator()
        result = generate_embeddings(pois, gen)
        assert len(result["1"]) == EMBED_DIM

    def test_uses_description_field(self, mock_model):
        """Verify that the description text is what gets passed to encode."""
        from embeddings.generator import EmbeddingGenerator, generate_embeddings
        pois = [make_poi("1", description="unique description text")]
        gen = EmbeddingGenerator()
        generate_embeddings(pois, gen)
        call_args = mock_model.encode.call_args
        texts_passed = call_args.args[0] if call_args.args else call_args.kwargs.get("sentences")
        assert "unique description text" in texts_passed

    def test_empty_poi_list_returns_empty_dict(self, mock_model):
        from embeddings.generator import EmbeddingGenerator, generate_embeddings
        gen = EmbeddingGenerator()
        result = generate_embeddings([], gen)
        assert result == {}


# ---------------------------------------------------------------------------
# save_embeddings / load_embeddings
# ---------------------------------------------------------------------------

class TestEmbeddingRoundTrip:
    def _sample_data(self):
        osm_ids = ["1", "2", "3"]
        embeddings = np.stack([random_unit_vector(EMBED_DIM, seed=i) for i in range(3)])
        return osm_ids, embeddings

    def test_save_creates_file(self, tmp_path):
        from embeddings.generator import save_embeddings
        osm_ids, embeddings = self._sample_data()
        out = tmp_path / "embs.npz"
        save_embeddings(osm_ids, embeddings, out)
        assert out.exists()

    def test_load_returns_tuple_of_two(self, tmp_path):
        from embeddings.generator import save_embeddings, load_embeddings
        osm_ids, embeddings = self._sample_data()
        out = tmp_path / "embs.npz"
        save_embeddings(osm_ids, embeddings, out)
        result = load_embeddings(out)
        assert len(result) == 2

    def test_roundtrip_preserves_osm_ids(self, tmp_path):
        from embeddings.generator import save_embeddings, load_embeddings
        osm_ids, embeddings = self._sample_data()
        out = tmp_path / "embs.npz"
        save_embeddings(osm_ids, embeddings, out)
        loaded_ids, _ = load_embeddings(out)
        assert list(loaded_ids) == osm_ids

    def test_roundtrip_preserves_embedding_values(self, tmp_path):
        from embeddings.generator import save_embeddings, load_embeddings
        osm_ids, embeddings = self._sample_data()
        out = tmp_path / "embs.npz"
        save_embeddings(osm_ids, embeddings, out)
        _, loaded_embs = load_embeddings(out)
        np.testing.assert_array_almost_equal(loaded_embs, embeddings)

    def test_roundtrip_preserves_shape(self, tmp_path):
        from embeddings.generator import save_embeddings, load_embeddings
        osm_ids, embeddings = self._sample_data()
        out = tmp_path / "embs.npz"
        save_embeddings(osm_ids, embeddings, out)
        _, loaded_embs = load_embeddings(out)
        assert loaded_embs.shape == embeddings.shape


# ---------------------------------------------------------------------------
# find_similar
# ---------------------------------------------------------------------------

class TestFindSimilar:
    def _corpus(self):
        """Three known unit vectors with controlled similarity."""
        v0 = np.array([1.0, 0.0, 0.0], dtype=np.float32)   # query will be identical
        v1 = np.array([0.0, 1.0, 0.0], dtype=np.float32)   # orthogonal  → sim = 0
        v2 = np.array([-1.0, 0.0, 0.0], dtype=np.float32)  # opposite    → sim = -1
        return np.stack([v0, v1, v2])

    def test_returns_list(self):
        from embeddings.generator import find_similar
        corpus = self._corpus()
        query = corpus[0]
        result = find_similar(query, corpus, top_k=2)
        assert isinstance(result, list)

    def test_returns_top_k_results(self):
        from embeddings.generator import find_similar
        corpus = self._corpus()
        result = find_similar(corpus[0], corpus, top_k=2)
        assert len(result) == 2

    def test_each_result_is_index_score_tuple(self):
        from embeddings.generator import find_similar
        corpus = self._corpus()
        result = find_similar(corpus[0], corpus, top_k=1)
        idx, score = result[0]
        assert isinstance(idx, (int, np.integer))
        assert isinstance(score, (float, np.floating))

    def test_identical_vector_scores_one(self):
        from embeddings.generator import find_similar
        corpus = self._corpus()
        query = corpus[0].copy()
        result = find_similar(query, corpus, top_k=3)
        # Index 0 is identical to query → similarity should be ~1.0
        scores_by_idx = {idx: score for idx, score in result}
        assert scores_by_idx[0] == pytest.approx(1.0, abs=1e-5)

    def test_orthogonal_vector_scores_zero(self):
        from embeddings.generator import find_similar
        corpus = self._corpus()
        query = corpus[0].copy()
        result = find_similar(query, corpus, top_k=3)
        scores_by_idx = {idx: score for idx, score in result}
        assert scores_by_idx[1] == pytest.approx(0.0, abs=1e-5)

    def test_results_sorted_descending_by_score(self):
        from embeddings.generator import find_similar
        corpus = self._corpus()
        result = find_similar(corpus[0], corpus, top_k=3)
        scores = [score for _, score in result]
        assert scores == sorted(scores, reverse=True)

    def test_top_k_larger_than_corpus_returns_all(self):
        from embeddings.generator import find_similar
        corpus = self._corpus()
        result = find_similar(corpus[0], corpus, top_k=100)
        assert len(result) == len(corpus)
