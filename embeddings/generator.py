"""
Embedding Generator.

Wraps a sentence-transformer model to:
  - encode lists of texts into float32 numpy arrays
  - embed a single user query
  - batch-generate embeddings for all BusinessPOI objects
  - persist / restore embeddings as compressed numpy archives (.npz)
  - find the most similar embeddings to a query via cosine similarity
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

from config import settings
from data_collector.normalizer import BusinessPOI


# ---------------------------------------------------------------------------
# Generator class
# ---------------------------------------------------------------------------

class EmbeddingGenerator:
    """Thin wrapper around a SentenceTransformer model."""

    def __init__(self, model_name: str | None = None) -> None:
        self._model: Any = SentenceTransformer(
            model_name or settings.embedding_model
        )

    def encode(self, texts: list[str]) -> np.ndarray:
        """
        Encode a list of strings.

        Returns
        -------
        np.ndarray of shape (N, D), dtype float32
        """
        vectors = self._model.encode(texts, show_progress_bar=False)
        return np.array(vectors, dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        """
        Encode a single query string.

        Returns
        -------
        np.ndarray of shape (D,), dtype float32
        """
        return self.encode([text])[0]


# ---------------------------------------------------------------------------
# Batch generation
# ---------------------------------------------------------------------------

def generate_embeddings(
    pois: list[BusinessPOI],
    generator: EmbeddingGenerator,
) -> dict[str, np.ndarray]:
    """
    Generate one embedding per POI, keyed by osm_id.

    The *description* field of each POI is used as the input text because
    it combines name + category + enrichment tags — giving the model the
    richest signal about what the business actually is.

    Returns
    -------
    dict mapping osm_id → 1-D float32 ndarray
    """
    if not pois:
        return {}

    descriptions = [p.description for p in pois]
    matrix = generator.encode(descriptions)   # shape (N, D)

    return {poi.osm_id: matrix[i] for i, poi in enumerate(pois)}


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_embeddings(
    osm_ids: list[str],
    embeddings: np.ndarray,
    path,
) -> None:
    """
    Save embeddings to a compressed numpy archive.

    Layout inside the .npz:
        "osm_ids"    — 1-D array of strings
        "embeddings" — 2-D float32 array of shape (N, D)
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        osm_ids=np.array(osm_ids, dtype=str),
        embeddings=embeddings.astype(np.float32),
    )


def load_embeddings(path) -> tuple[list[str], np.ndarray]:
    """
    Load embeddings previously saved with save_embeddings.

    Returns
    -------
    (osm_ids, embeddings)
        osm_ids    — plain Python list of strings
        embeddings — float32 ndarray of shape (N, D)
    """
    data = np.load(path, allow_pickle=False)
    osm_ids = data["osm_ids"].tolist()
    embeddings = data["embeddings"].astype(np.float32)
    return osm_ids, embeddings


# ---------------------------------------------------------------------------
# Similarity search
# ---------------------------------------------------------------------------

def find_similar(
    query_emb: np.ndarray,
    embeddings: np.ndarray,
    top_k: int = 5,
) -> list[tuple[int, float]]:
    """
    Return the top-k most similar rows in *embeddings* to *query_emb*
    using cosine similarity.

    Both query_emb and each row in embeddings are assumed to be already
    unit-normalised (which sentence-transformers produces by default).
    If they are not, we normalise them here to be safe.

    Returns
    -------
    List of (index, similarity_score) sorted by score descending.
    """
    # Normalise query
    q = query_emb.astype(np.float32)
    q_norm = np.linalg.norm(q)
    if q_norm > 0:
        q = q / q_norm

    # Normalise corpus rows
    e = embeddings.astype(np.float32)
    norms = np.linalg.norm(e, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)   # avoid division by zero
    e = e / norms

    # Cosine similarity = dot product of unit vectors
    scores = e @ q   # shape (N,)

    k = min(top_k, len(scores))
    top_indices = np.argsort(scores)[::-1][:k]

    return [(int(idx), float(scores[idx])) for idx in top_indices]


def match_category_by_embedding(
    query_emb: np.ndarray,
    category_names: list[str],
    generator: "EmbeddingGenerator",
    threshold: float = 0.3,
) -> str:
    """
    Return the category name most semantically similar to *query_emb*.

    Embeds each unique category label and picks the one with the highest
    cosine similarity to the query.  Falls back to an empty string when
    *category_names* is empty or no candidate meets *threshold*.

    Parameters
    ----------
    query_emb      : embedding of the user's query (1-D float32 array)
    category_names : list of known category strings to search over
    generator      : EmbeddingGenerator used to encode the category labels
    threshold      : minimum cosine similarity to accept a match (0-1)
    """
    if not category_names:
        return ""

    cat_matrix = generator.encode(category_names)          # (N, D)
    hits = find_similar(query_emb, cat_matrix, top_k=1)

    if hits and hits[0][1] >= threshold:
        return category_names[hits[0][0]]

    return ""
