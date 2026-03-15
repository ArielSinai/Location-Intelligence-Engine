"""
FastAPI dependency providers.

Each function is used with FastAPI's Depends() system.
Keeping them here (rather than inline in the router) makes them
easy to override in tests via mocker.patch.
"""

from neo4j import GraphDatabase

from config import settings
from embeddings.generator import EmbeddingGenerator

# Both singletons are loaded once on first use and reused for every request.
_generator: EmbeddingGenerator | None = None
_driver = None


def get_embedding_generator() -> EmbeddingGenerator:
    """Provide a shared EmbeddingGenerator instance (loaded once)."""
    global _generator
    if _generator is None:
        _generator = EmbeddingGenerator(settings.embedding_model)
    return _generator


def get_driver():
    """
    Provide the shared Neo4j driver.

    GraphDatabase.driver() manages an internal connection pool, so it must be
    created once and reused — not instantiated per request.
    """
    global _driver
    if _driver is None:
        _driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_username, settings.neo4j_password),
        )
    return _driver
