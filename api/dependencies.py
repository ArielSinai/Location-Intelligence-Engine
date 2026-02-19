"""
FastAPI dependency providers.

Each function is used with FastAPI's Depends() system.
Keeping them here (rather than inline in the router) makes them
easy to override in tests via mocker.patch.
"""

from neo4j import GraphDatabase

from config import settings
from embeddings.generator import EmbeddingGenerator


def get_embedding_generator() -> EmbeddingGenerator:
    """Provide a loaded EmbeddingGenerator instance."""
    return EmbeddingGenerator(settings.embedding_model)


def get_driver():
    """Provide a connected Neo4j driver, closed automatically after each request."""
    driver = GraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_username, settings.neo4j_password),
    )
    try:
        yield driver
    finally:
        driver.close()
