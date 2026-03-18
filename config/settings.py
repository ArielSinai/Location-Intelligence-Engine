from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

# Project root (one level above config/)
ROOT_DIR = Path(__file__).parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ROOT_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Neo4j
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_username: str = "neo4j"
    neo4j_password: str = "password"

    # Embeddings
    embedding_model: str = "all-MiniLM-L6-v2"

    # Neo4j database name (AuraDB uses the instance ID, e.g. "ef1ba9d9")
    neo4j_database: str = ""

    # Azure Blob Storage (only needed when deploying)
    azure_storage_connection_string: str = ""
    azure_blob_container: str = "embeddings"
    azure_blob_name: str = "embeddings.npz"

    # H3 grid resolution
    # 8 → avg cell area ~0.74 km²  (good for neighborhoods)
    # 7 → avg cell area ~5.16 km²  (good for districts)
    h3_resolution: int = 8

    # Overpass API
    overpass_url: str = "https://overpass-api.de/api/interpreter"

    # Data paths
    raw_data_dir: Path = ROOT_DIR / "data" / "raw"
    embeddings_dir: Path = ROOT_DIR / "data" / "embeddings"


# Singleton used across the whole application
settings = Settings()
