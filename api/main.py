"""
FastAPI application entry point.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.routers.recommendations import router as recommendations_router
from config import settings

logger = logging.getLogger(__name__)


def _download_embeddings_from_blob() -> None:
    """
    Download embeddings.npz from Azure Blob Storage if it is not already
    present on disk.  Skipped silently when no connection string is configured
    (local development).
    """
    if not settings.azure_storage_connection_string:
        return

    embeddings_path = settings.embeddings_dir / "embeddings.npz"
    if embeddings_path.exists():
        return

    try:
        from azure.storage.blob import BlobServiceClient

        logger.info("Downloading embeddings from Azure Blob Storage …")
        embeddings_path.parent.mkdir(parents=True, exist_ok=True)

        client = BlobServiceClient.from_connection_string(
            settings.azure_storage_connection_string
        )
        blob = client.get_blob_client(
            container=settings.azure_blob_container,
            blob=settings.azure_blob_name,
        )
        with open(embeddings_path, "wb") as f:
            f.write(blob.download_blob().readall())

        logger.info("Embeddings downloaded to %s", embeddings_path)

    except Exception as e:
        logger.error("Failed to download embeddings from blob storage: %s", e)
        raise


@asynccontextmanager
async def lifespan(app: FastAPI):
    _download_embeddings_from_blob()
    yield


app = FastAPI(
    title="Location Intelligence Engine",
    description="Recommends optimal locations to open a new business.",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(recommendations_router)


@app.get("/health")
def health_check():
    return {"status": "ok"}
