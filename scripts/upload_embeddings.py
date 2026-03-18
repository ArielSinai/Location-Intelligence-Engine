"""
Upload Embeddings Script
─────────────────────────
Uploads data/embeddings/embeddings.npz to Azure Blob Storage.

Run this once after generating embeddings locally, and again whenever
you regenerate them for a new area.

Usage:
  python -m scripts.upload_embeddings
"""

import sys
from azure.storage.blob import BlobServiceClient
from config import settings


def main() -> int:
    if not settings.azure_storage_connection_string:
        print("ERROR: AZURE_STORAGE_CONNECTION_STRING is not set in .env")
        return 1

    embeddings_path = settings.embeddings_dir / "embeddings.npz"
    if not embeddings_path.exists():
        print(f"ERROR: {embeddings_path} not found. Run run_embeddings.py first.")
        return 1

    print(f"[1/3] Connecting to Azure Blob Storage …")
    client = BlobServiceClient.from_connection_string(
        settings.azure_storage_connection_string
    )

    print(f"[2/3] Ensuring container '{settings.azure_blob_container}' exists …")
    container = client.get_container_client(settings.azure_blob_container)
    if not container.exists():
        container.create_container()

    print(f"[3/3] Uploading {embeddings_path} → {settings.azure_blob_name} …")
    blob = client.get_blob_client(
        container=settings.azure_blob_container,
        blob=settings.azure_blob_name,
    )
    with open(embeddings_path, "rb") as f:
        blob.upload_blob(f, overwrite=True)

    print(f"\nDone. Embeddings uploaded to blob '{settings.azure_blob_name}'.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
