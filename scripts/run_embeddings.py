"""
Embedding Generation Script
────────────────────────────
Loads a normalised POI file, generates embeddings for every business,
and saves a compressed numpy archive:
  data/embeddings/embeddings.npz

This script runs locally (CPU-heavy — not suitable for the free Azure tier).

Usage:
  python -m scripts.run_embeddings --input data/raw/tel_aviv_normalized.json
"""

import argparse
import sys

import numpy as np
from tqdm import tqdm

from config import settings
from data_collector.normalizer import load_normalized
from embeddings.generator import EmbeddingGenerator, generate_embeddings, save_embeddings


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Generate embeddings for normalised POIs")
    p.add_argument(
        "--input", required=True,
        help="Path to normalised JSON file (output of run_collector.py)",
    )
    p.add_argument(
        "--output",
        default=str(settings.embeddings_dir / "embeddings.npz"),
        help="Output path for the .npz file (default: data/embeddings/embeddings.npz)",
    )
    p.add_argument(
        "--model",
        default=settings.embedding_model,
        help=f"Sentence-transformer model name (default: {settings.embedding_model})",
    )
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    print(f"[1/4] Loading normalised POIs from {args.input} …")
    pois = load_normalized(args.input)
    print(f"      → {len(pois)} businesses loaded")

    print(f"[2/4] Loading embedding model: {args.model} …")
    generator = EmbeddingGenerator(args.model)

    print(f"[3/4] Generating embeddings …")
    emb_dict = generate_embeddings(pois, generator)

    osm_ids   = list(emb_dict.keys())
    emb_matrix = np.stack([emb_dict[oid] for oid in osm_ids])
    print(f"      → matrix shape: {emb_matrix.shape}")

    print(f"[4/4] Saving to {args.output} …")
    save_embeddings(osm_ids, emb_matrix, args.output)

    print(f"\nDone. Embeddings saved to: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
