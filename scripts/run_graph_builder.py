"""
Graph Building Script
──────────────────────
Loads a normalised POI file, connects to Neo4j, creates uniqueness
constraints, and upserts all Business / Category / Area nodes plus
their edges.

Runs locally — CPU-light but requires a running Neo4j instance.

Usage:
  # Make sure Neo4j is running (e.g. via docker-compose up neo4j)
  python -m scripts.run_graph_builder --input data/raw/tel_aviv_normalized.json

  # Clear the graph first (fresh rebuild):
  python -m scripts.run_graph_builder --input data/raw/tel_aviv_normalized.json --clear
"""

import argparse
import sys

from tqdm import tqdm
from neo4j import GraphDatabase

from config import settings
from data_collector.normalizer import load_normalized, SUPPORT_CATEGORIES
from graph.builder import GraphBuilder


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Build the Neo4j graph from normalised POIs")
    p.add_argument(
        "--input", required=True,
        help="Path to normalised JSON file (output of run_collector.py)",
    )
    p.add_argument(
        "--clear", action="store_true",
        help="Delete all existing nodes and relationships before building",
    )
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    print(f"[1/4] Loading normalised POIs from {args.input} …")
    pois = load_normalized(args.input)
    print(f"      → {len(pois)} businesses loaded")

    print(f"[2/4] Connecting to Neo4j at {settings.neo4j_uri} …")
    driver = GraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_username, settings.neo4j_password),
    )

    builder = GraphBuilder(driver)

    if args.clear:
        print("      Clearing existing graph …")
        with driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        print("      Done.")

    print("[3/4] Creating constraints …")
    builder.create_constraints()

    print(f"[4/4] Upserting {len(pois)} POIs …")
    for poi in tqdm(pois, unit="poi"):
        builder.upsert_business(poi)
        builder.upsert_category(poi.category, poi.category_group)
        builder.upsert_area(poi.h3_cell)
        builder.link_area_neighbors(poi.h3_cell)
        builder.link_business_to_category(poi.osm_id, poi.category)
        builder.link_business_to_area(poi.osm_id, poi.h3_cell)
        if poi.category in SUPPORT_CATEGORIES:
            builder.link_area_to_support_poi(poi.h3_cell, poi.osm_id)

    driver.close()
    print(f"\nDone. Graph built with {len(pois)} businesses.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
