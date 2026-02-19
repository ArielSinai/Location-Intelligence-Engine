"""
Data Collection Script
─────────────────────
Fetches POIs from OpenStreetMap for a bounding box, normalises them,
and saves two files:
  data/raw/<area_name>.json        ← raw OSM elements
  data/raw/<area_name>_normalized.json  ← cleaned BusinessPOI objects

Usage:
  python -m scripts.run_collector --area tel_aviv \
      --south 32.05 --west 34.76 --north 32.10 --east 34.82

  # Smaller test area (faster):
  python -m scripts.run_collector --area test \
      --south 32.07 --west 34.77 --north 32.08 --east 34.79
"""

import argparse
import sys
from pathlib import Path

from tqdm import tqdm

from config import settings
from data_collector.overpass_client import fetch_pois, save_raw
from data_collector.normalizer import normalize, save_normalized


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Collect POI data for an area")
    p.add_argument("--area",  required=True, help="Short name for this area (used in filenames)")
    p.add_argument("--south", required=True, type=float)
    p.add_argument("--west",  required=True, type=float)
    p.add_argument("--north", required=True, type=float)
    p.add_argument("--east",  required=True, type=float)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    bbox = (args.south, args.west, args.north, args.east)
    raw_path  = settings.raw_data_dir / f"{args.area}.json"
    norm_path = settings.raw_data_dir / f"{args.area}_normalized.json"

    print(f"[1/3] Fetching POIs for bbox {bbox} …")
    raw_pois = fetch_pois(bbox)
    print(f"      → {len(raw_pois)} raw elements fetched")

    print(f"[2/3] Saving raw data → {raw_path}")
    save_raw(raw_pois, raw_path)

    print(f"[3/3] Normalising …")
    norm_pois = normalize(raw_pois)
    print(f"      → {len(norm_pois)} valid businesses after normalisation")
    save_normalized(norm_pois, norm_path)

    print(f"\nDone. Normalised data saved to: {norm_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
