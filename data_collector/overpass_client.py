"""
Overpass API client.

Fetches Points of Interest (POIs) from OpenStreetMap for a given
bounding box or polygon using the public Overpass API — no API key needed.
"""

import json
import time
import requests
from dataclasses import dataclass
from typing import Optional
from config import settings


# ---------------------------------------------------------------------------
# Data shape returned by this module
# ---------------------------------------------------------------------------

@dataclass
class RawPOI:
    osm_id: str
    osm_type: str           # "node" | "way" | "relation"
    lat: float
    lon: float
    tags: dict


# ---------------------------------------------------------------------------
# OSM tag groups we care about
# The query asks for any element that has at least one of these tag keys.
# ---------------------------------------------------------------------------

RELEVANT_TAG_KEYS = [
    "amenity",
    "shop",
    "leisure",
    "tourism",
    "office",
    "healthcare",
]


# ---------------------------------------------------------------------------
# Query builder
# ---------------------------------------------------------------------------

def _build_query(bbox: tuple[float, float, float, float]) -> str:
    """
    Build an OverpassQL query that returns all nodes/ways/relations
    inside *bbox* that carry at least one business-relevant tag.

    bbox = (south, west, north, east)  ← Overpass convention
    """
    south, west, north, east = bbox
    bbox_str = f"{south},{west},{north},{east}"

    tag_unions = "\n  ".join(
        f"""(
    node["{key}"]({bbox_str});
    way["{key}"]({bbox_str});
    relation["{key}"]({bbox_str});
  );"""
        for key in RELEVANT_TAG_KEYS
    )

    query = f"""
[out:json][timeout:60];
(
  {tag_unions}
);
out center tags;
"""
    return query.strip()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_pois(
    bbox: tuple[float, float, float, float],
    *,
    retries: int = 3,
    retry_delay: float = 5.0,
) -> list[RawPOI]:
    """
    Fetch all business-related POIs inside *bbox* from Overpass.

    Parameters
    ----------
    bbox : (south, west, north, east) in decimal degrees
    retries : number of retry attempts on transient failure
    retry_delay : seconds to wait between retries

    Returns
    -------
    List of RawPOI dataclass instances.
    """
    query = _build_query(bbox)

    for attempt in range(1, retries + 1):
        try:
            response = requests.post(
                settings.overpass_url,
                data={"data": query},
                timeout=90,
            )
            response.raise_for_status()
            data = response.json()
            return _parse_elements(data.get("elements", []))

        except requests.exceptions.RequestException as exc:
            if attempt == retries:
                raise RuntimeError(
                    f"Overpass API failed after {retries} attempts: {exc}"
                ) from exc
            print(f"  [overpass] attempt {attempt} failed ({exc}), retrying in {retry_delay}s …")
            time.sleep(retry_delay)

    return []   # unreachable, but satisfies type checkers


def fetch_pois_from_polygon(
    polygon_coords: list[tuple[float, float]],
) -> list[RawPOI]:
    """
    Convenience wrapper: derive the bounding box from a list of
    (lat, lon) polygon vertices and call fetch_pois.
    """
    lats = [c[0] for c in polygon_coords]
    lons = [c[1] for c in polygon_coords]
    bbox = (min(lats), min(lons), max(lats), max(lons))
    return fetch_pois(bbox)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_elements(elements: list[dict]) -> list[RawPOI]:
    pois: list[RawPOI] = []

    for el in elements:
        tags = el.get("tags", {})
        if not tags:
            continue

        # Ways and relations expose a "center" key with lat/lon
        if el["type"] == "node":
            lat, lon = el.get("lat"), el.get("lon")
        else:
            center = el.get("center", {})
            lat, lon = center.get("lat"), center.get("lon")

        if lat is None or lon is None:
            continue

        pois.append(
            RawPOI(
                osm_id=str(el["id"]),
                osm_type=el["type"],
                lat=float(lat),
                lon=float(lon),
                tags=tags,
            )
        )

    return pois


def save_raw(pois: list[RawPOI], path) -> None:
    """Persist raw POIs to a JSON file for later processing."""
    import pathlib
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            [
                {
                    "osm_id": p.osm_id,
                    "osm_type": p.osm_type,
                    "lat": p.lat,
                    "lon": p.lon,
                    "tags": p.tags,
                }
                for p in pois
            ],
            f,
            ensure_ascii=False,
            indent=2,
        )


def load_raw(path) -> list[RawPOI]:
    """Load previously saved raw POIs from a JSON file."""
    with open(path, encoding="utf-8") as f:
        records = json.load(f)
    return [
        RawPOI(
            osm_id=r["osm_id"],
            osm_type=r["osm_type"],
            lat=r["lat"],
            lon=r["lon"],
            tags=r["tags"],
        )
        for r in records
    ]
