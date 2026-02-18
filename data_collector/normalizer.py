"""
Normalizer.

Converts raw OSM POIs into clean BusinessPOI objects:
- picks the best human-readable category from OSM tags
- fills a description string used later for embedding generation
- assigns an H3 hexagonal cell index to each POI
"""

import json
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import h3

from config import settings
from data_collector.overpass_client import RawPOI


# ---------------------------------------------------------------------------
# Output shape
# ---------------------------------------------------------------------------

@dataclass
class BusinessPOI:
    osm_id: str
    name: str
    lat: float
    lon: float
    category: str           # normalized top-level category  (e.g. "cafe")
    category_group: str     # broad group                    (e.g. "Food & Drink")
    description: str        # free-text used for embedding
    h3_cell: str            # H3 index at settings.h3_resolution
    tags: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# OSM tag → normalized category mapping
# Format: tag_key → { tag_value → (category, category_group) }
# ---------------------------------------------------------------------------

OSM_CATEGORY_MAP: dict[str, dict[str, tuple[str, str]]] = {
    "amenity": {
        # Food & drink
        "cafe":             ("cafe",             "Food & Drink"),
        "restaurant":       ("restaurant",        "Food & Drink"),
        "fast_food":        ("fast food",         "Food & Drink"),
        "bar":              ("bar",               "Food & Drink"),
        "pub":              ("pub",               "Food & Drink"),
        "food_court":       ("food court",        "Food & Drink"),
        "ice_cream":        ("ice cream shop",    "Food & Drink"),
        "juice_bar":        ("juice bar",         "Food & Drink"),
        # Health & body
        "pharmacy":         ("pharmacy",          "Health & Body"),
        "clinic":           ("clinic",            "Health & Body"),
        "dentist":          ("dentist",           "Health & Body"),
        "doctors":          ("medical clinic",    "Health & Body"),
        "hospital":         ("hospital",          "Health & Body"),
        # Education
        "school":           ("school",            "Education"),
        "university":       ("university",        "Education"),
        "college":          ("college",           "Education"),
        "library":          ("library",           "Education"),
        # Fitness
        "gym":              ("gym",               "Fitness & Sport"),
        # Finance
        "bank":             ("bank",              "Finance"),
        "atm":              ("atm",               "Finance"),
    },
    "shop": {
        # Retail food
        "supermarket":      ("supermarket",       "Retail – Food"),
        "convenience":      ("convenience store", "Retail – Food"),
        "bakery":           ("bakery",            "Retail – Food"),
        "butcher":          ("butcher",           "Retail – Food"),
        "greengrocer":      ("greengrocer",       "Retail – Food"),
        "deli":             ("deli",              "Retail – Food"),
        "health_food":      ("health food store", "Retail – Food"),
        # Retail general
        "clothes":          ("clothing store",    "Retail – General"),
        "shoes":            ("shoe store",        "Retail – General"),
        "electronics":      ("electronics store", "Retail – General"),
        "books":            ("bookstore",         "Retail – General"),
        "sports":           ("sports store",      "Retail – General"),
        # Beauty
        "hairdresser":      ("hair salon",        "Beauty & Wellness"),
        "beauty":           ("beauty salon",      "Beauty & Wellness"),
        "cosmetics":        ("cosmetics store",   "Beauty & Wellness"),
        "massage":          ("massage studio",    "Beauty & Wellness"),
        "nail_salon":       ("nail salon",        "Beauty & Wellness"),
    },
    "leisure": {
        "fitness_centre":   ("fitness center",    "Fitness & Sport"),
        "sports_centre":    ("sports center",     "Fitness & Sport"),
        "swimming_pool":    ("swimming pool",     "Fitness & Sport"),
        "dance":            ("dance studio",      "Fitness & Sport"),
        "yoga":             ("yoga studio",       "Fitness & Sport"),
        "park":             ("park",              "Outdoors"),
        "playground":       ("playground",        "Outdoors"),
    },
    "tourism": {
        "hotel":            ("hotel",             "Hospitality"),
        "hostel":           ("hostel",            "Hospitality"),
        "guest_house":      ("guest house",       "Hospitality"),
        "attraction":       ("attraction",        "Tourism"),
        "museum":           ("museum",            "Tourism"),
    },
    "office": {
        "company":          ("office",            "Office & Work"),
        "government":       ("government office", "Office & Work"),
        "coworking":        ("coworking space",   "Office & Work"),
        "it":               ("tech office",       "Office & Work"),
    },
    "healthcare": {
        "clinic":           ("clinic",            "Health & Body"),
        "physiotherapist":  ("physiotherapy",     "Health & Body"),
        "alternative":      ("alternative health","Health & Body"),
    },
}

# Tag keys checked in priority order when looking up the category
PRIORITY_TAG_KEYS = ["amenity", "shop", "leisure", "tourism", "office", "healthcare"]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def normalize(raw_pois: list[RawPOI]) -> list[BusinessPOI]:
    """
    Convert a list of RawPOI objects into cleaned BusinessPOI objects.
    POIs without a recognizable name or category are dropped.
    """
    result: list[BusinessPOI] = []

    for poi in raw_pois:
        name = _extract_name(poi.tags)
        if not name:
            continue

        category, group = _extract_category(poi.tags)
        if not category:
            continue

        description = _build_description(name, category, group, poi.tags)
        h3_cell = h3.latlng_to_cell(poi.lat, poi.lon, settings.h3_resolution)

        result.append(
            BusinessPOI(
                osm_id=poi.osm_id,
                name=name,
                lat=poi.lat,
                lon=poi.lon,
                category=category,
                category_group=group,
                description=description,
                h3_cell=h3_cell,
                tags=poi.tags,
            )
        )

    return result


def save_normalized(pois: list[BusinessPOI], path) -> None:
    """Write normalized POIs to a JSON file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump([asdict(p) for p in pois], f, ensure_ascii=False, indent=2)


def load_normalized(path) -> list[BusinessPOI]:
    """Load normalized POIs from a JSON file."""
    with open(path, encoding="utf-8") as f:
        records = json.load(f)
    return [BusinessPOI(**r) for r in records]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_name(tags: dict) -> Optional[str]:
    """Return the best available name from OSM tags."""
    for key in ("name:en", "name", "brand", "operator"):
        value = tags.get(key, "").strip()
        if value:
            return value
    return None


def _extract_category(tags: dict) -> tuple[Optional[str], Optional[str]]:
    """
    Walk PRIORITY_TAG_KEYS and return the first (category, group) match.
    Falls back to a cleaned version of the raw OSM tag value if no
    explicit mapping exists.
    """
    for key in PRIORITY_TAG_KEYS:
        value = tags.get(key)
        if not value:
            continue

        mapping = OSM_CATEGORY_MAP.get(key, {})
        if value in mapping:
            return mapping[value]

        # Unrecognized but has the key — use the raw value as category
        cleaned = _clean_tag_value(value)
        group = _guess_group(key)
        return cleaned, group

    return None, None


def _clean_tag_value(value: str) -> str:
    """Turn OSM underscored values like 'nail_salon' into 'nail salon'."""
    return re.sub(r"[_\-]+", " ", value).strip().lower()


def _guess_group(tag_key: str) -> str:
    """Fallback group name based on which OSM key was matched."""
    return {
        "amenity":    "General",
        "shop":       "Retail",
        "leisure":    "Fitness & Sport",
        "tourism":    "Tourism",
        "office":     "Office & Work",
        "healthcare": "Health & Body",
    }.get(tag_key, "Other")


def _build_description(
    name: str,
    category: str,
    group: str,
    tags: dict,
) -> str:
    """
    Build a short free-text description used as input to the embedding model.
    Combines name, category, group, and any extra tags that enrich meaning.
    """
    parts = [name, category, group]

    # Add cuisine info for food businesses
    if cuisine := tags.get("cuisine"):
        parts.append(cuisine.replace(";", " ").replace("_", " "))

    # Add sport type for fitness/sport
    if sport := tags.get("sport"):
        parts.append(sport.replace(";", " ").replace("_", " "))

    return " ".join(parts)
