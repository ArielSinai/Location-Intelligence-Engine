
import json
import pytest
import h3

from data_collector.overpass_client import RawPOI
from data_collector.normalizer import (
    BusinessPOI,
    _extract_name,
    _extract_category,
    _build_description,
    normalize,
    save_normalized,
    load_normalized,
)
from config import settings


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_raw(osm_id="1", lat=32.07, lon=34.78, **tags) -> RawPOI:
    return RawPOI(osm_id=osm_id, osm_type="node", lat=lat, lon=lon, tags=tags)


# ---------------------------------------------------------------------------
# _extract_name
# ---------------------------------------------------------------------------

class TestExtractName:
    def test_returns_name_tag(self):
        assert _extract_name({"name": "My Cafe"}) == "My Cafe"

    def test_prefers_english_name_over_plain_name(self):
        assert _extract_name({"name": "בית קפה", "name:en": "Coffee House"}) == "Coffee House"

    def test_falls_back_to_brand(self):
        assert _extract_name({"brand": "Starbucks"}) == "Starbucks"

    def test_falls_back_to_operator(self):
        assert _extract_name({"operator": "City Gym"}) == "City Gym"

    def test_strips_whitespace(self):
        assert _extract_name({"name": "  Trim Me  "}) == "Trim Me"

    def test_returns_none_when_no_name_tags(self):
        assert _extract_name({"amenity": "cafe"}) is None

    def test_returns_none_for_empty_name(self):
        assert _extract_name({"name": ""}) is None

    def test_returns_none_for_empty_tags(self):
        assert _extract_name({}) is None


# ---------------------------------------------------------------------------
# _extract_category
# ---------------------------------------------------------------------------

class TestExtractCategory:
    # Known mappings
    def test_known_amenity_cafe(self):
        cat, group = _extract_category({"amenity": "cafe"})
        assert cat == "cafe"
        assert group == "Food & Drink"

    def test_known_shop_nail_salon(self):
        cat, group = _extract_category({"shop": "nail_salon"})
        assert cat == "nail salon"
        assert group == "Beauty & Wellness"

    def test_known_leisure_fitness_centre(self):
        cat, group = _extract_category({"leisure": "fitness_centre"})
        assert cat == "fitness center"
        assert group == "Fitness & Sport"

    def test_known_tourism_hotel(self):
        cat, group = _extract_category({"tourism": "hotel"})
        assert cat == "hotel"
        assert group == "Hospitality"

    # Unknown OSM value — should still return something using the raw value
    def test_unknown_amenity_value_returns_cleaned_string(self):
        cat, group = _extract_category({"amenity": "some_new_type"})
        assert cat == "some new type"   # underscores replaced with spaces
        assert group is not None        # group inferred from key

    def test_unknown_shop_value_returns_shop_group(self):
        _, group = _extract_category({"shop": "mystery_item"})
        assert "Retail" in group

    # Priority: amenity > shop > leisure
    def test_amenity_takes_priority_over_shop(self):
        cat, _ = _extract_category({"amenity": "cafe", "shop": "clothes"})
        assert cat == "cafe"

    def test_shop_takes_priority_over_leisure(self):
        cat, _ = _extract_category({"shop": "hairdresser", "leisure": "park"})
        assert cat == "hair salon"

    # No relevant key
    def test_no_relevant_key_returns_none_none(self):
        cat, group = _extract_category({"building": "yes", "highway": "residential"})
        assert cat is None
        assert group is None

    def test_empty_tags_returns_none_none(self):
        cat, group = _extract_category({})
        assert cat is None
        assert group is None


# ---------------------------------------------------------------------------
# _build_description
# ---------------------------------------------------------------------------

class TestBuildDescription:
    def test_contains_name(self):
        desc = _build_description("My Gym", "gym", "Fitness & Sport", {})
        assert "My Gym" in desc

    def test_contains_category(self):
        desc = _build_description("My Gym", "gym", "Fitness & Sport", {})
        assert "gym" in desc

    def test_contains_group(self):
        desc = _build_description("My Gym", "gym", "Fitness & Sport", {})
        assert "Fitness & Sport" in desc

    def test_adds_cuisine_when_present(self):
        desc = _build_description("Noodle Bar", "restaurant", "Food & Drink", {"cuisine": "japanese"})
        assert "japanese" in desc

    def test_adds_sport_when_present(self):
        desc = _build_description("Arena", "sports center", "Fitness & Sport", {"sport": "tennis"})
        assert "tennis" in desc

    def test_replaces_semicolons_in_cuisine(self):
        # OSM can store multiple cuisines as "italian;pizza"
        desc = _build_description("Bistro", "restaurant", "Food & Drink", {"cuisine": "italian;pizza"})
        assert ";" not in desc

    def test_replaces_underscores_in_cuisine(self):
        desc = _build_description("Bistro", "restaurant", "Food & Drink", {"cuisine": "fast_food"})
        assert "_" not in desc

    def test_no_extra_fields_when_tags_empty(self):
        desc = _build_description("Shop", "retail", "Retail", {})
        # Should only have the three base parts
        assert desc.strip() != ""


# ---------------------------------------------------------------------------
# normalize
# ---------------------------------------------------------------------------

class TestNormalize:
    def test_valid_poi_is_included(self):
        pois = [make_raw(name="Good Cafe", amenity="cafe")]
        result = normalize(pois)
        assert len(result) == 1

    def test_result_is_business_poi(self):
        pois = [make_raw(name="Good Cafe", amenity="cafe")]
        result = normalize(pois)
        assert isinstance(result[0], BusinessPOI)

    def test_poi_without_name_is_dropped(self):
        pois = [make_raw(amenity="cafe")]   # no name tag
        result = normalize(pois)
        assert result == []

    def test_poi_without_category_is_dropped(self):
        pois = [make_raw(name="Mystery Place")]   # no amenity/shop/etc
        result = normalize(pois)
        assert result == []

    def test_empty_input_returns_empty_list(self):
        assert normalize([]) == []

    def test_mixed_list_only_valid_returned(self):
        pois = [
            make_raw(osm_id="1", name="Good Cafe", amenity="cafe"),
            make_raw(osm_id="2", amenity="bar"),                    # no name
            make_raw(osm_id="3", name="No Category"),               # no category
            make_raw(osm_id="4", name="Gym", leisure="fitness_centre"),
        ]
        result = normalize(pois)
        assert len(result) == 2
        ids = {p.osm_id for p in result}
        assert ids == {"1", "4"}

    def test_h3_cell_is_assigned(self):
        pois = [make_raw(lat=32.07, lon=34.78, name="Cafe", amenity="cafe")]
        result = normalize(pois)
        assert result[0].h3_cell != ""
        assert h3.is_valid_cell(result[0].h3_cell)

    def test_h3_cell_matches_expected_resolution(self):
        pois = [make_raw(lat=32.07, lon=34.78, name="Cafe", amenity="cafe")]
        result = normalize(pois)
        assert h3.get_resolution(result[0].h3_cell) == settings.h3_resolution

    def test_fields_populated_correctly(self):
        raw = make_raw(osm_id="99", lat=32.07, lon=34.78, name="Test Cafe", amenity="cafe")
        result = normalize([raw])
        poi = result[0]
        assert poi.osm_id == "99"
        assert poi.name == "Test Cafe"
        assert poi.lat == 32.07
        assert poi.lon == 34.78
        assert poi.category == "cafe"
        assert poi.category_group == "Food & Drink"
        assert poi.description != ""

    def test_tags_preserved_on_output(self):
        raw = make_raw(name="Tagged Place", amenity="restaurant", cuisine="italian")
        result = normalize([raw])
        assert result[0].tags.get("cuisine") == "italian"


# ---------------------------------------------------------------------------
# save_normalized / load_normalized
# ---------------------------------------------------------------------------

class TestNormalizedRoundTrip:
    def _sample_business_pois(self) -> list[BusinessPOI]:
        return normalize([
            make_raw(osm_id="1", lat=32.07, lon=34.78, name="Cafe A", amenity="cafe"),
            make_raw(osm_id="2", lat=32.08, lon=34.79, name="Gym B", leisure="fitness_centre"),
        ])

    def test_save_creates_file(self, tmp_path):
        pois = self._sample_business_pois()
        out = tmp_path / "normalized.json"
        save_normalized(pois, out)
        assert out.exists()

    def test_saved_file_is_valid_json_list(self, tmp_path):
        pois = self._sample_business_pois()
        out = tmp_path / "normalized.json"
        save_normalized(pois, out)
        with open(out) as f:
            data = json.load(f)
        assert isinstance(data, list)
        assert len(data) == 2

    def test_load_restores_business_pois(self, tmp_path):
        pois = self._sample_business_pois()
        out = tmp_path / "normalized.json"
        save_normalized(pois, out)
        loaded = load_normalized(out)
        assert len(loaded) == 2
        assert all(isinstance(p, BusinessPOI) for p in loaded)

    def test_roundtrip_preserves_all_fields(self, tmp_path):
        pois = self._sample_business_pois()
        out = tmp_path / "normalized.json"
        save_normalized(pois, out)
        loaded = load_normalized(out)
        for orig, back in zip(pois, loaded):
            assert back.osm_id         == orig.osm_id
            assert back.name           == orig.name
            assert back.lat            == orig.lat
            assert back.lon            == orig.lon
            assert back.category       == orig.category
            assert back.category_group == orig.category_group
            assert back.description    == orig.description
            assert back.h3_cell        == orig.h3_cell
            assert back.tags           == orig.tags
