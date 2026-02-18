"""
Tests for data_collector/overpass_client.py

Covered:
- _build_query          : correct OverpassQL structure for a given bbox
- _parse_elements       : handles nodes, ways (center), relations, missing coords
- fetch_pois            : happy path, HTTP error retry, permanent failure
- fetch_pois_from_polygon: bbox derived correctly from polygon vertices
- save_raw / load_raw   : round-trip JSON serialization
"""

import json
import pytest
import requests

from data_collector.overpass_client import (
    RawPOI,
    _build_query,
    _parse_elements,
    fetch_pois,
    fetch_pois_from_polygon,
    save_raw,
    load_raw,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_BBOX = (32.05, 34.76, 32.10, 34.82)   # small Tel Aviv area

SAMPLE_ELEMENTS = [
    # Node — has direct lat/lon
    {
        "type": "node",
        "id": 1,
        "lat": 32.07,
        "lon": 34.78,
        "tags": {"amenity": "cafe", "name": "Test Cafe"},
    },
    # Way — has center instead of direct lat/lon
    {
        "type": "way",
        "id": 2,
        "center": {"lat": 32.08, "lon": 34.79},
        "tags": {"shop": "supermarket", "name": "Big Store"},
    },
    # Relation — also uses center
    {
        "type": "relation",
        "id": 3,
        "center": {"lat": 32.09, "lon": 34.80},
        "tags": {"leisure": "fitness_centre", "name": "My Gym"},
    },
    # Element with no tags — should be skipped
    {
        "type": "node",
        "id": 4,
        "lat": 32.06,
        "lon": 34.77,
        "tags": {},
    },
    # Way with missing center — should be skipped
    {
        "type": "way",
        "id": 5,
        "tags": {"amenity": "restaurant", "name": "Ghost Place"},
    },
]


# ---------------------------------------------------------------------------
# _build_query
# ---------------------------------------------------------------------------

class TestBuildQuery:
    def test_contains_bbox_values(self):
        query = _build_query(SAMPLE_BBOX)
        south, west, north, east = SAMPLE_BBOX
        assert str(south) in query
        assert str(west)  in query
        assert str(north) in query
        assert str(east)  in query

    def test_json_output_format(self):
        query = _build_query(SAMPLE_BBOX)
        assert "[out:json]" in query

    def test_all_relevant_tag_keys_present(self):
        from data_collector.overpass_client import RELEVANT_TAG_KEYS
        query = _build_query(SAMPLE_BBOX)
        for key in RELEVANT_TAG_KEYS:
            assert f'"{key}"' in query, f"Tag key '{key}' missing from query"

    def test_all_element_types_queried(self):
        query = _build_query(SAMPLE_BBOX)
        for element_type in ("node", "way", "relation"):
            assert element_type in query

    def test_out_center_tags_present(self):
        # Needed so ways/relations return lat/lon via their center
        query = _build_query(SAMPLE_BBOX)
        assert "out center tags" in query


# ---------------------------------------------------------------------------
# _parse_elements
# ---------------------------------------------------------------------------

class TestParseElements:
    def test_node_parsed_correctly(self):
        pois = _parse_elements([SAMPLE_ELEMENTS[0]])
        assert len(pois) == 1
        p = pois[0]
        assert p.osm_id == "1"
        assert p.osm_type == "node"
        assert p.lat == 32.07
        assert p.lon == 34.78
        assert p.tags == {"amenity": "cafe", "name": "Test Cafe"}

    def test_way_uses_center(self):
        pois = _parse_elements([SAMPLE_ELEMENTS[1]])
        assert len(pois) == 1
        assert pois[0].lat == 32.08
        assert pois[0].lon == 34.79

    def test_relation_uses_center(self):
        pois = _parse_elements([SAMPLE_ELEMENTS[2]])
        assert len(pois) == 1
        assert pois[0].osm_type == "relation"

    def test_element_with_no_tags_is_skipped(self):
        pois = _parse_elements([SAMPLE_ELEMENTS[3]])
        assert pois == []

    def test_way_without_center_is_skipped(self):
        pois = _parse_elements([SAMPLE_ELEMENTS[4]])
        assert pois == []

    def test_full_list_returns_three_valid_pois(self):
        # 5 elements in sample; 2 should be skipped (no tags, no center)
        pois = _parse_elements(SAMPLE_ELEMENTS)
        assert len(pois) == 3

    def test_empty_input_returns_empty_list(self):
        assert _parse_elements([]) == []


# ---------------------------------------------------------------------------
# fetch_pois (mocked HTTP)
# ---------------------------------------------------------------------------

class TestFetchPois:
    def _make_response(self, elements: list, status_code: int = 200, mocker=None):
        """Build a mock requests.Response."""
        mock_resp = mocker.Mock()
        mock_resp.status_code = status_code
        mock_resp.json.return_value = {"elements": elements}
        if status_code >= 400:
            mock_resp.raise_for_status.side_effect = requests.exceptions.HTTPError(
                response=mock_resp
            )
        else:
            mock_resp.raise_for_status.return_value = None
        return mock_resp

    def test_happy_path_returns_pois(self, mocker):
        mock_post = mocker.patch("data_collector.overpass_client.requests.post")
        mock_post.return_value = self._make_response(SAMPLE_ELEMENTS[:3], mocker=mocker)

        pois = fetch_pois(SAMPLE_BBOX)

        assert len(pois) == 3
        assert all(isinstance(p, RawPOI) for p in pois)
        mock_post.assert_called_once()

    def test_query_includes_bbox(self, mocker):
        mock_post = mocker.patch("data_collector.overpass_client.requests.post")
        mock_post.return_value = self._make_response([], mocker=mocker)

        fetch_pois(SAMPLE_BBOX)

        call_data = mock_post.call_args.kwargs["data"]["data"]
        south, west, north, east = SAMPLE_BBOX
        assert str(south) in call_data

    def test_retries_on_transient_failure_then_succeeds(self, mocker):
        mock_post = mocker.patch("data_collector.overpass_client.requests.post")
        mocker.patch("data_collector.overpass_client.time.sleep")  # no real waiting

        good_response = self._make_response(SAMPLE_ELEMENTS[:1], mocker=mocker)
        bad_response = self._make_response([], status_code=503, mocker=mocker)

        # First call fails, second succeeds
        mock_post.side_effect = [
            requests.exceptions.RequestException("timeout"),
            good_response,
        ]

        pois = fetch_pois(SAMPLE_BBOX, retries=2)
        assert len(pois) == 1
        assert mock_post.call_count == 2

    def test_raises_after_all_retries_exhausted(self, mocker):
        mock_post = mocker.patch("data_collector.overpass_client.requests.post")
        mocker.patch("data_collector.overpass_client.time.sleep")

        mock_post.side_effect = requests.exceptions.RequestException("always fails")

        with pytest.raises(RuntimeError, match="Overpass API failed"):
            fetch_pois(SAMPLE_BBOX, retries=2)

        assert mock_post.call_count == 2

    def test_empty_area_returns_empty_list(self, mocker):
        mock_post = mocker.patch("data_collector.overpass_client.requests.post")
        mock_post.return_value = self._make_response([], mocker=mocker)

        pois = fetch_pois(SAMPLE_BBOX)
        assert pois == []


# ---------------------------------------------------------------------------
# fetch_pois_from_polygon
# ---------------------------------------------------------------------------

class TestFetchPoisFromPolygon:
    def test_bbox_derived_from_polygon(self, mocker):
        mock_fetch = mocker.patch("data_collector.overpass_client.fetch_pois", return_value=[])

        polygon = [
            (32.05, 34.76),
            (32.10, 34.76),
            (32.10, 34.82),
            (32.05, 34.82),
        ]
        fetch_pois_from_polygon(polygon)

        called_bbox = mock_fetch.call_args.args[0]
        assert called_bbox == (32.05, 34.76, 32.10, 34.82)

    def test_single_point_polygon(self, mocker):
        mock_fetch = mocker.patch("data_collector.overpass_client.fetch_pois", return_value=[])
        fetch_pois_from_polygon([(32.07, 34.78)])
        called_bbox = mock_fetch.call_args.args[0]
        assert called_bbox == (32.07, 34.78, 32.07, 34.78)


# ---------------------------------------------------------------------------
# save_raw / load_raw
# ---------------------------------------------------------------------------

class TestRoundTrip:
    def _sample_pois(self) -> list[RawPOI]:
        return [
            RawPOI("1", "node", 32.07, 34.78, {"amenity": "cafe", "name": "A"}),
            RawPOI("2", "way",  32.08, 34.79, {"shop": "supermarket", "name": "B"}),
        ]

    def test_save_creates_file(self, tmp_path):
        pois = self._sample_pois()
        out = tmp_path / "raw.json"
        save_raw(pois, out)
        assert out.exists()

    def test_saved_file_is_valid_json(self, tmp_path):
        pois = self._sample_pois()
        out = tmp_path / "raw.json"
        save_raw(pois, out)
        with open(out) as f:
            data = json.load(f)
        assert isinstance(data, list)
        assert len(data) == 2

    def test_load_restores_pois(self, tmp_path):
        pois = self._sample_pois()
        out = tmp_path / "raw.json"
        save_raw(pois, out)
        loaded = load_raw(out)
        assert len(loaded) == len(pois)
        assert all(isinstance(p, RawPOI) for p in loaded)

    def test_roundtrip_preserves_fields(self, tmp_path):
        original = self._sample_pois()
        out = tmp_path / "raw.json"
        save_raw(original, out)
        loaded = load_raw(out)

        for orig, back in zip(original, loaded):
            assert back.osm_id       == orig.osm_id
            assert back.osm_type     == orig.osm_type
            assert back.lat          == orig.lat
            assert back.lon          == orig.lon
            assert back.tags         == orig.tags
