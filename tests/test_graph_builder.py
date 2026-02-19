
import pytest
import h3

from data_collector.normalizer import BusinessPOI


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CELL = "882db0cc39fffff"   # real H3 cell (Tel Aviv area, res 8)

def make_poi(osm_id="1", h3_cell=CELL) -> BusinessPOI:
    return BusinessPOI(
        osm_id=osm_id,
        name="Test Cafe",
        lat=32.07,
        lon=34.78,
        category="cafe",
        category_group="Food & Drink",
        description="Test Cafe cafe Food & Drink",
        h3_cell=h3_cell,
        tags={},
    )


# ---------------------------------------------------------------------------
# Fixture: mock Neo4j driver + session
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_driver(mocker):
    """
    A mock neo4j.Driver whose session() is a context manager
    returning a mock session with a callable .run() method.
    """
    mock_session = mocker.MagicMock()
    mock_session.run.return_value = mocker.MagicMock()

    mock_ctx = mocker.MagicMock()
    mock_ctx.__enter__ = mocker.MagicMock(return_value=mock_session)
    mock_ctx.__exit__ = mocker.MagicMock(return_value=False)

    driver = mocker.MagicMock()
    driver.session.return_value = mock_ctx

    # Attach session to driver for easy access in tests
    driver._mock_session = mock_session
    return driver


# ---------------------------------------------------------------------------
# GraphBuilder — create_constraints
# ---------------------------------------------------------------------------

class TestCreateConstraints:
    def test_runs_at_least_one_query(self, mock_driver):
        from graph.builder import GraphBuilder
        gb = GraphBuilder(mock_driver)
        gb.create_constraints()
        assert mock_driver._mock_session.run.called

    def test_creates_business_constraint(self, mock_driver):
        from graph.builder import GraphBuilder
        gb = GraphBuilder(mock_driver)
        gb.create_constraints()
        all_queries = " ".join(
            str(call.args[0]) for call in mock_driver._mock_session.run.call_args_list
        )
        assert "Business" in all_queries

    def test_creates_category_constraint(self, mock_driver):
        from graph.builder import GraphBuilder
        gb = GraphBuilder(mock_driver)
        gb.create_constraints()
        all_queries = " ".join(
            str(call.args[0]) for call in mock_driver._mock_session.run.call_args_list
        )
        assert "Category" in all_queries

    def test_creates_area_constraint(self, mock_driver):
        from graph.builder import GraphBuilder
        gb = GraphBuilder(mock_driver)
        gb.create_constraints()
        all_queries = " ".join(
            str(call.args[0]) for call in mock_driver._mock_session.run.call_args_list
        )
        assert "Area" in all_queries


# ---------------------------------------------------------------------------
# GraphBuilder — upsert_business
# ---------------------------------------------------------------------------

class TestUpsertBusiness:
    def test_runs_a_query(self, mock_driver):
        from graph.builder import GraphBuilder
        gb = GraphBuilder(mock_driver)
        gb.upsert_business(make_poi())
        assert mock_driver._mock_session.run.called

    def test_query_uses_merge(self, mock_driver):
        from graph.builder import GraphBuilder
        gb = GraphBuilder(mock_driver)
        gb.upsert_business(make_poi())
        query = mock_driver._mock_session.run.call_args.args[0]
        assert "MERGE" in query.upper()

    def test_query_targets_business_label(self, mock_driver):
        from graph.builder import GraphBuilder
        gb = GraphBuilder(mock_driver)
        gb.upsert_business(make_poi())
        query = mock_driver._mock_session.run.call_args.args[0]
        assert "Business" in query

    def test_passes_osm_id_as_parameter(self, mock_driver):
        from graph.builder import GraphBuilder
        gb = GraphBuilder(mock_driver)
        gb.upsert_business(make_poi(osm_id="42"))
        kwargs = mock_driver._mock_session.run.call_args.kwargs
        all_params = str(kwargs)
        assert "42" in all_params

    def test_passes_name_as_parameter(self, mock_driver):
        from graph.builder import GraphBuilder
        gb = GraphBuilder(mock_driver)
        gb.upsert_business(make_poi())
        kwargs = mock_driver._mock_session.run.call_args.kwargs
        all_params = str(kwargs)
        assert "Test Cafe" in all_params


# ---------------------------------------------------------------------------
# GraphBuilder — upsert_category
# ---------------------------------------------------------------------------

class TestUpsertCategory:
    def test_runs_a_query(self, mock_driver):
        from graph.builder import GraphBuilder
        gb = GraphBuilder(mock_driver)
        gb.upsert_category("cafe", "Food & Drink")
        assert mock_driver._mock_session.run.called

    def test_query_targets_category_label(self, mock_driver):
        from graph.builder import GraphBuilder
        gb = GraphBuilder(mock_driver)
        gb.upsert_category("cafe", "Food & Drink")
        query = mock_driver._mock_session.run.call_args.args[0]
        assert "Category" in query

    def test_passes_category_name(self, mock_driver):
        from graph.builder import GraphBuilder
        gb = GraphBuilder(mock_driver)
        gb.upsert_category("nail salon", "Beauty & Wellness")
        kwargs = mock_driver._mock_session.run.call_args.kwargs
        assert "nail salon" in str(kwargs)

    def test_passes_group(self, mock_driver):
        from graph.builder import GraphBuilder
        gb = GraphBuilder(mock_driver)
        gb.upsert_category("nail salon", "Beauty & Wellness")
        kwargs = mock_driver._mock_session.run.call_args.kwargs
        assert "Beauty & Wellness" in str(kwargs)


# ---------------------------------------------------------------------------
# GraphBuilder — upsert_area
# ---------------------------------------------------------------------------

class TestUpsertArea:
    def test_runs_a_query(self, mock_driver):
        from graph.builder import GraphBuilder
        gb = GraphBuilder(mock_driver)
        gb.upsert_area(CELL)
        assert mock_driver._mock_session.run.called

    def test_query_targets_area_label(self, mock_driver):
        from graph.builder import GraphBuilder
        gb = GraphBuilder(mock_driver)
        gb.upsert_area(CELL)
        query = mock_driver._mock_session.run.call_args.args[0]
        assert "Area" in query

    def test_passes_h3_cell_as_parameter(self, mock_driver):
        from graph.builder import GraphBuilder
        gb = GraphBuilder(mock_driver)
        gb.upsert_area(CELL)
        kwargs = mock_driver._mock_session.run.call_args.kwargs
        assert CELL in str(kwargs)


# ---------------------------------------------------------------------------
# GraphBuilder — link_business_to_category
# ---------------------------------------------------------------------------

class TestLinkBusinessToCategory:
    def test_runs_a_query(self, mock_driver):
        from graph.builder import GraphBuilder
        gb = GraphBuilder(mock_driver)
        gb.link_business_to_category("1", "cafe")
        assert mock_driver._mock_session.run.called

    def test_query_creates_belongs_to_relationship(self, mock_driver):
        from graph.builder import GraphBuilder
        gb = GraphBuilder(mock_driver)
        gb.link_business_to_category("1", "cafe")
        query = mock_driver._mock_session.run.call_args.args[0]
        assert "BELONGS_TO" in query

    def test_passes_osm_id_and_category(self, mock_driver):
        from graph.builder import GraphBuilder
        gb = GraphBuilder(mock_driver)
        gb.link_business_to_category("99", "gym")
        kwargs = mock_driver._mock_session.run.call_args.kwargs
        all_params = str(kwargs)
        assert "99" in all_params
        assert "gym" in all_params


# ---------------------------------------------------------------------------
# GraphBuilder — link_business_to_area
# ---------------------------------------------------------------------------

class TestLinkBusinessToArea:
    def test_runs_a_query(self, mock_driver):
        from graph.builder import GraphBuilder
        gb = GraphBuilder(mock_driver)
        gb.link_business_to_area("1", CELL)
        assert mock_driver._mock_session.run.called

    def test_query_creates_located_in_relationship(self, mock_driver):
        from graph.builder import GraphBuilder
        gb = GraphBuilder(mock_driver)
        gb.link_business_to_area("1", CELL)
        query = mock_driver._mock_session.run.call_args.args[0]
        assert "LOCATED_IN" in query

    def test_passes_osm_id_and_h3_cell(self, mock_driver):
        from graph.builder import GraphBuilder
        gb = GraphBuilder(mock_driver)
        gb.link_business_to_area("42", CELL)
        kwargs = mock_driver._mock_session.run.call_args.kwargs
        all_params = str(kwargs)
        assert "42" in all_params
        assert CELL in all_params


# ---------------------------------------------------------------------------
# GraphBuilder — link_area_neighbors
# ---------------------------------------------------------------------------

class TestLinkAreaNeighbors:
    def test_runs_at_least_one_query(self, mock_driver):
        from graph.builder import GraphBuilder
        gb = GraphBuilder(mock_driver)
        gb.link_area_neighbors(CELL)
        assert mock_driver._mock_session.run.called

    def test_query_creates_near_relationship(self, mock_driver):
        from graph.builder import GraphBuilder
        gb = GraphBuilder(mock_driver)
        gb.link_area_neighbors(CELL)
        all_queries = " ".join(
            str(call.args[0]) for call in mock_driver._mock_session.run.call_args_list
        )
        assert "NEAR" in all_queries

    def test_links_all_six_neighbours(self, mock_driver):
        from graph.builder import GraphBuilder
        gb = GraphBuilder(mock_driver)
        gb.link_area_neighbors(CELL)
        # grid_disk(cell, 1) = cell + 6 neighbours = 7 cells
        # We should create one NEAR edge per neighbour (not self)
        neighbours = set(h3.grid_disk(CELL, 1)) - {CELL}
        assert mock_driver._mock_session.run.call_count == len(neighbours)

    def test_does_not_link_cell_to_itself(self, mock_driver):
        from graph.builder import GraphBuilder
        gb = GraphBuilder(mock_driver)
        gb.link_area_neighbors(CELL)
        all_params = str(mock_driver._mock_session.run.call_args_list)
        # Count how many times CELL appears as BOTH source and target
        # The cell itself should never be the neighbor target
        for call in mock_driver._mock_session.run.call_args_list:
            kwargs = call.kwargs
            neighbour = kwargs.get("neighbour_cell") or kwargs.get("neighbor_cell")
            if neighbour:
                assert neighbour != CELL


# ---------------------------------------------------------------------------
# GraphBuilder — build_from_pois
# ---------------------------------------------------------------------------

class TestBuildFromPois:
    def test_upserts_business_for_each_poi(self, mock_driver, mocker):
        from graph.builder import GraphBuilder
        gb = GraphBuilder(mock_driver)
        spy = mocker.patch.object(gb, "upsert_business")
        pois = [make_poi("1"), make_poi("2"), make_poi("3")]
        gb.build_from_pois(pois)
        assert spy.call_count == 3

    def test_upserts_category_for_each_poi(self, mock_driver, mocker):
        from graph.builder import GraphBuilder
        gb = GraphBuilder(mock_driver)
        spy = mocker.patch.object(gb, "upsert_category")
        pois = [make_poi("1"), make_poi("2")]
        gb.build_from_pois(pois)
        assert spy.call_count >= 1   # deduplicated categories OK

    def test_upserts_area_for_each_poi(self, mock_driver, mocker):
        from graph.builder import GraphBuilder
        gb = GraphBuilder(mock_driver)
        spy = mocker.patch.object(gb, "upsert_area")
        pois = [make_poi("1"), make_poi("2")]
        gb.build_from_pois(pois)
        assert spy.call_count >= 1

    def test_links_business_to_category(self, mock_driver, mocker):
        from graph.builder import GraphBuilder
        gb = GraphBuilder(mock_driver)
        spy = mocker.patch.object(gb, "link_business_to_category")
        pois = [make_poi("1"), make_poi("2")]
        gb.build_from_pois(pois)
        assert spy.call_count == 2

    def test_links_business_to_area(self, mock_driver, mocker):
        from graph.builder import GraphBuilder
        gb = GraphBuilder(mock_driver)
        spy = mocker.patch.object(gb, "link_business_to_area")
        pois = [make_poi("1"), make_poi("2")]
        gb.build_from_pois(pois)
        assert spy.call_count == 2

    def test_links_area_neighbors(self, mock_driver, mocker):
        from graph.builder import GraphBuilder
        gb = GraphBuilder(mock_driver)
        spy = mocker.patch.object(gb, "link_area_neighbors")
        pois = [make_poi("1")]
        gb.build_from_pois(pois)
        assert spy.call_count >= 1

    def test_empty_poi_list_runs_nothing(self, mock_driver, mocker):
        from graph.builder import GraphBuilder
        gb = GraphBuilder(mock_driver)
        spy_b = mocker.patch.object(gb, "upsert_business")
        gb.build_from_pois([])
        spy_b.assert_not_called()


# ---------------------------------------------------------------------------
# queries.py — get_competition_count
# ---------------------------------------------------------------------------

class TestGetCompetitionCount:
    def _make_result(self, mocker, count: int):
        record = mocker.MagicMock()
        record.__getitem__ = mocker.MagicMock(return_value=count)
        result = mocker.MagicMock()
        result.single.return_value = record
        return result

    def test_returns_integer(self, mock_driver, mocker):
        from graph.queries import get_competition_count
        mock_driver._mock_session.run.return_value = self._make_result(mocker, 3)
        result = get_competition_count(mock_driver, CELL, "cafe")
        assert isinstance(result, int)

    def test_returns_correct_count(self, mock_driver, mocker):
        from graph.queries import get_competition_count
        mock_driver._mock_session.run.return_value = self._make_result(mocker, 7)
        result = get_competition_count(mock_driver, CELL, "cafe")
        assert result == 7

    def test_passes_cell_and_category_to_query(self, mock_driver, mocker):
        from graph.queries import get_competition_count
        mock_driver._mock_session.run.return_value = self._make_result(mocker, 0)
        get_competition_count(mock_driver, CELL, "gym")
        kwargs = mock_driver._mock_session.run.call_args.kwargs
        all_params = str(kwargs)
        assert CELL in all_params
        assert "gym" in all_params

    def test_returns_zero_when_no_results(self, mock_driver, mocker):
        from graph.queries import get_competition_count
        result_mock = mocker.MagicMock()
        result_mock.single.return_value = None
        mock_driver._mock_session.run.return_value = result_mock
        result = get_competition_count(mock_driver, CELL, "cafe")
        assert result == 0


# ---------------------------------------------------------------------------
# queries.py — get_area_businesses
# ---------------------------------------------------------------------------

class TestGetAreaBusinesses:
    def _make_records(self, mocker, data: list[dict]):
        records = []
        for d in data:
            r = mocker.MagicMock()
            r.data.return_value = d
            records.append(r)
        result = mocker.MagicMock()
        result.__iter__ = mocker.MagicMock(return_value=iter(records))
        return result

    def test_returns_list(self, mock_driver, mocker):
        from graph.queries import get_area_businesses
        mock_driver._mock_session.run.return_value = self._make_records(mocker, [])
        result = get_area_businesses(mock_driver, CELL)
        assert isinstance(result, list)

    def test_returns_one_dict_per_business(self, mock_driver, mocker):
        from graph.queries import get_area_businesses
        data = [{"name": "Cafe A", "category": "cafe"}, {"name": "Gym B", "category": "gym"}]
        mock_driver._mock_session.run.return_value = self._make_records(mocker, data)
        result = get_area_businesses(mock_driver, CELL)
        assert len(result) == 2

    def test_passes_h3_cell_to_query(self, mock_driver, mocker):
        from graph.queries import get_area_businesses
        mock_driver._mock_session.run.return_value = self._make_records(mocker, [])
        get_area_businesses(mock_driver, CELL)
        kwargs = mock_driver._mock_session.run.call_args.kwargs
        assert CELL in str(kwargs)

    def test_empty_area_returns_empty_list(self, mock_driver, mocker):
        from graph.queries import get_area_businesses
        mock_driver._mock_session.run.return_value = self._make_records(mocker, [])
        result = get_area_businesses(mock_driver, CELL)
        assert result == []


# ---------------------------------------------------------------------------
# queries.py — get_neighbor_cells
# ---------------------------------------------------------------------------

class TestGetNeighborCells:
    def _make_cell_records(self, mocker, cells: list[str]):
        records = []
        for c in cells:
            r = mocker.MagicMock()
            r.__getitem__ = mocker.MagicMock(return_value=c)
            records.append(r)
        result = mocker.MagicMock()
        result.__iter__ = mocker.MagicMock(return_value=iter(records))
        return result

    def test_returns_list(self, mock_driver, mocker):
        from graph.queries import get_neighbor_cells
        mock_driver._mock_session.run.return_value = self._make_cell_records(mocker, [])
        result = get_neighbor_cells(mock_driver, CELL)
        assert isinstance(result, list)

    def test_returns_strings(self, mock_driver, mocker):
        from graph.queries import get_neighbor_cells
        neighbours = ["882db0cc07fffff", "882db0cc3dfffff"]
        mock_driver._mock_session.run.return_value = self._make_cell_records(mocker, neighbours)
        result = get_neighbor_cells(mock_driver, CELL)
        assert all(isinstance(c, str) for c in result)

    def test_passes_h3_cell_to_query(self, mock_driver, mocker):
        from graph.queries import get_neighbor_cells
        mock_driver._mock_session.run.return_value = self._make_cell_records(mocker, [])
        get_neighbor_cells(mock_driver, CELL)
        kwargs = mock_driver._mock_session.run.call_args.kwargs
        assert CELL in str(kwargs)

    def test_correct_number_of_neighbours_returned(self, mock_driver, mocker):
        from graph.queries import get_neighbor_cells
        neighbours = ["882db0cc07fffff", "882db0cc3dfffff", "882db0cc31fffff"]
        mock_driver._mock_session.run.return_value = self._make_cell_records(mocker, neighbours)
        result = get_neighbor_cells(mock_driver, CELL)
        assert len(result) == 3
