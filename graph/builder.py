"""
Graph Builder.

Creates and maintains the Neo4j graph structure:

  Nodes:   Business, Category, Area
  Edges:   BUSINESS -[:BELONGS_TO]->  CATEGORY
           BUSINESS -[:LOCATED_IN]->  AREA
           AREA     -[:NEAR]->        AREA
"""

from __future__ import annotations

import h3
from neo4j import Driver

from data_collector.normalizer import BusinessPOI, SUPPORT_CATEGORIES


class GraphBuilder:
    def __init__(self, driver: Driver) -> None:
        self._driver = driver

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def create_constraints(self) -> None:
        """Create uniqueness constraints so MERGE is idempotent."""
        statements = [
            "CREATE CONSTRAINT business_osm_id IF NOT EXISTS "
            "FOR (b:Business) REQUIRE b.osm_id IS UNIQUE",

            "CREATE CONSTRAINT category_name IF NOT EXISTS "
            "FOR (c:Category) REQUIRE c.name IS UNIQUE",

            "CREATE CONSTRAINT area_h3_index IF NOT EXISTS "
            "FOR (a:Area) REQUIRE a.h3_index IS UNIQUE",
        ]
        with self._driver.session() as session:
            for stmt in statements:
                session.run(stmt)

    # ------------------------------------------------------------------
    # Node upserts
    # ------------------------------------------------------------------

    def upsert_business(self, poi: BusinessPOI) -> None:
        """MERGE a Business node and set its properties."""
        with self._driver.session() as session:
            session.run(
                """
                MERGE (b:Business {osm_id: $osm_id})
                SET b.name     = $name,
                    b.lat      = $lat,
                    b.lon      = $lon,
                    b.category = $category,
                    b.h3_cell  = $h3_cell
                """,
                osm_id=poi.osm_id,
                name=poi.name,
                lat=poi.lat,
                lon=poi.lon,
                category=poi.category,
                h3_cell=poi.h3_cell,
            )

    def upsert_category(self, name: str, group: str) -> None:
        """MERGE a Category node."""
        with self._driver.session() as session:
            session.run(
                """
                MERGE (c:Category {name: $name})
                SET c.group = $group
                """,
                name=name,
                group=group,
            )

    def upsert_area(self, h3_cell: str) -> None:
        """MERGE an Area node for a single H3 cell."""
        with self._driver.session() as session:
            session.run(
                """
                MERGE (a:Area {h3_index: $h3_index})
                """,
                h3_index=h3_cell,
            )

    # ------------------------------------------------------------------
    # Edge creation
    # ------------------------------------------------------------------

    def link_business_to_category(self, osm_id: str, category: str) -> None:
        """MERGE (Business)-[:BELONGS_TO]->(Category)."""
        with self._driver.session() as session:
            session.run(
                """
                MATCH (b:Business {osm_id: $osm_id})
                MATCH (c:Category {name: $category})
                MERGE (b)-[:BELONGS_TO]->(c)
                """,
                osm_id=osm_id,
                category=category,
            )

    def link_business_to_area(self, osm_id: str, h3_cell: str) -> None:
        """MERGE (Business)-[:LOCATED_IN]->(Area)."""
        with self._driver.session() as session:
            session.run(
                """
                MATCH (b:Business {osm_id: $osm_id})
                MATCH (a:Area {h3_index: $h3_index})
                MERGE (b)-[:LOCATED_IN]->(a)
                """,
                osm_id=osm_id,
                h3_index=h3_cell,
            )

    def link_area_neighbors(self, h3_cell: str) -> None:
        """
        For each of the 6 H3 neighbours of *h3_cell*, MERGE an
        (Area)-[:NEAR]->(Area) edge.  The cell itself is excluded.
        """
        neighbours = set(h3.grid_disk(h3_cell, 1)) - {h3_cell}
        with self._driver.session() as session:
            for neighbour in neighbours:
                session.run(
                    """
                    MERGE (a:Area {h3_index: $h3_index})
                    MERGE (n:Area {h3_index: $neighbour_cell})
                    MERGE (a)-[:NEAR]->(n)
                    """,
                    h3_index=h3_cell,
                    neighbour_cell=neighbour,
                )

    def link_area_to_support_poi(self, h3_cell: str, osm_id: str) -> None:
        """
        MERGE (Area)-[:CLOSE_TO]->(Business) for a support POI.

        This edge captures demand anchors — universities, offices, transport
        hubs, and malls — that generate foot traffic near the area.
        """
        with self._driver.session() as session:
            session.run(
                """
                MATCH (a:Area {h3_index: $h3_index})
                MATCH (b:Business {osm_id: $osm_id})
                MERGE (a)-[:CLOSE_TO]->(b)
                """,
                h3_index=h3_cell,
                osm_id=osm_id,
            )

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def build_from_pois(self, pois: list[BusinessPOI]) -> None:
        """
        Upsert all nodes and edges for a list of POIs.

        Categories and areas are deduplicated — each unique value is
        upserted only once even if many POIs share it.
        """
        if not pois:
            return

        # Deduplicated sets to avoid redundant upserts
        seen_categories: set[str] = set()
        seen_areas: set[str] = set()

        for poi in pois:
            # Nodes
            self.upsert_business(poi)

            if poi.category not in seen_categories:
                self.upsert_category(poi.category, poi.category_group)
                seen_categories.add(poi.category)

            if poi.h3_cell not in seen_areas:
                self.upsert_area(poi.h3_cell)
                self.link_area_neighbors(poi.h3_cell)
                seen_areas.add(poi.h3_cell)

            # Edges
            self.link_business_to_category(poi.osm_id, poi.category)
            self.link_business_to_area(poi.osm_id, poi.h3_cell)

            # CLOSE_TO edge for demand-anchor support POIs
            if poi.category in SUPPORT_CATEGORIES:
                self.link_area_to_support_poi(poi.h3_cell, poi.osm_id)
