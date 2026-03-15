"""
Graph Queries.

Read-only Cypher queries used by the Opportunity Analyzer and
Recommendation Engine to interrogate the Neo4j graph.
"""

from __future__ import annotations

from neo4j import Driver



def get_area_businesses(driver: Driver, h3_cell: str) -> list[dict]:
    """
    Return all businesses located in *h3_cell* as plain dicts.
    """
    with driver.session() as session:
        result = session.run(
            """
            MATCH (b:Business)-[:LOCATED_IN]->(a:Area {h3_index: $h3_index})
            RETURN b.osm_id AS osm_id, b.name AS name,
                   b.category AS category, b.h3_cell AS h3_cell
            """,
            h3_index=h3_cell,
        )
        return [dict(record) for record in result]


def get_support_pois(driver: Driver, h3_cell: str) -> list[dict]:
    """
    Return all support POIs linked to *h3_cell* via CLOSE_TO edges.

    Support POIs are demand anchors (universities, offices, transport hubs,
    malls) that generate foot traffic near the area.

    Each dict contains 'osm_id', 'name', and 'category'.
    """
    with driver.session() as session:
        result = session.run(
            """
            MATCH (a:Area {h3_index: $h3_index})-[:CLOSE_TO]->(b:Business)
            RETURN b.osm_id AS osm_id, b.name AS name, b.category AS category
            """,
            h3_index=h3_cell,
        )
        return [dict(record) for record in result]


def get_neighbor_cells(driver: Driver, h3_cell: str) -> list[str]:
    """
    Return the H3 indices of all Area nodes directly NEAR *h3_cell*.
    """
    with driver.session() as session:
        result = session.run(
            """
            MATCH (a:Area {h3_index: $h3_index})-[:NEAR]->(n:Area)
            RETURN n.h3_index AS neighbour_cell
            """,
            h3_index=h3_cell,
        )
        return [record["neighbour_cell"] for record in result]
