"""
Graph Queries.

Read-only Cypher queries used by the Opportunity Analyzer and
Recommendation Engine to interrogate the Neo4j graph.
"""

from __future__ import annotations

from neo4j import Driver


def get_competition_count(driver: Driver, h3_cell: str, category: str) -> int:
    """
    Count how many businesses of *category* are LOCATED_IN *h3_cell*.

    Returns 0 if the cell or category does not exist in the graph.
    """
    with driver.session() as session:
        result = session.run(
            """
            MATCH (b:Business)-[:LOCATED_IN]->(a:Area {h3_index: $h3_index})
            WHERE b.category = $category
            RETURN count(b) AS count
            """,
            h3_index=h3_cell,
            category=category,
        )
        record = result.single()
        if record is None:
            return 0
        return int(record["count"])


def get_area_businesses(driver: Driver, h3_cell: str) -> list[dict]:
    """
    Return all businesses located in *h3_cell* as plain dicts.
    """
    with driver.session() as session:
        result = session.run(
            """
            MATCH (b:Business)-[:LOCATED_IN]->(a:Area {h3_index: $h3_index})
            RETURN b
            """,
            h3_index=h3_cell,
        )
        return [record.data() for record in result]


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
