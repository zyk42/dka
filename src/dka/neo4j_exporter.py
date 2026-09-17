"""
Neo4j exporter for DKA knowledge graphs.

Imports a DKA graph (NetworkX node-link JSON produced by `dka-build-kg`)
into a Neo4j database:

  - One node per entity, labelled by entity type (:Concept, :Person, ...).
    Nodes are keyed by `name` with a uniqueness constraint per label.
  - One relationship per edge, typed :RELATED by default, carrying
    weight / description / text_unit_ids (chunk provenance).

Node properties:
    name, type, description (list[str]), frequency (int), text_unit_ids (list[str])
Relationship properties:
    weight (float), description (list[str]), text_unit_ids (list[str])

Requires the optional `neo4j` package:  pip install dka[neo4j]
"""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from typing import Any

import networkx as nx

logger = logging.getLogger(__name__)

DEFAULT_REL_TYPE = "RELATED"


def _sanitize_value(val: Any) -> Any:
    """Coerce a property value into a Neo4j-storable primitive (or list of primitives)."""
    if val is None or isinstance(val, (str, int, float, bool)):
        return val
    if isinstance(val, (list, tuple, set)):
        arr = [_sanitize_value(v) for v in val]
        arr = ["null" if v is None else v for v in arr]
        if all(isinstance(v, (str, int, float, bool)) for v in arr):
            return arr
        return json.dumps(list(val), ensure_ascii=False, default=str)
    if isinstance(val, dict):
        return json.dumps(val, ensure_ascii=False, default=str)
    return str(val)


def _sanitize_props(props: dict) -> dict:
    return {k: _sanitize_value(v) for k, v in (props or {}).items() if v is not None}


def _label_for(entity_type: str | None) -> str:
    """Map an entity type to a safe Neo4j label, e.g. 'ORGANIZATION' -> 'Organization'."""
    if not entity_type:
        return "Entity"
    cleaned = re.sub(r"[^0-9A-Za-z_]", "_", str(entity_type).strip())
    if not cleaned:
        return "Entity"
    label = cleaned.capitalize()
    if label[0].isdigit():
        label = "T_" + label
    return label


def _rel_type_for(rel_type: str | None) -> str:
    if not rel_type:
        return DEFAULT_REL_TYPE
    cleaned = re.sub(r"[^0-9A-Za-z_]", "_", str(rel_type).strip().upper())
    if not cleaned:
        return DEFAULT_REL_TYPE
    if cleaned[0].isdigit():
        cleaned = "R_" + cleaned
    return cleaned


# ---------------------------------------------------------------------------
# Graph -> row tables
# ---------------------------------------------------------------------------

def graph_to_tables(G: nx.Graph, rel_type: str = DEFAULT_REL_TYPE) -> tuple[dict, dict]:
    """
    Convert a NetworkX graph to Neo4j import tables.

    Returns:
        entities_by_label: {label: [row, ...]}   row = {name, type, description, frequency, text_unit_ids}
        rel_groups:        {(src_label, rel_type, tgt_label): [row, ...]}
                           row = {source_name, target_name, weight, description, text_unit_ids}
    """
    entities_by_label: dict[str, list[dict]] = defaultdict(list)
    labels: dict[str, str] = {}

    for name, attrs in G.nodes(data=True):
        label = _label_for(attrs.get("type"))
        labels[name] = label
        row = {"name": name}
        row.update(_sanitize_props({
            "type": attrs.get("type", "UNKNOWN"),
            "description": attrs.get("description", []),
            "frequency": int(attrs.get("frequency", 1)),
            "text_unit_ids": attrs.get("text_unit_ids", []),
        }))
        entities_by_label[label].append(row)

    rel_groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for src, tgt, attrs in G.edges(data=True):
        if src not in labels or tgt not in labels:
            continue
        row = {"source_name": src, "target_name": tgt}
        row.update(_sanitize_props({
            "weight": float(attrs.get("weight", 1.0)),
            "description": attrs.get("description", []),
            "text_unit_ids": attrs.get("text_unit_ids", []),
        }))
        rel_groups[(labels[src], _rel_type_for(rel_type), labels[tgt])].append(row)

    return entities_by_label, rel_groups


# ---------------------------------------------------------------------------
# Neo4j import
# ---------------------------------------------------------------------------

def _create_constraints(session, labels: list[str]) -> None:
    for label in labels:
        session.run(
            f"CREATE CONSTRAINT `{label}_name_unique` IF NOT EXISTS "
            f"FOR (n:`{label}`) REQUIRE n.name IS UNIQUE"
        )


def _import_entities(session, entities_by_label: dict[str, list[dict]], batch_size: int) -> int:
    total = 0
    for label, rows in entities_by_label.items():
        for i in range(0, len(rows), batch_size):
            batch = rows[i : i + batch_size]
            session.run(
                f"UNWIND $rows AS row "
                f"MERGE (n:`{label}` {{name: row.name}}) "
                f"SET n.type = row.type, n.description = row.description, "
                f"n.frequency = row.frequency, n.text_unit_ids = row.text_unit_ids",
                rows=batch,
            )
            total += len(batch)
        logger.info("Imported %d :%s nodes", len(rows), label)
    return total


def _import_relations(session, rel_groups: dict[tuple[str, str, str], list[dict]], batch_size: int) -> int:
    total = 0
    for (src_label, rel_type, tgt_label), rows in rel_groups.items():
        for i in range(0, len(rows), batch_size):
            batch = rows[i : i + batch_size]
            session.run(
                f"UNWIND $rows AS row "
                f"MERGE (a:`{src_label}` {{name: row.source_name}}) "
                f"MERGE (b:`{tgt_label}` {{name: row.target_name}}) "
                f"MERGE (a)-[r:`{rel_type}`]->(b) "
                f"SET r.weight = row.weight, r.description = row.description, "
                f"r.text_unit_ids = row.text_unit_ids",
                rows=batch,
            )
            total += len(batch)
        logger.info(
            "Imported %d (:%s)-[:%s]->(:%s) relationships",
            len(rows), src_label, rel_type, tgt_label,
        )
    return total


def export_to_neo4j(
    G: nx.Graph,
    uri: str = "bolt://localhost:7687",
    user: str = "neo4j",
    password: str = "password",
    database: str = "neo4j",
    rel_type: str = DEFAULT_REL_TYPE,
    batch_size: int = 1000,
    clear: bool = False,
) -> dict[str, int]:
    """
    Import a DKA knowledge graph into Neo4j.

    Args:
        G:          NetworkX graph (from dka.graph_builder.load_graph).
        uri/user/password/database: Neo4j connection settings.
        rel_type:   Relationship type for all edges (default "RELATED").
        batch_size: UNWIND batch size.
        clear:      If True, delete all existing nodes/relationships first.

    Returns:
        {"nodes": N, "relationships": M}
    """
    try:
        from neo4j import GraphDatabase
    except ImportError as e:
        raise ImportError(
            "The 'neo4j' package is required for Neo4j export. "
            "Install it with: pip install dka[neo4j]"
        ) from e

    entities_by_label, rel_groups = graph_to_tables(G, rel_type=rel_type)

    driver = GraphDatabase.driver(uri, auth=(user, password))
    try:
        with driver.session(database=database) as session:
            if clear:
                logger.warning("Clearing existing graph in database '%s' …", database)
                session.run("MATCH (n) DETACH DELETE n")

            _create_constraints(session, list(entities_by_label.keys()))
            n_nodes = _import_entities(session, entities_by_label, batch_size)
            n_rels = _import_relations(session, rel_groups, batch_size)
    finally:
        driver.close()

    logger.info("Neo4j import done: %d nodes, %d relationships", n_nodes, n_rels)
    return {"nodes": n_nodes, "relationships": n_rels}
