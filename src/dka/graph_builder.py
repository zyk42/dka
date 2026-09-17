"""
Build a NetworkX Graph from extracted entities and relationships.
Each node carries chunk provenance (text_unit_ids) required for the
Knowledge Annealing weighted random walk.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import networkx as nx
import pandas as pd

logger = logging.getLogger(__name__)


def build_graph(
    entities: pd.DataFrame,
    relationships: pd.DataFrame,
    min_edge_weight: float = 1.0,
) -> nx.Graph:
    """
    Construct a directed graph from extracted DataFrames.

    Node attributes:
        type          - entity type (CONCEPT, MODEL, …)
        description   - list of descriptions collected across chunks
        text_unit_ids - list of chunk_ids where this entity appears  ← c(v)
        frequency     - number of chunks mentioning this entity

    Edge attributes:
        weight        - cumulative relationship strength
        description   - list of relationship descriptions
        text_unit_ids - chunk_ids where this relationship was observed
    """
    G = nx.Graph()

    for _, row in entities.iterrows():
        G.add_node(
            row["title"],
            type=row.get("type", "UNKNOWN"),
            description=row.get("description", []),
            text_unit_ids=row.get("text_unit_ids", []),
            frequency=int(row.get("frequency", 1)),
        )

    for _, row in relationships.iterrows():
        src, tgt = row["source"], row["target"]
        if not G.has_node(src) or not G.has_node(tgt):
            continue
        weight = float(row.get("weight", 1.0))
        if weight < min_edge_weight:
            continue
        G.add_edge(
            src,
            tgt,
            weight=weight,
            description=row.get("description", []),
            text_unit_ids=row.get("text_unit_ids", []),
        )

    logger.info(
        "Graph built: %d nodes, %d edges", G.number_of_nodes(), G.number_of_edges()
    )
    return G


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def save_graph(G: nx.Graph, path: str | Path) -> None:
    """Save graph to JSON (node-link format, human-readable)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = nx.node_link_data(G, edges="links")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info("Graph saved to %s", path)


def load_graph(path: str | Path) -> nx.Graph:
    """Load graph from JSON (node-link format)."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    G = nx.node_link_graph(data, directed=False, edges="links")
    logger.info(
        "Graph loaded: %d nodes, %d edges", G.number_of_nodes(), G.number_of_edges()
    )
    return G


def merge_graph(G_base: nx.Graph, G_new: nx.Graph) -> nx.Graph:
    """
    Merge G_new into G_base, combining node/edge attributes in-place.

    - Nodes present in both: extend descriptions and text_unit_ids, recount frequency.
    - Nodes only in G_new: add as-is.
    - Edges present in both: sum weights, extend descriptions and text_unit_ids.
    - Edges only in G_new: add as-is.

    Returns G_base (modified in place).
    """
    for node, attrs in G_new.nodes(data=True):
        if G_base.has_node(node):
            existing = G_base.nodes[node]
            # Merge descriptions (extend, preserve order, deduplicate)
            seen = set(existing.get("description", []))
            for d in attrs.get("description", []):
                if d not in seen:
                    existing.setdefault("description", []).append(d)
                    seen.add(d)
            # Merge text_unit_ids (extend, deduplicate)
            existing_ids = set(existing.get("text_unit_ids", []))
            for cid in attrs.get("text_unit_ids", []):
                if cid not in existing_ids:
                    existing.setdefault("text_unit_ids", []).append(cid)
                    existing_ids.add(cid)
            existing["frequency"] = len(existing.get("text_unit_ids", []))
        else:
            G_base.add_node(node, **attrs)

    for src, tgt, attrs in G_new.edges(data=True):
        if G_base.has_edge(src, tgt):
            existing = G_base[src][tgt]
            existing["weight"] = existing.get("weight", 0.0) + attrs.get("weight", 1.0)
            seen = set(existing.get("description", []))
            for d in attrs.get("description", []):
                if d not in seen:
                    existing.setdefault("description", []).append(d)
                    seen.add(d)
            existing_ids = set(existing.get("text_unit_ids", []))
            for cid in attrs.get("text_unit_ids", []):
                if cid not in existing_ids:
                    existing.setdefault("text_unit_ids", []).append(cid)
                    existing_ids.add(cid)
        else:
            if G_base.has_node(src) and G_base.has_node(tgt):
                G_base.add_edge(src, tgt, **attrs)

    logger.info(
        "After merge: %d nodes, %d edges",
        G_base.number_of_nodes(), G_base.number_of_edges(),
    )
    return G_base


def covered_chunk_ids(G: nx.Graph) -> set[str]:
    """Return the set of chunk IDs already represented in the graph."""
    ids: set[str] = set()
    for _, attrs in G.nodes(data=True):
        ids.update(attrs.get("text_unit_ids", []))
    return ids


def graph_stats(G: nx.Graph) -> dict[str, Any]:
    degrees = dict(G.degree())
    return {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "avg_degree": sum(degrees.values()) / max(len(degrees), 1),
        "max_degree_node": max(degrees, key=degrees.get) if degrees else None,
    }
