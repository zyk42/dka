"""
Weighted random walk on the knowledge graph following the Knowledge Annealing formula:

    P(v_{t+1} = v' | v_t = v) ∝ w(v, v') · α^[c(v') ≠ c(v)] · rel_score(v, v')

where:
  w(v, v')           – edge weight (cumulative relationship strength from extraction)
  α                  – cross-chunk multiplier (default 3.0; from config cross_chunk_multiplier)
  [c(v') ≠ c(v)]     – 1 if neighbour comes from a different chunk, 0 otherwise
  rel_score(v, v')   – proxy: same as w(v, v') (edge weight already captures relevance)

Walk strategy:
  - Start nodes are sampled from *rare* nodes (low frequency, appearing in few chunks).
  - No node revisiting within a single walk path.
  - Isolated nodes (out-degree = 0) are skipped as start candidates.
  - Paths span 3–5 hops across ≥2 distinct chunks.
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import networkx as nx

logger = logging.getLogger(__name__)


@dataclass
class WalkConfig:
    num_walks: int = 10_000
    min_hops: int = 3
    max_hops: int = 5
    min_chunks_crossed: int = 2
    cross_chunk_alpha: float = 3.0
    max_retries: int = 10          # retries if a walk dead-ends or fails quality check
    seed: int | None = None
    # Rare-node start: sample start nodes from low-frequency nodes
    rare_start: bool = True
    rare_frequency_threshold: int = 2   # nodes in ≤ this many chunks are "rare"


@dataclass
class Walk:
    nodes: list[str]             # ordered entity names
    node_descriptions: list[str] # description for each node (first description, or "")
    edges: list[dict]            # edge attrs for each hop
    chunk_ids: list[str]         # representative chunk_id per node (first occurrence)
    chunks_crossed: int          # number of distinct chunks in the path


def _primary_chunk(G: nx.DiGraph, node: str) -> str | None:
    ids = G.nodes[node].get("text_unit_ids", [])
    return ids[0] if ids else None


def _primary_description(G: nx.DiGraph, node: str) -> str:
    desc = G.nodes[node].get("description", [])
    if isinstance(desc, list):
        return desc[0] if desc else ""
    return str(desc) if desc else ""


def _node_frequency(G: nx.DiGraph, node: str) -> int:
    freq = G.nodes[node].get("frequency", 1)
    return int(freq) if isinstance(freq, (int, float)) else 1


def _sample_next(
    G: nx.DiGraph,
    current: str,
    current_chunk: str | None,
    alpha: float,
    rng: random.Random,
    visited: set[str],
) -> str | None:
    """Sample next node proportional to the walk formula, excluding visited nodes."""
    candidates = [n for n in G.neighbors(current) if n not in visited]
    if not candidates:
        return None

    weights: list[float] = []
    for nb in candidates:
        edge = G.edges[current, nb]
        w = float(edge.get("weight", 1.0))
        nb_chunk = _primary_chunk(G, nb)
        cross = alpha if (nb_chunk and nb_chunk != current_chunk) else 1.0
        weights.append(w * cross)

    total = sum(weights)
    if total <= 0:
        return None

    r = rng.uniform(0, total)
    cumulative = 0.0
    for nb, wt in zip(candidates, weights):
        cumulative += wt
        if r <= cumulative:
            return nb
    return candidates[-1]


def random_walk(
    G: nx.DiGraph,
    start: str,
    cfg: WalkConfig,
    rng: random.Random,
) -> Walk | None:
    """Attempt a single random walk from *start*. No node revisiting. Returns None if quality check fails."""
    path = [start]
    chunk_path = [_primary_chunk(G, start)]
    desc_path = [_primary_description(G, start)]
    edge_attrs: list[dict] = []
    visited: set[str] = {start}

    hops = rng.randint(cfg.min_hops, cfg.max_hops)

    for _ in range(hops):
        current = path[-1]
        current_chunk = chunk_path[-1]
        nxt = _sample_next(G, current, current_chunk, cfg.cross_chunk_alpha, rng, visited)
        if nxt is None:
            break
        visited.add(nxt)
        path.append(nxt)
        chunk_path.append(_primary_chunk(G, nxt))
        desc_path.append(_primary_description(G, nxt))
        edge_attrs.append(dict(G.edges[current, nxt]))

    if len(path) - 1 < cfg.min_hops:
        return None

    distinct_chunks = len({c for c in chunk_path if c is not None})
    if distinct_chunks < cfg.min_chunks_crossed:
        return None

    return Walk(
        nodes=path,
        node_descriptions=desc_path,
        edges=edge_attrs,
        chunk_ids=[c or "" for c in chunk_path],
        chunks_crossed=distinct_chunks,
    )


def _build_start_weights(G: nx.DiGraph, nodes: list[str], rare_start: bool, rare_threshold: int) -> list[float]:
    """
    Compute sampling weights for start nodes.
    rare_start=True  → rarity = 1 / (frequency × degree) weighting.
                       Only nodes with frequency ≤ rare_threshold are eligible.
                       This penalises high-degree "hub" nodes that appear rarely
                       in text but connect to many neighbours, which would otherwise
                       produce redundant walks radiating from the same hub.
    rare_start=False → degree-biased (high-degree nodes preferred).
    Nodes with degree = 0 get weight 0 (isolated, can't start a walk).
    """
    weights = []
    for n in nodes:
        deg = G.degree(n)
        if deg == 0:
            weights.append(0.0)
            continue
        if rare_start:
            freq = _node_frequency(G, n)
            # Eligible only if below frequency threshold; weight = 1 / (freq * deg)
            if freq <= rare_threshold:
                weights.append(1.0 / (freq * deg))
            else:
                weights.append(0.0)
        else:
            weights.append(float(deg))
    return weights


def generate_walks(
    G: nx.DiGraph,
    cfg: WalkConfig | None = None,
) -> Iterator[Walk]:
    """
    Yield up to cfg.num_walks walks.
    Start nodes are sampled from rare nodes (inverse-frequency) by default.
    Isolated nodes (out-degree=0) are skipped.
    """
    cfg = cfg or WalkConfig()
    rng = random.Random(cfg.seed)

    nodes = list(G.nodes())
    if not nodes:
        return

    start_weights = _build_start_weights(G, nodes, cfg.rare_start, cfg.rare_frequency_threshold)

    if sum(start_weights) == 0:
        logger.warning(
            "No eligible rare start nodes found (threshold=%d). Falling back to degree-biased.",
            cfg.rare_frequency_threshold,
        )
        start_weights = _build_start_weights(G, nodes, rare_start=False, rare_threshold=0)

    eligible = sum(1 for w in start_weights if w > 0)
    logger.info(
        "Walk start pool: %d eligible nodes (rare_start=%s, threshold=%d)",
        eligible, cfg.rare_start, cfg.rare_frequency_threshold,
    )

    generated = 0
    while generated < cfg.num_walks:
        start = rng.choices(nodes, weights=start_weights, k=1)[0]
        walk = None
        for _ in range(cfg.max_retries):
            walk = random_walk(G, start, cfg, rng)
            if walk is not None:
                break
        if walk is not None:
            generated += 1
            yield walk


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------

def save_walks(walks: list[Walk], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for w in walks:
            f.write(json.dumps({
                "nodes": w.nodes,
                "node_descriptions": w.node_descriptions,
                "chunk_ids": w.chunk_ids,
                "chunks_crossed": w.chunks_crossed,
                "edges": w.edges,
            }, ensure_ascii=False) + "\n")
    logger.info("Saved %d walks to %s", len(walks), path)


def load_walks(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
