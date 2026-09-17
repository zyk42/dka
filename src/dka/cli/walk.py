"""
Step 2 – Generate cross-chunk paths via weighted random walk.

Reads the knowledge graph produced by `ka-build-kg` and performs
weighted random walks, saving each path as a JSONL record for downstream
SFT data generation (inter-chunk QA synthesis).

Usage:
    dka-walk --graph data/knowledge_graph.json --output data/walks.jsonl
    dka-walk --graph data/knowledge_graph.json --config config.yaml --num-walks 5000
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import yaml
from tqdm import tqdm

from dka.graph_builder import load_graph
from dka.walker import WalkConfig, generate_walks, save_walks

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)
logger = logging.getLogger(__name__)


def _load_config(path: str | None) -> dict:
    if path and Path(path).exists():
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate weighted random walk paths over the knowledge graph."
    )
    parser.add_argument("--graph",  required=True, help="Path to knowledge_graph.json")
    parser.add_argument("--output", default=None,  help="Output walks JSONL path")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--num-walks", type=int, default=None, help="Override num_walks in config")
    parser.add_argument("--seed",      type=int, default=42,   help="Random seed")
    args = parser.parse_args()

    cfg = _load_config(args.config)
    walk_cfg_raw = cfg.get("walk", {})
    output_cfg = cfg.get("output", {})

    walk_cfg = WalkConfig(
        num_walks=args.num_walks or walk_cfg_raw.get("num_walks", 10_000),
        min_hops=walk_cfg_raw.get("min_hops", 3),
        max_hops=walk_cfg_raw.get("max_hops", 5),
        min_chunks_crossed=walk_cfg_raw.get("min_chunks_crossed", 2),
        cross_chunk_alpha=cfg.get("graph", {}).get("cross_chunk_multiplier", 3.0),
        max_retries=walk_cfg_raw.get("max_retries", 10),
        seed=args.seed,
        rare_start=walk_cfg_raw.get("rare_start", True),
        rare_frequency_threshold=walk_cfg_raw.get("rare_frequency_threshold", 2),
    )

    G = load_graph(args.graph)
    logger.info(
        "Generating %d walks (hops %d–%d, ≥%d chunks) …",
        walk_cfg.num_walks, walk_cfg.min_hops, walk_cfg.max_hops,
        walk_cfg.min_chunks_crossed,
    )

    walks = []
    with tqdm(total=walk_cfg.num_walks, unit="walk") as pbar:
        for w in generate_walks(G, walk_cfg):
            walks.append(w)
            pbar.update(1)

    out_path = args.output or output_cfg.get("walks_path", "data/walks.jsonl")
    save_walks(walks, out_path)

    # summary stats
    if walks:
        avg_hops = sum(len(w.nodes) - 1 for w in walks) / len(walks)
        avg_chunks = sum(w.chunks_crossed for w in walks) / len(walks)
        print(f"\nWalks saved → {out_path}")
        print(f"  total walks    : {len(walks)}")
        print(f"  avg hops       : {avg_hops:.2f}")
        print(f"  avg chunks crossed: {avg_chunks:.2f}")
    else:
        print("No walks generated – check graph connectivity.")


if __name__ == "__main__":
    main()
