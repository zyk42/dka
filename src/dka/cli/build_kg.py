"""
Step 1 – Build Knowledge Graph from a JSONL corpus.

Input JSONL format (one chunk per line):
    {"id": "chunk_001", "text": "Transformer uses self-attention ..."}

Usage:
    dka-build-kg --input data/chunks.jsonl --output data/knowledge_graph.json
    dka-build-kg --input data/chunks.jsonl --config config.yaml
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path

import yaml
from openai import AsyncOpenAI
from tqdm import tqdm

from dka.extractor import ExtractionConfig, extract_all
from dka.graph_builder import (
    build_graph, covered_chunk_ids, graph_stats, load_graph, merge_graph, save_graph,
)

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


def _load_chunks(path: str) -> list[dict[str, str]]:
    chunks = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    if not chunks:
        raise ValueError(f"No chunks found in {path}")
    logger.info("Loaded %d chunks from %s", len(chunks), path)
    return chunks


async def _run(args: argparse.Namespace, cfg: dict) -> None:
    llm_cfg = cfg.get("llm", {})
    ext_cfg = cfg.get("extraction", {})
    graph_cfg = cfg.get("graph", {})
    output_cfg = cfg.get("output", {})

    api_key = os.environ.get("OPENAI_API_KEY", llm_cfg.get("api_key", "dummy"))
    client = AsyncOpenAI(
        base_url=llm_cfg.get("base_url", "http://localhost:8000/v1"),
        api_key=api_key,
    )
    # Extra endpoints for load balancing (e.g. second GPU on port 8001)
    extra_urls = args.extra_urls or llm_cfg.get("extra_base_urls", [])
    extra_clients = [
        AsyncOpenAI(base_url=url, api_key=api_key)
        for url in extra_urls
    ] if extra_urls else None
    if extra_clients:
        logger.info("Load balancing across %d endpoints", 1 + len(extra_clients))

    extraction_config = ExtractionConfig(
        entity_types=ext_cfg.get("entity_types", [
            "CONCEPT", "MODEL", "METHOD", "DATASET", "METRIC",
            "PAPER", "ORGANIZATION", "PERSON",
        ]),
        max_gleanings=ext_cfg.get("max_gleanings", 1),
    )

    chunks = _load_chunks(args.input)

    # support --limit for quick tests
    if args.limit:
        chunks = chunks[: args.limit]
        logger.info("Limited to first %d chunks (--limit)", len(chunks))

    # ── Incremental merge: skip already-processed chunks ────────────────────
    G_base = None
    out_path = args.output or output_cfg.get("graph_path", "data/knowledge_graph.json")
    if args.merge and Path(out_path).exists():
        logger.info("Loading existing graph from %s for incremental merge …", out_path)
        G_base = load_graph(out_path)
        already_done = covered_chunk_ids(G_base)
        before = len(chunks)
        chunks = [c for c in chunks if c["id"] not in already_done]
        logger.info(
            "Incremental mode: %d / %d chunks already processed, %d remaining",
            before - len(chunks), before, len(chunks),
        )
        if not chunks:
            logger.info("All chunks already in graph — nothing to do.")
            stats = graph_stats(G_base)
            print(f"\nGraph up-to-date → {out_path}")
            print(f"  nodes : {stats['nodes']}")
            print(f"  edges : {stats['edges']}")
            print(f"  avg degree: {stats['avg_degree']:.2f}")
            return

    batch_size = args.checkpoint_every or llm_cfg.get("checkpoint_every", 1000)
    model      = llm_cfg.get("model", "Qwen/Qwen2.5-7B-Instruct")
    max_tokens = llm_cfg.get("max_tokens", 4096)
    temperature= llm_cfg.get("temperature", 0.0)
    concurrency= llm_cfg.get("concurrent_requests", 8)
    min_edge_w = graph_cfg.get("min_edge_weight", 1.0)

    G = G_base  # may be None for a fresh run

    total = len(chunks)
    for batch_start in range(0, total, batch_size):
        batch = chunks[batch_start : batch_start + batch_size]
        batch_end = min(batch_start + batch_size, total)
        logger.info(
            "Batch %d–%d / %d …", batch_start + 1, batch_end, total
        )

        entities, relationships = await extract_all(
            chunks=batch,
            client=client,
            model=model,
            config=extraction_config,
            max_tokens=max_tokens,
            temperature=temperature,
            concurrency=concurrency,
            extra_clients=extra_clients,
        )
        logger.info(
            "  Extracted: %d entities, %d relationships",
            len(entities), len(relationships),
        )

        G_new = build_graph(entities, relationships, min_edge_weight=min_edge_w)

        if G is None:
            G = G_new
        else:
            G = merge_graph(G, G_new)

        # ── Checkpoint save after every batch ──────────────────────────────
        save_graph(G, out_path)
        stats = graph_stats(G)
        logger.info(
            "Checkpoint saved → %s  (nodes=%d, edges=%d, progress=%d/%d)",
            out_path, stats["nodes"], stats["edges"], batch_end, total,
        )

    stats = graph_stats(G)
    print(f"\nGraph saved → {out_path}")
    print(f"  nodes : {stats['nodes']}")
    print(f"  edges : {stats['edges']}")
    print(f"  avg degree: {stats['avg_degree']:.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a knowledge graph from text chunks (GraphRAG-style extraction)."
    )
    parser.add_argument("--input",  required=True, help="Path to input JSONL chunks file")
    parser.add_argument("--output", default=None,  help="Output graph JSON path")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--limit",             type=int, default=None,  help="Process only first N chunks (for testing)")
    parser.add_argument("--merge",             action="store_true",     help="Incremental: load existing graph and skip already-processed chunks")
    parser.add_argument("--checkpoint-every",  type=int, default=None,  help="Save graph every N chunks (default: 1000)")
    parser.add_argument("--extra-urls",        nargs="*", default=None, help="Extra vLLM base URLs for load balancing")
    args = parser.parse_args()

    cfg = _load_config(args.config)
    asyncio.run(_run(args, cfg))


if __name__ == "__main__":
    main()
