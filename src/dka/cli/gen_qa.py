"""
Step 3 – Generate SFT training data from walks and chunks.

Usage:
    # inter-chunk only (from walks)
    dka-gen-qa --walks data/walks.jsonl --chunks data/chunks_all.jsonl --config config.yaml

    # intra-chunk only (from raw chunks)
    dka-gen-qa --chunks data/chunks_all.jsonl --config config.yaml --type intra

    # both types
    dka-gen-qa --walks data/walks.jsonl --chunks data/chunks_all.jsonl --config config.yaml --type both

Checkpoint / resume:
    If the output file already exists, the script counts existing lines and skips
    that many items from the input list, then appends new results.  Restart with
    the exact same command to continue from where it left off.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import random
from pathlib import Path

import yaml
from openai import AsyncOpenAI

from dka.qa_generator import (
    generate_inter_batch,
    generate_intra_batch,
    load_chunk_lookup,
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


def _load_walks(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _load_chunks(path: str, limit: int | None = None) -> list[dict]:
    chunks = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
                if limit and len(chunks) >= limit:
                    break
    return chunks


def _count_lines(path: Path) -> int:
    """Count non-empty lines in an existing JSONL file (= already-done items)."""
    if not path.exists():
        return 0
    count = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def _append_sft(items: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


async def _run_inter(
    walks: list[dict],
    chunk_lookup: dict[str, str],
    client: AsyncOpenAI,
    model: str,
    out_path: Path,
    max_tokens: int,
    temperature: float,
    concurrency: int,
    checkpoint_every: int,
    seed: int,
) -> None:
    # ── Checkpoint: skip already-done walks ──────────────────────────────────
    already_done = _count_lines(out_path)
    if already_done:
        logger.info(
            "Resuming inter-chunk: %d walks already done, skipping to walk %d",
            already_done, already_done,
        )
        walks = walks[already_done:]

    if not walks:
        logger.info("All inter-chunk walks already processed.")
        return

    logger.info("Inter-chunk: %d walks remaining", len(walks))
    rng = random.Random(seed)
    total_written = already_done
    total_processed = 0

    for batch_start in range(0, len(walks), checkpoint_every):
        batch = walks[batch_start: batch_start + checkpoint_every]
        batch_end_abs = already_done + batch_start + len(batch)

        items = await generate_inter_batch(
            walks=batch,
            chunk_lookup=chunk_lookup,
            client=client,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            concurrency=concurrency,
            rng=rng,
        )

        _append_sft(items, out_path)
        total_written += len(items)
        total_processed += len(batch)

        logger.info(
            "Checkpoint: walks processed %d/%d  |  samples written so far: %d  (batch yield: %d/%d)",
            already_done + total_processed, already_done + len(walks),
            total_written, len(items), len(batch),
        )

    print(f"\nInter-chunk SFT → {out_path}  ({total_written} total samples)")


async def _run_intra(
    chunks: list[dict],
    client: AsyncOpenAI,
    model: str,
    out_path: Path,
    max_tokens: int,
    temperature: float,
    concurrency: int,
    checkpoint_every: int,
) -> None:
    # ── Checkpoint: skip already-done chunks ─────────────────────────────────
    already_done = _count_lines(out_path)
    if already_done:
        logger.info(
            "Resuming intra-chunk: %d chunks already done, skipping to chunk %d",
            already_done, already_done,
        )
        chunks = chunks[already_done:]

    if not chunks:
        logger.info("All intra-chunk chunks already processed.")
        return

    logger.info("Intra-chunk: %d chunks remaining", len(chunks))
    total_written = already_done
    total_processed = 0

    for batch_start in range(0, len(chunks), checkpoint_every):
        batch = chunks[batch_start: batch_start + checkpoint_every]

        items = await generate_intra_batch(
            chunks=batch,
            client=client,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            concurrency=concurrency,
        )

        _append_sft(items, out_path)
        total_written += len(items)
        total_processed += len(batch)

        logger.info(
            "Checkpoint: chunks processed %d/%d  |  samples written so far: %d  (batch yield: %d/%d)",
            already_done + total_processed, already_done + len(chunks),
            total_written, len(items), len(batch),
        )

    print(f"Intra-chunk SFT  → {out_path}  ({total_written} total samples)")


async def _run(args: argparse.Namespace, cfg: dict) -> None:
    llm_cfg    = cfg.get("llm", {})
    output_cfg = cfg.get("output", {})

    client = AsyncOpenAI(
        base_url=llm_cfg.get("base_url", "http://localhost:8000/v1"),
        api_key=os.environ.get("OPENAI_API_KEY", llm_cfg.get("api_key", "dummy")),
    )
    model            = llm_cfg.get("model", "")
    concurrency      = llm_cfg.get("concurrent_requests", 8)
    sft_cfg = cfg.get("sft", {})
    checkpoint_every = args.checkpoint_every or sft_cfg.get("checkpoint_every", 500)

    out_dir = Path(args.output_dir or "data")
    mode    = args.type  # inter | intra | both

    # ------------------------------------------------------------------ inter
    if mode in ("inter", "both"):
        if not args.walks:
            raise ValueError("--walks is required for inter-chunk generation")
        walks = _load_walks(args.walks)
        if args.limit:
            walks = walks[: args.limit]
        logger.info("Loaded %d walks", len(walks))

        chunk_lookup = load_chunk_lookup(args.chunks)
        logger.info("Chunk lookup built (%d entries)", len(chunk_lookup))

        inter_max_tokens = args.max_tokens or llm_cfg.get("sft_max_tokens", 2048)
        inter_path = Path(args.inter_output) if args.inter_output else out_dir / output_cfg.get("sft_inter_path", "sft_inter.jsonl")

        await _run_inter(
            walks=walks,
            chunk_lookup=chunk_lookup,
            client=client,
            model=model,
            out_path=inter_path,
            max_tokens=inter_max_tokens,
            temperature=args.temperature,
            concurrency=concurrency,
            checkpoint_every=checkpoint_every,
            seed=args.seed,
        )

    # ------------------------------------------------------------------ intra
    if mode in ("intra", "both"):
        chunks = _load_chunks(args.chunks, limit=args.intra_limit or args.limit)
        logger.info("Loaded %d chunks for intra generation", len(chunks))

        intra_max_tokens = args.max_tokens or llm_cfg.get("sft_max_tokens", 1024)
        intra_path = Path(args.intra_output) if args.intra_output else out_dir / output_cfg.get("sft_intra_path", "sft_intra.jsonl")

        await _run_intra(
            chunks=chunks,
            client=client,
            model=model,
            out_path=intra_path,
            max_tokens=intra_max_tokens,
            temperature=args.temperature,
            concurrency=concurrency,
            checkpoint_every=checkpoint_every,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate SFT data from walks and chunks.")
    parser.add_argument("--walks",            default=None,   help="Path to walks.jsonl (required for inter)")
    parser.add_argument("--chunks",           required=True,  help="Path to chunks JSONL (for text lookup / intra)")
    parser.add_argument("--config",           default="config.yaml")
    parser.add_argument("--type",             default="inter", choices=["inter", "intra", "both"])
    parser.add_argument("--output-dir",       default="data")
    parser.add_argument("--limit",            type=int, default=None, help="Max inter-chunk walks to process")
    parser.add_argument("--intra-limit",      type=int, default=None, help="Max intra-chunk chunks (overrides --limit)")
    parser.add_argument("--temperature",      type=float, default=0.7)
    parser.add_argument("--max-tokens",       type=int, default=None, help="Override max_tokens for generation")
    parser.add_argument("--seed",             type=int, default=42)
    parser.add_argument("--checkpoint-every", type=int, default=None,
                        help="Save after every N walks/chunks (default: 500)")
    parser.add_argument("--inter-output",  default=None, help="Override output path for inter-chunk JSONL")
    parser.add_argument("--intra-output",  default=None, help="Override output path for intra-chunk JSONL")
    args = parser.parse_args()

    cfg = _load_config(args.config)
    asyncio.run(_run(args, cfg))


if __name__ == "__main__":
    main()
