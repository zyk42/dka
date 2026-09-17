"""
SFT data generation from knowledge graph walks.

Two types of data:
- inter-chunk: multi-hop QA requiring information from ≥2 chunks (uses walks)
- intra-chunk: single-chunk QA for Stage-1 fine-tuning (uses raw chunks)
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import re
from pathlib import Path

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

INTER_VALIDATE_PROMPT = """\
You are reviewing a generated multi-hop QA sample. The intended answer is: **{target_node}**

=== QUESTION ===
{question}

=== REASONING ===
{reasoning}

=== SOURCE PASSAGES ===
{chunks_text}

Output FAIL if ANY of the following is true:
1. Any reasoning step contains a fact not found in the passages (hallucination).
2. The question names or directly quotes "{target_node}".
3. The clues in the question are insufficient to uniquely identify {target_node} — another entity could reasonably fit.
4. The reasoning is circular: it merely restates the question or answer rather than building toward it step by step.
5. The question reads like a compressed version of the reasoning chain — it strings together clues from every reasoning step rather than picking the key identifying ones.
6. The reasoning chain connects entities only through coincidental shared attributes (same number, same year with no causal or thematic link) and has no real logical thread.

Output PASS otherwise.

Output only PASS or FAIL, nothing else.
"""

INTER_CHUNK_PROMPT = """\
You are constructing a multi-hop QA training sample. The answer is: **{target_node}**.

=== REASONING PATH ===
{path_text}

=== SOURCE PASSAGES ===
{chunks_text}

Step 1 — Write the reasoning chain:
Build step-by-step reasoning from facts in the passages that leads to {target_node}.
Each step must be grounded in the passages. Year-based connections are valid (e.g. "In the same year X happened, Y also occurred").

Step 2 — Derive the question:
From the reasoning chain, formulate ONE concise question whose answer is {target_node}.
Use only the MINIMUM clues needed to uniquely identify {target_node} — do not copy every detail from the reasoning.
Do NOT mention "{target_node}" in the question.

The question must be a single, natural sentence. Do NOT string together clues from every reasoning step — pick the ONE or TWO that most directly point to {target_node}.

Output format:
Reasoning:
Step 1: <inference step>
Step 2: <inference step>
[more steps as needed]
Question: <question>
Final Answer: {target_node}

Rules:
- Plain factual prose in reasoning steps, no passage labels.
- The question should read like a natural exam question, not a compressed summary of the reasoning chain.
- If you cannot describe {target_node} without naming it, or it is too generic to be uniquely identified, write: Question: SKIP
"""

INTRA_CHUNK_PROMPT = """\
You are constructing training data for a question answering model.

Below is a short document passage.

=== DOCUMENT ===
{chunk_text}

Step 1 — Choose the most specific, identifiable entity or fact in the passage as your answer.
Write a comprehensive description of it, covering as much relevant information from the passage as possible (its role, relationships, attributes, context, dates, etc.).
Write as plain factual statements — do NOT use phrases like "The document states that", "According to the passage", or any meta-reference to the text. State facts directly.

Step 2 — From the description, derive ONE question whose answer is that entity.
Use only the MINIMUM clues needed to uniquely identify it — do NOT name the answer in the question.
The question must be a single, natural sentence answerable from this passage alone.
If the answer is too generic to be uniquely identified without naming it, write: Question: SKIP

Output format (strictly follow this):
Description: <comprehensive factual description of the answer>
Question: <question text>
Final Answer: <answer entity or phrase>
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_path_text(walk: dict) -> str:
    """Format walk as a numbered reasoning chain, stripped of all chunk IDs and metadata."""
    nodes = walk["nodes"]
    descs = walk.get("node_descriptions", [""] * len(nodes))
    edges = walk.get("edges", [])
    lines = []
    for i, node in enumerate(nodes):
        desc = descs[i] if i < len(descs) else ""
        # Entity line: number + name + description
        entity_line = f"{i + 1}. {node}"
        if desc:
            entity_line += f"\n   ({desc})"
        lines.append(entity_line)
        # Relation line: unique description only, no IDs/weights
        if i < len(edges):
            rel_descs = edges[i].get("description", [])
            if isinstance(rel_descs, list):
                # Deduplicate while preserving order
                seen = set()
                unique = [r for r in rel_descs if r and not (r in seen or seen.add(r))]
                rel = unique[0] if unique else "related to"
            else:
                rel = rel_descs or "related to"
            lines.append(f"   → {rel}")
    return "\n".join(lines)


def _build_chunks_text(chunk_ids: list[str], chunk_lookup: dict[str, str]) -> str:
    seen = []
    for cid in chunk_ids:
        if cid and cid not in seen:
            seen.append(cid)
    parts = []
    for i, cid in enumerate(seen, 1):
        text = chunk_lookup.get(cid, "")
        if text:
            parts.append(f"[Passage {i}]\n{text}")
    return "\n\n".join(parts)


def _parse_output(text: str) -> dict | None:
    """Parse LLM output into {question, reasoning, answer}. Returns None on SKIP or parse failure.
    Supports:
      - inter-chunk format: Reasoning → Question → Final Answer
      - intra-chunk format: Description → Question → Final Answer
    """
    if text.strip().upper().startswith("SKIP"):
        return None
    q = re.search(r"Question:\s*(.+?)(?=\nFinal Answer:|\Z)", text, re.S)
    r = re.search(r"(?:Reasoning|Description):\s*(.+?)(?=\nQuestion:|\nFinal Answer:|\Z)", text, re.S)
    a = re.search(r"Final Answer:\s*(.+)", text)
    if not (q and a):
        return None
    answer = a.group(1).strip()
    if not answer:
        return None
    return {
        "question": q.group(1).strip(),
        "reasoning": r.group(1).strip() if r else "",
        "answer": answer,
    }


def _to_chat(question: str, reasoning: str, answer: str) -> dict:
    """Convert to sharegpt chat format compatible with LLaMA-Factory / TRL."""
    assistant_content = reasoning.rstrip()
    if assistant_content:
        assistant_content += f"\n\nFinal Answer: {answer}"
    else:
        assistant_content = f"Final Answer: {answer}"
    return {
        "conversations": [
            {"from": "human", "value": question},
            {"from": "gpt",   "value": assistant_content},
        ]
    }


# ---------------------------------------------------------------------------
# Inter-chunk generation
# ---------------------------------------------------------------------------

async def generate_inter(
    walk: dict,
    chunk_lookup: dict[str, str],
    client: AsyncOpenAI,
    model: str,
    max_tokens: int = 4096,
    temperature: float = 0.0,
    rng: random.Random | None = None,
) -> dict | None:
    """Generate one inter-chunk QA pair from a walk using two LLM calls:
    Call 1: generate the full reasoning chain + question.
    Call 2: validate the generated QA (PASS / FAIL).
    """
    nodes = walk["nodes"]
    if len(nodes) < 2:
        return None

    target_node = nodes[0]
    path_text   = _build_path_text(walk)
    chunks_text = _build_chunks_text(walk["chunk_ids"], chunk_lookup)
    if not chunks_text:
        return None

    # ── Call 1: QA generation ───────────────────────────────────────────────
    gen_prompt = INTER_CHUNK_PROMPT.format(
        path_text=path_text,
        chunks_text=chunks_text,
        target_node=target_node,
    )
    try:
        gen_resp = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": gen_prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        text = gen_resp.choices[0].message.content or ""
    except Exception:
        logger.exception("inter-chunk generation failed")
        return None

    parsed = _parse_output(text)
    if not parsed:
        logger.warning("Failed to parse inter-chunk output:\n%s", text[:300])
        return None

    # Fast code-level checks before spending a validation call
    if target_node.lower() in parsed["question"].lower():
        logger.debug("Entity blurring failed, discarding sample")
        return None

    reasoning = parsed.get("reasoning", "")
    if not reasoning.strip() or len(reasoning.strip()) < 30:
        logger.debug("Reasoning too short, discarding sample")
        return None

    # ── Call 2: post-generation validation ─────────────────────────────────
    val_prompt = INTER_VALIDATE_PROMPT.format(
        target_node=target_node,
        question=parsed["question"],
        reasoning=reasoning,
        chunks_text=chunks_text,
    )
    try:
        val_resp = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": val_prompt}],
            max_tokens=5,
            temperature=0.0,
        )
        verdict = (val_resp.choices[0].message.content or "").strip().upper()
    except Exception:
        logger.exception("validation call failed")
        return None

    if not verdict.startswith("PASS"):
        logger.info("Validator rejected sample: %s", " → ".join(nodes[:3]))
        return None

    item = _to_chat(parsed["question"], parsed["reasoning"], parsed["answer"])
    item["meta"] = {
        "type": "inter",
        "walk_nodes": nodes,
        "chunk_ids": list(dict.fromkeys(walk["chunk_ids"])),
        "chunks_crossed": walk["chunks_crossed"],
        "target_node": target_node,
    }
    return item


# ---------------------------------------------------------------------------
# Intra-chunk generation
# ---------------------------------------------------------------------------

async def generate_intra(
    chunk: dict,
    client: AsyncOpenAI,
    model: str,
    max_tokens: int = 1024,
    temperature: float = 0.7,
) -> dict | None:
    """Generate one intra-chunk QA pair from a single chunk."""
    prompt = INTRA_CHUNK_PROMPT.format(chunk_text=chunk["text"])

    try:
        resp = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        text = resp.choices[0].message.content or ""
    except Exception:
        logger.exception("intra-chunk generation failed for chunk %s", chunk["id"])
        return None

    parsed = _parse_output(text)
    if not parsed:
        logger.warning("Failed to parse intra-chunk output:\n%s", text[:300])
        return None

    item = _to_chat(parsed["question"], parsed["reasoning"], parsed["answer"])
    item["meta"] = {
        "type": "intra",
        "chunk_id": chunk["id"],
        "title": chunk.get("title", ""),
    }
    return item


# ---------------------------------------------------------------------------
# Batch runners
# ---------------------------------------------------------------------------

async def generate_inter_all(
    walks: list[dict],
    chunk_lookup: dict[str, str],
    client: AsyncOpenAI,
    model: str,
    max_tokens: int = 4096,
    temperature: float = 0.7,
    concurrency: int = 8,
    seed: int | None = None,
) -> list[dict]:
    rng = random.Random(seed)
    semaphore = asyncio.Semaphore(concurrency)

    async def _run(walk: dict) -> dict | None:
        async with semaphore:
            return await generate_inter(walk, chunk_lookup, client, model, max_tokens, temperature, rng)

    results = await asyncio.gather(*[_run(w) for w in walks])
    out = [r for r in results if r is not None]
    logger.info("Inter-chunk: %d / %d generated", len(out), len(walks))
    return out


async def generate_inter_batch(
    walks: list[dict],
    chunk_lookup: dict[str, str],
    client: AsyncOpenAI,
    model: str,
    max_tokens: int = 4096,
    temperature: float = 0.7,
    concurrency: int = 8,
    rng: random.Random | None = None,
) -> list[dict]:
    """Process a batch of walks concurrently; returns only successful items."""
    semaphore = asyncio.Semaphore(concurrency)

    async def _run(walk: dict) -> dict | None:
        async with semaphore:
            return await generate_inter(walk, chunk_lookup, client, model, max_tokens, temperature, rng)

    results = await asyncio.gather(*[_run(w) for w in walks])
    return [r for r in results if r is not None]


async def generate_intra_all(
    chunks: list[dict],
    client: AsyncOpenAI,
    model: str,
    max_tokens: int = 1024,
    temperature: float = 0.7,
    concurrency: int = 8,
) -> list[dict]:
    semaphore = asyncio.Semaphore(concurrency)

    async def _run(chunk: dict) -> dict | None:
        async with semaphore:
            return await generate_intra(chunk, client, model, max_tokens, temperature)

    results = await asyncio.gather(*[_run(c) for c in chunks])
    out = [r for r in results if r is not None]
    logger.info("Intra-chunk: %d / %d generated", len(out), len(chunks))
    return out


async def generate_intra_batch(
    chunks: list[dict],
    client: AsyncOpenAI,
    model: str,
    max_tokens: int = 1024,
    temperature: float = 0.7,
    concurrency: int = 8,
) -> list[dict]:
    """Process a batch of chunks concurrently; returns only successful items."""
    semaphore = asyncio.Semaphore(concurrency)

    async def _run(chunk: dict) -> dict | None:
        async with semaphore:
            return await generate_intra(chunk, client, model, max_tokens, temperature)

    results = await asyncio.gather(*[_run(c) for c in chunks])
    return [r for r in results if r is not None]


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_chunk_lookup(chunks_path: str | Path) -> dict[str, str]:
    """Build chunk_id → text lookup from a JSONL file."""
    lookup = {}
    with open(chunks_path, encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            lookup[item["id"]] = item["text"]
    return lookup


def save_sft(items: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    logger.info("Saved %d SFT samples → %s", len(items), path)
