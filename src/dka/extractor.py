# Adapted from Microsoft GraphRAG graph_extractor.py (MIT License)
# Changes: uses openai.AsyncOpenAI instead of graphrag-llm; no dependency on graphrag packages.

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from openai import AsyncOpenAI

from dka.prompts import CONTINUE_PROMPT, GRAPH_EXTRACTION_PROMPT, LOOP_PROMPT

logger = logging.getLogger(__name__)

TUPLE_DELIMITER = "<|>"
RECORD_DELIMITER = "##"
COMPLETION_DELIMITER = "<|COMPLETE|>"


def _clean(s: str) -> str:
    return s.strip().strip('"').strip("'").strip()


@dataclass
class ExtractionConfig:
    entity_types: list[str] = field(default_factory=lambda: [
        "CONCEPT", "MODEL", "METHOD", "DATASET", "METRIC", "PAPER",
        "ORGANIZATION", "PERSON",
    ])
    max_gleanings: int = 1
    prompt: str = GRAPH_EXTRACTION_PROMPT


class GraphExtractor:
    """
    Port of GraphRAG's GraphExtractor using openai.AsyncOpenAI.
    Extracts entities and relationships from a single text chunk,
    returning DataFrames with source_id (chunk_id) attached.
    """

    def __init__(
        self,
        client: AsyncOpenAI,
        model: str,
        config: ExtractionConfig | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ):
        self._client = client
        self._model = model
        self._cfg = config or ExtractionConfig()
        self._max_tokens = max_tokens
        self._temperature = temperature

    async def extract(
        self, text: str, chunk_id: str
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Extract entities + relationships from *text*. Returns (entities_df, rels_df)."""
        try:
            raw = await self._call_llm(text)
        except Exception:
            logger.exception("Extraction failed for chunk %s", chunk_id)
            return _empty_entities(), _empty_relationships()

        return self._parse(raw, chunk_id)

    async def _call_llm(self, text: str) -> str:
        messages: list[dict[str, str]] = [
            {
                "role": "user",
                "content": self._cfg.prompt.format(
                    entity_types=",".join(self._cfg.entity_types),
                    input_text=text,
                ),
            }
        ]

        response = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
        )
        result = response.choices[0].message.content or ""
        messages.append({"role": "assistant", "content": result})

        for i in range(self._cfg.max_gleanings):
            messages.append({"role": "user", "content": CONTINUE_PROMPT})
            cont = await self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
            )
            cont_text = cont.choices[0].message.content or ""
            result += cont_text
            messages.append({"role": "assistant", "content": cont_text})

            if i >= self._cfg.max_gleanings - 1:
                break

            messages.append({"role": "user", "content": LOOP_PROMPT})
            loop_resp = await self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                max_tokens=4,
                temperature=0.0,
            )
            if (loop_resp.choices[0].message.content or "").strip().upper() != "Y":
                break

        return result

    def _parse(
        self, result: str, chunk_id: str
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        entities: list[dict[str, Any]] = []
        relationships: list[dict[str, Any]] = []

        for raw_record in result.split(RECORD_DELIMITER):
            record = re.sub(r"^\(|\)$", "", raw_record.strip())
            if not record or COMPLETION_DELIMITER in record:
                continue

            parts = record.split(TUPLE_DELIMITER)
            rec_type = _clean(parts[0])

            if rec_type == "entity" and len(parts) >= 4:
                entities.append({
                    "title": _clean(parts[1]),
                    "type": _clean(parts[2]).upper(),
                    "description": _clean(parts[3]),
                    "source_id": chunk_id,
                })

            elif rec_type == "relationship" and len(parts) >= 5:
                try:
                    weight = float(_clean(parts[-1]))
                except ValueError:
                    weight = 1.0
                relationships.append({
                    "source": _clean(parts[1]),
                    "target": _clean(parts[2]),
                    "description": _clean(parts[3]),
                    "weight": weight,
                    "source_id": chunk_id,
                })

        return (
            pd.DataFrame(entities) if entities else _empty_entities(),
            pd.DataFrame(relationships) if relationships else _empty_relationships(),
        )


async def extract_all(
    chunks: list[dict[str, str]],  # [{"id": ..., "text": ...}, ...]
    client: AsyncOpenAI,
    model: str,
    config: ExtractionConfig | None = None,
    max_tokens: int = 4096,
    temperature: float = 0.0,
    concurrency: int = 8,
    extra_clients: list[AsyncOpenAI] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extract entities and relationships from all chunks concurrently.
    If extra_clients is provided, distributes chunks across all clients
    in round-robin for load balancing across multiple vLLM instances.
    Returns merged (entities_df, relationships_df) with text_unit_ids columns.
    """
    all_clients = [client] + (extra_clients or [])
    extractors = [
        GraphExtractor(c, model, config, max_tokens, temperature)
        for c in all_clients
    ]
    semaphore = asyncio.Semaphore(concurrency)

    async def _run(i: int, chunk: dict[str, str]) -> tuple[pd.DataFrame, pd.DataFrame]:
        extractor = extractors[i % len(extractors)]
        async with semaphore:
            return await extractor.extract(chunk["text"], chunk["id"])

    results = await asyncio.gather(*[_run(i, c) for i, c in enumerate(chunks)])

    entity_dfs = [r[0] for r in results]
    rel_dfs = [r[1] for r in results]

    entities = _merge_entities(entity_dfs)
    relationships = _merge_relationships(rel_dfs)
    relationships = _filter_orphans(relationships, entities)

    return entities, relationships


# ---------------------------------------------------------------------------
# Merging helpers (same logic as GraphRAG's extract_graph.py)
# ---------------------------------------------------------------------------

def _merge_entities(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    all_e = pd.concat(dfs, ignore_index=True)
    if all_e.empty:
        return _empty_entities()
    return (
        all_e
        .groupby(["title", "type"], sort=False)
        .agg(
            description=("description", list),
            text_unit_ids=("source_id", list),
            frequency=("source_id", "count"),
        )
        .reset_index()
    )


def _merge_relationships(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    all_r = pd.concat(dfs, ignore_index=True)
    if all_r.empty:
        return _empty_relationships()
    return (
        all_r
        .groupby(["source", "target"], sort=False)
        .agg(
            description=("description", list),
            text_unit_ids=("source_id", list),
            weight=("weight", "sum"),
        )
        .reset_index()
    )


def _filter_orphans(
    relationships: pd.DataFrame, entities: pd.DataFrame
) -> pd.DataFrame:
    if relationships.empty or entities.empty:
        return relationships
    valid = set(entities["title"])
    mask = relationships["source"].isin(valid) & relationships["target"].isin(valid)
    return relationships[mask].reset_index(drop=True)


def _empty_entities() -> pd.DataFrame:
    return pd.DataFrame(columns=["title", "type", "description", "source_id"])


def _empty_relationships() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["source", "target", "weight", "description", "source_id"]
    )
