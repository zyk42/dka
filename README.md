# DKA: Domain Knowledge Annealing

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![TRL](https://img.shields.io/badge/training-TRL-orange.svg)](https://github.com/huggingface/trl)
[![Gradio](https://img.shields.io/badge/UI-Gradio-yellow.svg)](https://gradio.app/)

**Domain Knowledge Annealing (DKA)** is a framework for building domain-expert LLMs from an unstructured corpus. Instead of treating training data as an unstructured collection, DKA organizes learning as a *"Local-to-Global"* annealing process:

1. **Domain knowledge graph construction with provenance tracking** — LLM-based entity/relation extraction over corpus chunks; every node and edge remembers which chunks it came from.
2. **KG-guided QA synthesis** — rare-node-guided weighted random walks over the graph produce cross-document reasoning paths, which are turned into multi-hop QA pairs (with an LLM self-validation pass); single-chunk QA pairs are synthesized directly from the corpus.
3. **Two-stage progressive training** — Stage 1 injects local (intra-document) knowledge; Stage 2 anneals the model on global (inter-document, multi-hop) reasoning. Both stages run on [TRL](https://github.com/huggingface/trl)'s `SFTTrainer`.

> Paper: *Domain Knowledge Annealing: A Structured Local-to-Global Paradigm for Constructing Expert LLMs*

## Pipeline Overview

```
raw docs (.txt/.md/.pdf)
      │  dka-prepare
      ▼
chunks.jsonl ────────────────┐
      │  dka-build-kg        │  dka-gen-qa --type intra
      ▼                      ▼
knowledge_graph.json     qa_intra.jsonl (Stage 1: local)
      │  ├─ dka-export-neo4j → Neo4j (optional)
      │  dka-walk
      ▼
walks.jsonl
      │  dka-gen-qa --type inter
      ▼
qa_inter.jsonl (Stage 2: global)
      │
      ▼  dka-train --stage 1  →  dka-train --stage 2
domain-expert LLM
```

## Installation

```bash
git clone https://github.com/zyk42/dka.git && cd dka
pip install -e .            # core pipeline
pip install -e ".[ui]"      # + Gradio web UI
pip install -e ".[train]"   # + two-stage training (TRL / PEFT / wandb)
pip install -e ".[neo4j]"   # + Neo4j export
pip install -e ".[pdf]"     # + PDF input for dka-prepare
```

You also need an OpenAI-compatible LLM endpoint for extraction and QA synthesis
(e.g. [vLLM](https://github.com/vllm-project/vllm)):

```bash
python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-32B-Instruct --port 8000
```

## Two Ways to Run

### A. Web UI (Gradio)

```bash
dka-ui                       # http://127.0.0.1:7860
dka-ui --host 0.0.0.0 --port 8000
dka-ui --share               # public *.gradio.live link
```

The UI walks you through the whole pipeline in five tabs:
**Corpus → Knowledge Graph (+ Neo4j export) → Random Walk → QA Synthesis → Training**.

The **LLM Endpoint Settings** panel at the top is shared by KG construction and
QA synthesis — just fill in the **Base URL** and **API Key** of any
OpenAI-compatible endpoint. For a locally deployed model (vLLM / Ollama /
SGLang), use its local URL (e.g. `http://localhost:8000/v1`) with any non-empty
key (e.g. `dummy`).

The Training tab generates the exact `torchrun` command (so you can copy it to
a terminal) and can also launch training directly with live log streaming.

### B. Command Line

Every UI step maps 1:1 to a CLI command — see Quick Start below.

## Quick Start

```bash
cp config.example.yaml config.yaml   # point llm.base_url / llm.model at your endpoint

# 0. Chunk your corpus (a folder of .txt/.md/.pdf)
dka-prepare --input docs/ --output data/chunks.jsonl

# 1. Build the knowledge graph (resumable, checkpointed)
dka-build-kg --input data/chunks.jsonl --config config.yaml

# 1b. (optional) Explore the graph in Neo4j
dka-export-neo4j --graph data/knowledge_graph.json --password <neo4j-password>

# 2. Rare-node-guided weighted random walks
dka-walk --graph data/knowledge_graph.json --config config.yaml --num-walks 10000

# 3. Synthesize QA pairs
dka-gen-qa --chunks data/chunks.jsonl --config config.yaml --type intra
dka-gen-qa --walks data/walks.jsonl --chunks data/chunks.jsonl --config config.yaml --type inter

# 4. Two-stage Local-to-Global training (TRL)
torchrun --nproc_per_node=2 -m dka.cli.train \
    --stage 1 --model_path /path/to/base-model \
    --data_path data/qa_intra.jsonl --output_dir models/dka_s1

torchrun --nproc_per_node=2 -m dka.cli.train \
    --stage 2 --model_path models/dka_s1 \
    --data_path data/qa_inter.jsonl --output_dir models/dka_s2
```

## Knowledge Graph Formats

DKA supports two graph representations:

| Format | Produced by | Description |
|---|---|---|
| **JSON (node-link)** | `dka-build-kg` | NetworkX node-link JSON. Nodes carry `type`, `description`, `frequency`, `text_unit_ids` (chunk provenance); edges carry `weight`, `description`, `text_unit_ids`. Used by `dka-walk`. |
| **Neo4j** | `dka-export-neo4j` | Entities imported with one label per entity type (`:Concept`, `:Person`, …), uniqueness constraint on `name`; edges as `:RELATED` relationships with weight and provenance. For visualization and interactive exploration. |

## QA Data Format

Both intra and inter QA samples use the ShareGPT-style chat format (compatible with TRL / LLaMA-Factory):

```json
{
  "conversations": [
    {"from": "human", "value": "Which researcher ...?"},
    {"from": "gpt", "value": "Step 1: ...\nStep 2: ...\n\nFinal Answer: ..."}
  ],
  "meta": {"type": "inter", "walk_nodes": [...], "chunks_crossed": 3, "target_node": "..."}
}
```

Inter-chunk samples are generated with a two-call protocol: the LLM first writes
a reasoning chain + question (answer-blurred), then a second call validates the
sample (PASS/FAIL) against hallucination, answer leakage, and uniqueness checks.

## Training Stages

| Stage | Data | Purpose | Default LR |
|---|---|---|---|
| 0 (Mix FT) | intra + inter, shuffled | baseline | 2e-5 |
| 1 (Local) | `qa_intra.jsonl` | inject discrete local facts (high entropy) | 2e-5 |
| 2 (Annealing) | `qa_inter.jsonl` | integrate facts into global reasoning (low entropy) | 1e-5 |

All stages use LoRA (r=8, α=16 by default) on top of TRL `SFTTrainer` with
`assistant_only_loss=True`. After each stage the LoRA adapter is merged and a
standard HF checkpoint is saved, so Stage 2 can initialize directly from
Stage 1's output directory.

## Repository Structure

```
dka/
├── src/dka/
│   ├── prompts.py            # entity/relation extraction prompts (adapted from GraphRAG)
│   ├── extractor.py          # async LLM extraction → entity/relation tables
│   ├── graph_builder.py      # NetworkX graph build / merge / IO, provenance tracking
│   ├── neo4j_exporter.py     # graph → Neo4j import
│   ├── walker.py             # rare-node-guided weighted random walk
│   ├── qa_generator.py       # intra / inter QA synthesis + PASS/FAIL validation
│   ├── cli/
│   │   ├── prepare_corpus.py # dka-prepare: docs → chunks.jsonl
│   │   ├── build_kg.py       # dka-build-kg: chunks → knowledge_graph.json
│   │   ├── export_neo4j.py   # dka-export-neo4j: graph JSON → Neo4j
│   │   ├── walk.py           # dka-walk: graph → walks.jsonl
│   │   ├── gen_qa.py         # dka-gen-qa: walks/chunks → QA JSONL (resumable)
│   │   └── train.py          # dka-train: two-stage TRL SFT
│   └── ui/
│       └── app.py            # dka-ui: Gradio web UI for the full pipeline
├── config.example.yaml
├── examples/
└── pyproject.toml
```

## Acknowledgements

- Entity/relation extraction is adapted from [Microsoft GraphRAG](https://github.com/microsoft/graphrag) (MIT).
- Training is built on [TRL](https://github.com/huggingface/trl), [PEFT](https://github.com/huggingface/peft) and [Transformers](https://github.com/huggingface/transformers).
- The Neo4j import design (label-per-type + `MERGE` by name) follows common community practice.

## Citation

If you find DKA useful in your research, please cite:

```bibtex
@article{dka2026,
  title  = {Domain Knowledge Annealing: A Structured Local-to-Global Paradigm for Constructing Expert LLMs},
  author = {DKA Authors},
  year   = {2026},
  note   = {arXiv preprint}
}
```

## License

MIT (see [LICENSE](LICENSE)).
