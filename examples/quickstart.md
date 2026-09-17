# Quickstart (toy example)

This example runs the full DKA pipeline on the tiny corpus in `examples/docs/`.

## 1. Start an LLM endpoint

Any OpenAI-compatible server works. With vLLM:

```bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-32B-Instruct --port 8000
```

## 2. Configure

```bash
cp examples/config.yaml config.yaml
# edit llm.model to match the served model name
```

## 3. Run the pipeline

```bash
# docs -> chunks (a folder mixing .txt / .md / .jsonl; convert PDFs to md/txt first)
dka-prepare --input examples/docs --output data/chunks.jsonl --chunk-size 256

# chunks -> knowledge graph (JSON)
dka-build-kg --input data/chunks.jsonl --config config.yaml

# (optional) graph -> Neo4j
dka-export-neo4j --graph data/knowledge_graph.json --password <pw>

# graph -> cross-chunk reasoning paths
dka-walk --graph data/knowledge_graph.json --config config.yaml --num-walks 100

# paths/chunks -> QA pairs (ShareGPT JSONL)
dka-gen-qa --chunks data/chunks.jsonl --config config.yaml --type intra
dka-gen-qa --walks data/walks.jsonl --chunks data/chunks.jsonl --config config.yaml --type inter

# two-stage training
dka-train --stage 1 --model_path /path/to/base-model \
    --data_path data/qa_intra.jsonl --output_dir models/dka_s1
dka-train --stage 2 --model_path models/dka_s1 \
    --data_path data/qa_inter.jsonl --output_dir models/dka_s2
```

Note: the toy corpus in `examples/docs/` is only a few sentences, so the graph
will be tiny and few (if any) walks survive the `min_chunks_crossed=2` filter.
For a meaningful run, point `dka-prepare` at a real document collection and
lower `walk.min_hops` / `walk.min_chunks_crossed` for small corpora.
