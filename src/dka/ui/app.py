"""
Gradio UI for the DKA pipeline.

Run:
    dka-ui                      # http://127.0.0.1:7860
    dka-ui --port 8000 --share

Tabs:
    1. Corpus      – upload/chunk documents → chunks.jsonl
    2. Knowledge Graph – LLM extraction → graph JSON (+ optional Neo4j export)
    3. Random Walk – rare-node weighted walks → walks.jsonl
    4. QA Synthesis    – intra / inter QA generation (resumable)
    5. Training    – two-stage TRL SFT launcher

The LLM Endpoint panel at the top is shared by KG construction and QA
synthesis. Any OpenAI-compatible endpoint works — for a locally deployed
model (e.g. vLLM) just fill in its URL (http://localhost:8000/v1) and any
non-empty API key.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import gradio as gr
import yaml
from openai import AsyncOpenAI

from dka import __version__
from dka.cli.prepare_corpus import SUPPORTED_SUFFIXES, prepare_corpus
from dka.extractor import ExtractionConfig, extract_all
from dka.graph_builder import (
    build_graph, graph_stats, load_graph, merge_graph, save_graph,
)
from dka.neo4j_exporter import export_to_neo4j
from dka.qa_generator import (
    generate_inter_batch, generate_intra_batch, load_chunk_lookup,
)
from dka.walker import WalkConfig, generate_walks, save_walks

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)
logger = logging.getLogger(__name__)

WORKDIR = Path(os.environ.get("DKA_WORKDIR", "dka_workspace"))
DEFAULT_ENTITY_TYPES = "CONCEPT, MODEL, METHOD, DATASET, METRIC, PAPER, ORGANIZATION, PERSON"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _client(base_url: str, api_key: str) -> AsyncOpenAI:
    return AsyncOpenAI(base_url=base_url.strip(), api_key=api_key.strip() or "dummy")


def _load_jsonl(path: str) -> list[dict]:
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items


def _preview_jsonl(path: str, n: int = 3, max_chars: int = 400) -> str:
    if not Path(path).exists():
        return "(file not found)"
    out = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            item = json.loads(line)
            s = json.dumps(item, ensure_ascii=False, indent=2)
            out.append(s[:max_chars] + (" …" if len(s) > max_chars else ""))
    return "\n\n".join(out)


def _count_lines(path: str) -> int:
    if not Path(path).exists():
        return 0
    with open(path, encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


# ---------------------------------------------------------------------------
# Tab 1: Corpus preparation
# ---------------------------------------------------------------------------

def run_prepare(files, input_dir, output_path, chunk_size, chunk_overlap):
    try:
        if files:
            tmp = Path(tempfile.mkdtemp(prefix="dka_docs_"))
            for f in files:
                shutil.copy(f.name, tmp / Path(f.name).name)
            src = tmp
        elif input_dir and input_dir.strip():
            src = Path(input_dir.strip())
            if not src.exists():
                return f"❌ Path not found: {src}", ""
        else:
            return "❌ Please upload files or provide a server-side directory path.", ""

        n = prepare_corpus(src, Path(output_path),
                           chunk_size=int(chunk_size), chunk_overlap=int(chunk_overlap))
        return f"✅ {n} chunks → {output_path}", _preview_jsonl(output_path)
    except Exception as e:
        logger.exception("prepare failed")
        return f"❌ {e}", ""


# ---------------------------------------------------------------------------
# Tab 2: KG construction (+ Neo4j export)
# ---------------------------------------------------------------------------

def run_build_kg(chunks_path, graph_path, base_url, api_key, model,
                 entity_types, max_gleanings, concurrency, batch_size,
                 limit, progress=gr.Progress()):
    try:
        chunks = _load_jsonl(chunks_path)
    except Exception as e:
        return f"❌ Cannot load chunks: {e}", ""
    if limit and int(limit) > 0:
        chunks = chunks[: int(limit)]

    cfg = ExtractionConfig(
        entity_types=[t.strip() for t in entity_types.split(",") if t.strip()],
        max_gleanings=int(max_gleanings),
    )
    client = _client(base_url, api_key)
    G = None
    total = len(chunks)
    bs = max(1, int(batch_size))
    log_lines = [f"Loaded {total} chunks from {chunks_path}"]

    for start in range(0, total, bs):
        batch = chunks[start : start + bs]
        progress((start + len(batch)) / total,
                 desc=f"Extracting chunks {start + 1}–{start + len(batch)} / {total}")
        try:
            entities, relationships = asyncio.run(extract_all(
                chunks=batch, client=client, model=model.strip(), config=cfg,
                concurrency=int(concurrency),
            ))
        except Exception as e:
            log_lines.append(f"⚠️ batch {start}: {e}")
            continue
        G_new = build_graph(entities, relationships)
        G = G_new if G is None else merge_graph(G, G_new)
        save_graph(G, graph_path)
        log_lines.append(
            f"batch {start + 1}–{start + len(batch)}: +{len(entities)} entities, "
            f"+{len(relationships)} rels → graph {G.number_of_nodes()} nodes / {G.number_of_edges()} edges"
        )

    if G is None:
        return "❌ No entities extracted. Check the LLM endpoint settings.", "\n".join(log_lines)

    stats = graph_stats(G)
    summary = (f"✅ Graph saved → {graph_path}\n"
               f"nodes: {stats['nodes']}   edges: {stats['edges']}   "
               f"avg degree: {stats['avg_degree']:.2f}")
    return summary + "\n\n" + "\n".join(log_lines[-20:]), _preview_graph(graph_path)


def _preview_graph(graph_path: str, n: int = 5) -> str:
    try:
        G = load_graph(graph_path)
    except Exception as e:
        return f"(cannot load graph: {e})"
    lines = ["Nodes:"]
    for i, (name, attrs) in enumerate(G.nodes(data=True)):
        if i >= n:
            break
        lines.append(f"  {name} [{attrs.get('type')}] freq={attrs.get('frequency')}")
    lines.append("Edges:")
    for i, (s, t, attrs) in enumerate(G.edges(data=True)):
        if i >= n:
            break
        lines.append(f"  {s} — {t} (w={attrs.get('weight')})")
    return "\n".join(lines)


def run_neo4j_export(graph_path, uri, user, password, database, rel_type, clear):
    try:
        G = load_graph(graph_path)
        stats = export_to_neo4j(G, uri=uri.strip(), user=user.strip(),
                                password=password, database=database.strip(),
                                rel_type=rel_type.strip() or "RELATED", clear=clear)
        return f"✅ Neo4j import complete: {stats['nodes']} nodes, {stats['relationships']} relationships"
    except Exception as e:
        logger.exception("neo4j export failed")
        return f"❌ {e}"


# ---------------------------------------------------------------------------
# Tab 3: Random walk
# ---------------------------------------------------------------------------

def run_walk(graph_path, walks_path, num_walks, min_hops, max_hops,
             min_chunks_crossed, alpha, rare_threshold, seed):
    try:
        G = load_graph(graph_path)
    except Exception as e:
        return f"❌ Cannot load graph: {e}", ""

    cfg = WalkConfig(
        num_walks=int(num_walks), min_hops=int(min_hops), max_hops=int(max_hops),
        min_chunks_crossed=int(min_chunks_crossed), cross_chunk_alpha=float(alpha),
        seed=int(seed), rare_start=True, rare_frequency_threshold=int(rare_threshold),
    )
    walks = list(generate_walks(G, cfg))
    if not walks:
        return ("❌ No walks generated — graph may be too small or poorly connected. "
                "Try lowering min_hops / min_chunks_crossed."), ""
    save_walks(walks, walks_path)

    avg_hops = sum(len(w.nodes) - 1 for w in walks) / len(walks)
    avg_chunks = sum(w.chunks_crossed for w in walks) / len(walks)
    summary = (f"✅ {len(walks)} walks → {walks_path}\n"
               f"avg hops: {avg_hops:.2f}   avg chunks crossed: {avg_chunks:.2f}")
    preview = "\n".join(" → ".join(w.nodes) for w in walks[:5])
    return summary, preview


# ---------------------------------------------------------------------------
# Tab 4: QA synthesis
# ---------------------------------------------------------------------------

def run_gen_qa(qa_type, chunks_path, walks_path, base_url, api_key, model,
               intra_out, inter_out, temperature, concurrency,
               checkpoint_every, limit, progress=gr.Progress()):
    logs = []
    try:
        if qa_type in ("intra", "both"):
            chunks = _load_jsonl(chunks_path)
            if limit and int(limit) > 0:
                chunks = chunks[: int(limit)]
            done = _count_lines(intra_out)
            if done:
                logs.append(f"Resuming intra: {done} samples exist, skipping {done} chunks")
                chunks = chunks[done:]
            client = _client(base_url, api_key)
            total = len(chunks)
            bs = max(1, int(checkpoint_every))
            written = done
            for start in range(0, total, bs):
                batch = chunks[start : start + bs]
                progress((start + len(batch)) / max(total, 1),
                         desc=f"intra QA {start + 1}–{start + len(batch)} / {total}")
                items = asyncio.run(generate_intra_batch(
                    chunks=batch, client=client, model=model.strip(),
                    temperature=float(temperature), concurrency=int(concurrency),
                ))
                with open(intra_out, "a", encoding="utf-8") as f:
                    for item in items:
                        f.write(json.dumps(item, ensure_ascii=False) + "\n")
                written += len(items)
                logs.append(f"intra {start + 1}–{start + len(batch)}: +{len(items)} (total {written})")

        if qa_type in ("inter", "both"):
            walks = _load_jsonl(walks_path)
            if limit and int(limit) > 0:
                walks = walks[: int(limit)]
            chunk_lookup = load_chunk_lookup(chunks_path)
            done = _count_lines(inter_out)
            if done:
                logs.append(f"Resuming inter: {done} samples exist, skipping {done} walks")
                walks = walks[done:]
            client = _client(base_url, api_key)
            total = len(walks)
            bs = max(1, int(checkpoint_every))
            written = done
            for start in range(0, total, bs):
                batch = walks[start : start + bs]
                progress((start + len(batch)) / max(total, 1),
                         desc=f"inter QA {start + 1}–{start + len(batch)} / {total}")
                items = asyncio.run(generate_inter_batch(
                    walks=batch, chunk_lookup=chunk_lookup, client=client,
                    model=model.strip(), temperature=float(temperature),
                    concurrency=int(concurrency),
                ))
                with open(inter_out, "a", encoding="utf-8") as f:
                    for item in items:
                        f.write(json.dumps(item, ensure_ascii=False) + "\n")
                written += len(items)
                logs.append(f"inter {start + 1}–{start + len(batch)}: +{len(items)} (total {written})")

        preview_path = inter_out if qa_type == "inter" else intra_out
        return "✅ Done\n" + "\n".join(logs[-20:]), _preview_jsonl(preview_path)
    except Exception as e:
        logger.exception("QA generation failed")
        return f"❌ {e}\n\n" + "\n".join(logs[-20:]), ""


# ---------------------------------------------------------------------------
# Tab 5: Training
# ---------------------------------------------------------------------------

TRAIN_PROC: subprocess.Popen | None = None


def build_train_cmd(stage, model_path, data_path, output_dir, run_name,
                    epochs, lr, batch, grad_accum, max_seq_len,
                    lora_rank, lora_alpha, nproc):
    cmd = [
        "torchrun", f"--nproc_per_node={int(nproc)}", "-m", "dka.cli.train",
        "--stage", str(stage), "--model_path", model_path.strip(),
        "--data_path", data_path.strip(), "--output_dir", output_dir.strip(),
        "--num_epochs", str(int(epochs)),
        "--per_device_batch_size", str(int(batch)),
        "--grad_accum", str(int(grad_accum)),
        "--max_seq_len", str(int(max_seq_len)),
        "--lora_rank", str(int(lora_rank)), "--lora_alpha", str(int(lora_alpha)),
    ]
    if run_name and run_name.strip():
        cmd += ["--run_name", run_name.strip()]
    if lr and float(lr) > 0:
        cmd += ["--lr", str(float(lr))]
    return " ".join(cmd)


def run_train(cmd, log_path):
    global TRAIN_PROC
    if TRAIN_PROC and TRAIN_PROC.poll() is None:
        return "⚠️ A training job is already running."
    if not cmd.strip():
        return "❌ Generate the command first."
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    logf = open(log_path, "w", encoding="utf-8")
    TRAIN_PROC = subprocess.Popen(
        cmd.split(), stdout=logf, stderr=subprocess.STDOUT,
        env={**os.environ},
    )
    return f"🚀 Training launched (pid {TRAIN_PROC.pid}), log → {log_path}"


def poll_train_log(log_path):
    """Generator: stream the training log into the UI."""
    last = ""
    while True:
        text = ""
        if Path(log_path).exists():
            with open(log_path, encoding="utf-8", errors="ignore") as f:
                text = f.read()[-6000:]
        status = ""
        if TRAIN_PROC is not None:
            rc = TRAIN_PROC.poll()
            status = "(running)" if rc is None else f"(exited, code {rc})"
        out = f"{status}\n{text}"
        if out != last:
            last = out
            yield out
        if TRAIN_PROC is not None and TRAIN_PROC.poll() is not None:
            yield out
            return
        time.sleep(3)


def stop_train():
    global TRAIN_PROC
    if TRAIN_PROC and TRAIN_PROC.poll() is None:
        TRAIN_PROC.terminate()
        return "⏹️ Stop signal sent."
    return "No running training job."


# ---------------------------------------------------------------------------
# App layout
# ---------------------------------------------------------------------------

def create_app() -> gr.Blocks:
    WORKDIR.mkdir(parents=True, exist_ok=True)
    W = str(WORKDIR)

    theme = gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="slate",
        font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
    )
    css = """
    .dka-header { text-align: center; margin-bottom: 0.2em; }
    .dka-header h1 { margin-bottom: 0.1em; }
    .dka-sub { color: #64748b; font-size: 0.95em; margin-top: 0; }
    .dka-panel { border: 1px solid #e2e8f0; border-radius: 10px; padding: 12px 14px; }
    .dka-run-btn { min-height: 44px; font-weight: 600; }
    footer { display: none !important; }
    """

    with gr.Blocks(title="DKA · Domain Knowledge Annealing", theme=theme, css=css) as app:
        # ── Header ───────────────────────────────────────────────────────────
        gr.Markdown(
            f"# 🕸️ DKA · Domain Knowledge Annealing\n"
            '<p class="dka-sub">Corpus → Knowledge Graph → Random Walks → QA Synthesis → Two-Stage Local-to-Global SFT'
            f" &nbsp;·&nbsp; v{__version__}</p>",
            elem_classes=["dka-header"],
        )

        # ── Shared LLM endpoint settings ─────────────────────────────────────
        with gr.Group(elem_classes=["dka-panel"]):
            gr.Markdown(
                "#### 🔑 LLM Endpoint\n"
                "共享于 **KG 构建** 与 **QA 合成**。任意 OpenAI 兼容端点均可;"
                "本地部署模型(vLLM / Ollama / SGLang)填本地 URL(如 `http://localhost:8000/v1`),"
                "API Key 任意非空即可(如 `dummy`)。"
            )
            with gr.Row():
                llm_url = gr.Textbox(label="Base URL", value="http://localhost:8000/v1",
                                     scale=3, placeholder="http://localhost:8000/v1")
                llm_key = gr.Textbox(label="API Key", value="dummy", type="password", scale=2)
                llm_model = gr.Textbox(label="Model", value="Qwen/Qwen2.5-32B-Instruct", scale=3)

        with gr.Tabs():
            # ── Tab 1: Corpus ────────────────────────────────────────────────
            with gr.Tab("① 语料准备 Corpus"):
                with gr.Row():
                    with gr.Column(scale=2):
                        gr.Markdown("#### 输入\n上传 `.txt / .md / .jsonl` 文件,或填写服务器上的目录路径(支持混合,自动处理)。\n"
                                    "`.jsonl` 每行一条语料记录(读取 text/content/paragraph 字段);PDF 请先自行转换为 md/txt。")
                        prep_files = gr.File(label="上传文档", file_count="multiple",
                                             file_types=list(SUPPORTED_SUFFIXES))
                        prep_dir = gr.Textbox(label="或服务器目录", placeholder="/path/to/docs/")
                        with gr.Accordion("切分参数", open=False):
                            prep_size = gr.Slider(label="Chunk 长度 (词)", value=512,
                                                  minimum=128, maximum=2048, step=64)
                            prep_overlap = gr.Slider(label="Chunk 重叠 (词)", value=64,
                                                     minimum=0, maximum=256, step=16)
                        prep_out = gr.Textbox(label="输出 chunks JSONL", value=f"{W}/chunks.jsonl")
                        prep_btn = gr.Button("▶ 开始切分", variant="primary",
                                             elem_classes=["dka-run-btn"])
                    with gr.Column(scale=3):
                        prep_status = gr.Textbox(label="状态", lines=3, interactive=False)
                        prep_preview = gr.Code(label="数据预览 (前 3 条)", language="json", lines=16)
                prep_btn.click(run_prepare,
                               [prep_files, prep_dir, prep_out, prep_size, prep_overlap],
                               [prep_status, prep_preview])

            # ── Tab 2: Knowledge graph ───────────────────────────────────────
            with gr.Tab("② 知识图谱 KG"):
                with gr.Row():
                    with gr.Column(scale=2):
                        gr.Markdown("#### 抽取配置\n基于 LLM 的实体/关系抽取,按批次自动保存断点,可随时中断重跑。")
                        kg_chunks = gr.Textbox(label="Chunks JSONL", value=f"{W}/chunks.jsonl")
                        kg_out = gr.Textbox(label="输出图谱 JSON", value=f"{W}/knowledge_graph.json")
                        kg_types = gr.Textbox(label="实体类型 (逗号分隔)", value=DEFAULT_ENTITY_TYPES, lines=2)
                        with gr.Accordion("高级参数", open=False):
                            with gr.Row():
                                kg_glean = gr.Number(label="Gleanings 轮数", value=0, minimum=0, maximum=5, step=1)
                                kg_conc = gr.Number(label="并发数", value=8, minimum=1, maximum=128, step=1)
                            with gr.Row():
                                kg_batch = gr.Number(label="断点批次大小", value=200, minimum=10, step=10)
                                kg_limit = gr.Number(label="限量调试 (0 = 全部)", value=0, minimum=0, step=10)
                        kg_btn = gr.Button("▶ 构建知识图谱", variant="primary",
                                           elem_classes=["dka-run-btn"])
                    with gr.Column(scale=3):
                        kg_status = gr.Textbox(label="状态", lines=9, interactive=False)
                        kg_preview = gr.Code(label="图谱预览", lines=13)
                kg_btn.click(run_build_kg,
                             [kg_chunks, kg_out, llm_url, llm_key, llm_model,
                              kg_types, kg_glean, kg_conc, kg_batch, kg_limit],
                             [kg_status, kg_preview])

                gr.Markdown("---")
                with gr.Accordion("⬆ 导出到 Neo4j(可选)", open=False):
                    with gr.Row():
                        neo_uri = gr.Textbox(label="Neo4j URI", value="bolt://localhost:7687", scale=2)
                        neo_user = gr.Textbox(label="用户名", value="neo4j", scale=1)
                        neo_pw = gr.Textbox(label="密码", type="password", value="password", scale=1)
                    with gr.Row():
                        neo_db = gr.Textbox(label="数据库", value="neo4j", scale=1)
                        neo_rel = gr.Textbox(label="关系类型", value="RELATED", scale=1)
                        neo_clear = gr.Checkbox(label="导入前清空数据库", value=False, scale=1)
                    neo_btn = gr.Button("⬆ 导出到 Neo4j", variant="secondary")
                    neo_status = gr.Textbox(label="导出状态", lines=2, interactive=False)
                    neo_btn.click(run_neo4j_export,
                                  [kg_out, neo_uri, neo_user, neo_pw, neo_db, neo_rel, neo_clear],
                                  neo_status)

            # ── Tab 3: Random walk ───────────────────────────────────────────
            with gr.Tab("③ 随机游走 Walk"):
                with gr.Row():
                    with gr.Column(scale=2):
                        gr.Markdown(
                            "#### 游走配置\n"
                            "稀有节点引导的加权随机游走:P(v′|v) ∝ w(v,v′)·α^(跨 chunk),起点权重 1/(freq×deg)。"
                        )
                        wk_graph = gr.Textbox(label="图谱 JSON", value=f"{W}/knowledge_graph.json")
                        wk_out = gr.Textbox(label="输出 walks JSONL", value=f"{W}/walks.jsonl")
                        wk_num = gr.Slider(label="游走条数", value=10000,
                                           minimum=100, maximum=100000, step=100)
                        with gr.Accordion("高级参数", open=False):
                            with gr.Row():
                                wk_min = gr.Number(label="最小跳数", value=3, minimum=1, step=1)
                                wk_max = gr.Number(label="最大跳数", value=5, minimum=1, step=1)
                                wk_cross = gr.Number(label="最少跨 chunk 数", value=2, minimum=1, step=1)
                            with gr.Row():
                                wk_alpha = gr.Number(label="跨 chunk α", value=3.0, step=0.5)
                                wk_rare = gr.Number(label="稀有节点频次阈值", value=2, minimum=1, step=1)
                                wk_seed = gr.Number(label="随机种子", value=42, step=1)
                        wk_btn = gr.Button("▶ 生成游走路径", variant="primary",
                                           elem_classes=["dka-run-btn"])
                    with gr.Column(scale=3):
                        wk_status = gr.Textbox(label="状态", lines=4, interactive=False)
                        wk_preview = gr.Code(label="样例路径 (前 5 条)", lines=14)
                wk_btn.click(run_walk,
                             [wk_graph, wk_out, wk_num, wk_min, wk_max,
                              wk_cross, wk_alpha, wk_rare, wk_seed],
                             [wk_status, wk_preview])

            # ── Tab 4: QA synthesis ──────────────────────────────────────────
            with gr.Tab("④ QA 合成"):
                with gr.Row():
                    with gr.Column(scale=2):
                        gr.Markdown(
                            "#### 合成配置\n"
                            "**intra** = 单文档 QA(Stage 1 局部注入)· "
                            "**inter** = 基于游走路径的多跳 QA(Stage 2 全局退火)。"
                            "支持断点续跑:输出文件已有样本自动跳过。"
                        )
                        qa_type = gr.Radio(["intra", "inter", "both"], value="both", label="QA 类型")
                        qa_chunks = gr.Textbox(label="Chunks JSONL", value=f"{W}/chunks.jsonl")
                        qa_walks = gr.Textbox(label="Walks JSONL (inter 需要)", value=f"{W}/walks.jsonl")
                        with gr.Row():
                            qa_intra_out = gr.Textbox(label="Intra 输出", value=f"{W}/qa_intra.jsonl")
                            qa_inter_out = gr.Textbox(label="Inter 输出", value=f"{W}/qa_inter.jsonl")
                        with gr.Accordion("高级参数", open=False):
                            with gr.Row():
                                qa_temp = gr.Slider(label="Temperature", value=0.7,
                                                    minimum=0.0, maximum=1.5, step=0.1)
                                qa_conc = gr.Number(label="并发数", value=8, minimum=1, maximum=128, step=1)
                            with gr.Row():
                                qa_ckpt = gr.Number(label="断点间隔 N", value=500, minimum=10, step=10)
                                qa_limit = gr.Number(label="限量调试 (0 = 全部)", value=0, minimum=0, step=10)
                        qa_btn = gr.Button("▶ 开始合成", variant="primary",
                                           elem_classes=["dka-run-btn"])
                    with gr.Column(scale=3):
                        qa_status = gr.Textbox(label="状态", lines=9, interactive=False)
                        qa_preview = gr.Code(label="样例输出", language="json", lines=13)
                qa_btn.click(run_gen_qa,
                             [qa_type, qa_chunks, qa_walks, llm_url, llm_key, llm_model,
                              qa_intra_out, qa_inter_out, qa_temp, qa_conc,
                              qa_ckpt, qa_limit],
                             [qa_status, qa_preview])

            # ── Tab 5: Training ──────────────────────────────────────────────
            with gr.Tab("⑤ 两阶段训练"):
                with gr.Row():
                    with gr.Column(scale=2):
                        gr.Markdown(
                            "#### 训练配置\n"
                            "**Stage 1** = intra 局部知识注入 → **Stage 2** = inter 全局退火"
                            "(Stage 2 的模型路径填 Stage 1 的输出目录)· **Stage 0** = mix 基线。"
                        )
                        with gr.Row():
                            tr_stage = gr.Radio([1, 2, 0], value=1, label="Stage")
                            tr_nproc = gr.Number(label="GPU 数", value=2, minimum=1, step=1)
                        tr_model = gr.Textbox(label="模型路径", value="/path/to/base-model")
                        tr_data = gr.Textbox(label="训练数据 JSONL(stage 0 用逗号分隔多个)",
                                             value=f"{W}/qa_intra.jsonl")
                        with gr.Row():
                            tr_outdir = gr.Textbox(label="输出目录", value="models/dka_s1")
                            tr_run = gr.Textbox(label="Run 名称 (wandb)", value="dka-s1")
                        with gr.Accordion("训练超参", open=False):
                            with gr.Row():
                                tr_epochs = gr.Number(label="Epochs", value=4, minimum=1, step=1)
                                tr_lr = gr.Number(label="LR (0 = 默认 2e-5/1e-5)", value=0)
                            with gr.Row():
                                tr_batch = gr.Number(label="Batch / 卡", value=4, minimum=1, step=1)
                                tr_accum = gr.Number(label="梯度累积", value=8, minimum=1, step=1)
                                tr_len = gr.Number(label="最大序列长", value=2048, step=256)
                            with gr.Row():
                                tr_r = gr.Number(label="LoRA r", value=8, minimum=1, step=1)
                                tr_alpha = gr.Number(label="LoRA α", value=16, minimum=1, step=1)
                        tr_cmd_btn = gr.Button("🛠 生成命令", variant="secondary")
                        tr_cmd = gr.Code(label="torchrun 命令(可复制到终端执行)", language="shell", lines=4)
                        tr_log_path = gr.Textbox(label="日志文件", value=f"{W}/train.log")
                        with gr.Row():
                            tr_launch = gr.Button("🚀 启动训练", variant="primary",
                                                  elem_classes=["dka-run-btn"])
                            tr_stop = gr.Button("⏹ 停止", variant="stop")
                        tr_status = gr.Textbox(label="状态", lines=2, interactive=False)
                    with gr.Column(scale=3):
                        tr_log = gr.Code(label="训练日志(自动刷新)", language="shell", lines=30)

                tr_cmd_btn.click(
                    build_train_cmd,
                    [tr_stage, tr_model, tr_data, tr_outdir, tr_run,
                     tr_epochs, tr_lr, tr_batch, tr_accum, tr_len,
                     tr_r, tr_alpha, tr_nproc],
                    tr_cmd,
                )
                tr_launch.click(run_train, [tr_cmd, tr_log_path], tr_status).then(
                    poll_train_log, tr_log_path, tr_log)
                tr_stop.click(stop_train, None, tr_status)

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="DKA Gradio UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    app = create_app()
    app.queue()
    app.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
