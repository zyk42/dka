"""
Two-stage Knowledge Annealing SFT training using TRL SFTTrainer.

Stage 0 (mix):   intra + inter shuffled together, single stage — Mix FT baseline
Stage 1 (intra): N epochs on single-chunk QA — local knowledge injection (DKA-S1)
Stage 2 (inter): N epochs on multi-hop QA — global reasoning annealing (DKA-S2)

All stages use LoRA by default. After training, LoRA weights are merged into
the base model so each stage's output is a plain HF checkpoint that can be
loaded directly as the next stage's --model_path.

Usage (2-GPU example):
    # Stage 1: local knowledge injection
    torchrun --nproc_per_node=2 -m dka.cli.train \
        --stage 1 --run_name dka-s1 \
        --model_path /path/to/base-model \
        --data_path data/qa_intra.jsonl \
        --output_dir models/dka_s1

    # Stage 2: global reasoning annealing (initialised from Stage 1)
    torchrun --nproc_per_node=2 -m dka.cli.train \
        --stage 2 --run_name dka-s2 \
        --model_path models/dka_s1 \
        --data_path data/qa_inter.jsonl \
        --output_dir models/dka_s2

    # Mix FT baseline (comma-separated paths → merged & shuffled)
    torchrun --nproc_per_node=2 -m dka.cli.train \
        --stage 0 --run_name mix-ft \
        --model_path /path/to/base-model \
        --data_path data/qa_intra.jsonl,data/qa_inter.jsonl \
        --output_dir models/mix_ft
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Chat templates with {% generation %} markers required by TRL assistant_only_loss.
# ---------------------------------------------------------------------------
LLAMA2_BASE_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] in ['user', 'human'] %}"
    "### Human: {{ message['content'] }}\n\n### Assistant: "
    "{% elif message['role'] in ['assistant', 'gpt'] %}"
    "{% generation %}{{ message['content'] }}{% endgeneration %}"
    "{{ eos_token }}"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}### Assistant: {% endif %}"
)

# Simplified Qwen2.5 template with {% generation %} markers (no tools support needed for training).
QWEN2_TRAINING_CHAT_TEMPLATE = (
    "{%- if messages[0]['role'] == 'system' %}"
    "{{- '<|im_start|>system\\n' + messages[0]['content'] + '<|im_end|>\\n' }}"
    "{%- else %}"
    "{{- '<|im_start|>system\\nYou are a helpful assistant.<|im_end|>\\n' }}"
    "{%- endif %}"
    "{%- for message in messages %}"
    "{%- if message['role'] == 'user' %}"
    "{{- '<|im_start|>user\\n' + message['content'] + '<|im_end|>\\n' }}"
    "{%- elif message['role'] == 'assistant' %}"
    "{{- '<|im_start|>assistant\\n' }}"
    "{% generation %}{{ message['content'] }}{% endgeneration %}"
    "{{- '<|im_end|>\\n' }}"
    "{%- endif %}"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n' }}{%- endif %}"
)

LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

ROLE_MAP = {"human": "user", "gpt": "assistant"}


def _load_samples(path: str) -> list[dict]:
    samples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                obj = json.loads(line)
                messages = [
                    {"role": ROLE_MAP.get(msg["from"], msg["from"]), "content": msg["value"]}
                    for msg in obj["conversations"]
                ]
                samples.append({"messages": messages})
    return samples


def load_dataset(data_path: str, stage: int, seed: int = 42) -> Dataset:
    """
    Load dataset from one or more comma-separated JSONL files.
    Stage 0 (Mix): merge and shuffle all provided files.
    """
    paths = [p.strip() for p in data_path.split(",")]

    if len(paths) == 1:
        samples = _load_samples(paths[0])
        logger.info("Loaded %d samples from %s", len(samples), paths[0])
    else:
        # Stage 0: merge + shuffle
        samples = []
        for p in paths:
            part = _load_samples(p)
            logger.info("  %s → %d samples", p, len(part))
            samples.extend(part)
        random.seed(seed)
        random.shuffle(samples)
        logger.info("Mix FT: %d total samples (shuffled, seed=%d)", len(samples), seed)

    return Dataset.from_list(samples)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, required=True, choices=[0, 1, 2],
                        help="0=mix, 1=intra (local/DKA-S1), 2=inter (DKA-S2)")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--data_path", required=True,
                        help="Path(s) to JSONL; comma-separated for stage 0 mix")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--run_name", default=None,
                        help="WandB run name (default: dka-mix/dka-s1/dka-s2)")
    # Training schedule
    parser.add_argument("--num_epochs", type=int, default=4)
    parser.add_argument("--per_device_batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=None,
                        help="LR override (default: 2e-5 for stage 0/1, 1e-5 for stage 2)")
    parser.add_argument("--warmup_steps", type=int, default=100)
    # LoRA
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    args = parser.parse_args()

    lr = args.lr or (1e-5 if args.stage == 2 else 2e-5)
    run_name = args.run_name or {0: "dka-mix", 1: "dka-s1", 2: "dka-s2"}[args.stage]

    logger.info(
        "Stage %d | run=%s | epochs=%d | lr=%g | model=%s",
        args.stage, run_name, args.num_epochs, lr, args.model_path,
    )

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"
    if tokenizer.chat_template is None:
        logger.info("No chat template found — setting LLaMA-2 base template")
        tokenizer.chat_template = LLAMA2_BASE_CHAT_TEMPLATE
    elif "{% generation %}" not in tokenizer.chat_template:
        # TRL requires {% generation %} markers for assistant_only_loss.
        # Qwen2.5 and similar models need a patched training-compatible template.
        logger.info("Chat template missing {% generation %} markers — applying training-compatible template")
        tokenizer.chat_template = QWEN2_TRAINING_CHAT_TEMPLATE

    # ── Model + LoRA ──────────────────────────────────────────────────────────
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        use_cache=False,
    )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    # Required for gradient checkpointing + LoRA
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    # ── Dataset ───────────────────────────────────────────────────────────────
    dataset = load_dataset(args.data_path, args.stage)

    # ── SFTConfig ─────────────────────────────────────────────────────────────
    output_dir = str(Path(args.output_dir))
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    sft_config = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=True,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_steps=args.warmup_steps,
        weight_decay=0.01,
        max_grad_norm=1.0,
        bf16=True,
        tf32=True,
        max_length=args.max_seq_len,
        assistant_only_loss=True,
        save_strategy="no",
        save_only_model=True,
        logging_steps=20,
        report_to="wandb",
        run_name=run_name,
        dataset_num_proc=4,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
    )

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    effective_batch = args.per_device_batch_size * args.grad_accum * world_size
    logger.info(
        "Training: %d samples × %d epochs | effective batch=%d | LoRA r=%d α=%d",
        len(dataset), args.num_epochs, effective_batch, args.lora_rank, args.lora_alpha,
    )

    trainer.train()

    # Merge LoRA weights into base model and save as a standard HF checkpoint.
    # This ensures Stage 2 can load Stage 1's output with plain AutoModelForCausalLM.
    if int(os.environ.get("RANK", 0)) == 0:
        peft_model = trainer.model
        if hasattr(peft_model, "module"):          # unwrap DDP
            peft_model = peft_model.module
        merged = peft_model.merge_and_unload()
        merged.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        logger.info("Model (LoRA merged) saved → %s", output_dir)


if __name__ == "__main__":
    main()
