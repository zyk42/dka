"""
Step 0 – Prepare a chunked corpus from raw documents.

Accepts a directory (or single file) of .txt / .md / .pdf documents and
produces the chunks JSONL consumed by `dka-build-kg`:

    {"id": "...", "text": "title\\n\\nchunk text", "title": "...", "source": "..."}

Documents are split into ~`chunk_size`-token chunks (whitespace tokenizer
approximation; 1 token ≈ 1 word for English) with `chunk_overlap` overlap.

Usage:
    dka-prepare --input docs/ --output data/chunks.jsonl
    dka-prepare --input paper.pdf --output data/chunks.jsonl --chunk-size 512
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)
logger = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = {".txt", ".md", ".pdf"}


def _read_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as e:
        raise ImportError(
            "PDF input requires the 'pypdf' package: pip install dka[pdf]"
        ) from e
    reader = PdfReader(str(path))
    return "\n\n".join(page.extract_text() or "" for page in reader.pages)


def read_document(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        return _read_pdf(path)
    return path.read_text(encoding="utf-8", errors="ignore")


def chunk_text(text: str, chunk_size: int = 512, chunk_overlap: int = 64) -> list[str]:
    """Split text into word-based chunks with overlap."""
    words = re.split(r"\s+", text.strip())
    words = [w for w in words if w]
    if not words:
        return []
    chunks = []
    step = max(1, chunk_size - chunk_overlap)
    for start in range(0, len(words), step):
        piece = " ".join(words[start : start + chunk_size])
        if piece:
            chunks.append(piece)
        if start + chunk_size >= len(words):
            break
    return chunks


def make_id(source: str, title: str, idx: int) -> str:
    h = hashlib.md5(f"{source}::{title}".encode()).hexdigest()[:8]
    return f"{source}_{h}_{idx}"


def iter_input_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    files = sorted(
        p for p in input_path.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
    )
    return files


def prepare_corpus(
    input_path: Path,
    output_path: Path,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> int:
    files = iter_input_files(input_path)
    if not files:
        raise ValueError(f"No .txt/.md/.pdf files found under {input_path}")
    logger.info("Found %d document(s) under %s", len(files), input_path)

    chunks: list[dict] = []
    for path in files:
        text = read_document(path)
        if not text.strip():
            logger.warning("Empty document, skipped: %s", path)
            continue
        title = path.stem
        pieces = chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        for i, piece in enumerate(pieces):
            chunks.append({
                "id": make_id("doc", f"{title}_{i}", len(chunks)),
                "text": f"{title}\n\n{piece}" if i == 0 else piece,
                "title": title,
                "source": str(path),
            })
        logger.info("  %s → %d chunks", path.name, len(pieces))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for item in chunks:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    logger.info("Wrote %d chunks → %s", len(chunks), output_path)
    return len(chunks)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Chunk raw documents (.txt/.md/.pdf) into a DKA corpus JSONL."
    )
    parser.add_argument("--input", required=True, help="Input file or directory")
    parser.add_argument("--output", required=True, help="Output chunks JSONL path")
    parser.add_argument("--chunk-size", type=int, default=512, help="Words per chunk")
    parser.add_argument("--chunk-overlap", type=int, default=64, help="Overlap in words")
    args = parser.parse_args()

    n = prepare_corpus(
        Path(args.input), Path(args.output),
        chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap,
    )
    print(f"\nDone: {n} chunks → {args.output}")
    print(f"Next: dka-build-kg --input {args.output} --config config.yaml")


if __name__ == "__main__":
    main()
