"""
Step 0 – Prepare a chunked corpus from raw documents.

Accepts a directory (or single file) containing any mix of:

  - ``.txt`` / ``.md``  – plain documents, split into ~``chunk_size``-word
                          chunks with ``chunk_overlap`` overlap.
  - ``.jsonl``          – pre-segmented corpus, one record per line. Each line
                          is a JSON object; the text is read from the first
                          available of the keys: text / content / paragraph /
                          paragraph_text / body. Optional ``id`` and ``title``
                          keys are reused. Records longer than ``chunk_size``
                          words are further split.

PDF is NOT supported — convert documents to .md / .txt first.

Output JSONL (one chunk per line) consumed by `dka-build-kg`:

    {"id": "...", "text": "title\\n\\nchunk text", "title": "...", "source": "..."}

Usage:
    dka-prepare --input docs/ --output data/chunks.jsonl
    dka-prepare --input corpus.jsonl --output data/chunks.jsonl --chunk-size 512
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

SUPPORTED_SUFFIXES = {".txt", ".md", ".jsonl"}

# Candidate keys for locating the text inside a JSONL record
TEXT_KEYS = ("text", "content", "paragraph", "paragraph_text", "body")


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


def _emit(text: str, title: str, source: str, chunks: list[dict],
          chunk_size: int, chunk_overlap: int, base_id: str | None = None) -> None:
    """Chunk one document/record and append to the corpus list."""
    pieces = chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    for i, piece in enumerate(pieces):
        if base_id and len(pieces) == 1:
            cid = base_id  # keep original id when the record is not split
        else:
            cid = make_id(source, f"{title}_{i}", len(chunks))
            if base_id:
                cid = f"{base_id}_{i}"
        chunks.append({
            "id": cid,
            "text": f"{title}\n\n{piece}" if (i == 0 and title) else piece,
            "title": title,
            "source": source,
        })


def _process_text_file(path: Path, chunks: list[dict],
                       chunk_size: int, chunk_overlap: int) -> int:
    text = path.read_text(encoding="utf-8", errors="ignore")
    if not text.strip():
        logger.warning("Empty document, skipped: %s", path)
        return 0
    before = len(chunks)
    _emit(text, title=path.stem, source=str(path), chunks=chunks,
          chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return len(chunks) - before


def _process_jsonl_file(path: Path, chunks: list[dict],
                        chunk_size: int, chunk_overlap: int) -> int:
    before = len(chunks)
    skipped = 0
    with open(path, encoding="utf-8", errors="ignore") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("%s:%d invalid JSON, skipped", path, lineno)
                skipped += 1
                continue

            if isinstance(record, str):
                text, title, rid = record, "", None
            elif isinstance(record, dict):
                text = next((str(record[k]) for k in TEXT_KEYS
                             if record.get(k) and str(record[k]).strip()), "")
                title = str(record.get("title") or "")
                rid = record.get("id")
            else:
                skipped += 1
                continue

            if not text.strip():
                skipped += 1
                continue
            _emit(text, title=title, source=str(path), chunks=chunks,
                  chunk_size=chunk_size, chunk_overlap=chunk_overlap,
                  base_id=str(rid) if rid else None)
    if skipped:
        logger.warning("%s: %d record(s) skipped (no text / invalid)", path, skipped)
    return len(chunks) - before


def iter_input_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    return sorted(
        p for p in input_path.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
    )


def prepare_corpus(
    input_path: Path,
    output_path: Path,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> int:
    files = iter_input_files(input_path)
    if not files:
        raise ValueError(
            f"No .txt/.md/.jsonl files found under {input_path} "
            "(PDF is not supported — convert to .md/.txt first)."
        )
    logger.info("Found %d file(s) under %s", len(files), input_path)

    chunks: list[dict] = []
    for path in files:
        suffix = path.suffix.lower()
        if suffix == ".jsonl":
            n = _process_jsonl_file(path, chunks, chunk_size, chunk_overlap)
        else:
            n = _process_text_file(path, chunks, chunk_size, chunk_overlap)
        logger.info("  %s → %d chunks", path.name, n)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for item in chunks:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    logger.info("Wrote %d chunks → %s", len(chunks), output_path)
    return len(chunks)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Chunk a folder of .txt/.md/.jsonl documents into a DKA corpus JSONL."
    )
    parser.add_argument("--input", required=True,
                        help="Input file or directory (mixed .txt / .md / .jsonl supported)")
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
