#!/usr/bin/env python3
"""Chonkie-based chunker for SOC analysis with model-specific presets.

Wraps the Chonkie library (https://github.com/chonkie-inc/chonkie) with defaults
tuned for stream-of-consciousness literary analysis.  Falls back to the built-in
soc_chunker.py if Chonkie is not installed.

Install:
    pip install chonkie                # base chunkers
    pip install chonkie[semantic]      # + SemanticChunker
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

from scripts.models import Chunk

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model presets — chunk_size in tokens, overlap in tokens
# ---------------------------------------------------------------------------

@dataclass
class ModelPreset:
    chunk_size: int
    overlap: int
    description: str


PRESETS: dict[str, ModelPreset] = {
    "qwen3-8b":       ModelPreset(9_000,   600,   "Qwen3 8B — conservative for 32K context"),
    "qwen3-14b":      ModelPreset(11_000,  800,   "Qwen3 14B/32B — moderate for 32K context"),
    "llama4-scout":   ModelPreset(12_000,  1_000, "Llama 4 Scout 17B — practical ceiling ~128K"),
    "llama4-maverick": ModelPreset(17_000, 1_000, "Llama 4 Maverick 400B MoE"),
    "gemini-pro":     ModelPreset(20_000,  1_200, "Gemini 2.5 Pro — strong long-context"),
    "gemini-flash":   ModelPreset(12_000,  800,   "Gemini 2.5 Flash — faster, tighter chunks"),
    "claude-sonnet":  ModelPreset(20_000,  1_000, "Claude Sonnet 4.x — nuanced analysis"),
    "claude-opus":    ModelPreset(25_000,  1_200, "Claude Opus 4.x — best for literary analysis"),
}


def list_presets() -> None:
    """Print available model presets."""
    print("Available model presets:\n")
    for name, p in PRESETS.items():
        print(f"  {name:20s}  chunk={p.chunk_size:,} tokens  overlap={p.overlap:,}  — {p.description}")


# ---------------------------------------------------------------------------
# Chonkie wrapper
# ---------------------------------------------------------------------------

def chunk_with_chonkie(
    text: str,
    source_file: str,
    chunker_type: str = "sentence",
    chunk_size: int = 20_000,
    overlap: int = 1_000,
    tokenizer: str = "gpt2",
) -> list[Chunk]:
    """Chunk text using the Chonkie library.

    Args:
        text: Full plain text to chunk.
        source_file: Original filename for metadata.
        chunker_type: One of 'sentence', 'recursive', 'semantic', 'token'.
        chunk_size: Target chunk size in tokens.
        overlap: Overlap context in tokens.
        tokenizer: Tokenizer name for Chonkie.

    Returns:
        List of Chunk objects.
    """
    try:
        import chonkie
    except ImportError:
        logger.error(
            "Chonkie is not installed. Install with: pip install chonkie\n"
            "Falling back to built-in chunker."
        )
        from scripts.soc_chunker import chunk_text
        return chunk_text(text, source_file, chunk_size, overlap)

    chunker_map = {
        "sentence": chonkie.SentenceChunker,
        "recursive": chonkie.RecursiveChunker,
        "token": chonkie.TokenChunker,
    }

    if chunker_type == "semantic":
        try:
            chunker_cls = chonkie.SemanticChunker
        except AttributeError:
            logger.error(
                "SemanticChunker not available. Install with: pip install chonkie[semantic]"
            )
            sys.exit(1)
    else:
        chunker_cls = chunker_map.get(chunker_type)
        if chunker_cls is None:
            logger.error("Unknown chunker type: %s. Choose from: %s", chunker_type, list(chunker_map))
            sys.exit(1)

    logger.info(
        "Chunking %s with %s (size=%d, overlap=%d, tokenizer=%s)",
        source_file, chunker_type, chunk_size, overlap, tokenizer,
    )

    # Build the chunker
    kwargs: dict = {"tokenizer": tokenizer, "chunk_size": chunk_size}
    if chunker_type == "semantic":
        kwargs["threshold"] = "auto"
    chunker = chunker_cls(**kwargs)

    raw_chunks = chunker(text)

    # Convert to our Chunk model
    chunks: list[Chunk] = []
    stem = Path(source_file).stem
    for i, rc in enumerate(raw_chunks):
        chunk_text = rc.text if hasattr(rc, "text") else str(rc)
        chunks.append(
            Chunk(
                source_file=source_file,
                chunk_id=f"{stem}_{i:03d}",
                chunk_label=f"Chunk {i + 1}",
                chunk_text=chunk_text,
                chunk_index=i,
                context_before=getattr(rc, "context_before", "") or "",
                context_after=getattr(rc, "context_after", "") or "",
            )
        )

    logger.info("Produced %d chunks from %s", len(chunks), source_file)
    return chunks


# ---------------------------------------------------------------------------
# Pipeline mode: chunk → refine overlap → export JSON
# ---------------------------------------------------------------------------

def run_pipeline(
    text: str,
    source_file: str,
    chunk_size: int = 20_000,
    overlap: int = 1_000,
    tokenizer: str = "gpt2",
    output_path: Path | None = None,
) -> list[Chunk]:
    """Full pipeline: chunk with Chonkie Pipeline API, export to JSON."""
    try:
        from chonkie import Pipeline
    except ImportError:
        logger.error("Chonkie Pipeline not available. Install with: pip install chonkie")
        sys.exit(1)

    out_file = output_path or Path(f"{Path(source_file).stem}_chunks.json")

    doc = (
        Pipeline()
        .process_with("text")
        .chunk_with("sentence", tokenizer=tokenizer, chunk_size=chunk_size)
        .refine_with("overlap", context_size=overlap)
        .export_with("json", file=str(out_file))
        .run(texts=text)
    )

    chunks: list[Chunk] = []
    stem = Path(source_file).stem
    for i, rc in enumerate(doc.chunks):
        chunks.append(
            Chunk(
                source_file=source_file,
                chunk_id=f"{stem}_{i:03d}",
                chunk_label=f"Chunk {i + 1}",
                chunk_text=rc.text,
                chunk_index=i,
                context_before=getattr(rc, "context_before", "") or "",
                context_after=getattr(rc, "context_after", "") or "",
            )
        )

    logger.info("Pipeline produced %d chunks → %s", len(chunks), out_file)
    return chunks


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Chonkie-based chunker for SOC literary analysis"
    )
    parser.add_argument("file", nargs="?", type=Path, help="Path to text file")
    parser.add_argument(
        "--model", "-m",
        choices=list(PRESETS),
        default="claude-sonnet",
        help="Model preset for chunk size (default: claude-sonnet)",
    )
    parser.add_argument(
        "--chunker", "-c",
        choices=["sentence", "recursive", "semantic", "token"],
        default="sentence",
        help="Chonkie chunker type (default: sentence)",
    )
    parser.add_argument("--pipeline", action="store_true", help="Use Chonkie Pipeline mode")
    parser.add_argument("--output", "-o", type=Path, help="Output JSON path")
    parser.add_argument("--list-presets", action="store_true", help="Show model presets and exit")
    parser.add_argument("--tokenizer", default="gpt2", help="Tokenizer for Chonkie")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if args.list_presets:
        list_presets()
        return

    if not args.file:
        parser.error("file is required (unless using --list-presets)")

    preset = PRESETS[args.model]
    text = args.file.read_text(encoding="utf-8")

    if args.pipeline:
        chunks = run_pipeline(
            text, args.file.name,
            chunk_size=preset.chunk_size,
            overlap=preset.overlap,
            tokenizer=args.tokenizer,
            output_path=args.output,
        )
    else:
        chunks = chunk_with_chonkie(
            text, args.file.name,
            chunker_type=args.chunker,
            chunk_size=preset.chunk_size,
            overlap=preset.overlap,
            tokenizer=args.tokenizer,
        )

    # Write output
    data = [c.model_dump() for c in chunks]
    output_json = json.dumps(data, indent=2, ensure_ascii=False)

    if args.output and not args.pipeline:
        args.output.write_text(output_json, encoding="utf-8")
        print(f"Wrote {len(chunks)} chunks to {args.output}")
    elif not args.pipeline:
        print(output_json)
    else:
        print(f"Pipeline complete: {len(chunks)} chunks")


if __name__ == "__main__":
    main()
