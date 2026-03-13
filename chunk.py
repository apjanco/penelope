#!/usr/bin/env python3
"""Penelope — Step 1: Extract and chunk literary texts.

Reads files from input/, chunks them, and writes annotated copies to chunking/
with <chunk-N>...</chunk-N> markup so boundaries can be inspected and manually
adjusted before running the LLM analysis.

Usage:
    # Chunk all files in input/ → chunking/
    python chunk.py --input input/

    # Custom chunk size
    python chunk.py --input input/ --chunk-size 15000 --chunk-overlap 800

    # Chunk a single file
    python chunk.py --input input/mrs_dalloway.txt

    # Specify output directory
    python chunk.py --input input/ --output chunking/

After running, review the files in chunking/ and adjust <chunk-N> boundaries
as needed, then run:  python run.py --input chunking/ --output results/
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from scripts.config import Config
from scripts.extract import extract_all, extract_text
from scripts.soc_chunker import chunk_text, write_chunked_file

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Penelope — extract and chunk literary texts with editable markup",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python chunk.py --input input/\n"
            "  python chunk.py --input input/mrs_dalloway.txt\n"
            "  python chunk.py --input input/ --chunk-size 15000\n"
            "\n"
            "After chunking, review and edit the files in chunking/,\n"
            "then run:  python run.py --input chunking/\n"
        ),
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        type=Path,
        help="Input file or directory containing literary texts (.txt, .doc, .pdf)",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("chunking"),
        help="Output directory for chunked/annotated files (default: chunking/)",
    )
    parser.add_argument(
        "--config", "-c",
        type=Path,
        default=None,
        help="Path to models.yaml (used to read chunk_size/overlap defaults)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Override chunk size in characters (default: from config or 20000)",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=None,
        help="Override chunk overlap in characters (default: from config or 1000)",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=None,
        help="Path to .env file (default: auto-detect in project root)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Load config (just for chunk_size/overlap defaults)
    config = Config.load(env_file=args.env_file, config_file=args.config)
    if args.chunk_size:
        config.chunk_size = args.chunk_size
    if args.chunk_overlap:
        config.chunk_overlap = args.chunk_overlap

    # ── Step 1: Extract text ───────────────────────────────────────────
    input_path: Path = args.input
    if input_path.is_file():
        logger.info("Single-file mode: %s", input_path.name)
        text = extract_text(input_path)
        files = [(input_path, text)]
    elif input_path.is_dir():
        logger.info("Directory mode: %s", input_path)
        files = extract_all(input_path)
    else:
        logger.error("Input path does not exist: %s", input_path)
        sys.exit(1)

    if not files:
        logger.error("No text files found. Exiting.")
        sys.exit(1)

    logger.info("Extracted text from %d file(s)", len(files))

    # ── Step 2: Chunk and write annotated files ────────────────────────
    output_dir: Path = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    total_chunks = 0
    for file_path, text in files:
        chunks = chunk_text(
            text,
            source_file=file_path.name,
            chunk_size=config.chunk_size,
            overlap=config.chunk_overlap,
        )
        logger.info("%s → %d chunks", file_path.name, len(chunks))

        # Write annotated file to chunking/
        out_name = file_path.stem + ".txt"
        out_path = output_dir / out_name
        write_chunked_file(chunks, out_path)
        logger.info("  → %s", out_path)

        total_chunks += len(chunks)

    # ── Summary ────────────────────────────────────────────────────────
    print(f"\nChunked {len(files)} file(s) → {total_chunks} total chunks")
    print(f"Output: {output_dir}/")
    print()
    print("Next steps:")
    print(f"  1. Review and edit chunk boundaries in {output_dir}/")
    print(f"  2. Run analysis:  python run.py --input {output_dir}/")


if __name__ == "__main__":
    main()
