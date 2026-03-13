#!/usr/bin/env python3
"""Penelope — Step 2: Run LLM analysis on pre-chunked texts.

Reads chunk-annotated files from chunking/ (produced by chunk.py),
sends each chunk to one or more LLMs, and exports results.

Usage:
    # Run analysis on all chunked files
    python run.py --input chunking/ --output results/

    # Multiple models (from models.yaml):
    python run.py --input chunking/ --output results/ --config models.yaml

    # Override: run only specific models from the config:
    python run.py --input chunking/ --model gpt-4o --model claude-sonnet

    # Dry run (list chunks, no LLM calls):
    python run.py --input chunking/ --dry-run

Two-step workflow:
    1. python chunk.py --input input/     # extract + chunk → chunking/
       (review and edit chunk boundaries in chunking/)
    2. python run.py --input chunking/    # LLM analysis → results/
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from scripts.analyze import analyze_chunks_multi
from scripts.config import Config
from scripts.export import export_results, print_summary
from scripts.models import ResultRow
from scripts.soc_chunker import is_chunked_file, parse_chunked_dir, parse_chunked_file

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Penelope — run LLM analysis on pre-chunked literary texts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python run.py --input chunking/ --output results/\n"
            "  python run.py --input chunking/ --config models.yaml\n"
            "  python run.py --input chunking/ --model gpt-4o --model claude-sonnet\n"
            "  python run.py --input chunking/ --dry-run\n"
            "\n"
            "First run:  python chunk.py --input input/\n"
            "to produce the chunked files.\n"
        ),
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        type=Path,
        help="Chunked file or directory (from chunk.py) with <chunk-N> markup",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("results"),
        help="Output directory for results (default: results/)",
    )
    parser.add_argument(
        "--config", "-c",
        type=Path,
        default=None,
        help="Path to models.yaml with LLM profiles (default: auto-detect in project root)",
    )
    parser.add_argument(
        "--model", "-m",
        action="append",
        dest="models",
        help=(
            "Run only these model label(s) from models.yaml. "
            "Can be specified multiple times. If omitted, all models in the config are run."
        ),
    )
    parser.add_argument(
        "--format", "-f",
        action="append",
        choices=["csv", "json"],
        dest="formats",
        help="Output format(s). Can be specified multiple times. Default: both csv and json.",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=None,
        help="Path to .env file (default: auto-detect in project root)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List chunks without making LLM calls (useful for verifying chunking)",
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

    # Load config
    config = Config.load(env_file=args.env_file, config_file=args.config)

    # Filter to requested models if --model was specified
    if args.models:
        requested = set(args.models)
        available = {p.label for p in config.get_model_profiles()}
        unknown = requested - available
        if unknown:
            logger.error(
                "Unknown model label(s): %s. Available: %s",
                ", ".join(sorted(unknown)),
                ", ".join(sorted(available)),
            )
            sys.exit(1)
        config.models = [p for p in config.get_model_profiles() if p.label in requested]

    profiles = config.get_model_profiles()
    logger.info("Models to run: %s", ", ".join(p.label for p in profiles))

    # ── Step 1: Read pre-chunked files ─────────────────────────────────
    input_path: Path = args.input
    if input_path.is_file():
        if not is_chunked_file(input_path):
            logger.error(
                "%s does not contain <chunk-N> markup. "
                "Run chunk.py first:  python chunk.py --input input/",
                input_path,
            )
            sys.exit(1)
        logger.info("Single-file mode: %s", input_path.name)
        all_chunks = parse_chunked_file(input_path)
    elif input_path.is_dir():
        # Check that at least one file has markup
        txt_files = sorted(input_path.glob("*.txt"))
        if not txt_files:
            logger.error("No .txt files found in %s", input_path)
            sys.exit(1)
        if not any(is_chunked_file(f) for f in txt_files):
            logger.error(
                "No files in %s contain <chunk-N> markup. "
                "Run chunk.py first:  python chunk.py --input input/ --output %s",
                input_path, input_path,
            )
            sys.exit(1)
        logger.info("Directory mode: %s", input_path)
        all_chunks = parse_chunked_dir(input_path)
    else:
        logger.error("Input path does not exist: %s", input_path)
        sys.exit(1)

    if not all_chunks:
        logger.error("No chunks found. Exiting.")
        sys.exit(1)

    logger.info("Total chunks: %d", len(all_chunks))

    if args.dry_run:
        print(f"\n[DRY RUN] Read {len(all_chunks)} chunks from {input_path}")
        print(f"Models configured: {', '.join(p.label for p in profiles)}")
        for c in all_chunks:
            print(f"  {c.chunk_id:30s}  {c.chunk_label:30s}  {len(c.chunk_text):,} chars")
        return

    # ── Step 2: LLM Analysis (all models) ─────────────────────────────
    all_rows: list[ResultRow] = analyze_chunks_multi(all_chunks, config)

    # ── Step 3: Export ─────────────────────────────────────────────────
    formats = args.formats or ["csv", "json"]
    created = export_results(all_rows, args.output, formats=formats)

    for p in created:
        print(f"  → {p}")

    # Summary (with per-model breakdown)
    print_summary(all_rows)


if __name__ == "__main__":
    main()
