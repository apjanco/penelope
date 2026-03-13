#!/usr/bin/env python3
"""SOC Analysis Pipeline — CLI entry point.

Usage:
    # Single model (from .env):
    python run.py --input input/ --output results/

    # Multiple models (from models.yaml):
    python run.py --input input/ --output results/ --config models.yaml

    # Override: run only specific models from the config:
    python run.py --input input/ --config models.yaml --model gpt-4o --model claude-sonnet

    # Dry run (test extraction + chunking, no LLM calls):
    python run.py --input input/ --dry-run

The pipeline:
    1. Extract plain text from .txt, .doc/.docx, .pdf files
    2. Chunk each text by structural markers (chapters/headings) or sentence boundaries
    3. Send each chunk to one or more LLMs for SOC classification (per SKILL.md)
    4. Parse and validate structured responses
    5. Export combined results as CSV and/or JSON (with model_label column for comparison)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from scripts.analyze import analyze_chunks, analyze_chunks_multi
from scripts.config import Config
from scripts.export import export_results, print_summary
from scripts.extract import extract_all, extract_text
from scripts.models import ResultRow
from scripts.soc_chunker import chunk_text

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SOC Analysis Pipeline — detect stream of consciousness in literary texts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python run.py --input input/ --output results/\n"
            "  python run.py --input input/ --config models.yaml\n"
            "  python run.py --input input/ --config models.yaml --model gpt-4o --model claude-sonnet\n"
            "  python run.py --input input/ --dry-run\n"
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
        "--chunk-size",
        type=int,
        default=None,
        help="Override chunk size in characters (default: from config/env)",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=None,
        help="Override chunk overlap in characters (default: from config/env)",
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
        help="Extract and chunk only — skip LLM analysis (useful for testing chunking)",
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

    # Load config (multi-model from YAML, or single-model from .env)
    config = Config.load(env_file=args.env_file, config_file=args.config)
    if args.chunk_size:
        config.chunk_size = args.chunk_size
    if args.chunk_overlap:
        config.chunk_overlap = args.chunk_overlap

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
    logger.info(
        "Models to run: %s", ", ".join(p.label for p in profiles)
    )

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

    # ── Step 2: Chunk ──────────────────────────────────────────────────
    all_chunks = []
    for file_path, text in files:
        chunks = chunk_text(
            text,
            source_file=file_path.name,
            chunk_size=config.chunk_size,
            overlap=config.chunk_overlap,
        )
        logger.info("%s → %d chunks", file_path.name, len(chunks))
        all_chunks.extend(chunks)

    logger.info("Total chunks: %d", len(all_chunks))

    if args.dry_run:
        print(f"\n[DRY RUN] Extracted & chunked {len(files)} file(s) into {len(all_chunks)} chunks.")
        print(f"Models configured: {', '.join(p.label for p in profiles)}")
        for c in all_chunks:
            print(f"  {c.chunk_id:30s}  {c.chunk_label:30s}  {len(c.chunk_text):,} chars")
        return

    # ── Step 3: LLM Analysis (all models) ─────────────────────────────
    all_rows: list[ResultRow] = analyze_chunks_multi(all_chunks, config)

    # ── Step 4: Export ─────────────────────────────────────────────────
    formats = args.formats or ["csv", "json"]
    created = export_results(all_rows, args.output, formats=formats)

    for p in created:
        print(f"  → {p}")

    # Summary (with per-model breakdown)
    print_summary(all_rows)


if __name__ == "__main__":
    main()
