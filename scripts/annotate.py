"""Unified annotation pipeline: silver-label extraction + consistency resampling.

This script replaces running silver.py and resample.py as separate manual steps.
For every new batch of chunked texts, run this script once and get both the
per-chunk silver labels (V1 training signal) and the within-model consistency
data needed for V3 ambiguity detection.

Pipeline
--------
  Phase 1 — Silver extraction (silver.py logic)
    For each unannotated chunk in --input, ask each model to identify and
    classify SoC passages.  Writes to --silver-dir/<model>.json.  Resumable.

  Phase 2 — Consistency resampling (resample.py logic)
    For canonical passages in the new source files, run each model k more
    times at temperature 0.8 to measure within-model consistency.
    Writes to --resample-dir/<model>.jsonl.  Resumable.

Usage
-----
    # Annotate all files in chunking_positives/ (k=10 resampling)
    python scripts/annotate.py --input chunking_positives/

    # Single new file, restrict to specific models
    python scripts/annotate.py --input chunking_positives/pg00209_the_turn_of_the_screw.txt \\
        --model mercury-2 --model qwen3-max

    # Silver extraction only (no resampling) — useful when iterating quickly
    python scripts/annotate.py --input chunking_positives/ --skip-resample

    # Dry run — shows call counts without hitting any API
    python scripts/annotate.py --input chunking_positives/ --dry-run

    # Re-annotate already-done chunks (e.g. after source text correction)
    python scripts/annotate.py --input chunking_positives/ --force

After annotation, build datasets:
    # V1 training data
    python scripts/consensus.py --silver-dir silver_data/ --output dataset/
    python scripts/build_dataset.py --silver-dir silver_data/ --neg-dir negatives/ --output dataset/

    # V3 ambiguity index
    python scripts/build_ambiguity_dataset.py --output silver_data/ambiguity_index.jsonl
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.config import Config
from scripts.soc_chunker import is_chunked_file, parse_chunked_dir, parse_chunked_file
import scripts.silver as _silver
import scripts.resample as _resample


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_silver(
    chunks,
    profiles,
    silver_dir: Path,
    *,
    delay: float,
    force: bool,
    dry_run: bool,
) -> None:
    """Phase 1: silver-label extraction for each model."""
    silver_dir.mkdir(parents=True, exist_ok=True)

    for profile in profiles:
        output_file = silver_dir / f"{profile.label}.json"
        existing_records, seen_chunk_ids = _silver._load_existing(output_file)

        pending = [c for c in chunks if force or c.chunk_id not in seen_chunk_ids]
        skipped = len(chunks) - len(pending)

        logger.info(
            "[%s] %d chunks to annotate, %d already done",
            profile.label, len(pending), skipped,
        )

        if dry_run:
            continue

        if not pending:
            logger.info("[%s] nothing to do", profile.label)
            continue

        profile.validate()
        new_records: list[dict] = []

        for i, chunk in enumerate(pending, 1):
            logger.info(
                "[%s] %d/%d  %s", profile.label, i, len(pending), chunk.chunk_id,
            )
            records = _silver._call_api(profile, chunk)
            if records:
                logger.info(
                    "  → %d instance(s): %s",
                    len(records),
                    ", ".join(r["soc_type"] for r in records),
                )
            else:
                logger.info("  → no SoC instances found")

            new_records.extend(records)
            _silver._save(output_file, existing_records + new_records)

            if i < len(pending):
                time.sleep(delay)

        logger.info(
            "[%s] silver complete — %d new, %d total",
            profile.label,
            len(new_records),
            len(existing_records) + len(new_records),
        )


def _run_resample(
    source_files: set[str],
    profiles,
    silver_dir: Path,
    resample_dir: Path,
    *,
    k: int,
    temperature: float,
    dry_run: bool,
) -> None:
    """Phase 2: consistency resampling for passages from the given source files."""
    canonical = _resample._build_canonical_passages(silver_dir)

    # Only resample passages from the files we just annotated
    filtered = [p for p in canonical if p["source_file"] in source_files]

    if not filtered:
        logger.warning(
            "No canonical passages found for source files: %s — "
            "silver extraction may have produced no instances, or source_file "
            "values don't match. Skipping resampling.",
            ", ".join(sorted(source_files)),
        )
        return

    logger.info(
        "Resampling %d canonical passages from %d source file(s)",
        len(filtered), len(source_files),
    )

    _resample.resample(
        filtered,
        profiles,
        resample_dir,
        k=k,
        temperature=temperature,
        source_file_filter=None,  # pre-filtered above
        dry_run=dry_run,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Silver-label extraction + consistency resampling in one step.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/annotate.py --input chunking_positives/\n"
            "  python scripts/annotate.py --input chunking_positives/new_file.txt\n"
            "  python scripts/annotate.py --input chunking_positives/ --skip-resample\n"
            "  python scripts/annotate.py --input chunking_positives/ --dry-run\n"
        ),
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        type=Path,
        help="Chunked file or directory (output of chunk.py)",
    )
    parser.add_argument(
        "--silver-dir",
        type=Path,
        default=Path("silver_data"),
        help="Directory for per-model silver JSON files (default: silver_data/)",
    )
    parser.add_argument(
        "--resample-dir",
        type=Path,
        default=Path("silver_data/resampled"),
        help="Output directory for resampled JSONL (default: silver_data/resampled/)",
    )
    parser.add_argument(
        "--model", "-m",
        action="append",
        dest="models",
        metavar="LABEL",
        help="Restrict to this model label (repeatable; default: all in models.yaml)",
    )
    parser.add_argument(
        "--resample-k",
        type=int,
        default=10,
        help="Resampling runs per passage for variable-temperature models (default: 10). "
             "gpt-5.4 always uses k=1.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature for resampling (default: 0.8)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Seconds between silver-extraction API calls (default: 1.0)",
    )
    parser.add_argument(
        "--config", "-c",
        type=Path,
        default=None,
        help="Path to models.yaml (default: auto-detect)",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=None,
        help="Path to .env file for API keys (default: auto-detect)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-annotate chunks already present in silver_data/",
    )
    parser.add_argument(
        "--skip-resample",
        action="store_true",
        help="Run silver extraction only; skip consistency resampling",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print call counts without making API requests",
    )
    args = parser.parse_args()

    # ── Config + model profiles ──────────────────────────────────────
    config = Config.load(env_file=args.env_file, config_file=args.config)
    profiles = config.get_model_profiles()

    if args.models:
        requested = set(args.models)
        profiles = [p for p in profiles if p.label in requested]
        unknown = requested - {p.label for p in profiles}
        if unknown:
            logger.error(
                "Unknown model label(s): %s. Available: %s",
                ", ".join(sorted(unknown)),
                ", ".join(p.label for p in config.get_model_profiles()),
            )
            sys.exit(1)

    if not profiles:
        logger.error("No model profiles found. Check models.yaml and .env.")
        sys.exit(1)

    logger.info("Models: %s", ", ".join(p.label for p in profiles))

    # ── Load chunks ──────────────────────────────────────────────────
    input_path = args.input
    if input_path.is_file():
        if not is_chunked_file(input_path):
            logger.error(
                "%s has no <chunk-N> markup. Run chunk.py first.", input_path,
            )
            sys.exit(1)
        chunks = parse_chunked_file(input_path)
    elif input_path.is_dir():
        chunks = parse_chunked_dir(input_path)
    else:
        logger.error("Input path not found: %s", input_path)
        sys.exit(1)

    if not chunks:
        logger.error("No chunks found in %s", input_path)
        sys.exit(1)

    # Collect the source files covered by this input (used to scope resampling)
    source_files: set[str] = {c.source_file for c in chunks}
    logger.info(
        "%d chunks from %d source file(s): %s",
        len(chunks),
        len(source_files),
        ", ".join(sorted(source_files)),
    )

    if args.dry_run:
        print(f"\n[DRY RUN] Phase 1 — Silver extraction")
        for profile in profiles:
            _, seen = _silver._load_existing(args.silver_dir / f"{profile.label}.json")
            pending = [c for c in chunks if args.force or c.chunk_id not in seen]
            print(
                f"  {profile.label}: {len(pending)} chunks to annotate "
                f"({len(chunks) - len(pending)} already done)"
            )

        if not args.skip_resample:
            print(f"\n[DRY RUN] Phase 2 — Consistency resampling")
            print(f"  k={args.resample_k} (variable-temp)  temperature={args.temperature}")
            canonical = _resample._build_canonical_passages(args.silver_dir)
            filtered = [p for p in canonical if p["source_file"] in source_files]
            for profile in profiles:
                model_k = _resample.GPT54_K if profile.label in _resample.NO_CUSTOM_TEMP_MODELS else args.resample_k
                print(
                    f"  {profile.label}: {len(filtered)} passages × k={model_k} "
                    f"= {len(filtered) * model_k} calls"
                )

        return

    # ── Phase 1: Silver extraction ───────────────────────────────────
    logger.info("=== Phase 1: Silver extraction ===")
    _run_silver(
        chunks, profiles, args.silver_dir,
        delay=args.delay,
        force=args.force,
        dry_run=False,
    )

    # ── Phase 2: Resampling ──────────────────────────────────────────
    if not args.skip_resample:
        logger.info("=== Phase 2: Consistency resampling ===")
        _run_resample(
            source_files, profiles,
            args.silver_dir, args.resample_dir,
            k=args.resample_k,
            temperature=args.temperature,
            dry_run=False,
        )

    logger.info("=== Annotation pipeline complete ===")


if __name__ == "__main__":
    main()
