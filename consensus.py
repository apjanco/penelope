#!/usr/bin/env python3
"""Penelope — Step 3: Build consensus datasets from multi-model results.

Reads per-model JSON files from results/, groups matching passages across
models, and filters/merges them according to tracks defined in consensus.yaml.

Usage:
    # Run the default track (moderate)
    python consensus.py

    # Run a specific track
    python consensus.py --track conservative

    # Run multiple tracks at once
    python consensus.py --track conservative --track liberal

    # List available tracks without running
    python consensus.py --list

    # Custom paths
    python consensus.py --input results/ --config consensus.yaml --output consensus/

Three-step workflow:
    1. python chunk.py --input input/           → chunking/
    2. python run.py --input chunking/          → results/
    3. python consensus.py --track conservative → results/consensus_conservative.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

from scripts.consensus import run_consensus, load_consensus_config, parse_track

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Penelope — build consensus datasets from multi-model SOC results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python consensus.py                             # default track\n"
            "  python consensus.py --track conservative\n"
            "  python consensus.py --track conservative --track liberal\n"
            "  python consensus.py --list                      # show available tracks\n"
        ),
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        default=Path("results"),
        help="Directory with per-model JSON results (default: results/)",
    )
    parser.add_argument(
        "--config", "-c",
        type=Path,
        default=Path("consensus.yaml"),
        help="Path to consensus YAML config (default: consensus.yaml)",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Output directory for consensus files (default: same as --input)",
    )
    parser.add_argument(
        "--track", "-t",
        action="append",
        dest="tracks",
        help="Track name(s) to run. Repeat for multiple. Omit for default.",
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        dest="list_tracks",
        help="List available tracks and exit.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose (DEBUG) logging.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)-8s %(message)s",
    )

    # Validate config exists
    if not args.config.exists():
        logger.error("Config file not found: %s", args.config)
        sys.exit(1)

    cfg = load_consensus_config(args.config)

    # --list mode
    if args.list_tracks:
        default = cfg.get("default_track", "?")
        tracks_raw = cfg.get("tracks", {})
        print(f"\nAvailable tracks in {args.config}:\n")
        for name, raw in tracks_raw.items():
            track = parse_track(name, raw)
            marker = " (default)" if name == default else ""
            print(f"  {name}{marker}")
            print(f"    agreement:      {track.agreement}")
            print(f"    min_models:     {track.min_models}")
            print(f"    min_confidence: {track.min_confidence}")
            print(f"    resolve_type:   {track.resolve_type}")
            if track.models:
                print(f"    models:         {', '.join(track.models)}")
            print()
        return

    # Validate input directory
    if not args.input.is_dir():
        logger.error("Results directory not found: %s", args.input)
        sys.exit(1)

    print(f"\n🧶 Penelope — Consensus Builder\n")

    results = run_consensus(
        results_dir=args.input,
        config_path=args.config,
        track_names=args.tracks,
        output_dir=args.output,
    )

    if not results:
        print("\n⚠  No output produced.")
        sys.exit(1)

    # Summary
    print(f"\n{'─' * 50}")
    for name, df in results.items():
        n_passages = len(df)
        n_full = (df["agreement"] == "full").sum() if "agreement" in df.columns else "?"
        n_partial = (df["agreement"] == "partial").sum() if "agreement" in df.columns else "?"
        print(f"Track '{name}': {n_passages} passages ({n_full} full agreement, {n_partial} partial)")
    print()


if __name__ == "__main__":
    main()
