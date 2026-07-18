"""Remove silver-data records that came from incorrectly-downloaded source texts.

When a Project Gutenberg ID is wrong, chunk.py and silver.py process the wrong
text.  This script purges those records from silver_data/*.json and optionally
deletes the contaminated chunking_positives/ files.

Usage
-----
    # Dry run — show what would be removed without changing anything
    python scripts/clean_silver.py --dry-run

    # Remove bad records from silver_data/ and delete bad chunking files
    python scripts/clean_silver.py

    # Only clean silver_data/, leave chunking_positives/ alone
    python scripts/clean_silver.py --skip-chunking

    # Point at non-default directories
    python scripts/clean_silver.py --silver-dir silver_data/ --chunking-dir chunking_positives/
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Contaminated source files
#
# Each entry maps the wrong filename (as it appears in silver_data records'
# "source_file" field) to a note explaining what the file actually contained.
# ---------------------------------------------------------------------------

BAD_SOURCES: dict[str, str] = {
    # Wrong PG IDs — text downloaded was NOT the labelled work
    "pg00690_to_the_lighthouse.txt":    "PG 690 = Bertrand Russell (not Woolf)",
    "pg05765_mrs_dalloway.txt":         "PG 5765 = Darwin's Insectivorous Plants",
    "pg05550_jacobs_room.txt":          "PG 5550 = Georg Ebers novel",
    "pg43524_the_waves.txt":            "PG 43524 = Catholic World magazine",
    "pg04517_pointed_roofs.txt":        "PG 4517 = Ethan Frome (Edith Wharton)",
    "pg04518_backwater.txt":            "PG 4518 = Madame de Treymes (Edith Wharton)",
    "pg04519_honeycomb.txt":            "PG 4519 = Wharton short stories",
    # Mislabelled — the text is a valid positive but the filename/title is wrong.
    # PG 209 = The Turn of the Screw; this will be re-downloaded with correct name.
    "pg00209_the_wings_of_the_dove.txt": "PG 209 = The Turn of the Screw (mislabelled as Wings of the Dove)",
    # Defoe travel writing was downloaded instead of Turn of the Screw.
    "pg01149_the_turn_of_the_screw.txt": "PG 1149 = Defoe 'From London to Land's End' (NOT Turn of the Screw)",
}

# Corresponding files that may exist in chunking_positives/ and should be deleted.
# Keys are filenames; values are same notes as above.
BAD_CHUNKING_FILES: dict[str, str] = BAD_SOURCES  # same filenames


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_json_array(path: Path) -> list[dict]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            logger.warning("%s: not a JSON array — skipping", path)
            return []
        return data
    except Exception as exc:
        logger.warning("Could not load %s: %s — skipping", path, exc)
        return []


def _save_json_array(path: Path, records: list[dict]) -> None:
    path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")


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
        description="Purge silver-data records from incorrectly-downloaded source texts.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--silver-dir",
        type=Path,
        default=Path("silver_data"),
        help="Directory containing per-model silver JSON files (default: silver_data/)",
    )
    parser.add_argument(
        "--chunking-dir",
        type=Path,
        default=Path("chunking_positives"),
        help="Directory of chunked positives to clean (default: chunking_positives/)",
    )
    parser.add_argument(
        "--skip-chunking",
        action="store_true",
        help="Only clean silver_data/, do not touch chunking_positives/",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be changed without writing anything",
    )
    args = parser.parse_args()

    any_action = False

    # ── 1. Clean silver_data/*.json ──────────────────────────────────
    silver_files = sorted(args.silver_dir.glob("*.json"))
    if not silver_files:
        logger.warning("No JSON files found in %s", args.silver_dir)
    else:
        print(f"\n{'[DRY RUN] ' if args.dry_run else ''}Silver data cleanup ({args.silver_dir})")
        print(f"  Contaminated sources to remove:")
        for src, note in sorted(BAD_SOURCES.items()):
            print(f"    {src}  ({note})")
        print()

        for json_file in silver_files:
            records = _load_json_array(json_file)
            if not records:
                continue

            bad = [r for r in records if r.get("source_file") in BAD_SOURCES]
            clean = [r for r in records if r.get("source_file") not in BAD_SOURCES]

            if not bad:
                logger.info("  %s — no contaminated records", json_file.name)
                continue

            # Summarise by source
            by_src: dict[str, int] = {}
            for r in bad:
                s = r.get("source_file", "?")
                by_src[s] = by_src.get(s, 0) + 1

            action = "would remove" if args.dry_run else "removing"
            logger.info(
                "  %s: %d total → %s %d contaminated → %d remain",
                json_file.name, len(records), action, len(bad), len(clean),
            )
            for src, n in sorted(by_src.items()):
                logger.info("      %s: %d records", src, n)

            if not args.dry_run:
                _save_json_array(json_file, clean)
                logger.info("  ✓ saved %s", json_file)

            any_action = True

    # ── 2. Delete contaminated chunking_positives/ files ────────────
    if not args.skip_chunking:
        print(f"\n{'[DRY RUN] ' if args.dry_run else ''}Chunking directory cleanup ({args.chunking_dir})")

        for filename, note in sorted(BAD_CHUNKING_FILES.items()):
            path = args.chunking_dir / filename
            if not path.exists():
                logger.info("  %s — not found (already deleted or never created)", filename)
                continue

            action = "would delete" if args.dry_run else "deleting"
            logger.info("  %s %s  (%s)", action, filename, note)

            if not args.dry_run:
                path.unlink()
                logger.info("    ✓ deleted")

            any_action = True

    if args.dry_run and any_action:
        print("\nRe-run without --dry-run to apply changes.")
    elif not args.dry_run and any_action:
        print("\nCleanup complete.")
        print("Next steps:")
        print("  1. python scripts/download_positives.py --output-dir positives/ --delay 2.0")
        print("  2. python chunk.py --input positives/ --output chunking_positives/")
        print("  3. python scripts/annotate.py --input chunking_positives/ --resample-k 10")


if __name__ == "__main__":
    main()
