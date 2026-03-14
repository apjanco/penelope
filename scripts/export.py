"""Export SOC analysis results to CSV and/or JSON."""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

from scripts.models import ResultRow

logger = logging.getLogger(__name__)

# The column order for CSV output
CSV_COLUMNS = [
    "model_label",
    "source_file",
    "chunk_id",
    "chunk_label",
    "chunk_index",
    "passage",
    "soc_type",
    "secondary_devices",
    "affective_register",
    "narrator_position",
    "character_pov",
    "explanation",
    "evidence",
    "confidence",
    "notes",
]


def export_csv(rows: list[ResultRow], output_path: Path) -> None:
    """Write results to a CSV file.

    Args:
        rows: List of ResultRow objects to export.
        output_path: Destination .csv file path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.model_dump())

    logger.info("Exported %d rows to %s", len(rows), output_path)


def export_json(rows: list[ResultRow], output_path: Path) -> None:
    """Write results to a JSON file.

    Args:
        rows: List of ResultRow objects to export.
        output_path: Destination .json file path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = [row.model_dump() for row in rows]
    output_path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    logger.info("Exported %d rows to %s", len(rows), output_path)


def export_results(
    rows: list[ResultRow],
    output_dir: Path,
    formats: list[str] | None = None,
) -> list[Path]:
    """Export results in one or more formats, one file per model.

    Creates separate CSV/JSON files for each model label, e.g.:
        results/gpt-5.csv, results/gpt-5.json
        results/gemini-3-pro-preview.csv, ...

    Args:
        rows: Analysis results to export.
        output_dir: Directory for output files.
        formats: List of format strings: 'csv', 'json', or both.
                 Defaults to both.

    Returns:
        List of paths to the created files.
    """
    if formats is None:
        formats = ["csv", "json"]

    output_dir.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []

    # Group rows by model label
    by_model: dict[str, list[ResultRow]] = {}
    for row in rows:
        label = row.model_label or "default"
        by_model.setdefault(label, []).append(row)

    for label, model_rows in sorted(by_model.items()):
        # Sanitise label for use as a filename
        safe_label = _safe_filename(label)

        if "csv" in formats:
            csv_path = output_dir / f"{safe_label}.csv"
            export_csv(model_rows, csv_path)
            created.append(csv_path)

        if "json" in formats:
            json_path = output_dir / f"{safe_label}.json"
            export_json(model_rows, json_path)
            created.append(json_path)

    return created


def _safe_filename(label: str) -> str:
    """Turn a model label into a filesystem-safe filename stem."""
    import re
    # Replace anything that isn't alphanumeric, hyphen, or underscore
    safe = re.sub(r"[^\w\-]", "_", label).strip("_")
    return safe or "default"


def print_summary(rows: list[ResultRow]) -> None:
    """Print a summary of the analysis results to stdout."""
    if not rows:
        print("\nNo SOC instances found.")
        return

    # Collect unique models
    model_labels = sorted(set(r.model_label for r in rows))
    multi_model = len(model_labels) > 1

    print(f"\n{'='*60}")
    print(f"SOC Analysis Summary: {len(rows)} instance(s) found")
    if multi_model:
        print(f"Models compared: {', '.join(model_labels)}")
    print(f"{'='*60}")

    # Per-model breakdown
    for label in model_labels:
        model_rows = [r for r in rows if r.model_label == label] if multi_model else rows
        header = f"\n── {label} " + "─" * (55 - len(label)) if multi_model else ""
        if header:
            print(header)
        print(f"  Instances: {len(model_rows)}")

        type_counts: dict[str, int] = {}
        confidence_counts: dict[str, int] = {}
        for row in model_rows:
            type_counts[row.soc_type] = type_counts.get(row.soc_type, 0) + 1
            confidence_counts[row.confidence] = confidence_counts.get(row.confidence, 0) + 1

        print(f"\n  By SOC type:")
        for stype, count in sorted(type_counts.items(), key=lambda x: -x[1]):
            print(f"    {stype:38s} {count:4d}")

        print(f"\n  By confidence:")
        for conf, count in sorted(confidence_counts.items()):
            print(f"    {conf:38s} {count:4d}")

        if not multi_model:
            break  # only one pass needed

    # Cross-model comparison summary
    if multi_model:
        print(f"\n── Cross-model comparison " + "─" * 34)
        print(f"  {'Model':<25s} {'Instances':>10s} {'High conf':>10s} {'Med conf':>10s} {'Low conf':>10s}")
        for label in model_labels:
            model_rows = [r for r in rows if r.model_label == label]
            high = sum(1 for r in model_rows if r.confidence == "high")
            med = sum(1 for r in model_rows if r.confidence == "medium")
            low = sum(1 for r in model_rows if r.confidence == "low")
            print(f"  {label:<25s} {len(model_rows):>10d} {high:>10d} {med:>10d} {low:>10d}")

    # File breakdown if multiple files
    file_counts: dict[str, int] = {}
    for row in rows:
        file_counts[row.source_file] = file_counts.get(row.source_file, 0) + 1
    if len(file_counts) > 1:
        print(f"\nBy source file:")
        for fname, count in sorted(file_counts.items()):
            print(f"  {fname:40s} {count:4d}")

    print()
