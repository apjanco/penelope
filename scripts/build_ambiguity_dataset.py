"""Build the ambiguity index from resampled silver passages.

For each canonical passage (as defined by scripts/resample.py), this script:
  1. Loads all resampled runs from silver_data/resampled/<model>.jsonl
  2. Computes within-model consistency: mode type + fraction of k runs matching.
  3. Classifies each (passage_key) as one of:
       clean-consensus   — all stable models agree on the same SoC type
       detection-boundary — stable models split between a SoC type and "none"
       typological-boundary — stable models split on exactly two SoC types
       noisy             — stable models disagree on 3+ distinct types
       single-model      — only one stable model (no cross-model signal)
  4. Assigns train/test split (--test-ratio, default 10%) stratified by boundary_type.
  5. Writes silver_data/ambiguity_index.jsonl (one record per productive-boundary passage).

Fallback mode (--single-pass):
  When no resampled data exists, loads single-pass silver_data/*.json instead.
  Each passage × model is treated as k=1 (consistency=1.0 by convention).
  This produces an upper-bound estimate of cross-model disagreements.

Usage
-----
    # Normal mode — from resampled data
    python scripts/build_ambiguity_dataset.py

    # Single-pass fallback (upper-bound estimate before resampling)
    python scripts/build_ambiguity_dataset.py --single-pass

    # Adjust consistency threshold
    python scripts/build_ambiguity_dataset.py --consistency-threshold 0.7

    # Custom output path
    python scripts/build_ambiguity_dataset.py --output silver_data/ambiguity_v2.jsonl

    # Print summary stats only, do not write output
    python scripts/build_ambiguity_dataset.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_CONSISTENCY_THRESHOLD = 0.8
DEFAULT_MAX_TYPE_SPLIT = 2
DEFAULT_TEST_RATIO = 0.10
DEFAULT_SEED = 42

# soc_type values that count as "not SoC" in detection-boundary logic
NONE_TYPES = {"none", "None", "", None}

# gpt-5.4 always runs at k=1; treat its single read as consistency=1.0
GPT54_LABEL = "gpt-5.4"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_resampled(resampled_dir: Path) -> dict[str, list[dict]]:
    """Load all resampled JSONL files.

    Returns a dict keyed by model_label.  Each value is the list of records
    (one per run) for that model.  Records with error=True or soc_type=None
    are excluded.
    """
    result: dict[str, list[dict]] = {}
    for f in sorted(resampled_dir.glob("*.jsonl")):
        records: list[dict] = []
        with f.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("error") or rec.get("soc_type") is None:
                    continue
                records.append(rec)
        model_label = f.stem
        result[model_label] = records
        logger.info("Loaded %d resampled records for %s", len(records), model_label)
    return result


def _load_single_pass(silver_dir: Path) -> dict[str, list[dict]]:
    """Load single-pass silver JSON files for fallback mode.

    Each record is wrapped to match the resampled schema (run_index=0,
    passage_key derived from chunk_id, temperature=None).
    """
    result: dict[str, list[dict]] = {}
    for f in sorted(silver_dir.glob("*.json")):
        if f.stem in ("results", "consensus"):
            continue
        try:
            raw = json.loads(f.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Could not load %s: %s", f, exc)
            continue
        if not isinstance(raw, list):
            continue

        # Build passage_key using the same clustering logic from resample.py
        # For single-pass data, we use chunk_id + an index within the chunk
        # (grouping by exact passage text for simplicity).
        passage_groups: dict[str, str] = {}  # normalised_text -> passage_key
        key_counter: Counter[str] = Counter()

        records: list[dict] = []
        model_label = f.stem

        for rec in raw:
            if not isinstance(rec, dict) or not rec.get("passage"):
                continue
            chunk_id = rec.get("chunk_id", "")
            norm = _normalise_text(rec["passage"])
            group_key = (chunk_id, " ".join(norm[:20]))  # prefix key
            if group_key not in passage_groups:
                idx = key_counter[chunk_id]
                passage_groups[group_key] = f"{chunk_id}__p{idx}"
                key_counter[chunk_id] += 1
            records.append({
                "passage_key": passage_groups[group_key],
                "chunk_id":    chunk_id,
                "run_index":   0,
                "model_label": model_label,
                "source_file": rec.get("source_file", ""),
                "passage":     rec["passage"],
                "soc_type":    rec.get("soc_type", "none"),
                "confidence":  rec.get("confidence", "medium"),
                "temperature": None,
            })

        result[model_label] = records
        logger.info("Loaded %d single-pass records for %s", len(records), model_label)

    return result


# ---------------------------------------------------------------------------
# Text normalisation (mirrors consensus.py / resample.py)
# ---------------------------------------------------------------------------

def _normalise_text(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return text.split()


def _token_overlap(a: str, b: str) -> float:
    ta = set(_normalise_text(a))
    tb = set(_normalise_text(b))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / min(len(ta), len(tb))


# ---------------------------------------------------------------------------
# Per-model consistency computation
# ---------------------------------------------------------------------------

def _compute_model_consistency(
    runs: list[dict],
    consistency_threshold: float,
    *,
    model_label: str = "",
    is_single_pass: bool = False,
) -> dict[str, dict[str, Any]]:
    """Compute within-model consistency for each passage_key.

    Returns a dict keyed by passage_key:
        {
            "mode_type":       str,   # most common soc_type
            "consistency":     float, # fraction of runs matching mode
            "n_runs":          int,
            "stable":          bool,  # consistency >= threshold
            "source_file":     str,
            "passage":         str,
        }

    For gpt-5.4 (k=1) and single-pass data, consistency is set to 1.0
    since we cannot measure within-model variance.
    """
    force_stable = (model_label == GPT54_LABEL) or is_single_pass

    by_key: dict[str, list[str]] = defaultdict(list)
    meta: dict[str, dict] = {}

    for rec in runs:
        pk = rec["passage_key"]
        soc = rec.get("soc_type") or "none"
        by_key[pk].append(soc)
        if pk not in meta:
            meta[pk] = {
                "source_file": rec.get("source_file", ""),
                "passage":     rec.get("passage", ""),
                "chunk_id":    rec.get("chunk_id", ""),
            }

    result: dict[str, dict[str, Any]] = {}
    for pk, type_list in by_key.items():
        counter = Counter(type_list)
        mode_type, mode_count = counter.most_common(1)[0]
        n_runs = len(type_list)
        consistency = 1.0 if force_stable else mode_count / n_runs

        result[pk] = {
            "mode_type":   mode_type,
            "consistency": consistency,
            "n_runs":      n_runs,
            "stable":      force_stable or consistency >= consistency_threshold,
            **meta[pk],
        }

    return result


# ---------------------------------------------------------------------------
# Cross-model boundary classification
# ---------------------------------------------------------------------------

def _classify_passage(
    passage_key: str,
    per_model_consistency: dict[str, dict[str, Any]],
    max_type_split: int,
) -> dict[str, Any] | None:
    """Classify one passage across all models.

    Returns an ambiguity_index record, or None if the passage does not
    qualify (clean-consensus, single-model, or noisy).
    """
    stable_models: list[str] = []
    unstable_models: list[str] = []
    within_model_consistency: dict[str, float] = {}
    type_split: dict[str, list[str]] = defaultdict(list)
    source_file = ""
    passage = ""
    chunk_id = ""

    for model_label, model_data in per_model_consistency.items():
        if passage_key not in model_data:
            continue
        rec = model_data[passage_key]
        within_model_consistency[model_label] = rec["consistency"]
        source_file = rec["source_file"]
        passage = rec["passage"]
        chunk_id = rec["chunk_id"]

        if rec["stable"]:
            stable_models.append(model_label)
            type_split[rec["mode_type"]].append(model_label)
        else:
            unstable_models.append(model_label)

    if len(stable_models) < 2:
        return None  # not enough information

    distinct_types = list(type_split.keys())
    n_distinct = len(distinct_types)

    if n_distinct == 1:
        return None  # clean-consensus — not a boundary case

    if n_distinct > max_type_split:
        # Noisy — more than two distinct types among stable models
        return {
            "passage_key":               passage_key,
            "chunk_id":                  chunk_id,
            "source_file":               source_file,
            "passage":                   passage,
            "boundary_type":             "noisy",
            "stable_models":             stable_models,
            "unstable_models":           unstable_models,
            "type_split":                dict(type_split),
            "within_model_consistency":  within_model_consistency,
            "noisy":                     True,
            "split":                     None,
        }

    # Exactly two distinct types among stable models
    types_pair = tuple(sorted(distinct_types))
    has_none = any(t in NONE_TYPES for t in types_pair)

    if has_none:
        boundary_type = "detection"
    else:
        boundary_type = "typological"

    return {
        "passage_key":               passage_key,
        "chunk_id":                  chunk_id,
        "source_file":               source_file,
        "passage":                   passage,
        "boundary_type":             boundary_type,
        "stable_models":             stable_models,
        "unstable_models":           unstable_models,
        "type_split":                dict(type_split),
        "within_model_consistency":  within_model_consistency,
        "noisy":                     False,
        "split":                     None,  # assigned later
    }


# ---------------------------------------------------------------------------
# Train/test split assignment
# ---------------------------------------------------------------------------

def _assign_splits(
    records: list[dict],
    test_ratio: float,
    seed: int,
) -> list[dict]:
    """Assign train/test split stratified by boundary_type.

    Noisy records are assigned split=None and excluded from training.
    """
    rng = random.Random(seed)

    # Group by boundary_type (excluding noisy)
    productive = [r for r in records if not r["noisy"]]
    noisy = [r for r in records if r["noisy"]]

    by_type: dict[str, list[dict]] = defaultdict(list)
    for r in productive:
        by_type[r["boundary_type"]].append(r)

    for bt, group in by_type.items():
        rng.shuffle(group)
        n_test = max(1, int(len(group) * test_ratio))
        for i, rec in enumerate(group):
            rec["split"] = "test" if i < n_test else "train"

    for rec in noisy:
        rec["split"] = None

    return records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_ambiguity_index(
    resampled_dir: Path,
    silver_dir: Path,
    output_path: Path,
    *,
    consistency_threshold: float = DEFAULT_CONSISTENCY_THRESHOLD,
    max_type_split: int = DEFAULT_MAX_TYPE_SPLIT,
    test_ratio: float = DEFAULT_TEST_RATIO,
    seed: int = DEFAULT_SEED,
    single_pass: bool = False,
    dry_run: bool = False,
) -> list[dict]:
    """Build and optionally write the ambiguity index. Returns all records."""

    # ── Load data ────────────────────────────────────────────────────
    if single_pass or not any(resampled_dir.glob("*.jsonl")):
        if not single_pass:
            logger.warning(
                "No resampled JSONL files found in %s — falling back to single-pass data.",
                resampled_dir,
            )
        logger.info("Single-pass fallback mode: consistency forced to 1.0 per model.")
        raw_by_model = _load_single_pass(silver_dir)
        is_single_pass = True
    else:
        raw_by_model = _load_resampled(resampled_dir)
        is_single_pass = False

    if not raw_by_model:
        logger.error("No data loaded. Check --resampled-dir / --silver-dir.")
        sys.exit(1)

    # ── Compute per-model consistency ────────────────────────────────
    per_model_consistency: dict[str, dict[str, Any]] = {}
    all_passage_keys: set[str] = set()

    for model_label, runs in raw_by_model.items():
        consistency = _compute_model_consistency(
            runs,
            consistency_threshold,
            model_label=model_label,
            is_single_pass=is_single_pass,
        )
        per_model_consistency[model_label] = consistency
        all_passage_keys.update(consistency.keys())

    logger.info(
        "Total unique passage keys across all models: %d", len(all_passage_keys)
    )

    # ── Classify each passage ────────────────────────────────────────
    records: list[dict] = []
    stats: Counter[str] = Counter()

    for passage_key in sorted(all_passage_keys):
        classified = _classify_passage(
            passage_key,
            per_model_consistency,
            max_type_split,
        )
        if classified is None:
            # clean-consensus or single-model
            stats["clean_or_single"] += 1
            continue
        records.append(classified)
        stats[classified["boundary_type"]] += 1

    logger.info("Classification summary:")
    logger.info("  clean-consensus / single-model: %d", stats["clean_or_single"])
    logger.info("  typological boundary:           %d", stats["typological"])
    logger.info("  detection boundary:             %d", stats["detection"])
    logger.info("  noisy (3+ types):               %d", stats["noisy"])

    # ── Assign train/test splits ─────────────────────────────────────
    records = _assign_splits(records, test_ratio, seed)

    productive = [r for r in records if not r["noisy"]]
    n_train = sum(1 for r in productive if r["split"] == "train")
    n_test = sum(1 for r in productive if r["split"] == "test")
    logger.info("Productive boundary cases: %d train / %d test", n_train, n_test)

    # ── Write output ─────────────────────────────────────────────────
    if not dry_run:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as fh:
            for rec in records:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        logger.info("Written %d records to %s", len(records), output_path)
    else:
        logger.info("[DRY RUN] would write %d records to %s", len(records), output_path)

    return records


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Build ambiguity_index.jsonl from resampled silver data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--resampled-dir",
        type=Path,
        default=Path("silver_data/resampled"),
        help="Directory of resampled JSONL files (default: silver_data/resampled/)",
    )
    p.add_argument(
        "--silver-dir",
        type=Path,
        default=Path("silver_data"),
        help="Directory of single-pass silver JSON files (default: silver_data/)",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("silver_data/ambiguity_index.jsonl"),
        help="Output JSONL path (default: silver_data/ambiguity_index.jsonl)",
    )
    p.add_argument(
        "--consistency-threshold",
        type=float,
        default=DEFAULT_CONSISTENCY_THRESHOLD,
        help=f"Min fraction of runs matching mode type for a model to be 'stable' "
             f"(default: {DEFAULT_CONSISTENCY_THRESHOLD})",
    )
    p.add_argument(
        "--max-type-split",
        type=int,
        default=DEFAULT_MAX_TYPE_SPLIT,
        help=f"Max distinct types among stable models before a passage is 'noisy' "
             f"(default: {DEFAULT_MAX_TYPE_SPLIT})",
    )
    p.add_argument(
        "--test-ratio",
        type=float,
        default=DEFAULT_TEST_RATIO,
        help=f"Fraction of productive boundary cases held out for test "
             f"(default: {DEFAULT_TEST_RATIO})",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed for train/test assignment (default: {DEFAULT_SEED})",
    )
    p.add_argument(
        "--single-pass",
        action="store_true",
        help="Use single-pass silver data instead of resampled (upper-bound estimate)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print stats without writing output file",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging",
    )
    return p


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    build_ambiguity_index(
        resampled_dir=args.resampled_dir,
        silver_dir=args.silver_dir,
        output_path=args.output,
        consistency_threshold=args.consistency_threshold,
        max_type_split=args.max_type_split,
        test_ratio=args.test_ratio,
        seed=args.seed,
        single_pass=args.single_pass,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
