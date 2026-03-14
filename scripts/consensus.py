"""Consensus filtering — combine multi-model SOC results with configurable strictness.

This module loads per-model JSON results, groups passages that refer to the
same text span, and filters/merges them according to a "track" defined in
consensus.yaml.
"""

from __future__ import annotations

import csv
import json
import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from io import StringIO
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

CONFIDENCE_RANK = {"high": 3, "medium": 2, "low": 1}
MIN_OVERLAP_RATIO = 0.35  # same as app.py


@dataclass
class TrackConfig:
    """One consensus track parsed from consensus.yaml."""

    name: str
    models: list[str] = field(default_factory=list)
    agreement: str = "partial"  # full | partial | any
    min_models: int = 2
    min_confidence: str = "medium"  # high | medium | low
    resolve_type: str = "majority"  # majority | longest | all
    output_formats: list[str] = field(default_factory=lambda: ["csv", "json"])


def load_consensus_config(path: Path) -> dict[str, Any]:
    """Load and return the raw consensus.yaml dict."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return raw


def parse_track(name: str, raw: dict[str, Any]) -> TrackConfig:
    """Parse a single track dict into a TrackConfig."""
    return TrackConfig(
        name=name,
        models=raw.get("models") or [],
        agreement=raw.get("agreement", "partial"),
        min_models=raw.get("min_models", 2),
        min_confidence=raw.get("min_confidence", "medium"),
        resolve_type=raw.get("resolve_type", "majority"),
        output_formats=raw.get("output_formats", ["csv", "json"]),
    )


# ---------------------------------------------------------------------------
# Data loading  (mirrors app.py logic, no Streamlit dependency)
# ---------------------------------------------------------------------------


def load_results(results_dir: Path) -> pd.DataFrame:
    """Load all per-model JSON result files into one DataFrame."""
    frames: list[pd.DataFrame] = []
    for f in sorted(results_dir.glob("*.json")):
        if f.stem in ("results", "consensus"):
            continue
        data = json.loads(f.read_text(encoding="utf-8"))
        if data:
            frames.append(pd.DataFrame(data))
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    combined["passage_norm"] = combined["passage"].apply(_normalise_text)
    combined["passage_tokens"] = combined["passage_norm"].apply(lambda t: set(t.split()))
    return combined


def _normalise_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _token_overlap(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    intersection = len(a & b)
    smaller = min(len(a), len(b))
    return intersection / smaller if smaller else 0.0


# ---------------------------------------------------------------------------
# Passage grouping  (same greedy algorithm as app.py)
# ---------------------------------------------------------------------------


def build_passage_groups(df: pd.DataFrame) -> list[dict]:
    """Cluster passages across models by token overlap within the same chunk."""
    groups: list[dict] = []
    assigned: set[int] = set()

    for chunk_id, chunk_df in df.groupby("chunk_id"):
        for i in chunk_df.index:
            if i in assigned:
                continue
            tokens_i = df.at[i, "passage_tokens"]
            matched_group = None
            for g in groups:
                if g["chunk_id"] != chunk_id:
                    continue
                for member_idx in g["rows"]:
                    if _token_overlap(tokens_i, df.at[member_idx, "passage_tokens"]) >= MIN_OVERLAP_RATIO:
                        matched_group = g
                        break
                if matched_group:
                    break

            if matched_group:
                matched_group["rows"].append(i)
                model = df.at[i, "model_label"]
                if model not in matched_group["models"]:
                    matched_group["models"].append(model)
            else:
                groups.append({
                    "group_id": len(groups),
                    "rows": [i],
                    "models": [df.at[i, "model_label"]],
                    "chunk_id": chunk_id,
                    "source_file": df.at[i, "source_file"],
                })
            assigned.add(i)

    # Enrich
    for g in groups:
        g["n_models"] = len(set(g["models"]))
        passages = [df.at[idx, "passage"] for idx in g["rows"]]
        g["representative"] = max(passages, key=len)
        types_in_group = set(df.at[idx, "soc_type"] for idx in g["rows"])
        if g["n_models"] == 1:
            g["agreement"] = "single"
        elif len(types_in_group) == 1:
            g["agreement"] = "full"
        else:
            g["agreement"] = "partial"

    return groups


# ---------------------------------------------------------------------------
# Filtering & merging
# ---------------------------------------------------------------------------

# Columns for output (omit internal/working columns)
OUTPUT_COLUMNS = [
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
    "n_models",
    "models_agreed",
    "agreement",
]


def apply_track(
    df: pd.DataFrame,
    groups: list[dict],
    track: TrackConfig,
) -> pd.DataFrame:
    """Filter and merge passage groups according to a track's rules.

    Returns a DataFrame with one row per retained passage.
    """
    min_conf_rank = CONFIDENCE_RANK.get(track.min_confidence, 1)
    out_rows: list[dict] = []

    for g in groups:
        # ── 1. Filter by model list ──────────────────────────────────
        if track.models:
            row_idxs = [
                idx for idx in g["rows"]
                if df.at[idx, "model_label"] in track.models
            ]
        else:
            row_idxs = list(g["rows"])
        if not row_idxs:
            continue

        # ── 2. Filter by confidence ──────────────────────────────────
        row_idxs = [
            idx for idx in row_idxs
            if CONFIDENCE_RANK.get(df.at[idx, "confidence"], 0) >= min_conf_rank
        ]
        if not row_idxs:
            continue

        # ── 3. Compute group-level stats on the surviving rows ───────
        models_present = sorted(set(df.at[idx, "model_label"] for idx in row_idxs))
        n_models = len(models_present)

        if n_models < track.min_models:
            continue

        types_present = set(df.at[idx, "soc_type"] for idx in row_idxs)
        agreement = "full" if len(types_present) == 1 else "partial"

        # ── 4. Filter by agreement requirement ───────────────────────
        if track.agreement == "full" and agreement != "full":
            continue
        # "partial" and "any" both accept partial & full

        # ── 5. Resolve / merge ───────────────────────────────────────
        if track.resolve_type == "all":
            # Keep every qualifying row individually
            for idx in row_idxs:
                row_dict = df.iloc[idx].to_dict()
                row_dict["n_models"] = n_models
                row_dict["models_agreed"] = ", ".join(models_present)
                row_dict["agreement"] = agreement
                out_rows.append(row_dict)
        else:
            # Produce a single merged row for the group
            merged = _resolve_group(df, row_idxs, track.resolve_type)
            merged["n_models"] = n_models
            merged["models_agreed"] = ", ".join(models_present)
            merged["agreement"] = agreement
            out_rows.append(merged)

    if not out_rows:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    result = pd.DataFrame(out_rows)
    # Keep only output columns that exist
    cols = [c for c in OUTPUT_COLUMNS if c in result.columns]
    result = result[cols].copy()
    result.sort_values(
        ["source_file", "chunk_index", "chunk_id"],
        inplace=True,
        ignore_index=True,
    )
    return result


def _resolve_group(
    df: pd.DataFrame,
    row_idxs: list[int],
    strategy: str,
) -> dict:
    """Merge multiple model rows for one passage group into a single row.

    strategy:
        majority — pick the SOC type that appears most often; on tie, pick
                   the one with highest average confidence.
        longest  — use the row whose passage text is longest.
    """
    rows = df.iloc[row_idxs]

    if strategy == "longest":
        best_idx = rows["passage"].str.len().idxmax()
        return rows.loc[best_idx].to_dict()

    # majority vote
    type_counts = Counter(rows["soc_type"])
    max_count = max(type_counts.values())
    candidates = [t for t, c in type_counts.items() if c == max_count]

    if len(candidates) == 1:
        winning_type = candidates[0]
    else:
        # Tie-break: pick type whose rows have highest avg confidence
        def _avg_conf(soc_type: str) -> float:
            subset = rows[rows["soc_type"] == soc_type]
            return subset["confidence"].map(CONFIDENCE_RANK).mean()

        winning_type = max(candidates, key=_avg_conf)

    # Pick the single best row for that type (highest confidence, longest passage)
    type_rows = rows[rows["soc_type"] == winning_type]
    type_rows = type_rows.copy()
    type_rows["_conf_rank"] = type_rows["confidence"].map(CONFIDENCE_RANK)
    type_rows["_pass_len"] = type_rows["passage"].str.len()
    best = type_rows.sort_values(
        ["_conf_rank", "_pass_len"], ascending=[False, False]
    ).iloc[0]
    result = best.to_dict()
    result.pop("_conf_rank", None)
    result.pop("_pass_len", None)
    return result


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def export_consensus(
    result: pd.DataFrame,
    output_dir: Path,
    track_name: str,
    formats: list[str],
) -> list[Path]:
    """Write the consensus DataFrame to CSV and/or JSON.

    Files are named  consensus_<track>.csv / .json
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []
    safe_name = re.sub(r"[^\w\-]", "_", track_name).strip("_") or "default"

    if "csv" in formats:
        csv_path = output_dir / f"consensus_{safe_name}.csv"
        result.to_csv(csv_path, index=False, quoting=csv.QUOTE_ALL)
        logger.info("Wrote %d rows → %s", len(result), csv_path)
        created.append(csv_path)

    if "json" in formats:
        json_path = output_dir / f"consensus_{safe_name}.json"
        records = json.loads(result.to_json(orient="records", force_ascii=False))
        json_path.write_text(
            json.dumps(records, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info("Wrote %d rows → %s", len(result), json_path)
        created.append(json_path)

    return created


# ---------------------------------------------------------------------------
# High-level driver
# ---------------------------------------------------------------------------


def run_consensus(
    results_dir: Path,
    config_path: Path,
    track_names: list[str] | None = None,
    output_dir: Path | None = None,
) -> dict[str, pd.DataFrame]:
    """Run one or more consensus tracks end-to-end.

    Returns a dict mapping track_name → filtered DataFrame.
    """
    cfg = load_consensus_config(config_path)
    default_track = cfg.get("default_track", "moderate")
    tracks_raw = cfg.get("tracks", {})

    if not track_names:
        track_names = [default_track]

    # Load results
    df = load_results(results_dir)
    if df.empty:
        logger.error("No result data found in %s", results_dir)
        return {}

    available_models = sorted(df["model_label"].unique())
    logger.info(
        "Loaded %d rows from %d models: %s",
        len(df), len(available_models), ", ".join(available_models),
    )

    # Build passage groups once
    groups = build_passage_groups(df)
    logger.info(
        "Passage groups: %d total, %d multi-model",
        len(groups),
        sum(1 for g in groups if g["n_models"] > 1),
    )

    out_dir = output_dir or results_dir
    results: dict[str, pd.DataFrame] = {}

    for name in track_names:
        if name not in tracks_raw:
            logger.error("Track '%s' not found in %s (available: %s)",
                         name, config_path, ", ".join(tracks_raw.keys()))
            continue

        track = parse_track(name, tracks_raw[name])
        logger.info(
            "── Track: %s — agreement=%s, min_models=%d, min_confidence=%s, resolve=%s",
            track.name, track.agreement, track.min_models,
            track.min_confidence, track.resolve_type,
        )
        if track.models:
            logger.info("   Models: %s", ", ".join(track.models))

        filtered = apply_track(df, groups, track)
        logger.info("   → %d rows survived filtering", len(filtered))

        files = export_consensus(filtered, out_dir, track.name, track.output_formats)
        for f in files:
            print(f"  ✓ {f}")

        results[name] = filtered

    return results
