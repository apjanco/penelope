#!/usr/bin/env python3
"""Penelope — Streamlit app for comparing SOC analysis results across models.

Launch:  streamlit run app.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RESULTS_DIR = Path("results")
MIN_OVERLAP_RATIO = 0.35  # minimum token overlap to consider passages "matched"

SOC_TYPE_LABELS: dict[str, str] = {
    "direct_interior_monologue": "Direct Interior Monologue",
    "indirect_interior_monologue": "Indirect Interior Monologue",
    "omniscient_description": "Omniscient Description",
    "soliloquy": "Soliloquy",
    "free_association": "Free Association",
    "space_montage": "Space-Montage",
    "orthographic_marker": "Orthographic Marker",
    "imagery": "Imagery",
    "simulation_state_of_mind": "Simulation of State of Mind",
    "reverie_fantasy": "Reverie / Fantasy",
    "hybrid": "Hybrid",
}

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


@st.cache_data
def load_results(results_dir: str = "results") -> pd.DataFrame:
    """Load all per-model JSON files from results/ into one DataFrame."""
    frames: list[pd.DataFrame] = []
    rdir = Path(results_dir)
    for f in sorted(rdir.glob("*.json")):
        # Skip the combined results.json if present
        if f.stem == "results":
            continue
        data = json.loads(f.read_text(encoding="utf-8"))
        if data:
            df = pd.DataFrame(data)
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    # Normalise whitespace in passages for better matching
    combined["passage_norm"] = combined["passage"].apply(_normalise_text)
    combined["passage_tokens"] = combined["passage_norm"].apply(lambda t: set(t.split()))
    return combined


def _normalise_text(text: str) -> str:
    """Lowercase, collapse whitespace, strip punctuation for matching."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# Passage matching — group passages across models that refer to the same text
# ---------------------------------------------------------------------------


def _token_overlap(a: set[str], b: set[str]) -> float:
    """Jaccard-like overlap ratio between two token sets."""
    if not a or not b:
        return 0.0
    intersection = len(a & b)
    smaller = min(len(a), len(b))
    return intersection / smaller if smaller else 0.0


@st.cache_data
def build_passage_groups(_df_json: str) -> list[dict]:
    """Cluster passages across models that overlap significantly.

    Uses a greedy approach: for each passage, find or create a group where
    token overlap with at least one existing member exceeds MIN_OVERLAP_RATIO.

    Returns a list of group dicts:
        {
            "group_id": int,
            "representative": str,       # longest passage text
            "models": list[str],
            "rows": list[int],           # DataFrame indices
            "chunk_id": str,
            "source_file": str,
            "n_models": int,
            "agreement": str,            # "full" / "partial" / "single"
        }
    """
    from io import StringIO
    df = pd.read_json(StringIO(_df_json), dtype={"chunk_index": int})
    if df.empty:
        return []

    df["passage_norm"] = df["passage"].apply(_normalise_text)
    df["passage_tokens"] = df["passage_norm"].apply(lambda t: set(t.split()))

    groups: list[dict] = []
    assigned: set[int] = set()

    # Process by chunk for efficiency (passages from different chunks can't match)
    for chunk_id, chunk_df in df.groupby("chunk_id"):
        idxs = chunk_df.index.tolist()
        for i in idxs:
            if i in assigned:
                continue
            tokens_i = df.at[i, "passage_tokens"]
            # Try to find a matching group
            matched_group = None
            for g in groups:
                if g["chunk_id"] != chunk_id:
                    continue
                for member_idx in g["rows"]:
                    tokens_m = df.at[member_idx, "passage_tokens"]
                    if _token_overlap(tokens_i, tokens_m) >= MIN_OVERLAP_RATIO:
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

    # Enrich groups
    all_models = sorted(df["model_label"].unique())
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

    # Sort: multi-model groups first, then by chunk
    groups.sort(key=lambda g: (-g["n_models"], g["chunk_id"], g["group_id"]))
    # Re-number
    for i, g in enumerate(groups):
        g["group_id"] = i

    return groups


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="Penelope — SOC Model Comparison",
        page_icon="🧶",
        layout="wide",
    )
    st.title("🧶 Penelope — SOC Model Comparison")

    # Load data
    df = load_results(str(RESULTS_DIR))
    if df.empty:
        st.error(f"No result JSON files found in `{RESULTS_DIR}/`.")
        st.stop()

    all_models = sorted(df["model_label"].unique())
    all_files = sorted(df["source_file"].unique())

    # --- Sidebar filters ---
    st.sidebar.header("Filters")
    sel_files = st.sidebar.multiselect(
        "Source files", all_files, default=all_files
    )
    sel_models = st.sidebar.multiselect(
        "Models", all_models, default=all_models
    )
    min_models = st.sidebar.slider(
        "Min models marking passage", 1, len(all_models), 2,
        help="Show only passage groups identified by at least N models",
    )

    mask = df["source_file"].isin(sel_files) & df["model_label"].isin(sel_models)
    filtered = df[mask].copy()

    if filtered.empty:
        st.warning("No data matches the current filters.")
        st.stop()

    # Tabs
    tab_overview, tab_compare, tab_detail, tab_data = st.tabs([
        "📊 Overview", "🔍 Passage Comparison", "📖 Detail View", "📋 Raw Data"
    ])

    # ── Tab 1: Overview ────────────────────────────────────────────────
    with tab_overview:
        _render_overview(filtered, all_models)

    # ── Tab 2: Passage Comparison ──────────────────────────────────────
    with tab_compare:
        _render_comparison(df, filtered, all_models, sel_models, min_models)

    # ── Tab 3: Detail View ─────────────────────────────────────────────
    with tab_detail:
        _render_detail(filtered, all_models)

    # ── Tab 4: Raw Data ────────────────────────────────────────────────
    with tab_data:
        _render_raw_data(filtered)


# ---------------------------------------------------------------------------
# Tab: Overview
# ---------------------------------------------------------------------------

def _render_overview(df: pd.DataFrame, all_models: list[str]) -> None:
    st.header("Overview")

    # KPI cards
    cols = st.columns(4)
    cols[0].metric("Total instances", len(df))
    cols[1].metric("Models", df["model_label"].nunique())
    cols[2].metric("Source files", df["source_file"].nunique())
    cols[3].metric("Chunks covered", df["chunk_id"].nunique())

    st.subheader("Instances per model")
    model_counts = df.groupby("model_label").size().reset_index(name="count")
    st.bar_chart(model_counts.set_index("model_label")["count"])

    # SOC type distribution
    st.subheader("SOC type distribution by model")
    type_model = (
        df.groupby(["model_label", "soc_type"])
        .size()
        .reset_index(name="count")
    )
    pivot = type_model.pivot(index="soc_type", columns="model_label", values="count").fillna(0)
    st.bar_chart(pivot)

    # Confidence breakdown
    st.subheader("Confidence breakdown")
    conf_model = (
        df.groupby(["model_label", "confidence"])
        .size()
        .reset_index(name="count")
    )
    conf_pivot = conf_model.pivot(index="confidence", columns="model_label", values="count").fillna(0)
    # Reorder
    for order_val in ["high", "medium", "low"]:
        if order_val not in conf_pivot.index:
            conf_pivot.loc[order_val] = 0
    conf_pivot = conf_pivot.loc[
        [v for v in ["high", "medium", "low"] if v in conf_pivot.index]
    ]
    st.bar_chart(conf_pivot)

    # Coverage heatmap: which chunks does each model annotate?
    st.subheader("Chunk coverage by model")
    coverage = (
        df.groupby(["chunk_id", "model_label"])
        .size()
        .reset_index(name="instances")
    )
    cov_pivot = coverage.pivot(index="chunk_id", columns="model_label", values="instances").fillna(0)
    cov_pivot = cov_pivot.sort_index()
    st.dataframe(
        cov_pivot.style.background_gradient(cmap="YlOrRd", axis=None),
        use_container_width=True,
        height=min(len(cov_pivot) * 35 + 50, 600),
    )


# ---------------------------------------------------------------------------
# Tab: Passage Comparison
# ---------------------------------------------------------------------------

def _render_comparison(
    full_df: pd.DataFrame,
    filtered: pd.DataFrame,
    all_models: list[str],
    sel_models: list[str],
    min_models: int,
) -> None:
    st.header("Passage Comparison")
    st.caption(
        "Passages from different models are grouped when they share significant "
        "token overlap (≥35% of the shorter passage). This catches near-identical "
        "quotes as well as passages where models quoted slightly different spans."
    )

    # Build groups from the full dataset (so matching works across all models)
    groups = build_passage_groups(full_df.drop(columns=["passage_tokens"]).to_json())

    # Filter groups
    visible_groups = [
        g for g in groups
        if g["n_models"] >= min_models
        and any(m in sel_models for m in g["models"])
        and g["source_file"] in filtered["source_file"].values
    ]

    if not visible_groups:
        st.info("No passage groups match the current filters. Try lowering the minimum models slider.")
        return

    # Summary metrics
    c1, c2, c3, c4 = st.columns(4)
    multi = [g for g in visible_groups if g["n_models"] > 1]
    full_agree = [g for g in multi if g["agreement"] == "full"]
    partial = [g for g in multi if g["agreement"] == "partial"]
    c1.metric("Passage groups", len(visible_groups))
    c2.metric("Multi-model groups", len(multi))
    c3.metric("Full type agreement", len(full_agree))
    c4.metric("Partial agreement", len(partial))

    # Agreement filter
    agree_filter = st.radio(
        "Show", ["All", "Full agreement", "Partial agreement", "Single model"],
        horizontal=True,
    )
    if agree_filter == "Full agreement":
        visible_groups = [g for g in visible_groups if g["agreement"] == "full"]
    elif agree_filter == "Partial agreement":
        visible_groups = [g for g in visible_groups if g["agreement"] == "partial"]
    elif agree_filter == "Single model":
        visible_groups = [g for g in visible_groups if g["agreement"] == "single"]

    st.divider()

    # Render each group
    for g in visible_groups:
        _render_group(g, full_df, sel_models)


def _render_group(group: dict, df: pd.DataFrame, sel_models: list[str]) -> None:
    """Render one passage group as an expandable card."""
    n = group["n_models"]
    agreement = group["agreement"]
    models_str = ", ".join(group["models"])

    # Badge colours
    if agreement == "full":
        badge = "🟢 Full agreement"
    elif agreement == "partial":
        badge = "🟡 Partial agreement"
    else:
        badge = "⚪ Single model"

    # Types in this group
    types_in_group = set()
    for idx in group["rows"]:
        if df.at[idx, "model_label"] in sel_models:
            types_in_group.add(df.at[idx, "soc_type"])

    preview = group["representative"][:120] + ("…" if len(group["representative"]) > 120 else "")
    header = f"{badge}  |  **{n} model(s)**  |  {', '.join(types_in_group)}  |  `{group['chunk_id']}`"

    with st.expander(f"**{preview}**\n\n{header}", expanded=False):
        # Collect each model's annotation(s) for this group
        relevant_rows = [
            idx for idx in group["rows"]
            if df.at[idx, "model_label"] in sel_models
        ]
        model_groups: dict[str, list[int]] = {}
        for idx in relevant_rows:
            model = df.at[idx, "model_label"]
            model_groups.setdefault(model, []).append(idx)

        # Build a comparison table: rows = fields, columns = models
        models_ordered = sorted(model_groups.keys())

        # Some models may have multiple matches in this group; show the
        # first one in the main table and note extras below.
        primary_idxs = {m: idxs[0] for m, idxs in model_groups.items()}
        extra_idxs = {m: idxs[1:] for m, idxs in model_groups.items() if len(idxs) > 1}

        fields = [
            ("SOC Type",           "soc_type",           lambda v: SOC_TYPE_LABELS.get(v, v)),
            ("Confidence",         "confidence",         None),
            ("Narrator Position",  "narrator_position",  None),
            ("Character POV",      "character_pov",      None),
            ("Secondary Devices",  "secondary_devices",  None),
            ("Affective Register", "affective_register", None),
            ("Passage",            "passage",            None),
            ("Explanation",        "explanation",        None),
            ("Evidence",           "evidence",           None),
            ("Notes",              "notes",              None),
        ]

        # Build markdown table
        header_row = "| Field | " + " | ".join(f"**{m}**" for m in models_ordered) + " |"
        sep_row = "|---|" + "|".join("---" for _ in models_ordered) + "|"
        table_rows = [header_row, sep_row]

        for label, key, fmt in fields:
            cells: list[str] = []
            for m in models_ordered:
                row = df.iloc[primary_idxs[m]]
                val = row.get(key, "")
                if pd.isna(val) or val == "":
                    val = "—"
                else:
                    val = str(val)
                    if fmt:
                        val = fmt(val)
                # Escape pipes and collapse newlines for markdown table cells
                val = val.replace("|", "\\|").replace("\n", " ")
                # Truncate very long cells to keep table readable
                if len(val) > 300:
                    val = val[:297] + "…"
                cells.append(val)
            table_rows.append(f"| **{label}** | " + " | ".join(cells) + " |")

        st.markdown("\n".join(table_rows))

        # If any model had multiple matches, show them below
        if extra_idxs:
            st.markdown("---")
            st.caption("Additional matches within this group:")
            for m, idxs in sorted(extra_idxs.items()):
                for idx in idxs:
                    row = df.iloc[idx]
                    soc_label = SOC_TYPE_LABELS.get(row["soc_type"], row["soc_type"])
                    st.caption(
                        f"**{m}** — {soc_label} ({row['confidence']}) — "
                        f"{str(row['passage'])[:100]}…"
                    )


# ---------------------------------------------------------------------------
# Tab: Detail View
# ---------------------------------------------------------------------------

def _render_detail(df: pd.DataFrame, all_models: list[str]) -> None:
    st.header("Detail View")
    st.caption("Browse individual passages. Select a chunk to see all annotations.")

    chunks = sorted(df["chunk_id"].unique())
    sel_chunk = st.selectbox("Chunk", chunks)

    chunk_df = df[df["chunk_id"] == sel_chunk].copy()
    if chunk_df.empty:
        st.info("No annotations for this chunk.")
        return

    st.subheader(f"Chunk: {sel_chunk}")
    if not chunk_df.empty:
        st.caption(f"Source: {chunk_df.iloc[0]['source_file']}  |  Label: {chunk_df.iloc[0].get('chunk_label', '')}")

    # Group by model
    for model in sorted(chunk_df["model_label"].unique()):
        model_df = chunk_df[chunk_df["model_label"] == model]
        st.markdown(f"### {model}  ({len(model_df)} instances)")

        for _, row in model_df.iterrows():
            soc_label = SOC_TYPE_LABELS.get(row["soc_type"], row["soc_type"])
            with st.expander(
                f"**{soc_label}** — {row['confidence']} confidence — "
                f"{row['passage'][:80]}…"
            ):
                st.markdown(f"**Passage:**\n> {row['passage']}")
                st.markdown(f"**SOC Type:** {soc_label}")
                st.markdown(f"**Confidence:** {row['confidence']}")
                st.markdown(f"**Narrator position:** {row.get('narrator_position', 'n/a')}")
                st.markdown(f"**Character POV:** {row.get('character_pov', 'n/a')}")
                if row.get("secondary_devices"):
                    st.markdown(f"**Secondary devices:** {row['secondary_devices']}")
                if row.get("affective_register") and row["affective_register"] != "n/a":
                    st.markdown(f"**Affective register:** {row['affective_register']}")
                st.markdown(f"**Explanation:** {row['explanation']}")
                if row.get("evidence"):
                    st.markdown(f"**Evidence:** {row['evidence']}")
                if row.get("notes"):
                    st.markdown(f"**Notes:** {row['notes']}")


# ---------------------------------------------------------------------------
# Tab: Raw Data
# ---------------------------------------------------------------------------

def _render_raw_data(df: pd.DataFrame) -> None:
    st.header("Raw Data")

    display_cols = [
        "model_label", "source_file", "chunk_id", "chunk_label",
        "passage", "soc_type", "secondary_devices", "confidence",
        "narrator_position", "character_pov", "explanation",
        "evidence", "notes",
    ]
    available = [c for c in display_cols if c in df.columns]

    st.dataframe(
        df[available],
        use_container_width=True,
        height=600,
    )

    # Download
    csv_data = df[available].to_csv(index=False)
    st.download_button(
        "⬇ Download filtered data as CSV",
        csv_data,
        file_name="penelope_filtered.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
