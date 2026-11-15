"""How much of the corpus cleanly fits Humphrey/Steinberg's twelve-slug
taxonomy (soc_type), and how much defies it?

Uses the fit_category field, which is an explicit per-instance judgment
(not inferred after the fact from cross-reader agreement):
  - technique_fit: soc_type and mediation_level both apply cleanly
  - boundary_tension: one other named type was seriously in contention
  - no_type_fits: the passage unambiguously renders consciousness but
    resists every named category

Reports both row-level (every passage/reader/run counted once, matching
how pilot_experiments_abc.py's Experiment B/C already report) and a
passage-level consensus version (majority vote across all reader/run rows
for a given passage) as a robustness check on the row-level read.

Usage: python humphrey_coverage.py
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pandas as pd

from pilot_experiments_abc import COMPLETE_FILES

ROOT = Path(__file__).resolve().parent

SOC_TYPES = [
    "direct_interior_monologue", "indirect_interior_monologue",
    "omniscient_description", "soliloquy", "free_association",
    "space_montage", "orthographic_marker", "imagery",
    "simulation_state_of_mind", "reverie_fantasy", "hybrid", "other_soc",
]
FIT_CATEGORIES = ["technique_fit", "boundary_tension", "no_type_fits"]


def load() -> pd.DataFrame:
    rows = []
    for name in COMPLETE_FILES:
        for line in (ROOT / "scored" / name).read_text().splitlines():
            rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    print(f"Loaded {len(df)} (passage, reader, run) rows from {len(COMPLETE_FILES)} files")
    return df


def main():
    df = load()
    soc = df[df["is_soc"] == True].dropna(subset=["fit_category"])  # noqa: E712
    print(f"{len(soc)} SoC-positive (passage, reader, run) rows "
          f"({soc['passage_id'].nunique()} unique passages)\n")

    out: list[str] = ["# Humphrey/Steinberg taxonomy coverage\n"]
    out.append(
        f"Run on the full corpus: {len(COMPLETE_FILES)} files, "
        f"{soc['passage_id'].nunique()} SoC-positive passages, "
        f"{len(soc)} total (passage, reader, run) rows where is_soc=true.\n"
    )

    # ---- Row-level fit_category distribution -----------------------------
    out.append("## Overall fit — row level\n")
    out.append("Every (passage, reader, run) judgment counted once:\n")
    counts = soc["fit_category"].value_counts()
    props = soc["fit_category"].value_counts(normalize=True)
    out.append("| fit_category | proportion | raw n |")
    out.append("|---|---|---|")
    for cat in FIT_CATEGORIES:
        out.append(f"| {cat} | {props.get(cat, 0):.1%} | {counts.get(cat, 0)} |")
    out.append("")

    # ---- Passage-level: reader-mode agreement, not a raw row-level mode --
    # A plain mode() over all (reader, run) rows silently tiebreaks ties
    # alphabetically, which matters a lot here: exact technique_fit vs.
    # boundary_tension ties turn out to be extremely common (one reader
    # saying technique_fit on all k runs, the other saying boundary_tension
    # on all k runs — i.e. genuine stable cross-reader disagreement, not
    # noise). Classify at the reader level first, then compare readers,
    # matching this project's own resampling methodology (AMBIGUITY.md):
    # stable disagreement between internally-consistent readers is signal,
    # not something to average away.
    out.append("## Overall fit — passage-level, reader-mode comparison\n")
    out.append(
        "For each passage, each reader's own k=3 runs are first collapsed to "
        "that reader's modal fit_category. Passages are then classified by "
        "whether the readers' modes agree:\n"
    )

    def classify_passage(g: pd.DataFrame) -> tuple[str, str]:
        reader_modes = g.groupby("reader_id")["fit_category"].agg(lambda s: s.mode().iloc[0])
        if reader_modes.nunique() == 1:
            return ("consensus", reader_modes.iloc[0])
        return ("stable_disagreement", "/".join(sorted(reader_modes.unique())))

    classified = soc.groupby("passage_id").apply(
        lambda g: pd.Series(classify_passage(g), index=["kind", "detail"])
    )
    kind_counts = classified["kind"].value_counts()
    kind_props = classified["kind"].value_counts(normalize=True)
    out.append("| | proportion of passages | raw n passages |")
    out.append("|---|---|---|")
    out.append(f"| readers agree (consensus) | {kind_props.get('consensus', 0):.1%} | {kind_counts.get('consensus', 0)} |")
    out.append(f"| readers stably disagree | {kind_props.get('stable_disagreement', 0):.1%} | {kind_counts.get('stable_disagreement', 0)} |")
    out.append("")

    out.append("Where readers agree, what they agree on:\n")
    cons = classified[classified["kind"] == "consensus"]["detail"].value_counts(normalize=True)
    cons_n = classified[classified["kind"] == "consensus"]["detail"].value_counts()
    out.append("| fit_category | proportion of consensus passages | raw n |")
    out.append("|---|---|---|")
    for cat in FIT_CATEGORIES:
        out.append(f"| {cat} | {cons.get(cat, 0):.1%} | {cons_n.get(cat, 0)} |")
    out.append("")

    out.append("Where readers stably disagree, which pairing:\n")
    disagree = classified[classified["kind"] == "stable_disagreement"]["detail"].value_counts()
    out.append("| pairing | raw n passages |")
    out.append("|---|---|")
    for pairing, n in disagree.items():
        out.append(f"| {pairing} | {n} |")
    out.append("")

    # ---- soc_type distribution --------------------------------------------
    out.append("## soc_type distribution (row level)\n")
    out.append("Which of the twelve Humphrey/Steinberg slugs the corpus actually uses:\n")
    st_counts = soc["soc_type"].value_counts()
    st_props = soc["soc_type"].value_counts(normalize=True)
    out.append("| soc_type | proportion | raw n |")
    out.append("|---|---|---|")
    for t in SOC_TYPES:
        if t in st_counts.index:
            out.append(f"| {t} | {st_props[t]:.1%} | {st_counts[t]} |")
    missing = [t for t in SOC_TYPES if t not in st_counts.index]
    if missing:
        out.append("")
        out.append(f"Never assigned in this corpus: {', '.join(missing)}.")
    out.append("")

    # ---- Cross-tab: soc_type x fit_category --------------------------------
    out.append("## Which named types fit cleanly, and which strain the taxonomy\n")
    out.append(
        "fit_category rate by soc_type (row level) — where a given named "
        "type most often lands once assigned:\n"
    )
    ct = pd.crosstab(soc["soc_type"], soc["fit_category"], normalize="index")
    out.append("| soc_type | technique_fit | boundary_tension | no_type_fits | n |")
    out.append("|---|---|---|---|---|")
    type_n = soc["soc_type"].value_counts()
    for t in SOC_TYPES:
        if t in ct.index:
            row = ct.loc[t]
            out.append(
                f"| {t} | {row.get('technique_fit', 0):.1%} | "
                f"{row.get('boundary_tension', 0):.1%} | "
                f"{row.get('no_type_fits', 0):.1%} | {type_n[t]} |"
            )
    out.append("")

    # ---- fit_category by author ---------------------------------------------
    out.append("## fit_category by author (row level)\n")
    out.append(
        "Whether the taxonomy strains more for some authors than others — "
        "connects to the genre-pole/technique-author distinction "
        "(Experiment A) and the diachronic frontier (Experiment C):\n"
    )
    act = pd.crosstab(soc["author"], soc["fit_category"], normalize="index")
    author_n = soc["author"].value_counts()
    out.append("| author | technique_fit | boundary_tension | no_type_fits | n |")
    out.append("|---|---|---|---|---|")
    for author in act.index:
        row = act.loc[author]
        out.append(
            f"| {author} | {row.get('technique_fit', 0):.1%} | "
            f"{row.get('boundary_tension', 0):.1%} | "
            f"{row.get('no_type_fits', 0):.1%} | {author_n[author]} |"
        )
    out.append("")

    # ---- antimimetic cross-check -------------------------------------------
    out.append("## antimimetic flag (the stricter genre-pole criterion)\n")
    out.append(
        "antimimetic=true is reserved for no_type_fits passages with an "
        "explicit textual reason (Richardson 2006 / Alber 2010's unnatural "
        "narratology), not merely low-confidence calls. Rate among "
        "no_type_fits rows specifically:\n"
    )
    ntf = soc[soc["fit_category"] == "no_type_fits"]
    if len(ntf):
        am_rate = (ntf["antimimetic"] == True).mean()  # noqa: E712
        out.append(
            f"{am_rate:.1%} of no_type_fits rows ({len(ntf)} total) are also "
            f"flagged antimimetic=true — the rest are no_type_fits without "
            "that stricter theoretical justification attached.\n"
        )

    report_path = ROOT / "humphrey_coverage_report.md"
    report_path.write_text("\n".join(out))
    print(f"\nWritten: {report_path}")


if __name__ == "__main__":
    main()
