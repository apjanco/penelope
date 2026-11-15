"""Full-corpus run of Experiments A, B, C (HANDOFF_della.md sec. 5).

Every file below is complete (or 99%+ complete — a handful of passages in
Ulysses, The Sound and the Fury, and The Ambassadors were permanently
rejected by DashScope's content moderation and are instead read by a local
open-weight model, reader_id "qwen3.6-local", substituted in place of the
missing qwen3-max reading for just those ~11 passages — see
fill_qwen_gaps.slurm). Every other passage in the corpus is read by both
gpt-5.4 and qwen3-max, k=3.

Usage: python pilot_experiments_abc.py
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent

COMPLETE_FILES = [
    "as_I_lay_dying.jsonl",
    "between_the_acts.jsonl",
    "flush.jsonl",
    "jacobs_room.jsonl",
    "night_and_day_cleaned.jsonl",
    "orlando_cleaned.jsonl",
    "pg00432_the_ambassadors.jsonl",
    "pg02775_the_good_soldier.jsonl",
    "pg02814_dubliners.jsonl",
    "pg03019_pointed_roofs.jsonl",
    "pg04217_a_portrait_of_the_artist_as_a_young_man.jsonl",
    "pg04300_ulysses.jsonl",
    "pg29452_the_wings_of_the_dove_volume_1.jsonl",
    "pg30059_the_wings_of_the_dove_volume_2.jsonl",
    "pg56221_backwater.jsonl",
    "pg56664_honeycomb.jsonl",
    "pg75170_sound_and_fury.jsonl",
    "the_voyage_out.jsonl",
    "the_waves_cleaned.jsonl",
    "the_years.jsonl",
    "woolf_lighthouse_dalloway.jsonl",
]


def load() -> pd.DataFrame:
    rows = []
    for name in COMPLETE_FILES:
        for line in (ROOT / "scored" / name).read_text().splitlines():
            r = json.loads(line)
            rows.append(r)
    df = pd.DataFrame(rows)

    # The Wings of the Dove was split across two passage files (vol. 1/2)
    # purely for corpus-building reasons — it is one novel, not two
    # independent works. Counting it as 2 "titles" biases any per-title or
    # between-title variance calculation (James would get credit for
    # between-book variation that's actually just within-book position in
    # a single continuous novel). Merge before anything downstream uses title.
    wings_mask = df["title"].isin(["The Wings of the Dove (vol. 1)", "The Wings of the Dove (vol. 2)"])
    df.loc[wings_mask, "title"] = "The Wings of the Dove"

    print(f"Loaded {len(df)} (passage, reader, run) rows from {len(COMPLETE_FILES)} files")
    print(f"  {df['passage_id'].nunique()} unique passages, "
          f"{df['author'].nunique()} authors, "
          f"{df['title'].nunique()} titles")
    return df


def passage_mode_table(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse (passage, reader, k runs) -> one row per (passage, reader)
    using the modal mediation_level/fit_category, with a consistency score
    (fraction of runs matching the mode) — AMBIGUITY.md's own within-model
    stability measure."""
    records = []
    for (pid, reader), g in df.groupby(["passage_id", "reader_id"]):
        base = g.iloc[0]
        soc_runs = g[g["is_soc"] == True]  # noqa: E712
        if len(soc_runs) == 0:
            records.append({
                "passage_id": pid, "reader_id": reader,
                "author": base["author"], "title": base["title"], "year": base["year"],
                "is_soc_rate": 0.0, "mediation_level": np.nan, "fit_category": None,
                "consistency": (g["is_soc"] == False).mean(),  # noqa: E712
                "n_runs": len(g),
            })
            continue
        ml_mode = soc_runs["mediation_level"].mode()
        ml_mode = ml_mode.iloc[0] if len(ml_mode) else np.nan
        fc_mode = soc_runs["fit_category"].mode()
        fc_mode = fc_mode.iloc[0] if len(fc_mode) else None
        consistency = (soc_runs["mediation_level"] == ml_mode).mean() if not pd.isna(ml_mode) else np.nan
        records.append({
            "passage_id": pid, "reader_id": reader,
            "author": base["author"], "title": base["title"], "year": base["year"],
            "is_soc_rate": (g["is_soc"] == True).mean(),  # noqa: E712
            "mediation_level": ml_mode, "fit_category": fc_mode,
            "consistency": consistency, "n_runs": len(g),
        })
    return pd.DataFrame(records)


def experiment_a(df: pd.DataFrame, out: list[str]) -> None:
    out.append("## Experiment A — per-author fingerprints\n")
    soc = df[df["is_soc"] == True].dropna(subset=["mediation_level"])  # noqa: E712
    soc = soc.copy()
    soc["mediation_level"] = soc["mediation_level"].astype(float)

    out.append("| author | n (passage x reader x run) | mean level | std (spread) | level 0 | 1 | 2 | 3 |")
    out.append("|---|---|---|---|---|---|---|---|")
    author_stats = {}
    for author, g in soc.groupby("author"):
        counts = g["mediation_level"].value_counts(normalize=True).reindex([0, 1, 2, 3], fill_value=0.0)
        mean, std = g["mediation_level"].mean(), g["mediation_level"].std()
        author_stats[author] = (mean, std)
        out.append(
            f"| {author} | {len(g)} | {mean:.2f} | {std:.2f} | "
            f"{counts[0]:.0%} | {counts[1]:.0%} | {counts[2]:.0%} | {counts[3]:.0%} |"
        )
    out.append("")

    out.append("**By title** (finer grain, previews the Joyce/Woolf developmental-arc question):\n")
    out.append("| author | title | year | n | mean level | std |")
    out.append("|---|---|---|---|---|---|")
    for (author, title, year), g in soc.groupby(["author", "title", "year"]):
        out.append(f"| {author} | {title} | {year} | {len(g)} | {g['mediation_level'].mean():.2f} | {g['mediation_level'].std():.2f} |")
    out.append("")

    # Prediction check against HANDOFF sec 5, Experiment A
    out.append("**Predictions (HANDOFF sec. 5, Experiment A):**\n")
    faulkner = author_stats.get("Faulkner")
    woolf = author_stats.get("Woolf")
    richardson = author_stats.get("Richardson")
    james = author_stats.get("James")
    ford = author_stats.get("Ford")
    if faulkner:
        out.append(f"- Faulkner (genre pole): predicted spread >1.2 — full corpus shows std={faulkner[1]:.2f} "
                    f"({'CONFIRMED' if faulkner[1] > 1.2 else 'NOT CONFIRMED'}).")
    if woolf:
        out.append(f"- Woolf (technique author): predicted spread ~0.5, peak at level 1 — full corpus shows std={woolf[1]:.2f}, mean={woolf[0]:.2f}.")
    if richardson:
        out.append(f"- Richardson (technique author): predicted spread ~0.5, peak at level 1 — full corpus shows std={richardson[1]:.2f}, mean={richardson[0]:.2f} "
                    f"(now all three Pilgrimage volumes in the corpus: Pointed Roofs, Backwater, Honeycomb).")
    if james:
        out.append(f"- James (control, floor): predicted mass at levels 0-1 — full corpus mean={james[0]:.2f} "
                    f"({'CONFIRMED, lowest or near-lowest mean in panel' if james[0] == min(v[0] for v in author_stats.values()) else 'mean=' + str(round(james[0],2))}).")
    if ford:
        out.append(f"- Ford (control, boundary case): full corpus mean={ford[0]:.2f}, std={ford[1]:.2f}.")
    out.append(
        "- Joyce developmental arc (Dubliners -> Portrait -> Ulysses marching rightward): "
        "now testable with all three works at full or near-full completion."
    )
    joyce = soc[soc["author"] == "Joyce"]
    if len(joyce):
        joyce_means = []
        for title, g in joyce.groupby("title"):
            m = g["mediation_level"].mean()
            joyce_means.append((title, m))
            out.append(f"  - {title}: mean={m:.2f}, n={len(g)}")
        year_order = joyce.groupby("title")["year"].first().sort_values().index.tolist()
        ordered_means = [dict(joyce_means)[t] for t in year_order]
        marches_rightward = all(ordered_means[i] <= ordered_means[i + 1] for i in range(len(ordered_means) - 1))
        out.append(f"  - Marches rightward by publication order: {'CONFIRMED' if marches_rightward else 'NOT CONFIRMED'} ({' -> '.join(year_order)}).")
    out.append("")


def experiment_b(df: pd.DataFrame, out: list[str]) -> None:
    out.append("## Experiment B — geometry of the cline\n")
    soc = df[df["is_soc"] == True].dropna(subset=["mediation_level"])  # noqa: E712
    counts = soc["mediation_level"].value_counts(normalize=True).reindex([0, 1, 2, 3], fill_value=0.0)
    raw_counts = soc["mediation_level"].value_counts().reindex([0, 1, 2, 3], fill_value=0)
    out.append("Pooled mediation_level density across all authors/readers/runs in the full corpus:\n")
    out.append("| level | name | proportion | raw n |")
    out.append("|---|---|---|---|")
    names = {0: "psycho-narration", 1: "narrated monologue / FID", 2: "quoted monologue", 3: "autonomous monologue"}
    for lvl in [0, 1, 2, 3]:
        out.append(f"| {lvl} | {names[lvl]} | {counts[lvl]:.1%} | {raw_counts[lvl]} |")
    out.append("")

    vals = counts.values
    # Discrete "dip" check: a level is a valley if its count is lower than
    # both neighbors (interior levels only).
    valleys = [lvl for lvl in [1, 2] if vals[lvl] < vals[lvl - 1] and vals[lvl] < vals[lvl + 1]]
    monotonic = all(vals[i] >= vals[i + 1] for i in range(3)) or all(vals[i] <= vals[i + 1] for i in range(3))
    out.append(f"Valleys detected at level(s): {valleys or 'none'}. Monotonic across 0-3: {monotonic}.")
    out.append(
        "Interpretation: with only 4 discrete bins, this is still a coarse read, not a real "
        "dip-test/GMM fit (per HANDOFF sec. 5 Experiment B, that would want a continuous soft "
        "score). Reported here as a first look at full-corpus scale, not a final verdict.\n"
    )

    # Bonus check: does boundary_tension concentrate in the valleys?
    bt_by_level = soc.groupby("mediation_level")["fit_category"].apply(
        lambda s: (s == "boundary_tension").mean()
    ).reindex([0, 1, 2, 3])
    out.append("`fit_category == boundary_tension` rate by mediation_level (bonus check from HANDOFF sec. 5):\n")
    out.append("| level | boundary_tension rate |")
    out.append("|---|---|")
    for lvl in [0, 1, 2, 3]:
        out.append(f"| {lvl} | {bt_by_level.get(lvl, float('nan')):.1%} |")
    out.append("")


def experiment_c(df: pd.DataFrame, out: list[str]) -> None:
    out.append("## Experiment C — moving frontier, diachronic arm\n")
    out.append(
        f"Year range: {int(df['year'].min())}-{int(df['year'].max())} "
        f"({df['title'].nunique()} titles, {df['author'].nunique()} authors).\n"
    )
    by_title = df.groupby(["title", "year"]).apply(
        lambda g: pd.Series({
            "no_type_fits_rate": (g["fit_category"] == "no_type_fits").mean(),
            "boundary_tension_rate": (g["fit_category"] == "boundary_tension").mean(),
            "antimimetic_rate": (g["antimimetic"] == True).mean(),  # noqa: E712
            "n": len(g),
        }),
        include_groups=False,
    ).reset_index().sort_values("year")
    out.append("| year | title | no_type_fits | boundary_tension | antimimetic | n |")
    out.append("|---|---|---|---|---|---|")
    for _, r in by_title.iterrows():
        out.append(
            f"| {int(r['year'])} | {r['title']} | {r['no_type_fits_rate']:.1%} | "
            f"{r['boundary_tension_rate']:.1%} | {r['antimimetic_rate']:.2%} | {int(r['n'])} |"
        )
    out.append("")
    corr = stats.spearmanr(by_title["year"], by_title["no_type_fits_rate"])
    out.append(
        f"Spearman correlation, year vs. no_type_fits_rate across titles: "
        f"rho={corr.statistic:.3f}, p={corr.pvalue:.3f}. "
        f"{'Suggestive of a frontier trend' if corr.pvalue < 0.1 else 'No clear trend at the title level'} "
        f"— this is a by-title ({by_title.shape[0]} titles) summary; HANDOFF sec. 5's "
        "passage-level diachronic curve would be a finer-grained follow-up.\n"
    )


def author_variance_decomposition(df: pd.DataFrame, out: list[str]) -> None:
    out.append("## Author variance decomposition — within-oeuvre texture\n")
    out.append(
        "For authors with multiple titles in the corpus, splits total variance in "
        "mediation_level into a between-title component (which book you're reading) and a "
        "within-title component (passage-to-passage noise inside one book). Each passage is "
        "first collapsed to its mean level across readers/runs, so titles aren't weighted by "
        "how many reader-runs happen to exist for them. eta^2 = between-title SS / total SS: "
        "the fraction of an author's overall spread that is explained by which book you're "
        "reading, rather than by ordinary passage-to-passage variation.\n"
    )
    soc = df[df["is_soc"] == True].dropna(subset=["mediation_level"])  # noqa: E712
    soc = soc.copy()
    soc["mediation_level"] = soc["mediation_level"].astype(float)
    per_passage = (
        soc.groupby(["passage_id", "author", "title", "year"])["mediation_level"]
        .mean()
        .reset_index()
    )

    out.append("| author | titles | n passages | std of title means | between-title variance share (eta^2) |")
    out.append("|---|---|---|---|---|")
    results = []
    for author, g in per_passage.groupby("author"):
        n_titles = g["title"].nunique()
        if n_titles < 2:
            continue
        grand_mean = g["mediation_level"].mean()
        title_means = g.groupby("title")["mediation_level"].mean()
        title_ns = g.groupby("title")["mediation_level"].count()
        ss_between = sum(title_ns[t] * (title_means[t] - grand_mean) ** 2 for t in title_means.index)
        ss_total = ((g["mediation_level"] - grand_mean) ** 2).sum()
        eta_sq = ss_between / ss_total if ss_total else float("nan")
        results.append((author, n_titles, len(g), title_means))
        out.append(f"| {author} | {n_titles} | {len(g)} | {title_means.std():.3f} | {eta_sq:.3f} |")
    out.append("")

    out.append("Per-title means, low to high, within each author:\n")
    for author, n_titles, n, title_means in results:
        out.append(f"- **{author}** ({n_titles} titles, n={n} passages):")
        for t in title_means.sort_values().index:
            out.append(f"    - {t}: {title_means[t]:.2f}")
    out.append("")
    out.append(
        "Authors with only one title in the corpus (Ford) can't be scored on this axis at all "
        "— not evidence of a fixed style, just no within-oeuvre comparison available yet.\n"
    )


def main():
    df = load()
    modes = passage_mode_table(df)
    modes.to_csv(ROOT / "pilot_passage_modes.csv", index=False)

    n_local = (df["reader_id"] == "qwen3.6-local").sum()
    out: list[str] = [
        "# Full-corpus run — Experiments A, B, C\n",
        f"Run on the complete corpus: {len(COMPLETE_FILES)} files, "
        f"{df['passage_id'].nunique()} passages, {df['author'].nunique()} authors, "
        f"{df['title'].nunique()} titles, {len(df)} total (passage, reader, run) rows. "
        "Every passage is read by gpt-5.4 at k=3; nearly all are also read by qwen3-max at k=3. "
        f"{n_local} rows ({n_local // 3} passages) across Ulysses, The Sound and the Fury, and "
        "The Ambassadors were permanently rejected by DashScope's content moderation "
        "(data_inspection_failed) and are instead read by a local open-weight model "
        "(reader_id `qwen3.6-local`, Qwen/Qwen3.6-27B served via vLLM — see "
        "`fill_qwen_gaps.slurm`), substituted in place of the missing qwen3-max reading for "
        "just those passages. This is a genuinely different model, correctly labeled as its "
        "own reader, not merged into the qwen3-max column.\n",
    ]
    experiment_a(df, out)
    author_variance_decomposition(df, out)
    experiment_b(df, out)
    experiment_c(df, out)

    report_path = ROOT / "pilot_report.md"
    report_path.write_text("\n".join(out))
    print(f"\nWritten: {report_path}")
    print(f"Written: {ROOT / 'pilot_passage_modes.csv'}")


if __name__ == "__main__":
    main()
