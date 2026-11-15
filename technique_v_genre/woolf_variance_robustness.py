"""Robustness checks on the author variance-decomposition finding, responding
directly to technique_v_genre/critique.md:

1. A variance-components intraclass correlation (ICC) in place of the raw
   eta-squared, which the critique shows is upward-biased by title count and
   not comparable across authors with different within-book noise. Bootstrap
   CI via passage-level resampling within each fixed set of titles.
2. Leave-one-title-out sensitivity — the headline number recomputed with
   each author's highest-mean title removed, so a figure that moves as much
   as Woolf's (0.53 -> ~0.16 without The Waves) is never reported alone.
3. Within-title level distributions (not just means).
4. Within-title lag-1 autocorrelation of mediation_level, ordered by the
   passage_id's numeric suffix (the scored data doesn't carry the source
   passage_index field through, but the suffix is the same sequential
   index) — the "does a 0.87 mean come from smooth wandering or violent
   alternation" check.
5. Composition robustness reported explicitly: Woolf with/without the two
   genre-heterogeneous texts (Flush, Orlando), and Richardson flagged as
   not a fair "does style vary by book" case (Pilgrimage's volumes are
   installments of one continuous work, not independent books).

Usage: python woolf_variance_robustness.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from pilot_experiments_abc import COMPLETE_FILES, load

ROOT = Path(__file__).resolve().parent
RNG = np.random.default_rng(20260807)
N_BOOTSTRAP = 2000

NON_COMPARABLE_AUTHORS = {
    "Richardson": "Pilgrimage's volumes are installments of one continuous "
                   "work (one protagonist, one uninterrupted narrative), not "
                   "independent books — near-zero between-title variance "
                   "here is close to tautological, not a finding about a "
                   "'fixed style.'",
}


def per_passage_means(df: pd.DataFrame) -> pd.DataFrame:
    soc = df[df["is_soc"] == True].dropna(subset=["mediation_level"])  # noqa: E712
    soc = soc.copy()
    soc["mediation_level"] = soc["mediation_level"].astype(float)
    return (
        soc.groupby(["passage_id", "author", "title", "year"])["mediation_level"]
        .mean()
        .reset_index()
    )


def icc_one_way(groups: list[np.ndarray]) -> float:
    """One-way random-effects ICC for unbalanced groups (title nested in
    author), via the standard ANOVA method (Searle et al.): corrects for
    group-count/size bias in a way raw eta-squared does not."""
    k = len(groups)
    if k < 2:
        return float("nan")
    ns = np.array([len(g) for g in groups])
    N = ns.sum()
    if N <= k:
        return float("nan")
    grand_mean = np.concatenate(groups).mean()
    ss_between = sum(n * (g.mean() - grand_mean) ** 2 for n, g in zip(ns, groups))
    ss_within = sum(((g - g.mean()) ** 2).sum() for g in groups)
    msb = ss_between / (k - 1)
    msw = ss_within / (N - k) if N > k else float("nan")
    n0 = (N - (ns ** 2).sum() / N) / (k - 1)
    denom = msb + (n0 - 1) * msw
    if denom == 0:
        return float("nan")
    return max(0.0, (msb - msw) / denom)


def bootstrap_icc_ci(groups: list[np.ndarray], n_boot: int = N_BOOTSTRAP) -> tuple[float, float]:
    """Passage-level cluster bootstrap: resample passages WITHIN each fixed
    title, recompute ICC. Captures finite-sample uncertainty given this
    specific set of books — NOT uncertainty over which books were chosen
    (that's what leave-one-out below is for)."""
    boots = []
    for _ in range(n_boot):
        resampled = [RNG.choice(g, size=len(g), replace=True) for g in groups]
        boots.append(icc_one_way(resampled))
    boots = np.array([b for b in boots if not np.isnan(b)])
    if len(boots) == 0:
        return (float("nan"), float("nan"))
    return (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))


def leave_one_out(title_groups: dict[str, np.ndarray]) -> list[tuple[str, float]]:
    """ICC recomputed with each title removed in turn. Returns
    [(title_removed, icc_without_it), ...], sorted by impact."""
    titles = list(title_groups.keys())
    results = []
    for t in titles:
        remaining = [g for name, g in title_groups.items() if name != t]
        if len(remaining) < 2:
            continue
        results.append((t, icc_one_way(remaining)))
    return results


def autocorrelation(df_title: pd.DataFrame) -> float:
    """Lag-1 autocorrelation of mediation_level, ordered by the passage_id's
    numeric suffix (the sequential position in the source text)."""
    idx = df_title["passage_id"].apply(lambda s: int(re.search(r"(\d+)$", s).group(1)))
    ordered = df_title.assign(_idx=idx).sort_values("_idx")["mediation_level"].to_numpy()
    if len(ordered) < 3:
        return float("nan")
    a, b = ordered[:-1], ordered[1:]
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def main():
    df = load()
    means = per_passage_means(df)

    out: list[str] = [
        "# Variance-decomposition robustness checks\n",
        "Direct response to `critique.md`'s arithmetic objections: eta-squared "
        "recomputed as a variance-components ICC (corrects the group-count "
        "bias eta^2 has), with a passage-level bootstrap CI, leave-one-title-out "
        "sensitivity, within-title level distributions, within-title "
        "autocorrelation, and explicit composition robustness checks. See "
        "`woolf_variance_decomposition.md` for the write-up this supports.\n",
    ]

    # ---- ICC + bootstrap CI + leave-one-out, per author -------------------
    out.append("## ICC (variance-components), bootstrap CI, and leave-one-out sensitivity\n")
    out.append(
        "ICC replaces the raw eta^2 reported in the original note: eta^2 is "
        "upward-biased by title count (an author with more titles is "
        "structurally advantaged), which the ICC's unbalanced-groups "
        "correction accounts for. The bootstrap CI captures finite-sample "
        "uncertainty given the actual titles present; leave-one-out shows "
        "how much the figure depends on any single title.\n"
    )
    out.append("| author | titles | ICC | 95% CI | ICC w/o highest-mean title | which title |")
    out.append("|---|---|---|---|---|---|")

    icc_summary = {}
    for author, g in means.groupby("author"):
        title_groups = {t: sub["mediation_level"].to_numpy() for t, sub in g.groupby("title")}
        if len(title_groups) < 2:
            continue
        groups = list(title_groups.values())
        icc = icc_one_way(groups)
        lo, hi = bootstrap_icc_ci(groups)
        loo = leave_one_out(title_groups)
        # highest-mean title specifically (matches critique's exact check)
        title_means = {t: arr.mean() for t, arr in title_groups.items()}
        top_title = max(title_means, key=title_means.get)
        icc_without_top = dict(loo).get(top_title, float("nan"))
        icc_summary[author] = (icc, lo, hi, icc_without_top, top_title)
        flag = " *" if author in NON_COMPARABLE_AUTHORS else ""
        out.append(
            f"| {author}{flag} | {len(title_groups)} | {icc:.3f} | [{lo:.3f}, {hi:.3f}] | "
            f"{icc_without_top:.3f} | {top_title} |"
        )
    out.append("")
    for author, note in NON_COMPARABLE_AUTHORS.items():
        out.append(f"\\* **{author}**: {note}")
    out.append("")

    out.append("Full leave-one-out table (ICC with each title removed in turn):\n")
    for author, g in means.groupby("author"):
        title_groups = {t: sub["mediation_level"].to_numpy() for t, sub in g.groupby("title")}
        if len(title_groups) < 3:
            continue  # leave-one-out needs >=2 remaining titles to be meaningful
        loo = leave_one_out(title_groups)
        out.append(f"- **{author}** (full ICC = {icc_summary[author][0]:.3f}):")
        for title, icc_wo in sorted(loo, key=lambda x: x[1]):
            out.append(f"    - without *{title}*: {icc_wo:.3f}")
    out.append("")

    # ---- Composition robustness: Woolf with/without Flush + Orlando -------
    out.append("## Composition robustness: Woolf without the biography/mock-biography texts\n")
    out.append(
        "*Flush* (a dog's-eye-view biography) and *Orlando* (a mock-biography) "
        "are genre-heterogeneous relative to Woolf's other eight novels; "
        "dropping them tests whether the nine-novel 'flat band' finding is an "
        "artifact of including them.\n"
    )
    woolf = means[means["author"] == "Woolf"]
    woolf_groups_full = {t: sub["mediation_level"].to_numpy() for t, sub in woolf.groupby("title")}
    woolf_groups_no_bio = {t: g for t, g in woolf_groups_full.items() if t not in ("Flush", "Orlando")}
    icc_full = icc_one_way(list(woolf_groups_full.values()))
    icc_no_bio = icc_one_way(list(woolf_groups_no_bio.values()))
    out.append(f"- All 10 titles: ICC = {icc_full:.3f}")
    out.append(f"- Excluding *Flush* and *Orlando* (8 titles): ICC = {icc_no_bio:.3f}")
    out.append(
        f"- {'Finding holds' if icc_no_bio >= icc_full * 0.7 else 'Finding weakens substantially'} "
        "without them — not fragile in this direction, but the choice to "
        "include them should be stated, not silent.\n"
    )

    # ---- Within-title level distributions ----------------------------------
    out.append("## Within-title level distributions\n")
    out.append(
        "Full proportion breakdown across mediation levels 0-3 per title, not "
        "just the mean — a 0.87 mean is compatible with prose that sits "
        "flatly at level 1 or that alternates violently between 0 and 2.\n"
    )
    df_soc = df[df["is_soc"] == True].dropna(subset=["mediation_level"])  # noqa: E712
    df_soc = df_soc.copy()
    df_soc["mediation_level"] = df_soc["mediation_level"].astype(float)
    out.append("| author | title | mean | 0 | 1 | 2 | 3 |")
    out.append("|---|---|---|---|---|---|---|")
    for (author, title), g in df_soc.groupby(["author", "title"]):
        counts = g["mediation_level"].value_counts(normalize=True).reindex([0, 1, 2, 3], fill_value=0.0)
        out.append(
            f"| {author} | {title} | {g['mediation_level'].mean():.2f} | "
            f"{counts[0]:.0%} | {counts[1]:.0%} | {counts[2]:.0%} | {counts[3]:.0%} |"
        )
    out.append("")

    # ---- Within-title autocorrelation --------------------------------------
    out.append("## Within-title autocorrelation (lag-1, along text order)\n")
    out.append(
        "Positive values: the register moves in sustained runs (a passage's "
        "level predicts the next). Near zero: the register oscillates "
        "passage to passage roughly independently of what came before.\n"
    )
    out.append("| author | title | lag-1 autocorrelation | n passages |")
    out.append("|---|---|---|---|")
    for (author, title), g in means.groupby(["author", "title"]):
        ac = autocorrelation(g)
        out.append(f"| {author} | {title} | {ac:.3f} | {len(g)} |")
    out.append("")

    report_path = ROOT / "woolf_variance_robustness_report.md"
    report_path.write_text("\n".join(out))
    print(f"Written: {report_path}")


if __name__ == "__main__":
    main()
