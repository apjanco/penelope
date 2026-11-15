"""Does mediation level on Cohn's scale drift with publication year, 1875-1941?

The chapter's existing "no diachronic trend" figure (rho = 0.02, p = 0.91,
recompute_chapter_stats.py Exp C) correlates year with the *no_type_fits* rate,
i.e. taxonomy-resistant prose. It is not a test of the scale itself. This script
tests the scale directly, at the title level, for each reader:

  1. corpus-wide: Spearman(year, title mean level), plus year vs share of
     passages at level 0 and at levels 2-3; permutation p-values.
  2. within-author: year and level both centred on each author's mean, so the
     test asks whether authors move toward less mediation *across their own
     careers*; permutation shuffles years within author only.
  3. between-author: author mean year vs author mean level (n = 7, descriptive).
  4. sensitivity: without The Waves; without James and Ford (the non-SOC
     comparison set, which sits early and mediated and could carry a trend alone);
     only titles from 1910 on.

Title-level means are per-judgment means over is_soc passages, as in
recompute_chapter_stats.py; vol. 1/2 folded into one title.

Usage (della-vis2: activate the env by its /scratch path or python hangs):
  conda activate /scratch/gpfs/MM4/.conda-aj7878/envs/penelope
  python mediation_trend.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

TV = Path(__file__).resolve().parent
SKIP = {"time_passes.jsonl", "qwen3_reasoning_smoketest.jsonl"}
RNG = np.random.default_rng(20261018)
N_PERM = 20_000


def load() -> pd.DataFrame:
    rows = []
    for sdir, readers in ((TV / "scored", {"gpt-5.4", "qwen3-max", "qwen3.6-local"}),
                          (TV / "scored_jev", {"jev"})):
        for f in sorted(sdir.glob("*.jsonl")):
            if f.name in SKIP:
                continue
            for line in f.open():
                r = json.loads(line)
                if r.get("reader_id") not in readers or r.get("error"):
                    continue
                rows.append({k: r.get(k) for k in ("passage_id", "author", "title", "year",
                                                    "reader_id", "is_soc", "mediation_level")})
    df = pd.DataFrame(rows)
    df["title"] = df["title"].str.replace(r"\s*\(vol\. \d\)", "", regex=True)
    df = df[(df.is_soc == True) & df.mediation_level.notna()].copy()  # noqa: E712
    df["mediation_level"] = df.mediation_level.astype(float)
    return df


def titles(d: pd.DataFrame) -> pd.DataFrame:
    g = d.groupby(["author", "title"])
    return pd.DataFrame({
        "year": g.year.min(),
        "n": g.size(),
        "mean": g.mediation_level.mean(),
        "L0": g.mediation_level.apply(lambda v: (v == 0).mean()),
        "L23": g.mediation_level.apply(lambda v: (v >= 2).mean()),
    }).reset_index()


def perm_p(x, y, groups=None) -> float:
    """Two-sided permutation p for Spearman rho; shuffle x within groups if given."""
    obs = stats.spearmanr(x, y).statistic
    x = np.asarray(x, float)
    idx = [np.flatnonzero(groups == g) for g in np.unique(groups)] if groups is not None else [np.arange(len(x))]
    hits = 0
    for _ in range(N_PERM):
        xp = x.copy()
        for ix in idx:
            xp[ix] = RNG.permutation(x[ix])
        hits += abs(stats.spearmanr(xp, y).statistic) >= abs(obs) - 1e-12
    return (hits + 1) / (N_PERM + 1)


def report(label: str, t: pd.DataFrame) -> None:
    out = []
    for col in ("mean", "L0", "L23"):
        rho = stats.spearmanr(t.year, t[col]).statistic
        out.append(f"{col} rho={rho:+.2f} p={perm_p(t.year.values, t[col].values):.3f}")
    # within-author: centre both on author means; shuffle years within author
    multi = t.groupby("author").title.transform("size") >= 3
    w = t[multi].copy()
    w["yc"] = w.year - w.groupby("author").year.transform("mean")
    w["mc"] = w["mean"] - w.groupby("author")["mean"].transform("mean")
    rho_w = stats.spearmanr(w.yc, w.mc).statistic
    p_w = perm_p(w.yc.values, w.mc.values, groups=w.author.values)
    out.append(f"within-author rho={rho_w:+.2f} p={p_w:.3f} (authors>=3 titles: {w.author.nunique()})")
    print(f"  {label:<34} n={len(t):>2} | " + " | ".join(out))


def main() -> None:
    df = load()
    readers = {
        "pooled (gpt+qwen+local)": df[df.reader_id != "jev"],
        "gpt-5.4": df[df.reader_id == "gpt-5.4"],
        "qwen3-max": df[df.reader_id == "qwen3-max"],
        "jev (all 54 works)": df[df.reader_id == "jev"],
    }
    for name, d in readers.items():
        t = titles(d)
        print(f"\n== {name}: {len(t)} titles, {t.year.min()}-{t.year.max()}")
        report("all titles", t)
        report("without The Waves", t[t.title != "The Waves"])
        report("without James and Ford", t[~t.author.isin(["James", "Ford"])])
        report("1910 onward", t[t.year >= 1910])
        a = t.groupby("author").agg(year=("year", "mean"), mean=("mean", "mean"), k=("title", "size"))
        print("  between-author (descriptive, n=7):",
              f"rho={stats.spearmanr(a.year, a['mean']).statistic:+.2f}  ",
              ", ".join(f"{i} {r.year:.0f}/{r['mean']:.2f}" for i, r in a.sort_values("year").iterrows()))
        if name.startswith("jev"):
            print("  per-author Spearman(year, mean) on jev:")
            for au, g in t.groupby("author"):
                if len(g) >= 3:
                    print(f"    {au:<11} k={len(g):>2} rho={stats.spearmanr(g.year, g['mean']).statistic:+.2f}")


if __name__ == "__main__":
    main()
