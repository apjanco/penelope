"""Recompute the draft chapter's results-section statistics on the current
(expanded, 44-work) corpus, using the same method as the August reports
(pilot_report.md Exp A/C, woolf_variance_robustness.py), plus per-reader splits.

The Exp C Spearman here uses scipy's asymptotic p; for the antimimetic trend use
antimimetic_trend.py (permutation p, per-reader and pre-1900 sensitivity).

Usage (della-vis2: activate the env by its /scratch path or python hangs):
  conda activate /scratch/gpfs/MM4/.conda-aj7878/envs/penelope
  python recompute_chapter_stats.py
"""
import json, re, sys, collections
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

TV = Path("/scratch/gpfs/MM4/apjanco/penelope/technique_v_genre")
RNG = np.random.default_rng(20260807)
N_BOOTSTRAP = 2000

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

SKIP = {"time_passes.jsonl", "qwen3_reasoning_smoketest.jsonl"}
READERS = {"gpt-5.4", "qwen3-max", "qwen3.6-local"}
rows = []
for f in sorted((TV / "scored").glob("*.jsonl")):
    if f.name in SKIP:
        continue
    for line in f.open():
        r = json.loads(line)
        if r.get("reader_id") not in READERS or r.get("error"):
            continue
        rows.append({k: r.get(k) for k in ("passage_id", "author", "title", "year", "reader_id", "run",
                                            "is_soc", "mediation_level", "soc_type", "fit_category", "antimimetic")})
df = pd.DataFrame(rows)
df["title"] = df["title"].str.replace(r"\s*\(vol\. \d\)", "", regex=True)
print("works", df.title.nunique(), "passages", df.passage_id.nunique(), "rows", len(df))
print(df.reader_id.value_counts().to_dict())
loc = df[df.reader_id == "qwen3.6-local"]
print("local-substitute passages", loc.passage_id.nunique(), loc.groupby("title").passage_id.nunique().to_dict())

soc = df[(df.is_soc == True) & df.mediation_level.notna()].copy()  # noqa
soc["mediation_level"] = soc.mediation_level.astype(float)

def prof(g):
    v = g.mediation_level
    p = v.value_counts(normalize=True).reindex([0, 1, 2, 3], fill_value=0) * 100
    return pd.Series({"n": len(v), "mean": v.mean(), "sd": v.std(), "L0": p[0], "L1": p[1], "L2": p[2], "L3": p[3]})

pd.set_option("display.width", 200); pd.set_option("display.float_format", "{:.2f}".format)
print("\n== Exp A pooled (gpt+qwen+local), per judgment, by author")
print(soc.groupby("author").apply(prof))
for rd in ("gpt-5.4", "qwen3-max"):
    print(f"\n== Exp A {rd} only, by author")
    print(soc[soc.reader_id == rd].groupby("author").apply(prof))
print("\n== by title pooled")
bt = soc.groupby(["author", "title", "year"]).apply(prof).reset_index().sort_values(["author", "year"])
print(bt.to_string(index=False))
for rd in ("gpt-5.4", "qwen3-max"):
    print(f"\n== Woolf by title {rd}")
    print(soc[(soc.reader_id == rd) & (soc.author == "Woolf")].groupby(["title", "year"]).apply(prof).reset_index().sort_values("year").to_string(index=False))

# per-passage mean across readers/runs (Aug method)
pp = soc.groupby(["passage_id", "author", "title", "year"]).mediation_level.mean().reset_index()

def icc_block(pp, label):
    print(f"\n== ICC ({label})")
    for a, g in pp.groupby("author"):
        tg = {t: gg.mediation_level.to_numpy() for t, gg in g.groupby("title")}
        icc = icc_one_way(list(tg.values()))
        lo, hi = bootstrap_icc_ci(list(tg.values()), n_boot=N_BOOTSTRAP)
        top = g.groupby("title").mediation_level.mean().idxmax()
        loo = dict(leave_one_out(tg))
        print(f"{a:11s} titles={len(tg):2d} ICC={icc:.3f} [{lo:.3f},{hi:.3f}] w/o {top}: {loo.get(top, float('nan')):.3f}")
        if a in ("Woolf", "Joyce", "Sinclair", "James", "Ford", "Faulkner", "Richardson"):
            print("    LOO:", {k: round(v, 3) for k, v in sorted(loo.items(), key=lambda x: x[1])})
    w = pp[pp.author == "Woolf"]
    tg = {t: gg.mediation_level.to_numpy() for t, gg in w.groupby("title") if t not in ("Flush", "Orlando")}
    print("Woolf w/o Flush+Orlando:", round(icc_one_way(list(tg.values())), 3))

icc_block(pp, "pooled per-passage mean, gpt+qwen(+local)")
for rd in ("gpt-5.4", "qwen3-max"):
    ppr = soc[soc.reader_id == rd].groupby(["passage_id", "author", "title", "year"]).mediation_level.mean().reset_index()
    icc_block(ppr, rd)

print("\n== per-passage title distributions + lag-1 autocorr (pooled per-passage mean, rounded for dist)")
out = []
for (a, t), g in pp.groupby(["author", "title"]):
    lv = g.mediation_level.round().clip(0, 3)
    p = lv.value_counts(normalize=True).reindex([0, 1, 2, 3], fill_value=0) * 100
    out.append((a, t, int(g.year.iloc[0]), len(g), g.mediation_level.mean(), *p.values, autocorrelation(g)))
print(pd.DataFrame(out, columns=["author", "title", "year", "n", "mean", "L0", "L1", "L2", "L3", "ac1"]).sort_values(["author", "year"]).to_string(index=False))

print("\n== Exp C: year vs no_type_fits / antimimetic rate by title (gpt+qwen rows w/ fit_category)")
fc = df[df.fit_category.notna()]
c = fc.groupby(["title", "year"]).agg(ntf=("fit_category", lambda s: (s == "no_type_fits").mean()),
                                      anti=("antimimetic", lambda s: (s == True).mean()), n=("fit_category", "size")).reset_index()  # noqa
print(c.sort_values("anti", ascending=False).head(12).to_string(index=False))
print("spearman year~no_type_fits", stats.spearmanr(c.year, c.ntf), "n titles", len(c))
print("spearman year~antimimetic", stats.spearmanr(c.year, c.anti))
print("antimimetic total rows", int((fc.antimimetic == True).sum()), "of", len(fc))  # noqa
