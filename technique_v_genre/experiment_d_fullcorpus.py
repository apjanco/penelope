"""Experiment D at corpus scale.

The original validation gate (analyze_experiment_d.py) correlated mediation
level with independently counted surface features on 97 co-detected
passages from a 150-passage sample of the 21-title corpus. This reruns the
same features, unchanged, on every passage of the current main corpus that
both gpt-5.4 and qwen3-max identify as rendering consciousness — using the
k=3 judgements already in scored/, so no new API calls.

Three things the small sample could not do, and this does:

1. Per reader. The scale is not calibrated between readers, so each
   reader's modal level is correlated separately, alongside the two-reader
   mean the original used. Jev (scored_jev/, asked the level question
   independently of the type question) is reported as a third check.
2. Within author. At corpus scale a pooled correlation can be carried by
   between-author differences alone (Faulkner is both first-person-heavy
   and high-level). Each feature is also correlated after removing each
   author's mean from both variables, so the estimate reflects what the
   feature predicts inside one author's prose.
3. Disagreement. The original found no feature predicting cross-reader
   spread at n=97; with ~17k passages a null is far more informative.

Writes experiment_d_fullcorpus_report.md.
"""
from __future__ import annotations

import collections
import json
from pathlib import Path

import numpy as np
from scipy import stats

from analyze_experiment_d import surface_features

TV = Path(__file__).resolve().parent
OUT = TV / "experiment_d_fullcorpus_report.md"
READERS = ("gpt-5.4", "qwen3-max")
FEATURES = [
    ("reporting_verb_density", "-"),
    ("first_person_ratio", "+"),
    ("past_tense_ratio", "-"),
    ("dash_ellipsis_density", "+ (original prediction; reversed at n=97)"),
    ("semicolon_density", "-"),
]


def main_corpus_passages() -> dict[str, dict]:
    man = json.loads((TV / "corpus_manifest.json").read_text())["corpus"]
    cop = {l.strip() for l in (TV / "jev_new_works.txt").read_text().splitlines() if l.strip()}
    out = {}
    for e in man:
        stem = e["source_file"].rsplit(".", 1)[0]
        if stem in cop:
            continue
        for line in (TV / "passages" / f"{stem}.jsonl").read_text().splitlines():
            if line.strip():
                p = json.loads(line)
                out[p["passage_id"]] = {"author": e["author"], "text": p["passage_text"]}
    return out


def load(dirname: str, readers) -> dict:
    by = collections.defaultdict(lambda: collections.defaultdict(dict))
    for f in sorted((TV / dirname).glob("*.jsonl")):
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "error" in r or r.get("reader_id") not in readers:
                continue
            by[r["passage_id"]][r["reader_id"]][int(r.get("run", 0))] = r
    return by


def modal(runs: dict, field: str):
    c = collections.Counter(r.get(field) for r in runs.values() if r.get(field) is not None)
    return c.most_common(1)[0][0] if c else None


def demean(x: np.ndarray, groups: np.ndarray) -> np.ndarray:
    out = x.astype(float).copy()
    for g in np.unique(groups):
        m = groups == g
        out[m] -= np.nanmean(out[m])
    return out


def corr(x, y, groups=None):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y = x[ok], y[ok]
    if groups is not None:
        g = np.asarray(groups)[ok]
        x, y = demean(x, g), demean(y, g)
    rho, p = stats.spearmanr(x, y)
    return rho, p, int(ok.sum())


def fmt(rho, p, n):
    ps = "<0.001" if p < 0.001 else f"{p:.3f}"
    return f"{rho:+.3f} | {ps} | {n:,}"


def main() -> None:
    P = main_corpus_passages()
    S = load("scored", set(READERS))
    J = load("scored_jev", {"jev"})

    recs = []
    for pid, meta in P.items():
        d = S.get(pid, {})
        if not all(rd in d and modal(d[rd], "is_soc") for rd in READERS):
            continue
        lv = {rd: modal(d[rd], "mediation_level") for rd in READERS}
        if None in lv.values():
            continue
        jr = J.get(pid, {}).get("jev", {})
        jl = modal(jr, "mediation_level") if jr and modal(jr, "is_soc") else None
        recs.append({
            "author": meta["author"],
            **{f"lvl_{rd}": lv[rd] for rd in READERS},
            "lvl_mean": np.mean(list(lv.values())),
            "spread": abs(lv[READERS[0]] - lv[READERS[1]]),
            "lvl_jev": np.nan if jl is None else jl,
            **surface_features(meta["text"]),
        })

    authors = np.array([r["author"] for r in recs])
    col = lambda k: [r[k] for r in recs]
    L = []
    L.append("# Experiment D at corpus scale\n")
    L.append(f"Passages in the main corpus: {len(P):,}. Co-detected as rendering "
             f"consciousness by both gpt-5.4 and qwen3-max (modal is_soc over k=3): "
             f"**{len(recs):,}**. Features are `analyze_experiment_d.surface_features`, "
             "unchanged. Spearman ρ throughout; *within-author* = both variables "
             "demeaned by author first.\n")

    L.append("## Feature vs. level\n")
    L.append("| feature | predicted | target | pooled ρ | p | n | within-author ρ | p | n |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for feat, sign in FEATURES:
        for tgt, lab in (("lvl_mean", "two-reader mean"), ("lvl_gpt-5.4", "gpt-5.4"),
                         ("lvl_qwen3-max", "qwen3-max"), ("lvl_jev", "jev")):
            L.append(f"| {feat} | {sign} | {lab} | {fmt(*corr(col(feat), col(tgt)))} | "
                     f"{fmt(*corr(col(feat), col(tgt), authors))} |")

    L.append("\n## Feature vs. cross-reader disagreement (|gpt-5.4 − qwen3-max|)\n")
    L.append("| feature | pooled ρ | p | n | within-author ρ | p | n |")
    L.append("|---|---|---|---|---|---|---|")
    for feat, _ in FEATURES:
        L.append(f"| {feat} | {fmt(*corr(col(feat), col('spread')))} | "
                 f"{fmt(*corr(col(feat), col('spread'), authors))} |")

    L.append("\n## By author (two-reader mean level)\n")
    L.append("| author | n | mean level | mean spread | ρ first-person | ρ past-tense |")
    L.append("|---|---|---|---|---|---|")
    for a in sorted(set(authors)):
        sub = [r for r in recs if r["author"] == a]
        fp = corr([r["first_person_ratio"] for r in sub], [r["lvl_mean"] for r in sub])
        pt = corr([r["past_tense_ratio"] for r in sub], [r["lvl_mean"] for r in sub])
        L.append(f"| {a} | {len(sub):,} | {np.mean([r['lvl_mean'] for r in sub]):.2f} | "
                 f"{np.mean([r['spread'] for r in sub]):.2f} | {fp[0]:+.2f} | {pt[0]:+.2f} |")

    OUT.write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
