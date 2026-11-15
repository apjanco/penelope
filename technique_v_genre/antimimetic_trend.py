"""Does the rate of antimimetic flags rise with publication year?

Title-level Spearman correlation between year and the share of judgements
flagged antimimetic, over the main corpus (gpt-5.4 + qwen3-max, k=3), with an
exact-by-permutation p-value (many titles sit at a rate of exactly zero, so
the t/normal approximation to Spearman's null is unreliable here).
Sensitivity checks: each reader separately; excluding pre-1900 titles (the
zero-rate James/Sinclair works that could carry the correlation alone);
no_type_fits rate as the comparison measure.

Usage: python antimimetic_trend.py
"""
from __future__ import annotations

import collections
import json
import re
from pathlib import Path

import numpy as np

TV = Path(__file__).resolve().parent
SKIP = {"time_passes.jsonl", "qwen3_reasoning_smoketest.jsonl"}
READERS = ("gpt-5.4", "qwen3-max")
N_PERM = 100_000
RNG = np.random.default_rng(20261012)


def load():
    # (title, year, reader) -> [n_rows, n_antimimetic, n_no_type_fits]
    c = collections.defaultdict(lambda: [0, 0, 0])
    for f in sorted((TV / "scored").glob("*.jsonl")):
        if f.name in SKIP:
            continue
        for line in f.open():
            r = json.loads(line)
            if r.get("reader_id") not in READERS or r.get("error") or r.get("fit_category") is None:
                continue
            title = re.sub(r"\s*\(vol\. \d\)", "", r["title"])
            k = c[(title, r["year"], r["reader_id"])]
            k[0] += 1
            k[1] += r.get("antimimetic") is True
            k[2] += r.get("fit_category") == "no_type_fits"
    return c


def spearman(x, y):
    rx = np.argsort(np.argsort(x)).astype(float)
    # average ranks for ties
    def avg_rank(v):
        order = np.argsort(v, kind="mergesort")
        ranks = np.empty(len(v))
        sv = v[order]
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and sv[j + 1] == sv[i]:
                j += 1
            ranks[order[i:j + 1]] = (i + j) / 2
            i = j + 1
        return ranks
    rx, ry = avg_rank(np.asarray(x, float)), avg_rank(np.asarray(y, float))
    return float(np.corrcoef(rx, ry)[0, 1]), rx, ry


def perm_test(x, y):
    rho, rx, ry = spearman(x, y)
    rx = (rx - rx.mean()) / rx.std()
    ry = (ry - ry.mean()) / ry.std()
    perms = np.array([np.mean(rx * RNG.permutation(ry)) for _ in range(N_PERM)])
    p = (np.sum(np.abs(perms) >= abs(rho) - 1e-12) + 1) / (N_PERM + 1)
    return rho, p


def report(c, readers, measure, min_year=0, label=""):
    agg = collections.defaultdict(lambda: [0, 0, 0])
    for (t, y, rd), v in c.items():
        if rd in readers and y >= min_year:
            a = agg[(t, y)]
            for i in range(3):
                a[i] += v[i]
    keys = sorted(agg)
    years = [y for _, y in keys]
    idx = 1 if measure == "antimimetic" else 2
    rates = [agg[k][idx] / agg[k][0] for k in keys]
    rho, p = perm_test(years, rates)
    zero = sum(r == 0 for r in rates)
    print(f"  {label:38s} titles={len(keys):2d} (zero-rate {zero:2d})  rho={rho:+.3f}  permutation p={p:.4f}")


def main():
    c = load()
    for measure in ("antimimetic", "no_type_fits"):
        print(f"\n{measure} rate vs publication year")
        report(c, READERS, measure, label="both readers, all titles")
        report(c, READERS, measure, min_year=1900, label="both readers, 1900 onward")
        for rd in READERS:
            report(c, (rd,), measure, label=f"{rd} only, all titles")
            report(c, (rd,), measure, min_year=1900, label=f"{rd} only, 1900 onward")
    tot = collections.Counter()
    for (t, y, rd), v in c.items():
        tot[(t, rd)] += v[1]
    print("\nantimimetic flags by title and reader (nonzero):")
    for (t, rd), n in sorted(tot.items(), key=lambda kv: -kv[1]):
        if n:
            print(f"  {t:42s} {rd:10s} {n}")


if __name__ == "__main__":
    main()
