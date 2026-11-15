"""Export tidy CSVs for the chapter figures.

One export step so every figure reads the same numbers, and so the figures can
be rebuilt without re-reading 25k-row scored files.

Three deliberate choices baked in here:

1. Rows are per (work x reader), never pooled across readers. The mediation
   scale is NOT calibrated between readers — gpt-5.4 puts 52% of judgements at
   level 0 where llama puts 2.6% — so a pooled mean is an average over
   instruments, not a measurement. Figures pick a reader and say which.

2. llama is excluded. It failed the calibration gate (79.7% of judgements in one
   soc_type, entropy 0.93 bits) and assigns level 1 to 90.7% of passages, so its
   per-work means hover at 1.00 regardless of the text.

3. `n_works` travels with every author row. Faulkner has 3 works and Woolf 10,
   and range is an extreme-value statistic — the reader has to see n to judge a
   spread claim.
"""
from __future__ import annotations

import collections
import csv
import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
TV = ROOT / "technique_v_genre"
OUT = Path(__file__).resolve().parent.parent / "data"

READERS = ("gpt-5.4", "qwen3-max", "jev")   # llama excluded, see docstring
MIN_PASSAGES = 100


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    man = {e["source_file"][:-4]: e for e in
           json.loads((TV / "corpus_manifest.json").read_text())["corpus"]}

    levels = collections.defaultdict(lambda: collections.defaultdict(list))
    amb_gap = []
    for d in (TV / "scored", TV / "scored_jev"):
        for f in sorted(d.glob("*.jsonl")):
            for line in f.read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "error" in row or not row.get("is_soc"):
                    continue
                stem = row.get("source_file", "")[:-4]
                r = row.get("reader_id")
                if r in READERS and row.get("mediation_level") is not None:
                    levels[stem][r].append(row["mediation_level"])
                probs = row.get("jev_mediation_probs")
                if probs:
                    v = sorted((float(x) for x in probs.values()), reverse=True)
                    if len(v) > 1:
                        amb_gap.append((stem, row["passage_id"], v[0] - v[1]))

    # ---- per work x reader -------------------------------------------------
    with (OUT / "work_mediation.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["author", "title", "year", "role", "source_file", "reader",
                    "n_passages", "mean_mediation", "sd_mediation",
                    "pct_L0", "pct_L1", "pct_L2", "pct_L3"])
        for stem, byr in sorted(levels.items()):
            e = man.get(stem)
            if not e:
                continue
            for r in READERS:
                v = byr.get(r, [])
                if len(v) < MIN_PASSAGES:
                    continue
                c = collections.Counter(v)
                w.writerow([e["author"], e["title"], e["year"], e["role"], stem, r,
                            len(v), round(st.mean(v), 4),
                            round(st.pstdev(v), 4) if len(v) > 1 else 0,
                            *[round(100 * c.get(k, 0) / len(v), 2) for k in range(4)]])

    # ---- per author x reader ---------------------------------------------
    rows = list(csv.DictReader((OUT / "work_mediation.csv").open()))
    with (OUT / "author_spread.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["author", "reader", "n_works", "min", "max", "range",
                    "sd", "iqr", "median", "mean", "first_year", "last_year"])
        grp = collections.defaultdict(list)
        for r in rows:
            grp[(r["author"], r["reader"])].append(r)
        for (a, rd), rs in sorted(grp.items()):
            v = sorted(float(x["mean_mediation"]) for x in rs)
            ys = [int(x["year"]) for x in rs]
            if len(v) < 2:
                continue
            q1, q3 = v[len(v) // 4], v[(3 * len(v)) // 4]
            w.writerow([a, rd, len(v), round(v[0], 4), round(v[-1], 4),
                        round(v[-1] - v[0], 4), round(st.pstdev(v), 4),
                        round(q3 - q1, 4), round(st.median(v), 4),
                        round(st.mean(v), 4), min(ys), max(ys)])

    # ---- Jev's direct ambiguity measure ----------------------------------
    with (OUT / "ambiguity_gap.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["source_file", "author", "passage_id", "top2_prob_gap"])
        for stem, pid, gap in amb_gap:
            e = man.get(stem)
            w.writerow([stem, e["author"] if e else "", pid, round(gap, 4)])

    print(f"  work_mediation.csv  {len(rows)} rows")
    print(f"  author_spread.csv   {sum(1 for _ in csv.DictReader((OUT/'author_spread.csv').open()))-0} rows")
    print(f"  ambiguity_gap.csv   {len(amb_gap):,} passages")


if __name__ == "__main__":
    main()
