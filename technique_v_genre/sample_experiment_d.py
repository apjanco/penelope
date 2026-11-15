"""Stratified sample of 50 passages each from the Woolf / Faulkner / James
validation set, evenly spaced by position in the novel (not random) so the
sample covers the whole book rather than clustering in one section.

Output: technique_v_genre/experiment_d_sample.jsonl (150 rows)
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PASSAGES_DIR = ROOT / "passages"
N_PER_WORK = 50

SAMPLE_FILES = [
    "to_the_lighthouse_cleaned",   # Woolf
    "pg75170_sound_and_fury",       # Faulkner
    "pg00432_the_ambassadors",      # James (control)
]


def stratified_sample(rows: list[dict], n: int) -> list[dict]:
    if len(rows) <= n:
        return rows
    step = len(rows) / n
    return [rows[int(i * step)] for i in range(n)]


def main():
    out_rows = []
    for stem in SAMPLE_FILES:
        rows = [json.loads(l) for l in (PASSAGES_DIR / f"{stem}.jsonl").read_text().splitlines()]
        sample = stratified_sample(rows, N_PER_WORK)
        out_rows.extend(sample)
        print(f"{stem}: sampled {len(sample)} / {len(rows)} passages")

    out_path = ROOT / "experiment_d_sample.jsonl"
    out_path.write_text("\n".join(json.dumps(r) for r in out_rows) + "\n")
    print(f"\nWrote {len(out_rows)} passages -> {out_path}")


if __name__ == "__main__":
    main()
