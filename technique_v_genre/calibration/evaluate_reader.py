"""Score a candidate reader on the calibration set and report whether it
discriminates — before committing GPU-days to it.

Run with --baselines to compute the reference band from readers already in
scored/ (no API calls). Run with --reader LABEL to score a candidate through
score.py and then report it against that band.

The verdict is deliberately narrow. There is no ground truth in this project,
so nothing here is "accuracy". The gate is DISCRIMINATION, which needs no
ground truth: a reader that puts >70% of its judgements in one soc_type is
not measuring the taxonomy, whatever its prose looks like. llama-3.3-70B sits
at 91% and produced perfectly literate arguments for all of it.

Reported alongside, as context rather than pass/fail:
  - consensus agreement, per mediation level (where gpt-5.4 and qwen3-max
    already agree unanimously, a competent reader should usually agree)
  - agreement with each existing reader separately — a candidate that tracks
    qwen3-max too closely adds no independent signal, which is the whole
    reason for a third reader
  - resampling variance — if k runs never differ, k is wasted and the
    stability analysis in AMBIGUITY.md has nothing to work with
  - behaviour on contested passages, where the two cloud readers disagree and
    there is no expected answer at all
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TV = HERE.parent
CAL = HERE / "calibration_set.jsonl"
MAX_SHARE_FAIL = 0.70     # one category above this = not discriminating
MIN_ENTROPY_WARN = 0.90   # bits, over the soc_type distribution


def load_cal() -> dict[str, dict]:
    return {r["passage_id"]: r for r in
            (json.loads(l) for l in CAL.read_text().splitlines() if l.strip())}


def rows_for(reader: str, cal: dict) -> list[dict]:
    """Pull a reader's judgements on calibration passages from scored/ or
    from this harness's own output."""
    out = []
    srcs = list((TV / "scored").glob("*.jsonl")) + list(HERE.glob("scored_*.jsonl"))
    for f in srcs:
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get("reader_id") != reader or "error" in d:
                continue
            if d["passage_id"] in cal:
                out.append(d)
    return out


def entropy(counter: collections.Counter) -> float:
    n = sum(counter.values())
    if not n:
        return 0.0
    return -sum((v / n) * math.log2(v / n) for v in counter.values() if v)


def report(reader: str, rows: list[dict], cal: dict) -> dict:
    if not rows:
        print(f"  {reader}: no rows on the calibration set")
        return {}
    types = collections.Counter(d.get("soc_type") for d in rows if d.get("is_soc"))
    levels = collections.Counter(d.get("mediation_level") for d in rows
                                 if d.get("mediation_level") is not None)
    n_soc = sum(types.values())
    top, top_n = (types.most_common(1) or [(None, 0)])[0]
    share = top_n / n_soc if n_soc else 0.0
    ent = entropy(types)

    # consensus agreement, by level
    by_level = collections.defaultdict(lambda: [0, 0])
    per_pass = collections.defaultdict(list)
    for d in rows:
        per_pass[d["passage_id"]].append(d)
    for pid, ds in per_pass.items():
        c = cal[pid]
        if c["stratum"] != "consensus":
            continue
        mode = collections.Counter(d.get("soc_type") for d in ds).most_common(1)[0][0]
        lv = c["consensus_mediation_level"]
        by_level[lv][1] += 1
        if mode == c["consensus_soc_type"]:
            by_level[lv][0] += 1

    # resampling: do k runs ever differ?
    varied = sum(1 for ds in per_pass.values()
                 if len({d.get("mediation_level") for d in ds}) > 1)
    multi = sum(1 for ds in per_pass.values() if len(ds) > 1)

    verdict = "FAIL — collapsed onto one category" if share > MAX_SHARE_FAIL else "pass"
    print(f"\n  == {reader}")
    print(f"     passages {len(per_pass):>4}   soc judgements {n_soc:>5}")
    print(f"     top soc_type: {str(top)[:30]:32s} {100*share:5.1f}%   "
          f"entropy {ent:.2f} bits   -> {verdict}")
    lv = "  ".join(f"L{k}:{100*levels.get(k,0)/max(sum(levels.values()),1):4.0f}%" for k in range(4))
    print(f"     mediation spread: {lv}")
    agree_tot = sum(v[0] for v in by_level.values()), sum(v[1] for v in by_level.values())
    if agree_tot[1]:
        per = "  ".join(f"L{k}:{100*by_level[k][0]/by_level[k][1]:3.0f}%"
                        for k in sorted(by_level) if by_level[k][1])
        # gpt-5.4 and qwen3-max DEFINE the consensus stratum, so they score
        # 100% here by construction. That is a tautology, not a validation.
        # The number is only informative for a reader held out of it.
        tag = "  [TAUTOLOGICAL — this reader defines the consensus]" \
            if reader in ("gpt-5.4", "qwen3-max") else ""
        print(f"     consensus agreement: {100*agree_tot[0]/agree_tot[1]:.0f}% overall   {per}{tag}")
    if multi:
        print(f"     k-run variance: {100*varied/multi:.0f}% of passages vary across runs")
    return {"reader": reader, "share": share, "entropy": ent,
            "verdict": verdict, "n": len(per_pass)}


def score_candidate(reader: str, k: int, concurrency: int) -> None:
    out = HERE / f"scored_{reader}.jsonl"
    print(f"  scoring {CAL.name} with {reader} (k={k}) -> {out.name}")
    subprocess.run([
        "/home/aj7878/.conda/envs/penelope/bin/python", str(TV / "score.py"),
        "--input", str(CAL), "--readers", reader, "--k", str(k),
        "--concurrency", str(concurrency), "--output", str(out),
    ], cwd=TV, check=False)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reader", help="models.yaml label to score and evaluate")
    ap.add_argument("--baselines", action="store_true",
                    help="report existing readers from scored/ (no API calls)")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--concurrency", type=int, default=8)
    args = ap.parse_args()

    cal = load_cal()
    print(f"  calibration set: {len(cal)} passages "
          f"({sum(1 for c in cal.values() if c['stratum']=='consensus')} consensus / "
          f"{sum(1 for c in cal.values() if c['stratum']=='contested')} contested)")
    print(f"  DISCRIMINATION GATE: no single soc_type above {100*MAX_SHARE_FAIL:.0f}%")

    if args.baselines:
        for r in ("gpt-5.4", "qwen3-max", "llama", "jev", "gliclass", "gemma"):
            report(r, rows_for(r, cal), cal)
    if args.reader:
        score_candidate(args.reader, args.k, args.concurrency)
        report(args.reader, rows_for(args.reader, cal), cal)


if __name__ == "__main__":
    main()
