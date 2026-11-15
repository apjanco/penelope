"""Is a local reader a valid stand-in for a cloud reader?

The discrimination gate in evaluate_reader.py asks whether a candidate uses
the taxonomy at all. That is necessary but not sufficient here, because the
local reader has a different job from the ones before it.

Seven of the eight coauthor-supplied texts are in US copyright, so they
cannot be sent to OpenAI, DashScope or TypeSafe. A locally-served model
keeps them inside Princeton. But if those works are read by an instrument
nobody has calibrated, their scores are not on the same scale as the other
46 works, and no cross-work comparison — which is the entire chapter — can
include them.

So the local reader has to clear a second bar: on the PUBLIC-DOMAIN works,
where both it and the cloud readers can be run, does it track qwen3-max
closely enough to serve as its licensed stand-in?

The yardstick is not perfection. It is the agreement the cloud readers
already achieve with each other: gpt-5.4 and qwen3-max agree on soc_type
about 56% of the time and on mediation_level about 67%. A local reader that
reaches qwen3-max at or above that rate is as close to it as the two cloud
readers are to each other, which is the strongest claim the data can
support. Falling well below it means the in-copyright scores sit on their
own scale and must be reported separately rather than pooled.

Usage:  python calibration/proxy_validity.py --candidate qwen3.6-local
"""
from __future__ import annotations

import argparse
import collections
import json
import math
from pathlib import Path

TV = Path(__file__).resolve().parent.parent


def load(readers: set[str]) -> dict[str, dict[str, list]]:
    per: dict[str, dict[str, list]] = collections.defaultdict(lambda: collections.defaultdict(list))
    for d in (TV / "scored", TV / "scored_jev", TV / "scored_local"):
        if not d.exists():
            continue
        for f in d.glob("*.jsonl"):
            for line in f.read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "error" in r or r.get("reader_id") not in readers:
                    continue
                per[r["passage_id"]][r["reader_id"]].append(r)
    return per


def mode(rows: list, field: str):
    c = collections.Counter(x.get(field) for x in rows if x.get(field) is not None)
    return c.most_common(1)[0][0] if c else None


def agree(per, a: str, b: str) -> tuple[int, float, float, float]:
    st = ml = adj = n = 0
    for _pid, rv in per.items():
        if a not in rv or b not in rv:
            continue
        n += 1
        if mode(rv[a], "soc_type") == mode(rv[b], "soc_type"):
            st += 1
        la, lb = mode(rv[a], "mediation_level"), mode(rv[b], "mediation_level")
        if la is not None and lb is not None:
            if la == lb:
                ml += 1
            if abs(la - lb) <= 1:
                adj += 1
    if not n:
        return 0, 0.0, 0.0, 0.0
    return n, 100 * st / n, 100 * ml / n, 100 * adj / n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--target", default="qwen3-max",
                    help="the cloud reader the candidate is standing in for")
    args = ap.parse_args()

    cand, tgt = args.candidate, args.target
    per = load({cand, tgt, "gpt-5.4", "jev"})

    n_ref, st_ref, ml_ref, adj_ref = agree(per, "gpt-5.4", tgt)
    print(f"\n  REFERENCE — how well the two cloud readers agree (the realistic ceiling)")
    print(f"    gpt-5.4 vs {tgt:14s} n={n_ref:>6,}  soc_type {st_ref:4.1f}%  "
          f"level {ml_ref:4.1f}%  within-1 {adj_ref:4.1f}%")

    n, st, ml, adj = agree(per, cand, tgt)
    if not n:
        print(f"\n  {cand} has no overlapping passages with {tgt} yet — score it first.")
        return
    print(f"\n  CANDIDATE as a stand-in for {tgt}")
    print(f"    {cand:14s} vs {tgt:14s} n={n:>6,}  soc_type {st:4.1f}%  "
          f"level {ml:4.1f}%  within-1 {adj:4.1f}%")
    n2, st2, ml2, adj2 = agree(per, cand, "gpt-5.4")
    if n2:
        print(f"    {cand:14s} vs {'gpt-5.4':14s} n={n2:>6,}  soc_type {st2:4.1f}%  "
              f"level {ml2:4.1f}%  within-1 {adj2:4.1f}%")

    print(f"\n  VERDICT")
    for label, got, ref in (("soc_type", st, st_ref), ("mediation level", ml, ml_ref),
                            ("within-1 level", adj, adj_ref)):
        delta = got - ref
        mark = "PASS" if delta >= -5 else ("MARGINAL" if delta >= -12 else "FAIL")
        print(f"    {label:16s} {got:5.1f}% vs {ref:5.1f}% reference  ({delta:+5.1f}pp)  -> {mark}")
    print(f"\n  PASS on mediation level means in-copyright works scored only by {cand}")
    print(f"  can be read on the same scale as the rest of the corpus, with the")
    print(f"  single-reader caveat stated. FAIL means they must be reported separately.")


if __name__ == "__main__":
    main()
