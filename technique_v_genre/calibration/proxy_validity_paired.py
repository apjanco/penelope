"""Paired version of proxy_validity.py: is qwen3.6-local a valid stand-in for
qwen3-max on the in-copyright works?

proxy_validity.py compares the candidate (on its few hundred overlap passages)
against a reference computed over the whole corpus, and divides level
agreement by every overlapping passage, including ones where one reader found
no consciousness. Both distort the verdict: the overlap is mostly Faulkner,
whose level distribution differs from the corpus, and detection disagreement
leaks into the "within-1 level" figure (which came out barely above exact
agreement). This script fixes both:

  - every pair is compared on the SAME passages (scored by all three readers)
  - detection (is_soc) agreement is reported separately
  - level agreement is computed only where both readers call the passage SOC
  - it also reports each reader's mean level on those passages, since the
    chapter's claims rest on work-level means and a constant offset matters
    more there than per-passage agreement

Usage:  python calibration/proxy_validity_paired.py [--candidate qwen3.6-local]
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

TV = Path(__file__).resolve().parent.parent
SKIP = {"time_passes.jsonl", "qwen3_reasoning_smoketest.jsonl"}


def load(readers: set[str]) -> dict:
    per = collections.defaultdict(lambda: collections.defaultdict(list))
    for d in (TV / "scored", TV / "scored_local"):
        for f in d.glob("*.jsonl"):
            if f.name in SKIP:
                continue
            for line in f.open():
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "error" in r or r.get("reader_id") not in readers:
                    continue
                per[r["passage_id"]][r["reader_id"]].append(r)
    return per


def modal(rows, field):
    c = collections.Counter(x.get(field) for x in rows if x.get(field) is not None)
    return c.most_common(1)[0][0] if c else None


def soc(rows):
    """Majority is_soc over the reader's runs."""
    v = [bool(x.get("is_soc")) for x in rows]
    return sum(v) * 2 > len(v)


def level(rows):
    return modal([x for x in rows if x.get("is_soc")], "mediation_level")


def compare(per, pids, a, b):
    det = both = exact = adj = 0
    la_sum = lb_sum = 0.0
    for p in pids:
        ra, rb = per[p][a], per[p][b]
        sa, sb = soc(ra), soc(rb)
        det += sa == sb
        if sa and sb:
            la, lb = level(ra), level(rb)
            if la is None or lb is None:
                continue
            both += 1
            exact += la == lb
            adj += abs(la - lb) <= 1
            la_sum += la
            lb_sum += lb
    n = len(pids)
    return {
        "n": n, "detect_agree": 100 * det / n, "n_both_soc": both,
        "level_exact": 100 * exact / both if both else float("nan"),
        "level_within1": 100 * adj / both if both else float("nan"),
        "mean_a": la_sum / both if both else float("nan"),
        "mean_b": lb_sum / both if both else float("nan"),
    }


def show(label, r, a, b):
    print(f"  {label:34s} n={r['n']:4d}  detect {r['detect_agree']:5.1f}%  "
          f"| both-SOC n={r['n_both_soc']:4d}  level {r['level_exact']:5.1f}%  "
          f"within-1 {r['level_within1']:5.1f}%  | mean {a} {r['mean_a']:.2f} vs {b} {r['mean_b']:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", default="qwen3.6-local")
    ap.add_argument("--target", default="qwen3-max")
    args = ap.parse_args()
    cand, tgt, ref = args.candidate, args.target, "gpt-5.4"
    per = load({cand, tgt, ref})
    pids = sorted(p for p, rv in per.items() if cand in rv and tgt in rv and ref in rv)
    if not pids:
        print("no passages scored by all three readers")
        return
    by_title = collections.defaultdict(list)
    for p in pids:
        by_title[per[p][cand][0]["title"]].append(p)

    print(f"\nPassages scored by {cand}, {tgt} and {ref}: {len(pids)}")
    for groups in ([("ALL", pids)], sorted(by_title.items())):
        for t, ps in groups:
            print(f"\n[{t}]")
            show(f"{ref} vs {tgt}  (reference)", compare(per, ps, ref, tgt), ref, tgt)
            show(f"{cand} vs {tgt}  (stand-in)", compare(per, ps, cand, tgt), cand, tgt)
            show(f"{cand} vs {ref}", compare(per, ps, cand, ref), cand, ref)

    r_ref, r_c = compare(per, pids, ref, tgt), compare(per, pids, cand, tgt)
    print("\nVERDICT (stand-in minus reference, same passages; PASS >= -5pp, MARGINAL >= -12pp)")
    for k in ("detect_agree", "level_exact", "level_within1"):
        d = r_c[k] - r_ref[k]
        mark = "PASS" if d >= -5 else ("MARGINAL" if d >= -12 else "FAIL")
        print(f"  {k:14s} {r_c[k]:5.1f}% vs {r_ref[k]:5.1f}%  ({d:+5.1f}pp) -> {mark}")
    print(f"  mean-level offset {cand} - {tgt}: {r_c['mean_a'] - r_c['mean_b']:+.2f} "
          f"(reference offset {ref} - {tgt}: {r_ref['mean_a'] - r_ref['mean_b']:+.2f})")


if __name__ == "__main__":
    main()
