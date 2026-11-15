"""Cross-reader agreement, gpt-5.4 vs qwen3-max: categorical (Humphrey
soc_type) and ordinal (Cohn mediation_level), with the disagreement
breakdown the chapter cites.

Replaces chapter_viz/data/reliability.json, which no script in the repo
produced. Its n=17,510 turned out to use an "any run" inclusion rule — a
passage counts if each reader found consciousness in at least one of its
k=3 runs. Everything else in the pipeline (Experiment D, figure data,
AMBIGUITY.md) uses the majority rule, which gives n=16,953. The kappas
differ only in the third decimal; both are written here so the choice is
explicit rather than inherited.

Cohen's kappa is unweighted on each reader's modal label over its
SOC-positive runs.

Usage:  python reader_agreement.py      # writes reader_agreement.json
"""
from __future__ import annotations

import collections
import json
from pathlib import Path

TV = Path(__file__).resolve().parent
OUT = TV / "reader_agreement.json"
A, B = "gpt-5.4", "qwen3-max"


def mode(xs):
    c = collections.Counter(x for x in xs if x is not None)
    return c.most_common(1)[0][0] if c else None


def kappa(pairs):
    n = len(pairs)
    obs = sum(a == b for a, b in pairs) / n
    ca = collections.Counter(a for a, _ in pairs)
    cb = collections.Counter(b for _, b in pairs)
    exp = sum(ca[k] * cb[k] for k in ca) / n / n
    return {"observed": round(100 * obs, 1), "chance": round(100 * exp, 1),
            "kappa": round((obs - exp) / (1 - exp), 4)}


def main() -> None:
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for f in sorted((TV / "scored").glob("*.jsonl")):
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "error" in r or r.get("reader_id") not in (A, B):
                continue
            by[r["passage_id"]][r["reader_id"]].append(r)

    rules = {
        "majority": lambda runs: sum(bool(x.get("is_soc")) for x in runs) * 2 > len(runs),
        "any_run": lambda runs: any(x.get("is_soc") for x in runs),
    }
    out = {}
    for name, detected in rules.items():
        types, levels = [], []
        for d in by.values():
            if A not in d or B not in d or not (detected(d[A]) and detected(d[B])):
                continue
            pos = {rd: [x for x in d[rd] if x.get("is_soc")] for rd in (A, B)}
            t = [mode([x.get("soc_type") for x in pos[rd]]) for rd in (A, B)]
            lv = [mode([x.get("mediation_level") for x in pos[rd]]) for rd in (A, B)]
            if None in t or None in lv:
                continue
            types.append(tuple(t))
            levels.append(tuple(lv))
        n = len(types)
        dis = [tuple(sorted(p)) for p in types if p[0] != p[1]]
        ldis = [p for p in levels if p[0] != p[1]]
        pairs = collections.Counter(dis).most_common(8)
        out[name] = {
            "n_passages": n,
            "categorical": kappa(types),
            "ordinal": kappa(levels),
            "level_within_one_pct_of_all": round(100 * sum(abs(a - b) <= 1 for a, b in levels) / n, 1),
            "level_one_apart_pct_of_disagreements": round(100 * sum(abs(a - b) == 1 for a, b in ldis) / len(ldis), 1),
            "type_disagreements": len(dis),
            "top_type_disagreements": [
                {"pair": " vs ".join(p), "n": c,
                 "pct_of_passages": round(100 * c / n, 1),
                 "pct_of_disagreements": round(100 * c / len(dis), 1)} for p, c in pairs],
        }
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
