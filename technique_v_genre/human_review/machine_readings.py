"""Gather every machine reading of the 63 annotation passages, and score how
contested each passage is.

Two uses. The annotation page sorts by the contention score so a reader can
go straight to the passages the machines handle worst; and after a passage is
coded, the page reveals what each reader said, so the session doubles as a
way to see how the machines behave.

CONTENTION. Each qualifying reader gets one vote regardless of its k, so a
k=10 reader does not drown out a k=1 one: a reader's runs are first collapsed
into its own distribution over mediation levels, those per-reader
distributions are averaged, and contention is the Simpson index of the
average, 1 - sum(p^2). It is 0 when every reader on every run gives the same
level and rises both when readers disagree with each other and when a single
reader is unstable across its own runs — which is what "least certain or most
disagree" has to mean when the two sources of uncertainty are not separable
from the outside.

QUALIFYING READERS. Only readers that passed the calibration gate count
toward contention: gpt-5.4, qwen3-max, jev, qwen3.6-local. llama and
GLiClass collapsed onto one category (llama put 90/90 Time Passes passages
at level 1) and would flatten the score toward whatever they always say.
They are still carried into the reveal panel, labelled, because a reader that
fails this way is worth seeing.
"""
import collections, glob, json, math
from pathlib import Path

HERE = Path(__file__).resolve().parent
TV = HERE.parent

SCORING = ["gpt-5.4", "qwen3-max", "jev", "qwen3.6-local"]
FAILED = {"llama": "collapsed onto one category",
          "gliclass": "collapsed onto one category"}

key = json.loads((HERE / "passage_key.json").read_text())
pid2code = {v["passage_id"]: c for c, v in key.items()}

runs = collections.defaultdict(lambda: collections.defaultdict(list))
jev_probs = collections.defaultdict(list)

for d in ("scored", "scored_jev", "scored_local", "calibration"):
    for f in glob.glob(str(TV / d / "*.jsonl")):
        if f.endswith(".bak"):
            continue
        for line in Path(f).read_text(errors="replace").splitlines():
            if '"passage_id"' not in line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            code = pid2code.get(r.get("passage_id"))
            if not code or "error" in r or not r.get("reader_id"):
                continue
            rd = r["reader_id"]
            if rd not in SCORING and rd not in FAILED:
                continue
            runs[code][rd].append({
                "is_soc": bool(r.get("is_soc")),
                "level": r.get("mediation_level"),
                "type": r.get("soc_type"),
            })
            if rd == "jev" and r.get("jev_mediation_probs"):
                jev_probs[code].append(r["jev_mediation_probs"])


def dist(vals):
    """Distribution over levels 0-3 from one reader's runs. None -> no soc."""
    c = collections.Counter(v for v in vals if v is not None)
    n = sum(c.values())
    return [c.get(i, 0) / n for i in range(4)] if n else None


out = {}
for code in key:
    per = {}
    for rd, rows in runs[code].items():
        levels = [x["level"] for x in rows if x["is_soc"]]
        d = dist(levels)
        types = collections.Counter(x["type"] for x in rows if x["is_soc"] and x["type"])
        per[rd] = {
            "k": len(rows),
            "soc_share": round(sum(1 for x in rows if x["is_soc"]) / len(rows), 3),
            "dist": [round(p, 3) for p in d] if d else None,
            "modal": max(range(4), key=lambda i: d[i]) if d else None,
            "k_consistency": round(max(d), 3) if d else None,
            "modal_type": types.most_common(1)[0][0] if types else None,
            "failed_gate": FAILED.get(rd),
        }

    good = [per[r]["dist"] for r in SCORING if r in per and per[r]["dist"]]
    if good:
        avg = [sum(d[i] for d in good) / len(good) for i in range(4)]
        contention = 1 - sum(p * p for p in avg)
        ent = -sum(p * math.log(p, 4) for p in avg if p > 0)   # log base 4 -> 0..1
        modals = [per[r]["modal"] for r in SCORING if r in per and per[r]["modal"] is not None]
        spread = (max(modals) - min(modals)) if modals else 0
        top = collections.Counter(modals).most_common(1)[0][1] if modals else 0
        split = round(1 - top / len(modals), 3) if modals else 0.0
    else:
        avg, contention, ent, spread, split = None, 0.0, 0.0, 0, 0.0

    socs = [per[r]["soc_share"] for r in SCORING if r in per]
    out[code] = {
        "readers": per,
        "avg_dist": [round(p, 3) for p in avg] if avg else None,
        "contention": round(contention, 4),
        "entropy": round(ent, 3),
        "spread": spread,                       # widest gap between reader modes
        "reader_split": split,                  # share of readers off the plurality
        "soc_split": round(max(socs) - min(socs), 3) if len(socs) > 1 else 0.0,
        "n_scoring_readers": len(good),
        "jev_mediation_probs": (
            [round(sum(p.get(str(i), 0) for p in jev_probs[code]) / len(jev_probs[code]), 3)
             for i in range(4)] if jev_probs.get(code) else None),
    }

(HERE / "machine_readings.json").write_text(json.dumps(out, indent=1))

ranked = sorted(out, key=lambda c: -out[c]["contention"])
cov = collections.Counter(out[c]["n_scoring_readers"] for c in out)
print(f"{len(out)} passages; scoring readers per passage: {dict(sorted(cov.items()))}")
print(f"contention: max {out[ranked[0]]['contention']:.3f}  "
      f"median {out[ranked[len(ranked)//2]]['contention']:.3f}  "
      f"min {out[ranked[-1]]['contention']:.3f}")
print("\nmost contested:")
for c in ranked[:8]:
    k, o = key[c], out[c]
    modes = " ".join(f"{r.split('-')[0][:4]}:{o['readers'][r]['modal']}"
                     for r in SCORING if r in o["readers"] and o["readers"][r]["modal"] is not None)
    print(f"  {c} {k['block']:5s} {k['title'][:26]:26s} contention {o['contention']:.3f}  {modes}")
print("\nleast contested:")
for c in ranked[-4:]:
    k, o = key[c], out[c]
    print(f"  {c} {k['block']:5s} {k['title'][:26]:26s} contention {o['contention']:.3f}")
