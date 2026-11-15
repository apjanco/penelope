"""Build the reader-calibration set from the already-scored corpus.

Why this exists: llama-3.3-70B ran for 12h47m and wrote 62,792 rows before
anyone noticed it assigns indirect_interior_monologue to 56,966 passages and
omniscient_description to 1,495 — a 13x divergence from gpt-5.4 on the one
distinction that drives the whole mediation cline. Every cheap check it
would plausibly have been given, it passes: valid JSON, coherent arguments,
a sensible direct/indirect split *within* SoC. The failure is visible only
in the marginal distribution, so that is what this harness measures.

IMPORTANT — there is no ground truth here, and this set does not pretend
otherwise. dataset/test.jsonl is silver (model-generated), carries no
mediation_level, and predates this schema. So a candidate is never scored
for "accuracy". It is scored for:

  (a) DISCRIMINATION — does it use the taxonomy at all, or collapse onto one
      category? This is the axis llama failed and the only one with a
      defensible pass/fail threshold.
  (b) AGREEMENT — how close is it to each existing reader? Useful in both
      directions: too far out is a llama-style outlier, too close to
      qwen3-max means it adds no independent signal to the ensemble.
  (c) RESAMPLING BEHAVIOUR — llama returned identical levels on 97% of its
      k=3 reruns, so resampling measured nothing. A reader whose k runs never
      vary cannot support the stability analysis in AMBIGUITY.md.

Two strata, because they answer different questions:

  CONSENSUS  — passages where gpt-5.4 and qwen3-max agree on soc_type across
               all their runs. A competent reader should mostly agree too.
               Stratified across mediation levels so the set is not 90% FID.
  CONTESTED  — passages where those two disagree. These carry no expected
               answer at all; they test whether a candidate reproduces the
               ambiguity or papers over it, which is the chapter's actual
               subject.
"""
from __future__ import annotations

import collections
import json
import random
from pathlib import Path

TV = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "calibration_set.jsonl"
PER_LEVEL = 40      # consensus passages per mediation level
CONTESTED = 120
SEED = 20260930


def main() -> None:
    man = {e["source_file"][:-4]: e for e in
           json.loads((TV / "corpus_manifest.json").read_text())["corpus"]}

    # passage_id -> reader -> [soc_type...] and the passage row itself
    votes: dict[str, dict[str, list]] = collections.defaultdict(lambda: collections.defaultdict(list))
    lvl: dict[str, dict[str, list]] = collections.defaultdict(lambda: collections.defaultdict(list))
    meta: dict[str, dict] = {}
    for f in sorted((TV / "scored").glob("*.jsonl")):
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "error" in d or d.get("reader_id") not in ("gpt-5.4", "qwen3-max"):
                continue
            pid = d["passage_id"]
            votes[pid][d["reader_id"]].append(d.get("soc_type"))
            if d.get("mediation_level") is not None:
                lvl[pid][d["reader_id"]].append(d["mediation_level"])
            if pid not in meta and d.get("passage_text"):
                stem = d["source_file"][:-4]
                e = man.get(stem, {})
                meta[pid] = {
                    "passage_id": pid, "source_file": d["source_file"],
                    "author": e.get("author", d.get("author")),
                    "title": e.get("title", d.get("title")), "year": e.get("year", d.get("year")),
                    "role": "calibration", "passage_text": d["passage_text"],
                }

    consensus: dict[int, list] = collections.defaultdict(list)
    contested: list = []
    for pid, rv in votes.items():
        if pid not in meta or len(rv) < 2:
            continue
        g, q = rv.get("gpt-5.4", []), rv.get("qwen3-max", [])
        if not g or not q:
            continue
        # unanimous within each reader AND between them
        if len(set(g)) == 1 and len(set(q)) == 1 and g[0] == q[0] and g[0] is not None:
            levels = lvl[pid].get("gpt-5.4", [])
            if levels and len(set(levels)) == 1:
                row = dict(meta[pid])
                row.update(stratum="consensus", consensus_soc_type=g[0],
                           consensus_mediation_level=levels[0])
                consensus[levels[0]].append(row)
        elif set(g) & set(q) == set():
            row = dict(meta[pid])
            row.update(stratum="contested",
                       gpt_soc_types=sorted(set(x for x in g if x)),
                       qwen_soc_types=sorted(set(x for x in q if x)))
            contested.append(row)

    rnd = random.Random(SEED)
    picked: list = []
    for level in range(4):
        pool = consensus.get(level, [])
        rnd.shuffle(pool)
        take = pool[:PER_LEVEL]
        picked += take
        print(f"  consensus level {level}: {len(pool):>6,} available, took {len(take)}")
    rnd.shuffle(contested)
    picked += contested[:CONTESTED]
    print(f"  contested       : {len(contested):>6,} available, took {min(len(contested), CONTESTED)}")

    rnd.shuffle(picked)
    for i, r in enumerate(picked):
        r["calibration_index"] = i
    OUT.write_text("\n".join(json.dumps(r) for r in picked) + "\n")

    authors = collections.Counter(r["author"] for r in picked)
    print(f"\n  wrote {len(picked)} passages -> {OUT.name}")
    print(f"  authors: {dict(authors)}")


if __name__ == "__main__":
    main()
