"""Build the blind annotation set for the two human readers.

Two blocks:

  CORE  — all 33 passages of "Time Passes", the section every machine reader
          has now been run on at k=10. This is the comparison the chapter
          wants, and it is deliberately NOT a random sample: it is the crux
          section, so agreement measured on it describes a hard case.

  CLINE — 30 passages drawn from the existing calibration set, stratified
          across all four of Cohn's levels. "Time Passes" collapses to a
          near-binary L0/L1 problem (57%/40%, two L2s, no L3), so agreement
          measured there alone says nothing about the top of the cline.
          This block supplies the full-range number to sit beside it.

Both blocks are emitted under opaque display codes (P01...P63). The code ->
passage_id mapping is written to passage_key.json, which stays in the repo
and is NOT shipped to the annotation page: an annotator who can see that a
passage comes from Ulysses has a prior about it before reading a word.

Order is NOT fixed here. The page shuffles per annotator from a seed derived
from their user id, so the two readers meet the passages in different orders
and any fatigue or drift effect does not land on the same passages for both.
"""
import collections, glob, json, random
from pathlib import Path

HERE = Path(__file__).resolve().parent
TV = HERE.parent

cal = {json.loads(l)["passage_id"]: json.loads(l)
       for l in (TV / "calibration/calibration_set.jsonl").read_text().splitlines() if l.strip()}

# Reference level per calibration passage: the modal mediation_level across
# every reader that has scored it. Used ONLY to stratify the sample, never
# shown to annotators and never treated as ground truth.
levels = collections.defaultdict(list)
for f in glob.glob(str(TV / "calibration/scored_*.jsonl")):
    reader = Path(f).stem.replace("scored_", "")
    if reader in ("gliclass", "gemma"):      # failed the calibration gate
        continue
    for line in Path(f).read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("passage_id") in cal and r.get("is_soc") and r.get("mediation_level") is not None:
            levels[r["passage_id"]].append(r["mediation_level"])
ref = {p: collections.Counter(v).most_common(1)[0][0] for p, v in levels.items()}

rng = random.Random(20261027)

# 30 cline passages: 4 consensus + ~3-4 contested per level, spread across
# authors so no single novel dominates a level.
picked = []
for level in (0, 1, 2, 3):
    for stratum, want in (("consensus", 4), ("contested", 4 if level < 2 else 3)):
        pool = [p for p in ref
                if ref[p] == level and cal[p]["stratum"] == stratum]
        rng.shuffle(pool)
        seen = collections.Counter()
        chosen = []
        for p in pool:                       # at most 2 per author per cell
            a = cal[p]["author"]
            if seen[a] >= 2:
                continue
            seen[a] += 1
            chosen.append(p)
            if len(chosen) == want:
                break
        picked.extend(chosen)

core = [json.loads(l) for l in (TV / "passages/time_passes.jsonl").read_text().splitlines() if l.strip()]

items, key = [], {}
for i, p in enumerate(core, start=1):
    code = f"P{i:02d}"
    items.append({"code": code, "block": "core", "text": p["passage_text"], "n_words": p["n_words"]})
    key[code] = {"passage_id": p["passage_id"], "block": "core",
                 "source_file": p["source_file"], "title": p["title"],
                 "author": p["author"], "year": p["year"]}
for j, pid in enumerate(picked, start=len(core) + 1):
    c = cal[pid]
    code = f"P{j:02d}"
    items.append({"code": code, "block": "cline", "text": c["passage_text"],
                  "n_words": len(c["passage_text"].split())})
    key[code] = {"passage_id": pid, "block": "cline", "source_file": c["source_file"],
                 "title": c["title"], "author": c["author"], "year": c["year"],
                 "stratum": c["stratum"], "ref_level": ref[pid]}

(HERE / "annotation_set.json").write_text(json.dumps(items, indent=1))
(HERE / "passage_key.json").write_text(json.dumps(key, indent=1))

print(f"{len(items)} passages  ({sum(1 for i in items if i['block']=='core')} core, "
      f"{sum(1 for i in items if i['block']=='cline')} cline)")
print("cline block by reference level:",
      dict(sorted(collections.Counter(key[c]['ref_level'] for c in key
                                      if key[c]['block'] == 'cline').items())))
print("cline block by author:",
      dict(collections.Counter(key[c]['author'] for c in key if key[c]['block'] == 'cline')))
print("total words:", sum(i["n_words"] for i in items))
