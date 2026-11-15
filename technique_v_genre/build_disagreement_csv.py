"""Build a co-author-facing CSV of Woolf passages where gpt-5.4 and qwen3-max
both find SoC but diverge on how to categorize it (boundary_tension_convergent
or strong_divergence per analyze_stability.py's classification). Reuses that
script's stability logic directly rather than reimplementing it.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_stability import summarize_reader, classify_passage  # noqa: E402

IN_PATH = Path("/tmp/woolf_all_scored.jsonl")
OUT_PATH = Path("/scratch/gpfs/MM4/apjanco/penelope/technique_v_genre/woolf_disagreement_examples.csv")
THRESHOLD = 0.8

rows = [json.loads(l) for l in IN_PATH.read_text().splitlines() if not json.loads(l).get("error")]

by_passage_reader = defaultdict(list)
meta = {}
text_by_passage = {}
for r in rows:
    key = (r["passage_id"], r["reader_id"])
    by_passage_reader[key].append(r)
    meta[r["passage_id"]] = {"author": r["author"], "title": r["title"], "year": r["year"]}
    text_by_passage[r["passage_id"]] = r["passage_text"]

by_passage = defaultdict(dict)
for (pid, reader), runs in by_passage_reader.items():
    by_passage[pid][reader] = summarize_reader(runs, THRESHOLD)

readers = ["gpt-5.4", "qwen3-max"]
out_rows = []
for pid, reader_summaries in by_passage.items():
    cls = classify_passage(reader_summaries)
    if cls not in ("boundary_tension_convergent", "strong_divergence"):
        continue
    if not all(r in reader_summaries for r in readers):
        continue
    m = meta[pid]
    row = {
        "disagreement_strength": "STRONG" if cls == "strong_divergence" else "boundary_tension",
        "title": m["title"],
        "year": m["year"],
        "passage_id": pid,
        "passage_text": text_by_passage[pid],
    }
    for reader in readers:
        s = reader_summaries[reader]
        prefix = reader.replace(".", "_").replace("-", "_")
        row[f"{prefix}_soc_type"] = s.get("soc_type_mode", "")
        row[f"{prefix}_mediation_level"] = s.get("mediation_level_mode", "")
        row[f"{prefix}_fit_category"] = s.get("fit_category_mode", "")
        ex = s.get("example", {})
        row[f"{prefix}_argument"] = ex.get("argument", "")
        row[f"{prefix}_alternative_reasoning"] = ex.get("alternative_reasoning", "")
        row[f"{prefix}_antimimetic"] = ex.get("antimimetic", False)
    out_rows.append(row)

# strongest disagreements first, then grouped by novel
out_rows.sort(key=lambda r: (r["disagreement_strength"] != "STRONG", r["title"], r["passage_id"]))

fieldnames = [
    "disagreement_strength", "title", "year", "passage_id", "passage_text",
    "gpt_5_4_soc_type", "gpt_5_4_mediation_level", "gpt_5_4_fit_category",
    "gpt_5_4_argument", "gpt_5_4_alternative_reasoning", "gpt_5_4_antimimetic",
    "qwen3_max_soc_type", "qwen3_max_mediation_level", "qwen3_max_fit_category",
    "qwen3_max_argument", "qwen3_max_alternative_reasoning", "qwen3_max_antimimetic",
]
with OUT_PATH.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for row in out_rows:
        writer.writerow(row)

print(f"Wrote {len(out_rows)} disagreement examples -> {OUT_PATH}")
strong = sum(1 for r in out_rows if r["disagreement_strength"] == "STRONG")
print(f"  {strong} strong_divergence, {len(out_rows) - strong} boundary_tension")
by_title = defaultdict(int)
for r in out_rows:
    by_title[r["title"]] += 1
for title, n in sorted(by_title.items()):
    print(f"  {title}: {n}")
