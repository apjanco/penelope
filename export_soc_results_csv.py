"""Export all cross-model SoC confound-testing results to CSV for direct
review (e.g. by a co-author in Google Sheets after uploading to Drive).

Produces two files:
  soc_results_detailed.csv - one row per (passage, model, run): every call
  soc_results_summary.csv  - one row per passage: aggregated across runs

Pulls passage text from the PASSAGES lists in each test script and merges
it with the raw JSONL results, since the results files only store
passage_key, not the passage text itself.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, "/scratch/gpfs/MM4/apjanco/penelope")

import cross_model_confound_test as t_confound
import cross_model_aspect_test as t_aspect
import cross_model_richardson_test as t_richardson
import cross_model_architecture_test as t_architecture

ROOT = Path("/scratch/gpfs/MM4/apjanco/penelope")

# Author, by source_file.
AUTHOR_BY_SOURCE = {
    "to_the_lighthouse_cleaned.txt": "Woolf",
    "mrs_dalloway_cleaned.txt": "Woolf",
    "jacobs_room.txt": "Woolf",
    "the_voyage_out.txt": "Woolf",
    "between_the_acts.txt": "Woolf",
    "pg03019_pointed_roofs.txt": "Richardson",
    "pg56221_backwater.txt": "Richardson",
    "pg56664_honeycomb.txt": "Richardson",
    "pg02413_madame_bovary.txt": "Flaubert",
}

# Sentence architecture, by passage_key — close-reading judgment described
# in woolf_richardson_comparison.md (elaborate = long/periodic/subordinated;
# fragmented = short/paratactic/dash-or-ellipsis-heavy). Passages not yet
# classified (from the original confound test) are filled in here too.
ARCHITECTURE = {
    "lighthouse_iterative_1": "elaborate",
    "lighthouse_iterative_2": "elaborate",
    "lighthouse_punctual_1": "elaborate",
    "lighthouse_punctual_2": "fragmented",
    "dalloway_iterative_1": "elaborate",
    "dalloway_iterative_2": "fragmented",
    "dalloway_punctual_1": "fragmented",
    "dalloway_punctual_2": "elaborate",
    "jacobsroom_iterative_1": "elaborate",
    "jacobsroom_punctual_1": "fragmented",
    "voyageout_iterative_1": "elaborate",
    "voyageout_punctual_1": "elaborate",
    "betweentheacts_iterative_1": "fragmented",
    "betweentheacts_punctual_1": "fragmented",
    "pointedroofs_iterative_1": "fragmented",
    "pointedroofs_iterative_2": "elaborate",
    "pointedroofs_punctual_1": "fragmented",
    "pointedroofs_punctual_2": "fragmented",
    "backwater_iterative_1": "fragmented",
    "backwater_punctual_1": "fragmented",
    "honeycomb_iterative_1": "fragmented",
    "honeycomb_iterative_2": "elaborate",
    "honeycomb_punctual_1": "elaborate",
    "honeycomb_punctual_2": "fragmented",
    "lighthouse_punctual_elaborate_2": "elaborate",
    "honeycomb_punctual_elaborate": "elaborate",
    "honeycomb_iterative_elaborate": "elaborate",
    "madame-bovary_003_bertaux": "elaborate",
    "pointed-roofs_002_embrace": "elaborate",
}

DATASETS = [
    ("original_confound", t_confound, ROOT / "cross_model_confound_results.jsonl"),
    ("woolf_aspect", t_aspect, ROOT / "cross_model_aspect_results.jsonl"),
    ("richardson_aspect", t_richardson, ROOT / "cross_model_richardson_results.jsonl"),
    ("architecture_disentangling", t_architecture, ROOT / "cross_model_architecture_results.jsonl"),
]

DETAILED_FIELDS = [
    "test_batch", "passage_key", "author", "source_file", "aspect", "architecture",
    "passage_text", "model", "run", "soc_type", "narrator_position", "is_soc",
    "confidence", "explanation",
]
SUMMARY_FIELDS = [
    "test_batch", "passage_key", "author", "source_file", "aspect", "architecture",
    "passage_text", "gpt-5.4_types", "mercury-2_types", "qwen3-max_types",
    "omniscient_or_hybrid_pull", "n_distinct_stable_positions",
]


def main():
    passage_meta: dict[str, dict] = {}
    for batch_name, module, _ in DATASETS:
        for p in module.PASSAGES:
            key = p["key"]
            source_file = p["source_file"]
            passage_meta[key] = {
                "test_batch": batch_name,
                "passage_key": key,
                "author": AUTHOR_BY_SOURCE.get(source_file, "unknown"),
                "source_file": source_file,
                "aspect": p.get("aspect", ""),
                "architecture": p.get("architecture", ARCHITECTURE.get(key, "")),
                "passage_text": p["passage"],
            }

    detailed_rows = []
    by_passage_model = defaultdict(lambda: defaultdict(list))

    for batch_name, _, jsonl_path in DATASETS:
        if not jsonl_path.exists():
            continue
        for line in jsonl_path.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            key = r["passage_key"]
            meta = passage_meta.get(key, {})
            row = {
                "test_batch": batch_name,
                "passage_key": key,
                "author": meta.get("author", ""),
                "source_file": meta.get("source_file", r.get("source_file", "")),
                "aspect": meta.get("aspect", r.get("aspect", "")),
                "architecture": meta.get("architecture", r.get("architecture", "")),
                "passage_text": meta.get("passage_text", ""),
                "model": r.get("model", ""),
                "run": r.get("run", ""),
                "soc_type": r.get("soc_type", ""),
                "narrator_position": r.get("narrator_position", ""),
                "is_soc": r.get("is_soc", ""),
                "confidence": r.get("confidence", ""),
                "explanation": r.get("explanation", ""),
            }
            detailed_rows.append(row)
            by_passage_model[(batch_name, key)][r.get("model", "")].append(r.get("soc_type"))

    detailed_path = ROOT / "soc_results_detailed.csv"
    with detailed_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=DETAILED_FIELDS)
        writer.writeheader()
        writer.writerows(detailed_rows)
    print(f"Wrote {len(detailed_rows)} rows to {detailed_path}")

    reliable = {"gpt-5.4", "qwen3-max"}
    summary_rows = []
    for (batch_name, key), by_model in by_passage_model.items():
        meta = passage_meta.get(key, {})
        stable_positions = set()
        pull = False
        for m in reliable:
            types = by_model.get(m, [])
            if types and len(set(types)) == 1:
                stable_positions.add(types[0])
            if "omniscient_description" in types or "hybrid" in types:
                pull = True
        summary_rows.append({
            "test_batch": batch_name,
            "passage_key": key,
            "author": meta.get("author", ""),
            "source_file": meta.get("source_file", ""),
            "aspect": meta.get("aspect", ""),
            "architecture": meta.get("architecture", ""),
            "passage_text": meta.get("passage_text", ""),
            "gpt-5.4_types": ";".join(by_model.get("gpt-5.4", [])),
            "mercury-2_types": ";".join(by_model.get("mercury-2", [])),
            "qwen3-max_types": ";".join(by_model.get("qwen3-max", [])),
            "omniscient_or_hybrid_pull": pull,
            "n_distinct_stable_positions": len(stable_positions),
        })

    summary_path = ROOT / "soc_results_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Wrote {len(summary_rows)} rows to {summary_path}")


if __name__ == "__main__":
    main()
