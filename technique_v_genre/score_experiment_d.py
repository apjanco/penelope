"""Score the 150-passage Experiment D sample with the frontier ensemble.

Reader choice: gpt-5.4 and qwen3-max only. mercury-2 is excluded — per
prior findings in this project (woolf_richardson_comparison.md flagged its
general unreliability and recommended dropping it from aspect-style tests)
and per this round's explicit direction not to treat it as a proper
frontier-model reader.

K=1 (single pass per model) at this validation-gate stage: Experiment D
only needs to check whether cross-reader convergence tracks measurable
surface features, not within-model stability under resampling — that's a
later step (boundary-tension confirmation), not this one.

Output: technique_v_genre/experiment_d_results.jsonl, one row per
(passage, model).
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, "/scratch/gpfs/MM4/apjanco/penelope")

from scripts.config import Config
from openai import OpenAI

from prompts import SYSTEM_PROMPT

ROOT = Path(__file__).resolve().parent
SAMPLE_PATH = ROOT / "experiment_d_sample.jsonl"
OUT_PATH = ROOT / "experiment_d_results.jsonl"
READERS = {"gpt-5.4", "qwen3-max"}


def score(profile, passage_text: str, source_file: str) -> dict | None:
    user_content = (
        f"Source: {source_file}\n\n"
        "Score the following passage per the schema. If it renders no "
        "consciousness at all, return {\"instances\": []}. Respond with "
        "JSON only.\n\n"
        f"{passage_text}"
    )
    client = OpenAI(base_url=profile.base_url, api_key=profile.api_key)
    kwargs = dict(
        model=profile.model_name,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
    )
    if profile.temperature is not None:
        kwargs["temperature"] = profile.temperature
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(**kwargs)
            raw = resp.choices[0].message.content or ""
            cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
            cleaned = re.sub(r"</think>", "", cleaned)
            cleaned = re.sub(r"```(?:json)?", "", cleaned).strip()
            start = cleaned.find("{")
            data = json.loads(cleaned[start:] if start >= 0 else cleaned)
            instances = data.get("instances") or []
            if not instances:
                return {"is_soc": False}
            item = instances[0]
            return {
                "is_soc": item.get("is_soc", True),
                "mediation_level": item.get("mediation_level"),
                "soc_type": item.get("soc_type"),
                "secondary_devices": item.get("secondary_devices") or [],
                "fit_category": item.get("fit_category"),
                "fit_alternatives": item.get("fit_alternatives") or [],
                "antimimetic": item.get("antimimetic", False),
                "antimimetic_defense": item.get("antimimetic_defense", ""),
                "narrator_position": item.get("narrator_position"),
                "character_pov": item.get("character_pov", ""),
                "confidence": item.get("confidence"),
                "explanation": (item.get("explanation") or "")[:300],
                "evidence": item.get("evidence") or [],
            }
        except Exception as exc:
            print(f"    [{profile.label}] attempt {attempt + 1} failed: {exc}")
            time.sleep(3)
    return None


def main():
    cfg = Config.load()
    profiles = [p for p in cfg.models if p.label in READERS]
    found = {p.label for p in profiles}
    missing = READERS - found
    if missing:
        sys.exit(f"Missing model profiles in models.yaml: {missing}")
    print("Scoring with:", [p.label for p in profiles])

    rows = [json.loads(l) for l in SAMPLE_PATH.read_text().splitlines()]
    print(f"{len(rows)} passages x {len(profiles)} readers = {len(rows) * len(profiles)} calls\n")

    done_keys = set()
    mode = "a"
    if OUT_PATH.exists():
        for l in OUT_PATH.read_text().splitlines():
            r = json.loads(l)
            done_keys.add((r["passage_id"], r["reader_id"]))
        print(f"Resuming: {len(done_keys)} (passage, reader) pairs already done.")
    else:
        mode = "w"

    with OUT_PATH.open(mode) as fh:
        for i, row in enumerate(rows):
            for profile in profiles:
                key = (row["passage_id"], profile.label)
                if key in done_keys:
                    continue
                result = score(profile, row["passage_text"], row["source_file"])
                record = {
                    "passage_id": row["passage_id"],
                    "source_file": row["source_file"],
                    "author": row["author"],
                    "title": row["title"],
                    "year": row["year"],
                    "role": row["role"],
                    "reader_id": profile.label,
                    "run": 0,
                    "passage_text": row["passage_text"],
                    **(result or {"error": True}),
                }
                fh.write(json.dumps(record) + "\n")
                fh.flush()
            if (i + 1) % 10 == 0:
                print(f"  {i + 1}/{len(rows)} passages done")

    print("\nWritten to", OUT_PATH)


if __name__ == "__main__":
    main()
