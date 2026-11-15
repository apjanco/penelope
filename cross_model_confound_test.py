"""One-off cross-model stability test for two category-2 confound candidates
(narrator_position: present paired with indirect_interior_monologue). Reuses
the project's existing Config/ModelProfile (scripts/config.py) and
TRAINING_SYSTEM_PROMPT (scripts/prompts.py) so this uses the same system
prompt / model endpoints as the rest of the silver_data pipeline. Captures
narrator_position explicitly (resample.py's own parser drops it), which is
the field this specific test cares about.

Run from a node with internet access (della-vis1/vis2), not a compute node.
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, "/scratch/gpfs/MM4/apjanco/penelope")

from scripts.config import Config
from scripts.prompts import TRAINING_SYSTEM_PROMPT
from openai import OpenAI

K = 3  # samples per model per passage

PASSAGES = [
    {
        "key": "madame-bovary_003_bertaux",
        "source_file": "pg02413_madame_bovary.txt",
        "passage": (
            "As to Charles, he did not stop to ask himself why it was a pleasure to "
            "him to go to the Bertaux. Had he done so, he would, no doubt, have "
            "attributed his zeal to the importance of the case, or perhaps to the "
            "money he hoped to make by it. Was it for this, however, that his visits "
            "to the farm formed a delightful exception to the meagre occupations of "
            "his life? On these days he rose early, set off at a gallop, urging on "
            "his horse, then got down to wipe his boots in the grass and put on "
            "black gloves before entering. He liked going into the courtyard, and "
            "noticing the gate turn against his shoulder, the cock crow on the wall, "
            "the lads run to meet him. He liked the granary and the stables; he "
            "liked old Rouault, who pressed his hand and called him his saviour; he "
            "liked the small wooden shoes of Mademoiselle Emma on the scoured flags "
            "of the kitchen--her high heels made her a little taller; and when she "
            "walked in front of him, the wooden soles springing up quickly struck "
            "with a sharp sound against the leather of her boots."
        ),
        "original_label": "indirect_interior_monologue",
        "original_narrator_position": "present",
    },
    {
        "key": "pointed-roofs_002_embrace",
        "source_file": "pg03019_pointed_roofs.txt",
        "passage": (
            "The sense of that sudden little embrace recurred often to Miriam "
            "during the course of the first day. It was unlike any contact she had "
            "known--more motherly than her mother's. Neither of her sisters could "
            "have embraced her like that. She did not know that a human form could "
            "bring such a sense of warm nearness, that human contours could be "
            "eloquent--or anyone so sweetly daring."
        ),
        "original_label": "indirect_interior_monologue",
        "original_narrator_position": "present",
    },
]


def classify(profile, passage: str, source_file: str) -> dict | None:
    user_content = (
        f"Source: {source_file}\n\n"
        "Classify the following passage. "
        "If it contains stream-of-consciousness, return a JSON 'instances' list with one entry. "
        "If it does not, return {\"instances\": []}. Respond with JSON only.\n\n"
        f"{passage}"
    )
    client = OpenAI(base_url=profile.base_url, api_key=profile.api_key)
    kwargs = dict(
        model=profile.model_name,
        messages=[
            {"role": "system", "content": TRAINING_SYSTEM_PROMPT},
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
            instances = data.get("instances") or data.get("soc_instances") or []
            if not instances:
                return {"soc_type": "none"}
            item = instances[0]
            return {
                "soc_type": item.get("soc_type"),
                "narrator_position": item.get("narrator_position"),
                "is_soc": item.get("is_soc"),
                "confidence": item.get("confidence"),
                "explanation": (item.get("explanation") or "")[:220],
            }
        except Exception as exc:
            print(f"  [{profile.label}] attempt {attempt+1} failed: {exc}")
            time.sleep(3)
    return None


def main():
    cfg = Config.load()
    profiles = cfg.models
    print("Models loaded:", [p.label for p in profiles])
    print()

    out_path = Path(__file__).parent / "cross_model_confound_results.jsonl"
    with out_path.open("w") as fh:
        for p_info in PASSAGES:
            print(f"=== {p_info['key']} (original: {p_info['original_label']}, "
                  f"narrator={p_info['original_narrator_position']}) ===")
            for profile in profiles:
                for run in range(K):
                    result = classify(profile, p_info["passage"], p_info["source_file"])
                    record = {
                        "passage_key": p_info["key"],
                        "model": profile.label,
                        "run": run,
                        "original_label": p_info["original_label"],
                        "original_narrator_position": p_info["original_narrator_position"],
                        **(result or {"error": True}),
                    }
                    fh.write(json.dumps(record) + "\n")
                    fh.flush()
                    print(f"  [{profile.label} run {run}] -> "
                          f"soc_type={record.get('soc_type')} "
                          f"narrator={record.get('narrator_position')} "
                          f"is_soc={record.get('is_soc')}")
            print()
    print("Written to", out_path)


if __name__ == "__main__":
    main()
