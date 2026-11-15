"""Disentangling test: does sentence architecture (elaborate/periodic vs.
fragmented/paratactic) predict the omniscient_description pull independent
of habitual/punctual aspect, or is it the conjunction of both that matters?

Targets the two thinnest, most important cells from the existing 24-passage
dataset: elaborate+punctual (previously only 4 passages, one flawed by
hypothetical mood) and — as a special test of whether architecture
transcends authorial habit — Richardson writing in her rare elaborate mode,
for both aspects, since her default style is otherwise uniformly fragmented.
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

K = 3

PASSAGES = [
    {
        "key": "lighthouse_punctual_elaborate_2",
        "source_file": "to_the_lighthouse_cleaned.txt",
        "aspect": "punctual",
        "architecture": "elaborate",
        "passage": (
            "He had followed her into the drawing-room, that young man "
            "they laughed at; he was standing by the table, fidgeting "
            "with something, awkwardly, feeling himself out of things, "
            "as she knew without looking round."
        ),
    },
    {
        "key": "honeycomb_punctual_elaborate",
        "source_file": "pg56664_honeycomb.txt",
        "aspect": "punctual",
        "architecture": "elaborate",
        "passage": (
            "But to herself the laugh meant much more than liking Mrs. "
            "Corrie and holding her up and begging her to go on. It meant "
            "the large dark room, the dark invisible picture, the big "
            "pieces of strange dark furniture in gloomy corners, the huge "
            "screen near the door where the parlourmaid came in and out; "
            "the table like an island under the dome of the low-hanging "
            "rose-shaded lamp, the table-centre thickly embroidered with "
            "beetles' wings, the little dishes stuck about, sweets, "
            "curiously crusted brown almonds, sheeny grey-green olives."
        ),
    },
    {
        "key": "honeycomb_iterative_elaborate",
        "source_file": "pg56664_honeycomb.txt",
        "aspect": "iterative",
        "architecture": "elaborate",
        "passage": (
            "They had never known anything else. For years life had been "
            "for them just what it was to-day--breakfast in bed, chirping "
            "at their mother from the dressing-rooms where they slept, "
            "and scolding at Stokes as she waited on their toilet; "
            "jocularly and impatiently learning lessons from little "
            "text-books for an hour or so in the morning, spending their "
            "afternoons cantering about the commons and along the sandy "
            "roadways with the groom; driving with their mother or "
            "walking with the governess and every day coming in at the "
            "end of the afternoon to this cosy, dainty grown-up tea, with "
            "their strange untroubled brooding faces. They would grow up "
            "and be exactly like their parents."
        ),
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

    out_path = Path(__file__).parent / "cross_model_architecture_results.jsonl"
    with out_path.open("w") as fh:
        for p_info in PASSAGES:
            print(f"=== {p_info['key']} ({p_info['aspect']}, {p_info['architecture']}) ===")
            for profile in profiles:
                for run in range(K):
                    result = classify(profile, p_info["passage"], p_info["source_file"])
                    record = {
                        "passage_key": p_info["key"],
                        "aspect": p_info["aspect"],
                        "architecture": p_info["architecture"],
                        "source_file": p_info["source_file"],
                        "model": profile.label,
                        "run": run,
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
