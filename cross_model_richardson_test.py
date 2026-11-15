"""Cross-model stability test for the habitual/iterative-aspect hypothesis,
extended to Richardson's Pilgrimage (Pointed Roofs, Backwater, Honeycomb),
to check whether the omniscient-pull pattern found in Woolf holds across an
author with a very different sentence style (fragmentary, present-immediate,
resistant to generalizing summary, vs. Woolf's long periodic sentences).
Same methodology as cross_model_aspect_test.py.
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
    # --- Pointed Roofs ---
    {
        "key": "pointedroofs_iterative_1",
        "source_file": "pg03019_pointed_roofs.txt",
        "aspect": "iterative",
        "passage": (
            "Why had he come to tea every Sunday--never missing a single "
            "Sunday--all the winter? Why did he say, \"Play 'Abide with "
            "me,'\" \"Play 'Abide with me'\" yesterday, if he didn't care? "
            "What was the good of being so quiet and saying nothing? Why "
            "didn't he say \"Don't go\" or \"When are you coming back?\" "
            "Eve said he looked perfectly miserable."
        ),
    },
    {
        "key": "pointedroofs_iterative_2",
        "source_file": "pg03019_pointed_roofs.txt",
        "aspect": "iterative",
        "passage": (
            "The sense of that sudden little embrace recurred often to "
            "Miriam during the course of the first day. It was unlike any "
            "contact she had known--more motherly than her mother's. "
            "Neither of her sisters could have embraced her like that. She "
            "did not know that a human form could bring such a sense of "
            "warm nearness, that human contours could be eloquent--or "
            "anyone so sweetly daring."
        ),
    },
    {
        "key": "pointedroofs_punctual_1",
        "source_file": "pg03019_pointed_roofs.txt",
        "aspect": "punctual",
        "passage": (
            "She drew back from the window as the bass chords began "
            "thumping gently in the darkness. It was better that it should "
            "come now than later on, at dinner-time. She could get over it "
            "alone up here."
        ),
    },
    {
        "key": "pointedroofs_punctual_2",
        "source_file": "pg03019_pointed_roofs.txt",
        "aspect": "punctual",
        "passage": (
            "She flushed and fumbled--thought of dressing-tables and the "
            "little objects of which she had made so many hanging to the "
            "mirror by ribbons; \"toilet-tidies\" haunted her--but that was "
            "not it--she smoothed out her work as if to show it to "
            "Fräulein--\"Na, na,\" came the delicate caustic voice. \"Was "
            "wird das wohl sein?\" Then she remembered."
        ),
    },
    # --- Backwater ---
    {
        "key": "backwater_iterative_1",
        "source_file": "pg56221_backwater.txt",
        "aspect": "iterative",
        "passage": (
            "That darkness was dreadful. It was his own life. She would "
            "never know it. However well they got to know each other they "
            "would always be strangers. Probably he never thought about "
            "her when he was alone. Only of Shakespeare and politics."
        ),
    },
    {
        "key": "backwater_punctual_1",
        "source_file": "pg56221_backwater.txt",
        "aspect": "punctual",
        "passage": (
            "Miriam turned sharply to her mother. She was sitting with her "
            "tired look--bright eyes, and moist flushed face. How had she "
            "heard about the little classes? Had there been little "
            "classes? She could not remember them."
        ),
    },
    # --- Honeycomb ---
    {
        "key": "honeycomb_iterative_1",
        "source_file": "pg56664_honeycomb.txt",
        "aspect": "iterative",
        "passage": (
            "It meant never being able to agree with people about books, "
            "never liking them for the same reasons as other people.... "
            "But it was true and exciting. It meant ... things coming to "
            "you out of books, people, not the people in the books, but "
            "knowing, absolutely, everything about the author."
        ),
    },
    {
        "key": "honeycomb_iterative_2",
        "source_file": "pg56664_honeycomb.txt",
        "aspect": "iterative",
        "passage": (
            "He had never thought for a moment in his life of the world as "
            "anything than what it appeared to be. He had no suspicion "
            "that anyone ever did."
        ),
    },
    {
        "key": "honeycomb_punctual_1",
        "source_file": "pg56664_honeycomb.txt",
        "aspect": "punctual",
        "passage": (
            "It sounds like a housekeeper giving an order to a servant she "
            "knows won't obey her, thought Miriam, swishing more "
            "comfortably into her chair. If Mrs. Craven would talk there "
            "would be no need to do anything."
        ),
    },
    {
        "key": "honeycomb_punctual_2",
        "source_file": "pg56664_honeycomb.txt",
        "aspect": "punctual",
        "passage": (
            "\"This dress is a nightmare in this room,\" she thought, "
            "puffing up her hair under her fringe-net with a hat-pin. "
            "\"Never mind, I mustn't think about it,\" she added hurriedly, "
            "disconcerted for a moment by the frightened look in her eyes."
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

    out_path = Path(__file__).parent / "cross_model_richardson_results.jsonl"
    with out_path.open("w") as fh:
        for p_info in PASSAGES:
            print(f"=== {p_info['key']} ({p_info['aspect']}) ===")
            for profile in profiles:
                for run in range(K):
                    result = classify(profile, p_info["passage"], p_info["source_file"])
                    record = {
                        "passage_key": p_info["key"],
                        "aspect": p_info["aspect"],
                        "source_file": p_info["source_file"],
                        "author": "richardson",
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
