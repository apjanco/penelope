"""Cross-model stability test for the habitual/iterative-aspect hypothesis:
does disagreement between independent models cluster around free indirect
discourse passages describing habitual/recurring states (vs. a punctual,
single-moment thought), independent of author? Passages are hand-selected
from the raw text by grammatical aspect markers, before any model has seen
them, to avoid defining the groups by what a model already said.

Companion to cross_model_confound_test.py (same methodology, reused here
for a designed comparison rather than a post-hoc example pull).
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
    # --- To the Lighthouse ---
    {
        "key": "lighthouse_iterative_1",
        "source_file": "to_the_lighthouse_cleaned.txt",
        "aspect": "iterative",
        "passage": (
            "Since he belonged, even at the age of six, to that great clan which "
            "cannot keep this feeling separate from that, but must let future "
            "prospects, with their joys and sorrows, cloud what is actually at "
            "hand, since to such people even in earliest childhood any turn in "
            "the wheel of sensation has the power to crystallise and transfix "
            "the moment upon which its gloom or radiance rests, James Ramsay, "
            "sitting on the floor cutting out pictures from the illustrated "
            "catalogue of the Army and Navy Stores, endowed the picture of a "
            "refrigerator as his mother spoke with heavenly bliss."
        ),
    },
    {
        "key": "lighthouse_iterative_2",
        "source_file": "to_the_lighthouse_cleaned.txt",
        "aspect": "iterative",
        "passage": (
            "It was his power, his gift, suddenly to shed all superfluities, to "
            "shrink and diminish so that he looked barer and felt sparer, even "
            "physically, yet lost none of his intensity of mind, and so to "
            "stand on his little ledge facing the dark of human ignorance, how "
            "we know nothing and the sea eats away the ground we stand on — "
            "that was his fate, his gift."
        ),
    },
    {
        "key": "lighthouse_punctual_1",
        "source_file": "to_the_lighthouse_cleaned.txt",
        "aspect": "punctual",
        "passage": (
            "Had there been an axe handy, a poker, or any weapon that would "
            "have gashed a hole in his father's breast and killed him, there "
            "and then, James would have seized it."
        ),
    },
    {
        "key": "lighthouse_punctual_2",
        "source_file": "to_the_lighthouse_cleaned.txt",
        "aspect": "punctual",
        "passage": (
            "A bit of a hypocrite? she repeated. Oh no — the most sincere of "
            "men, the truest (here he was), the best; but, looking down, she "
            "thought, he is absorbed in himself, he is tyrannical, he is "
            "unjust; and kept looking down, purposely, for only so could she "
            "keep steady, staying with the Ramsays."
        ),
    },
    # --- Mrs Dalloway ---
    {
        "key": "dalloway_iterative_1",
        "source_file": "mrs_dalloway_cleaned.txt",
        "aspect": "iterative",
        "passage": (
            "For having lived in Westminster—how many years now? over "
            "twenty,—one feels even in the midst of the traffic, or waking at "
            "night, Clarissa was positive, a particular hush, or solemnity; an "
            "indescribable pause; a suspense (but that might be her heart, "
            "affected, they said, by influenza) before Big Ben strikes."
        ),
    },
    {
        "key": "dalloway_iterative_2",
        "source_file": "mrs_dalloway_cleaned.txt",
        "aspect": "iterative",
        "passage": (
            "For they might be parted for hundreds of years, she and Peter; "
            "she never wrote a letter and his were dry sticks; but suddenly it "
            "would come over her, If he were with me now what would he say?"
            "—some days, some sights bringing him back to her calmly, without "
            "the old bitterness; which perhaps was the reward of having cared "
            "for people; they came back in the middle of St James's Park on a "
            "fine morning—indeed they did."
        ),
    },
    {
        "key": "dalloway_punctual_1",
        "source_file": "mrs_dalloway_cleaned.txt",
        "aspect": "punctual",
        "passage": (
            "There! Out it boomed. First a warning, musical; then the hour, "
            "irrevocable. The leaden circles dissolved in the air. Such fools "
            "we are, she thought, crossing Victoria Street."
        ),
    },
    {
        "key": "dalloway_punctual_2",
        "source_file": "mrs_dalloway_cleaned.txt",
        "aspect": "punctual",
        "passage": (
            "Was Evelyn ill again? Evelyn was a good deal out of sorts, said "
            "Hugh, intimating by a kind of pout or swell of his very "
            "well-covered, manly, extremely handsome, perfectly upholstered "
            "body (he was almost too well dressed always, but presumably had "
            "to be, with his little job at Court) that his wife had some "
            "internal ailment, nothing serious, which, as an old friend, "
            "Clarissa Dalloway would quite understand without requiring him "
            "to specify. Ah yes, she did of course; what a nuisance; and felt "
            "very sisterly and oddly conscious at the same time of her hat. "
            "Not the right hat for the early morning, was that it?"
        ),
    },
    # --- Jacob's Room ---
    {
        "key": "jacobsroom_iterative_1",
        "source_file": "jacobs_room.txt",
        "aspect": "iterative",
        "passage": (
            "For the landscape needed it. It was too pale—greys flowing into "
            "lavenders, and one star or a white gull suspended just so—too "
            "pale as usual. The critics would say it was too pale, for he "
            "was an unknown man exhibiting obscurely, a favourite with his "
            "landladies' children, wearing a cross on his watch chain, and "
            "much gratified if his landladies liked his pictures—which they "
            "often did."
        ),
    },
    {
        "key": "jacobsroom_punctual_1",
        "source_file": "jacobs_room.txt",
        "aspect": "punctual",
        "passage": (
            "No, but not in lodgings, thought Mrs Flanders. It's a great "
            "experiment coming so far with young children. There's no man "
            "to help with the perambulator. And Jacob is such a handful; so "
            "obstinate already."
        ),
    },
    # --- The Voyage Out ---
    {
        "key": "voyageout_iterative_1",
        "source_file": "the_voyage_out.txt",
        "aspect": "iterative",
        "passage": (
            "As for the mass of streets, squares, and public buildings "
            "which parted them, she only felt at this moment how little "
            "London had done to make her love it, although thirty of her "
            "forty years had been spent in a street. She knew how to read "
            "the people who were passing her; there were the rich who were "
            "running to and from each others' houses at this hour; there "
            "were the bigoted workers driving in a straight line to their "
            "offices; there were the poor who were unhappy and rightly "
            "malignant."
        ),
    },
    {
        "key": "voyageout_punctual_1",
        "source_file": "the_voyage_out.txt",
        "aspect": "punctual",
        "passage": (
            "But this lady looked neither up nor down; the only thing she "
            "had seen, since she stood there, was a circular iridescent "
            "patch slowly floating past with a straw in the middle of it. "
            "The straw and the patch swam again and again behind the "
            "tremulous medium of a great welling tear, and the tear rose "
            "and fell and dropped into the river. Yes, she knew she must go "
            "back to all that, but at present she must weep."
        ),
    },
    # --- Between the Acts ---
    {
        "key": "betweentheacts_iterative_1",
        "source_file": "between_the_acts.txt",
        "aspect": "iterative",
        "passage": (
            "She had met him at a Bazaar; and at a tennis party. He had "
            "handed her a cup and a racquet—that was all. But in his "
            "ravaged face she always felt mystery; and in his silence, "
            "passion. At the tennis party she had felt this, and at the "
            "Bazaar. Now a third time, if anything more strongly, she felt "
            "it again."
        ),
    },
    {
        "key": "betweentheacts_punctual_1",
        "source_file": "between_the_acts.txt",
        "aspect": "punctual",
        "passage": (
            "Isa raised her head. The words made two rings, perfect rings, "
            "that floated them, herself and Haines, like two swans down "
            "stream. But his snow-white breast was circled with a tangle of "
            "dirty duckweed; and she too, in her webbed feet was entangled, "
            "by her husband, the stockbroker."
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

    out_path = Path(__file__).parent / "cross_model_aspect_results.jsonl"
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
