"""Penelope — generate interpretive <think> traces for training data.

Optional utility: pre-generate SFT traces using a frontier model before
fine-tuning. In the default training pipeline this step is SKIPPED — GRPO
bootstraps trace quality directly from the Qwen3-72B reward signal, so
pre-generated traces are not required.

Use this script only if you want to inspect or curate traces before SFT,
or if you want to compare SFT-then-GRPO against GRPO-only.

Usage
-----
    python scripts/generate_traces.py \\
        --dataset dataset/ \\
        --output training_data/traces/ \\
        [--model <label-from-models.yaml>] \\
        [--candidates 6] \\
        [--temperature 0.8] \\
        [--split train]          # train | val | test | all

Pipeline
--------
1. For each record in the split, call the configured model k times
   (temperature=0.8) asking for an interpretive trace in the format
   defined in SKILL.md.
2. Score each candidate on grounding + typological specificity
   (both automatic, no API call needed).
3. Select the highest-scoring trace and write the rebuilt record to
   training_data/traces/<split>/<passage_id>.jsonl.
4. Re-run build_dataset.py --traces training_data/traces/ to fold the
   new traces back into dataset/*.jsonl.

Note: negative records (is_soc=false) are prioritised — their current
<think> content is boilerplate. Positive records with existing rich traces
are also regenerated for consistency.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Taxonomy keywords for automatic specificity scoring
_TAXONOMY_KEYWORDS = {
    "direct_interior_monologue",
    "indirect_interior_monologue",
    "omniscient_description",
    "soliloquy",
    "free_association",
    "space_montage",
    "orthographic_marker",
    "imagery",
    "simulation_state_of_mind",
    "reverie_fantasy",
    "hybrid",
    "other_soc",
    # Humphrey/Steinberg proper names as additional signal
    "humphrey",
    "steinberg",
    "free indirect",
    "interior monologue",
}

# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

_TRACE_GENERATION_PROMPT = """\
You are a literary analyst specialising in stream-of-consciousness technique.
Analyse the passage below and produce a reasoning trace in the interpretive format.

## Interpretive reasoning trace format

Work through each step in prose. Do NOT use numbered headers — write flowing
deliberation that moves through these questions naturally:

1. First reading: describe the immediate texture — voice, syntax, register.
   What is the reader's relationship to interiority here?
2. Candidate types: name 1-3 types from the taxonomy that could plausibly apply.
   For each, cite a verbatim phrase from the passage.
3. Strongest reading: argue for the primary type. What makes it the most
   productive lens, not just the most accurate label?
4. Counter-reading: what would a reasonable alternative say? Why does your
   reading hold up against it?
5. Skepticism gate: could this passage appear in a non-SoC text? If yes,
   what specifically marks it as interior consciousness? If you cannot answer,
   conclude is_soc: false.

The known classification for this passage is: {label}

Keep the trace to 150-250 words. Do not repeat the passage verbatim at length.
Output ONLY the trace text — no JSON, no headers, no preamble.

## Passage
{passage}
"""


def _build_prompt(passage: str, label: str) -> str:
    return _TRACE_GENERATION_PROMPT.format(passage=passage.strip(), label=label)


# ---------------------------------------------------------------------------
# Automatic scoring
# ---------------------------------------------------------------------------


def _grounding_score(trace: str, passage: str) -> float:
    """Score 0/0.5/1 based on verbatim phrase presence in passage."""
    # Extract quoted phrases (text between " or ' that is ≥4 words)
    quotes = re.findall(r'["\u2018\u2019\u201c\u201d]([^"\']+)["\u2018\u2019\u201c\u201d]', trace)
    passage_lower = passage.lower()
    for q in quotes:
        if len(q.split()) >= 3 and q.strip().lower() in passage_lower:
            return 1.0
    # Partial: any 4-word run from the trace appears in the passage
    words = trace.split()
    for i in range(len(words) - 3):
        fragment = " ".join(words[i : i + 4]).lower()
        fragment = re.sub(r"[^\w\s]", "", fragment)
        if fragment in passage_lower:
            return 0.5
    return 0.0


def _specificity_score(trace: str) -> float:
    """Score 0/0.5/1 based on taxonomy keyword presence."""
    trace_lower = trace.lower()
    hits = sum(1 for kw in _TAXONOMY_KEYWORDS if kw in trace_lower)
    if hits >= 2:
        return 1.0
    if hits == 1:
        return 0.5
    return 0.0


def _skepticism_score(trace: str, is_soc: bool) -> float:
    """Heuristic: does the trace mention the skepticism gate concern?"""
    lower = trace.lower()
    skepticism_phrases = [
        "non-soc", "not soc", "could appear", "conventional narration",
        "narrated psychology", "narrator retains", "psycho-narration",
        "would need", "falls short", "does not qualify", "narrator's voice",
        "narrator describes", "no interior", "no unmediated",
    ]
    hits = sum(1 for p in skepticism_phrases if p in lower)
    if is_soc:
        # For positives, reward any engagement with the skepticism question
        return min(1.0, hits * 0.4)
    else:
        # For negatives, higher bar — must articulate why it fails
        return min(1.0, hits * 0.5)


def score_candidate(trace: str, passage: str, is_soc: bool) -> float:
    """Combined automatic score for a candidate trace."""
    g = _grounding_score(trace, passage)
    s = _specificity_score(trace)
    k = _skepticism_score(trace, is_soc)
    return 0.4 * g + 0.35 * s + 0.25 * k


# ---------------------------------------------------------------------------
# API call (reuses models.yaml config)
# ---------------------------------------------------------------------------


def _call_model(
    client: Any,
    model_name: str,
    prompt: str,
    temperature: float,
    n: int,
) -> list[str]:
    """Call the model and return n candidate traces."""
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        n=n,
        max_tokens=512,
    )
    return [choice.message.content.strip() for choice in response.choices]


# ---------------------------------------------------------------------------
# Record processing
# ---------------------------------------------------------------------------


def _passage_id(record: dict, split: str) -> str:
    """Stable ID for a record based on passage content hash."""
    digest = hashlib.md5(record.get("passage", "").encode()).hexdigest()[:10]
    chunk = record.get("chunk_id", "unk").replace("/", "_")
    return f"{split}_{chunk}_{digest}"


def process_record(
    record: dict,
    client: Any,
    model_name: str,
    n_candidates: int,
    temperature: float,
    passage_id: str,
    traces_dir: Path,
    force: bool = False,
) -> dict:
    """Generate traces for one record; return the record with best trace injected."""
    out_path = traces_dir / f"{passage_id}.jsonl"

    passage = record.get("passage", "")
    is_soc = record.get("is_soc", False)
    label = "is_soc: true, soc_type: " + record.get("soc_type", "unknown") if is_soc else "is_soc: false"

    # Load cached candidates if they exist
    candidates: list[str] = []
    if out_path.exists() and not force:
        with out_path.open(encoding="utf-8") as fh:
            for line in fh:
                obj = json.loads(line)
                candidates.append(obj.get("trace", ""))
        logger.debug("Loaded %d cached candidates for %s", len(candidates), passage_id)
    else:
        prompt = _build_prompt(passage, label)
        candidates = _call_model(client, model_name, prompt, temperature, n_candidates)
        # Write candidates to disk for audit
        traces_dir.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as fh:
            for trace in candidates:
                fh.write(json.dumps({"trace": trace, "passage_id": passage_id}) + "\n")
        logger.debug("Generated %d candidates for %s", len(candidates), passage_id)

    # Select best by automatic score
    scored = [(score_candidate(t, passage, is_soc), t) for t in candidates if t.strip()]
    if not scored:
        logger.warning("No valid candidates for %s — keeping original trace", passage_id)
        return record

    best_score, best_trace = max(scored, key=lambda x: x[0])
    logger.debug("Best score %.3f for %s", best_score, passage_id)

    # Inject the new trace
    updated = record.copy()
    updated["think_content"] = best_trace
    updated["trace_score"] = best_score
    return updated


# ---------------------------------------------------------------------------
# JSONL record reconstruction for dataset rebuild
# ---------------------------------------------------------------------------


def record_to_messages(record: dict, system_prompt: str) -> dict:
    """Convert a record with updated think_content to messages format."""
    think_block = record.get("think_content", "")
    assistant_json = record.get("assistant_json", '{"instances": []}')
    if think_block:
        assistant_content = f"<think>\n{think_block}\n</think>\n{assistant_json}"
    else:
        assistant_content = assistant_json

    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": record.get("passage", "")},
            {"role": "assistant", "content": assistant_content},
        ],
        # Preserve metadata for downstream inspection
        "_passage_id": record.get("_passage_id", ""),
        "_is_soc": record.get("is_soc", False),
        "_trace_score": record.get("trace_score", None),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate interpretive <think> traces for training data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset", type=Path, default=Path("dataset"),
        help="Directory with train.jsonl / val.jsonl / test.jsonl (default: dataset/)",
    )
    parser.add_argument(
        "--output", "-o", type=Path, default=Path("training_data/traces"),
        help="Directory to store candidate traces (default: training_data/traces/)",
    )
    parser.add_argument(
        "--rebuilt", type=Path, default=None,
        help="Write rebuilt JSONL here (default: dataset/<split>_traces.jsonl)",
    )
    parser.add_argument(
        "--model", default=None,
        help="Model label from models.yaml to use as trace generator. "
             "Defaults to the first model in the config.",
    )
    parser.add_argument(
        "--config", type=Path, default=Path("models.yaml"),
        help="Path to models.yaml (default: models.yaml)",
    )
    parser.add_argument(
        "--split", default="train",
        choices=["train", "val", "test", "all"],
        help="Which split(s) to process (default: train)",
    )
    parser.add_argument(
        "--candidates", "-n", type=int, default=6,
        help="Number of candidate traces to generate per passage (default: 6)",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.8,
        help="Sampling temperature for trace generation (default: 0.8)",
    )
    parser.add_argument(
        "--negatives-first", action="store_true", default=True,
        help="Process negative records before positive ones (default: True)",
    )
    parser.add_argument(
        "--force", action="store_true", default=False,
        help="Regenerate traces even if cached candidates exist.",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Process at most N records (useful for testing).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
    )

    # ── Load config ────────────────────────────────────────────────────
    import yaml
    with args.config.open(encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    model_cfgs = cfg.get("models", [])
    if not model_cfgs:
        logger.error("No models found in %s", args.config)
        sys.exit(1)

    # Select model
    if args.model:
        model_cfg = next((m for m in model_cfgs if m["label"] == args.model), None)
        if not model_cfg:
            logger.error("Model '%s' not found in %s", args.model, args.config)
            sys.exit(1)
    else:
        model_cfg = model_cfgs[0]
    logger.info("Using model: %s", model_cfg["label"])

    # ── Build OpenAI client ────────────────────────────────────────────
    try:
        import os
        from openai import OpenAI
    except ImportError:
        logger.error("openai package required: pip install openai")
        sys.exit(1)

    api_key = model_cfg.get("api_key", "")
    # Resolve env var references like ${OPENAI_API_KEY}
    env_match = re.match(r"\$\{(\w+)\}", api_key)
    if env_match:
        api_key = os.environ.get(env_match.group(1), "not-set")

    client = OpenAI(
        api_key=api_key,
        base_url=model_cfg.get("base_url"),
    )

    # ── Load system prompt ─────────────────────────────────────────────
    from scripts.prompts import TRAINING_SYSTEM_PROMPT

    # ── Process splits ─────────────────────────────────────────────────
    splits = ["train", "val", "test"] if args.split == "all" else [args.split]

    for split in splits:
        jsonl_path = args.dataset / f"{split}.jsonl"
        if not jsonl_path.exists():
            logger.warning("Skipping %s — not found", jsonl_path)
            continue

        logger.info("Loading %s ...", jsonl_path)
        records: list[dict] = []
        with jsonl_path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

        # Reconstruct raw record fields from messages format
        raw_records: list[dict] = []
        for rec in records:
            msgs = rec.get("messages", [])
            user_msg = next((m["content"] for m in msgs if m["role"] == "user"), "")
            asst_msg = next((m["content"] for m in msgs if m["role"] == "assistant"), "")

            think_match = re.search(r"<think>(.*?)</think>", asst_msg, re.DOTALL)
            think_content = think_match.group(1).strip() if think_match else ""
            assistant_json = re.sub(r"<think>.*?</think>", "", asst_msg, flags=re.DOTALL).strip()

            # Determine is_soc from JSON
            try:
                parsed = json.loads(assistant_json)
                is_soc = bool(parsed.get("instances"))
                instances = parsed.get("instances", [])
                soc_type = instances[0].get("soc_type", "") if instances else ""
            except Exception:
                is_soc = False
                soc_type = ""

            raw_records.append({
                "passage": user_msg,
                "is_soc": is_soc,
                "soc_type": soc_type,
                "think_content": think_content,
                "assistant_json": assistant_json,
                "chunk_id": rec.get("_chunk_id", ""),
            })

        # Sort: negatives first (most important to fix)
        if args.negatives_first:
            raw_records.sort(key=lambda r: (r["is_soc"], 0))

        if args.limit:
            raw_records = raw_records[: args.limit]

        logger.info(
            "Processing %d records (%d negative, %d positive)",
            len(raw_records),
            sum(1 for r in raw_records if not r["is_soc"]),
            sum(1 for r in raw_records if r["is_soc"]),
        )

        split_traces_dir = args.output / split
        rebuilt_records: list[dict] = []

        for i, rec in enumerate(raw_records):
            pid = _passage_id(rec, split)
            rec["_passage_id"] = pid
            try:
                updated = process_record(
                    record=rec,
                    client=client,
                    model_name=model_cfg["model_name"],
                    n_candidates=args.candidates,
                    temperature=args.temperature,
                    passage_id=pid,
                    traces_dir=split_traces_dir,
                    force=args.force,
                )
            except Exception as exc:
                logger.warning("Failed on record %d (%s): %s", i, pid, exc)
                updated = rec

            rebuilt_records.append(record_to_messages(updated, TRAINING_SYSTEM_PROMPT))

            if (i + 1) % 50 == 0:
                logger.info("  %d / %d processed", i + 1, len(raw_records))

        # Write rebuilt JSONL
        out_path = args.rebuilt or (args.dataset / f"{split}_traces.jsonl")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as fh:
            for r in rebuilt_records:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")

        logger.info("Wrote %d records to %s", len(rebuilt_records), out_path)

    logger.info("Done. Review traces in %s before re-training.", args.output)
    logger.info(
        "To rebuild dataset: python scripts/build_dataset.py "
        "--traces %s", args.output
    )


if __name__ == "__main__":
    main()
