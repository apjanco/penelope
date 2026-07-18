"""Resample silver-label passages for within-model consistency estimation.

Loads all extracted passages from silver_data/*.json, deduplicates them by
token-overlap (the same grouping logic used in consensus.py), then re-classifies
each unique passage independently k times per model.

Output: silver_data/resampled/<model_label>.jsonl
  One JSON object per line per (passage_key, run_index).
  The file is append-friendly — re-running resumes from where it left off.

Usage
-----
    # All models, all passages, k=10
    python scripts/resample.py

    # Specific model(s) only
    python scripts/resample.py --model mercury-2 --model qwen3-max

    # Limit to one source file (useful for testing)
    python scripts/resample.py --source-file as_I_lay_dying.txt

    # Dry run — show stats without calling APIs
    python scripts/resample.py --dry-run

    # Custom k (overrides per-model defaults)
    python scripts/resample.py --k 5

    # Custom temperature for variable-temp models (default 0.8)
    python scripts/resample.py --temperature 0.9

Notes
-----
- gpt-5.4 rejects custom temperature; it always runs at k=1 (default temp).
  Within-model consistency for gpt-5.4 is therefore treated as 1.0 by convention
  in build_ambiguity_dataset.py.
- Deduplication uses MIN_OVERLAP_RATIO=0.35 (same as consensus.py). Within each
  (source_file, chunk_id) group, passages with token overlap >= 0.35 are merged
  into one cluster; the longest passage in the cluster becomes the canonical text.
- passage_key format: "<chunk_id>__p<cluster_index>" (e.g. "as-i-lay-dying_000__p0")
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)

# Prepend project root so we can import scripts.*
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.config import Config, ModelProfile
from scripts.prompts import TRAINING_SYSTEM_PROMPT

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MIN_OVERLAP_RATIO = 0.35   # mirrors consensus.py

DEFAULT_K = 10             # runs per passage for variable-temperature models
GPT54_K = 1                # gpt-5.4 does not support custom temperature
DEFAULT_TEMP = 0.8         # sampling temperature for variable-temp models

# Models that cannot accept a custom temperature parameter
NO_CUSTOM_TEMP_MODELS = {"gpt-5.4", "gpt-5"}

# ---------------------------------------------------------------------------
# Text normalisation and token-overlap (mirrors consensus.py)
# ---------------------------------------------------------------------------

def _normalise_text(text: str) -> list[str]:
    """Lowercase, strip punctuation, split on whitespace -> token list."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return text.split()


def _token_overlap(a: str, b: str) -> float:
    """Return token overlap ratio: |A ∩ B| / min(|A|, |B|)."""
    ta = set(_normalise_text(a))
    tb = set(_normalise_text(b))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / min(len(ta), len(tb))


# ---------------------------------------------------------------------------
# Passage deduplication
# ---------------------------------------------------------------------------

def _build_canonical_passages(silver_dir: Path) -> list[dict]:
    """Load all silver JSON files and return a deduplicated passage list.

    Within each (source_file, chunk_id) group, passages are clustered by
    token-overlap >= MIN_OVERLAP_RATIO.  The longest passage in each cluster
    becomes the canonical text. passage_key is assigned as
    "<chunk_id>__p<cluster_index>".

    Returns a list of dicts:
        {passage_key, chunk_id, source_file, passage}
    """
    # Collect all records from all silver JSON files
    raw: list[dict] = []
    for f in sorted(silver_dir.glob("*.json")):
        if f.stem in ("results", "consensus"):
            continue
        try:
            records = json.loads(f.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Could not load %s: %s", f, exc)
            continue
        if not isinstance(records, list):
            continue
        for r in records:
            if isinstance(r, dict) and r.get("passage"):
                raw.append(r)

    logger.info("Loaded %d raw records from %s", len(raw), silver_dir)

    # Group by (source_file, chunk_id)
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in raw:
        key = (r.get("source_file", ""), r.get("chunk_id", ""))
        groups[key].append(r)

    canonical: list[dict] = []

    for (source_file, chunk_id), records in sorted(groups.items()):
        # Greedy token-overlap clustering: sort by passage length desc
        sorted_records = sorted(records, key=lambda r: len(r["passage"]), reverse=True)
        clusters: list[list[dict]] = []

        for rec in sorted_records:
            placed = False
            for cluster in clusters:
                rep = cluster[0]  # longest passage is the representative
                if _token_overlap(rec["passage"], rep["passage"]) >= MIN_OVERLAP_RATIO:
                    cluster.append(rec)
                    placed = True
                    break
            if not placed:
                clusters.append([rec])

        for idx, cluster in enumerate(clusters):
            # Canonical passage = longest in the cluster
            canonical_rec = max(cluster, key=lambda r: len(r["passage"]))
            passage_key = f"{chunk_id}__p{idx}"
            canonical.append({
                "passage_key": passage_key,
                "chunk_id": chunk_id,
                "source_file": source_file,
                "passage": canonical_rec["passage"],
            })

    logger.info(
        "Deduplicated to %d canonical passages from %d groups",
        len(canonical), len(groups),
    )
    return canonical


# ---------------------------------------------------------------------------
# Resume helpers
# ---------------------------------------------------------------------------

def _load_done(output_file: Path) -> set[tuple[str, int]]:
    """Return set of (passage_key, run_index) already written to JSONL."""
    if not output_file.exists():
        return set()
    done: set[tuple[str, int]] = set()
    with output_file.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                done.add((rec["passage_key"], rec["run_index"]))
            except Exception:
                pass
    return done


# ---------------------------------------------------------------------------
# Classification API call
# ---------------------------------------------------------------------------

def _classify_passage(
    profile: ModelProfile,
    passage: str,
    source_file: str,
    temperature: float | None,
    *,
    max_retries: int = 3,
    retry_delay: float = 5.0,
) -> dict | None:
    """Ask one model to classify a single passage.  Returns parsed record or None."""
    try:
        from openai import OpenAI, APIError, RateLimitError
    except ImportError:
        logger.error("openai package not installed. Run: pip install openai>=1.0.0")
        sys.exit(1)

    user_content = (
        f"Source: {source_file}\n\n"
        "Classify the following passage. "
        "If it contains stream-of-consciousness, return a JSON 'instances' list with one entry. "
        "If it does not, return {\"instances\": []}. Respond with JSON only.\n\n"
        f"{passage}"
    )

    client = OpenAI(base_url=profile.base_url, api_key=profile.api_key)
    kwargs: dict = dict(
        model=profile.model_name,
        messages=[
            {"role": "system", "content": TRAINING_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        response_format={"type": "json_object"},
    )
    if temperature is not None:
        kwargs["temperature"] = temperature

    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(**kwargs)
            raw = response.choices[0].message.content or ""
            return _parse_classification(raw)
        except RateLimitError as exc:
            wait = retry_delay * (2 ** (attempt - 1))
            logger.warning(
                "Rate limit on %s (attempt %d/%d). Retrying in %.0fs…",
                profile.label, attempt, max_retries, wait,
            )
            time.sleep(wait)
            last_exc = exc
        except APIError as exc:
            # 400 errors (content filter, bad request) are deterministic —
            # retrying will never succeed.
            if getattr(exc, "status_code", None) == 400:
                logger.warning(
                    "Non-retriable 400 from %s (content filter or bad request): %s",
                    profile.label, exc,
                )
                last_exc = exc
                break
            logger.warning(
                "API error on %s (attempt %d/%d): %s",
                profile.label, attempt, max_retries, exc,
            )
            time.sleep(retry_delay)
            last_exc = exc
        except Exception as exc:
            logger.warning("Unexpected error on %s: %s", profile.label, exc)
            last_exc = exc
            break

    logger.error(
        "Failed to get response from %s after %d attempts: %s",
        profile.label, max_retries, last_exc,
    )
    return None


def _parse_classification(raw: str) -> dict | None:
    """Parse a single-passage classification response into a normalised dict."""
    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
    cleaned = re.sub(r"```(?:json)?", "", cleaned).strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        logger.warning("JSON parse error: %s", exc)
        return None

    instances = data.get("instances") or data.get("soc_instances") or []
    if not isinstance(instances, list):
        return None

    # A passage-level call should return 0 or 1 instances.
    # If the model returns none → soc_type = "none"
    if not instances:
        return {"soc_type": "none", "secondary_devices": [], "confidence": "high",
                "explanation": "", "evidence": []}

    item = instances[0]
    if not isinstance(item, dict):
        return None

    devices = item.get("secondary_devices", [])
    if isinstance(devices, str):
        devices = [d.strip() for d in devices.split(",") if d.strip()]
    evidence = item.get("evidence", [])
    if isinstance(evidence, str):
        evidence = [e.strip() for e in evidence.split(",") if e.strip()]

    return {
        "soc_type":          str(item.get("soc_type", "none")),
        "secondary_devices": devices,
        "confidence":        str(item.get("confidence", "medium")),
        "explanation":       str(item.get("explanation", "")),
        "evidence":          evidence,
    }


# ---------------------------------------------------------------------------
# Main resampling loop
# ---------------------------------------------------------------------------

def resample(
    passages: list[dict],
    profiles: list[ModelProfile],
    output_dir: Path,
    *,
    k: int = DEFAULT_K,
    temperature: float = DEFAULT_TEMP,
    source_file_filter: str | None = None,
    dry_run: bool = False,
) -> None:
    """Run resampling for all (passage, model) pairs."""
    if source_file_filter:
        passages = [p for p in passages if p["source_file"] == source_file_filter]
        logger.info("Filtered to %d passages for source_file=%s", len(passages), source_file_filter)

    output_dir.mkdir(parents=True, exist_ok=True)

    for profile in profiles:
        output_file = output_dir / f"{profile.label}.jsonl"
        done = _load_done(output_file)

        # gpt-5.4 and similar models can't use custom temperature
        is_no_temp = profile.label in NO_CUSTOM_TEMP_MODELS
        model_k = GPT54_K if is_no_temp else k
        model_temp = None if is_no_temp else temperature

        total_calls = len(passages) * model_k
        already_done = sum(
            1 for p in passages
            for run_idx in range(model_k)
            if (p["passage_key"], run_idx) in done
        )
        pending_calls = total_calls - already_done

        logger.info(
            "%s: %d passages × k=%d = %d calls (%d done, %d pending)",
            profile.label, len(passages), model_k, total_calls, already_done, pending_calls,
        )

        if dry_run:
            continue

        if pending_calls == 0:
            logger.info("%s: all passages already resampled, skipping", profile.label)
            continue

        profile.validate()

        completed = 0
        with output_file.open("a", encoding="utf-8") as fh:
            for passage_info in passages:
                for run_idx in range(model_k):
                    if (passage_info["passage_key"], run_idx) in done:
                        continue

                    result = _classify_passage(
                        profile,
                        passage_info["passage"],
                        passage_info["source_file"],
                        model_temp,
                    )

                    record: dict = {
                        "passage_key":      passage_info["passage_key"],
                        "chunk_id":         passage_info["chunk_id"],
                        "run_index":        run_idx,
                        "model_label":      profile.label,
                        "source_file":      passage_info["source_file"],
                        "passage":          passage_info["passage"],
                        "temperature":      model_temp,
                    }
                    if result is not None:
                        record.update(result)
                    else:
                        # API failure — record a sentinel so we can retry later
                        record["soc_type"] = None
                        record["error"] = True

                    fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                    fh.flush()
                    done.add((passage_info["passage_key"], run_idx))
                    completed += 1

                    if completed % 100 == 0:
                        logger.info(
                            "%s: %d / %d calls completed",
                            profile.label, completed, pending_calls,
                        )

        logger.info("%s: finished — %d calls written to %s", profile.label, completed, output_file)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Resample silver passages for consistency estimation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--silver-dir",
        type=Path,
        default=Path("silver_data"),
        help="Directory containing per-model silver JSON files (default: silver_data/)",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("silver_data/resampled"),
        help="Output directory for resampled JSONL files (default: silver_data/resampled/)",
    )
    p.add_argument(
        "--models-yaml",
        type=Path,
        default=Path("models.yaml"),
        help="Path to models.yaml (default: models.yaml)",
    )
    p.add_argument(
        "--model",
        dest="models",
        action="append",
        metavar="LABEL",
        help="Restrict to this model label (repeatable; default: all models in models.yaml)",
    )
    p.add_argument(
        "--source-file",
        metavar="FILENAME",
        default=None,
        help="Restrict to passages from this source_file (e.g. as_I_lay_dying.txt)",
    )
    p.add_argument(
        "--k",
        type=int,
        default=DEFAULT_K,
        help=f"Resample runs per passage for variable-temperature models (default: {DEFAULT_K}). "
             f"gpt-5.4 always uses k=1.",
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMP,
        help=f"Sampling temperature for variable-temperature models (default: {DEFAULT_TEMP})",
    )
    p.add_argument(
        "--env-file",
        type=Path,
        default=None,
        help="Path to .env file for API keys",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print stats and exit without making API calls",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging",
    )
    return p


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    # ── Load config ──────────────────────────────────────────────────
    cfg = Config.load(env_file=args.env_file, config_file=args.models_yaml)
    profiles: list[ModelProfile] = cfg.models
    if not profiles:
        logger.error("No model profiles found. Check models.yaml and .env.")
        sys.exit(1)

    if args.models:
        label_set = set(args.models)
        profiles = [p for p in profiles if p.label in label_set]
        if not profiles:
            logger.error("No matching profiles for: %s", args.models)
            sys.exit(1)

    logger.info("Models: %s", ", ".join(p.label for p in profiles))

    # ── Build canonical passage list ─────────────────────────────────
    canonical = _build_canonical_passages(args.silver_dir)
    if not canonical:
        logger.error("No passages found in %s", args.silver_dir)
        sys.exit(1)

    if args.dry_run:
        sources = sorted(set(p["source_file"] for p in canonical))
        print(f"\n[DRY RUN] {len(canonical)} canonical passages")
        print(f"Models: {', '.join(p.label for p in profiles)}")
        print(f"k (variable-temp models): {args.k}  |  temperature: {args.temperature}")
        print(f"k (gpt-5.4): {GPT54_K}  |  temperature: default")
        print(f"Output dir: {args.output_dir.resolve()}")
        print(f"\nSource files ({len(sources)}):")
        for s in sources:
            n = sum(1 for p in canonical if p["source_file"] == s)
            print(f"  {n:5d}  {s}")
        total_variable = sum(
            len(canonical) * args.k
            for p in profiles if p.label not in NO_CUSTOM_TEMP_MODELS
        )
        total_fixed = sum(
            len(canonical) * GPT54_K
            for p in profiles if p.label in NO_CUSTOM_TEMP_MODELS
        )
        print(f"\nTotal API calls: ~{total_variable + total_fixed:,}")
        return

    # ── Run ──────────────────────────────────────────────────────────
    resample(
        canonical,
        profiles,
        args.output_dir,
        k=args.k,
        temperature=args.temperature,
        source_file_filter=args.source_file,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
