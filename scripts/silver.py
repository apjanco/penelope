"""Generate silver-label training data by running external LLMs on chunked texts.

Reads pre-chunked files from chunking/ (produced by chunk.py), calls each
configured model via its OpenAI-compatible API, and appends results to
training_data/<model_label>.json.  Existing records for the same chunk_id
are skipped, so the script is safe to re-run or interrupt and resume.

Usage
-----
    # Run all models in models.yaml on all files in chunking/
    python scripts/silver.py --input chunking/

    # Single file, specific models only
    python scripts/silver.py --input chunking/mrs_dalloway.txt \\
        --model gpt-5 --model gemini-3-pro-preview

    # Custom output directory (default: training_data/)
    python scripts/silver.py --input chunking/ --output-dir training_data/

    # Dry run — list chunks without calling APIs
    python scripts/silver.py --input chunking/ --dry-run

    # Use a specific .env file
    python scripts/silver.py --input chunking/ --env-file .env.production

Prerequisites
-------------
    pip install openai>=1.0.0
    cp .env.example .env   # then fill in your API keys
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)

# Prepend project root so we can import scripts.*
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.config import Config, ModelProfile
from scripts.prompts import TRAINING_SYSTEM_PROMPT
from scripts.soc_chunker import is_chunked_file, parse_chunked_dir, parse_chunked_file
from scripts.models import Chunk


# ---------------------------------------------------------------------------
# API call
# ---------------------------------------------------------------------------

def _call_api(
    profile: ModelProfile,
    chunk: Chunk,
    *,
    max_retries: int = 3,
    retry_delay: float = 5.0,
) -> list[dict]:
    """Call one LLM and return a list of silver-label record dicts."""
    try:
        from openai import OpenAI, APIError, RateLimitError
    except ImportError:
        logger.error("openai package not installed. Run: pip install openai>=1.0.0")
        sys.exit(1)

    user_content = (
        f"Source: {chunk.source_file}\n"
        f"Chunk: {chunk.chunk_id} ({chunk.chunk_label})\n"
        f"Position: {chunk.chunk_index}\n"
    )
    if chunk.context_before:
        user_content += f"\n[CONTEXT — end of previous chunk]\n{chunk.context_before}\n"
    user_content += f"\n[TEXT TO ANALYZE]\n{chunk.chunk_text}\n"
    if chunk.context_after:
        user_content += f"\n[CONTEXT — start of next chunk]\n{chunk.context_after}\n"
    user_content += (
        "\nAnalyze the [TEXT TO ANALYZE] section above. "
        "Identify and classify every SoC passage. "
        "Respond with JSON only."
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
    if profile.temperature is not None:
        kwargs["temperature"] = profile.temperature

    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(**kwargs)
            raw = response.choices[0].message.content or ""
            return _parse_instances(raw, profile.label, chunk)
        except RateLimitError as exc:
            wait = retry_delay * (2 ** (attempt - 1))
            logger.warning(
                "Rate limit on %s (attempt %d/%d). Retrying in %.0fs…",
                profile.label, attempt, max_retries, wait,
            )
            time.sleep(wait)
            last_exc = exc
        except APIError as exc:
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
        "Failed to get response from %s for chunk %s after %d attempts: %s",
        profile.label, chunk.chunk_id, max_retries, last_exc,
    )
    return []


def _parse_instances(raw: str, model_label: str, chunk: Chunk) -> list[dict]:
    """Parse the JSON response into a list of silver-label record dicts."""
    import re

    # Strip any markdown fences or think blocks the model may emit
    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
    cleaned = re.sub(r"```(?:json)?", "", cleaned).strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        logger.warning("JSON parse error for %s / %s: %s", model_label, chunk.chunk_id, exc)
        return []

    instances_raw = data.get("instances") or data.get("soc_instances") or []
    if not isinstance(instances_raw, list):
        logger.warning("Unexpected 'instances' shape from %s: %s", model_label, type(instances_raw))
        return []

    records = []
    for item in instances_raw:
        if not isinstance(item, dict):
            continue
        if not item.get("is_soc", True):
            continue  # skip explicit non-SoC entries

        # Normalise list fields to comma-separated strings to match existing schema
        devices = item.get("secondary_devices", "")
        if isinstance(devices, list):
            devices = ", ".join(str(d) for d in devices)
        evidence = item.get("evidence", "")
        if isinstance(evidence, list):
            evidence = ", ".join(str(e) for e in evidence)

        records.append({
            "model_label":       model_label,
            "source_file":       chunk.source_file,
            "chunk_id":          chunk.chunk_id,
            "chunk_label":       chunk.chunk_label,
            "chunk_index":       chunk.chunk_index,
            "passage":           str(item.get("passage", "")),
            "soc_type":          str(item.get("soc_type", "")),
            "secondary_devices": devices,
            "affective_register": str(item.get("affective_register", "n/a")),
            "narrator_position": str(item.get("narrator_position", "")),
            "character_pov":     str(item.get("character_pov", "")),
            "explanation":       str(item.get("explanation", "")),
            "evidence":          evidence,
            "confidence":        str(item.get("confidence", "medium")),
            "notes":             str(item.get("notes", "")),
        })

    return records


# ---------------------------------------------------------------------------
# Per-model output file helpers
# ---------------------------------------------------------------------------

def _load_existing(output_file: Path) -> tuple[list[dict], set[str]]:
    """Load existing records and return (records, set_of_chunk_ids)."""
    if not output_file.exists():
        return [], set()
    try:
        records = json.loads(output_file.read_text(encoding="utf-8"))
        if not isinstance(records, list):
            records = []
    except Exception as exc:
        logger.warning("Could not load %s: %s — starting fresh", output_file, exc)
        records = []
    seen = {r.get("chunk_id", "") for r in records}
    return records, seen


def _save(output_file: Path, records: list[dict]) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Generate silver-label training data via external LLM APIs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/silver.py --input chunking/\n"
            "  python scripts/silver.py --input chunking/ --model gpt-5 --model gemini-3-pro-preview\n"
            "  python scripts/silver.py --input chunking/ --dry-run\n"
        ),
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        type=Path,
        help="Chunked file or directory (from chunk.py) with <chunk-N> markup",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=Path("training_data"),
        help="Directory to write <model_label>.json files (default: training_data/)",
    )
    parser.add_argument(
        "--model", "-m",
        action="append",
        dest="models",
        metavar="LABEL",
        help="Run only this model label (from models.yaml). Repeatable. Default: all.",
    )
    parser.add_argument(
        "--config", "-c",
        type=Path,
        default=None,
        help="Path to models.yaml (default: auto-detect in project root)",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=None,
        help="Path to .env file (default: auto-detect in project root)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Seconds to wait between API calls per model (default: 1.0)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List chunks without making API calls",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-annotate chunks that are already in the output file",
    )
    args = parser.parse_args()

    # ── Load config ──────────────────────────────────────────────────
    config = Config.load(env_file=args.env_file, config_file=args.config)
    profiles = config.get_model_profiles()

    if args.models:
        requested = set(args.models)
        profiles = [p for p in profiles if p.label in requested]
        unknown = requested - {p.label for p in profiles}
        if unknown:
            logger.error(
                "Unknown model label(s): %s. Available: %s",
                ", ".join(sorted(unknown)),
                ", ".join(p.label for p in config.get_model_profiles()),
            )
            sys.exit(1)

    if not profiles:
        logger.error("No model profiles found. Check models.yaml and .env.")
        sys.exit(1)

    logger.info("Models: %s", ", ".join(p.label for p in profiles))

    # ── Load chunks ──────────────────────────────────────────────────
    input_path = args.input
    if input_path.is_file():
        if not is_chunked_file(input_path):
            logger.error(
                "%s has no <chunk-N> markup. Run chunk.py first.", input_path
            )
            sys.exit(1)
        chunks = parse_chunked_file(input_path)
    elif input_path.is_dir():
        chunks = parse_chunked_dir(input_path)
    else:
        logger.error("Input path not found: %s", input_path)
        sys.exit(1)

    if not chunks:
        logger.error("No chunks found in %s", input_path)
        sys.exit(1)

    logger.info("Chunks loaded: %d", len(chunks))

    if args.dry_run:
        print(f"\n[DRY RUN] {len(chunks)} chunks × {len(profiles)} models "
              f"= {len(chunks) * len(profiles)} API calls")
        print(f"Output dir: {args.output_dir.resolve()}")
        print(f"Models: {', '.join(p.label for p in profiles)}")
        for c in chunks:
            print(f"  {c.chunk_id:40s}  {len(c.chunk_text):,} chars")
        return

    # ── Validate API keys before making any calls ────────────────────
    for profile in profiles:
        try:
            profile.validate()
        except ValueError as exc:
            logger.error("%s", exc)
            sys.exit(1)

    # ── Run each model ───────────────────────────────────────────────
    for profile in profiles:
        output_file = args.output_dir / f"{profile.label}.json"
        existing_records, seen_chunk_ids = _load_existing(output_file)

        pending = [c for c in chunks if args.force or c.chunk_id not in seen_chunk_ids]
        skipped = len(chunks) - len(pending)

        logger.info(
            "[%s] %d chunks to annotate, %d already done",
            profile.label, len(pending), skipped,
        )

        new_records: list[dict] = []
        for i, chunk in enumerate(pending, 1):
            logger.info(
                "[%s] %d/%d  %s", profile.label, i, len(pending), chunk.chunk_id
            )
            records = _call_api(profile, chunk)
            if records:
                logger.info(
                    "  → %d instance(s): %s",
                    len(records),
                    ", ".join(r["soc_type"] for r in records),
                )
            else:
                logger.info("  → no SoC instances found")
            new_records.extend(records)

            # Save after every chunk so progress survives interruption
            _save(output_file, existing_records + new_records)

            if i < len(pending):
                time.sleep(args.delay)

        total_new = len(new_records)
        total_all = len(existing_records) + total_new
        logger.info(
            "[%s] Done. %d new instances added. %d total in %s",
            profile.label, total_new, total_all, output_file,
        )

    logger.info("All models complete.")


if __name__ == "__main__":
    main()
