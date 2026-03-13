"""LLM-based SOC analysis — send chunks to an OpenAI-compatible API."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from openai import OpenAI

from scripts.config import Config, ModelProfile
from scripts.models import Chunk, LLMResponse, ResultRow, SocInstance

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt construction
# ---------------------------------------------------------------------------

def _load_skill(skill_path: Path) -> str:
    """Read the SKILL.md file and return its contents."""
    if not skill_path.exists():
        raise FileNotFoundError(
            f"SKILL.md not found at {skill_path}. "
            "Ensure it exists in the project root."
        )
    return skill_path.read_text(encoding="utf-8")


def build_system_prompt(skill_path: Path) -> str:
    """Build the system prompt from SKILL.md."""
    skill_text = _load_skill(skill_path)
    return (
        "You are an expert literary analyst specialising in stream of consciousness "
        "(SOC) techniques in modernist fiction. Your task is to identify and classify "
        "EVERY instance of SOC in the provided text chunk.\n\n"
        "Follow the taxonomy, classification procedure, and output format defined "
        "in the skill document below. Be exhaustive — find ALL SOC passages, not "
        "just a sample. For each passage, quote the text verbatim.\n\n"
        "When a passage is ambiguous or could fit multiple types, acknowledge the "
        "ambiguity in the notes field and explain what signals point toward each "
        "possible classification. Use medium or low confidence when warranted.\n\n"
        "If no SOC is present in the chunk, return: {\"soc_instances\": []}\n\n"
        "---\n\n"
        f"{skill_text}\n\n"
        "---\n\n"
        "CRITICAL REMINDERS:\n"
        "- Be EXHAUSTIVE. Identify every SOC passage, even brief ones.\n"
        "- Quote passages VERBATIM from the text.\n"
        "- Respond ONLY with valid JSON matching the output format.\n"
        "- Include secondary_devices, affective_register, narrator_position, "
        "character_pov, evidence, and notes for EVERY instance.\n\n"
        "JSON SCHEMA — use these exact field names:\n"
        '{"soc_instances": [{"passage": "...", "soc_type": "...", '
        '"secondary_devices": [...], "affective_register": "...", '
        '"narrator_position": "...", "character_pov": "...", '
        '"explanation": "...", "evidence": [...], "confidence": "...", '
        '"notes": "..."}]}\n'
    )


def build_user_prompt(chunk: Chunk) -> str:
    """Build the user message for a single chunk."""
    parts = [
        f"Source: {chunk.source_file}",
        f"Chunk: {chunk.chunk_id} ({chunk.chunk_label})",
        f"Position: {chunk.chunk_index}",
    ]
    if chunk.context_before:
        parts.append(f"\n[CONTEXT — end of previous chunk]\n{chunk.context_before}\n")
    parts.append(f"\n[TEXT TO ANALYZE]\n{chunk.chunk_text}\n")
    if chunk.context_after:
        parts.append(f"\n[CONTEXT — start of next chunk]\n{chunk.context_after}\n")

    parts.append(
        "\nAnalyze the [TEXT TO ANALYZE] section above. "
        "Identify and classify every SOC passage. "
        "Respond with JSON only."
    )
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# LLM call with retry
# ---------------------------------------------------------------------------

def _call_llm(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float | None = 0.1,
    max_retries: int = 3,
    retry_delay: float = 2.0,
) -> str:
    """Send a chat completion request with simple retry logic."""
    for attempt in range(1, max_retries + 1):
        try:
            kwargs: dict = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "response_format": {"type": "json_object"},
            }
            # Only include temperature if explicitly set — some models
            # (e.g. GPT-5 via Portkey) reject non-default values.
            if temperature is not None:
                kwargs["temperature"] = temperature

            response = client.chat.completions.create(**kwargs)
            content = response.choices[0].message.content or ""
            return content

        except Exception as exc:
            logger.warning(
                "LLM call attempt %d/%d failed: %s", attempt, max_retries, exc
            )
            if attempt < max_retries:
                time.sleep(retry_delay * attempt)
            else:
                raise


# ---------------------------------------------------------------------------
# Response normalization — remap common LLM field-name variants
# ---------------------------------------------------------------------------

# Maps human-readable / variant keys → Pydantic field names
_KEY_MAP: dict[str, str] = {
    "primary type": "soc_type",
    "primary_type": "soc_type",
    "type": "soc_type",
    "soc type": "soc_type",
    "secondary devices": "secondary_devices",
    "secondary_devices": "secondary_devices",
    "affective register": "affective_register",
    "affective_register": "affective_register",
    "narrator position": "narrator_position",
    "narrator_position": "narrator_position",
    "character pov": "character_pov",
    "character_pov": "character_pov",
    "character": "character_pov",
    "pov": "character_pov",
    "passage": "passage",
    "text": "passage",
    "quote": "passage",
    "explanation": "explanation",
    "reasoning": "explanation",
    "evidence": "evidence",
    "confidence": "confidence",
    "confidence level": "confidence",
    "notes": "notes",
}


def _normalize_instance(raw: dict) -> dict:
    """Remap variant field names to canonical Pydantic names."""
    normalized: dict = {}
    for key, value in raw.items():
        canonical = _KEY_MAP.get(key.lower().strip(), key)
        # First match wins — don't overwrite if we already set it
        if canonical not in normalized:
            normalized[canonical] = value
    return normalized


def _normalize_response(data: dict) -> dict:
    """Normalize top-level response and each soc_instance inside it."""
    # Handle variant top-level keys ("soc_instances", "instances", "results")
    instances = None
    for key in ("soc_instances", "instances", "results", "annotations"):
        if key in data:
            instances = data[key]
            break
    if instances is None:
        # Maybe the LLM returned a bare list
        if isinstance(data, list):
            instances = data
        else:
            instances = []

    if isinstance(instances, list):
        instances = [_normalize_instance(inst) if isinstance(inst, dict) else inst for inst in instances]

    return {"soc_instances": instances}


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def _parse_response(
    raw_json: str, chunk: Chunk, model_label: str = ""
) -> list[ResultRow]:
    """Parse and validate the LLM's JSON response."""
    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        logger.error("Invalid JSON from LLM for chunk %s: %s", chunk.chunk_id, exc)
        logger.debug("Raw response: %s", raw_json[:500])
        return []

    # Normalize field names before validation
    data = _normalize_response(data)

    try:
        llm_response = LLMResponse.model_validate(data)
    except Exception as exc:
        logger.error(
            "Response validation failed for chunk %s: %s", chunk.chunk_id, exc
        )
        logger.debug("Normalized data keys: %s", list(data.get("soc_instances", [{}])[0].keys()) if data.get("soc_instances") else "empty")
        return []

    rows: list[ResultRow] = []
    for inst in llm_response.soc_instances:
        rows.append(ResultRow.from_chunk_and_instance(chunk, inst, model_label=model_label))
    return rows


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyze_chunk(
    chunk: Chunk,
    client: OpenAI,
    model: str,
    system_prompt: str,
    model_label: str = "",
    temperature: float | None = 0.1,
    max_retries: int = 3,
) -> list[ResultRow]:
    """Analyze a single chunk for SOC passages.

    Args:
        chunk: The text chunk to analyze.
        client: Configured OpenAI client.
        model: Model name to use.
        system_prompt: Pre-built system prompt with SKILL.md.
        model_label: Human-readable label for this model (for output).
        temperature: Temperature for sampling. None = omit (model default).
        max_retries: Number of retry attempts on failure.

    Returns:
        List of ResultRow objects (one per SOC instance found).
    """
    user_prompt = build_user_prompt(chunk)
    logger.info("[%s] Analyzing chunk %s (%d chars)", model_label, chunk.chunk_id, len(chunk.chunk_text))

    raw = _call_llm(
        client, model, system_prompt, user_prompt,
        temperature=temperature, max_retries=max_retries,
    )
    rows = _parse_response(raw, chunk, model_label=model_label)

    logger.info(
        "[%s] Chunk %s: found %d SOC instance(s)", model_label, chunk.chunk_id, len(rows)
    )
    return rows


def analyze_chunks(
    chunks: list[Chunk],
    profile: ModelProfile,
    skill_path: Path,
) -> list[ResultRow]:
    """Analyze all chunks sequentially with a single model profile.

    Args:
        chunks: Ordered list of chunks from a single work.
        profile: The LLM model profile to use.
        skill_path: Path to the SKILL.md file.

    Returns:
        Combined list of ResultRow objects from all chunks.
    """
    profile.validate()

    client = OpenAI(base_url=profile.base_url, api_key=profile.api_key)
    system_prompt = build_system_prompt(skill_path)

    all_rows: list[ResultRow] = []
    for i, chunk in enumerate(chunks):
        logger.info(
            "[%s] Processing chunk %d/%d: %s",
            profile.label, i + 1, len(chunks), chunk.chunk_id,
        )
        try:
            rows = analyze_chunk(
                chunk, client, profile.model_name, system_prompt,
                model_label=profile.label,
                temperature=profile.temperature,
            )
            all_rows.extend(rows)
        except Exception:
            logger.exception(
                "[%s] Failed to analyze chunk %s", profile.label, chunk.chunk_id
            )

    logger.info(
        "[%s] Analysis complete: %d SOC instances across %d chunks",
        profile.label, len(all_rows), len(chunks),
    )
    return all_rows


def analyze_chunks_multi(
    chunks: list[Chunk],
    config: Config,
) -> list[ResultRow]:
    """Run analysis across all configured model profiles.

    Each model processes the same chunks independently. Results are
    combined with a model_label column so they can be compared.

    Args:
        chunks: Ordered list of chunks.
        config: Pipeline configuration with one or more model profiles.

    Returns:
        Combined results from all models.
    """
    profiles = config.get_model_profiles()
    logger.info(
        "Running analysis with %d model(s): %s",
        len(profiles), ", ".join(p.label for p in profiles),
    )

    all_rows: list[ResultRow] = []
    for profile in profiles:
        logger.info("\n" + "=" * 60)
        logger.info("Starting model: %s (%s)", profile.label, profile.model_name)
        logger.info("=" * 60)
        rows = analyze_chunks(chunks, profile, config.skill_path)
        all_rows.extend(rows)

    logger.info(
        "All models complete: %d total SOC instances", len(all_rows)
    )
    return all_rows
