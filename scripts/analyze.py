"""Local-model SoC analysis -- run the fine-tuned Qwen3-4B model on text chunks.

The model is loaded once per process from model_config.yaml and held in
memory for the lifetime of the run.  The public interface
(analyze_chunks_multi) is intentionally kept compatible with run.py.
"""

from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

import yaml

from scripts.models import Chunk, InferenceResponse, ResultRow, SocInstance
from scripts.prompts import INFERENCE_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

# Default location of the model config (project root)
_DEFAULT_MODEL_CONFIG = Path(__file__).resolve().parent.parent / "model_config.yaml"


def build_user_prompt(chunk: Chunk) -> str:
    """Build the user message for a single chunk."""
    parts = [
        f"Source: {chunk.source_file}",
        f"Chunk: {chunk.chunk_id} ({chunk.chunk_label})",
        f"Position: {chunk.chunk_index}",
    ]
    if chunk.context_before:
        parts.append(f"\n[CONTEXT -- end of previous chunk]\n{chunk.context_before}\n")
    parts.append(f"\n[TEXT TO ANALYZE]\n{chunk.chunk_text}\n")
    if chunk.context_after:
        parts.append(f"\n[CONTEXT -- start of next chunk]\n{chunk.context_after}\n")
    parts.append(
        "\nAnalyze the [TEXT TO ANALYZE] section above. "
        "Identify and classify every SoC passage. "
        "Respond with JSON only."
    )
    return "\n".join(parts)


def _parse_response(raw: str) -> InferenceResponse:
    """Strip think block and parse the trailing JSON into InferenceResponse."""
    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        logger.debug("No JSON found in response; returning empty instances")
        return InferenceResponse(instances=[])
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        logger.warning("JSON decode error: %s -- raw snippet: %.200s", exc, cleaned)
        return InferenceResponse(instances=[])

    instances_raw = data.get("instances") or data.get("soc_instances") or []
    instances: list[SocInstance] = []
    for item in instances_raw:
        if not isinstance(item, dict):
            continue
        devices = item.get("secondary_devices", [])
        if isinstance(devices, str):
            devices = [d.strip() for d in devices.split(",") if d.strip()]
        evidence = item.get("evidence", [])
        if isinstance(evidence, str):
            evidence = [e.strip() for e in evidence.split(",") if e.strip()]
        try:
            instances.append(
                SocInstance(
                    passage=str(item.get("passage", "")),
                    soc_type=str(item.get("soc_type", "other_soc")),
                    secondary_devices=devices,
                    affective_register=str(item.get("affective_register", "n/a")),
                    narrator_position=str(item.get("narrator_position", "minimal")),
                    character_pov=str(item.get("character_pov", "")),
                    is_soc=bool(item.get("is_soc", True)),
                    explanation=str(item.get("explanation", "")),
                    evidence=evidence,
                    confidence=str(item.get("confidence", "medium")),
                    notes=str(item.get("notes", "")),
                )
            )
        except Exception as exc:
            logger.warning("Could not parse SocInstance: %s -- %s", exc, item)
    return InferenceResponse(instances=instances)


class ModelConfig:
    """Holds the local-model inference settings from model_config.yaml."""

    def __init__(self, data: dict) -> None:
        cfg = data.get("model", data)
        self.path: str = str(cfg.get("path", "apjanco/penelope-soc-v1"))
        self.device: str = str(cfg.get("device", "auto"))
        self.max_new_tokens: int = int(cfg.get("max_new_tokens", 4096))
        self.enable_thinking: bool = bool(cfg.get("enable_thinking", True))
        self.batch_size: int = int(cfg.get("batch_size", 1))
        self.temperature: float = float(cfg.get("temperature", 0.6))
        self.top_p: float = float(cfg.get("top_p", 0.95))
        self.top_k: int = int(cfg.get("top_k", 20))

    @classmethod
    def from_file(cls, path: Path) -> "ModelConfig":
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return cls(data)

    @classmethod
    def load_default(cls) -> "ModelConfig":
        if _DEFAULT_MODEL_CONFIG.exists():
            return cls.from_file(_DEFAULT_MODEL_CONFIG)
        return cls({})


class SocAnalyzer:
    """Wraps the fine-tuned model for SoC detection and classification."""

    def __init__(
        self,
        config: ModelConfig | None = None,
        model_config_path: Path | None = None,
        fast: bool = False,
    ) -> None:
        if config is None:
            path = model_config_path or _DEFAULT_MODEL_CONFIG
            config = ModelConfig.from_file(path) if path.exists() else ModelConfig({})
        self.cfg = config
        self.fast = fast

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            logger.error(
                "transformers not installed. "
                "Run: pip install transformers>=4.51.0 torch>=2.2.0\n%s",
                exc,
            )
            sys.exit(1)

        logger.info("Loading model: %s (device=%s)", config.path, config.device)
        self._tokenizer = AutoTokenizer.from_pretrained(
            config.path, trust_remote_code=True
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            config.path,
            torch_dtype=torch.bfloat16,
            device_map=config.device,
            trust_remote_code=True,
        )
        self._model.eval()
        logger.info("Model loaded.")

    def analyze_chunk(self, chunk: Chunk, model_label: str = "") -> list[ResultRow]:
        """Analyse a single chunk and return a list of ResultRow objects."""
        import torch

        enable_thinking = self.cfg.enable_thinking and not self.fast
        temperature = self.cfg.temperature if enable_thinking else 0.7
        top_p = self.cfg.top_p if enable_thinking else 0.8
        top_k = self.cfg.top_k if enable_thinking else 20

        messages = [
            {"role": "system", "content": INFERENCE_SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(chunk)},
        ]
        text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        inputs = self._tokenizer(text, return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.cfg.max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=True,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        generated = output_ids[0][inputs["input_ids"].shape[1]:]
        raw = self._tokenizer.decode(generated, skip_special_tokens=True)
        logger.debug("Chunk %s -- raw output (first 300 chars): %.300s", chunk.chunk_id, raw)

        response = _parse_response(raw)
        label = model_label or self.cfg.path.split("/")[-1]
        rows = [
            ResultRow.from_chunk_and_instance(chunk, inst, model_label=label)
            for inst in response.instances
            if inst.is_soc
        ]
        if rows:
            logger.info("Chunk %s -- %d SoC instance(s).", chunk.chunk_id, len(rows))
        else:
            logger.info("Chunk %s -- no SoC instances.", chunk.chunk_id)
        return rows


def analyze_chunks(chunks: list[Chunk], analyzer: SocAnalyzer) -> list[ResultRow]:
    """Run analysis on a list of chunks sequentially."""
    all_rows: list[ResultRow] = []
    for i, chunk in enumerate(chunks, 1):
        logger.info("Processing chunk %d / %d: %s", i, len(chunks), chunk.chunk_id)
        all_rows.extend(analyzer.analyze_chunk(chunk))
    return all_rows


def analyze_chunks_multi(
    chunks: list[Chunk],
    config: Any,
    model_config_path: Path | None = None,
    fast: bool = False,
) -> list[ResultRow]:
    """Run analysis with the local fine-tuned model.

    Keeps the same external signature as the original multi-model API
    version so that run.py requires only minimal changes.

    Args:
        chunks: Ordered list of chunks.
        config: Pipeline Config object (for context; model settings come
                from model_config.yaml, not from this object).
        model_config_path: Optional override path to model_config.yaml.
        fast: Use non-thinking mode.

    Returns:
        ResultRow list for all detected SoC instances.
    """
    path = model_config_path or _DEFAULT_MODEL_CONFIG
    if not path.exists():
        logger.warning(
            "model_config.yaml not found at %s -- using built-in defaults. "
            "Run export_model.py first or point --model-config at a valid file.",
            path,
        )
    analyzer = SocAnalyzer(model_config_path=path, fast=fast)
    logger.info("Running local-model analysis on %d chunk(s).", len(chunks))
    rows = analyze_chunks(chunks, analyzer)
    logger.info("Analysis complete: %d SoC instance(s) found.", len(rows))
    return rows
