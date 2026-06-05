"""Penelope — GRPO fine-tuning on interpretive reward.

Applies Group Relative Policy Optimization on top of an SFT adapter to
sharpen interpretive quality. Requires a running vLLM judge server
(see judge_della.slurm).

Usage
-----
    python scripts/train_grpo.py \\
        --dataset dataset/ \\
        --base-model Qwen/Qwen3-4B \\
        --sft-adapter adapter-sft/ \\
        --output adapter-grpo/ \\
        --judge-url http://della9.princeton.edu:8000/v1 \\
        [--epochs 1] \\
        [--group-size 8] \\
        [--kl-coef 0.1]

Prerequisites
-------------
- Run scripts/generate_traces.py to rebuild dataset/ with interpretive traces.
- Run scripts/train.py --mask-thinking False to produce the SFT adapter.
- Start judge_della.slurm and wait for judge_endpoint.txt to be written.

Reward function
---------------
reward = 0.2 * grounding + 0.2 * skepticism + 0.2 * specificity + 0.4 * type_coherence

grounding + specificity are scored automatically (no API call).
skepticism + type_coherence are scored by the vLLM judge (JUDGE_PROMPT).

When is_soc=false, specificity and type_coherence default to 0.5 (neutral)
so the model is not penalised for correct negative predictions.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Taxonomy keywords for automatic scoring (mirrors generate_traces.py)
_TAXONOMY_KEYWORDS = {
    "direct_interior_monologue", "indirect_interior_monologue",
    "omniscient_description", "soliloquy", "free_association",
    "space_montage", "orthographic_marker", "imagery",
    "simulation_state_of_mind", "reverie_fantasy", "hybrid", "other_soc",
    "humphrey", "steinberg", "free indirect", "interior monologue",
}


# ---------------------------------------------------------------------------
# Automatic reward components
# ---------------------------------------------------------------------------


def _grounding(think: str, passage: str) -> float:
    quotes = re.findall(r'["\u201c\u201d\u2018\u2019]([^"\']+)["\u201c\u201d\u2018\u2019]', think)
    passage_lower = passage.lower()
    for q in quotes:
        if len(q.split()) >= 3 and q.strip().lower() in passage_lower:
            return 1.0
    words = think.split()
    for i in range(len(words) - 3):
        frag = re.sub(r"[^\w\s]", "", " ".join(words[i : i + 4]).lower())
        if frag in passage_lower:
            return 0.5
    return 0.0


def _specificity(think: str) -> float:
    lower = think.lower()
    hits = sum(1 for kw in _TAXONOMY_KEYWORDS if kw in lower)
    if hits >= 2:
        return 1.0
    if hits == 1:
        return 0.5
    return 0.0


# ---------------------------------------------------------------------------
# Judge client
# ---------------------------------------------------------------------------


def _build_judge_client(judge_url: str):
    try:
        from openai import OpenAI
    except ImportError:
        logger.error("openai package required: pip install openai")
        sys.exit(1)
    return OpenAI(api_key="not-needed", base_url=judge_url)


def _call_judge(
    client: Any,
    judge_model: str,
    judge_prompt_template: str,
    passage: str,
    think_trace: str,
    json_output: str,
) -> dict:
    prompt = judge_prompt_template.format(
        passage=passage,
        think_trace=think_trace,
        json_output=json_output,
    )
    try:
        response = client.chat.completions.create(
            model=judge_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=256,
        )
        raw = response.choices[0].message.content.strip()
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            return json.loads(match.group(0))
    except Exception as exc:
        logger.warning("Judge call failed: %s", exc)
    return {"grounding": 0.5, "skepticism": 0.5, "specificity": 0.5, "type_coherence": 0.5, "rationale": "judge unavailable"}


# ---------------------------------------------------------------------------
# Combined reward
# ---------------------------------------------------------------------------


def compute_reward(
    completions: list[str],
    passages: list[str],
    judge_client: Any,
    judge_model: str,
    judge_prompt: str,
) -> list[float]:
    rewards = []
    for completion, passage in zip(completions, passages):
        # Extract think and JSON
        think_match = re.search(r"<think>(.*?)</think>", completion, re.DOTALL)
        think = think_match.group(1).strip() if think_match else ""
        json_str = re.sub(r"<think>.*?</think>", "", completion, flags=re.DOTALL).strip()
        jmatch = re.search(r"\{.*\}", json_str, re.DOTALL)
        json_output = jmatch.group(0) if jmatch else "{}"

        # Determine is_soc from JSON
        try:
            parsed = json.loads(json_output)
            is_soc = bool(parsed.get("instances"))
        except Exception:
            is_soc = False

        # Automatic scores
        g = _grounding(think, passage)
        s = _specificity(think)

        # Judge scores
        judge_scores = _call_judge(
            judge_client, judge_model, judge_prompt,
            passage, think, json_output,
        )
        skepticism = float(judge_scores.get("skepticism", 0.5))
        type_coherence = float(judge_scores.get("type_coherence", 0.5))

        # Neutral on type dimensions for negatives
        if not is_soc:
            s = 0.5
            type_coherence = 0.5

        reward = 0.2 * g + 0.2 * skepticism + 0.2 * s + 0.4 * type_coherence
        rewards.append(reward)

        logger.debug(
            "reward=%.3f  grounding=%.2f  skepticism=%.2f  specificity=%.2f  "
            "type_coherence=%.2f  rationale=%s",
            reward, g, skepticism, s, type_coherence,
            judge_scores.get("rationale", ""),
        )

    return rewards


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train(args: argparse.Namespace) -> None:
    try:
        import torch
        from datasets import Dataset
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as exc:
        logger.error(
            "Missing dependency: %s\n"
            "Install with: pip install trl>=0.8.0 transformers peft torch",
            exc,
        )
        sys.exit(1)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
    )

    # ── Judge setup ────────────────────────────────────────────────────
    judge_url = args.judge_url or os.environ.get("JUDGE_URL")
    if not judge_url:
        logger.error(
            "Judge URL required: --judge-url or JUDGE_URL env var. "
            "Start judge_della.slurm and wait for judge_endpoint.txt."
        )
        sys.exit(1)

    logger.info("Judge endpoint: %s", judge_url)
    judge_client = _build_judge_client(judge_url)

    from prompts import JUDGE_PROMPT, INFERENCE_SYSTEM_PROMPT

    # ── Load dataset ───────────────────────────────────────────────────
    train_records = []
    with (args.dataset / "train.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                train_records.append(json.loads(line))

    # Extract passages for reward computation
    passages = []
    prompts = []
    for rec in train_records:
        msgs = rec.get("messages", [])
        user_msg = next((m["content"] for m in msgs if m["role"] == "user"), "")
        passages.append(user_msg)
        # GRPOTrainer expects prompt messages without assistant turn
        prompts.append([
            {"role": "system", "content": INFERENCE_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ])

    dataset = Dataset.from_dict({"prompt": prompts, "passage": passages})
    logger.info("GRPO training set: %d examples", len(dataset))

    # ── Load model ─────────────────────────────────────────────────────
    # Resolve to absolute path so PEFT's os.path.isdir() check succeeds and
    # it never falls through to Hub lookup.
    sft_adapter_path = Path(args.sft_adapter).resolve()
    if not sft_adapter_path.is_dir():
        raise FileNotFoundError(f"SFT adapter directory not found: {sft_adapter_path}")
    logger.info("Loading SFT adapter: %s on %s", sft_adapter_path, args.base_model)
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, trust_remote_code=True, local_files_only=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # GRPOTrainer expects left-padding

    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
    )
    model = PeftModel.from_pretrained(base, str(sft_adapter_path), local_files_only=True)

    # ── GRPO config ────────────────────────────────────────────────────
    grpo_config = GRPOConfig(
        output_dir=str(args.output / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_generations=args.group_size,     # k completions per passage
        max_completion_length=args.max_new_tokens,
        temperature=0.8,
        beta=args.kl_coef,                  # KL penalty vs reference policy
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        logging_steps=5,
        save_steps=100,
        save_total_limit=2,
        report_to="none",
        seed=args.seed,
        remove_unused_columns=False,
    )

    # ── Reward wrapper ─────────────────────────────────────────────────
    judge_model_name = args.judge_model

    def reward_fn(completions: list[str], passage: list[str], **_) -> list[float]:  # type: ignore[override]
        return compute_reward(
            completions=completions,
            passages=passage,
            judge_client=judge_client,
            judge_model=judge_model_name,
            judge_prompt=JUDGE_PROMPT,
        )

    # ── Train ──────────────────────────────────────────────────────────
    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=dataset,
        reward_funcs=reward_fn,
        processing_class=tokenizer,
    )

    logger.info("Starting GRPO training...")
    logger.warning(
        "Monitor KL divergence in logs. If KL spikes while reward climbs, "
        "reduce --kl-coef or stop early to prevent reward hacking."
    )
    trainer.train()

    # ── Save ───────────────────────────────────────────────────────────
    args.output.mkdir(parents=True, exist_ok=True)
    logger.info("Saving GRPO adapter to %s", args.output)
    trainer.model.save_pretrained(str(args.output))
    tokenizer.save_pretrained(str(args.output))
    logger.info("GRPO training complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GRPO fine-tuning with interpretive reward for SoC classification.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    parser.add_argument("--base-model", default="Qwen/Qwen3-4B")
    parser.add_argument(
        "--sft-adapter", type=Path, default=Path("adapter-sft"),
        help="Path to SFT adapter (trained WITHOUT --mask-thinking).",
    )
    parser.add_argument("--output", type=Path, default=Path("adapter-grpo"))
    parser.add_argument(
        "--judge-url", default=None,
        help="Base URL of vLLM judge server, e.g. http://della9.princeton.edu:8000/v1. "
             "Reads JUDGE_URL env var if not provided.",
    )
    parser.add_argument(
        "--judge-model", default="Qwen/Qwen3.6-27B",
        help="Model name served by the judge vLLM instance.",
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument(
        "--group-size", type=int, default=8,
        help="Number of completions per passage for GRPO (k). Default: 8.",
    )
    parser.add_argument(
        "--kl-coef", type=float, default=0.1,
        help="KL divergence penalty coefficient. Increase if reward hacking occurs.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
