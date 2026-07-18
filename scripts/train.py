"""QLoRA supervised fine-tuning of Qwen/Qwen3-4B for SoC classification.

Usage
-----
python scripts/train.py \\
    --dataset dataset/ \\
    [--base-model Qwen/Qwen3-4B] \\
    [--adapter-output adapter/] \\
    [--epochs 3] \\
    [--batch-size 2] \\
    [--grad-accum 8] \\
    [--lr 2e-4] \\
    [--lora-rank 64] \\
    [--max-seq-len 4096] \\
    [--seed 42]

Training format
---------------
Each JSONL record has a "messages" key (system/user/assistant turns).
The assistant turn is:
  <think>\\n...reasoning...\\n</think>\\n{"instances": [...]}

Loss masking
------------
The custom ThinkMaskingCollator sets labels to -100 for all tokens
from the start of the assistant turn up to and including </think>.
Only the JSON tokens after </think> contribute to the loss.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _import_training_deps():
    """Import heavy training dependencies (deferred to avoid import cost at inference time)."""
    try:
        import torch
        from datasets import Dataset, DatasetDict
        from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            Trainer,
            TrainingArguments,
        )
        return (
            torch, Dataset, DatasetDict, LoraConfig, TaskType,
            get_peft_model, prepare_model_for_kbit_training,
            AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
            Trainer, TrainingArguments,
        )
    except ImportError as exc:
        logger.error(
            "Missing training dependency: %s\n"
            "Install with: pip install transformers>=4.51.0 peft>=0.10.0 "
            "datasets>=2.19.0 bitsandbytes>=0.43.0 torch>=2.2.0",
            exc,
        )
        sys.exit(1)


# ---------------------------------------------------------------------------
# Think-masking data collator
# ---------------------------------------------------------------------------


class ThinkMaskingCollator:
    """Pads sequences and masks <think>...</think> tokens from the loss.

    For each sequence in the batch, finds the LAST occurrence of the
    </think> token and sets labels[0 : pos+1] = -100.  Padding tokens
    and the prompt prefix (everything before <|im_start|>assistant) are
    also masked.

    Legacy collator — use WeightedThinkCollator for v3 training.
    """

    def __init__(self, tokenizer, think_end_token_ids: list[int]):
        self.tokenizer = tokenizer
        self.think_end_ids = think_end_token_ids  # token ids for </think>

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        # Pad the batch
        batch = self.tokenizer.pad(
            features,
            padding=True,
            return_tensors="pt",
        )
        input_ids = batch["input_ids"]          # (B, L)
        attention_mask = batch["attention_mask"] # (B, L)

        labels = input_ids.clone()
        # Mask padding tokens
        labels[attention_mask == 0] = -100

        # Mask everything up to and including </think> per sequence
        for i, seq in enumerate(input_ids):
            seq_list = seq.tolist()
            # Find last occurrence of each think_end_id token
            last_pos = -1
            for tid in self.think_end_ids:
                positions = [j for j, x in enumerate(seq_list) if x == tid]
                if positions:
                    last_pos = max(last_pos, positions[-1])
            if last_pos >= 0:
                labels[i, : last_pos + 1] = -100

        batch["labels"] = labels
        return batch


class WeightedThinkCollator:
    """Pads sequences and applies per-region loss weights.

    Loss regions per sequence:
      - Prompt (system + user turns, up to and excluding <think>):
            labels = -100 (masked from loss entirely)
      - Reasoning trace (<think> ... </think> inclusive):
            labels = token IDs, token_weight = think_weight
      - Structured output (tokens after </think>):
            labels = token IDs, token_weight = 1.0

    The companion WeightedThinkTrainer reads the ``token_weights`` tensor
    produced here and applies it in a custom compute_loss, so the gradient
    signal from the reasoning trace is scaled down relative to the
    structured JSON output while still flowing back through the network.

    Args:
        tokenizer: the model tokenizer (for padding).
        think_start_token_ids: token ids that encode ``<think>``.
        think_end_token_ids: token ids that encode ``</think>``.
        think_weight: loss weight for reasoning-trace tokens (0.0–1.0).
            0.0 replicates ThinkMaskingCollator behaviour; 1.0 gives equal
            weight to reasoning and output tokens.  Recommended: 0.3–0.5.
    """

    def __init__(
        self,
        tokenizer,
        think_start_token_ids: list[int],
        think_end_token_ids: list[int],
        think_weight: float = 0.4,
    ):
        self.tokenizer = tokenizer
        self.think_start_ids = think_start_token_ids
        self.think_end_ids = think_end_token_ids
        self.think_weight = think_weight

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        batch = self.tokenizer.pad(
            features,
            padding=True,
            return_tensors="pt",
        )
        input_ids = batch["input_ids"]           # (B, L)
        attention_mask = batch["attention_mask"]  # (B, L)
        B, L = input_ids.shape

        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        token_weights = torch.zeros(B, L, dtype=torch.float32)

        for i, seq in enumerate(input_ids):
            seq_list = seq.tolist()

            # Locate </think> (last occurrence)
            think_end_pos = -1
            for tid in self.think_end_ids:
                for j in range(len(seq_list) - 1, -1, -1):
                    if seq_list[j] == tid:
                        think_end_pos = max(think_end_pos, j)
                        break

            # Locate <think> (last occurrence before </think>)
            think_start_pos = -1
            for tid in self.think_start_ids:
                for j in range(think_end_pos, -1, -1):
                    if seq_list[j] == tid:
                        think_start_pos = max(think_start_pos, j)
                        break

            if think_start_pos >= 0 and think_end_pos >= think_start_pos:
                # Mask prompt (everything before <think>)
                labels[i, :think_start_pos] = -100

                # Think region: keep labels at think_weight
                token_weights[i, think_start_pos : think_end_pos + 1] = self.think_weight

                # JSON output region: weight 1.0 up to the last real (non-padding) token
                seq_len = int(attention_mask[i].sum().item())
                token_weights[i, think_end_pos + 1 : seq_len] = 1.0
            else:
                # Fallback: no think block found — mask the whole sequence
                labels[i, :] = -100

        batch["labels"] = labels
        batch["token_weights"] = token_weights
        return batch


# ---------------------------------------------------------------------------
# Weighted-think trainer
# ---------------------------------------------------------------------------


def _make_weighted_think_trainer_class():
    """Return a Trainer subclass that applies per-token weights from the collator.

    Defined as a factory so the heavy ``transformers.Trainer`` import happens
    only inside the training process, not at module import time.
    """
    from transformers import Trainer
    import torch

    class WeightedThinkTrainer(Trainer):
        """Trainer that scales the reasoning-trace contribution to the loss.

        The ``WeightedThinkCollator`` adds a ``token_weights`` tensor to each
        batch.  This trainer pops it before the model forward pass and uses it
        to compute a weighted cross-entropy:

            loss = Σ(weight_i * ce_i) / Σ(weight_i)   for all non-masked tokens

        where ``weight_i`` is ``think_weight`` for reasoning-trace tokens and
        ``1.0`` for JSON-output tokens.  Prompt and padding tokens are masked
        with labels=-100 and weight=0.0 so they contribute nothing.
        """

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
            token_weights = inputs.pop("token_weights", None)

            outputs = model(**inputs)

            if token_weights is None:
                # Fallback: no weights tensor (e.g. legacy collator)
                loss = outputs.loss
                return (loss, outputs) if return_outputs else loss

            logits = outputs.logits
            labels = inputs["labels"]

            # Causal LM: predict token t+1 from token t, so shift by 1.
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_weights = token_weights[..., 1:].to(logits.device)

            loss_fct = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
            per_token_loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            ).view(shift_labels.size())  # (B, L-1)

            # Weighted mean over all non-masked positions.
            # ignore_index=-100 already zeroes masked positions in per_token_loss,
            # but we also zero them in the weight denominator so the scale is correct.
            non_masked = (shift_labels != -100).float()
            weighted_sum = (per_token_loss * shift_weights).sum()
            weight_total = (shift_weights * non_masked).sum().clamp(min=1e-8)
            loss = weighted_sum / weight_total

            return (loss, outputs) if return_outputs else loss

    return WeightedThinkTrainer


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def apply_chat_template(records: list[dict], tokenizer) -> list[dict]:
    """Tokenize each record's messages using the tokenizer's chat template."""
    tokenized = []
    for rec in records:
        messages = rec["messages"]
        # Apply chat template with thinking mode enabled for assistant turn
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        enc = tokenizer(
            text,
            truncation=True,
            max_length=tokenizer.model_max_length,
            return_tensors=None,
        )
        tokenized.append({"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]})
    return tokenized


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------


def train(args: argparse.Namespace) -> None:
    (
        torch, Dataset, DatasetDict, LoraConfig, TaskType,
        get_peft_model, prepare_model_for_kbit_training,
        AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
        Trainer, TrainingArguments,
    ) = _import_training_deps()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
    )

    # ── Tokenizer ──────────────────────────────────────────────────────
    logger.info("Loading tokenizer: %s", args.base_model)
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.model_max_length = args.max_seq_len

    # Find the token id(s) for <think> and </think>
    think_start_str = "<think>"
    think_start_ids = tokenizer.encode(think_start_str, add_special_tokens=False)
    logger.info("<think>  token ids: %s", think_start_ids)

    think_end_str = "</think>"
    think_end_ids = tokenizer.encode(think_end_str, add_special_tokens=False)
    logger.info("</think> token ids: %s", think_end_ids)

    # ── Model ──────────────────────────────────────────────────────────
    logger.info("Loading model in 4-bit: %s", args.base_model)

    # Auto-detect flash_attention_2: requires sm_80+ (A100/H100) and the flash-attn package.
    # Enabled explicitly with --flash-attn; otherwise falls back to sdpa then eager.
    if args.flash_attn:
        attn_impl = "flash_attention_2"
    elif torch.cuda.is_available():
        # Use scaled_dot_product_attention (built into PyTorch >= 2.0) as a middle ground
        attn_impl = "sdpa"
    else:
        attn_impl = "eager"
    logger.info("Attention implementation: %s", attn_impl)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation=attn_impl,
    )
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False

    # ── LoRA ──────────────────────────────────────────────────────────
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_rank * 2,
        lora_dropout=0.05,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── Dataset ────────────────────────────────────────────────────────
    dataset_dir = args.dataset
    train_records = load_jsonl(dataset_dir / "train.jsonl")
    val_records = load_jsonl(dataset_dir / "val.jsonl")
    logger.info(
        "Dataset sizes — train: %d, val: %d", len(train_records), len(val_records)
    )

    train_tokenized = apply_chat_template(train_records, tokenizer)
    val_tokenized = apply_chat_template(val_records, tokenizer)

    train_dataset = Dataset.from_list(train_tokenized)
    val_dataset = Dataset.from_list(val_tokenized)

    # ── Data collator ──────────────────────────────────────────────────
    # Resolve effective think weight.  --mask-thinking is kept for backward
    # compat with v1/v2 slurm scripts and forces the weight to 0.0.
    effective_think_weight = 0.0 if args.mask_thinking else args.think_loss_weight

    collator = WeightedThinkCollator(
        tokenizer,
        think_start_token_ids=think_start_ids,
        think_end_token_ids=think_end_ids,
        think_weight=effective_think_weight,
    )
    logger.info(
        "Think loss weight: %.2f  (%s)",
        effective_think_weight,
        "reasoning trace excluded from loss" if effective_think_weight == 0.0
        else "reasoning trace included at reduced weight" if effective_think_weight < 1.0
        else "reasoning trace at full weight",
    )

    # ── Training arguments ─────────────────────────────────────────────
    output_dir = args.adapter_output
    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_steps=10,
        lr_scheduler_type="cosine",
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        seed=args.seed,
        dataloader_num_workers=args.dataloader_workers,
        remove_unused_columns=False,
    )

    # ── Trainer ────────────────────────────────────────────────────────
    # Use WeightedThinkTrainer (not SFTTrainer) since we pre-tokenize the
    # dataset and apply a custom weighted collator.  SFTTrainer's internal
    # formatting logic conflicts with pre-tokenized data in trl >= 0.8.
    WeightedThinkTrainer = _make_weighted_think_trainer_class()
    trainer = WeightedThinkTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
        processing_class=tokenizer,
    )

    logger.info("Starting training...")
    trainer.train()

    # ── Save ───────────────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Saving adapter to %s", output_dir)
    trainer.model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    logger.info("Training complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="QLoRA fine-tune Qwen3-4B for stream-of-consciousness classification."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("dataset"),
        help="Directory with train.jsonl, val.jsonl (default: dataset/)",
    )
    parser.add_argument(
        "--base-model",
        default="Qwen/Qwen3-4B",
        help="HuggingFace model ID for the base model (default: Qwen/Qwen3-4B)",
    )
    parser.add_argument(
        "--adapter-output",
        type=Path,
        default=Path("adapter"),
        help="Directory to save the LoRA adapter (default: adapter/)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: 3)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Per-device train batch size (default: 2)",
    )
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=8,
        help="Gradient accumulation steps (default: 8; effective batch = 16)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-4,
        help="Learning rate (default: 2e-4)",
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=64,
        help="LoRA rank (default: 64; alpha=2x rank)",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=4096,
        help="Maximum tokenised sequence length (default: 4096)",
    )
    parser.add_argument(
        "--flash-attn",
        action="store_true",
        help="Use flash_attention_2 (requires: pip install flash-attn --no-build-isolation)",
    )
    parser.add_argument(
        "--dataloader-workers",
        type=int,
        default=4,
        help="DataLoader worker processes (default: 4; use 0 for debugging)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--think-loss-weight",
        type=float,
        default=0.4,
        metavar="W",
        help=(
            "Loss weight for <think>...</think> reasoning-trace tokens relative to "
            "the JSON output tokens (weight=1.0).  Range 0.0–1.0.  "
            "0.0 excludes the reasoning trace entirely (same as --mask-thinking).  "
            "1.0 gives equal weight to reasoning and output.  "
            "Recommended for v3: 0.4 (default)."
        ),
    )
    parser.add_argument(
        "--mask-thinking",
        action="store_true",
        default=False,
        help=(
            "DEPRECATED — use --think-loss-weight 0.0 instead.  "
            "Forces think_loss_weight to 0.0, reproducing v1/v2 training behaviour "
            "where <think>...</think> tokens are excluded from the loss entirely."
        ),
    )
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
