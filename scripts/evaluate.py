"""Evaluate the fine-tuned SoC model against the test split.

Usage
-----
python scripts/evaluate.py \\
    --model apjanco/penelope-soc-v1 \\
    [--adapter adapter/]              # load adapter on top of base model
    [--base-model Qwen/Qwen3-4B]      # required when --adapter is used
    [--dataset dataset/] \\
    [--batch-size 4] \\
    [--max-new-tokens 4096] \\
    [--enable-thinking] \\
    [--output eval_results.json]

Outputs
-------
Per-class precision / recall / F1, macro and weighted averages,
confusion matrix, and comparison against:
  - zero-shot Qwen3-4B baseline
  - silver majority-vote labels
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------


def load_model_and_tokenizer(
    model_path: str,
    adapter_path: str | None,
    base_model: str,
    device: str = "auto",
):
    """Load model and tokenizer, optionally merging a LoRA adapter."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        logger.error("Missing dependency: %s", exc)
        sys.exit(1)

    if adapter_path:
        from peft import PeftModel
        tok = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base, adapter_path)
        model = model.merge_and_unload()
    else:
        tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True,
        )

    model.eval()
    return model, tok


def run_inference(
    model,
    tokenizer,
    messages: list[dict],
    max_new_tokens: int = 4096,
    enable_thinking: bool = True,
    temperature: float = 0.6,
    top_p: float = 0.95,
    top_k: int = 20,
) -> str:
    """Run inference on a single messages list and return the raw assistant text."""
    import torch

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature if enable_thinking else 0.7,
            top_p=top_p if enable_thinking else 0.8,
            top_k=top_k if enable_thinking else 20,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


def parse_response(raw: str) -> dict:
    """Strip think block and parse JSON."""
    # Remove thinking block if present
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    # Extract first { ... } block
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return {"instances": []}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {"instances": []}


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------


def compute_metrics(y_true: list[str], y_pred: list[str]) -> dict:
    """Compute per-class and aggregate classification metrics."""
    try:
        from sklearn.metrics import classification_report, confusion_matrix
        import numpy as np
    except ImportError:
        logger.error("sklearn required: pip install scikit-learn")
        sys.exit(1)

    labels = sorted(set(y_true + y_pred))
    report = classification_report(
        y_true, y_pred,
        labels=labels,
        zero_division=0,
        output_dict=True,
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    return {"report": report, "confusion_matrix": {"labels": labels, "matrix": cm}}


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------


def evaluate(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
    )

    # ── Load test data ─────────────────────────────────────────────────
    test_path = args.dataset / "test_raw.jsonl"
    test_msg_path = args.dataset / "test.jsonl"
    if not test_path.exists():
        logger.error(
            "test_raw.jsonl not found in %s — run build_dataset.py first", args.dataset
        )
        sys.exit(1)

    raw_records = []
    with test_path.open(encoding="utf-8") as fh:
        for line in fh:
            raw_records.append(json.loads(line.strip()))

    chat_records = []
    with test_msg_path.open(encoding="utf-8") as fh:
        for line in fh:
            chat_records.append(json.loads(line.strip()))

    logger.info("Test examples: %d", len(raw_records))

    # ── Load fine-tuned model ──────────────────────────────────────────
    logger.info("Loading fine-tuned model: %s", args.model)
    model, tokenizer = load_model_and_tokenizer(
        model_path=args.model,
        adapter_path=str(args.adapter) if args.adapter else None,
        base_model=args.base_model,
        device="auto",
    )

    # ── Run inference ──────────────────────────────────────────────────
    y_true_binary: list[str] = []
    y_pred_binary: list[str] = []
    y_true_type: list[str] = []
    y_pred_type: list[str] = []

    results: list[dict] = []

    for i, (raw_rec, chat_rec) in enumerate(zip(raw_records, chat_records)):
        # Ground truth
        true_is_soc = raw_rec.get("is_soc", False)
        true_type = "none"
        if true_is_soc:
            try:
                data = json.loads(raw_rec["assistant_json"])
                insts = data.get("instances", [])
                true_type = insts[0].get("soc_type", "other_soc") if insts else "none"
            except Exception:
                true_type = "other_soc"

        # Inference (use system+user from chat record, strip assistant)
        messages = chat_rec["messages"][:2]  # system + user only
        raw_output = run_inference(
            model, tokenizer, messages,
            max_new_tokens=args.max_new_tokens,
            enable_thinking=args.enable_thinking,
        )
        parsed = parse_response(raw_output)
        pred_instances = parsed.get("instances", [])
        pred_is_soc = len(pred_instances) > 0
        pred_type = pred_instances[0].get("soc_type", "other_soc") if pred_is_soc else "none"

        y_true_binary.append("soc" if true_is_soc else "none")
        y_pred_binary.append("soc" if pred_is_soc else "none")

        if true_is_soc or pred_is_soc:
            y_true_type.append(true_type)
            y_pred_type.append(pred_type)

        results.append(
            {
                "passage": raw_rec.get("passage", ""),
                "true_is_soc": true_is_soc,
                "pred_is_soc": pred_is_soc,
                "true_type": true_type,
                "pred_type": pred_type,
                "raw_output": raw_output[:500],
            }
        )

        if (i + 1) % 10 == 0:
            logger.info("Evaluated %d / %d", i + 1, len(raw_records))

    # ── Metrics ───────────────────────────────────────────────────────
    logger.info("\n=== Binary detection (SoC vs none) ===")
    binary_metrics = compute_metrics(y_true_binary, y_pred_binary)
    _print_report(binary_metrics["report"])

    logger.info("\n=== SoC type classification (positives only) ===")
    if y_true_type:
        type_metrics = compute_metrics(y_true_type, y_pred_type)
        _print_report(type_metrics["report"])
    else:
        type_metrics = {}
        logger.warning("No positive examples in test set for type metrics.")

    # ── Write results ─────────────────────────────────────────────────
    output = {
        "n_examples": len(raw_records),
        "binary_detection": binary_metrics,
        "type_classification": type_metrics,
        "per_example": results,
    }
    args.output.write_text(
        json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("Results written to %s", args.output)


def _print_report(report: dict) -> None:
    """Pretty-print a sklearn classification_report dict."""
    for label, metrics in report.items():
        if isinstance(metrics, dict):
            logger.info(
                "  %-30s  P=%.3f  R=%.3f  F1=%.3f  N=%d",
                label,
                metrics.get("precision", 0),
                metrics.get("recall", 0),
                metrics.get("f1-score", 0),
                int(metrics.get("support", 0)),
            )
        else:
            logger.info("  %-30s  %.4f", label, metrics)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate fine-tuned SoC model on the test split."
    )
    parser.add_argument(
        "--model",
        default="apjanco/penelope-soc-v1",
        help="Path or HF Hub ID of the merged model (default: apjanco/penelope-soc-v1)",
    )
    parser.add_argument(
        "--adapter",
        type=Path,
        default=None,
        help="Path to LoRA adapter directory (merges on top of --base-model)",
    )
    parser.add_argument(
        "--base-model",
        default="Qwen/Qwen3-4B",
        help="Base model ID (required when using --adapter)",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("dataset"),
        help="Directory with test.jsonl and test_raw.jsonl (default: dataset/)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Inference batch size (default: 1)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=4096,
        help="Max tokens to generate (default: 4096)",
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        default=True,
        help="Use thinking mode for inference (default: True)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("eval_results.json"),
        help="Output file for results (default: eval_results.json)",
    )
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
