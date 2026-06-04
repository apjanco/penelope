"""Merge LoRA adapter into the base model and optionally push to HuggingFace Hub.

Usage
-----
python scripts/export_model.py \\
    [--base-model Qwen/Qwen3-4B] \\
    [--adapter adapter/] \\
    [--output models/penelope-soc-v1] \\
    [--push-to-hub] \\
    [--hub-id apjanco/penelope-soc-v1] \\
    [--hub-private]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def export(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
    )

    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        logger.error(
            "Missing dependency: %s\n"
            "Install with: pip install transformers>=4.51.0 peft>=0.10.0 torch>=2.2.0",
            exc,
        )
        sys.exit(1)

    # ── Load base model ────────────────────────────────────────────────
    logger.info("Loading base model: %s", args.base_model)
    tokenizer = AutoTokenizer.from_pretrained(
        args.adapter,  # tokenizer was saved alongside the adapter
        trust_remote_code=True,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="cpu",           # CPU merge avoids GPU OOM
        trust_remote_code=True,
    )

    # ── Load and merge adapter ─────────────────────────────────────────
    logger.info("Loading adapter from: %s", args.adapter)
    model = PeftModel.from_pretrained(base_model, str(args.adapter))
    logger.info("Merging LoRA weights into base model...")
    model = model.merge_and_unload()
    model.eval()

    # ── Save merged model ──────────────────────────────────────────────
    output_path = args.output
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info("Saving merged model to %s", output_path)
    model.save_pretrained(str(output_path), safe_serialization=True)
    tokenizer.save_pretrained(str(output_path))
    logger.info("Saved %s", output_path)

    # ── Write model card ───────────────────────────────────────────────
    card_content = f"""\
---
language: en
license: apache-2.0
base_model: {args.base_model}
tags:
  - stream-of-consciousness
  - literary-analysis
  - text-classification
  - qwen3
  - peft
  - qlora
pipeline_tag: text-generation
---

# penelope-soc-v1

Fine-tuned [{args.base_model}](https://huggingface.co/{args.base_model}) for
stream-of-consciousness (SoC) detection and classification in literary texts.

## Model description

This model analyses a passage of prose and returns structured JSON identifying
SoC instances, their type (direct interior monologue, free association, etc.),
and supporting evidence.

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{args.hub_id or str(output_path)}")
tokenizer = AutoTokenizer.from_pretrained("{args.hub_id or str(output_path)}")

messages = [
    {{"role": "user", "content": "Source: Mrs Dalloway\\n\\n<passage text>"}},
]
text = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
)
inputs = tokenizer(text, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=4096, temperature=0.6, top_p=0.95)
print(tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True))
```
"""
    (output_path / "README.md").write_text(card_content, encoding="utf-8")

    # ── Push to Hub ────────────────────────────────────────────────────
    if args.push_to_hub:
        hub_id = args.hub_id
        logger.info("Pushing to HuggingFace Hub: %s", hub_id)
        model.push_to_hub(hub_id, private=args.hub_private)
        tokenizer.push_to_hub(hub_id, private=args.hub_private)
        logger.info("Successfully pushed to %s", hub_id)

    logger.info("Export complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge LoRA adapter and export the final model."
    )
    parser.add_argument(
        "--base-model",
        default="Qwen/Qwen3-4B",
        help="Base model HF ID (default: Qwen/Qwen3-4B)",
    )
    parser.add_argument(
        "--adapter",
        type=Path,
        default=Path("adapter"),
        help="Path to saved LoRA adapter directory (default: adapter/)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/penelope-soc-v1"),
        help="Output directory for merged model (default: models/penelope-soc-v1)",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push merged model to HuggingFace Hub after export",
    )
    parser.add_argument(
        "--hub-id",
        default="apjanco/penelope-soc-v1",
        help="HuggingFace Hub repository ID (default: apjanco/penelope-soc-v1)",
    )
    parser.add_argument(
        "--hub-private",
        action="store_true",
        help="Make the Hub repository private",
    )
    args = parser.parse_args()
    export(args)


if __name__ == "__main__":
    main()
