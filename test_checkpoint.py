#!/usr/bin/env python3
"""Quick smoke-test for adapter-grpo/checkpoints/checkpoint-4100.

Loads the LoRA adapter on top of the Qwen/Qwen3-4B base model and runs a
short Virginia Woolf passage through the model, printing the raw output.

Usage:
    python test_checkpoint.py
    python test_checkpoint.py --checkpoint adapter-grpo/checkpoints/checkpoint-4100
    python test_checkpoint.py --no-thinking   # faster, no <think> block
"""

import argparse
import json
import re
import sys
from pathlib import Path

CHECKPOINT_DEFAULT = "adapter-grpo/checkpoints/checkpoint-4100"
BASE_MODEL = "Qwen/Qwen3-4B"

# A passage rich in indirect interior monologue / free indirect discourse
# (Mrs. Dalloway, Virginia Woolf)
TEST_PASSAGE = """\
What a lark! What a plunge! For so it had always seemed to her when, with a
little squeak of the hinges, which she could hear now, she had burst open the
French windows and plunged at Bourton into the open air. How fresh, how calm,
stiller than this of course, the air was in the early morning; like the flap
of a wave; the kiss of a wave; chill and sharp and yet (for a girl of
eighteen as she then was) solemn, feeling as she did, standing there at the
open window, that something awful was about to happen; looking at the flowers,
at the trees with the smoke winding off them and the rooks rising, falling;
standing and looking until Peter Walsh said, "Musing among the vegetables?"
"""


def build_messages(passage: str) -> list[dict]:
    # Import here so the script is importable without heavy deps at module level
    from scripts.prompts import TRAINING_SYSTEM_PROMPT

    system = TRAINING_SYSTEM_PROMPT
    user = (
        "Source: Mrs Dalloway (Virginia Woolf)\n"
        "Chunk: 1 (opening)\n"
        "Position: 0\n\n"
        "[TEXT TO ANALYZE]\n"
        f"{passage}\n\n"
        "Analyze the [TEXT TO ANALYZE] section above. "
        "Identify and classify every SoC passage. "
        "Respond with JSON only."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def load_model(checkpoint: str, device: str = "auto"):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading base model: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )

    print(f"Applying LoRA adapter: {checkpoint}")
    model = PeftModel.from_pretrained(base, checkpoint)
    model.eval()
    print("Model ready.\n")
    return tokenizer, model


def run_inference(
    tokenizer, model, messages: list[dict], enable_thinking: bool = True
) -> str:
    import torch

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=2048,
            temperature=0.6 if enable_thinking else 0.7,
            top_p=0.95 if enable_thinking else 0.8,
            top_k=20,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = output_ids[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(generated, skip_special_tokens=True)


def pretty_print(raw: str) -> None:
    # Separate think block from JSON
    think_match = re.search(r"<think>(.*?)</think>", raw, re.DOTALL)
    if think_match:
        print("=== <think> block ===")
        think_text = think_match.group(1).strip()
        # Print first / last 300 chars if long
        if len(think_text) > 800:
            print(think_text[:400])
            print(f"\n... [{len(think_text) - 800} chars omitted] ...\n")
            print(think_text[-400:])
        else:
            print(think_text)
        print()

    json_match = re.search(r"\{.*\}", raw, re.DOTALL)
    if json_match:
        print("=== JSON output ===")
        try:
            parsed = json.loads(json_match.group(0))
            print(json.dumps(parsed, indent=2))
        except json.JSONDecodeError:
            print("[Could not parse JSON]\n", json_match.group(0))
    else:
        print("=== Raw output (no JSON found) ===")
        print(raw)


def main() -> None:
    parser = argparse.ArgumentParser(description="Test a GRPO checkpoint")
    parser.add_argument(
        "--checkpoint",
        default=CHECKPOINT_DEFAULT,
        help=f"Path to the LoRA checkpoint directory (default: {CHECKPOINT_DEFAULT})",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device map for from_pretrained (default: auto)",
    )
    parser.add_argument(
        "--no-thinking",
        action="store_true",
        help="Disable Qwen3 thinking mode for faster (but shallower) output",
    )
    args = parser.parse_args()

    checkpoint = str(Path(args.checkpoint).resolve())
    if not Path(checkpoint).exists():
        print(f"ERROR: checkpoint not found: {checkpoint}", file=sys.stderr)
        sys.exit(1)

    tokenizer, model = load_model(checkpoint, device=args.device)
    messages = build_messages(TEST_PASSAGE)

    print("=== Test passage ===")
    print(TEST_PASSAGE)
    print()

    enable_thinking = not args.no_thinking
    print(f"Running inference (thinking={'on' if enable_thinking else 'off'}) …\n")
    raw = run_inference(tokenizer, model, messages, enable_thinking=enable_thinking)

    pretty_print(raw)


if __name__ == "__main__":
    main()
