#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.prompts import TRAINING_SYSTEM_PROMPT

pd.set_option("display.max_colwidth", 220)

BASE_MODEL_CANDIDATE = Path(
    "/scratch/gpfs/MM4/apjanco/.cache/huggingface/hub/models--Qwen--Qwen3-4B/"
    "snapshots/1cfa9a7208912126459214e8b04321603b3df60c"
)
BASE_MODEL = str(BASE_MODEL_CANDIDATE) if BASE_MODEL_CANDIDATE.exists() else "Qwen/Qwen3-4B"

SFT_ADAPTER_PATH = Path("adapter-v3")
GRPO_CKPT_CANDIDATE = Path("adapter-v3-grpo/checkpoints/checkpoint-5291")
GRPO_ADAPTER_PATH = GRPO_CKPT_CANDIDATE if GRPO_CKPT_CANDIDATE.exists() else Path("adapter-v3-grpo")

MODEL_SPECS = {
    "v3_sft": {"label": "v3_sft", "adapter_path": SFT_ADAPTER_PATH},
    "v3_grpo": {"label": "v3_grpo", "adapter_path": GRPO_ADAPTER_PATH},
}

CURATED_PASSAGES = [
    {
        "id": "soc_direct_1",
        "source": "Curated direct interior",
        "kind": "clear_soc",
        "gold_is_soc": True,
        "gold_primary": "direct_interior_monologue",
        "gold_alt_types": ["free_association"],
        "text": "I cannot keep the thought still, not one of them, they run ahead of me and double back, and if I close my eyes I hear the bell again, no, not a bell, the spoon against the glass, and why did I say yes, why did I nod, foolish, foolish.",
    },
    {
        "id": "soc_indirect_1",
        "source": "Curated indirect interior",
        "kind": "clear_soc",
        "gold_is_soc": True,
        "gold_primary": "indirect_interior_monologue",
        "gold_alt_types": ["omniscient_description"],
        "text": "She crossed the square more quickly than she meant to, for of course he would be there, and of course she would not look, and was that not exactly the sort of weakness she had promised herself she had finished with years ago.",
    },
    {
        "id": "non_soc_1",
        "source": "Curated external narration",
        "kind": "clear_non_soc",
        "gold_is_soc": False,
        "gold_primary": "none",
        "gold_alt_types": [],
        "text": "The train arrived at half past six. Three porters unloaded the trunks, stacked them by the station wall, and waited for instructions from the conductor.",
    },
    {
        "id": "non_soc_2",
        "source": "Curated descriptive narration",
        "kind": "clear_non_soc",
        "gold_is_soc": False,
        "gold_primary": "none",
        "gold_alt_types": [],
        "text": "Rain moved across the field in narrow bands. The barn roof shone, then dulled, while crows gathered along the fence and settled for the night.",
    },
    {
        "id": "ambig_hybrid_1",
        "source": "Curated hybrid candidate",
        "kind": "ambiguous",
        "gold_is_soc": True,
        "gold_primary": "hybrid",
        "gold_alt_types": ["free_association", "imagery", "simulation_state_of_mind"],
        "text": "The room turns to water and then to wool, soft and choking, and her name is a thread in my mouth, blue thread, bitter thread, pull it and the whole evening comes apart.",
    },
    {
        "id": "ambig_reverie_1",
        "source": "Curated reverie candidate",
        "kind": "ambiguous",
        "gold_is_soc": True,
        "gold_primary": "reverie_fantasy",
        "gold_alt_types": ["simulation_state_of_mind", "imagery"],
        "text": "If I stepped through that doorway now the whole city would fold like paper, and mother would be in the kitchen again, humming, though the smell of smoke says this is no kitchen and no year I can name.",
    },
    {
        "id": "ambig_indirect_omniscient_1",
        "source": "Curated indirect-omniscient edge",
        "kind": "ambiguous",
        "gold_is_soc": True,
        "gold_primary": "indirect_interior_monologue",
        "gold_alt_types": ["omniscient_description"],
        "text": "He paused before answering, astonished at how quickly pride dressed itself as caution; perhaps that was always his trick, to call fear prudence and feel wise for it.",
    },
    {
        "id": "soc_orthographic_1",
        "source": "Curated orthographic marker",
        "kind": "clear_soc",
        "gold_is_soc": True,
        "gold_primary": "orthographic_marker",
        "gold_alt_types": ["direct_interior_monologue"],
        "text": "No no no -- not again -- if the door clicks I will -- I cannot -- stop -- breathe -- count -- one two three --",
    },
]

CONFIDENCE_MAP = {"high": 0.9, "medium": 0.7, "low": 0.55}
NON_SOC_CONFIDENCE_DEFAULT = 0.65


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the v3 ambiguity comparison as a batch script.")
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--models", nargs="+", default=["v3_sft", "v3_grpo"])
    parser.add_argument("--output-dir", default="outputs/notebook_demo")
    parser.add_argument("--tag", default=None, help="Optional run tag prefix. Defaults to a UTC timestamp.")
    parser.add_argument("--real-data-path", default="dataset/test_raw.jsonl")
    parser.add_argument("--n-real-soc", type=int, default=3)
    parser.add_argument("--n-real-non-soc", type=int, default=3)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--max-new-tokens", type=int, default=1200)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--disable-thinking", action="store_true")
    parser.add_argument("--allow-cpu-fallback", action="store_true")
    parser.add_argument("--no-real-samples", action="store_true")
    return parser.parse_args()


def resolve_runtime_config(allow_cpu_fallback: bool) -> dict[str, Any]:
    if not torch.cuda.is_available():
        if allow_cpu_fallback:
            return {
                "device_map": "cpu",
                "torch_dtype": torch.float32,
                "attn_implementation": "eager",
                "runtime_note": "CUDA not available; using CPU inference.",
                "cuda_usable": False,
            }
        raise RuntimeError("CUDA is not available. Re-run on a GPU node or pass --allow-cpu-fallback.")

    supported_arches = {arch.replace("sm_", "") for arch in torch.cuda.get_arch_list()}
    device_props = torch.cuda.get_device_properties(0)
    current_arch = f"{device_props.major}{device_props.minor}"
    cuda_usable = current_arch in supported_arches

    if not cuda_usable:
        message = (
            f"CUDA device {device_props.name} has compute capability {device_props.major}.{device_props.minor}, "
            f"but this PyTorch build only supports {sorted(supported_arches)}."
        )
        if allow_cpu_fallback:
            return {
                "device_map": "cpu",
                "torch_dtype": torch.float32,
                "attn_implementation": "eager",
                "runtime_note": f"{message} Falling back to CPU.",
                "cuda_usable": False,
            }
        raise RuntimeError(f"{message} Use an sm_80+ node or pass --allow-cpu-fallback.")

    torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return {
        "device_map": "auto",
        "torch_dtype": torch_dtype,
        "attn_implementation": "sdpa",
        "runtime_note": f"Using CUDA on {device_props.name} (cc {device_props.major}.{device_props.minor}).",
        "cuda_usable": True,
    }


def build_messages(passage: str, source: str) -> list[dict[str, str]]:
    user = (
        f"Source: {source}\n"
        "Chunk: demo-0 (notebook-demo)\n"
        "Position: 0\n\n"
        "[TEXT TO ANALYZE]\n"
        f"{passage}\n\n"
        "Analyze the [TEXT TO ANALYZE] section above. "
        "Identify and classify every SoC passage. "
        "Respond with JSON only."
    )
    return [
        {"role": "system", "content": TRAINING_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def load_model_bundle(
    model_id: str,
    base_model: str,
    runtime_config: dict[str, Any],
) -> tuple[AutoTokenizer, PeftModel]:
    if model_id not in MODEL_SPECS:
        raise KeyError(f"Unknown model_id: {model_id}")

    adapter_path = Path(MODEL_SPECS[model_id]["adapter_path"])
    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter path does not exist for {model_id}: {adapter_path}")

    tokenizer = AutoTokenizer.from_pretrained(str(adapter_path), trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=runtime_config["torch_dtype"],
        device_map=runtime_config["device_map"],
        trust_remote_code=True,
        attn_implementation=runtime_config["attn_implementation"],
    )
    model = PeftModel.from_pretrained(base, str(adapter_path))
    model.eval()
    return tokenizer, model


def run_inference(
    model,
    tokenizer,
    messages: list[dict[str, str]],
    max_new_tokens: int,
    enable_thinking: bool,
    temperature: float,
    top_p: float,
    top_k: int,
) -> str:
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

    generated = outputs[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(generated, skip_special_tokens=True)


def parse_response(raw: str) -> dict[str, Any]:
    raw_clean = raw.strip()
    think_match = re.search(r"<think>(.*?)</think>", raw_clean, re.DOTALL)
    think_text = think_match.group(1).strip() if think_match else ""

    json_candidate = re.sub(r"<think>.*?</think>", "", raw_clean, flags=re.DOTALL).strip()
    json_match = re.search(r"\{.*\}", json_candidate, re.DOTALL)
    if not json_match:
        return {"instances": [], "_think": think_text, "_raw": raw_clean}

    try:
        parsed = json.loads(json_match.group(0))
    except json.JSONDecodeError:
        parsed = {"instances": []}

    if "instances" not in parsed or not isinstance(parsed["instances"], list):
        parsed["instances"] = []

    parsed["_think"] = think_text
    parsed["_raw"] = raw_clean
    return parsed


def top_instance(parsed: dict[str, Any]) -> dict[str, Any]:
    instances = parsed.get("instances", [])
    if isinstance(instances, list) and instances and isinstance(instances[0], dict):
        return instances[0]
    return {}


def normalize_secondary_devices(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return []


def has_multi_type_signal(inst: dict[str, Any]) -> bool:
    secondary = normalize_secondary_devices(inst.get("secondary_devices", []))
    notes = str(inst.get("notes", "")).lower()
    explanation = str(inst.get("explanation", "")).lower()
    hedge_terms = ["hybrid", "ambiguous", "could", "also", "overlap", "mixed", "alternative"]
    hedged = any(term in notes or term in explanation for term in hedge_terms)
    return bool(secondary) or hedged


def prediction_confidence(pred_is_soc: bool, inst: dict[str, Any]) -> float:
    if pred_is_soc:
        return float(CONFIDENCE_MAP.get(str(inst.get("confidence", "medium")).lower(), 0.7))
    return float(NON_SOC_CONFIDENCE_DEFAULT)


def compute_ece(correct: np.ndarray, conf: np.ndarray, n_bins: int = 10) -> float:
    if len(correct) == 0:
        return 0.0
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for index in range(n_bins):
        low, high = bins[index], bins[index + 1]
        if index < n_bins - 1:
            mask = (conf >= low) & (conf < high)
        else:
            mask = (conf >= low) & (conf <= high)
        if not np.any(mask):
            continue
        acc_bin = float(np.mean(correct[mask]))
        conf_bin = float(np.mean(conf[mask]))
        weight = float(np.mean(mask))
        ece += weight * abs(acc_bin - conf_bin)
    return float(ece)


def compute_risk_coverage(correct: np.ndarray, conf: np.ndarray) -> tuple[pd.DataFrame, float]:
    if len(correct) == 0:
        return pd.DataFrame(columns=["coverage", "risk"]), 0.0

    order = np.argsort(-conf)
    correct_sorted = correct[order]
    coverages = []
    risks = []
    for size in range(1, len(correct_sorted) + 1):
        coverage = size / len(correct_sorted)
        risk = 1.0 - float(np.mean(correct_sorted[:size]))
        coverages.append(coverage)
        risks.append(risk)

    curve = pd.DataFrame({"coverage": coverages, "risk": risks})
    aurc = float(np.trapz(curve["risk"].values, curve["coverage"].values))
    return curve, aurc


def load_real_samples(path: Path, n_soc: int, n_non_soc: int, seed: int) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    soc = [record for record in records if bool(record.get("is_soc"))]
    non_soc = [record for record in records if not bool(record.get("is_soc"))]

    rng = random.Random(seed)
    if len(soc) > n_soc:
        soc = rng.sample(soc, n_soc)
    if len(non_soc) > n_non_soc:
        non_soc = rng.sample(non_soc, n_non_soc)

    output = []
    for index, record in enumerate(soc + non_soc):
        gold_soc = bool(record.get("is_soc"))
        output.append(
            {
                "id": f"real_{index:02d}",
                "source": str(record.get("source_file", path)),
                "kind": "real_soc" if gold_soc else "real_non_soc",
                "gold_is_soc": gold_soc,
                "gold_primary": "unknown",
                "gold_alt_types": [],
                "text": str(record.get("passage", ""))[:3000],
            }
        )
    return output


def flatten_compare_table(df_compare: pd.DataFrame) -> pd.DataFrame:
    flattened = df_compare.copy()
    flattened.columns = [
        f"{left}_{right}" if isinstance(left, str) else f"{left[0]}_{left[1]}"
        for left, right in flattened.columns.to_flat_index()
    ]
    return flattened.reset_index()


def export_run(
    export_dir: Path,
    run_tag: str,
    runtime_config: dict[str, Any],
    args: argparse.Namespace,
    df: pd.DataFrame,
    df_compare: pd.DataFrame,
    summary_df: pd.DataFrame,
    ambig_view: pd.DataFrame,
    risk_curves: dict[str, pd.DataFrame],
) -> dict[str, Path]:
    export_dir.mkdir(parents=True, exist_ok=True)
    compare_flat = flatten_compare_table(df_compare)
    risk_curve_df = pd.concat(risk_curves.values(), ignore_index=True) if risk_curves else pd.DataFrame()

    outputs = {
        "predictions_csv": export_dir / f"{run_tag}_predictions.csv",
        "predictions_json": export_dir / f"{run_tag}_predictions.json",
        "summary_csv": export_dir / f"{run_tag}_summary.csv",
        "summary_json": export_dir / f"{run_tag}_summary.json",
        "ambiguity_csv": export_dir / f"{run_tag}_ambiguity.csv",
        "compare_csv": export_dir / f"{run_tag}_compare.csv",
        "risk_curves_csv": export_dir / f"{run_tag}_risk_curves.csv",
        "manifest_json": export_dir / f"{run_tag}_manifest.json",
    }

    df.to_csv(outputs["predictions_csv"], index=False)
    df.to_json(outputs["predictions_json"], orient="records", indent=2)
    summary_df.to_csv(outputs["summary_csv"], index=False)
    summary_df.to_json(outputs["summary_json"], orient="records", indent=2)
    ambig_view.to_csv(outputs["ambiguity_csv"], index=False)
    compare_flat.to_csv(outputs["compare_csv"], index=False)
    risk_curve_df.to_csv(outputs["risk_curves_csv"], index=False)

    manifest = {
        "run_tag": run_tag,
        "base_model": args.base_model,
        "models": args.models,
        "runtime_config": {
            key: str(value) if isinstance(value, torch.dtype) else value
            for key, value in runtime_config.items()
        },
        "n_rows": int(len(df)),
        "n_cases": int(df["id"].nunique()) if not df.empty else 0,
        "output_files": {key: str(path) for key, path in outputs.items()},
    }
    outputs["manifest_json"].write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return outputs


def main() -> None:
    args = parse_args()
    runtime_config = resolve_runtime_config(allow_cpu_fallback=args.allow_cpu_fallback)

    print(f"BASE_MODEL={args.base_model}")
    for model_id in args.models:
        model_path = Path(MODEL_SPECS[model_id]["adapter_path"])
        print(f"{model_id}: {model_path.resolve()} (exists={model_path.exists()})")
    print(runtime_config["runtime_note"])
    print(
        "device_map={}, dtype={}, cuda_available={}, cuda_usable={}".format(
            runtime_config["device_map"],
            runtime_config["torch_dtype"],
            torch.cuda.is_available(),
            runtime_config["cuda_usable"],
        )
    )

    real_samples = []
    if not args.no_real_samples:
        real_samples = load_real_samples(
            Path(args.real_data_path),
            args.n_real_soc,
            args.n_real_non_soc,
            args.seed,
        )
    print(f"Loaded real samples: {len(real_samples)}")

    passages = CURATED_PASSAGES + real_samples
    print(f"Total passages to run: {len(passages)}")

    rows: list[dict[str, Any]] = []
    enable_thinking = not args.disable_thinking
    for model_id in args.models:
        print(f"Running model: {model_id}")
        tokenizer, model = load_model_bundle(model_id, args.base_model, runtime_config)
        for passage in passages:
            messages = build_messages(passage["text"], passage["source"])
            raw = run_inference(
                model,
                tokenizer,
                messages,
                max_new_tokens=args.max_new_tokens,
                enable_thinking=enable_thinking,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
            )
            parsed = parse_response(raw)
            inst = top_instance(parsed)
            pred_instances = parsed.get("instances", [])
            pred_is_soc = isinstance(pred_instances, list) and len(pred_instances) > 0
            secondary_list = normalize_secondary_devices(inst.get("secondary_devices", []))
            conf_num = prediction_confidence(pred_is_soc, inst)
            correct_binary = bool(pred_is_soc == passage["gold_is_soc"])

            rows.append(
                {
                    "model_id": model_id,
                    "id": passage["id"],
                    "kind": passage["kind"],
                    "source": passage["source"],
                    "gold_is_soc": passage["gold_is_soc"],
                    "gold_primary": passage["gold_primary"],
                    "gold_alt_types": ", ".join(passage["gold_alt_types"]) if passage["gold_alt_types"] else "",
                    "pred_is_soc": pred_is_soc,
                    "pred_type": inst.get("soc_type", "none") if pred_is_soc else "none",
                    "pred_secondary": ", ".join(secondary_list),
                    "pred_confidence": inst.get("confidence", ""),
                    "pred_confidence_num": conf_num,
                    "correct_binary": correct_binary,
                    "pred_explanation": inst.get("explanation", ""),
                    "pred_notes": inst.get("notes", ""),
                    "multi_type_signal": has_multi_type_signal(inst) if pred_is_soc else False,
                    "think_preview": parsed.get("_think", "")[:320],
                    "raw_output": parsed.get("_raw", raw),
                }
            )

        del model
        del tokenizer
        if torch.cuda.is_available() and runtime_config["cuda_usable"]:
            torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    compare_cols = [
        "pred_is_soc",
        "pred_type",
        "pred_secondary",
        "pred_confidence",
        "pred_confidence_num",
        "multi_type_signal",
    ]
    df_compare = (
        df.set_index(["id", "kind", "gold_is_soc", "gold_primary", "gold_alt_types", "model_id"])[compare_cols]
        .unstack("model_id")
    )
    print(f"Rows generated: {len(df)}")

    eval_df = df[df["gold_is_soc"].isin([True, False])].copy()
    summary_rows = []
    risk_curves: dict[str, pd.DataFrame] = {}
    for model_id, group in eval_df.groupby("model_id"):
        y_true = group["gold_is_soc"].astype(bool).values
        y_pred = group["pred_is_soc"].astype(bool).values
        tp = int(np.sum((y_true == True) & (y_pred == True)))
        tn = int(np.sum((y_true == False) & (y_pred == False)))
        fp = int(np.sum((y_true == False) & (y_pred == True)))
        fn = int(np.sum((y_true == True) & (y_pred == False)))
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        accuracy = (tp + tn) / len(group) if len(group) else 0.0
        conf = group["pred_confidence_num"].astype(float).values
        correct = group["correct_binary"].astype(float).values
        ece = compute_ece(correct, conf, n_bins=10)
        brier = float(np.mean((conf - correct) ** 2))
        curve_df, aurc = compute_risk_coverage(correct, conf)
        curve_df["model_id"] = model_id
        risk_curves[model_id] = curve_df
        summary_rows.append(
            {
                "model_id": model_id,
                "n_cases": int(len(group)),
                "tp": tp,
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "precision": round(precision, 3),
                "recall": round(recall, 3),
                "f1": round(f1, 3),
                "accuracy": round(accuracy, 3),
                "ece_proxy": round(ece, 4),
                "brier_proxy": round(brier, 4),
                "aurc_proxy": round(aurc, 4),
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values("model_id").reset_index(drop=True)
    ambig_view = df[(df["kind"] == "ambiguous") | (df["multi_type_signal"] == True)].copy()

    run_tag = args.tag or pd.Timestamp.utcnow().strftime("v3_demo_%Y%m%d_%H%M%S")
    outputs = export_run(
        Path(args.output_dir),
        run_tag,
        runtime_config,
        args,
        df,
        df_compare,
        summary_df,
        ambig_view,
        risk_curves,
    )

    print("Exported files:")
    for path in outputs.values():
        print(f"- {path}")


if __name__ == "__main__":
    main()