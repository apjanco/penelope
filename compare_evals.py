#!/usr/bin/env python3
"""Compare two Penelope eval result files and report improvements / regressions.

Usage:
    python compare_evals.py logs/eval_A_results.json logs/eval_B_results.json
    python compare_evals.py logs/eval_A_results.json logs/eval_B_results.json --label-a "SFT" --label-b "GRPO-4762"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


# ── helpers ─────────────────────────────────────────────────────────────────

def load(path: Path) -> dict:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def delta_str(a: float, b: float) -> str:
    d = b - a
    sign = "+" if d >= 0 else ""
    arrow = "▲" if d > 0.001 else ("▼" if d < -0.001 else "·")
    return f"{arrow} {sign}{d:+.4f}"


def pct(v: float) -> str:
    return f"{v * 100:.2f}%"


# ── section printers ─────────────────────────────────────────────────────────

def compare_binary(a: dict, b: dict, la: str, lb: str) -> None:
    print("\n━━━ Binary detection (SoC vs. none) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    rep_a = a["binary_detection"]["report"]
    rep_b = b["binary_detection"]["report"]

    metrics = ["precision", "recall", "f1-score"]
    keys = ["soc", "none", "macro avg", "weighted avg"]

    # Header
    col = 18
    print(f"{'':>{col}}  {'precision':>10}  {'recall':>10}  {'f1-score':>10}")
    for key in keys:
        if key not in rep_a or key not in rep_b:
            continue
        ra, rb = rep_a[key], rep_b[key]
        print(f"\n  {key}")
        print(f"  {'':>{col-2}}  {'precision':>10}  {'recall':>10}  {'f1-score':>10}")
        print(f"  {la:>{col}}  {pct(ra['precision']):>10}  {pct(ra['recall']):>10}  {pct(ra['f1-score']):>10}")
        print(f"  {lb:>{col}}  {pct(rb['precision']):>10}  {pct(rb['recall']):>10}  {pct(rb['f1-score']):>10}")
        row = "  " + " " * col + "  "
        row += "  ".join(f"{delta_str(ra[m], rb[m]):>10}" for m in metrics)
        print(row)

    acc_a = rep_a.get("accuracy", 0.0)
    acc_b = rep_b.get("accuracy", 0.0)
    print(f"\n  {'accuracy':>{col}}  {pct(acc_a):>10}   →   {pct(acc_b):>10}   {delta_str(acc_a, acc_b)}")


def compare_types(a: dict, b: dict, la: str, lb: str) -> None:
    print("\n━━━ Type classification (per-class F1) ━━━━━━━━━━━━━━━━━━━━━━━━━━")
    rep_a = a["type_classification"]["report"]
    rep_b = b["type_classification"]["report"]

    # Collect all SoC type keys (skip aggregate rows)
    skip = {"accuracy", "macro avg", "weighted avg"}
    all_keys = sorted((set(rep_a) | set(rep_b)) - skip)

    col = 32
    print(f"  {'class':>{col}}  {la:>10}  {lb:>10}  {'delta':>10}  {'support':>8}")
    print("  " + "─" * (col + 44))

    improved = regressed = unchanged = 0

    for key in all_keys:
        ra = rep_a.get(key, {})
        rb = rep_b.get(key, {})
        f1_a = ra.get("f1-score", 0.0)
        f1_b = rb.get("f1-score", 0.0)
        sup = int(ra.get("support", rb.get("support", 0)))
        d = f1_b - f1_a
        marker = "▲" if d > 0.001 else ("▼" if d < -0.001 else "·")
        if d > 0.001:
            improved += 1
        elif d < -0.001:
            regressed += 1
        else:
            unchanged += 1
        print(
            f"  {key:>{col}}  {pct(f1_a):>10}  {pct(f1_b):>10}  "
            f"{marker} {d:>+8.4f}  {sup:>8}"
        )

    print("  " + "─" * (col + 44))
    # Aggregates
    for agg in ("macro avg", "weighted avg", "accuracy"):
        if agg == "accuracy":
            va = rep_a.get("accuracy", 0.0)
            vb = rep_b.get("accuracy", 0.0)
            print(f"  {'accuracy':>{col}}  {pct(va):>10}  {pct(vb):>10}  {delta_str(va, vb):>10}")
        else:
            ra = rep_a.get(agg, {})
            rb = rep_b.get(agg, {})
            f1_a = ra.get("f1-score", 0.0)
            f1_b = rb.get("f1-score", 0.0)
            print(f"  {agg:>{col}}  {pct(f1_a):>10}  {pct(f1_b):>10}  {delta_str(f1_a, f1_b):>10}")

    print(f"\n  Classes improved: {improved}  |  regressed: {regressed}  |  unchanged: {unchanged}")


def compare_summary(a: dict, b: dict, la: str, lb: str) -> None:
    print("\n━━━ Top-level summary ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"  Examples evaluated:  {a.get('n_examples', '?')} ({la})  /  {b.get('n_examples', '?')} ({lb})")

    def _get(results: dict, section: str, key: str, metric: str) -> float:
        return results.get(section, {}).get("report", {}).get(key, {}).get(metric, 0.0)

    rows = [
        ("Binary SoC F1",       "binary_detection",   "soc",        "f1-score"),
        ("Binary macro F1",     "binary_detection",   "macro avg",  "f1-score"),
        ("Type weighted F1",    "type_classification","weighted avg","f1-score"),
        ("Type macro F1",       "type_classification","macro avg",  "f1-score"),
    ]
    col = 24
    print(f"\n  {'metric':>{col}}  {la:>10}  {lb:>10}  {'delta':>12}")
    print("  " + "─" * (col + 38))
    for label, section, key, metric in rows:
        va = _get(a, section, key, metric)
        vb = _get(b, section, key, metric)
        print(f"  {label:>{col}}  {pct(va):>10}  {pct(vb):>10}  {delta_str(va, vb):>12}")


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two Penelope eval result JSON files")
    parser.add_argument("file_a", type=Path, help="Baseline results (e.g. SFT adapter)")
    parser.add_argument("file_b", type=Path, help="New results (e.g. GRPO checkpoint)")
    parser.add_argument("--label-a", default=None, help="Label for file A (default: filename)")
    parser.add_argument("--label-b", default=None, help="Label for file B (default: filename)")
    args = parser.parse_args()

    for p in (args.file_a, args.file_b):
        if not p.exists():
            print(f"ERROR: file not found: {p}", file=sys.stderr)
            sys.exit(1)

    la = args.label_a or args.file_a.stem
    lb = args.label_b or args.file_b.stem

    a = load(args.file_a)
    b = load(args.file_b)

    print(f"\nComparing eval results")
    print(f"  A (baseline) : {args.file_a}  [{la}]")
    print(f"  B (new)      : {args.file_b}  [{lb}]")

    compare_summary(a, b, la, lb)
    compare_binary(a, b, la, lb)
    compare_types(a, b, la, lb)
    print()


if __name__ == "__main__":
    main()
