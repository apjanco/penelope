"""Stability-confirmed analysis of a k-resampled scored file.

Reusable across batches (Lighthouse+Dalloway now, further Woolf/other-author
runs later). Implements AMBIGUITY.md's own distinction, applied to the new
mediation_level/fit_category schema instead of the old flat soc_type:

For each (passage_id, reader_id): mode mediation_level/soc_type across k
runs, and within-model consistency = fraction of valid runs matching the
mode. With k=3 "stable" means a majority (>=2/3) rather than AMBIGUITY.md's
0.8 threshold, which was calibrated for k=10 and is unsatisfiable at k=3
except by unanimity.

Cross-reader, per passage:
- both readers stable, mode is_soc=False           -> confirmed_negative
- readers split on is_soc                          -> detection_boundary
- both stable, same soc_type, mediation_level exact -> technique_fit_convergent
- both stable, differ by <=1 mediation_level        -> boundary_tension_convergent
- both stable, differ by >=2, or either no_type_fits -> strong_divergence
- either reader unstable within its own k runs      -> noisy (excluded from
  the above — random disagreement, not yet a finding, per AMBIGUITY.md)
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def mode_and_consistency(values: list):
    if not values:
        return None, 0.0
    counts = Counter(values)
    mode_val, mode_n = counts.most_common(1)[0]
    return mode_val, mode_n / len(values)


def summarize_reader(runs: list[dict], threshold: float) -> dict:
    """runs: the k result dicts for one (passage_id, reader_id), errors excluded.

    threshold: AMBIGUITY.md's stability bar is 0.8, calibrated for k=10 (8/10
    runs matching the mode). At k=3 that's unsatisfiable except by unanimity,
    so callers running k=3 data should pass a lower threshold (e.g. 2/3); at
    k=10 pass 0.8 directly, which is what this function defaults to.
    """
    is_soc_vals = [r.get("is_soc") for r in runs]
    is_soc_mode, is_soc_consistency = mode_and_consistency(is_soc_vals)

    soc_runs = [r for r in runs if r.get("is_soc")]
    if not soc_runs:
        return {"is_soc_mode": is_soc_mode, "is_soc_consistency": is_soc_consistency, "stable": is_soc_consistency >= threshold}

    med_vals = [r.get("mediation_level") for r in soc_runs if r.get("mediation_level") is not None]
    type_vals = [r.get("soc_type") for r in soc_runs if r.get("soc_type")]
    fit_vals = [r.get("fit_category") for r in soc_runs if r.get("fit_category")]

    med_mode, med_consistency = mode_and_consistency(med_vals)
    type_mode, type_consistency = mode_and_consistency(type_vals)
    fit_mode, fit_consistency = mode_and_consistency(fit_vals)

    # A reader is "stable" on this passage if it consistently says SoC AND
    # its mediation_level mode holds for at least `threshold` of runs.
    stable = is_soc_consistency >= threshold and med_consistency >= threshold

    # keep one representative run (highest-consistency example) for quoting
    best_run = max(soc_runs, key=lambda r: 1 if r.get("mediation_level") == med_mode else 0)

    return {
        "is_soc_mode": is_soc_mode,
        "is_soc_consistency": is_soc_consistency,
        "mediation_level_mode": med_mode,
        "mediation_level_consistency": med_consistency,
        "soc_type_mode": type_mode,
        "fit_category_mode": fit_mode,
        "stable": stable,
        "example": {
            "argument": best_run.get("argument", ""),
            "alternative_reasoning": best_run.get("alternative_reasoning", ""),
            "antimimetic": best_run.get("antimimetic", False),
            "antimimetic_defense": best_run.get("antimimetic_defense", ""),
            "evidence": best_run.get("evidence", []),
        },
    }


def classify_passage(reader_summaries: dict[str, dict]) -> str:
    labels = list(reader_summaries.values())
    if len(labels) < 2:
        return "single_reader_only"

    if any(not l["stable"] for l in labels if "mediation_level_mode" in l or l["is_soc_mode"] is False):
        # unstable readers (no majority even on is_soc, or on level) -> noisy,
        # UNLESS both simply and consistently say not-SoC (that's a real negative).
        pass

    is_soc_modes = {l["is_soc_mode"] for l in labels}
    if is_soc_modes == {False}:
        return "confirmed_negative"
    if len(is_soc_modes) > 1:
        return "detection_boundary"

    # both say is_soc True from here
    if not all(l["stable"] for l in labels):
        return "noisy"

    med_levels = [l["mediation_level_mode"] for l in labels]
    fit_modes = [l["fit_category_mode"] for l in labels]
    types = [l["soc_type_mode"] for l in labels]

    if any(f == "no_type_fits" for f in fit_modes) and len(set(fit_modes)) > 1:
        return "strong_divergence"
    if None in med_levels:
        return "noisy"

    spread = max(med_levels) - min(med_levels)
    if spread == 0 and len(set(types)) == 1:
        return "technique_fit_convergent"
    if spread <= 1:
        return "boundary_tension_convergent"
    return "strong_divergence"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default=None, help="Markdown report path (default: <input>_report.md)")
    ap.add_argument("--examples-per-category", type=int, default=3)
    ap.add_argument(
        "--stability-threshold", type=float, default=0.8,
        help="Fraction of k runs that must agree to call a reader 'stable' on a passage "
        "(default 0.8, AMBIGUITY.md's k=10 bar; use ~0.67 for k=3 data).",
    )
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output) if args.output else in_path.with_name(in_path.stem + "_report.md")

    rows = [json.loads(l) for l in in_path.read_text().splitlines() if json.loads(l).get("error") is not True]
    # (re-parse once more cleanly, above line double-parses but file is small; fine)
    rows = [json.loads(l) for l in in_path.read_text().splitlines()]
    rows = [r for r in rows if not r.get("error")]

    by_passage_reader = defaultdict(list)
    meta = {}
    for r in rows:
        key = (r["passage_id"], r["reader_id"])
        by_passage_reader[key].append(r)
        meta[r["passage_id"]] = {"author": r["author"], "title": r["title"], "year": r["year"]}

    by_passage = defaultdict(dict)
    for (pid, reader), runs in by_passage_reader.items():
        by_passage[pid][reader] = summarize_reader(runs, args.stability_threshold)

    classification = {}
    for pid, reader_summaries in by_passage.items():
        classification[pid] = classify_passage(reader_summaries)

    # ---- aggregate stats ----
    by_title = defaultdict(lambda: Counter())
    for pid, cls in classification.items():
        by_title[meta[pid]["title"]][cls] += 1

    overall = Counter(classification.values())
    total = len(classification)

    lines = []
    lines.append("# Stability-confirmed analysis: " + in_path.name + "\n")
    lines.append(f"{total} passages, {len(rows)} valid (non-error) reader-runs.\n")

    lines.append("## Overall classification\n")
    lines.append("| category | n | % |")
    lines.append("|---|---|---|")
    for cat, n in overall.most_common():
        lines.append(f"| {cat} | {n} | {100*n/total:.1f}% |")

    lines.append("\n## By work\n")
    cats = sorted({c for counts in by_title.values() for c in counts})
    lines.append("| title | " + " | ".join(cats) + " | n |")
    lines.append("|---|" + "---|" * (len(cats) + 1))
    for title, counts in by_title.items():
        row_total = sum(counts.values())
        cells = [str(counts.get(c, 0)) for c in cats]
        lines.append(f"| {title} | " + " | ".join(cells) + f" | {row_total} |")

    # mean mediation_level per work, over passages where both readers were stable
    lines.append("\n## Mean mediation_level per work (stable, is_soc=true cases only)\n")
    lines.append("| title | n | mean mediation_level (avg of both readers' modes) |")
    lines.append("|---|---|---|")
    med_by_title = defaultdict(list)
    for pid, reader_summaries in by_passage.items():
        if classification[pid] not in ("technique_fit_convergent", "boundary_tension_convergent", "strong_divergence"):
            continue
        levels = [s.get("mediation_level_mode") for s in reader_summaries.values() if s.get("mediation_level_mode") is not None]
        if levels:
            med_by_title[meta[pid]["title"]].append(sum(levels) / len(levels))
    for title, vals in med_by_title.items():
        lines.append(f"| {title} | {len(vals)} | {sum(vals)/len(vals):.2f} |")

    # quotable examples per category
    lines.append("\n## Examples\n")
    for cat in ["technique_fit_convergent", "boundary_tension_convergent", "strong_divergence"]:
        pids = [pid for pid, c in classification.items() if c == cat][: args.examples_per_category]
        if not pids:
            continue
        lines.append(f"### {cat}\n")
        for pid in pids:
            m = meta[pid]
            lines.append(f"**{pid}** ({m['title']}, {m['year']})\n")
            for reader, s in by_passage[pid].items():
                if "example" not in s:
                    continue
                lines.append(f"- *{reader}* — level {s.get('mediation_level_mode')}, {s.get('soc_type_mode')}, {s.get('fit_category_mode')}")
                lines.append(f"  - argument: {s['example']['argument']}")
                if s["example"]["alternative_reasoning"]:
                    lines.append(f"  - alternative_reasoning: {s['example']['alternative_reasoning']}")
                if s["example"]["antimimetic"]:
                    lines.append(f"  - **antimimetic**: {s['example']['antimimetic_defense']}")
            lines.append("")

    out_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out_path}")
    print("\n".join(lines[:60]))


if __name__ == "__main__":
    main()
