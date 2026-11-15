"""Strip recorded-failure rows from scored/*.jsonl.

score.py now retries failed triples instead of treating them as done, so
purging is not required for correctness any more. It is still worth doing
before analysis: an error row carries no soc_type, mediation_level or
is_soc, and every consumer that iterates scored/*.jsonl has to remember to
skip it. Leaving 29k of them in the corpus is an invitation to a silent
denominator bug.

Writes a .bak alongside each modified file. Dry-run by default.
"""
from __future__ import annotations
import argparse, json, shutil
from pathlib import Path

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true", help="actually rewrite files")
    ap.add_argument("--dir", default="scored")
    args = ap.parse_args()

    tot_kept = tot_dropped = 0
    for f in sorted(Path(args.dir).glob("*.jsonl")):
        kept, dropped = [], 0
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                kept.append(line); continue
            if "error" in r:
                dropped += 1
            else:
                kept.append(line)
        if not dropped:
            continue
        tot_kept += len(kept); tot_dropped += dropped
        print(f"  {f.name:48s} drop {dropped:>6,}  keep {len(kept):>7,}")
        if args.apply:
            shutil.copy2(f, f.with_suffix(".jsonl.bak"))
            f.write_text("\n".join(kept) + ("\n" if kept else ""))
    verb = "dropped" if args.apply else "would drop"
    print(f"\n  {verb} {tot_dropped:,} error rows; {tot_kept:,} real rows retained in touched files")
    if not args.apply:
        print("  (dry run — pass --apply to rewrite, .bak files are kept)")

if __name__ == "__main__":
    main()
