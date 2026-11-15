"""Report per-reader coverage gaps that are actually fillable.

A gap is only fillable if the work's existing rows live in scored/<stem>.jsonl.
score.py resumes by reading its own --output file, so for a work whose rows sit
under a different filename (To the Lighthouse and Mrs Dalloway both live in
woolf_lighthouse_dalloway.jsonl) pointing a top-up job at scored/<stem>.jsonl
finds an empty file, skips nothing, and rescores the whole work. Job 14787262
did exactly that: it burned its hour duplicating 455 passages and timed out at
work 14 of 39.

Usage:  python find_reader_gaps.py --reader llama [--out llama_gap_files.txt]
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

TV = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reader", required=True)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--out")
    args = ap.parse_args()

    man = {e["source_file"][:-4]: e for e in
           json.loads((TV / "corpus_manifest.json").read_text())["corpus"]}

    # Count the reader's rows per passage_id, and record which FILES hold them.
    have = collections.Counter()
    homes = collections.defaultdict(set)
    # scored_local/ and scored_jev/ hold readers that never write into scored/:
    # the locally-served models (in-copyright texts must not leave Princeton)
    # and Jev. Searching only scored/ made this report every work as missing
    # for qwen3.6-local when most were already done.
    for sdir in (TV / "scored", TV / "scored_local", TV / "scored_jev"):
        if not sdir.exists():
            continue
        for f in sdir.glob("*.jsonl"):
            for line in f.read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "error" in row or row.get("reader_id") != args.reader:
                    continue
                have[row["passage_id"]] += 1
                homes[row["passage_id"]].add(f.name)


    fillable, foreign = [], []
    for stem in sorted(man):
        p = TV / "passages" / f"{stem}.jsonl"
        if not p.exists():
            continue
        ids = [json.loads(l)["passage_id"] for l in p.read_text().splitlines()]
        missing = [i for i in ids if have[i] == 0]
        shallow = [i for i in ids if 0 < have[i] < args.k]
        if not (missing or shallow):
            continue
        # Where do this work's EXISTING rows live?
        elsewhere = set()
        for i in ids:
            elsewhere |= {h for h in homes[i] if h != f"{stem}.jsonl"}
        own = (TV / "scored" / f"{stem}.jsonl").exists()
        if elsewhere and not own:
            foreign.append((stem, len(missing), len(shallow), sorted(elsewhere)))
        else:
            fillable.append((stem, len(missing), len(shallow)))

    print(f"  reader={args.reader}  k={args.k}")
    print(f"  fillable works: {len(fillable)}")
    tm = sum(m for _, m, _ in fillable)
    ts = sum(s for _, _, s in fillable)
    for s, m, sh in sorted(fillable, key=lambda r: -(r[1] + r[2]))[:12]:
        print(f"     {s[:46]:48s} missing={m:<4} partial={sh}")
    print(f"  -> {tm} missing + {ts} partial = {tm * args.k + ts} calls")

    if foreign:
        print(f"\n  NOT fillable by stem ({len(foreign)}) - rows live under another filename;")
        print("  topping these up would rescore the whole work and duplicate data:")
        for s, m, sh, where in foreign:
            print(f"     {s[:40]:42s} missing={m:<4} partial={sh:<4} lives in {', '.join(where)}")

    if args.out:
        Path(args.out).write_text("\n".join(s for s, _, _ in fillable) + "\n")
        print(f"\n  wrote {args.out} ({len(fillable)} works)")


if __name__ == "__main__":
    main()
