"""General k-resampled scorer for the technique-vs-genre schema.

Supersedes score_experiment_d.py's single-pass, one-off validation scorer.
Resampling (k>=3 runs per passage per reader) is the default unit of
evidence here, per AMBIGUITY.md's existing methodology: within-model
consistency (mode label across k runs, fraction matching) is what
distinguishes a stable, defensible reading from a model just being noisy
on that passage. A single-pass call is a preliminary read, not a finding.

Usage:
    python score.py --input passages/to_the_lighthouse_cleaned.jsonl \\
        --readers gpt-5.4 qwen3-max --k 3 --output scored/lighthouse.jsonl

Resumable by (passage_id, reader_id, run) key, same pattern as
score_experiment_d.py and this project's existing resample.py.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, "/scratch/gpfs/MM4/apjanco/penelope")

from scripts.config import Config
from openai import OpenAI

# A dead API key is not a transient failure, but score.py could not tell the
# difference: on 2026-09-30 the gpt-5.4 key exhausted its credit mid-run and
# the next stage spent 49,932 attempts — three retries with backoff per call —
# re-asking a key that had already answered "no credits remaining" thousands
# of times. It "completed" every file and wrote nothing. These markers trip a
# process-wide breaker so the run stops and says why.
FATAL_MARKERS = (
    # credit / quota exhaustion
    "no credits remaining", "insufficient_quota", "exceeded your current quota",
    "quota_exceeded", "credit_balance_exhausted",
    # configured spend caps — these surface differently from running out of
    # credit, and a budget cap that reads "hard limit reached" would not have
    # matched the original marker list.
    "billing", "billing_hard_limit_reached", "hard limit", "spend limit",
    "budget", "monthly limit", "usage limit",
    # credential problems
    "invalid_api_key", "account is not active", "access_terminated",
    "account deactivated", "permission_denied",
)
FATAL_STREAK = 25   # consecutive fatal-looking failures before aborting


class DeadKeyError(RuntimeError):
    """Raised when a reader's credentials or quota are conclusively gone."""


from prompts import SYSTEM_PROMPT

ROOT = Path(__file__).resolve().parent
DEFAULT_READERS = {"gpt-5.4", "qwen3-max"}  # mercury-2 excluded per project direction


_fatal_streak = {}


def _note_failure(label: str, msg: str) -> None:
    """Track consecutive unrecoverable failures per reader."""
    low = msg.lower()
    if any(m in low for m in FATAL_MARKERS):
        _fatal_streak[label] = _fatal_streak.get(label, 0) + 1
        if _fatal_streak[label] >= FATAL_STREAK:
            raise DeadKeyError(
                f"{label}: {_fatal_streak[label]} consecutive unrecoverable API "
                f"failures — aborting rather than burning retries.\n  last error: {msg[:200]}")
    else:
        _fatal_streak[label] = 0


def _note_success(label: str) -> None:
    _fatal_streak[label] = 0


def score_one(profile, passage_text: str, source_file: str) -> dict | None:
    user_content = (
        f"Source: {source_file}\n\n"
        "Score the following passage per the schema. If it renders no "
        "consciousness at all, return {\"instances\": []}. Respond with "
        "JSON only.\n\n"
        f"{passage_text}"
    )
    client = OpenAI(base_url=profile.base_url, api_key=profile.api_key)
    last_error = "unknown error"
    kwargs = dict(
        model=profile.model_name,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
    )
    if profile.temperature is not None:
        kwargs["temperature"] = profile.temperature
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(**kwargs)
            raw = resp.choices[0].message.content or ""
            # Reasoning models (e.g. Qwen3.6's thinking mode) sometimes emit
            # only the CLOSING </think> tag — the opening <think> is added by
            # the chat template as a prompt prefix, not part of the returned
            # completion text — so a <think>.*?</think> strip matches nothing.
            # Worse, the reasoning prose itself often contains inline draft
            # JSON snippets (the model talking through the schema before
            # committing), so a naive "first { in the text" grab latches onto
            # one of those instead of the real answer. Anchor on the LAST
            # </think> instead: everything after it is the model's actual
            # output, everything before is safe to discard regardless of
            # whether an opening tag is present.
            if "</think>" in raw:
                cleaned = raw.rsplit("</think>", 1)[1]
            else:
                cleaned = raw
            cleaned = re.sub(r"```(?:json)?", "", cleaned).strip()
            start = cleaned.find("{")
            data = json.loads(cleaned[start:] if start >= 0 else cleaned)
            instances = data.get("instances") or []
            _note_success(profile.label)
            if not instances:
                return {"is_soc": False}
            item = instances[0]
            return {
                "is_soc": item.get("is_soc", True),
                "mediation_level": item.get("mediation_level"),
                "soc_type": item.get("soc_type"),
                "secondary_devices": item.get("secondary_devices") or [],
                "fit_category": item.get("fit_category"),
                "fit_alternatives": item.get("fit_alternatives") or [],
                "antimimetic": item.get("antimimetic", False),
                "antimimetic_defense": item.get("antimimetic_defense", ""),
                "narrator_position": item.get("narrator_position"),
                "character_pov": item.get("character_pov", ""),
                "confidence": item.get("confidence"),
                "argument": item.get("argument", ""),
                "alternative_reasoning": item.get("alternative_reasoning", ""),
                "evidence": item.get("evidence") or [],
            }
        except Exception as exc:
            msg = str(exc)
            print(f"    [{profile.label}] attempt {attempt + 1} failed: {msg}")
            # Trips DeadKeyError once the same unrecoverable condition repeats;
            # a transient failure resets the streak instead.
            _note_failure(profile.label, msg)
            last_error = msg
            time.sleep(3)
    return {"_error_message": last_error}


def _as_record(result: dict | None) -> dict:
    """Error rows used to store `error: True` and nothing else, which made a
    dead API key indistinguishable from a parse failure when diagnosing a run
    afterwards. Keep the message."""
    if result is None:
        return {"error": True, "error_message": "no result"}
    if "_error_message" in result:
        return {"error": True, "error_message": result["_error_message"][:300]}
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="Passages JSONL (from segment.py)")
    ap.add_argument("--output", required=True, help="Output JSONL path")
    ap.add_argument("--readers", nargs="*", default=None, help="Reader labels from models.yaml (default: gpt-5.4 qwen3-max)")
    ap.add_argument("--k", type=int, default=3, help="Resamples per (passage, reader) (default 3)")
    ap.add_argument("--limit", type=int, default=None, help="Only score the first N passages (for smoke tests)")
    ap.add_argument(
        "--concurrency", type=int, default=1,
        help="Concurrent requests (default 1 = serial). ~13s/call serially; "
        "each provider is a separate endpoint so concurrency mostly helps.",
    )
    args = ap.parse_args()

    wanted_readers = set(args.readers) if args.readers else DEFAULT_READERS
    cfg = Config.load()
    profiles = {p.label: p for p in cfg.models if p.label in wanted_readers}
    missing = wanted_readers - set(profiles)
    if missing:
        sys.exit(f"Missing model profiles in models.yaml: {missing}")
    print("Scoring with:", list(profiles), f"k={args.k}, concurrency={args.concurrency}")

    in_path = Path(args.input)
    rows = [json.loads(l) for l in in_path.read_text().splitlines()]
    if args.limit:
        rows = rows[: args.limit]
    total_calls = len(rows) * len(profiles) * args.k
    print(f"{len(rows)} passages x {len(profiles)} readers x k={args.k} = {total_calls} calls\n")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done_keys = set()
    failed_keys = set()
    mode = "a"
    if out_path.exists():
        for l in out_path.read_text().splitlines():
            if not l.strip():
                continue
            r = json.loads(l)
            key = (r["passage_id"], r["reader_id"], r["run"])
            # A recorded failure is NOT a completed call. Counting error rows
            # as done made resume permanently skip them: when the gpt-5.4 key
            # ran out of credit mid-run on 2026-09-29, 29,495 calls failed and
            # a plain rerun would have reported "nothing left to do" while five
            # works sat unscored by that reader. Errors are retried; only real
            # results are skipped.
            if "error" in r:
                failed_keys.add(key)
            else:
                done_keys.add(key)
        failed_keys -= done_keys  # a later success supersedes an earlier failure
        print(f"Resuming: {len(done_keys)} triples already done.")
        if failed_keys:
            print(f"{len(failed_keys)} previously-failed triples will be retried.")
    else:
        mode = "w"

    # Flatten to individual (row, reader_label, run) tasks so concurrency
    # isn't bottlenecked by iterating passage-by-passage.
    tasks = [
        (row, label, run)
        for row in rows
        for label in profiles
        for run in range(args.k)
        if (row["passage_id"], label, run) not in done_keys
    ]
    print(f"{len(tasks)} calls remaining after resume-skip.\n")

    write_lock = threading.Lock()
    done_count = 0

    def process(task):
        row, label, run = task
        result = score_one(profiles[label], row["passage_text"], row["source_file"])
        return row, label, run, result

    with out_path.open(mode) as fh:

        def write_record(row, label, run, result):
            record = {
                "passage_id": row["passage_id"],
                "source_file": row["source_file"],
                "author": row["author"],
                "title": row["title"],
                "year": row["year"],
                "role": row["role"],
                "reader_id": label,
                "run": run,
                "passage_text": row["passage_text"],
                **_as_record(result),
            }
            with write_lock:
                fh.write(json.dumps(record) + "\n")
                fh.flush()

        try:
            if args.concurrency <= 1:
                for i, task in enumerate(tasks):
                    row, label, run, result = process(task)
                    write_record(row, label, run, result)
                    if (i + 1) % 20 == 0:
                        print(f"  {i + 1}/{len(tasks)} calls done")
            else:
                with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                    futures = [pool.submit(process, t) for t in tasks]
                    for future in as_completed(futures):
                        row, label, run, result = future.result()
                        write_record(row, label, run, result)
                        done_count += 1
                        if done_count % 20 == 0:
                            print(f"  {done_count}/{len(tasks)} calls done")
        except DeadKeyError as e:
            print(f"\nABORTING: {e}", file=sys.stderr)
            print(f"Partial results kept in {out_path}; rerun to resume once fixed.",
                  file=sys.stderr)
            fh.flush()
            sys.exit(3)

    print("\nWritten to", out_path)


if __name__ == "__main__":
    main()
