#!/bin/bash
# gpt-5.4 backfill round 2, after OpenAI credits were topped up again on
# 2026-08-07 afternoon (round 1 ran out mid-Sound-and-Fury; see
# logs/gpt54_backfill_TODO.txt for the full history). Ulysses, both Wings of
# the Dove volumes, Backwater, and Honeycomb are already complete and are not
# included here. score.py resumes by (passage_id, reader_id, run) key, so
# this is safe to re-run if credits run out again.

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

FILES=(
  pg75170_sound_and_fury.jsonl
  the_voyage_out.jsonl
  the_waves_cleaned.jsonl
  the_years.jsonl
)

mkdir -p scored

for f in "${FILES[@]}"; do
  name="${f%.jsonl}"
  echo "=========================================="
  echo "$(date '+%Y-%m-%d %H:%M:%S') gpt-5.4 backfill round 2: $f -> scored/${name}.jsonl"
  echo "=========================================="
  /home/aj7878/.conda/envs/penelope/bin/python score.py \
      --input "passages/$f" \
      --readers gpt-5.4 \
      --k 3 \
      --concurrency 15 \
      --output "scored/${name}.jsonl"
  echo "$(date '+%Y-%m-%d %H:%M:%S') Done with $f"
  echo
done

echo "$(date '+%Y-%m-%d %H:%M:%S') GPT-5.4 BACKFILL ROUND 2 COMPLETE"
