#!/bin/bash
# qwen3-max-only continuation while OpenAI credits are exhausted.
# --readers qwen3-max means gpt-5.4 is never called (no wasted quota-blocked
# attempts), and score.py's normal resume logic skips whatever qwen3-max
# work is already complete in each file, so this is safe to run against
# files that are already fully done for qwen3-max (Ulysses, Wings vol 1/2,
# Backwater, Honeycomb, Sound and Fury, Voyage Out all pass through as
# near-instant no-ops) as well as the genuinely incomplete ones (The Waves,
# The Years). gpt-5.4 backfill for all affected files is a separate,
# explicit follow-up once credits are restored — see logs/gpt54_backfill_TODO.txt.

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

FILES=(
  pg04300_ulysses.jsonl
  pg29452_the_wings_of_the_dove_volume_1.jsonl
  pg30059_the_wings_of_the_dove_volume_2.jsonl
  pg56221_backwater.jsonl
  pg56664_honeycomb.jsonl
  pg75170_sound_and_fury.jsonl
  the_voyage_out.jsonl
  the_waves_cleaned.jsonl
  the_years.jsonl
)

mkdir -p scored

for f in "${FILES[@]}"; do
  name="${f%.jsonl}"
  echo "=========================================="
  echo "$(date '+%Y-%m-%d %H:%M:%S') qwen3-max only: $f -> scored/${name}.jsonl"
  echo "=========================================="
  /home/aj7878/.conda/envs/penelope/bin/python score.py \
      --input "passages/$f" \
      --readers qwen3-max \
      --k 3 \
      --concurrency 15 \
      --output "scored/${name}.jsonl"
  echo "$(date '+%Y-%m-%d %H:%M:%S') Done with $f"
  echo
done

echo "$(date '+%Y-%m-%d %H:%M:%S') QWEN3-MAX PASS COMPLETE"
