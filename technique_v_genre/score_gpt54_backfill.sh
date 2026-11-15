#!/bin/bash
# gpt-5.4 backfill after OpenAI quota was restored on 2026-08-07 (credits added).
# score.py resumes by (passage_id, reader_id, run) key, so this only does the
# gap rows documented in logs/gpt54_backfill_TODO.txt (~16,575 calls) and is
# safe to re-run if interrupted.

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
)

mkdir -p scored

for f in "${FILES[@]}"; do
  name="${f%.jsonl}"
  echo "=========================================="
  echo "$(date '+%Y-%m-%d %H:%M:%S') gpt-5.4 backfill: $f -> scored/${name}.jsonl"
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


# the_waves_cleaned.jsonl and the_years.jsonl are excluded above because the
# concurrent score_remaining_qwen_only.sh job may still be writing to those
# same output files (single shared append-mode file handle, no cross-process
# lock in score.py -> concurrent writers risk row corruption, as happened
# once already this session). Wait for that script's driver process to exit
# before touching them.
QWEN_SCRIPT_PID=309848
if kill -0 "$QWEN_SCRIPT_PID" 2>/dev/null; then
  echo "$(date '+%Y-%m-%d %H:%M:%S') Waiting for score_remaining_qwen_only.sh (pid $QWEN_SCRIPT_PID) to finish before touching the_waves_cleaned/the_years..."
  while kill -0 "$QWEN_SCRIPT_PID" 2>/dev/null; do
    sleep 30
  done
  echo "$(date '+%Y-%m-%d %H:%M:%S') qwen job finished, proceeding."
fi

for f in the_waves_cleaned.jsonl the_years.jsonl; do
  name="${f%.jsonl}"
  echo "=========================================="
  echo "$(date '+%Y-%m-%d %H:%M:%S') gpt-5.4 backfill: $f -> scored/${name}.jsonl"
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

echo "$(date '+%Y-%m-%d %H:%M:%S') GPT-5.4 BACKFILL COMPLETE"
