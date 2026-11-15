#!/bin/bash
# Refill gpt-5.4 after the 2026-09-29 credit exhaustion (credits restored
# 2026-09-30). Two stages, split by which files another process owns.
#
# FILE OWNERSHIP IS THE CONSTRAINT, not rate limits: gpt-5.4 (OpenAI) and
# qwen3-max (DashScope) are separate providers and do not contend. But both
# append to the same scored/<work>.jsonl, and score.py's write lock is
# per-process — two processes appending to one file can interleave partial
# lines. So stage 1 runs only on files the qwen catch-up is NOT touching,
# and stage 2 waits for it.
#
# Resumable and retry-correct: since the 2026-09-29 patch, score.py retries
# recorded failures instead of counting them as done, so the ~9.8k rows that
# failed on the credit error will be re-attempted rather than skipped.
#
# Usage:  ./score_gpt_refill.sh sinclair     # stage 1, safe to run now
#         ./score_gpt_refill.sh james_se     # stage 2, only after qwen exits
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

STAGE="${1:?usage: $0 sinclair|james_se}"

SINCLAIR=(
  pg17867_the_helpmate
  pg28461_the_combined_maze
  pg11876_the_three_sisters
  pg13883_the_tree_of_heaven
  pg09366_mary_olivier_a_life
  pg13292_the_romantic
  pg09298_life_and_death_of_harriett_frean
)
JAMES_SE=(
  pg00176_roderick_hudson
  pg02833_the_portrait_of_a_lady_vol_1
  pg02834_the_portrait_of_a_lady_vol_2
  pg07118_what_maisie_knew
  pg04264_the_golden_bowl
  se_william_faulkner_soldiers_pay
  se_ford_madox_ford_a_man_could_stand_up
  se_ford_madox_ford_the_last_post
)

case "$STAGE" in
  sinclair) FILES=( "${SINCLAIR[@]}" ) ;;
  james_se) FILES=( "${JAMES_SE[@]}" ) ;;
  *) echo "unknown stage: $STAGE" >&2; exit 2 ;;
esac

echo "$(date '+%Y-%m-%d %H:%M:%S') verifying corpus integrity before scoring..."
if ! /home/aj7878/.conda/envs/penelope/bin/python verify_corpus.py --online; then
  echo "ABORT: corpus verification failed." >&2; exit 1
fi
echo

for name in "${FILES[@]}"; do
  echo "=========================================="
  echo "$(date '+%Y-%m-%d %H:%M:%S') gpt-5.4 [$STAGE]: $name"
  echo "=========================================="
  /home/aj7878/.conda/envs/penelope/bin/python score.py \
      --input "passages/${name}.jsonl" --readers gpt-5.4 \
      --k 3 --concurrency 15 --output "scored/${name}.jsonl"
  rc=$?
  if [ "$rc" -eq 3 ]; then
    echo "ABORT: score.py reported a dead API key (exit 3). Stopping the run" >&2
    echo "       rather than repeating it on every remaining file." >&2
    exit 3
  fi
  echo "$(date '+%Y-%m-%d %H:%M:%S') Done with $name"
done
echo "$(date '+%Y-%m-%d %H:%M:%S') GPT REFILL COMPLETE [$STAGE]"
