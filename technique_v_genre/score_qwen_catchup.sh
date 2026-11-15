#!/bin/bash
# Fill qwen3-max's gap on the James and Standard Ebooks works.
#
# That gap is not a qwen failure. gpt-5.4's key ran out of credit on
# 2026-09-29 mid-run, and because both cloud readers ride in one score.py
# invocation, killing the arms to stop gpt burning 429s stopped qwen too.
# qwen was probed on 2026-09-30 and still works, so this is unblocked work
# that does not wait on the OpenAI billing fix.
#
# qwen-only deliberately: including gpt-5.4 here would spend three retries
# per call on a key that is still refusing, turning a ~2h run into a much
# longer one that produces nothing extra. gpt-5.4 gets its own pass once
# credits are restored; score.py resumes per (passage_id, reader_id, run)
# and — since the 2026-09-29 patch — retries recorded failures rather than
# skipping them, so the two passes converge on the same result.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

echo "$(date '+%Y-%m-%d %H:%M:%S') verifying corpus integrity before scoring..."
if ! /home/aj7878/.conda/envs/penelope/bin/python verify_corpus.py --online; then
  echo "ABORT: corpus verification failed." >&2; exit 1
fi
echo

FILES=(
  pg00176_roderick_hudson
  pg02833_the_portrait_of_a_lady_vol_1
  pg02834_the_portrait_of_a_lady_vol_2
  pg07118_what_maisie_knew
  pg04264_the_golden_bowl
  se_william_faulkner_soldiers_pay
  se_ford_madox_ford_a_man_could_stand_up
  se_ford_madox_ford_the_last_post
)

for name in "${FILES[@]}"; do
  echo "=========================================="
  echo "$(date '+%Y-%m-%d %H:%M:%S') qwen3-max: $name"
  echo "=========================================="
  /home/aj7878/.conda/envs/penelope/bin/python score.py \
      --input "passages/${name}.jsonl" --readers qwen3-max \
      --k 3 --concurrency 15 --output "scored/${name}.jsonl"
  rc=$?
  if [ "$rc" -eq 3 ]; then
    echo "ABORT: score.py reported a dead API key (exit 3). Stopping the run" >&2
    echo "       rather than repeating it on every remaining file." >&2
    exit 3
  fi
  echo "$(date '+%Y-%m-%d %H:%M:%S') Done with $name"
done
echo "$(date '+%Y-%m-%d %H:%M:%S') QWEN CATCH-UP COMPLETE"
