#!/bin/bash
# Score the eight coauthor-supplied works with Jev, k=1.
#
# Same reader, same protocol and same k as the 46 works already in
# scored_jev/, which is the point: Jev is the only reader with 100% coverage
# of the existing corpus, so works it scores land on a scale that is already
# calibrated against gpt-5.4 (per-work rank correlation 0.942) and qwen3-max
# (0.871). A locally-served model would have needed that calibration built
# from scratch.
#
# Seven of these eight are in US copyright. Scoring them here sends passage
# text to TypeSafe. That is the user's decision, taken knowingly.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

echo "$(date '+%Y-%m-%d %H:%M:%S') verifying corpus integrity before scoring..."
if ! /home/aj7878/.conda/envs/penelope/bin/python verify_corpus.py --online; then
  echo "ABORT: corpus verification failed." >&2; exit 1
fi
echo

mapfile -t FILES < jev_new_works.txt
for name in "${FILES[@]}"; do
  [ -n "$name" ] || continue
  echo "$(date '+%Y-%m-%d %H:%M:%S') jev: $name"
  /home/aj7878/.conda/envs/penelope/bin/python calibration/score_jev.py \
      --input "passages/${name}.jsonl" --k 1 --concurrency 6 \
      --out "scored_jev/${name}.jsonl"
  rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "ABORT: score_jev.py exited $rc on $name — stopping." >&2; exit "$rc"
  fi
done
echo "$(date '+%Y-%m-%d %H:%M:%S') JEV NEW WORKS COMPLETE"
