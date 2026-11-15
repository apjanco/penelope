#!/bin/bash
# Cloud arm for the James 1875-1904 arc (second pass, 2026-09-29).
#
# Separate from score_expansion.sh because that run was already in flight
# when James was added. Deliberately NOT run concurrently with it: both
# hit the same two rate-limited APIs, and doubling in-flight requests
# risks throttling errors in a run that is already hours deep. Chained
# instead — see the launcher that waits on the first run's PID.
#
# llama covers these through the SLURM arm; they are in expansion_files.txt.
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
)

for name in "${FILES[@]}"; do
  echo "=========================================="
  echo "$(date '+%Y-%m-%d %H:%M:%S') Scoring $name -> scored/${name}.jsonl"
  echo "=========================================="
  /home/aj7878/.conda/envs/penelope/bin/python score.py \
      --input "passages/${name}.jsonl" \
      --readers gpt-5.4 qwen3-max \
      --k 3 \
      --concurrency 15 \
      --output "scored/${name}.jsonl"
  rc=$?
  if [ "$rc" -eq 3 ]; then
    echo "ABORT: score.py reported a dead API key (exit 3). Stopping the run" >&2
    echo "       rather than repeating it on every remaining file." >&2
    exit 3
  fi
  echo "$(date '+%Y-%m-%d %H:%M:%S') Done with $name"
done
echo "$(date '+%Y-%m-%d %H:%M:%S') JAMES CLOUD ARM COMPLETE"
