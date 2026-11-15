#!/bin/bash
# Cloud arm for the three Standard Ebooks titles (third pass, 2026-09-29):
# Faulkner's Soldiers' Pay and Ford's last two Parade's End volumes.
# Chained behind the James arm rather than run alongside it — same two
# rate-limited APIs. llama covers these via SLURM job 14697078.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

echo "$(date '+%Y-%m-%d %H:%M:%S') verifying corpus integrity before scoring..."
if ! /home/aj7878/.conda/envs/penelope/bin/python verify_corpus.py --online; then
  echo "ABORT: corpus verification failed." >&2; exit 1
fi
echo

for name in se_william_faulkner_soldiers_pay \
            se_ford_madox_ford_a_man_could_stand_up \
            se_ford_madox_ford_the_last_post; do
  echo "=========================================="
  echo "$(date '+%Y-%m-%d %H:%M:%S') Scoring $name -> scored/${name}.jsonl"
  echo "=========================================="
  /home/aj7878/.conda/envs/penelope/bin/python score.py \
      --input "passages/${name}.jsonl" --readers gpt-5.4 qwen3-max \
      --k 3 --concurrency 15 --output "scored/${name}.jsonl"
  rc=$?
  if [ "$rc" -eq 3 ]; then
    echo "ABORT: score.py reported a dead API key (exit 3). Stopping the run" >&2
    echo "       rather than repeating it on every remaining file." >&2
    exit 3
  fi
  echo "$(date '+%Y-%m-%d %H:%M:%S') Done with $name"
done
echo "$(date '+%Y-%m-%d %H:%M:%S') STANDARD EBOOKS CLOUD ARM COMPLETE"
