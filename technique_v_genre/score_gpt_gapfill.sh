#!/bin/bash
# Fill gpt-5.4's remaining coverage (credits restored 2026-10-01).
#
# Work list comes from find_reader_gaps.py, which excludes works whose rows
# live under another filename — To the Lighthouse is in
# woolf_lighthouse_dalloway.jsonl, and pointing a top-up job at
# scored/to_the_lighthouse_cleaned.jsonl rescores all 455 passages instead of
# the 4 that are short. That mistake cost the llama job its walltime.
#
# ~6,405 missing + 90 partial = 19,305 calls. score.py now aborts with exit 3
# after 25 consecutive quota/auth failures, so a second credit exhaustion
# stops in seconds instead of grinding through 50k doomed retries, and the
# loop below stops with it rather than repeating on every remaining file.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

echo "$(date '+%Y-%m-%d %H:%M:%S') verifying corpus integrity before scoring..."
if ! /home/aj7878/.conda/envs/penelope/bin/python verify_corpus.py --online; then
  echo "ABORT: corpus verification failed." >&2; exit 1
fi
echo

mapfile -t FILES < gpt_gap_files.txt

for name in "${FILES[@]}"; do
  [ -n "$name" ] || continue
  echo "=========================================="
  echo "$(date '+%Y-%m-%d %H:%M:%S') gpt-5.4: $name"
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
echo "$(date '+%Y-%m-%d %H:%M:%S') GPT GAPFILL COMPLETE"
