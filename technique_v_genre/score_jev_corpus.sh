#!/bin/bash
# Score the whole corpus with Jev.
#
# k=1 deliberately. Every other reader needs resampling because a hard label
# carries no uncertainty; Jev returns a probability distribution per question,
# so the ambiguity is in the answer itself. Its k-run variance on the
# calibration set was 5%, so k=3 would triple the bill to re-derive what the
# probabilities already state. This is also the measure AMBIGUITY.md wants and
# cross-reader disagreement only proxies.
#
# Output goes to scored_jev/, NOT scored/. The gpt-5.4 gapfill is appending to
# scored/<work>.jsonl right now, and score.py's write lock is per-process — two
# processes appending to one file can interleave partial lines. Separate
# directory removes the hazard entirely, and these rows carry extra fields
# (per-category probabilities, confidence) that the others do not.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

echo "$(date '+%Y-%m-%d %H:%M:%S') verifying corpus integrity before scoring..."
if ! /home/aj7878/.conda/envs/penelope/bin/python verify_corpus.py --online; then
  echo "ABORT: corpus verification failed." >&2; exit 1
fi
echo

python3 - <<'PYEOF' > /tmp/jev_worklist.txt
import json
from pathlib import Path
man = json.loads(Path("corpus_manifest.json").read_text())["corpus"]
for e in man:
    stem = e["source_file"][:-4]
    if (Path("passages") / f"{stem}.jsonl").exists():
        print(stem)
PYEOF
mapfile -t FILES < /tmp/jev_worklist.txt
echo "  ${#FILES[@]} works to score"

for name in "${FILES[@]}"; do
  [ -n "$name" ] || continue
  echo "$(date '+%Y-%m-%d %H:%M:%S') jev: $name"
  /home/aj7878/.conda/envs/penelope/bin/python calibration/score_jev.py \
      --input "passages/${name}.jsonl" --k 1 --concurrency 6 \
      --out "scored_jev/${name}.jsonl"
  rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "ABORT: score_jev.py exited $rc on $name — stopping." >&2
    exit "$rc"
  fi
done
echo "$(date '+%Y-%m-%d %H:%M:%S') JEV CORPUS COMPLETE"
