#!/bin/bash
# Score the remaining corpus (everything except Woolf's Lighthouse+Dalloway,
# already in scored/woolf_lighthouse_dalloway.jsonl) with the same two cloud
# readers, same k, same schema. Run from a node with internet access
# (della-vis2/vis1), detached via nohup so it survives disconnection —
# same reasoning as resample_run.sh's screen/tmux instructions, since this
# is CPU/network-bound work with no SLURM allocation needed.
#
# Resumable: score.py itself skips (passage_id, reader_id, run) triples
# already present in the output file, so killing and rerunning this script
# picks up where it left off, both within a file and across files.

set -uo pipefail  # not -e: one file's failure shouldn't stop the rest

cd "$(dirname "${BASH_SOURCE[0]}")"

FILES=(
  as_I_lay_dying.jsonl
  between_the_acts.jsonl
  flush.jsonl
  jacobs_room.jsonl
  night_and_day_cleaned.jsonl
  orlando_cleaned.jsonl
  pg00432_the_ambassadors.jsonl
  pg02775_the_good_soldier.jsonl
  pg02814_dubliners.jsonl
  pg03019_pointed_roofs.jsonl
  pg04217_a_portrait_of_the_artist_as_a_young_man.jsonl
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
  echo "$(date '+%Y-%m-%d %H:%M:%S') Scoring $f -> scored/${name}.jsonl"
  echo "=========================================="
  /home/aj7878/.conda/envs/penelope/bin/python score.py \
      --input "passages/$f" \
      --readers gpt-5.4 qwen3-max \
      --k 3 \
      --concurrency 15 \
      --output "scored/${name}.jsonl"
  echo "$(date '+%Y-%m-%d %H:%M:%S') Done with $f"
  echo
done

echo "$(date '+%Y-%m-%d %H:%M:%S') ALL FILES COMPLETE"
