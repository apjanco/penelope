#!/bin/bash
# Score the 2026-09-29 non-Woolf career-arc expansion: Richardson's
# Pilgrimage vols 4-9, Ford's Parade's End vols 1-2, and May Sinclair's
# 1897-1922 arc. Same readers, same
# k, same schema as score_remaining_corpus.sh — the point of the expansion
# is an apples-to-apples comparison with the works already scored, so
# nothing about the scoring setup may drift here.
#
# THIS IS THE CLOUD ARM ONLY (gpt-5.4 + qwen3-max).
#
# The expansion needs three readers, not two: every volume it will be
# compared against (pg03019_pointed_roofs, pg56221_backwater,
# pg56664_honeycomb, pg02775_the_good_soldier) carries gpt-5.4, qwen3-max
# AND llama. Measuring the new works' cross-reader spread over a smaller
# ensemble than the baseline would break the comparison this expansion
# exists to enable.
#
# But llama is NOT scored from here. serve_score_model.slurm runs the vLLM
# server and the scoring client inside the SAME job, deliberately — see its
# header and fill_qwen_gaps.slurm for why that must never be split across
# separately queued jobs. The llama arm is therefore:
#
#     sbatch --job-name=score-llama-expansion --gres=gpu:2 --time=24:00:00 \
#       --export=ALL,LABEL=llama,HFMODEL=meta-llama/Llama-3.3-70B-Instruct,\
# TP=2,MAXLEN=4096,FILES="$(cat expansion_files.txt | tr '\n' ' ')" \
#       technique_v_genre/serve_score_model.slurm
#
# Both arms append to the same scored/<name>.jsonl and are resumable by
# (passage_id, reader_id, run), so they can run concurrently or in any
# order without conflicting.
#
# ~7,657 passages x 2 cloud readers x k=3 -> ~46k rows here, plus ~23k
# more from the llama arm: ~69k total, about 59% on top of the 116k
# already in scored/. Sinclair is over half of that on her own: she
# wrote long, and her eight novels are 4,087 passages to Richardson's and
# Ford's 3,570 combined. Trim SINCLAIR_FILES if that is more than the
# budget allows — the arc still reads with Audrey Craven, The Tree of
# Heaven, Mary Olivier and Harriett Frean alone, which is the 1918 coinage
# straddled at four points instead of eight.
#
# Run from a node with internet access (della-vis2/vis1), detached via
# nohup. Resumable: score.py skips (passage_id, reader_id, run) triples
# already in the output, so killing and rerunning picks up where it left off.

set -uo pipefail  # not -e: one file's failure shouldn't stop the rest

cd "$(dirname "${BASH_SOURCE[0]}")"

# Refuse to score a corpus that hasn't been verified. The whole point of
# verify_corpus.py is to catch a wrong-book download BEFORE it is scored,
# analysed and written up; running it after the fact would defeat that.
echo "$(date '+%Y-%m-%d %H:%M:%S') verifying corpus integrity before scoring..."
if ! /home/aj7878/.conda/envs/penelope/bin/python verify_corpus.py --online; then
  echo "ABORT: corpus verification failed — fix the manifest before scoring." >&2
  exit 1
fi
echo

RICHARDSON_FORD_FILES=(
  pg56447_the_tunnel.jsonl
  pg57395_interim.jsonl
  pg56925_deadlock.jsonl
  pg62967_revolving_lights.jsonl
  pg64585_the_trap.jsonl
  pg77646_oberland.jsonl
  pg64248_some_do_not.jsonl
  pg67622_no_more_parades.jsonl
)

SINCLAIR_FILES=(
  pg29766_audrey_craven.jsonl
  pg17867_the_helpmate.jsonl
  pg28461_the_combined_maze.jsonl
  pg11876_the_three_sisters.jsonl
  pg13883_the_tree_of_heaven.jsonl
  pg09366_mary_olivier_a_life.jsonl
  pg13292_the_romantic.jsonl
  pg09298_life_and_death_of_harriett_frean.jsonl
)

FILES=( "${RICHARDSON_FORD_FILES[@]}" "${SINCLAIR_FILES[@]}" )

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
  rc=$?
  if [ "$rc" -eq 3 ]; then
    echo "ABORT: score.py reported a dead API key (exit 3). Stopping the run" >&2
    echo "       rather than repeating it on every remaining file." >&2
    exit 3
  fi
  echo "$(date '+%Y-%m-%d %H:%M:%S') Done with $f"
  echo
done

echo "$(date '+%Y-%m-%d %H:%M:%S') ALL FILES COMPLETE"
