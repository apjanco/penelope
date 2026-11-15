#!/bin/bash
# Submit the llama arm of the 2026-09-29 expansion.
#
# The cloud arm (score_expansion.sh, gpt-5.4 + qwen3-max) runs on
# della-vis2; this is the third reader. It must be a SLURM job rather than
# another client against a served endpoint: serve_score_model.slurm runs
# the vLLM server and the scoring client inside the SAME job by design, and
# its header plus fill_qwen_gaps.slurm both say never to split those across
# separately queued jobs.
#
# Both arms append to the same scored/<name>.jsonl and are resumable by
# (passage_id, reader_id, run), so order and overlap do not matter.
#
# Prerequisite: meta-llama/Llama-3.3-70B-Instruct in the HF cache. Compute
# nodes are offline (HF_HUB_OFFLINE=1), so the weights must already be
# there — the cache was cleared some time after the August run, hence
# download_with_retry.sh. Verified below rather than discovered 30 minutes
# into a GPU allocation.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

HF_CACHE=/scratch/gpfs/MM4/apjanco/.cache/huggingface
MODEL_DIR="$HF_CACHE/hub/models--meta-llama--Llama-3.3-70B-Instruct"
EXPECTED_SHARDS=30

n=$(find "$MODEL_DIR" -name "*.safetensors" 2>/dev/null | wc -l)
if [ "$n" -lt "$EXPECTED_SHARDS" ]; then
    echo "ABORT: only $n/$EXPECTED_SHARDS safetensors shards in the HF cache." >&2
    echo "Run: ./download_with_retry.sh meta-llama/Llama-3.3-70B-Instruct --exclude 'original/*'" >&2
    exit 1
fi
echo "HF cache OK: $n safetensors shards"

FILES="$(tr '\n' ' ' < expansion_files.txt)"
echo "Files: $(wc -w <<< "$FILES") works"

cd /scratch/gpfs/MM4/apjanco/penelope
sbatch --job-name=score-llama-expansion --gres=gpu:2 --time=24:00:00 \
  --export=ALL,LABEL=llama,HFMODEL=meta-llama/Llama-3.3-70B-Instruct,TP=2,MAXLEN=4096,FILES="$FILES" \
  technique_v_genre/serve_score_model.slurm
