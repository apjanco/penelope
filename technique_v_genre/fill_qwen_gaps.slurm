#!/bin/bash
# ============================================================================
# Penelope technique-vs-genre — fill DashScope-blocked qwen3-max gaps with a
# local open-weight reader.
#
# 11 passages (8 Ulysses, 1 Sound and the Fury, 2 The Ambassadors) permanently
# fail qwen3-max ("qwen3-max" = Alibaba's commercial DashScope API) with
# data_inspection_failed content-moderation errors — retrying does not help,
# it's a persistent content-policy block, not a transient failure. This job
# serves the open-weight Qwen/Qwen3.6-27B checkpoint locally (no external
# moderation) and scores just those 11 passages under it.
#
# Important: this is scored as its own reader, "qwen3.6-local", NOT written
# in as "qwen3-max" — it is a genuinely different model (different weights),
# and mislabeling it would silently mix two models' outputs under one reader
# column, corrupting the k-run within-reader consistency analysis the rest
# of the pipeline depends on.
#
# Server + client run in the SAME job (never split across two separately
# queued jobs — see project history for why).
# ============================================================================

#SBATCH --job-name=penelope-fill-qwen-gaps
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --time=01:00:00
#SBATCH --qos=gpu-short
#SBATCH --output=technique_v_genre/logs/fill_qwen_gaps_%j.out
#SBATCH --error=technique_v_genre/logs/fill_qwen_gaps_%j.err
#SBATCH --mail-type=begin,fail
#SBATCH --mail-user=aj7878@princeton.edu

set -eo pipefail   # NOT -u: conda activation scripts reference unset $PS1 in batch jobs

module purge
module load anaconda3/2025.12
module load cudatoolkit/13.2

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate penelope

export HF_HOME=/scratch/gpfs/MM4/apjanco/.cache/huggingface
export HF_HUB_OFFLINE=1
export VLLM_USE_FLASHINFER_SAMPLER=0
export CPATH="/usr/local/cuda-13.2/targets/x86_64-linux/include${CPATH:+:$CPATH}"

PROJECT_DIR="/scratch/gpfs/MM4/apjanco/penelope"
TV_DIR="$PROJECT_DIR/technique_v_genre"
PORT=8000
MODEL="Qwen/Qwen3.6-27B"
SERVED_NAME="qwen3.6-local"
MAX_MODEL_LEN=32768

cd "$PROJECT_DIR"
mkdir -p "$TV_DIR/logs" "$TV_DIR/endpoints"

echo "Job ID  : $SLURM_JOB_ID"
echo "Node    : $SLURMD_NODENAME  ($(hostname -f))"
echo "GPUs    : $CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "Starting vLLM server ($MODEL as $SERVED_NAME)..."
vllm serve "$MODEL" \
    --port "$PORT" \
    --served-model-name "$SERVED_NAME" \
    --tensor-parallel-size 1 \
    --max-model-len "$MAX_MODEL_LEN" \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.92 \
    --enable-prefix-caching \
    &
VLLM_PID=$!
echo "vLLM PID: $VLLM_PID"

cleanup() { echo "Shutting down vLLM ($VLLM_PID)..."; kill "$VLLM_PID" 2>/dev/null; }
trap cleanup EXIT

MAX_WAIT=900
WAITED=0
INTERVAL=15
echo "Waiting for server to become ready (max ${MAX_WAIT}s)..."
until curl -sf "http://localhost:$PORT/health" > /dev/null 2>&1; do
    sleep $INTERVAL
    WAITED=$((WAITED + INTERVAL))
    if [ $WAITED -ge $MAX_WAIT ]; then
        echo "ERROR: vLLM did not start within ${MAX_WAIT}s. Exiting."
        exit 1
    fi
    echo "  waited ${WAITED}s..."
done
echo "Server ready after ${WAITED}s."

ENDPOINT="http://$(hostname -f):$PORT/v1"
echo "$ENDPOINT" > "$TV_DIR/endpoints/${SERVED_NAME}.txt"
echo "Endpoint: $ENDPOINT"

# Register (or update) this reader in models.yaml so score.py's Config.load()
# can find it by label, following the convention already documented there.
python3 - "$ENDPOINT" "$SERVED_NAME" "$MODEL" <<'PYEOF'
import sys, re
from pathlib import Path

endpoint, served_name, model = sys.argv[1], sys.argv[2], sys.argv[3]
p = Path("models.yaml")
text = p.read_text()
entry = (
    f"  - label: {served_name}\n"
    f"    base_url: {endpoint}   # job {__import__('os').environ.get('SLURM_JOB_ID','?')}, "
    f"fills content-filtered qwen3-max gaps (see technique_v_genre/fill_qwen_gaps.slurm)\n"
    f"    model_name: {served_name}\n"
    f"    api_key: not-needed\n"
)
label_re = re.compile(rf"  - label: {re.escape(served_name)}\n(?:    .*\n)*")
if label_re.search(text):
    text = label_re.sub(entry, text)
else:
    marker = "  # Example: local model via Ollama / vLLM"
    text = text.replace(marker, entry + "\n" + marker) if marker in text else text + "\n" + entry
p.write_text(text)
print(f"models.yaml updated: {served_name} -> {endpoint}")
PYEOF

# ── Score the gap passages, one book at a time, appending to the existing
#    per-book scored/*.jsonl files (same structure gpt-5.4/qwen3-max already
#    share) ────────────────────────────────────────────────────────────────
cd "$TV_DIR"
for pair in \
    "passages/gaps/pg04300_ulysses.jsonl:scored/pg04300_ulysses.jsonl" \
    "passages/gaps/pg75170_sound_and_fury.jsonl:scored/pg75170_sound_and_fury.jsonl" \
    "passages/gaps/pg00432_the_ambassadors.jsonl:scored/pg00432_the_ambassadors.jsonl" \
; do
    input="${pair%%:*}"
    output="${pair##*:}"
    echo "=========================================="
    echo "Scoring $input -> $output (reader: $SERVED_NAME)"
    echo "=========================================="
    python score.py \
        --input "$input" \
        --readers "$SERVED_NAME" \
        --k 3 \
        --concurrency 5 \
        --output "$output"
done

echo "GAP-FILL COMPLETE"
