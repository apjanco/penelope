#!/bin/bash
# ============================================================================
# Penelope technique-vs-genre — generalized vLLM reader server
#
# Parameterized version of ../judge_della.slurm for the local reader
# ensemble (see technique_v_genre/README or the project conversation log
# for the model roster and rationale). One job = one served model.
#
# Usage — gres/constraint MUST be sbatch CLI flags, not --export vars:
# #SBATCH directives are parsed from the literal script text before any
# job env var is set, so they can't be templated at runtime.
#
#   sbatch --gres=gpu:1 --constraint=gpu80 \
#     --export=MODEL="Qwen/Qwen3.6-27B",SERVED_NAME="qwen3-reasoning",TP=1 \
#     --job-name=penelope-qwen3-reasoning \
#     technique_v_genre/serve_vllm.slurm
#
# Writes the endpoint to technique_v_genre/endpoints/<SERVED_NAME>.txt when
# ready — register that URL as a new profile in models.yaml to add this
# reader to score.py's ensemble (same pattern judge_della.slurm already
# established for the GRPO judge).
#
# Required --export vars: MODEL, SERVED_NAME, TP
# Optional: PORT (default 8000), MAX_MODEL_LEN (default 8192)
# ============================================================================

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --qos=gpu-short
#SBATCH --output=technique_v_genre/logs/serve_%x_%j.out
#SBATCH --error=technique_v_genre/logs/serve_%x_%j.err

set -eo pipefail
# NOT set -u: conda's activation scripts reference $PS1, which is unset in a
# non-interactive batch job and would kill the script here with "unbound
# variable" before it ever reaches vLLM. Confirmed by the first submission
# attempt (job 11838353/11934158, failed after 3s at the conda activate line).

: "${MODEL:?Set MODEL, e.g. --export=MODEL=\"Qwen/Qwen3.6-27B\"}"
: "${SERVED_NAME:?Set SERVED_NAME, e.g. --export=SERVED_NAME=qwen3-reasoning}"
: "${TP:?Set TP (tensor-parallel-size), e.g. --export=TP=1}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
# e.g. EXTRA_ARGS="--tokenizer-mode mistral --config-format mistral" for
# Mistral checkpoints — vLLM's "auto" tokenizer-mode was silently falling
# back to the generic HF tokenizer instead of mistral_common's, leaking
# raw BPE glyphs (Ġ/Ċ) into decoded text and breaking every JSON parse
# (job 12590292, 100% error rate — see conversation log 2026-08-20).

# ── Environment ─────────────────────────────────────────────────────────────
module purge
module load anaconda3/2025.12
module load cudatoolkit/13.2

source /usr/licensed/anaconda3/2025.12/etc/profile.d/conda.sh
conda activate penelope

export HF_HOME=/scratch/gpfs/MM4/apjanco/.cache/huggingface
export HF_HUB_OFFLINE=1

# flashinfer JIT fix — see judge_della.slurm for the full explanation
# (module-based CUDA install puts curand.h under targets/x86_64-linux/include,
# not the top-level include/, which breaks flashinfer's JIT sampling kernel).
export VLLM_USE_FLASHINFER_SAMPLER=0
export CPATH="/usr/local/cuda-13.2/targets/x86_64-linux/include${CPATH:+:$CPATH}"

PROJECT_DIR="/scratch/gpfs/MM4/apjanco/penelope"
ENDPOINT_DIR="$PROJECT_DIR/technique_v_genre/endpoints"
ENDPOINT_FILE="$ENDPOINT_DIR/${SERVED_NAME}.txt"

cd "$PROJECT_DIR"
mkdir -p "$ENDPOINT_DIR" technique_v_genre/logs
rm -f "$ENDPOINT_FILE"

echo "Job ID       : $SLURM_JOB_ID"
echo "Node         : $SLURMD_NODENAME  ($(hostname -f))"
echo "GPUs         : $CUDA_VISIBLE_DEVICES"
echo "Model        : $MODEL"
echo "Served name  : $SERVED_NAME"
echo "TP           : $TP"
echo "Max model len: $MAX_MODEL_LEN"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "Starting vLLM server..."
vllm serve "$MODEL" \
    --port "$PORT" \
    --served-model-name "$SERVED_NAME" \
    --tensor-parallel-size "$TP" \
    --max-model-len "$MAX_MODEL_LEN" \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.92 \
    --enable-prefix-caching \
    $EXTRA_ARGS \
    &

VLLM_PID=$!
echo "vLLM PID: $VLLM_PID"

MAX_WAIT=1800   # 30 min — generous for a 70B/TP=2 cold load
WAITED=0
INTERVAL=15

echo "Waiting for server to become ready (max ${MAX_WAIT}s)..."
until curl -sf "http://localhost:$PORT/health" > /dev/null 2>&1; do
    sleep $INTERVAL
    WAITED=$((WAITED + INTERVAL))
    if [ $WAITED -ge $MAX_WAIT ]; then
        echo "ERROR: vLLM did not start within ${MAX_WAIT}s. Exiting."
        kill $VLLM_PID 2>/dev/null || true
        exit 1
    fi
    echo "  waited ${WAITED}s..."
done

echo "Server ready after ${WAITED}s."

ENDPOINT="http://$(hostname -f):$PORT/v1"
echo "$ENDPOINT" > "$ENDPOINT_FILE"
echo "Endpoint written to $ENDPOINT_FILE: $ENDPOINT"

wait $VLLM_PID
