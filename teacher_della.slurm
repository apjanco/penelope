#!/bin/bash
# ============================================================================
# Penelope SoC — teacher-gold relabeling job (TEACHER_GOLD.md Stage A)
#
# Serves Qwen3.6-27B on 1×A100 80GB via vLLM, then runs scripts/teacher_gold.py
# against it to produce passage-grounded, differentiated gold in
# training_data/teacher.json. Server + client run in the SAME job; the server is
# shut down when the client finishes.
#
# Smoke test first (32 chunks), then full run:
#   LIMIT=32 sbatch teacher_della.slurm      # smoke — inspect training_data/teacher.json
#   sbatch teacher_della.slurm               # full run (resumes from checkpoint)
#
# Requests run CONCURRENCY-wide against the server (default 4 — sized to fit
# MAX_MODEL_LEN=32768 within the A100's ~189K-token KV cache pool at
# gpu-memory-utilization=0.92; raising MAX_MODEL_LEN further should lower
# CONCURRENCY roughly proportionally, e.g. CONCURRENCY*MAX_MODEL_LEN <~ 180000).
# Override with e.g. `CONCURRENCY=2 MAX_MODEL_LEN=65536 sbatch teacher_della.slurm`.
#
# Monitor:
#   tail -f logs/teacher_<JOBID>.out
#   cat teacher_endpoint.txt                 # appears when server is ready
# ============================================================================

#SBATCH --job-name=penelope-teacher
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80         # 1×A100 80GB for Qwen3.6-27B in BF16
#SBATCH --time=24:00:00
#SBATCH --qos=gpu-short
#SBATCH --output=logs/teacher_%j.out
#SBATCH --error=logs/teacher_%j.err
#SBATCH --mail-type=begin,fail
#SBATCH --mail-user=aj7878@princeton.edu

# ── Environment ─────────────────────────────────────────────────────────────
module purge
module load anaconda3/2025.12
module load cudatoolkit/13.2

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate penelope

export HF_HOME=/scratch/gpfs/MM4/apjanco/.cache/huggingface
export HF_HUB_OFFLINE=1

# ── flashinfer JIT fix (same as judge_della.slurm) ──────────────────────────
export VLLM_USE_FLASHINFER_SAMPLER=0
export CPATH="/usr/local/cuda-13.2/targets/x86_64-linux/include${CPATH:+:$CPATH}"

PROJECT_DIR="/scratch/gpfs/MM4/apjanco/penelope"
ENDPOINT_FILE="$PROJECT_DIR/teacher_endpoint.txt"
PORT=8000
MODEL="Qwen/Qwen3.6-27B"

# Client knobs (override at submit time, e.g. `LIMIT=32 sbatch teacher_della.slurm`)
INPUT="${INPUT:-chunking_positives/}"
OUTPUT="${OUTPUT:-training_data/teacher.json}"
TEMPERATURE="${TEMPERATURE:-0.2}"
MAX_TOKENS="${MAX_TOKENS:-16384}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
RETRIES="${RETRIES:-2}"
CONCURRENCY="${CONCURRENCY:-4}"

cd "$PROJECT_DIR" || { echo "Project dir not found: $PROJECT_DIR"; exit 1; }
mkdir -p logs

rm -f "$ENDPOINT_FILE"

echo "Job ID  : $SLURM_JOB_ID"
echo "Node    : $SLURMD_NODENAME  ($(hostname -f))"
echo "GPUs    : $CUDA_VISIBLE_DEVICES"
echo "Model   : $MODEL"
echo "Input   : $INPUT"
echo "Output  : $OUTPUT"
echo "Limit   : ${LIMIT:-<none, full run>}"
echo "Concur. : $CONCURRENCY"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# ── Start vLLM server ────────────────────────────────────────────────────────
echo "Starting vLLM server..."
vllm serve "$MODEL" \
    --port "$PORT" \
    --served-model-name Qwen/Qwen3.6-27B \
    --tensor-parallel-size 1 \
    --max-model-len "$MAX_MODEL_LEN" \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.92 \
    --enable-prefix-caching \
    &
VLLM_PID=$!
echo "vLLM PID: $VLLM_PID"

# Always tear down the server on exit (normal, error, or cancellation).
cleanup() { echo "Shutting down vLLM ($VLLM_PID)..."; kill "$VLLM_PID" 2>/dev/null; }
trap cleanup EXIT

# ── Wait for server readiness ────────────────────────────────────────────────
MAX_WAIT=600
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

# ── Publish endpoint (for reference / external clients) ──────────────────────
ENDPOINT="http://$(hostname -f):$PORT/v1"
echo "$ENDPOINT" > "$ENDPOINT_FILE"
echo "Endpoint written to $ENDPOINT_FILE: $ENDPOINT"

# ── Run the teacher-gold client ──────────────────────────────────────────────
LIMIT_ARG=""
if [ -n "${LIMIT:-}" ]; then
    LIMIT_ARG="--limit $LIMIT"
fi

VERBOSE_ARG=""
if [ -n "${VERBOSE:-}" ]; then
    VERBOSE_ARG="--verbose"
fi

echo "Running teacher_gold.py..."
python scripts/teacher_gold.py \
    --input "$INPUT" \
    --output "$OUTPUT" \
    --model "$MODEL" \
    --base-url "http://localhost:$PORT/v1" \
    --temperature "$TEMPERATURE" \
    --max-tokens "$MAX_TOKENS" \
    --max-model-len "$MAX_MODEL_LEN" \
    --retries "$RETRIES" \
    --concurrency "$CONCURRENCY" \
    --resume \
    $LIMIT_ARG \
    $VERBOSE_ARG

STATUS=$?
echo "teacher_gold.py exited with status $STATUS"
exit $STATUS
