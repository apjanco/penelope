#!/bin/bash
# ============================================================================
# Penelope SoC — GRPO smoke test on Princeton Della
#
# Minimal end-to-end test of the production GRPO path:
# - starts the local vLLM judge on GPU 0
# - runs GRPO training on GPU 1
# - limits work to a tiny slice so you can confirm wiring before a full run
#
# Submit:
#   sbatch test_train_grpo.slurm
#
# Monitor:
#   squeue -u $USER
#   tail -f logs/test_grpo_<JOBID>.out
# ============================================================================

#SBATCH --job-name=penelope-grpo-test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gres=gpu:2
#SBATCH --constraint=gpu80
#SBATCH --time=00:45:00
#SBATCH --qos=gpu-short
#SBATCH --output=logs/test_grpo_%j.out
#SBATCH --error=logs/test_grpo_%j.err
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=

module purge
module load anaconda3/2025.12
module load cudatoolkit/13.2

conda activate penelope

export HF_HOME=/scratch/gpfs/MM4/apjanco/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_USE_FLASHINFER_SAMPLER=0
export CPATH="/usr/local/cuda-13.2/targets/x86_64-linux/include${CPATH:+:$CPATH}"

PROJECT_DIR="/scratch/gpfs/MM4/apjanco/penelope"
ENDPOINT_FILE="$PROJECT_DIR/judge_endpoint.txt"
PORT=8000
TEST_OUTPUT_DIR="$PROJECT_DIR/adapter-v3-grpo-smoketest"

cd "$PROJECT_DIR" || { echo "Project dir not found: $PROJECT_DIR"; exit 1; }
mkdir -p logs
rm -f "$ENDPOINT_FILE"
rm -rf "$TEST_OUTPUT_DIR"

echo "Job ID     : $SLURM_JOB_ID"
echo "Node       : $SLURMD_NODENAME"
echo "GPUs       : $CUDA_VISIBLE_DEVICES"
echo "Working dir: $PWD"
echo "Test output: $TEST_OUTPUT_DIR"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "import torch; print('PyTorch', torch.__version__, '| CUDA', torch.version.cuda)"

JUDGE_MODEL=$(find "$HF_HOME/hub/models--Qwen--Qwen3.6-27B/snapshots" \
    -maxdepth 1 -mindepth 1 -type d 2>/dev/null | head -1)
if [ -z "$JUDGE_MODEL" ]; then
    echo "ERROR: Qwen3.6-27B not found in HF cache."
    exit 1
fi
echo "Judge model: $JUDGE_MODEL"

CUDA_VISIBLE_DEVICES=0 vllm serve "$JUDGE_MODEL" \
    --port "$PORT" \
    --served-model-name Qwen/Qwen3.6-27B \
    --tensor-parallel-size 1 \
    --max-model-len 8192 \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.92 \
    --enable-prefix-caching \
    &
VLLM_PID=$!
echo "vLLM PID: $VLLM_PID (GPU 0)"

MAX_WAIT=600
WAITED=0
INTERVAL=15
echo "Waiting for judge server (max ${MAX_WAIT}s)..."
until curl -sf "http://localhost:$PORT/health" > /dev/null 2>&1; do
    sleep $INTERVAL
    WAITED=$((WAITED + INTERVAL))
    if [ $WAITED -ge $MAX_WAIT ]; then
        echo "ERROR: Judge did not start within ${MAX_WAIT}s."
        kill $VLLM_PID 2>/dev/null
        exit 1
    fi
    echo "  waited ${WAITED}s..."
done
echo "Judge ready after ${WAITED}s."

export JUDGE_URL="http://localhost:$PORT/v1"
echo "$JUDGE_URL" > "$ENDPOINT_FILE"
echo "Judge URL  : $JUDGE_URL"

BASE_MODEL=$(find "$HF_HOME/hub/models--Qwen--Qwen3-4B/snapshots" \
    -maxdepth 1 -mindepth 1 -type d 2>/dev/null | head -1)
if [ -z "$BASE_MODEL" ]; then
    echo "ERROR: Qwen3-4B not found in HF cache ($HF_HOME/hub/)."
    kill $VLLM_PID 2>/dev/null
    exit 1
fi
echo "Base model : $BASE_MODEL"

echo "Starting minimal GRPO smoke test..."
CUDA_VISIBLE_DEVICES=1 python scripts/train_grpo.py \
    --dataset            dataset/ \
    --base-model         "$BASE_MODEL" \
    --sft-adapter        "$PROJECT_DIR/adapter-v3" \
    --output             "$TEST_OUTPUT_DIR" \
    --judge-url          "$JUDGE_URL" \
    --judge-model        Qwen/Qwen3.6-27B \
    --max-train-examples 2 \
    --max-steps          1 \
    --epochs             1 \
    --batch-size         1 \
    --grad-accum         2 \
    --lr                 5e-6 \
    --group-size         2 \
    --kl-coef            0.1 \
    --max-new-tokens     256 \
    --save-steps         1

STATUS=$?
kill $VLLM_PID 2>/dev/null
wait $VLLM_PID 2>/dev/null

if [ $STATUS -ne 0 ]; then
    echo "GRPO smoke test failed with exit code $STATUS"
    exit $STATUS
fi

echo "GRPO smoke test complete. Output written to $TEST_OUTPUT_DIR"
