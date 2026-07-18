#!/bin/bash
# ============================================================================
# Penelope SoC — GRPO training on Princeton Della
#
# Self-contained 2-GPU job: starts vLLM judge on GPU 0, runs GRPO training
# on GPU 1. No separate judge_della.slurm needed.
#
# Submit:
#   sbatch train_grpo.slurm
#
# Monitor:
#   tail -f logs/grpo_<JOBID>.out
# ============================================================================

#SBATCH --job-name=penelope-grpo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gres=gpu:2
#SBATCH --constraint=gpu80         # 2×A100 80GB: GPU 0 = vLLM judge, GPU 1 = GRPO training
#SBATCH --time=24:00:00
#SBATCH --qos=gpu-medium           # gpu-medium allows longer wall time than gpu-short
#SBATCH --output=logs/grpo_%j.out
#SBATCH --error=logs/grpo_%j.err
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=               # fill in your Princeton email

# ── Environment ─────────────────────────────────────────────────────────────
module purge
module load anaconda3/2025.12
module load cudatoolkit/13.2

conda activate penelope

export HF_HOME=/scratch/gpfs/MM4/apjanco/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

PROJECT_DIR="/scratch/gpfs/MM4/apjanco/penelope"
ENDPOINT_FILE="$PROJECT_DIR/judge_endpoint.txt"
PORT=8000

cd "$PROJECT_DIR" || { echo "Project dir not found: $PROJECT_DIR"; exit 1; }
mkdir -p logs

echo "Job ID     : $SLURM_JOB_ID"
echo "Node       : $SLURMD_NODENAME"
echo "GPUs       : $CUDA_VISIBLE_DEVICES"
echo "Working dir: $PWD"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "import torch; print('PyTorch', torch.__version__, '| CUDA', torch.version.cuda)"

# ── Start judge (vLLM) on GPU 0 ─────────────────────────────────────────────
# GPU 0 = Qwen3.6-27B judge (~55 GB BF16)
# GPU 1 = QLoRA GRPO training (~8 GB)
rm -f "$ENDPOINT_FILE"

export VLLM_USE_FLASHINFER_SAMPLER=0
export CPATH="/usr/local/cuda-13.2/targets/x86_64-linux/include${CPATH:+:$CPATH}"

JUDGE_MODEL=$(find "$HF_HOME/hub/models--Qwen--Qwen3.6-27B/snapshots" \
    -maxdepth 1 -mindepth 1 -type d 2>/dev/null | head -1)
if [ -z "$JUDGE_MODEL" ]; then
    echo "ERROR: Qwen3.6-27B not found in HF cache. Pre-download on a login node."
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

# Wait for judge to be ready
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

# ── Resolve base model path from HF cache ───────────────────────────────────
# Passing the Hub ID offline can fail if refs/main is missing from the cache.
# Resolve the snapshot path directly to avoid the lookup entirely.
BASE_MODEL=$(find "$HF_HOME/hub/models--Qwen--Qwen3-4B/snapshots" \
    -maxdepth 1 -mindepth 1 -type d 2>/dev/null | head -1)
if [ -z "$BASE_MODEL" ]; then
    echo "ERROR: Qwen3-4B not found in HF cache ($HF_HOME/hub/). Pre-download it on a login node."
    exit 1
fi
echo "Base model : $BASE_MODEL"

# ── Auto-resume if a checkpoint exists ──────────────────────────────────────
RESUME_FLAG=""
if compgen -G "$PROJECT_DIR/adapter-v3-grpo/checkpoints/checkpoint-*" > /dev/null 2>&1; then
    RESUME_FLAG="--resume"
    LATEST=$(ls -d "$PROJECT_DIR/adapter-v3-grpo/checkpoints/checkpoint-"* 2>/dev/null | sort -V | tail -1)
    echo "Checkpoint detected: $LATEST — resuming."
else
    echo "No checkpoint found — starting from scratch."
fi

# ── GRPO Training on GPU 1 ───────────────────────────────────────────────────
CUDA_VISIBLE_DEVICES=1 python scripts/train_grpo.py \
    --dataset        dataset/ \
    --base-model     "$BASE_MODEL" \
    --sft-adapter    "$PROJECT_DIR/adapter-v3" \
    --output         adapter-v3-grpo/ \
    --judge-url      "$JUDGE_URL" \
    --judge-model    Qwen/Qwen3.6-27B \
    --epochs         1 \
    --batch-size     1 \
    --grad-accum     8 \
    --lr             5e-6 \
    --group-size     8 \
    --kl-coef        0.1 \
    --max-new-tokens 2048 \
    $RESUME_FLAG

echo "GRPO training complete. Adapter written to adapter-grpo/"

# ── Shut down judge ──────────────────────────────────────────────────────────
kill $VLLM_PID 2>/dev/null
echo "Judge server stopped."

# ── Optional: cancel judge job when done ────────────────────────────────────
# Uncomment if you want to automatically release the judge GPU:
# scancel "$SLURM_JOB_ID"   # this would cancel ourselves — use JUDGE_JOB_ID
# To cancel the judge: store the judge job ID and pass it here via env var
# e.g. JUDGE_JOB_ID=... sbatch train_grpo.slurm  and then:
# [ -n "$JUDGE_JOB_ID" ] && scancel "$JUDGE_JOB_ID" && echo "Cancelled judge job $JUDGE_JOB_ID"
