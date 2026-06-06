#!/bin/bash
# ============================================================================
# Penelope SoC — GRPO training on Princeton Della
#
# Reads the judge endpoint written by judge_della.slurm, then runs GRPO
# reinforcement learning on top of the SFT adapter.
#
# Submit AFTER judge_della.slurm:
#   JUDGE_JOB=$(sbatch --parsable judge_della.slurm)
#   sbatch --dependency=after:$JUDGE_JOB train_grpo.slurm
#
# Or submit independently if the judge is already running:
#   sbatch train_grpo.slurm
#
# Monitor:
#   tail -f logs/grpo_<JOBID>.out
#   watch -n 30 cat judge_endpoint.txt    # confirm judge is reachable
# ============================================================================

#SBATCH --job-name=penelope-grpo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80         # A100 80GB; QLoRA Qwen3-4B needs ~8 GB GPU
#SBATCH --time=24:00:00
#SBATCH --qos=gpu-short            # 24-hour blocks; resubmit with auto-resume
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

cd "$PROJECT_DIR" || { echo "Project dir not found: $PROJECT_DIR"; exit 1; }
mkdir -p logs

echo "Job ID     : $SLURM_JOB_ID"
echo "Node       : $SLURMD_NODENAME"
echo "GPUs       : $CUDA_VISIBLE_DEVICES"
echo "Working dir: $PWD"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "import torch; print('PyTorch', torch.__version__, '| CUDA', torch.version.cuda)"

# ── Wait for judge endpoint ──────────────────────────────────────────────────
MAX_WAIT=1200    # judge may still be loading when this job starts
WAITED=0
INTERVAL=15

echo "Waiting for judge endpoint ($ENDPOINT_FILE)..."
until [ -f "$ENDPOINT_FILE" ]; do
    sleep $INTERVAL
    WAITED=$((WAITED + INTERVAL))
    if [ $WAITED -ge $MAX_WAIT ]; then
        echo "ERROR: Judge endpoint not available after ${MAX_WAIT}s. Is judge_della.slurm running?"
        exit 1
    fi
    echo "  waited ${WAITED}s..."
done

export JUDGE_URL=$(cat "$ENDPOINT_FILE")
echo "Judge URL  : $JUDGE_URL"

# Verify judge is reachable before starting expensive training
if ! curl -sf "${JUDGE_URL%/v1}/health" > /dev/null 2>&1; then
    echo "ERROR: Judge health check failed at $JUDGE_URL"
    exit 1
fi
echo "Judge health check passed."

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
if compgen -G "$PROJECT_DIR/adapter-grpo/checkpoints/checkpoint-*" > /dev/null 2>&1; then
    RESUME_FLAG="--resume"
    LATEST=$(ls -d "$PROJECT_DIR/adapter-grpo/checkpoints/checkpoint-"* 2>/dev/null | sort -V | tail -1)
    echo "Checkpoint detected: $LATEST — resuming."
else
    echo "No checkpoint found — starting from scratch."
fi

# ── GRPO Training ─────────────────────────────────────────────────────────────
python scripts/train_grpo.py \
    --dataset        dataset/ \
    --base-model     "$BASE_MODEL" \
    --sft-adapter    "$PROJECT_DIR/adapter" \
    --output         adapter-grpo/ \
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

# ── Optional: cancel judge job when done ────────────────────────────────────
# Uncomment if you want to automatically release the judge GPU:
# scancel "$SLURM_JOB_ID"   # this would cancel ourselves — use JUDGE_JOB_ID
# To cancel the judge: store the judge job ID and pass it here via env var
# e.g. JUDGE_JOB_ID=... sbatch train_grpo.slurm  and then:
# [ -n "$JUDGE_JOB_ID" ] && scancel "$JUDGE_JOB_ID" && echo "Cancelled judge job $JUDGE_JOB_ID"
