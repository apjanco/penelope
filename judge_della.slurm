#!/bin/bash
# ============================================================================
# Penelope SoC — vLLM judge server for GRPO training
#
# Starts a Qwen3.6-27B judge server on 1×A100 80GB.
# Writes the endpoint URL to judge_endpoint.txt when ready.
# The GRPO training job reads that file to discover the judge URL.
#
# Submit BEFORE train_grpo.slurm:
#   JUDGE_JOB=$(sbatch --parsable judge_della.slurm)
#   sbatch --dependency=after:$JUDGE_JOB train_grpo.slurm
#
# Monitor:
#   tail -f logs/judge_<JOBID>.out
#   cat judge_endpoint.txt        # appears when server is ready
#
# Alternative: Qwen/Qwen3-32B also fits on 1×80GB (~64GB BF16)
# ============================================================================

#SBATCH --job-name=penelope-judge
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80         # 1×A100 80GB for Qwen3.6-27B in BF16
#SBATCH --time=12:00:00            # keep alive for the duration of GRPO run
#SBATCH --output=logs/judge_%j.out
#SBATCH --error=logs/judge_%j.err
#SBATCH --mail-type=begin,fail
#SBATCH --mail-user=               # fill in your Princeton email

# ── Environment ─────────────────────────────────────────────────────────────
module purge
module purge
module load anaconda3/2025.12
module load cudatoolkit/13.2

conda activate penelope

export HF_HOME=/scratch/gpfs/MM4/apjanco/.cache/huggingface
export HF_HUB_OFFLINE=1

# ── flashinfer JIT fix ───────────────────────────────────────────────────────
# flashinfer tries to JIT-compile its sampling kernel and fails with
# "curand.h: No such file or directory" because the module-based CUDA install
# puts headers under targets/x86_64-linux/include, not the top-level include/.
# Setting VLLM_USE_FLASHINFER_SAMPLER=0 tells vLLM to use native PyTorch
# sampling instead, bypassing the JIT entirely (negligible throughput impact
# for a judge serving sequential requests).
export VLLM_USE_FLASHINFER_SAMPLER=0
# Belt-and-suspenders: expose the targets include path in case any other
# flashinfer kernel (attention) also needs curand.h at build time.
export CPATH="/usr/local/cuda-13.2/targets/x86_64-linux/include${CPATH:+:$CPATH}"

PROJECT_DIR="/scratch/gpfs/MM4/apjanco/penelope"
ENDPOINT_FILE="$PROJECT_DIR/judge_endpoint.txt"
PORT=8000
MODEL="Qwen/Qwen3.6-27B"    # alternative: Qwen/Qwen3-32B (~64 GB BF16, also fits on 1×80GB)

cd "$PROJECT_DIR" || { echo "Project dir not found: $PROJECT_DIR"; exit 1; }
mkdir -p logs

# Remove stale endpoint file from previous runs
rm -f "$ENDPOINT_FILE"

echo "Job ID  : $SLURM_JOB_ID"
echo "Node    : $SLURMD_NODENAME  ($(hostname -f))"
echo "GPUs    : $CUDA_VISIBLE_DEVICES"
echo "Model   : $MODEL"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# ── Start vLLM server ────────────────────────────────────────────────────────
echo "Starting vLLM server..."

vllm serve "$MODEL" \
    --port "$PORT" \
    --tensor-parallel-size 1 \
    --max-model-len 8192 \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.92 \
    &

VLLM_PID=$!
echo "vLLM PID: $VLLM_PID"

# ── Wait for server readiness (with timeout) ─────────────────────────────────
MAX_WAIT=600    # 10 minutes — 27B loads in ~3-5 min
WAITED=0
INTERVAL=15

echo "Waiting for server to become ready (max ${MAX_WAIT}s)..."
until curl -sf "http://localhost:$PORT/health" > /dev/null 2>&1; do
    sleep $INTERVAL
    WAITED=$((WAITED + INTERVAL))
    if [ $WAITED -ge $MAX_WAIT ]; then
        echo "ERROR: vLLM did not start within ${MAX_WAIT}s. Exiting."
        kill $VLLM_PID 2>/dev/null
        exit 1
    fi
    echo "  waited ${WAITED}s..."
done

echo "Server ready after ${WAITED}s."

# ── Publish endpoint ─────────────────────────────────────────────────────────
ENDPOINT="http://$(hostname -f):$PORT/v1"
echo "$ENDPOINT" > "$ENDPOINT_FILE"
echo "Endpoint written to $ENDPOINT_FILE: $ENDPOINT"

# ── Keep job alive until GRPO finishes ──────────────────────────────────────
# The training job will cancel this job when it's done, or it will
# expire naturally at the --time limit.
wait $VLLM_PID
