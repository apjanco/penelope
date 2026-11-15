#!/bin/bash
# ============================================================================
# Penelope technique-vs-genre — consolidated multi-model vLLM server
#
# One SLURM job, one 4x A100-80GB node, three readers pinned to their own
# GPU(s) via CUDA_VISIBLE_DEVICES. Built as a single job rather than one
# job per model because gpu-medium/gpu-long jobs queue for a long time
# under this account's fairshare (see conversation log 2026-08-02) — one
# trip through the priority queue for the whole local roster, not three.
#
# Roster (fits exactly on 4 GPUs):
#   qwen3-reasoning : Qwen/Qwen3.6-27B              1 GPU  (already cached)
#   mistral         : Mistral-Small-24B-Instruct    1 GPU  (ungated)
#   llama           : Llama-3.3-70B-Instruct        2 GPUs, TP=2 (gated;
#                     requires HF_TOKEN in .env with the license accepted)
#
# Weights must be pre-downloaded on a login/vis node first (compute nodes
# have no internet — see della_instructions.md) — this job runs with
# HF_HUB_OFFLINE=1 and will fail to start any model whose weights aren't
# already in the HF cache.
#
# Submit:
#   sbatch --gres=gpu:a100:4 --constraint=gpu80 technique_v_genre/serve_vllm_multi.slurm
#
# Each model writes its own endpoint file to technique_v_genre/endpoints/
# as soon as it's ready (not gated on the others) — register each one as
# its own profile in models.yaml.
# ============================================================================

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=72:00:00
#SBATCH --qos=gpu-medium
#SBATCH --output=technique_v_genre/logs/serve_multi_%j.out
#SBATCH --error=technique_v_genre/logs/serve_multi_%j.err

set -eo pipefail
# NOT set -u: conda's activation scripts reference $PS1, unset in a batch
# job (see serve_vllm.slurm for the full explanation / the failure this
# caused on first submission).

module purge
module load anaconda3/2025.12
module load cudatoolkit/13.2

source /usr/licensed/anaconda3/2025.12/etc/profile.d/conda.sh
conda activate penelope

export HF_HOME=/scratch/gpfs/MM4/apjanco/.cache/huggingface
export HF_HUB_OFFLINE=1
export VLLM_USE_FLASHINFER_SAMPLER=0
export CPATH="/usr/local/cuda-13.2/targets/x86_64-linux/include${CPATH:+:$CPATH}"

PROJECT_DIR="/scratch/gpfs/MM4/apjanco/penelope"
ENDPOINT_DIR="$PROJECT_DIR/technique_v_genre/endpoints"
LOG_DIR="$PROJECT_DIR/technique_v_genre/logs"
cd "$PROJECT_DIR"
mkdir -p "$ENDPOINT_DIR" "$LOG_DIR"

echo "Job ID : $SLURM_JOB_ID"
echo "Node   : $SLURMD_NODENAME  ($(hostname -f))"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# label | model | cuda_visible_devices | port | tensor_parallel_size | max_model_len
SPECS=(
  "qwen3-reasoning|Qwen/Qwen3.6-27B|0|8000|1|8192"
  "mistral|mistralai/Mistral-Small-24B-Instruct-2501|1|8001|1|8192"
  "llama|meta-llama/Llama-3.3-70B-Instruct|2,3|8002|2|4096"
)
# Llama gets a smaller max_model_len than the others: weights alone are
# ~140GB across 2x80GB GPUs, leaving little headroom for KV cache at
# gpu-memory-utilization 0.92 — 4096 is still far more than this project's
# passages + system prompt need (well under 2000 tokens combined).

PIDS=()

for spec in "${SPECS[@]}"; do
  IFS='|' read -r LABEL MODEL DEVICES PORT TP MAXLEN <<< "$spec"
  ENDPOINT_FILE="$ENDPOINT_DIR/${LABEL}.txt"
  rm -f "$ENDPOINT_FILE"
  echo "Launching $LABEL ($MODEL) on GPU(s) $DEVICES, port $PORT, TP=$TP, max_model_len=$MAXLEN"

  CUDA_VISIBLE_DEVICES="$DEVICES" vllm serve "$MODEL" \
      --port "$PORT" \
      --served-model-name "$LABEL" \
      --tensor-parallel-size "$TP" \
      --max-model-len "$MAXLEN" \
      --dtype bfloat16 \
      --gpu-memory-utilization 0.92 \
      --enable-prefix-caching \
      > "$LOG_DIR/serve_multi_${LABEL}_${SLURM_JOB_ID}.out" \
      2> "$LOG_DIR/serve_multi_${LABEL}_${SLURM_JOB_ID}.err" &

  PIDS+=($!)
  echo "  PID: $!"
done

echo "Launched PIDs: ${PIDS[*]}"

# ── Wait for each server independently, write its endpoint as soon as it's ready ──
declare -A READY
MAX_WAIT=1800
WAITED=0
INTERVAL=15

while [ "${#READY[@]}" -lt "${#SPECS[@]}" ] && [ "$WAITED" -lt "$MAX_WAIT" ]; do
  for spec in "${SPECS[@]}"; do
    IFS='|' read -r LABEL MODEL DEVICES PORT TP MAXLEN <<< "$spec"
    if [ -z "${READY[$LABEL]:-}" ] && curl -sf "http://localhost:$PORT/health" > /dev/null 2>&1; then
      ENDPOINT="http://$(hostname -f):$PORT/v1"
      echo "$ENDPOINT" > "$ENDPOINT_DIR/${LABEL}.txt"
      echo "[$LABEL] ready after ${WAITED}s -> $ENDPOINT"
      READY[$LABEL]=1
    fi
  done
  if [ "${#READY[@]}" -lt "${#SPECS[@]}" ]; then
    sleep $INTERVAL
    WAITED=$((WAITED + INTERVAL))
  fi
done

for spec in "${SPECS[@]}"; do
  IFS='|' read -r LABEL MODEL DEVICES PORT TP MAXLEN <<< "$spec"
  if [ -z "${READY[$LABEL]:-}" ]; then
    echo "WARNING: $LABEL did not become ready within ${MAX_WAIT}s — check $LOG_DIR/serve_multi_${LABEL}_${SLURM_JOB_ID}.err"
  fi
done

echo "All servers launched (${#READY[@]}/${#SPECS[@]} ready). Serving until wall-clock limit or cancellation."
wait
