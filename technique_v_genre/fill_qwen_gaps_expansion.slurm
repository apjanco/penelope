#!/bin/bash
# ============================================================================
# Fill the two content-filtered qwen3-max gaps from the 2026-09-29 corpus
# expansion with the same local fallback reader fill_qwen_gaps.slurm used.
#
#   pg07118_what_maisie_knew_0286
#   se_ford_madox_ford_the_last_post_0341
#
# Both fail qwen3-max with data_inspection_failed on every attempt (retried
# 2026-10-02, 6/6 rejected), so they get the same treatment as the original
# 11: Qwen/Qwen3.6-27B served locally, scored at k=3 as its own reader
# ("qwen3.6-local"), appended to the per-book scored/*.jsonl.
#
# models.yaml entry is written byte-identical to score_qwen_local.slurm's
# (localhost:8000, under the same flock), so this job and that one can run
# at the same time without pointing each other's clients at the wrong node.
# ============================================================================
#SBATCH --job-name=qwen-gapfill2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --qos=gpu-test
#SBATCH --time=00:30:00
#SBATCH --output=technique_v_genre/logs/fill_qwen_gaps2_%j.out
#SBATCH --error=technique_v_genre/logs/fill_qwen_gaps2_%j.err
#SBATCH --mail-type=fail
#SBATCH --mail-user=aj7878@princeton.edu

set -eo pipefail

LABEL=qwen3.6-local
HFMODEL=Qwen/Qwen3.6-27B
PORT=8000

module purge
module load anaconda3/2025.12
module load cudatoolkit/13.2
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate penelope

export HF_HOME=/scratch/gpfs/MM4/apjanco/.cache/huggingface
export HF_HUB_OFFLINE=1
export VLLM_USE_FLASHINFER_SAMPLER=0

PROJ=/scratch/gpfs/MM4/apjanco/penelope
TV="$PROJ/technique_v_genre"
cd "$PROJ"

echo "Job $SLURM_JOB_ID on $SLURMD_NODENAME — $LABEL"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

vllm serve "$HFMODEL" --port "$PORT" --served-model-name "$LABEL" \
    --tensor-parallel-size 1 --max-model-len 8192 --dtype bfloat16 \
    --gpu-memory-utilization 0.92 --enable-prefix-caching \
    > "$TV/logs/serve_${LABEL}_${SLURM_JOB_ID}.out" \
    2> "$TV/logs/serve_${LABEL}_${SLURM_JOB_ID}.err" &
VLLM_PID=$!
trap 'kill "$VLLM_PID" 2>/dev/null || true' EXIT

WAITED=0
until curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1; do
    sleep 15; WAITED=$((WAITED+15))
    [ $WAITED -ge 1200 ] && { echo "vLLM failed to start"; exit 1; }
done
echo "server ready after ${WAITED}s"

( flock -x 200
  python3 - "$LABEL" <<'PYEOF'
import re, sys, yaml
from pathlib import Path
label=sys.argv[1]; p=Path("models.yaml"); text=p.read_text()
entry=(f"  - label: {label}\n    base_url: http://localhost:8000/v1\n"
       f"    model_name: {label}\n    api_key: not-needed\n")
rx=re.compile(rf"  - label: {re.escape(label)}\n(?:    .*\n)*")
text = rx.sub(entry, text) if rx.search(text) else text.replace(
    "  # Example: local model via Ollama / vLLM", entry+"\n  # Example: local model via Ollama / vLLM",1)
doc=yaml.safe_load(text)
assert any(m.get("label")==label for m in doc.get("models",[])), "edit would corrupt models.yaml"
p.write_text(text); print(f"models.yaml: {label} registered")
PYEOF
) 200>"$PROJ/models.yaml.lock"

cd "$TV"
for name in pg07118_what_maisie_knew se_ford_madox_ford_the_last_post; do
    echo "--- $name"
    python score.py --input "passages/gaps/${name}.jsonl" --readers "$LABEL" \
        --k 3 --concurrency 3 --output "scored/${name}.jsonl"
done

echo "GAP-FILL COMPLETE"
