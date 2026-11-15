#!/bin/bash
# ============================================================================
# Qwen3.6-27B on the 33 "Time Passes" passages at k=10.
#
# WHY: the manual-review section compares every reader on one section, against
# two human annotators. gpt-5.4 and qwen3-max already have k=10 there and Jev
# now does too; qwen3.6-local — the second reader for the in-copyright works,
# and the only open-weights reader that passed the calibration gate — did not.
# Leaving it out would mean the human comparison covers the cloud readers only.
#
# k=10 matches the other readers so self-consistency is measured on the same
# footing. 330 calls at concurrency 64 is a few minutes; the walltime is almost
# entirely vLLM loading the 27B.
#
# The calibration gate still runs. Its rows are banked from the earlier job, so
# score.py's resume returns them in seconds, but the check itself is cheap and
# a reader must not reach the corpus ungated.
# ============================================================================
#SBATCH --job-name=tp_qwenlocal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --time=01:30:00
#SBATCH --qos=gpu-short
#SBATCH --output=technique_v_genre/logs/tp_qwenlocal_%j.out
#SBATCH --error=technique_v_genre/logs/tp_qwenlocal_%j.err
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

echo "Job $SLURM_JOB_ID on $SLURMD_NODENAME — $LABEL, Time Passes k=10"

if [[ "${CUDA_VISIBLE_DEVICES:-}" == *MIG-* ]]; then
    echo "ABORT: scheduled onto a MIG instance (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)." >&2
    exit 2
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

vllm serve "$HFMODEL" --port "$PORT" --served-model-name "$LABEL" \
    --tensor-parallel-size 1 --max-model-len 8192 --dtype bfloat16 \
    --gpu-memory-utilization 0.92 --enable-prefix-caching \
    > "$TV/logs/serve_tp_${SLURM_JOB_ID}.out" \
    2> "$TV/logs/serve_tp_${SLURM_JOB_ID}.err" &
VLLM_PID=$!
trap 'kill "$VLLM_PID" 2>/dev/null || true' EXIT

WAITED=0
until curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1; do
    sleep 15; WAITED=$((WAITED+15))
    [ $WAITED -ge 1800 ] && { echo "vLLM failed to start"; exit 1; }
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

echo "=== calibration gate (resumes from banked rows) ==="
python score.py --input calibration/calibration_set.jsonl --readers "$LABEL" \
    --k 3 --concurrency 64 --output "calibration/scored_${LABEL}.jsonl"

python - "$LABEL" <<'PYEOF'
import collections, json, sys
from pathlib import Path
label=sys.argv[1]
rows=[json.loads(l) for l in Path(f"calibration/scored_{label}.jsonl").read_text().splitlines() if l.strip()]
types=collections.Counter(d.get("soc_type") for d in rows if d.get("is_soc") and "error" not in d)
n=sum(types.values()); top,cnt=(types.most_common(1) or [(None,0)])[0]
share=cnt/n if n else 1.0
print(f"  top soc_type {top!r} = {100*share:.1f}% of {n} judgements")
if share > 0.70:
    print("  GATE FAILED: collapsed onto one category (>70%)."); sys.exit(42)
print("  GATE PASSED")
PYEOF

echo "=== Time Passes, k=10 ==="
python score.py --input passages/time_passes.jsonl --readers "$LABEL" \
    --k 10 --concurrency 64 --output "scored_local/time_passes.jsonl"

echo "TIME PASSES LOCAL SCORING COMPLETE"
