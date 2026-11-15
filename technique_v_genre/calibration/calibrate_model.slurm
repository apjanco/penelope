#!/bin/bash
# ============================================================================
# Score the reader-calibration set with a local model, on a GPU node.
#
# Separate from serve_score_model.slurm because that one reads passages/<name>
# and writes scored/<name> — the corpus layout. This writes into calibration/
# so a candidate under evaluation never lands in the corpus by accident.
#
# Server and client run in the SAME job, as everywhere else in this project
# (see serve_score_model.slurm's header and fill_qwen_gaps.slurm for why that
# must not be split).
#
# Submit:
#   sbatch --job-name=cal-gemma --gres=gpu:1 --time=02:00:00 \
#     --export=ALL,LABEL=gemma,HFMODEL=google/gemma-3-27b-it,TP=1,MAXLEN=8192 \
#     technique_v_genre/calibration/calibrate_model.slurm
# ============================================================================
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --time=02:00:00
#SBATCH --qos=gpu-short
#SBATCH --output=technique_v_genre/logs/cal_%x_%j.out
#SBATCH --error=technique_v_genre/logs/cal_%x_%j.err
#SBATCH --mail-type=fail
#SBATCH --mail-user=aj7878@princeton.edu

set -eo pipefail   # NOT -u: conda activation scripts reference unset $PS1

: "${LABEL:?set LABEL}"
: "${HFMODEL:?set HFMODEL}"
TP="${TP:-1}"
MAXLEN="${MAXLEN:-8192}"
K="${K:-3}"
PORT=8000

module purge
module load anaconda3/2025.12
module load cudatoolkit/13.2
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate penelope

export HF_HOME=/scratch/gpfs/MM4/apjanco/.cache/huggingface
export HF_HUB_OFFLINE=1
export VLLM_USE_FLASHINFER_SAMPLER=0

PROJECT_DIR="/scratch/gpfs/MM4/apjanco/penelope"
TV="$PROJECT_DIR/technique_v_genre"
cd "$PROJECT_DIR"
mkdir -p "$TV/logs" "$TV/calibration"

echo "Job $SLURM_JOB_ID on $SLURMD_NODENAME — $LABEL ($HFMODEL) TP=$TP"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

vllm serve "$HFMODEL" --port "$PORT" --served-model-name "$LABEL" \
    --tensor-parallel-size "$TP" --max-model-len "$MAXLEN" --dtype bfloat16 \
    --gpu-memory-utilization 0.92 --enable-prefix-caching \
    > "$TV/logs/serve_${LABEL}_${SLURM_JOB_ID}.out" \
    2> "$TV/logs/serve_${LABEL}_${SLURM_JOB_ID}.err" &
VLLM_PID=$!
trap 'kill "$VLLM_PID" 2>/dev/null || true' EXIT

MAX_WAIT=1800; WAITED=0
until curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1; do
    sleep 15; WAITED=$((WAITED+15))
    [ $WAITED -ge $MAX_WAIT ] && { echo "vLLM failed to start"; exit 1; }
done
echo "server ready after ${WAITED}s"

# Register the endpoint under flock, same as serve_score_model.slurm.
( flock -x 200
  python3 - "$LABEL" <<'PYEOF'
import re, sys
import yaml
from pathlib import Path
label = sys.argv[1]
p = Path("models.yaml"); text = p.read_text()
entry = (f"  - label: {label}\n    base_url: http://localhost:8000/v1\n"
         f"    model_name: {label}\n    api_key: not-needed\n")
rx = re.compile(rf"  - label: {re.escape(label)}\n(?:    .*\n)*")
if rx.search(text):
    text = rx.sub(entry, text)
else:
    # Insert INSIDE the models: list, before the commented example. Appending
    # to the end of the file instead put the entry after the top-level
    # chunk_size/chunk_overlap keys, orphaning it outside `models:` and making
    # the whole file unparseable — which killed job 14787255 after the server
    # had already spent 225s loading. Refuse to write anything that does not
    # round-trip through the YAML parser.
    marker = "  # Example: local model via Ollama / vLLM"
    if marker in text:
        text = text.replace(marker, entry + "\n" + marker, 1)
    else:
        lines = text.splitlines(keepends=True)
        end = max((i for i, l in enumerate(lines)
                   if l.startswith("  - label:") or l.startswith("    ")), default=None)
        if end is None:
            raise SystemExit("models.yaml: cannot locate the models: list")
        lines.insert(end + 1, entry)
        text = "".join(lines)
try:
    doc = yaml.safe_load(text)
    assert any(m.get("label") == label for m in doc.get("models", []))
except Exception as e:
    raise SystemExit(f"models.yaml edit would corrupt the file ({e}) - aborting")
p.write_text(text); print(f"models.yaml: {label} registered")
PYEOF
) 200>"$PROJECT_DIR/models.yaml.lock"

cd "$TV"
python score.py \
    --input calibration/calibration_set.jsonl \
    --readers "$LABEL" --k "$K" --concurrency 16 \
    --output "calibration/scored_${LABEL}.jsonl"

echo "CALIBRATION COMPLETE for $LABEL"
