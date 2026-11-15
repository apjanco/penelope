#!/bin/bash
# ============================================================================
# Penelope technique-vs-genre — self-contained per-model full-corpus scorer.
#
# Replaces the monolithic serve_vllm_multi.slurm (4 GPUs x 72h, cancelled
# 2026-08-18 after ~6 days pending under Priority). One model per job: the
# vLLM server and the scoring client run in the SAME job (never split — see
# fill_qwen_gaps.slurm), so the job holds only the GPU(s) that one model
# needs and exits the moment scoring finishes. Small, single-/dual-GPU,
# gpu-short jobs backfill far better than one big long reservation.
#
# Parametrised by environment so all three local readers share this file;
# GPU count / walltime are set per model on the sbatch command line
# (see submit_local_scoring.sh). Required env:
#   LABEL   served-model name + models.yaml label + reader_id (e.g. mistral)
#   HFMODEL HuggingFace repo id (weights must already be in the HF cache;
#           compute nodes are offline, HF_HUB_OFFLINE=1)
# Optional env (defaults):
#   TP=1            tensor-parallel-size (llama needs 2)
#   MAXLEN=8192     max-model-len (llama uses 4096)
#   CONCURRENCY=32  concurrent scoring requests against the local server
#   FILES           space-separated passage-file stems to score. Defaults to
#                   the original 21-work corpus, so existing submissions are
#                   unchanged. Set it to score a subset — the 2026-09-29
#                   expansion passes the 16 stems in expansion_files.txt.
#
# Submit (see submit_local_scoring.sh for the three concrete commands):
#   sbatch --job-name=score-mistral --gres=gpu:1 --time=08:00:00 \
#     --export=ALL,LABEL=mistral,HFMODEL=mistralai/Mistral-Small-24B-Instruct-2501 \
#     technique_v_genre/serve_score_model.slurm
# ============================================================================

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --time=12:00:00
#SBATCH --qos=gpu-short
#SBATCH --output=technique_v_genre/logs/score_%x_%j.out
#SBATCH --error=technique_v_genre/logs/score_%x_%j.err
#SBATCH --mail-type=begin,fail
#SBATCH --mail-user=aj7878@princeton.edu

set -eo pipefail   # NOT -u: conda activation scripts reference unset $PS1 in batch jobs

: "${LABEL:?set LABEL (served-model name / reader_id)}"
: "${HFMODEL:?set HFMODEL (HuggingFace repo id)}"
TP="${TP:-1}"
MAXLEN="${MAXLEN:-8192}"
CONCURRENCY="${CONCURRENCY:-32}"
PORT=8000

# vLLM's tokenizer-mode=auto silently falls back to the generic HF tokenizer
# for Mistral checkpoints instead of mistral_common's, leaking raw BPE glyphs
# (Ġ/Ċ) into "decoded" text and breaking every JSON parse — 100% error rate,
# discovered on job 12590292 (2026-08-20) after it had already run through
# 8 files' worth of Mistral output as silent garbage. Force the correct mode
# whenever the HF repo id says mistralai, rather than trusting auto-detect.
EXTRA_ARGS=""
if [[ "$HFMODEL" == mistralai/* ]]; then
    EXTRA_ARGS="--tokenizer-mode mistral --config-format mistral"
    echo "Detected Mistral checkpoint -> forcing $EXTRA_ARGS"
fi

module purge
module load anaconda3/2025.12
module load cudatoolkit/13.2

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate penelope

export HF_HOME=/scratch/gpfs/MM4/apjanco/.cache/huggingface
export HF_HUB_OFFLINE=1
export VLLM_USE_FLASHINFER_SAMPLER=0
export CPATH="/usr/local/cuda-13.2/targets/x86_64-linux/include${CPATH:+:$CPATH}"

PROJECT_DIR="/scratch/gpfs/MM4/apjanco/penelope"
TV_DIR="$PROJECT_DIR/technique_v_genre"
cd "$PROJECT_DIR"
mkdir -p "$TV_DIR/logs" "$TV_DIR/endpoints"

echo "Job ID  : $SLURM_JOB_ID  ($SLURM_JOB_NAME)"
echo "Node    : $SLURMD_NODENAME  ($(hostname -f))"
echo "Reader  : $LABEL  ($HFMODEL)  TP=$TP  max_model_len=$MAXLEN"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# ── Launch the vLLM server in the background; kill it when the job exits ──
echo "Starting vLLM server..."
vllm serve "$HFMODEL" \
    --port "$PORT" \
    --served-model-name "$LABEL" \
    --tensor-parallel-size "$TP" \
    --max-model-len "$MAXLEN" \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.92 \
    --enable-prefix-caching \
    $EXTRA_ARGS \
    > "$TV_DIR/logs/serve_${LABEL}_${SLURM_JOB_ID}.out" \
    2> "$TV_DIR/logs/serve_${LABEL}_${SLURM_JOB_ID}.err" &
VLLM_PID=$!
echo "vLLM PID: $VLLM_PID"
cleanup() { echo "Shutting down vLLM ($VLLM_PID)..."; kill "$VLLM_PID" 2>/dev/null || true; }
trap cleanup EXIT

# ── Wait for readiness ──
MAX_WAIT=1800; WAITED=0; INTERVAL=15
echo "Waiting for server to become ready (max ${MAX_WAIT}s)..."
until curl -sf "http://localhost:$PORT/health" > /dev/null 2>&1; do
    sleep $INTERVAL; WAITED=$((WAITED + INTERVAL))
    if [ $WAITED -ge $MAX_WAIT ]; then
        echo "ERROR: vLLM did not start within ${MAX_WAIT}s (see logs/serve_${LABEL}_${SLURM_JOB_ID}.err)."
        exit 1
    fi
    echo "  waited ${WAITED}s..."
done
echo "Server ready after ${WAITED}s."

ENDPOINT="http://localhost:$PORT/v1"
echo "$ENDPOINT" > "$TV_DIR/endpoints/${LABEL}.txt"

# ── Register (or update) this reader in models.yaml, under flock so the
#    three concurrent per-model jobs never corrupt the shared file ──
(
  flock -x 200
  python3 - "$ENDPOINT" "$LABEL" <<'PYEOF'
import os, re, sys
from pathlib import Path
endpoint, label = sys.argv[1], sys.argv[2]
p = Path("models.yaml")
text = p.read_text()
entry = (
    f"  - label: {label}\n"
    f"    base_url: {endpoint}   # job {os.environ.get('SLURM_JOB_ID','?')} "
    f"(local vLLM, see technique_v_genre/serve_score_model.slurm)\n"
    f"    model_name: {label}\n"
    f"    api_key: not-needed\n"
)
label_re = re.compile(rf"  - label: {re.escape(label)}\n(?:    .*\n)*")
if label_re.search(text):
    text = label_re.sub(entry, text)
else:
    marker = "  # Example: local model via Ollama / vLLM"
    text = text.replace(marker, entry + "\n" + marker) if marker in text else text + "\n" + entry
p.write_text(text)
print(f"models.yaml updated: {label} -> {endpoint}")
PYEOF
) 200>"$PROJECT_DIR/models.yaml.lock"

# ── Score the full corpus (the 21 files pilot_experiments_abc.COMPLETE_FILES
#    reads), appending this reader to the existing per-book scored/*.jsonl.
#    Resumable by (passage_id, reader_id, run): a re-submit continues. ──
cd "$TV_DIR"
DEFAULT_CORPUS=(
    as_I_lay_dying between_the_acts flush jacobs_room night_and_day_cleaned
    orlando_cleaned pg00432_the_ambassadors pg02775_the_good_soldier
    pg02814_dubliners pg03019_pointed_roofs
    pg04217_a_portrait_of_the_artist_as_a_young_man pg04300_ulysses
    pg29452_the_wings_of_the_dove_volume_1 pg30059_the_wings_of_the_dove_volume_2
    pg56221_backwater pg56664_honeycomb pg75170_sound_and_fury the_voyage_out
    the_waves_cleaned the_years woolf_lighthouse_dalloway
)
# FILES overrides the default corpus; unset means "the original 21 works",
# so every submission written before this parameter existed behaves as before.
if [ -n "${FILES:-}" ]; then
    read -r -a CORPUS <<< "$FILES"
else
    CORPUS=( "${DEFAULT_CORPUS[@]}" )
fi

MISSING=0
for name in "${CORPUS[@]}"; do
    [ -f "passages/${name}.jsonl" ] || { echo "ERROR: passages/${name}.jsonl not found"; MISSING=1; }
done
[ "$MISSING" -eq 0 ] || { echo "ABORT: named passage files are missing."; exit 1; }
echo "Scoring ${#CORPUS[@]} works with reader $LABEL"
for name in "${CORPUS[@]}"; do
    echo "=========================================="
    echo "Scoring $name  (reader: $LABEL)"
    echo "=========================================="
    python score.py \
        --input "passages/${name}.jsonl" \
        --readers "$LABEL" \
        --k 3 \
        --concurrency "$CONCURRENCY" \
        --output "scored/${name}.jsonl"
done

echo "FULL-CORPUS SCORING COMPLETE for reader $LABEL"
