#!/bin/bash
# ============================================================================
# Serve Qwen3.6-27B locally and score the corpus with it — the second reader
# for the in-copyright works.
#
# WHY LOCAL: seven of the eight coauthor-supplied texts are in US copyright.
# Jev has scored them (user's decision), but that leaves them on one reader
# where every other work has three, and Faulkner — whose shape those texts
# just changed — was already the most reader-sensitive author in the corpus.
# A locally-served model adds a second reading without any text leaving
# Princeton.
#
# THE GATE: this job scores the 277-passage calibration set FIRST and stops
# if the model collapses onto one soc_type. llama-3.3-70B passed every cheap
# check and still put 90.7% of its judgements in one category, after burning
# 12h47m of GPU time. That must not happen twice, so the check is in the job
# rather than in someone's intentions.
#
# THEN, in order:
#   2. a stratified sample of the public-domain works (k=1) — enough for
#      proxy_validity.py to ask
#      whether this reader tracks qwen3-max closely enough to stand in for it.
#      Scoring only the in-copyright works would leave them unanchored.
#   3. the 8 coauthor works (k=1).
#
# k=1 for stages 2-3: k-run variance has been 1-6% for every reader measured,
# so resampling buys almost nothing, and k=3 would push this past gpu-short's
# 24h cap. Stage 1 keeps k=3 precisely to measure that variance here.
#
# Output goes to scored_local/, never scored/ — a reader under evaluation
# must not land in the corpus before it has passed.
# ============================================================================
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
# gres and constraint are set on the sbatch command line, so this script can
# target either a single gpu80 (TP=1) or two gpu40s (TP=2). The 27B needs
# ~54GB in bf16: one 80GB card fits it, two 40GB cards fit it sharded, one
# 40GB card does not.
#SBATCH --time=12:00:00
#SBATCH --qos=gpu-short
#SBATCH --output=technique_v_genre/logs/qwenlocal_%j.out
#SBATCH --error=technique_v_genre/logs/qwenlocal_%j.err
#SBATCH --mail-type=fail
#SBATCH --mail-user=aj7878@princeton.edu

set -eo pipefail

LABEL=qwen3.6-local
HFMODEL=Qwen/Qwen3.6-27B
PORT=8000
TP="${TP:-1}"

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

echo "Job $SLURM_JOB_ID on $SLURMD_NODENAME — $LABEL (TP=$TP)"

# Fail fast on MIG. The gpu40 feature is carried by two different node types on
# this cluster: della-i14g[1-20] with real 2x A100 40GB cards, and
# della-l01g[3-12] with 3g.40gb MIG slices. Job 14868449 landed on a MIG node,
# vLLM was handed CUDA_VISIBLE_DEVICES=MIG-cf72618f-... , its config parser
# called int() on that, and the job sat for the full 30-minute readiness
# timeout before reporting "vLLM failed to start". Tensor parallelism across
# MIG slices would not work regardless — NCCL peer-to-peer is unsupported
# between instances. Submit with --constraint="gpu40&nomig" (or gpu80); this
# check is the backstop.
if [[ "${CUDA_VISIBLE_DEVICES:-}" == *MIG-* ]]; then
    echo "ABORT: scheduled onto a MIG instance (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)." >&2
    echo "       vLLM cannot parse MIG UUIDs and TP>1 is unsupported across slices." >&2
    echo "       Resubmit with --constraint=\"gpu40&nomig\" or --constraint=gpu80." >&2
    exit 2
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

vllm serve "$HFMODEL" --port "$PORT" --served-model-name "$LABEL" \
    --tensor-parallel-size "$TP" --max-model-len 8192 --dtype bfloat16 \
    --gpu-memory-utilization 0.92 --enable-prefix-caching \
    > "$TV/logs/serve_${LABEL}_${SLURM_JOB_ID}.out" \
    2> "$TV/logs/serve_${LABEL}_${SLURM_JOB_ID}.err" &
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

# ---- stage 1: the gate -----------------------------------------------------
echo "=== STAGE 1: calibration gate ==="
# Resumable: score.py skips (passage_id, reader_id, run) triples already
# present, so a resubmit after job 14838700 re-reads the banked result in
# seconds rather than re-spending 3h on it. The gate still RUNS — the check
# below is what gates the corpus pass, and it must not be skipped just
# because the rows exist.
python score.py --input calibration/calibration_set.jsonl --readers "$LABEL" \
    --k 3 --concurrency 64 --output "calibration/scored_${LABEL}.jsonl"

python - "$LABEL" <<'PYEOF'
import collections, json, sys
from pathlib import Path
label=sys.argv[1]
rows=[json.loads(l) for l in Path(f"calibration/scored_{label}.jsonl").read_text().splitlines() if l.strip()]
types=collections.Counter(d.get("soc_type") for d in rows if d.get("is_soc") and "error" not in d)
n=sum(types.values())
top,cnt=(types.most_common(1) or [(None,0)])[0]
share=cnt/n if n else 1.0
print(f"  top soc_type {top!r} = {100*share:.1f}% of {n} judgements")
if share > 0.70:
    print(f"  GATE FAILED: collapsed onto one category (>70%). Stopping before the corpus pass.")
    sys.exit(42)
print("  GATE PASSED — proceeding to the corpus.")
PYEOF

# ---- stages 2 and 3 --------------------------------------------------------
echo "=== STAGE 2+3: full corpus, k=1 ==="
WORKLIST="${WORKLIST:-}"
# Take the work list from a file when one is supplied. Both earlier runs
# timed out because they were sized for all 54 works when the deliverable was
# a subset — the walltime went to works nobody needed.
if [ -n "$WORKLIST" ] && [ -f "$WORKLIST" ]; then
    cp "$WORKLIST" /tmp/qwenlocal_worklist.txt
    echo "using supplied worklist: $WORKLIST ($(wc -l < "$WORKLIST") works)"
else
python - > /tmp/qwenlocal_worklist.txt <<'PYEOF'
import json
from pathlib import Path
man=json.loads(Path("corpus_manifest.json").read_text())["corpus"]
new=[e for e in man if e.get("provenance_note","").startswith("Supplied by the project coauthor")]
newset={e["source_file"] for e in new}
for e in new: print(e["source_file"][:-4])
for e in man:
    if e["source_file"] not in newset and (Path("passages")/f"{e['source_file'][:-4]}.jsonl").exists():
        print(e["source_file"][:-4])
PYEOF
fi

mapfile -t FILES < /tmp/qwenlocal_worklist.txt
echo "${#FILES[@]} works queued"

for name in "${FILES[@]}"; do
    [ -n "$name" ] || continue
    echo "--- $name"
    python score.py --input "passages/${name}.jsonl" --readers "$LABEL" \
        --k 1 --concurrency 64 --output "scored_local/${name}.jsonl"
done

echo "QWEN LOCAL SCORING COMPLETE"
