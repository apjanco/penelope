#!/bin/bash
# =============================================================================
# resample_run.sh — run scripts/resample.py from the Della login node.
#
# Compute nodes have no internet access, so this script must be run from a
# login node (della8, della9, etc.), NOT submitted via sbatch.
#
# The script is safe to kill and re-run: resample.py checks which
# (passage_key, run_index) pairs are already written and skips them.
#
# Recommended workflow:
#   1. SSH to a visualization node (both have internet access):
#        ssh <NetID>@della-vis1.princeton.edu   # 80 cores, 1 TB RAM, A100 40 GB
#        ssh <NetID>@della-vis2.princeton.edu   # 28 cores, 256 GB RAM, 4× P100
#   2. Open a persistent session so it survives disconnection:
#        screen -S resample        # or: tmux new -s resample
#   3. cd to the project dir:
#        cd /scratch/gpfs/MM4/apjanco/penelope
#   4. Run one model at a time (each writes to a separate file):
#        bash resample_run.sh mercury-2
#        bash resample_run.sh qwen3-max
#        bash resample_run.sh gpt-5.4
#      Or run all three in parallel in separate screen windows (Ctrl-A c):
#        bash resample_run.sh mercury-2 &
#        bash resample_run.sh qwen3-max &
#        bash resample_run.sh gpt-5.4 &
#   5. Detach (screen: Ctrl-A D  |  tmux: Ctrl-B D) and reconnect later.
#   6. Monitor progress:
#        tail -f logs/resample_mercury-2.log
#        wc -l silver_data/resampled/mercury-2.jsonl
#
# Priority order (lowest agreement → most valuable boundary cases first):
#   Richardson texts: pg04517_pointed_roofs.txt  pg04518_backwater.txt  pg04519_honeycomb.txt
#   Use --source-file to run a single file first if you want to test:
#        bash resample_run.sh mercury-2 pg04517_pointed_roofs.txt
#
# Estimated API calls (full corpus):
#   mercury-2:  6,302 passages × k=10 = 63,020 calls
#   qwen3-max:  6,302 passages × k=10 = 63,020 calls
#   gpt-5.4:    6,302 passages × k=1  =  6,302 calls
#   Total: ~132,342 calls
# =============================================================================

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON=/home/aj7878/.conda/envs/penelope/bin/python
ENV_FILE="$PROJECT_DIR/.env"
LOG_DIR="$PROJECT_DIR/logs"
RESAMPLE_DIR="$PROJECT_DIR/silver_data/resampled"

MODEL="${1:-}"
SOURCE_FILE="${2:-}"

if [[ -z "$MODEL" ]]; then
    echo "Usage: bash resample_run.sh <model-label> [source-file]"
    echo "  model-label:  mercury-2 | qwen3-max | gpt-5.4"
    echo "  source-file:  optional, e.g. pg04517_pointed_roofs.txt"
    exit 1
fi

mkdir -p "$LOG_DIR" "$RESAMPLE_DIR"

LOG_FILE="$LOG_DIR/resample_${MODEL}.log"

ARGS=(
    --model "$MODEL"
    --silver-dir "$PROJECT_DIR/silver_data"
    --output-dir "$RESAMPLE_DIR"
    --models-yaml "$PROJECT_DIR/models.yaml"
    --k 10
    --temperature 0.8
)

if [[ -n "$SOURCE_FILE" ]]; then
    ARGS+=(--source-file "$SOURCE_FILE")
    LOG_FILE="$LOG_DIR/resample_${MODEL}_${SOURCE_FILE%.txt}.log"
fi

if [[ -f "$ENV_FILE" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
fi

echo "Starting resample for model: $MODEL"
echo "Log: $LOG_FILE"
echo "Output: $RESAMPLE_DIR/${MODEL}.jsonl"
if [[ -f "$RESAMPLE_DIR/${MODEL}.jsonl" ]]; then
    echo "Resuming — $(wc -l < "$RESAMPLE_DIR/${MODEL}.jsonl") lines already written"
fi
echo ""

# Dry-run first to show scope
"$PYTHON" "$PROJECT_DIR/scripts/resample.py" \
    "${ARGS[@]}" \
    --dry-run 2>&1 | grep -E "passages|calls|pending|done"

echo ""
echo "Starting in 3 seconds… (Ctrl-C to abort)"
sleep 3

exec "$PYTHON" "$PROJECT_DIR/scripts/resample.py" \
    "${ARGS[@]}" \
    --verbose \
    2>&1 | tee -a "$LOG_FILE"
