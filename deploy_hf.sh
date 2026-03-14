#!/usr/bin/env bash
set -euo pipefail
#
# Deploy the Penelope Streamlit app to Hugging Face Spaces.
#
# Prerequisites:
#   pip install huggingface_hub
#   huggingface-cli login          # authenticate once
#
# Usage:
#   ./deploy_hf.sh                         # default: apjanco/penelope
#   ./deploy_hf.sh myorg/my-space-name     # custom space
#
# What this script does:
#   1. Refreshes the hf_space/ folder with the latest per-model JSON results
#   2. Pushes the entire hf_space/ folder to the HF Space repo
#

SPACE="${1:-apjanco/penelope}"
HF_SPACE_DIR="$(cd "$(dirname "$0")" && pwd)/hf_space"
RESULTS_DIR="$(cd "$(dirname "$0")" && pwd)/results"

echo "🧶 Deploying Penelope to HF Spaces: $SPACE"
echo ""

# ── 1. Refresh data ──────────────────────────────────────────────────
echo "📦 Copying latest per-model results to hf_space/results/ ..."
mkdir -p "$HF_SPACE_DIR/results"

# Copy only per-model JSON files (not consensus, results.json, CSVs, etc.)
for f in "$RESULTS_DIR"/*.json; do
    stem=$(basename "$f" .json)
    # Skip combined/consensus files
    case "$stem" in
        results|consensus_*) continue ;;
    esac
    cp "$f" "$HF_SPACE_DIR/results/"
    echo "  ✓ $(basename "$f")"
done

echo ""

# ── 2. Push to HF ───────────────────────────────────────────────────
echo "🚀 Pushing to https://huggingface.co/spaces/$SPACE ..."
python -c "
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(
    folder_path='$HF_SPACE_DIR',
    repo_id='$SPACE',
    repo_type='space',
)
print('✅ Deploy complete!')
print(f'   https://huggingface.co/spaces/$SPACE')
"
