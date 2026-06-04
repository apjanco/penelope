#!/bin/bash
# ============================================================================
# Penelope — merge LoRA adapter and push to HuggingFace Hub
#
# Submit with:
#   sbatch export_della.slurm
#
# Monitor with:
#   squeue -u $USER
#   tail -f logs/export_<JOBID>.out
# ============================================================================

#SBATCH --job-name=penelope-export
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G                  # merging Qwen3-4B bf16 needs ~16 GB RAM
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --time=01:00:00
#SBATCH --output=logs/export_%j.out
#SBATCH --error=logs/export_%j.err
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=               # fill in your Princeton email

# ── Environment ─────────────────────────────────────────────────────────────
module purge
module load anaconda3/2025.12
module load cudatoolkit/13.2

conda activate penelope

export HF_HOME=/scratch/gpfs/MM4/apjanco/.cache/huggingface
# Leave HF_HUB_OFFLINE unset so push-to-hub can reach the internet

PROJECT_DIR=/scratch/gpfs/MM4/apjanco/penelope

cd "$PROJECT_DIR"

# ── Export ───────────────────────────────────────────────────────────────────
python scripts/export_model.py \
    --base-model  Qwen/Qwen3-4B \
    --adapter     "$PROJECT_DIR/adapter/" \
    --output      "$PROJECT_DIR/models/penelope-soc-v1"

echo "Done."
