#!/bin/bash
# ============================================================================
# Penelope — quick checkpoint smoke-test on a GPU node
#
# Submit with:
#   sbatch test_checkpoint.slurm
#
# Monitor with:
#   squeue -u $USER
#   tail -f logs/test_ckpt_<JOBID>.out
# ============================================================================

#SBATCH --job-name=penelope-ckpt-test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80          # A100 80 GB; change to gpu40 if needed
#SBATCH --time=00:30:00
#SBATCH --output=logs/test_ckpt_%j.out
#SBATCH --error=logs/test_ckpt_%j.err
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=                # fill in your Princeton email

# ── Environment ─────────────────────────────────────────────────────────────
module purge
module load anaconda3/2025.12
module load cudatoolkit/13.2

conda activate penelope

export HF_HOME=/scratch/gpfs/MM4/apjanco/.cache/huggingface
export HF_HUB_OFFLINE=1            # no network calls on compute nodes

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_DIR="/scratch/gpfs/MM4/apjanco/penelope"
CHECKPOINT="$PROJECT_DIR/adapter-grpo/checkpoints/checkpoint-4100"

cd "$PROJECT_DIR" || { echo "Project dir not found: $PROJECT_DIR"; exit 1; }
mkdir -p logs

# ── Diagnostics ──────────────────────────────────────────────────────────────
echo "Job ID     : $SLURM_JOB_ID"
echo "Node       : $SLURMD_NODENAME"
echo "GPUs       : $CUDA_VISIBLE_DEVICES"
echo "Working dir: $PWD"
echo "Checkpoint : $CHECKPOINT"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python -c "import torch; print('PyTorch', torch.__version__, '| CUDA', torch.version.cuda, '| bf16:', torch.cuda.is_bf16_supported())"

# ── Test ─────────────────────────────────────────────────────────────────────
python test_checkpoint.py \
    --checkpoint "$CHECKPOINT" \
    --device     auto

echo "Done."
