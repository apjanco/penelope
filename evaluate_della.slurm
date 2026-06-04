#!/bin/bash
# ============================================================================
# Penelope SoC evaluation — Princeton Della SLURM batch script
#
# Submit with:
#   sbatch evaluate_della.slurm
#
# Monitor with:
#   squeue -u $USER
#   tail -f logs/eval_<JOBID>.out
#
# Pre-download model files (run once interactively before submitting):
#   python -c "
#   from transformers import AutoModelForCausalLM, AutoTokenizer
#   AutoTokenizer.from_pretrained('Qwen/Qwen3-4B')
#   AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-4B', torch_dtype='auto')
#   "
# ============================================================================

#SBATCH --job-name=penelope-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80          # A100 80 GB; change to gpu40 if needed
#SBATCH --time=05:00:00
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=                 # fill in your Princeton email

# ── Environment ─────────────────────────────────────────────────────────────
module purge
module load anaconda3/2025.12
module load cudatoolkit/13.2

conda activate penelope

export HF_HOME=/scratch/gpfs/MM4/apjanco/.cache/huggingface
export HF_HUB_OFFLINE=1             # no network calls on compute nodes

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_DIR="/scratch/gpfs/MM4/apjanco/penelope"
DATASET_DIR="$PROJECT_DIR/dataset"
ADAPTER_DIR="$PROJECT_DIR/adapter"
OUTPUT="$PROJECT_DIR/logs/eval_${SLURM_JOB_ID}_results.json"

cd "$PROJECT_DIR" || { echo "Project dir not found: $PROJECT_DIR"; exit 1; }
mkdir -p logs

# ── Diagnostics ──────────────────────────────────────────────────────────────
echo "Job ID     : $SLURM_JOB_ID"
echo "Node       : $SLURMD_NODENAME"
echo "GPUs       : $CUDA_VISIBLE_DEVICES"
echo "Working dir: $PWD"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python -c "import torch; print('PyTorch', torch.__version__, '| CUDA', torch.version.cuda, '| bf16 support:', torch.cuda.is_bf16_supported())"

# ── Evaluation ────────────────────────────────────────────────────────────────
python scripts/evaluate.py \
    --adapter     "$ADAPTER_DIR" \
    --base-model  Qwen/Qwen3-4B \
    --dataset     "$DATASET_DIR" \
    --batch-size  4 \
    --output      "$OUTPUT"

echo "Results written to $OUTPUT"
echo "Done."
