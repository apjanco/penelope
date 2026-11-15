#!/bin/bash
# ============================================================================
# Penelope — v3 ambiguity demo batch runner
#
# Submit with:
#   sbatch v3_ambiguity_demo.slurm
#
# Monitor with:
#   tail -f logs/v3_ambiguity_demo_<JOBID>.out
# ============================================================================

#SBATCH --job-name=penelope-v3-demo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --time=02:00:00
#SBATCH --qos=gpu-short
#SBATCH --output=logs/v3_ambiguity_demo_%j.out
#SBATCH --error=logs/v3_ambiguity_demo_%j.err
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=

module purge
module load anaconda3/2025.12
module load cudatoolkit/13.2

conda activate penelope

export HF_HOME=/scratch/gpfs/MM4/apjanco/.cache/huggingface
export HF_HUB_OFFLINE=1

PROJECT_DIR="/scratch/gpfs/MM4/apjanco/penelope"
OUTPUT_DIR="$PROJECT_DIR/outputs/notebook_demo"
RUN_TAG="slurm_${SLURM_JOB_ID}"
RUN_LOG="$OUTPUT_DIR/${RUN_TAG}_stdout.log"

cd "$PROJECT_DIR" || { echo "Project dir not found: $PROJECT_DIR"; exit 1; }
mkdir -p logs "$OUTPUT_DIR"

echo "Job ID     : $SLURM_JOB_ID"
echo "Node       : $SLURMD_NODENAME"
echo "GPUs       : $CUDA_VISIBLE_DEVICES"
echo "Working dir: $PWD"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python -c "import torch; print('PyTorch', torch.__version__, '| CUDA', torch.version.cuda, '| bf16:', torch.cuda.is_bf16_supported())"

python scripts/v3_ambiguity_demo.py \
    --output-dir "$OUTPUT_DIR" \
    --tag "$RUN_TAG" \
    2>&1 | tee "$RUN_LOG"

echo "Saved stdout log to: $RUN_LOG"