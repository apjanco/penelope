#!/bin/bash
# ============================================================================
# Penelope SoC fine-tuning — Princeton Della SLURM batch script
#
# Submit with:
#   sbatch train_della.slurm
#
# Monitor with:
#   squeue -u $USER
#   tail -f logs/train_<JOBID>.out
# ============================================================================

#SBATCH --job-name=penelope-soc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8          # 4 for DataLoader workers + headroom
#SBATCH --mem=64G                  # 4-bit model ~6 GB GPU, ~20 GB CPU overhead
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80         # A100 80 GB gives ample room for seq_len=4096
                                   # Change to gpu40 if 80 GB nodes are busy;
                                   # both work for QLoRA Qwen3-4B.
#SBATCH --time=08:00:00            # 3 epochs on ~500 examples typically <2 h
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=                # fill in your Princeton email

# ── Environment ─────────────────────────────────────────────────────────────
module purge
module load anaconda3/2025.12       # check available: module avail anaconda3
module load cudatoolkit/13.2        # A100 compute capability 8.0

conda activate penelope             # see setup instructions below

export HF_HOME=/scratch/gpfs/MM4/apjanco/.cache/huggingface
export HF_HUB_OFFLINE=1            # no network calls on compute nodes

# ── Paths ────────────────────────────────────────────────────────────────────
# Use /scratch/gpfs for all I/O (fast parallel filesystem; NOT /home).
# Example: copy project to scratch before submitting, then point paths there.
PROJECT_DIR="/scratch/gpfs/MM4/apjanco/penelope"
DATASET_DIR="$PROJECT_DIR/dataset"
ADAPTER_DIR="$PROJECT_DIR/adapter"

cd "$PROJECT_DIR" || { echo "Project dir not found: $PROJECT_DIR"; exit 1; }
mkdir -p logs

# ── Diagnostics ──────────────────────────────────────────────────────────────
echo "Job ID     : $SLURM_JOB_ID"
echo "Node       : $SLURMD_NODENAME"
echo "GPUs       : $CUDA_VISIBLE_DEVICES"
echo "Working dir: $PWD"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python -c "import torch; print('PyTorch', torch.__version__, '| CUDA', torch.version.cuda, '| bf16 support:', torch.cuda.is_bf16_supported())"

# ── Training ──────────────────────────────────────────────────────────────────
python scripts/train.py \
    --dataset         "$DATASET_DIR" \
    --base-model      Qwen/Qwen3-4B \
    --adapter-output  "$ADAPTER_DIR" \
    --epochs          3 \
    --batch-size      2 \
    --grad-accum      8 \
    --lr              2e-4 \
    --lora-rank       64 \
    --max-seq-len     4096 \
    --flash-attn \
    --dataloader-workers 4

# ── Post-training: merge and export ──────────────────────────────────────────
# Uncomment to automatically merge and push after training:
#
# python scripts/export_model.py \
#     --base-model  Qwen/Qwen3-4B \
#     --adapter     "$ADAPTER_DIR" \
#     --output      "$PROJECT_DIR/models/penelope-soc-v1" \
#     --push-to-hub \
#     --hub-id      apjanco/penelope-soc-v1

echo "Done."

# ============================================================================
# SETUP INSTRUCTIONS (run once, interactively on della-gpu login node)
# ============================================================================
#
# 1. Transfer project to scratch:
#    scp -r ~/projects/penelope $USER@della-gpu.princeton.edu:/scratch/gpfs/$USER/
#
# 2. Connect to Della GPU login node:
#    ssh $USER@della-gpu.princeton.edu
#
# 3. Create conda environment:
#    module load anaconda3/2024.6
#    conda create -n penelope python=3.11 -y
#    conda activate penelope
#
# 4. Install PyTorch with CUDA 12:
#    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
#
# 5. Install remaining dependencies:
#    cd /scratch/gpfs/$USER/penelope
#    pip install -r requirements.txt
#
# 6. Install flash-attn (takes ~10 min to compile; recommended for A100):
#    pip install flash-attn --no-build-isolation
#
# 7. (Optional) Pre-download the base model to avoid HF Hub timeouts in the job:
#    python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; \
#               AutoTokenizer.from_pretrained('Qwen/Qwen3-4B'); \
#               AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-4B')"
#
# 8. Build the dataset (can also be done interactively):
#    python scripts/build_dataset.py \
#        --silver-dir training_data/ \
#        --neg-dir /path/to/gutenberg/ \
#        --output dataset/
#
# 9. Submit the job:
#    sbatch train_della.slurm
