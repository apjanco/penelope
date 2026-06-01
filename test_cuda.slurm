#!/bin/bash
# ============================================================================
# CUDA toolkit compatibility probe
# Loads each available toolkit in turn and reports what PyTorch sees.
#
# Submit with:
#   sbatch test_cuda.slurm
#
# Results in: logs/cuda_probe_<JOBID>.out
# ============================================================================

#SBATCH --job-name=cuda-probe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80          # or gpu40 — change to whichever is free
#SBATCH --time=00:15:00
#SBATCH --output=logs/cuda_probe_%j.out
#SBATCH --error=logs/cuda_probe_%j.err

mkdir -p logs

echo "================================================================"
echo "Node         : $SLURMD_NODENAME"
echo "CUDA_VISIBLE : $CUDA_VISIBLE_DEVICES"
echo "================================================================"
echo ""

# -- GPU hardware info (driver version determines max CUDA) ------------------
echo ">>> nvidia-smi"
nvidia-smi --query-gpu=name,driver_version,compute_cap,memory.total \
           --format=csv,noheader
echo ""

# Driver version -> max CUDA support:
#   >= 525  -> CUDA 12.x
#   >= 545  -> CUDA 12.3+
#   >= 560  -> CUDA 12.6+
#   >= 565  -> CUDA 12.7+
#   >= 570  -> CUDA 12.8+
#   >= 576  -> CUDA 13.x (preview)
DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | tr -d ' ')
COMPUTE=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' ')
echo "Driver version  : $DRIVER"
echo "Compute cap     : $COMPUTE"
echo ""

module purge
module load anaconda3/2025.12
conda activate penelope

# -- Test each available toolkit ---------------------------------------------
TOOLKITS=(11.8 12.6 12.8 12.9 13.0 13.1 13.2)

echo "================================================================"
echo "PyTorch CUDA compatibility sweep"
echo "================================================================"
BEST_KIT=""
for KIT in "${TOOLKITS[@]}"; do
    echo ""
    echo "--- cudatoolkit/$KIT ---"
    module load "cudatoolkit/$KIT" 2>/dev/null || { echo "  [SKIP] module not found"; continue; }

    python - <<'PYEOF'
import sys, torch
ok     = torch.cuda.is_available()
ver    = torch.version.cuda
dev    = torch.cuda.get_device_name(0) if ok else "N/A"
mem    = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1) if ok else 0
bf16   = torch.cuda.is_bf16_supported()          if ok else False
fp8    = hasattr(torch, 'float8_e4m3fn')
tf32   = torch.backends.cuda.matmul.allow_tf32   if ok else False

print(f"  torch        : {torch.__version__}")
print(f"  CUDA (torch) : {ver}")
print(f"  CUDA avail   : {ok}")
print(f"  Device       : {dev}")
print(f"  VRAM         : {mem} GB")
print(f"  bf16 support : {bf16}")
print(f"  fp8 avail    : {fp8}")
print(f"  tf32 allowed : {tf32}")
if ok:
    print("  [PASS] GPU is usable with this toolkit")
else:
    print("  [FAIL] CUDA not available — toolkit/driver mismatch")
    sys.exit(1)
PYEOF

    STATUS=$?
    if [[ $STATUS -eq 0 ]]; then
        BEST_KIT=$KIT
    fi
    module unload "cudatoolkit/$KIT" 2>/dev/null
done

echo ""
echo "================================================================"
if [[ -n "$BEST_KIT" ]]; then
    echo "RECOMMENDATION: module load cudatoolkit/$BEST_KIT"
    echo "(highest-versioned toolkit that passed)"
else
    echo "WARNING: No toolkit passed — check driver/PyTorch versions."
fi
echo "================================================================"
