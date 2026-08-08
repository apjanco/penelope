 ---
description: "Use when writing SLURM scripts, configuring jobs for Della (Princeton HPC), managing conda environments, or debugging queue/scheduling issues. Covers offline model loading, fairshare, GPU constraints, and Anaconda best practices."
applyTo: "**/*.slurm"
---

# Della HPC — Good Citizen Practices

## Compute nodes have NO internet access

This is the single most common gotcha. Any code that touches HuggingFace Hub,
pip, or external APIs **will fail** on compute nodes. Always set:

```bash
export HF_HOME="/scratch/gpfs/MM4/apjanco/.cache/huggingface"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

Pre-download models and datasets on a **login node** before submitting:

```bash
# On login node (has internet):
python -c "from transformers import AutoProcessor; AutoProcessor.from_pretrained('Qwen/Qwen3-VL-8B-Instruct')"
```

If using `datasets`, also cache on the login node first:

```bash
python -c "from datasets import load_dataset; load_dataset('Teklia/IAM-line')"
```

## Anaconda on Della

```bash
module purge
module load anaconda3/2025.12    # check `module avail anaconda3` for versions
conda activate <env_name>        # envs live in ~/.conda/envs/ or scratch
```

Key points:
- Base environment is **read-only** — always create your own env
- Create envs on scratch for large installs: `conda create -p /scratch/gpfs/MM4/apjanco/envs/<name> python=3.11`
- Install GPU packages (torch, etc.) on login node **before** submitting jobs
- Pin `cudatoolkit` module to match your env's CUDA build (check `module avail cudatoolkit`)
- Current envs: `churro_exp`, `penelope`, `htr2hpc`, `nara_trainer`

## GPU scheduling — avoid long queues

The `mm4` account fairshare depletes quickly because usage is concentrated on
one user. Key levers to reduce wait time:

### 1. Relax the constraint

```bash
# BAD — restricts to A100 nodes only, highest demand
#SBATCH --constraint=a100

# BETTER — any 80GB GPU (targets della-l02g* 4×A100-80GB nodes)
#SBATCH --constraint=gpu80

# BEST for small models — accept any GPU node in the partition
# (omit --constraint entirely)
```

### 2. Request only what you need

| Resource | Guideline |
|----------|-----------|
| `--time` | Set to 1.5× expected runtime, not 4h for a 40min job. Shorter jobs backfill into gaps. |
| `--mem`  | 64G is fine for 8B models; don't request 128G+ unless proven OOM. |
| `--gres=gpu:1` | Never request more GPUs than your code actually uses. |
| `--cpus-per-task` | 4 is usually enough for dataloader workers. |

### 3. Use `gputest` for iteration

```bash
#SBATCH --partition=gputest
#SBATCH --time=00:30:00
#SBATCH --constraint=gpu80
```

Jobs start in **seconds** on `gputest` (vs hours on `gpu`). Use for:
- Smoke tests, debugging, validating model loads
- Max 30 minutes walltime

### 4. QOS tiers

| QOS | Max walltime | Max GPUs/user | Use for |
|-----|-------------|---------------|---------|
| `gpu-short` | 1 day | 44 | Standard runs |
| `gpu-medium` | 3 days | 20 | Multi-day training |
| `gpu-long` | 6 days | 16 | Long fine-tuning |
| `gpu-test` | (unlimited) | (unlimited) | Quick debugging |

### 5. Monitor fairshare

```bash
sshare -A mm4 -a          # check account fairshare status
sprio -u $USER            # see your job's priority breakdown
squeue -u $USER           # check pending reason (Priority, Resources, etc.)
```

### 6. Modify a queued job in place

Use `scontrol update` to tweak a pending job without cancel/resubmit:

```bash
scontrol update JobId=<id> TimeLimit=01:00:00
scontrol update JobId=<id> NumCPUs=4
scontrol update JobId=<id> MinMemoryNode=32G
```

**Can change** (while pending):
- `TimeLimit` — **reduce only**, cannot increase past original request
- `NumCPUs` / `MinCPUsNode` — reduce CPUs
- `MinMemoryNode` — reduce memory
- `Partition` — move to a different partition (e.g. `gputest`)
- `Name` — rename the job
- `Dependency` — add/remove dependencies
- `ExcNodeList` / `Features` — change node preferences/constraints
- `Nice` — lower your own priority to let other jobs pass

**Cannot change** (must cancel and resubmit):
- `TimeLimit` increase (admin-only)
- `Gres` (GPU count/type)
- `Account` / `QOS`
- The submission script itself (command, env vars, paths)

Reducing time is the highest-impact change: a 1h job backfills into gaps
that a 4h job never fits.

## Node types on Della GPU partition

| Nodes | CPUs | GPU | VRAM | Constraint |
|-------|------|-----|------|-----------|
| della-i14g1–20 | AMD Rome | 2× A100 PCIe | 40GB | `a100,gpu40` |
| della-l01g* | Intel Icelake | 8× A100 MIG slices | 40GB | `a100,gpu40` |
| della-l02g* | Intel Icelake | 4× A100 PCIe | 80GB | `a100,gpu80` |

## SLURM script template

```bash
#!/bin/bash
#SBATCH --job-name=<name>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --account=mm4
#SBATCH --constraint=gpu80
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

module purge
module load anaconda3/2025.12
module load cudatoolkit/12.9
conda activate churro_exp

export HF_HOME="/scratch/gpfs/MM4/apjanco/.cache/huggingface"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs

python your_script.py --resume
```
