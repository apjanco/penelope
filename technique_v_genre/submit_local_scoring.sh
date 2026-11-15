#!/bin/bash
# Submit the three self-contained per-model full-corpus scoring jobs that
# replace the cancelled serve_vllm_multi.slurm (job 12296047). Each is a
# separate, small gpu-short job (server + scoring wrapped together) so they
# backfill independently instead of waiting behind one 4-GPU/72h reservation.
#
# gpu-short walltime cap is 24h. Walltimes below are generous first guesses;
# scoring is resumable, so if one hits its limit just re-run the same sbatch
# and it continues from where it stopped. Llama is gated + needs 2 GPUs and
# its weights pre-cached (HF_HUB_OFFLINE=1).
#
# Usage:  bash technique_v_genre/submit_local_scoring.sh
set -euo pipefail
cd /scratch/gpfs/MM4/apjanco/penelope

sbatch --job-name=score-qwen3-reasoning --gres=gpu:1 --time=16:00:00 \
  --export=ALL,LABEL=qwen3-reasoning,HFMODEL=Qwen/Qwen3.6-27B,TP=1,MAXLEN=8192 \
  technique_v_genre/serve_score_model.slurm

sbatch --job-name=score-mistral --gres=gpu:1 --time=16:00:00 \
  --export=ALL,LABEL=mistral,HFMODEL=mistralai/Mistral-Small-24B-Instruct-2501,TP=1,MAXLEN=8192 \
  technique_v_genre/serve_score_model.slurm

sbatch --job-name=score-llama --gres=gpu:2 --time=24:00:00 \
  --export=ALL,LABEL=llama,HFMODEL=meta-llama/Llama-3.3-70B-Instruct,TP=2,MAXLEN=4096 \
  technique_v_genre/serve_score_model.slurm
