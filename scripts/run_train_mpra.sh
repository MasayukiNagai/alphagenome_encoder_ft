#!/bin/bash
#SBATCH --job-name=train_ag_mpra
#SBATCH --output=out/%x_%j.log
#SBATCH --error=out/%x_%j.log
#SBATCH --export=ALL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-gpu=10
#SBATCH --mem-per-gpu=128G
#SBATCH --partition=gpuq
#SBATCH --qos=bio_ai
#SBATCH --time=48:00:00

if command -v job_notify_slurm >/dev/null 2>&1; then
  source "$(command -v job_notify_slurm)"
  notify_job_start || true
fi

set -ex

REPO_ROOT="/grid/koo/home/nagai/projects/ag_mpra_torch/alphagenome-encoder-ft"
cd "$REPO_ROOT"

PYTHON="$REPO_ROOT/.venv/bin/python"
script="$REPO_ROOT/scripts/train_mpra.py"
celltype=${1:-"K562"}
config="$REPO_ROOT/configs/lentimpra_${celltype}.json"

input_tsv="/grid/koo/home/shared/data/lentimpra/agarwal_2025/${celltype}.tsv"
pretrained_weights="/grid/koo/home/shared/models/alphagenome_pytorch/model_all_folds.safetensors"

timestamp=$(date +"%m%d_%H%M")
wandb_name="mpra_${celltype}_${timestamp}"
checkpoint_dir="./results/mpra_${celltype}_${timestamp}"
mkdir -p "$checkpoint_dir"

cmd="CUDA_VISIBLE_DEVICES=0 \
  $PYTHON $script \
  --config $config \
  --input_tsv $input_tsv \
  --pretrained_weights $pretrained_weights \
  --wandb_name $wandb_name \
  --checkpoint_dir $checkpoint_dir"

echo "Running command:"
echo "$cmd"
eval "$cmd"
