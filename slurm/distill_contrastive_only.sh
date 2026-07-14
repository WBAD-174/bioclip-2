#!/usr/bin/env bash
#SBATCH --nodes=2
#SBATCH --account=PAS2136
#SBATCH --gpus-per-node=2
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu
#SBATCH --job-name=bioclip-distill-ctrl
#SBATCH --time=48:00:00
#SBATCH --mem=200GB

# Control run for the Hypothesis-1 disentangling question (see
# Hypothesis1_Geometry_Findings.md, "What's not yet checked" -> "Disentangle
# data-scale/coverage from the distillation objective"). Identical to distill.sh
# (same --model, same warm start, same 10M shards, same 30 epochs, same seed) with the
# --distill-model/--distill-pretrained/--teacher-embed-dir args removed, so
# args.distill = args.distill_model is not None and args.distill_pretrained is not None
# (src/training/main.py:208) evaluates False and training falls back to plain
# contrastive loss against the ground-truth taxonomic text labels only -- no teacher
# supervision at all. This isolates: does training on the same 10M sample without any
# distillation signal already show the same inter-species accuracy drop / intra-species
# orthogonality preservation as the logit-KD student? If yes, that's a data-scale
# effect, not a distillation-objective effect.
#
# Deliberately NOT passing --resume (unlike distill.sh, which resumes a specific
# mid-training KD checkpoint) -- this is a fresh 30-epoch run from the same warm start.
#
# After this finishes, add a third entry to slurm/eval_hypothesis1.sh's
# MODEL_NAMES/MODEL_TYPES/PRETRAINED_PATHS arrays (e.g. name "contrastive_only", type
# "ViT-B-16-1024", pretrained_path pointing at this run's final epoch checkpoint under
# --logs-dir below), then re-run it and
# src/evaluation/summarize_hypothesis1_classification.py (already generalized to any
# number of model subdirectories under --logs-root).

module load miniconda3/24.1.2-py310
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate bioclip

# -m src.xxx needs the repo root as CWD. Not using $0-based resolution: SLURM may run
# a spooled copy of this script, so $0 does not reliably point back into the repo.
cd /users/PAS2136/chenxujiang/bioclip-2/bioclip-2

echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "
echo "Nodelist:= " $SLURM_JOB_NODELIST
echo "Number of nodes:= " $SLURM_JOB_NUM_NODES
echo "Ntasks per node:= "  $SLURM_NTASKS_PER_NODE
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "

host_node=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
echo $host_node

export RDZV_HOST=$host_node
export RDZV_PORT=29400

srun torchrun --nnodes=2 --nproc_per_node 2 \
  --rdzv_id=$RANDOM --rdzv_backend=c10d --rdzv_endpoint=$RDZV_HOST:$RDZV_PORT \
  -m src.training.main \
  --model ViT-B-16-1024 \
  --pretrained '/fs/scratch/PAS2136/chenxujiang/student_warm_start.pt' \
  --train-data '/fs/scratch/PAS2136/bioclip-distillation/10M/shards/shard-{00000..01001}.tar' \
  --train-num-samples 10000000 \
  --dataset-type 'webdataset' \
  --dataset-resampled \
  --save-frequency 1 \
  --warmup 1000 \
  --batch-size 256 \
  --accum-freq 1 \
  --epochs 30 \
  --workers 8 \
  --text_type 'random' \
  --log-every-n-steps 1 \
  --lr 1e-4 \
  --seed 42 \
  --local-loss \
  --gather-with-grad \
  --grad-checkpointing \
  --logs-dir '/fs/scratch/PAS2136/chenxujiang/bioclip-distill-contrastive-only-logs' \
  --precision amp \
