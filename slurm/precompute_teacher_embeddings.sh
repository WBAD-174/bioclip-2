#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --account=PAS2136
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu
#SBATCH --job-name=bioclip-teacher-embed
#SBATCH --time=48:00:00
#SBATCH --mem=200GB

module load miniconda3/24.1.2-py310
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate bioclip

# -m src.xxx needs the repo root as CWD, regardless of where sbatch was invoked from.
cd "$(dirname "$(readlink -f "$0")")/.."

echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "
echo "Nodelist:= " $SLURM_JOB_NODELIST
echo "Number of nodes:= " $SLURM_JOB_NUM_NODES
echo "Ntasks per node:= "  $SLURM_NTASKS_PER_NODE
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "

host_node=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
echo $host_node

export RDZV_HOST=$host_node
export RDZV_PORT=29400

# Embarrassingly parallel across ranks (whole shards per rank, no gradient sync needed), so this
# scales cleanly by bumping --gpus-per-node / --nodes if it's too slow on one node.
srun torchrun --nnodes=1 --nproc_per_node 4 \
  --rdzv_id=$RANDOM --rdzv_backend=c10d --rdzv_endpoint=$RDZV_HOST:$RDZV_PORT \
  -m src.training.precompute_teacher_embeddings \
  --teacher-model 'hf-hub:imageomics/bioclip-2.5-vith14' \
  --input-data '/fs/scratch/PAS2136/bioclip-distillation/10M/shards/shard-{00000..01001}.tar' \
  --output-dir '/fs/scratch/PAS2136/bioclip-distillation/10M/teacher-embeddings' \
  --batch-size 512 \
  --workers 2 \
  --precision amp \
