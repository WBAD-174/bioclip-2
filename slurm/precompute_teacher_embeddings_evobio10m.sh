#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --account=PAS2136
#SBATCH --gpus-per-node=2
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu
#SBATCH --job-name=bioclip-teacher-embed-evobio10m
#SBATCH --time=12:00:00
#SBATCH --mem=200GB

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

# Frozen teacher (BioCLIP 2.5) image embeddings for the ORIGINAL evobio10m-v3.3 TOL-10M
# shards (ablation-study data, distinct from the earlier Matt-provided 10M subset --
# output goes to its own directory so the earlier embeddings stay untouched).
#
# Embarrassingly parallel across ranks (whole shards per rank, no gradient sync needed), so this
# scales cleanly by bumping --gpus-per-node / --nodes if it's too slow on one node.
srun torchrun --nnodes=1 --nproc_per_node 2 \
  --rdzv_id=$RANDOM --rdzv_backend=c10d --rdzv_endpoint=$RDZV_HOST:$RDZV_PORT \
  -m src.training.precompute_teacher_embeddings \
  --teacher-model 'hf-hub:imageomics/bioclip-2.5-vith14' \
  --input-data '/fs/ess/PAS2136/open_clip/data/evobio10m-v3.3/224x224/train/shard-{000000..000159}.tar' \
  --output-dir '/fs/scratch/PAS2136/chenxujiang/bioclip-teacher-embeddings-evobio10m' \
  --batch-size 512 \
  --workers 2 \
  --precision amp \
