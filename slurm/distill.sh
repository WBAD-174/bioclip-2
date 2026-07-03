#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --account=PAS2136
#SBATCH --gpus-per-node=2
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu
#SBATCH --job-name=bioclip-distill
#SBATCH --time=48:00:00
#SBATCH --mem=200GB

module load miniconda3/24.1.2-py310
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate bioclip

echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "
echo "Nodelist:= " $SLURM_JOB_NODELIST
echo "Number of nodes:= " $SLURM_JOB_NUM_NODES
echo "Ntasks per node:= "  $SLURM_NTASKS_PER_NODE
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "

host_node=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
echo $host_node

export RDZV_HOST=$host_node
export RDZV_PORT=29400

srun torchrun --nnodes=1 --nproc_per_node 2 \
  --rdzv_id=$RANDOM --rdzv_backend=c10d --rdzv_endpoint=$RDZV_HOST:$RDZV_PORT \
  -m src.training.main \
  --model ViT-B-16-1024 \
  --pretrained '/fs/scratch/PAS2136/bioclip-distillation/10M/student_warm_start.pt' \
  --distill-model 'hf-hub:imageomics/bioclip-2.5-vith14' \
  --distill-pretrained 'unused' \
  --teacher-embed-dir '/fs/scratch/PAS2136/bioclip-distillation/10M/teacher-embeddings' \
  --train-data '/fs/scratch/PAS2136/bioclip-distillation/10M/shards/shard-{00000..01001}.tar' \
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
  --logs-dir './logs' \
  --precision amp \
