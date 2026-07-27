#!/usr/bin/env bash
#SBATCH --nodes=2
#SBATCH --account=PAS2136
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu
#SBATCH --job-name=bioclip-exp1-no-distill
#SBATCH --time=48:00:00
#SBATCH --mem=800GB

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

# Back to 2 nodes (2026-07-27, user-confirmed working on Cardinal after the earlier
# single-node-workaround comment below was written) -- needs the multi-node rdzv setup
# again, --standalone won't work across nodes.
host_node=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
echo $host_node

export RDZV_HOST=$host_node
export RDZV_PORT=29400

# Controlled ablation, Exp1: reproduce original BioCLIP -- OpenAI-CLIP-init ViT-B/16,
# plain contrastive training on TOL-10M (evobio10m-v3.3), NO distillation, NO LAION
# replay. Exp2 (slurm/exp2_distill.sh) is this exact same config with only the
# --distill-* / --teacher-embed-dir lines added -- diff the two files to confirm
# nothing else differs (seed, lr, batch size, epochs, augmentation all identical).
srun torchrun --nnodes=2 --nproc_per_node 4 \
  --rdzv_id=$RANDOM --rdzv_backend=c10d --rdzv_endpoint=$RDZV_HOST:$RDZV_PORT \
  -m src.training.main \
  --name 'exp1-no-distill-evobio10m' \
  --model ViT-B-16 \
  --pretrained 'openai' \
  --resume '/fs/scratch/PAS2136/chenxujiang/bioclip-ablation-logs/exp1-no-distill-evobio10m/checkpoints/epoch_15.pt' \
  --train-data '/fs/ess/PAS2136/open_clip/data/evobio10m-v3.3/224x224/train/shard-{000000..000159}.tar' \
  --val-data '/fs/ess/PAS2136/open_clip/data/evobio10m-v3.3/224x224/val/shard-{000000..000064}.tar' \
  --dataset-type 'webdataset' \
  --dataset-resampled \
  --save-frequency 1 \
  --warmup 1000 \
  --batch-size 4096 \
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
  --logs-dir '/fs/scratch/PAS2136/chenxujiang/bioclip-ablation-logs' \
  --precision pure_bf16 \
  --torchcompile \