#!/usr/bin/env bash
#SBATCH --nodes=2
#SBATCH --account=PAS2136
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu
#SBATCH --job-name=bioclip-exp3-distill-tuned
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

host_node=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
echo $host_node

export RDZV_HOST=$host_node
export RDZV_PORT=29400

# Exp3: same config as slurm/exp2_distill.sh (same init, data, seed, lr, batch-size,
# epochs, augmentation) but with the new --distill-temperature / --distill-loss-weight
# knobs (src/open_clip/loss.py DistillClipLoss, added for this experiment) turned on --
# these 2 lines are the sole diff from Exp2. Diff the two files to confirm.
#
# T=2.0: softens both teacher and student logits before the KD softmax/log_softmax
# (Hinton et al. 2015 style; loss is rescaled by T^2 to keep gradient magnitude
# comparable to T=1). Previously the KD loss ran directly on each model's own
# contrastively-learned logit_scale (~14-100, i.e. effectively T=1, very peaked) --
# T=2 is a conservative first step to let more of the teacher's non-argmax signal
# through instead of an almost one-hot target.
#
# distill-loss-weight=0.5: previously contrastive_loss and distill_loss were summed
# 1:1 (both implicitly weight 1.0, see src/open_clip/loss.py's old DistillClipLoss).
# Halving distill_loss's weight is a first step toward treating distillation as a
# regularizer on top of the main contrastive objective rather than an equal co-task,
# given eval showed Exp2 already skews toward the teacher's (cleaner-image-domain)
# behavior at the cost of a small intra-species-orthogonality regression on
# camera-trap benchmarks -- see the exp1-vs-exp2 ablation eval discussion.
#
# NOTE ON COMPARISON: unlike exp1/exp2 (which both got --resume'd mid-training after a
# batch-size 256->4096 / node-count change, see those scripts' git history), this run
# starts fresh at the final batch-size/node config for the full 30 epochs -- no
# mid-training regime switch. That makes it directly comparable to itself, but means
# it is NOT trained under the identical step-by-step history as exp1/exp2's checkpoints
# (only same total epochs/data/seed/hyperparameters). It also trains on the 1M-sample
# evobio10m-v3.3/224x224/train_small shard set (64 shards) instead of the full 10M
# train/ set exp1/exp2 used, as a faster first pass at this hyperparameter direction --
# not a like-for-like data comparison, see conversation notes before promoting any
# result here to the main ablation table.
#
# NOTE ON TEACHER EMBEDDINGS: exp1/exp2 read precomputed frozen teacher image
# embeddings from --teacher-embed-dir, matched to --train-data shards by basename (see
# src/training/data.py's add_teacher_embed_url/tarfile_pairs_to_samples_nothrow). That
# precomputed dir was built from the full train/ shards, not train_small/ -- if
# train_small isn't byte-identical to the first 64 shards of train/ (e.g. it's a
# resampled repackaging), the basename-matched embeddings would mostly miss on sample
# key and get silently dropped (safely, not misaligned -- but that could starve this
# run of most of its distillation signal without erroring). --teacher-embed-dir is
# intentionally omitted below so the teacher's image tower runs live on train_small's
# actual images instead, trading some extra per-step compute for guaranteed
# correctness. Switch back to a --teacher-embed-dir once/if a train_small-specific
# embedding dir is precomputed via slurm/precompute_teacher_embeddings.sh.
#
# NOTE ON --warmup: exp1/exp2's --warmup 1000 was sized for the 10M train/ set's
# ~9150 total steps (global_batch_size = batch-size(4096) * world_size(8) = 32768;
# ~305 steps/epoch * 30 epochs). train_small has ~1M samples -> ~24-32 steps/epoch
# (see src/training/data.py:534-541's per-worker batch chunking) -> ~700-960 total
# steps for the same 30 epochs, which is LESS than a 1000-step warmup -- LR would
# still be ramping up linearly when training ends, never reaching peak or entering
# cosine decay. Lowered to 100 (roughly the same ~11% warmup fraction as exp1/exp2:
# 1000/9150 approx 100/900).
srun torchrun --nnodes=2 --nproc_per_node 4 \
  --rdzv_id=$RANDOM --rdzv_backend=c10d --rdzv_endpoint=$RDZV_HOST:$RDZV_PORT \
  -m src.training.main \
  --name 'exp3-distill-tuned-evobio10m' \
  --model ViT-B-16 \
  --pretrained 'openai' \
  --distill-model 'hf-hub:imageomics/bioclip-2.5-vith14' \
  --distill-pretrained 'unused' \
  --distill-temperature 2.0 \
  --distill-loss-weight 0.5 \
  --train-data '/fs/ess/PAS2136/open_clip/data/evobio10m-v3.3/224x224/train_small/shard-{000000..000063}.tar' \
  --val-data '/fs/ess/PAS2136/open_clip/data/evobio10m-v3.3/224x224/val/shard-{000000..000064}.tar' \
  --dataset-type 'webdataset' \
  --dataset-resampled \
  --save-frequency 1 \
  --warmup 100 \
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
