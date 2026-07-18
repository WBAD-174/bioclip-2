#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --account=PAS2136
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --job-name=bioclip-eval-ablation
#SBATCH --time=8:00:00
#SBATCH --mem=400GB

module load miniconda3/24.1.2-py310
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate bioclip

cd /users/PAS2136/chenxujiang/bioclip-2/bioclip-2

export CUDA_VISIBLE_DEVICES=0

# Controlled ablation eval: Exp1 (no distillation, OpenAI-CLIP init, plain contrastive
# on evobio10m-v3.3 TOL-10M) vs Exp2 (identical config + BioCLIP 2.5 distillation).
# Both compared at epoch 12 -- the highest epoch BOTH runs had actually completed as of
# 2026-07-14 (Exp1 had reached 15, Exp2 12; neither had finished all 30 -- both got cut
# off by the 48h wall-time limit). Epoch-matched on purpose: comparing Exp1@15 vs
# Exp2@12 would confound "distillation helped/hurt" with "trained longer", defeating the
# point of a single-variable ablation. Both experiments are being --resume'd toward
# epoch 30 separately; re-run this eval against later matching epochs once more data is
# in, don't just swap in whichever checkpoint is newest for one side only.
#
# Reuses the same eval harness as slurm/eval_hypothesis1.sh (classification.py =
# proposal Option 1 inter-species accuracy, geometry_eval.py = Option 3 FDR / intra-
# species orthogonality) against the same four benchmark families. See that script's
# header comment for the one-time data-prep commands (NABirds/Meta-Album/Rare Species/
# CameraTrap) -- already done, not repeated here.
MODEL_NAMES=("exp1_no_distill" "exp2_distill")
MODEL_TYPES=(
  "ViT-B-16"
  "ViT-B-16"
)
PRETRAINED_PATHS=(
  "/fs/scratch/PAS2136/chenxujiang/bioclip-ablation-logs/exp1-no-distill-evobio10m/checkpoints/epoch_12.pt"
  "/fs/scratch/PAS2136/chenxujiang/bioclip-ablation-logs/exp2-distill-evobio10m/checkpoints/epoch_12.pt"
)

LOG_FILEPATH="/fs/scratch/PAS2136/chenxujiang/bioclip-ablation-eval/logs"

for i in "${!MODEL_NAMES[@]}"; do
  NAME=${MODEL_NAMES[$i]}
  MODEL_TYPE=${MODEL_TYPES[$i]}
  PRETRAINED=${PRETRAINED_PATHS[$i]}

  echo "==== $NAME : $MODEL_TYPE / $PRETRAINED ===="

  # --- Option 1: inter-species classification ---
  TEXT_TYPE="taxon_com"
  CAMERA_TRAP_ROOT="/fs/scratch/PAS2136/chenxujiang/IDLE-OO-Camera-Traps"
  DATA_ROOT="$CAMERA_TRAP_ROOT/data/test"
  LABEL_FILES=(
    "$CAMERA_TRAP_ROOT/desert-lion-balanced.csv"
    "$CAMERA_TRAP_ROOT/ENA24-balanced.csv"
    "$CAMERA_TRAP_ROOT/island-balanced.csv"
    "$CAMERA_TRAP_ROOT/orinoquia-balanced.csv"
    "$CAMERA_TRAP_ROOT/ohio-small-animals-balanced.csv"
  )
  for j in "${!LABEL_FILES[@]}"; do
    python -m src.evaluation.classification \
      --model "$MODEL_TYPE" \
      --batch-size 256 \
      --data_root "$DATA_ROOT" \
      --pretrained "$PRETRAINED" \
      --label_filename "${LABEL_FILES[$j]}" \
      --logs "$LOG_FILEPATH/$NAME" \
      --text_type "$TEXT_TYPE" \
      --task_type all \
      --classification-tasks zero_shot few_shot \
      --nfold 5 \
      --kshot_list 1 5
  done

  META_ALBUM_TEXT_TYPE="asis"
  META_ALBUM_DATA_ROOTS=(
    "/fs/ess/PAS2136/meta-album/set0/PLK_Mini/val"
    "/fs/ess/PAS2136/meta-album/set2/INS_Mini/val"
    "/fs/ess/PAS2136/meta-album/set1/INS_2_Mini/val"
    "/fs/ess/PAS2136/meta-album/set1/PLT_NET_Mini/val"
    "/fs/ess/PAS2136/meta-album/set2/FNG_Mini/val"
    "/fs/ess/PAS2136/meta-album/set0/PLT_VIL_Mini/val"
    "/fs/ess/PAS2136/meta-album/set1/MED_LF_Mini/val"
    "/fs/ess/PAS2136/hierarchical-vision/datasets/nabird/nabirds/images"
  )
  META_ALBUM_LABEL_FILES=(
    "/fs/scratch/PAS2136/chenxujiang/meta-album-metadata/PLK_Mini.csv"
    "/fs/scratch/PAS2136/chenxujiang/meta-album-metadata/INS_Mini.csv"
    "/fs/scratch/PAS2136/chenxujiang/meta-album-metadata/INS_2_Mini.csv"
    "/fs/scratch/PAS2136/chenxujiang/meta-album-metadata/PLT_NET_Mini.csv"
    "/fs/scratch/PAS2136/chenxujiang/meta-album-metadata/FNG_Mini.csv"
    "/fs/scratch/PAS2136/chenxujiang/meta-album-metadata/PLT_VIL_Mini.csv"
    "/fs/scratch/PAS2136/chenxujiang/meta-album-metadata/MED_LF_Mini.csv"
    "/fs/scratch/PAS2136/chenxujiang/nabirds_metadata.csv"
  )
  for j in "${!META_ALBUM_DATA_ROOTS[@]}"; do
    python -m src.evaluation.classification \
      --model "$MODEL_TYPE" \
      --batch-size 256 \
      --data_root "${META_ALBUM_DATA_ROOTS[$j]}" \
      --pretrained "$PRETRAINED" \
      --label_filename "${META_ALBUM_LABEL_FILES[$j]}" \
      --logs "$LOG_FILEPATH/$NAME" \
      --text_type "$META_ALBUM_TEXT_TYPE" \
      --task_type all \
      --classification-tasks zero_shot few_shot \
      --nfold 5 \
      --kshot_list 1 5
  done

  DATA_ROOT="/fs/scratch/PAS2136/chenxujiang/rare-species-export"
  LABEL_FILE="/fs/scratch/PAS2136/chenxujiang/rare-species-export/metadata.csv"
  python -m src.evaluation.classification \
    --model "$MODEL_TYPE" \
    --batch-size 256 \
    --data_root "$DATA_ROOT" \
    --pretrained "$PRETRAINED" \
    --label_filename "$LABEL_FILE" \
    --logs "$LOG_FILEPATH/$NAME" \
    --text_type taxon_com \
    --task_type all \
    --classification-tasks zero_shot few_shot \
    --nfold 5 \
    --kshot_list 1 5

  # --- Option 3: intra-species geometry ---
  python -m src.evaluation.geometry_eval \
    --model "$MODEL_TYPE" \
    --batch-size 256 \
    --data_root "$DATA_ROOT" \
    --pretrained "$PRETRAINED" \
    --label_filename metadata.csv \
    --text_type taxon_com \
    --workers 8 \
    --logs "$LOG_FILEPATH/$NAME" \
    --orthogonality-components 10

done

echo "Done. Compare $LOG_FILEPATH/exp1_no_distill/.../geometry_results.json against"
echo "$LOG_FILEPATH/exp2_distill/.../geometry_results.json, and the classification logs"
echo "under the same two directories, for the distillation ablation (epoch 12 vs epoch 12)."
