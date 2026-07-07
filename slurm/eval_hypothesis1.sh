#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --account=PAS2136
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --job-name=bioclip-eval-h1
#SBATCH --time=8:00:00
#SBATCH --mem=400GB

module load miniconda3/24.1.2-py310
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate bioclip

cd /users/PAS2136/chenxujiang/bioclip-2/bioclip-2

export CUDA_VISIBLE_DEVICES=0

# Hypothesis 1 (mentor, 2026-07-01): logit-KD should mostly preserve inter-species
# capability but show highly degraded intra-species capability. This runs BOTH the
# teacher and the distilled student through the same eval sets so the two are directly
# comparable:
#   - classification.py (proposal Option 1): inter-species zero-/few-shot accuracy.
#   - geometry_eval.py   (proposal Option 3): FDR (inter-species) + intra-species
#     orthogonality, model-geometry-only, no classification head involved.
#
# Loop is (name, model, pretrained) so it's a two-line change to add a third model
# later (e.g. a feature-alignment student for hypothesis 2).
MODEL_NAMES=("teacher" "student")
MODEL_TYPES=(
  "hf-hub:imageomics/bioclip-2.5-vith14"
  "ViT-B-16-1024"
)
PRETRAINED_PATHS=(
  "False"
  "/fs/scratch/PAS2136/chenxujiang/bioclip-distill-logs/2026_07_06-00_27_36-model_ViT-B-16-1024-lr_0.0001-b_256-j_8-p_amp/checkpoints/epoch_30.pt"
)

LOG_FILEPATH="/fs/scratch/PAS2136/chenxujiang/bioclip-hypothesis1-eval/logs"

for i in "${!MODEL_NAMES[@]}"; do
  NAME=${MODEL_NAMES[$i]}
  MODEL_TYPE=${MODEL_TYPES[$i]}
  PRETRAINED=${PRETRAINED_PATHS[$i]}

  echo "==== $NAME : $MODEL_TYPE / $PRETRAINED ===="

  # --- Option 1: inter-species classification (same benchmarks as slurm/eval.sh) ---
  TEXT_TYPE="taxon_com"
  DATA_ROOTS=(
    "[test-set-dir]/CameraTrap/images/desert-lion/"
    "[test-set-dir]/CameraTrap/images/ENA24/"
    "[test-set-dir]/CameraTrap/images/island/"
    "[test-set-dir]/CameraTrap/images/orinoquia/"
    "[test-set-dir]/CameraTrap/images/ohio-small-animals/"
  )
  LABEL_FILES=(
    "[test-set-dir]/CameraTrap/desert-lion-balanced.csv"
    "[test-set-dir]/CameraTrap/ENA24-balanced.csv"
    "[test-set-dir]/CameraTrap/island-balanced.csv"
    "[test-set-dir]/CameraTrap/orinoquia-balanced.csv"
    "[test-set-dir]/CameraTrap/ohio-small-animals-balanced.csv"
  )
  for j in "${!DATA_ROOTS[@]}"; do
    python -m src.evaluation.classification \
      --model "$MODEL_TYPE" \
      --batch-size 256 \
      --data_root "${DATA_ROOTS[$j]}" \
      --pretrained "$PRETRAINED" \
      --label_filename "${LABEL_FILES[$j]}" \
      --logs "$LOG_FILEPATH/$NAME" \
      --text_type $TEXT_TYPE \
      --task_type all \
      --classification-tasks zero_shot few_shot \
      --nfold 5 \
      --kshot_list 1 5
  done

  DATA_ROOT="[test-set-dir]/rare-species/"
  LABEL_FILE="[test-set-dir]/rare-species/metadata.csv"
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

  # --- Option 3: intra-species geometry (rare-species has full taxonomic columns) ---
  python -m src.evaluation.geometry_eval \
    --model "$MODEL_TYPE" \
    --batch-size 256 \
    --data_root "$DATA_ROOT" \
    --pretrained "$PRETRAINED" \
    --label_filename metadata.csv \
    --text_type asis \
    --workers 8 \
    --logs "$LOG_FILEPATH/$NAME" \
    --orthogonality-components 10

done

echo "Done. Compare $LOG_FILEPATH/teacher/.../geometry_results.json against"
echo "$LOG_FILEPATH/student/.../geometry_results.json, and the classification logs"
echo "under the same two directories, for hypothesis 1."
