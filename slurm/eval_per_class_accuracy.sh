#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --account=PAS2136
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --job-name=bioclip-per-class-acc
#SBATCH --time=8:00:00
#SBATCH --mem=200GB

# Per-class zero-shot accuracy for teacher and student, for the "long-tail
# data-coverage" disentangling check in Hypothesis1_Geometry_Findings.md. Same
# resource shape as slurm/eval_hypothesis1.sh's classification.py calls, since
# src/evaluation/per_class_accuracy.py reuses the same model/feature-extraction path --
# just a different aggregation at the end (per-class instead of pooled).
#
# Scoped to a few representative eval sets rather than all 14, to keep this cheap:
# rare-species (has full taxonomic columns, most direct link to
# count_training_species_frequency.py output), ohio-small-animals and orinoquia (hit
# hardest in the original classification results), nabirds and PLT_NET_Mini (barely
# hit -- useful contrast). Add more DATA_ROOTS/LABEL_FILES entries below if you want
# full coverage later.

module load miniconda3/24.1.2-py310
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate bioclip

cd /users/PAS2136/chenxujiang/bioclip-2/bioclip-2

export CUDA_VISIBLE_DEVICES=0

LOG_FILEPATH="/fs/scratch/PAS2136/chenxujiang/bioclip-hypothesis1-eval/logs"

MODEL_NAMES=("teacher" "student")
MODEL_TYPES=(
  "hf-hub:imageomics/bioclip-2.5-vith14"
  "ViT-B-16-1024"
)
PRETRAINED_PATHS=(
  "False"
  "/fs/scratch/PAS2136/chenxujiang/bioclip-distill-logs/2026_07_06-00_27_36-model_ViT-B-16-1024-lr_0.0001-b_256-j_8-p_amp/checkpoints/epoch_30.pt"
)

DATA_ROOTS=(
  "/fs/scratch/PAS2136/chenxujiang/rare-species-export"
  "/fs/scratch/PAS2136/chenxujiang/IDLE-OO-Camera-Traps/data/test"
  "/fs/scratch/PAS2136/chenxujiang/IDLE-OO-Camera-Traps/data/test"
  "/fs/ess/PAS2136/hierarchical-vision/datasets/nabird/nabirds/images"
  "/fs/ess/PAS2136/meta-album/set1/PLT_NET_Mini/val"
)
LABEL_FILES=(
  "/fs/scratch/PAS2136/chenxujiang/rare-species-export/metadata.csv"
  "/fs/scratch/PAS2136/chenxujiang/IDLE-OO-Camera-Traps/ohio-small-animals-balanced.csv"
  "/fs/scratch/PAS2136/chenxujiang/IDLE-OO-Camera-Traps/orinoquia-balanced.csv"
  "/fs/scratch/PAS2136/chenxujiang/nabirds_metadata.csv"
  "/fs/scratch/PAS2136/chenxujiang/meta-album-metadata/PLT_NET_Mini.csv"
)
TEXT_TYPES=("taxon_com" "taxon_com" "taxon_com" "asis" "asis")

for i in "${!MODEL_NAMES[@]}"; do
  NAME=${MODEL_NAMES[$i]}
  MODEL_TYPE=${MODEL_TYPES[$i]}
  PRETRAINED=${PRETRAINED_PATHS[$i]}

  for j in "${!DATA_ROOTS[@]}"; do
    python -m src.evaluation.per_class_accuracy \
      --model "$MODEL_TYPE" \
      --batch-size 256 \
      --data_root "${DATA_ROOTS[$j]}" \
      --pretrained "$PRETRAINED" \
      --label_filename "${LABEL_FILES[$j]}" \
      --text_type "${TEXT_TYPES[$j]}" \
      --workers 8 \
      --logs "$LOG_FILEPATH/$NAME"
  done
done

echo "Done. Each dataset's per_class_accuracy.csv is under $LOG_FILEPATH/{teacher,student}/<timestamp>-...-per_class_accuracy/"
