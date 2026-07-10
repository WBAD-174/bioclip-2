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
#
# Prerequisite one-time data prep (run before this script, not part of it; all done as
# of 2026-07-07, kept here for reproducibility / re-running from scratch):
#
#   NABirds:
#     python -m src.evaluation.build_nabirds_metadata \
#       --nabirds-root /fs/ess/PAS2136/hierarchical-vision/datasets/nabird/nabirds \
#       --output /fs/scratch/PAS2136/chenxujiang/nabirds_metadata.csv
#
#   Meta-Album (7 subsets; INS_Mini/FNG_Mini need the Unicode-normalization fallback in
#   build_meta_album_metadata.py -- their labels.csv CATEGORY strings and val/ folder
#   names use different Unicode normalization forms for accented characters):
#     for SUBSET in PLK_Mini:set0 INS_Mini:set2 INS_2_Mini:set1 PLT_NET_Mini:set1 \
#                   FNG_Mini:set2 PLT_VIL_Mini:set0 MED_LF_Mini:set1; do
#       NAME=${SUBSET%%:*}; SET=${SUBSET##*:}
#       python -m src.evaluation.build_meta_album_metadata \
#         --labels-csv "/fs/ess/PAS2136/meta-album/$SET/$NAME/labels.csv" \
#         --split-dir "/fs/ess/PAS2136/meta-album/$SET/$NAME/val" \
#         --output "/fs/scratch/PAS2136/chenxujiang/meta-album-metadata/$NAME.csv"
#     done
#
#   Rare Species (not on OSC -- downloaded from HF, parquet-backed, needs exporting to
#   real files before this repo's eval code can read it):
#     huggingface-cli download imageomics/rare-species --repo-type dataset \
#       --local-dir /fs/scratch/PAS2136/chenxujiang/rare-species
#     cd /fs/scratch/PAS2136/chenxujiang/rare-species
#     pip install polars datasets  # deps for the export script below, not in requirements.txt
#     python scripts/export_rare_species.py --dataset-path /fs/scratch/PAS2136/chenxujiang/rare-species-export
#     cd /users/PAS2136/chenxujiang/bioclip-2/bioclip-2
#     python -m src.evaluation.build_rare_species_metadata \
#       --input /fs/scratch/PAS2136/chenxujiang/rare-species/metadata.csv \
#       --output /fs/scratch/PAS2136/chenxujiang/rare-species-export/metadata.csv
#
#   CameraTrap (IDLE-OO-Camera-Traps, not on OSC -- downloaded from HF; needs a login
#   token, anonymous downloads of this dataset hit HF's rate limit):
#     huggingface-cli login   # or export HF_TOKEN=...
#     huggingface-cli download imageomics/IDLE-OO-Camera-Traps --repo-type dataset \
#       --local-dir /fs/scratch/PAS2136/chenxujiang/IDLE-OO-Camera-Traps --max-workers 1
#     # no conversion needed -- its own CSV columns already match what this repo's
#     # eval code expects (kingdom/phylum/cls/order/family/genus/species/common_name/filepath)
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
  # IDLE-OO-Camera-Traps' own 'filepath' column already includes the subset name
  # (e.g. "ENA24/<uuid>.png"), so unlike the old eval.sh assumption, all 5 subsets
  # share ONE data_root (data/test/) rather than each getting its own subset-specific
  # data_root -- see project memory / conversation notes for how this was confirmed.
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
  # metadata.csv for these is generated by build_meta_album_metadata.py /
  # build_nabirds_metadata.py -- see comment block above the loop.
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

  # --- Option 3: intra-species geometry (rare-species has full taxonomic columns) ---
  # --text_type must be something DatasetFromFile can actually compute from this CSV's
  # columns (kingdom/phylum/cls/order/family/genus/species/common_name) -- 'asis' needs
  # a literal pre-existing 'class' column, which this CSV doesn't have (it has 'cls',
  # matching the Taxon dataclass field name). geometry_eval.py itself never reads the
  # resulting 'class' text value -- it reads the taxonomic columns directly -- so any
  # working --text_type is fine; 'taxon_com' matches the classification.py call above.
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

echo "Done. Compare $LOG_FILEPATH/teacher/.../geometry_results.json against"
echo "$LOG_FILEPATH/student/.../geometry_results.json, and the classification logs"
echo "under the same two directories, for hypothesis 1."
