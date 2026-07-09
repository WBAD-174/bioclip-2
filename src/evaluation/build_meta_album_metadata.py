"""
Build a metadata.csv for a Meta-Album subset in the format src/evaluation/data.py's
DatasetFromFile expects, from that subset's raw labels.csv.

Meta-Album's own labels.csv has columns FILE_NAME,CATEGORY,SUPER_CATEGORY (not the
filepath/class columns DatasetFromFile reads), and its val/ directory is already laid
out ImageFolder-style (one subdirectory per CATEGORY). This script just re-expresses
that same information as a flat metadata.csv, verifying each file actually exists
under val/<CATEGORY>/<FILE_NAME> along the way (rows with missing files are dropped
and counted, not silently included).

By default underscores in CATEGORY are replaced with spaces for the 'class' column
(nicer as literal zero-shot template text, e.g. "Amphidinium_sp" -> "Amphidinium sp")
-- pass --no-clean-names to keep CATEGORY verbatim.

Usage (run once per subset):
    python -m src.evaluation.build_meta_album_metadata \
        --labels-csv /fs/ess/PAS2136/meta-album/set0/PLK_Mini/labels.csv \
        --split-dir /fs/ess/PAS2136/meta-album/set0/PLK_Mini/val \
        --output /fs/scratch/PAS2136/chenxujiang/meta-album-metadata/PLK_Mini.csv
"""
import argparse
import os

import pandas as pd


def build_rows(labels_csv, split_dir, clean_names):
    labels = pd.read_csv(labels_csv)
    for col in ("FILE_NAME", "CATEGORY"):
        if col not in labels.columns:
            raise ValueError(f"{labels_csv} is missing expected column '{col}' (found {list(labels.columns)}).")

    rows = []
    skipped_missing = 0
    for _, row in labels.iterrows():
        category = str(row["CATEGORY"])
        filename = str(row["FILE_NAME"])
        rel_path = os.path.join(category, filename)
        if not os.path.exists(os.path.join(split_dir, rel_path)):
            skipped_missing += 1
            continue
        class_name = category.replace("_", " ") if clean_names else category
        rows.append({"class": class_name, "filepath": rel_path})

    return rows, skipped_missing


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-csv", required=True, help="Path to the subset's labels.csv.")
    parser.add_argument("--split-dir", required=True, help="Path to the subset's val/ directory (ImageFolder-style).")
    parser.add_argument("--output", required=True, help="Where to write metadata.csv.")
    parser.add_argument(
        "--no-clean-names", dest="clean_names", action="store_false",
        help="Keep CATEGORY verbatim (default: replace underscores with spaces).",
    )
    args = parser.parse_args()

    rows, skipped_missing = build_rows(args.labels_csv, args.split_dir, args.clean_names)
    if not rows:
        raise RuntimeError(
            "No rows produced -- check --labels-csv/--split-dir point at a real Meta-Album subset."
        )

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    df.to_csv(args.output)  # default index=True writes the leading unnamed index column
    # DatasetFromFile.data expects, matching e.g. data/annotation/rare_species/metadata.csv

    print(
        f"Wrote {len(rows)} rows ({df['class'].nunique()} distinct classes) to {args.output} "
        f"(skipped {skipped_missing} rows whose file wasn't found under {args.split_dir})"
    )


if __name__ == "__main__":
    main()
