"""
Rename columns in the HF `imageomics/rare-species` dataset's own metadata.csv to match
what src/evaluation/data.py's DatasetFromFile / imageomics/naming_eval.py's Taxon
dataclass expect: 'class' -> 'cls', 'common' -> 'common_name', 'file_name' -> 'filepath'
(everything else -- kingdom/phylum/order/family/genus/species -- already matches).

Run this against the metadata.csv paired with images actually extracted via that
dataset's own scripts/export_rare_species.py (the HF copy is parquet-backed; this repo's
eval code needs real files on disk). --output should sit next to the exported
`dataset/` directory so --data_root == the export's --dataset-path still resolves
'filepath' values (which already start with "dataset/...") correctly.

Usage:
    python scripts/export_rare_species.py --dataset-path /fs/scratch/PAS2136/chenxujiang/rare-species-export
    python -m src.evaluation.build_rare_species_metadata \
        --input /fs/scratch/PAS2136/chenxujiang/rare-species/metadata.csv \
        --output /fs/scratch/PAS2136/chenxujiang/rare-species-export/metadata.csv
"""
import argparse
import os

import pandas as pd

RENAME = {"class": "cls", "common": "common_name", "file_name": "filepath"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Raw metadata.csv from the HF rare-species download.")
    parser.add_argument("--output", required=True, help="Where to write the renamed metadata.csv.")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    missing = [c for c in RENAME if c not in df.columns]
    if missing:
        raise ValueError(f"{args.input} is missing expected column(s) {missing} (found {list(df.columns)}).")

    df = df.rename(columns=RENAME)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    df.to_csv(args.output)  # default index=True writes the leading unnamed index column

    print(f"Wrote {len(df)} rows to {args.output} (renamed {RENAME})")


if __name__ == "__main__":
    main()
