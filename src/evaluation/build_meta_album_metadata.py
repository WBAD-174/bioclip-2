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

Some subsets (seen in INS_Mini, FNG_Mini) have CATEGORY strings in labels.csv that are
byte-for-byte different from the matching val/ subdirectory name despite looking
identical when printed -- Unicode normalization form mismatch (accented characters as
one precomposed codepoint vs. a base letter + combining accent, e.g. macOS tends to
decompose accents in filenames it creates). This script normalizes both sides (NFC) to
match on meaning, not byte layout, before falling back to "no match" -- but does NOT
fix separately-corrupted (mojibake) category strings, e.g. a "ç" that was already
mangled into "Ã§" upstream; those still won't resolve and are reported separately from
plain not-on-disk-at-all misses.

Usage (run once per subset):
    python -m src.evaluation.build_meta_album_metadata \
        --labels-csv /fs/ess/PAS2136/meta-album/set0/PLK_Mini/labels.csv \
        --split-dir /fs/ess/PAS2136/meta-album/set0/PLK_Mini/val \
        --output /fs/scratch/PAS2136/chenxujiang/meta-album-metadata/PLK_Mini.csv
"""
import argparse
import os
import unicodedata

import pandas as pd


def _nfc(s):
    return unicodedata.normalize("NFC", s)


def build_rows(labels_csv, split_dir, clean_names):
    labels = pd.read_csv(labels_csv)
    for col in ("FILE_NAME", "CATEGORY"):
        if col not in labels.columns:
            raise ValueError(f"{labels_csv} is missing expected column '{col}' (found {list(labels.columns)}).")

    # disk folder name, keyed by its NFC-normalized form, so a CATEGORY string that's a
    # different Unicode normalization of the same text still resolves to the real folder
    disk_categories = os.listdir(split_dir)
    disk_by_nfc = {_nfc(name): name for name in disk_categories}

    rows = []
    skipped_exact = 0
    skipped_normalized = 0
    unmatched_categories = 0
    seen_categories = set()
    for _, row in labels.iterrows():
        category = str(row["CATEGORY"])
        filename = str(row["FILE_NAME"])

        rel_path = os.path.join(category, filename)
        if os.path.exists(os.path.join(split_dir, rel_path)):
            class_name = category.replace("_", " ") if clean_names else category
            rows.append({"class": class_name, "filepath": rel_path})
            continue

        disk_category = disk_by_nfc.get(_nfc(category))
        if disk_category is not None:
            rel_path = os.path.join(disk_category, filename)
            if os.path.exists(os.path.join(split_dir, rel_path)):
                class_name = category.replace("_", " ") if clean_names else category
                rows.append({"class": class_name, "filepath": rel_path})
                skipped_normalized += 1  # recovered via normalization, not a clean match
                continue

        skipped_exact += 1
        if category not in seen_categories:
            seen_categories.add(category)
            unmatched_categories += 1

    # skipped_normalized counts rows recovered by the NFC fallback (still included in
    # rows), skipped_exact counts rows dropped entirely (category truly not resolvable,
    # e.g. mojibake-corrupted names -- see module docstring)
    return rows, skipped_exact, skipped_normalized, unmatched_categories


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

    rows, skipped_exact, skipped_normalized, unmatched_categories = build_rows(
        args.labels_csv, args.split_dir, args.clean_names
    )
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
        f"-- {skipped_normalized} recovered via Unicode-normalization fallback, "
        f"{skipped_exact} rows still dropped (unresolvable, from {unmatched_categories} "
        f"distinct categories -- likely mojibake-corrupted names, see module docstring)"
    )


if __name__ == "__main__":
    main()
