"""
Build a metadata.csv for NABirds in the format src/evaluation/data.py's
DatasetFromFile expects, from the raw NABirds distribution (images.txt,
image_class_labels.txt, classes.txt, train_test_split.txt).

The raw download (Caltech/CUB-style pairing files) doesn't include the flat CSV this
repo's eval code reads. eval.sh / eval_hypothesis1.sh use --text_type asis for
NABirds, so DatasetFromFile only needs a 'class' column (used as-is, no taxonomy
parsing) plus a 'filepath' column relative to the images/ directory -- this script
writes exactly that, using NABirds' own class names verbatim as 'class'. It does not
attempt to reconstruct full kingdom/phylum/.../species columns (NABirds' classes.txt
gives common names, not a structured taxonomy, so that mapping isn't available here) --
if a taxonomic breakdown is needed later (e.g. to run geometry_eval.py on NABirds),
this file will need a separate common-name-to-taxonomy lookup.

By default only the test split (train_test_split.txt == 0) is written, matching
standard evaluation convention -- pass --include-train to keep everything.

Usage:
    python -m src.evaluation.build_nabirds_metadata \
        --nabirds-root /fs/ess/PAS2136/hierarchical-vision/datasets/nabird/nabirds \
        --output /fs/ess/PAS2136/hierarchical-vision/datasets/nabird/nabirds/metadata.csv
"""
import argparse
import os

import pandas as pd


def read_pairs(path, value_has_spaces=False):
    result = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if value_has_spaces:
                key, value = line.split(" ", 1)
            else:
                key, value = line.split()
            result[key] = value
    return result


def build_rows(nabirds_root, include_train):
    images = read_pairs(os.path.join(nabirds_root, "images.txt"))
    image_class_labels = read_pairs(os.path.join(nabirds_root, "image_class_labels.txt"))
    classes = read_pairs(os.path.join(nabirds_root, "classes.txt"), value_has_spaces=True)
    split = read_pairs(os.path.join(nabirds_root, "train_test_split.txt"))

    rows = []
    skipped_split = 0
    skipped_no_class = 0
    for image_id, rel_path in images.items():
        if not include_train and split.get(image_id) == "1":
            skipped_split += 1
            continue
        class_id = image_class_labels.get(image_id)
        class_name = classes.get(class_id) if class_id is not None else None
        if class_name is None:
            skipped_no_class += 1
            continue
        # filepath is relative to the images/ directory itself, matching how
        # eval.sh/eval_hypothesis1.sh point --data_root at .../nabirds/images/
        rows.append({"class": class_name, "filepath": rel_path})

    return rows, skipped_split, skipped_no_class


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--nabirds-root", required=True,
        help="Directory containing images.txt, classes.txt, image_class_labels.txt, "
             "train_test_split.txt (the raw NABirds download root).",
    )
    parser.add_argument("--output", required=True, help="Where to write metadata.csv")
    parser.add_argument(
        "--include-train", action="store_true",
        help="Include training-split images too (default: test split only, matching "
             "standard evaluation convention).",
    )
    args = parser.parse_args()

    rows, skipped_split, skipped_no_class = build_rows(args.nabirds_root, args.include_train)
    if not rows:
        raise RuntimeError("No rows produced -- check --nabirds-root points at the raw NABirds directory.")

    df = pd.DataFrame(rows)
    df.to_csv(args.output)  # default index=True writes the leading unnamed index column
    # DatasetFromFile.data expects, matching e.g. data/annotation/rare_species/metadata.csv

    print(
        f"Wrote {len(rows)} rows ({df['class'].nunique()} distinct classes) to {args.output} "
        f"(skipped {skipped_split} train-split images, {skipped_no_class} with no class mapping)"
    )


if __name__ == "__main__":
    main()
