#!/usr/bin/env python3
"""
Count how many times each species (or other taxon field) appears in the student's
webdataset training shards, without touching image bytes or needing to join against
the upstream TreeOfLife-200M catalog/parquet files -- the per-sample taxon strings are
already baked into each shard as plain-text members alongside the .jpg
(see /fs/scratch/PAS2136/bioclip-distillation/10M/shards/*.tar, one member per sample
per suffix: .com.txt, .common_name.txt, .sci.txt, .sci_com.txt, .scientific_name.txt,
.taxon.txt, .taxonTag.txt, .taxonTag_com.txt, .taxon_com.txt, .taxonomic_name.txt).

This exists to test the "long-tail data-coverage" explanation in
Hypothesis1_Geometry_Findings.md: join this frequency table against each eval
dataset's species list (or against per_class_accuracy.py's output) to see whether the
accuracy drop concentrates in species the 10M sample barely saw.

Usage:
    python -m src.evaluation.count_training_species_frequency \
        --shards-glob '/fs/scratch/PAS2136/bioclip-distillation/10M/shards/shard-*.tar' \
        --field scientific_name \
        --out /fs/scratch/PAS2136/chenxujiang/bioclip-hypothesis1-eval/training_species_frequency.csv \
        --workers 16

--field picks which per-sample suffix to count (default: scientific_name, expected to
be a clean "Genus species" string -- check one shard's *.scientific_name.txt content
first with `tar xf shard-00000.tar --wildcards '*.scientific_name.txt' -O | head` to
confirm the format before trusting the counts). Any of the suffixes listed above works;
pass e.g. --field taxon for the full 7-rank string instead of just species.

Runs on a login node or a plain (no-GPU) Slurm job -- this is pure text I/O, streams
through each shard sequentially (tarfile mode 'r|') so it never seeks or loads image
bytes into memory.
"""
import argparse
import csv
import glob
import multiprocessing
import tarfile
from collections import Counter


def count_one_shard(args):
    shard_path, field = args
    suffix = f".{field}.txt"
    counts = Counter()
    n_samples = 0
    with tarfile.open(shard_path, mode="r|") as tf:
        for member in tf:
            if not member.name.endswith(suffix):
                continue
            f = tf.extractfile(member)
            if f is None:
                continue
            text = f.read().decode("utf-8", errors="replace").strip()
            if text:
                counts[text] += 1
                n_samples += 1
    return counts, n_samples, shard_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards-glob", required=True)
    ap.add_argument(
        "--field", default="scientific_name",
        choices=[
            "com", "common_name", "sci", "sci_com", "scientific_name",
            "taxon", "taxonTag", "taxonTag_com", "taxon_com", "taxonomic_name",
        ],
    )
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    shard_paths = sorted(glob.glob(args.shards_glob))
    if not shard_paths:
        raise SystemExit(f"No shards matched {args.shards_glob!r}")
    print(f"Found {len(shard_paths)} shards. Counting field={args.field!r} with {args.workers} workers...")

    total_counts = Counter()
    total_samples = 0
    work = [(p, args.field) for p in shard_paths]
    with multiprocessing.Pool(args.workers) as pool:
        for i, (counts, n_samples, shard_path) in enumerate(pool.imap_unordered(count_one_shard, work), 1):
            total_counts.update(counts)
            total_samples += n_samples
            if i % 50 == 0 or i == len(shard_paths):
                print(f"  [{i}/{len(shard_paths)}] shards done, {total_samples} samples, "
                      f"{len(total_counts)} unique values so far ({shard_path})")

    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([args.field, "count"])
        for value, count in total_counts.most_common():
            writer.writerow([value, count])

    print(f"\nDone. {total_samples} samples, {len(total_counts)} unique {args.field!r} values.")
    print(f"Wrote frequency table to {args.out}")

    singletons = sum(1 for c in total_counts.values() if c == 1)
    print(f"Sanity check: {singletons} values ({singletons / max(len(total_counts), 1) * 100:.1f}%) "
          f"appear exactly once in the 10M sample -- these are the ones most likely to explain "
          f"a rare-species accuracy drop if the eval sets overlap with them.")


if __name__ == "__main__":
    main()
