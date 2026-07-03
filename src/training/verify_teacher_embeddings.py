"""
Sanity-check the output of `precompute_teacher_embeddings.py` after a run finishes.

Three checks, cheapest first:

1. Completeness: does every expected output shard exist? (catches a job that got killed/timed
   out partway through -- just re-`sbatch` the precompute job to fill in what's missing).
2. Format: for a sample of shards, are the embeddings the right shape/dtype, unit-norm, and free
   of NaN/Inf?
3. Key alignment: for a sample of shards, does the *set* of sample keys in the output shard
   exactly match the set of keys in the corresponding input shard? (catches images that were
   silently dropped by a decode error during precompute -- `log_and_continue` swallows those).
4. (optional, --semantic-check) Coarse sanity check that the embeddings are meaningful, not just
   well-formed: within a shard, mean cosine similarity between same-species pairs should be
   higher than between different-species pairs. This is NOT a substitute for the real evaluation
   in src/evaluation/ -- it's a 30-second smoke test to catch a badly broken teacher/preprocessing
   before spending a training run on it.

Usage:
    python -m src.training.verify_teacher_embeddings \\
        --input-data '/fs/scratch/PAS2136/bioclip-distillation/10M/shards/shard-{00000..01001}.tar' \\
        --embed-dir '/fs/scratch/PAS2136/bioclip-distillation/10M/teacher-embeddings' \\
        --num-shards-to-check 5 \\
        --semantic-check
"""
import argparse
import os
import random
import tarfile

import braceexpand
import numpy as np
import webdataset as wds


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-data", required=True, help="Same brace-expand pattern used for precompute's --input-data.")
    parser.add_argument("--embed-dir", required=True, help="Same directory used for precompute's --output-dir.")
    parser.add_argument("--num-shards-to-check", type=int, default=5, help="How many shards to sample for the format/key/semantic checks.")
    parser.add_argument("--semantic-check", action="store_true", help="Also run the same-species-vs-different-species similarity sanity check.")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def tar_keys(path, strip_exts=(".jpg", ".png", ".jpeg", ".webp", ".sci.txt", ".com.txt", ".taxon.txt", ".sci_com.txt", ".taxon_com.txt", ".teacher_emb.npy")):
    keys = set()
    with tarfile.open(path) as tf:
        for member in tf.getnames():
            for ext in strip_exts:
                if member.endswith(ext):
                    keys.add(member[: -len(ext)])
                    break
    return keys


def check_completeness(expected_basenames, embed_dir):
    present = set(os.listdir(embed_dir)) if os.path.isdir(embed_dir) else set()
    missing = [b for b in expected_basenames if b not in present]
    print(f"[completeness] expected {len(expected_basenames)} shards, found {len(present & set(expected_basenames))}, missing {len(missing)}")
    if missing:
        preview = ", ".join(missing[:10])
        print(f"[completeness] missing (first 10): {preview}")
        print("[completeness] -> re-submit the precompute sbatch job; it skips shards that already exist.")
    return missing


def check_format(embed_path):
    ds = wds.WebDataset(embed_path).decode()
    dims, dtypes, norms, n_nan = set(), set(), [], 0
    n = 0
    for sample in ds:
        emb = sample["teacher_emb.npy"]
        dims.add(emb.shape)
        dtypes.add(str(emb.dtype))
        if not np.isfinite(emb).all():
            n_nan += 1
        else:
            norms.append(float(np.linalg.norm(emb.astype("float32"))))
        n += 1
    ok = len(dims) == 1 and len(dtypes) == 1 and n_nan == 0
    mean_norm = float(np.mean(norms)) if norms else float("nan")
    print(f"[format] {os.path.basename(embed_path)}: n={n}, shape={dims}, dtype={dtypes}, "
          f"mean_norm={mean_norm:.4f}, non_finite={n_nan} -> {'OK' if ok else 'PROBLEM'}")
    return ok


def check_key_alignment(input_shard_path, embed_shard_path):
    img_keys = tar_keys(input_shard_path)
    emb_keys = tar_keys(embed_shard_path)
    only_in_input = img_keys - emb_keys
    only_in_embed = emb_keys - img_keys
    ok = not only_in_input and not only_in_embed
    print(f"[keys] {os.path.basename(embed_shard_path)}: {len(img_keys)} image keys, {len(emb_keys)} embedding keys, "
          f"missing_from_embed={len(only_in_input)}, extra_in_embed={len(only_in_embed)} -> {'OK' if ok else 'PROBLEM'}")
    if only_in_input:
        print(f"[keys]   dropped during precompute (first 10): {list(only_in_input)[:10]}")
    return ok


def semantic_check(input_shard_path, embed_shard_path, max_pairs=20000, seed=0):
    embeds = {}
    for sample in wds.WebDataset(embed_shard_path).decode():
        embeds[sample["__key__"]] = sample["teacher_emb.npy"].astype("float32")

    species = {}
    for sample in wds.WebDataset(input_shard_path).decode():
        key = sample["__key__"]
        if key in embeds and "sci.txt" in sample:
            # first two words of the scientific name ~= genus + species
            species[key] = " ".join(sample["sci.txt"].split()[:2])

    keys = [k for k in species if k in embeds]
    if len(keys) < 20:
        print(f"[semantic] {os.path.basename(embed_shard_path)}: too few labeled samples ({len(keys)}) to check, skipping.")
        return

    rng = random.Random(seed)
    same_sims, diff_sims = [], []
    for _ in range(max_pairs):
        a, b = rng.sample(keys, 2)
        sim = float(np.dot(embeds[a], embeds[b]))  # already unit-norm, so dot == cosine
        (same_sims if species[a] == species[b] else diff_sims).append(sim)

    same_mean = np.mean(same_sims) if same_sims else float("nan")
    diff_mean = np.mean(diff_sims) if diff_sims else float("nan")
    verdict = "OK (same-species more similar, as expected)" if same_mean > diff_mean else "SUSPICIOUS (same-species not more similar than different-species)"
    print(f"[semantic] {os.path.basename(embed_shard_path)}: same-species pairs n={len(same_sims)} mean_cos={same_mean:.4f}, "
          f"different-species pairs n={len(diff_sims)} mean_cos={diff_mean:.4f} -> {verdict}")


def main():
    args = parse_args()
    random.seed(args.seed)

    all_input_shards = sorted(braceexpand.braceexpand(args.input_data))
    expected_basenames = [os.path.basename(s) for s in all_input_shards]

    missing = check_completeness(expected_basenames, args.embed_dir)

    present_shards = [s for s in all_input_shards if os.path.basename(s) not in missing]
    if not present_shards:
        print("No completed shards to check further.")
        return
    sample = random.sample(present_shards, min(args.num_shards_to_check, len(present_shards)))

    print(f"\nSampling {len(sample)} shard(s) for format/key checks...")
    for input_path in sample:
        embed_path = os.path.join(args.embed_dir, os.path.basename(input_path))
        check_format(embed_path)
        check_key_alignment(input_path, embed_path)
        if args.semantic_check:
            semantic_check(input_path, embed_path, seed=args.seed)


if __name__ == "__main__":
    main()
