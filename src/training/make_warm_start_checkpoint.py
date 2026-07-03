"""
Build a warm-start checkpoint for the distillation student.

Matt asked for the student to warm-start from a LAION-2B-pretrained ViT-B/16, matching how
BioCLIP 2 itself warm-started its (ViT-L/14) backbone from a LAION-2B checkpoint rather than
training from scratch (see the bioclip-2 README's "Model" section).

This can't be done with a plain `--pretrained laion2b_s34b_b88k` flag, though: the student uses
the custom `ViT-B-16-1024` config (embed_dim=1024, chosen to match the BioCLIP 2.5 teacher for
feature-level comparisons), while every public LAION-2B ViT-B/16 checkpoint is trained as the
standard `ViT-B-16` config (embed_dim=512). The two architectures are identical everywhere
*except* the final image/text projection matrices (`visual.proj`, `text_projection`), whose
shapes depend on embed_dim. `open_clip.factory.load_checkpoint` loads with `strict=True`, so
pointing `--pretrained` straight at a 512-dim checkpoint for a 1024-dim model would fail with a
shape-mismatch error, not silently degrade.

So instead: load the standard pretrained ViT-B-16, load a freshly-initialized ViT-B-16-1024,
copy over every parameter whose name AND shape match (i.e. everything except the two final
projections, which stay at their random init), and save the result as a normal checkpoint file.
That file's shapes match ViT-B-16-1024 exactly, so it loads with the default `strict=True` path
like any other checkpoint -- `--pretrained /path/to/this/output` just works.

Usage: this is a short, one-off, network-dependent step (it downloads the LAION-2B checkpoint
from HF Hub), so just run it directly on an OSC login node -- no sbatch needed:

    python -m src.training.make_warm_start_checkpoint \
        --output /fs/scratch/PAS2136/bioclip-distillation/10M/student_warm_start.pt

The output is reused by every slurm/distill.sh run afterwards (--pretrained points at it), so
this only needs to be run once, and re-run only if --source-model/--source-pretrained/
--target-model change.
"""
import argparse
import logging

import torch

from ..open_clip import create_model


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-model", default="ViT-B-16")
    parser.add_argument("--source-pretrained", default="laion2b_s34b_b88k")
    parser.add_argument("--target-model", default="ViT-B-16-1024")
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache-dir", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    logging.info(f"Loading source {args.source_model} ({args.source_pretrained})...")
    source = create_model(args.source_model, args.source_pretrained, cache_dir=args.cache_dir)
    logging.info(f"Building fresh (randomly initialized) target {args.target_model}...")
    target = create_model(args.target_model, pretrained=None, cache_dir=args.cache_dir)

    src_sd = source.state_dict()
    tgt_sd = target.state_dict()

    copied, skipped_shape, skipped_missing = [], [], []
    for key, tgt_param in tgt_sd.items():
        if key not in src_sd:
            skipped_missing.append(key)
            continue
        src_param = src_sd[key]
        if src_param.shape != tgt_param.shape:
            skipped_shape.append((key, tuple(src_param.shape), tuple(tgt_param.shape)))
            continue
        tgt_sd[key] = src_param.clone()
        copied.append(key)

    logging.info(f"Copied {len(copied)}/{len(tgt_sd)} parameters from source into target.")
    if skipped_shape:
        logging.info("Left at random init (shape mismatch -- expected for the final projections "
                      "since embed_dim differs, 512 vs 1024):")
        for key, src_shape, tgt_shape in skipped_shape:
            logging.info(f"  {key}: source {src_shape} vs target {tgt_shape}")
    if skipped_missing:
        logging.info(f"Left at random init (not present in source checkpoint): {skipped_missing}")

    # Sanity check: only the projection layers (and anything genuinely new to the target
    # architecture) should differ. If the backbone itself doesn't line up 1:1, something about
    # --target-model isn't actually the same body as --source-model and this warm start is
    # silently doing much less than intended.
    unexpected_mismatches = [k for k, *_ in skipped_shape if 'proj' not in k]
    if unexpected_mismatches or skipped_missing:
        raise RuntimeError(
            f"Warm start body did not line up as expected: unexpected shape mismatches "
            f"{unexpected_mismatches}, missing keys {skipped_missing}. Refusing to write a "
            f"checkpoint that warm-starts less of the model than intended -- check that "
            f"--source-model and --target-model really do share the same backbone."
        )

    torch.save(tgt_sd, args.output)
    logging.info(f"Wrote warm-start checkpoint to {args.output}")


if __name__ == "__main__":
    main()
