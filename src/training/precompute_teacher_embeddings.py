"""
Precompute frozen teacher image embeddings over the distillation training shards.

The teacher (e.g. BioCLIP 2.5, `hf-hub:imageomics/bioclip-2.5-vith14`) never changes during
distillation, so its image-tower forward pass doesn't need to be re-run every training step.
This script runs it once per image and writes the result to disk; distillation training can
then read the embedding straight off disk instead of paying for a teacher forward pass.

Text embeddings are intentionally NOT precomputed here: `--text_type random` in
`src/training/train.py` picks a different text field per step, and the text tower is cheap
relative to a ViT-H/14 image tower, so it's still run live at train time.

For every input webdataset shard `<name>.tar` (containing `<key>.jpg` + text fields), writes
an output shard of the same name into `--output-dir` containing, per sample,
`<key>.teacher_emb.npy` -- the teacher's L2-normalized image embedding (float16, shape
[embed_dim]). Keeping the same shard names/sample keys lets a future training-side reader zip
the image shards and embedding shards together shard-for-shard, so it stays correct even though
`--dataset-resampled` picks shards in a random (but matched, if driven by the same shard list)
order at train time.

Shards are assigned whole to ranks (`shards[rank::world_size]`), and a rank skips any shard
whose output file already exists -- so a killed/resumed job just needs to be re-launched.

Usage: see slurm/precompute_teacher_embeddings.sh
"""
import argparse
import logging
import os

import numpy as np
import torch
import webdataset as wds
import braceexpand

from ..open_clip import create_model_and_transforms, get_input_dtype
from .data import log_and_continue, tarfile_to_samples_nothrow
from .distributed import init_distributed_device
from .logger import setup_logging
from .precision import get_autocast


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--teacher-model", default="hf-hub:imageomics/bioclip-2.5-vith14",
        help="open_clip model identifier for the frozen teacher.")
    parser.add_argument(
        "--teacher-pretrained", default=None,
        help="Pretrained tag/path; ignored when --teacher-model uses the hf-hub: schema.")
    parser.add_argument(
        "--input-data", required=True,
        help="Brace-expand webdataset shard pattern, e.g. '[training-dir]/shard-{00000..24235}.tar'.")
    parser.add_argument(
        "--output-dir", required=True,
        help="Directory to write one teacher-embedding shard per input shard.")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument(
        "--workers", type=int, default=2,
        help="Dataloader workers. Shards are processed one at a time, so this only helps by "
             "prefetching/decoding the next shard while the current one is still on GPU.")
    parser.add_argument("--precision", default="amp", choices=["amp", "amp_bfloat16", "amp_bf16", "fp32"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dist-backend", default=None)
    parser.add_argument("--dist-url", default="env://")
    parser.add_argument("--cache-dir", default=None)
    return parser.parse_args()


@torch.no_grad()
def embed_shard(model, shard_url, out_path, preprocess, autocast, input_dtype, device, batch_size, workers):
    dataset = wds.DataPipeline(
        wds.SimpleShardList(shard_url),
        wds.split_by_worker,
        tarfile_to_samples_nothrow,
        wds.decode("pilrgb", handler=log_and_continue),
        wds.rename(image="jpg;png;jpeg;webp"),
        wds.map_dict(image=preprocess),
        wds.to_tuple("__key__", "image"),
        wds.batched(batch_size, partial=True),
    )
    loader = wds.WebLoader(dataset, batch_size=None, shuffle=False, num_workers=workers)

    tmp_path = out_path + ".tmp"
    n_written = 0
    with wds.TarWriter(tmp_path) as sink:
        for keys, images in loader:
            images = images.to(device=device, dtype=input_dtype, non_blocking=True)
            with autocast():
                image_features, _ = model.encode_image(images, normalize=True)
            image_features = image_features.float().to("cpu").to(torch.float16).numpy()
            for key, feat in zip(keys, image_features):
                sink.write({"__key__": key, "teacher_emb.npy": feat})
                n_written += 1
    os.replace(tmp_path, out_path)
    return n_written


def main():
    args = parse_args()
    device = init_distributed_device(args)
    setup_logging(None, logging.INFO, include_host=True)

    os.makedirs(args.output_dir, exist_ok=True)

    all_shards = sorted(braceexpand.braceexpand(args.input_data))
    my_shards = all_shards[args.rank::args.world_size]
    logging.info(f"[rank {args.rank}/{args.world_size}] assigned {len(my_shards)}/{len(all_shards)} shards.")

    model, _, preprocess = create_model_and_transforms(
        args.teacher_model,
        args.teacher_pretrained,
        precision=args.precision,
        device=device,
        output_dict=True,
        cache_dir=args.cache_dir,
    )
    model.eval()

    autocast = get_autocast(args.precision, device_type=torch.device(device).type)
    input_dtype = get_input_dtype(args.precision)

    for i, shard_url in enumerate(my_shards):
        out_path = os.path.join(args.output_dir, os.path.basename(shard_url))
        if os.path.exists(out_path):
            logging.info(f"[rank {args.rank}] ({i + 1}/{len(my_shards)}) skip (exists): {out_path}")
            continue
        n = embed_shard(
            model, shard_url, out_path, preprocess, autocast, input_dtype, device,
            args.batch_size, args.workers,
        )
        logging.info(f"[rank {args.rank}] ({i + 1}/{len(my_shards)}) wrote {n} embeddings -> {out_path}")

    logging.info(f"[rank {args.rank}] done.")


if __name__ == "__main__":
    main()
