#!/usr/bin/env python3
"""
Zero-shot top-1 accuracy broken down per class, instead of the single pooled number
classification.py's evaluate_zero_shot() reports. Same model-loading and feature
extraction path as classification.py/geometry_eval.py, just a different aggregation at
the end.

This exists to test the "long-tail data-coverage" explanation in
Hypothesis1_Geometry_Findings.md: run this for teacher and student on the same eval set,
then join both output CSVs' class_name column against
count_training_species_frequency.py's output to see whether the accuracy drop
concentrates in species the 10M training sample barely covered.

CAVEAT: class_name here is whatever label the eval dataset's own CSV uses (via
DatasetFromFile / --text_type), which is not guaranteed to string-match the training
shards' scientific_name text exactly (capitalization, common name vs. scientific name,
etc.) -- check both sides' formatting before joining on class_name directly; a
normalization step per eval dataset may be needed.

Usage (mirrors classification.py's zero-shot args):
    python -m src.evaluation.per_class_accuracy \
        --model hf-hub:imageomics/bioclip-2.5-vith14 \
        --pretrained False \
        --data_root /fs/scratch/PAS2136/chenxujiang/rare-species-export \
        --label_filename metadata.csv \
        --text_type taxon_com \
        --batch-size 256 \
        --workers 8 \
        --logs /fs/scratch/PAS2136/chenxujiang/bioclip-hypothesis1-eval/logs/teacher
"""
import csv
import logging
import os
import sys

import numpy as np
import torch

from ..training.imagenet_zeroshot_data import openai_imagenet_template
from .classification import (
    build_classnames,
    build_dataloader,
    create_model,
    extract_feature_bundle,
    zero_shot_classifier,
)
from .params import parse_args
from .utils import (
    configure_logging,
    configure_torch_backends,
    init_device,
    log_params,
    normalize_force_image_size,
    random_seed,
)


def compute_per_class_top1(model, bundle, args):
    classnames = build_classnames(bundle["class_to_idx"])
    classifier = zero_shot_classifier(model, classnames, openai_imagenet_template, args)
    idx_to_class = {v: k for k, v in bundle["class_to_idx"].items()}

    features = bundle["features"]
    target = bundle["target"]
    n_classes = len(classnames)
    logit_scale = model.logit_scale.exp()

    correct = np.zeros(n_classes, dtype=np.int64)
    total = np.zeros(n_classes, dtype=np.int64)

    with torch.no_grad():
        for start in range(0, len(features), args.batch_size):
            end = start + args.batch_size
            image_features = torch.from_numpy(features[start:end]).to(
                args.device, dtype=classifier.dtype
            )
            target_batch = target[start:end]
            logits = logit_scale * image_features @ classifier
            pred = logits.argmax(dim=1).cpu().numpy()
            correct_mask = pred == target_batch

            for cls_idx in np.unique(target_batch):
                mask = target_batch == cls_idx
                total[cls_idx] += int(mask.sum())
                correct[cls_idx] += int((mask & correct_mask).sum())

    rows = []
    for cls_idx in range(n_classes):
        n = int(total[cls_idx])
        c = int(correct[cls_idx])
        rows.append({
            "class_name": idx_to_class.get(cls_idx, str(cls_idx)),
            "n_samples": n,
            "top1_correct": c,
            "top1_accuracy": (c / n) if n else float("nan"),
        })
    return rows


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    random_seed(args.seed, 0)
    configure_torch_backends(deterministic=True)
    device = init_device(args)

    log_base_path = configure_logging(args, "per_class_accuracy", include_workers=True, log_filename="out.log")
    normalize_force_image_size(args)
    log_params(args, log_base_path)

    model, preprocess_val = create_model(args, device)
    dataloader = build_dataloader(args, preprocess_val)
    bundle = extract_feature_bundle(model, dataloader, args)

    rows = compute_per_class_top1(model, bundle, args)
    rows.sort(key=lambda r: r["top1_accuracy"])

    output_dir = log_base_path or os.getcwd()
    output_path = os.path.join(output_dir, "per_class_accuracy.csv")
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["class_name", "n_samples", "top1_correct", "top1_accuracy"])
        writer.writeheader()
        writer.writerows(rows)

    total_n = sum(r["n_samples"] for r in rows)
    total_correct = sum(r["top1_correct"] for r in rows)
    overall = total_correct / total_n if total_n else float("nan")
    logging.info(f"Saved per-class accuracy for {len(rows)} classes to {output_path}")
    logging.info(f"Overall top-1 (sanity check, should ~match classification.py's val-unseen-top1): {overall * 100:.2f}")


if __name__ == "__main__":
    main()
