"""
Emergent-property geometry eval (distillation proposal's "Option 3").

Tests the geometry directly, independent of any classification head -- the two
properties the BioCLIP 2 paper identifies as emergent at scale:

  - Inter-species ecological alignment: species should separate cleanly by taxonomy.
    Measured here with the Fisher Discriminant Ratio (FDR) at each taxonomic level
    present in the label file: between-class variance / within-class variance. High
    FDR = clean separation at that level.
  - Intra-species variation preservation: within-species variation should survive in a
    subspace roughly orthogonal to the inter-species discriminative directions, rather
    than being flattened away by species-level contrastive pressure. Measured here as
    the orthogonality between the top principal directions of within-species residuals
    and the top principal directions of the between-species-mean scatter. Score near 1
    = intra-species variation lives in directions the model isn't using to discriminate
    species (property preserved). Score near 0 = that variation has collapsed into the
    same directions used for species discrimination (no independent signal left).

This is this repo's own operationalization of the description in
`BioCLIP2_Distillation_Proposal.md`'s Option 3, written directly from that text -- it
does NOT claim to reproduce the exact rho-separation formula from BioCLIP 2 paper
section 5.4. Cross-check against the paper before treating these numbers as directly
comparable to any numbers quoted from it.

Run once per model (teacher, student, ...) on the same labeled eval set (needs a
metadata.csv with kingdom/phylum/cls/order/family/genus/species columns, e.g.
data/annotation/rare_species/metadata.csv) and compare the printed/saved tables.

Usage: see slurm/eval_hypothesis1.sh
"""
import json
import logging
import os
import sys

import numpy as np

from .classification import build_dataloader, create_model, extract_feature_bundle
from .params import parse_args
from .utils import (
    configure_logging,
    configure_torch_backends,
    init_device,
    log_params,
    normalize_force_image_size,
    random_seed,
)

TAXON_LEVELS = ["species", "genus", "family", "order", "cls", "phylum", "kingdom"]


def fisher_discriminant_ratio(features, labels):
    """Between-class variance / within-class variance, pooled over all embedding dims."""
    classes, inv = np.unique(labels, return_inverse=True)
    overall_mean = features.mean(axis=0)
    between = 0.0
    within = 0.0
    n_total = len(features)
    for i in range(len(classes)):
        class_feats = features[inv == i]
        n_c = len(class_feats)
        if n_c < 2:
            continue
        class_mean = class_feats.mean(axis=0)
        between += n_c * np.sum((class_mean - overall_mean) ** 2)
        within += np.sum((class_feats - class_mean) ** 2)
    if within <= 0:
        return float("nan")
    return float((between / n_total) / (within / n_total))


def intra_species_orthogonality(features, species_labels, n_components=10):
    """See module docstring for the definition. Returns (score, n_components_used)."""
    classes, inv = np.unique(species_labels, return_inverse=True)
    overall_mean = features.mean(axis=0)

    class_means = []
    residuals = []
    for i in range(len(classes)):
        class_feats = features[inv == i]
        if len(class_feats) < 2:
            continue
        class_mean = class_feats.mean(axis=0)
        class_means.append(class_mean)
        residuals.append(class_feats - class_mean)

    if len(class_means) < 2 or not residuals:
        return float("nan"), 0

    between_matrix = np.stack(class_means) - overall_mean  # [n_classes_used, D]
    residual_matrix = np.concatenate(residuals, axis=0)  # [n_residual_samples, D]

    k = min(
        n_components,
        between_matrix.shape[0] - 1,
        residual_matrix.shape[0] - 1,
        features.shape[1],
    )
    k = int(k)
    if k < 1:
        return float("nan"), 0

    # Both matrices are already mean-subtracted (class means minus overall mean;
    # residuals minus their own class mean), so SVD right singular vectors are the
    # principal directions directly, no extra centering step needed.
    _, _, inter_basis = np.linalg.svd(between_matrix, full_matrices=False)
    inter_basis = inter_basis[:k]  # [k, D], rows are orthonormal directions
    _, _, intra_basis = np.linalg.svd(residual_matrix, full_matrices=False)
    intra_basis = intra_basis[:k]  # [k, D]

    overlap = inter_basis @ intra_basis.T  # [k, k] cosine similarities between bases
    mean_sq_overlap = float(np.mean(overlap ** 2))
    return 1.0 - mean_sq_overlap, k


def run_geometry_eval(features, taxon_table, args):
    results = {"n_samples": int(len(features)), "fdr": {}, "intra_species_orthogonality": None}

    for level in TAXON_LEVELS:
        if level not in taxon_table.columns:
            continue
        labels = taxon_table[level].astype(str).values
        n_classes = len(set(labels))
        if n_classes < 2:
            continue
        fdr = fisher_discriminant_ratio(features, labels)
        results["fdr"][level] = {"value": fdr, "n_classes": int(n_classes)}
        logging.info(f"[FDR] {level}: {fdr:.4f} (n_classes={n_classes})")

    if "species" in taxon_table.columns:
        # group by genus+species so identically-named species in different genera
        # (rare, but possible in large taxonomic tables) aren't merged together
        if "genus" in taxon_table.columns:
            species_key = (
                taxon_table["genus"].astype(str) + "_" + taxon_table["species"].astype(str)
            ).values
        else:
            species_key = taxon_table["species"].astype(str).values
        score, k = intra_species_orthogonality(features, species_key, args.orthogonality_components)
        results["intra_species_orthogonality"] = {"value": score, "n_components": k}
        logging.info(f"[intra-species orthogonality] {score:.4f} (k={k} components)")
    else:
        logging.warning("No 'species' column in label file; skipping intra-species orthogonality.")

    return results


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    parser_extra = _extra_arg_parser()
    args, _ = parser_extra.parse_known_args(sys.argv[1:] if argv is None else argv, namespace=args)

    random_seed(args.seed, 0)
    configure_torch_backends(deterministic=True)
    device = init_device(args)

    log_base_path = configure_logging(args, "geometry_eval", include_workers=True, log_filename="out.log")
    normalize_force_image_size(args)
    log_params(args, log_base_path)

    model, preprocess_val = create_model(args, device)
    dataloader = build_dataloader(args, preprocess_val)
    bundle = extract_feature_bundle(model, dataloader, args)

    # dataloader has shuffle=False (see evaluation/utils.get_dataloader), so row i of
    # the dataset's underlying label table lines up with bundle["features"][i].
    taxon_table = dataloader.dataset.data.reset_index(drop=True)
    assert len(taxon_table) == len(bundle["features"]), (
        f"label table has {len(taxon_table)} rows but extracted "
        f"{len(bundle['features'])} feature vectors -- alignment assumption broken."
    )

    results = run_geometry_eval(bundle["features"], taxon_table, args)
    results["model"] = args.model
    results["pretrained"] = args.pretrained
    results["data_root"] = args.data_root

    output_dir = log_base_path or os.getcwd()
    output_path = os.path.join(output_dir, "geometry_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logging.info(f"Saved geometry eval results to {output_path}")


def _extra_arg_parser():
    import argparse

    # --label_filename, --text_type, --data_root etc. already come from
    # evaluation/params.py's parse_args(); this only adds what's new here.
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--orthogonality-components",
        type=int,
        default=10,
        help="Number of top principal directions used on each side of the "
             "intra-species-vs-inter-species orthogonality comparison.",
    )
    return parser


if __name__ == "__main__":
    main()
