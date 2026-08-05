#!/usr/bin/env python3
"""
Aggregate all classification.py + geometry_eval.py runs under eval_ablation.sh's log
dirs and print an exp1-vs-exp2 comparison table straight to stdout. Pass --out to also
dump the full (all runs, not just latest) data as CSV.

Usage (on the cluster, where the logs actually live):
    python -m src.evaluation.summarize_ablation_eval \
        --logs-root /fs/scratch/PAS2136/chenxujiang/bioclip-ablation-eval-epoch30/logs

Only the most recent run per (model, dataset) is used for the printed table --
eval_ablation.sh can get re-submitted more than once (run_name's leading timestamp
sorts lexically, so "most recent" = max(run_name) per (model, dataset)); duplicate
submissions just get silently collapsed to their latest run here.

For each <logs-root>/{exp1_no_distill,exp2_distill}/*classification*/ dir, reads
params.txt for data_root/label_filename (to identify the dataset) and out.log for the
"Results:" block (metric: value% lines). For each *geometry_eval*/ dir, reads
geometry_results.json and flattens it into fdr-<level> and intra-species-orthogonality
metrics, identified by data_root (geometry_eval only ran against RareSpecies here).
"""
import argparse
import csv
import json
import os
import re


def parse_params(params_path):
    info = {}
    with open(params_path) as f:
        for line in f:
            if ":" not in line:
                continue
            key, _, val = line.partition(":")
            info[key.strip()] = val.strip()
    return info


def parse_classification_results(out_log_path):
    """Return dict of metric -> float(percent) from the 'Results:' block in out.log."""
    with open(out_log_path) as f:
        lines = f.readlines()

    results = {}
    in_block = False
    metric_re = re.compile(r"\|\s*([\w.\-/@ ]+?):\s*([\-\d.]+)\s*$")
    for line in lines:
        if "Results:" in line:
            in_block = True
            continue
        if in_block:
            m = metric_re.search(line)
            if m:
                results[m.group(1).strip()] = float(m.group(2))
            else:
                if results:
                    break
    return results


def parse_geometry_results(json_path):
    """Flatten geometry_results.json into metric -> float."""
    with open(json_path) as f:
        data = json.load(f)

    results = {}
    for level, entry in data.get("fdr", {}).items():
        results[f"fdr-{level}"] = entry["value"]
    ortho = data.get("intra_species_orthogonality")
    if ortho is not None:
        results["intra-species-orthogonality"] = ortho["value"]
    return results


def dataset_name(params):
    label = params.get("label_filename", "")
    if label and label != "metadata.csv":
        return os.path.basename(label.rstrip("/"))
    # geometry_eval calls pass a bare "metadata.csv" relative to data_root -- use the
    # data_root's basename instead so it doesn't collapse every geometry run into one
    # indistinguishable "metadata.csv" row.
    data_root = params.get("data_root", "")
    return os.path.basename(data_root.rstrip("/")) or "unknown"


# Headline-only view for --compact: the 3 classification metrics and geometry metrics
# actually discussed when summarizing results, dropping top3/top5/std noise.
COMPACT_CLS_METRICS = ["val-unseen-top1", "few-shot-1-mean", "few-shot-5-mean"]
COMPACT_GEO_METRICS = [
    "fdr-species", "fdr-genus", "fdr-family", "fdr-order", "fdr-cls", "fdr-phylum",
    "intra-species-orthogonality",
]


def _fmt(v, decimals=2):
    return f"{v:.{decimals}f}" if v is not None else "n/a"


def _fmt_delta(v, base, decimals=2):
    if v is None or base is None:
        return "n/a"
    return f"{v - base:+.{decimals}f}"


def print_compact_markdown(by_dataset, model_names, baseline):
    """Print paste-ready markdown tables restricted to the headline metrics."""
    other_models = [m for m in model_names if m != baseline]

    cls_datasets = sorted(
        d for d, metrics in by_dataset.items()
        if any(metric in metrics for metric in COMPACT_CLS_METRICS)
    )
    if cls_datasets:
        header = ["Benchmark", baseline] + other_models
        print("| " + " | ".join(header) + " |")
        print("|" + "|".join(["---"] * len(header)) + "|")
        for dataset in cls_datasets:
            metrics = by_dataset[dataset]
            base_vals = [metrics.get(m, {}).get(baseline) for m in COMPACT_CLS_METRICS]
            row = [dataset, " / ".join(_fmt(v) for v in base_vals)]
            for other in other_models:
                other_vals = [metrics.get(m, {}).get(other) for m in COMPACT_CLS_METRICS]
                row.append(" / ".join(_fmt(v) for v in other_vals))
            print("| " + " | ".join(row) + " |")
        print()

    geo_datasets = sorted(
        d for d, metrics in by_dataset.items()
        if any(metric in metrics for metric in COMPACT_GEO_METRICS)
    )
    for dataset in geo_datasets:
        metrics = by_dataset[dataset]
        present = [m for m in COMPACT_GEO_METRICS if m in metrics]
        if not present:
            continue
        print(f"Geometry ({dataset}):")
        header = ["Metric", baseline] + [f"{m} (delta)" for m in other_models]
        print("| " + " | ".join(header) + " |")
        print("|" + "|".join(["---"] * len(header)) + "|")
        for metric in present:
            base_v = metrics[metric].get(baseline)
            row = [metric, _fmt(base_v, 4)]
            for other in other_models:
                row.append(_fmt_delta(metrics[metric].get(other), base_v, 4))
            print("| " + " | ".join(row) + " |")
        print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs-root", required=True)
    ap.add_argument(
        "--baseline", default="exp1_no_distill",
        help="model name to compute diff/rel%% against (must match a subdir name under --logs-root)",
    )
    ap.add_argument("--out", default=None, help="optional: also write all-runs CSV here")
    ap.add_argument(
        "--compact", action="store_true",
        help="print only the headline metrics (zero-top1/few1/few5 + geometry) as paste-ready markdown, instead of the full metric-by-metric table",
    )
    args = ap.parse_args()

    model_names = sorted(
        d for d in os.listdir(args.logs_root)
        if os.path.isdir(os.path.join(args.logs_root, d))
    )

    rows = []
    for model_name in model_names:
        model_dir = os.path.join(args.logs_root, model_name)
        for run_name in sorted(os.listdir(model_dir)):
            run_dir = os.path.join(model_dir, run_name)
            params_path = os.path.join(run_dir, "params.txt")
            if not os.path.isfile(params_path):
                continue
            params = parse_params(params_path)
            dataset = dataset_name(params)

            if "geometry_eval" in run_name:
                geo_path = os.path.join(run_dir, "geometry_results.json")
                if not os.path.isfile(geo_path):
                    continue
                results = parse_geometry_results(geo_path)
            elif "classification" in run_name:
                out_log_path = os.path.join(run_dir, "out.log")
                if not os.path.isfile(out_log_path):
                    continue
                results = parse_classification_results(out_log_path)
            else:
                continue

            for metric, value in results.items():
                rows.append({
                    "model": model_name,
                    "run_dir": run_name,
                    "dataset": dataset,
                    "data_root": params.get("data_root", ""),
                    "metric": metric,
                    "value": value,
                })

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True) if os.path.dirname(args.out) else None
        with open(args.out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["model", "run_dir", "dataset", "data_root", "metric", "value"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {len(rows)} rows to {args.out}\n")

    # --- keep only the latest run per (model, dataset) -- collapses duplicate submissions ---
    latest_run = {}  # (model, dataset) -> run_name
    for r in rows:
        key = (r["model"], r["dataset"])
        if key not in latest_run or r["run_dir"] > latest_run[key]:
            latest_run[key] = r["run_dir"]

    pivot = {}
    for r in rows:
        if latest_run[(r["model"], r["dataset"])] != r["run_dir"]:
            continue
        pivot.setdefault((r["dataset"], r["metric"]), {})[r["model"]] = r["value"]

    if args.compact:
        by_dataset = {}
        for (dataset, metric), vals in pivot.items():
            by_dataset.setdefault(dataset, {})[metric] = vals
        print_compact_markdown(by_dataset, model_names, args.baseline)
        return

    other_models = [m for m in model_names if m != args.baseline]

    col_widths = {"dataset": 22, "metric": 26, "model": 9, "diff": 8, "rel": 8}
    header_parts = [f"{'dataset':<{col_widths['dataset']}}", f"{'metric':<{col_widths['metric']}}"]
    for m in model_names:
        header_parts.append(f"{m:>{col_widths['model']}}")
    for m in other_models:
        header_parts.append(f"{'diff(' + m + ')':>{col_widths['diff']}}")
        header_parts.append(f"{'rel%(' + m + ')':>{col_widths['rel']}}")
    header = " ".join(header_parts)
    print(header)
    print("-" * len(header))

    for (dataset, metric) in sorted(pivot):
        vals = pivot[(dataset, metric)]
        row_parts = [f"{dataset:<{col_widths['dataset']}}", f"{metric:<{col_widths['metric']}}"]
        for m in model_names:
            v = vals.get(m)
            row_parts.append(f"{(f'{v:.4f}' if v is not None else 'n/a'):>{col_widths['model']}}")
        baseline_v = vals.get(args.baseline)
        for m in other_models:
            v = vals.get(m)
            if v is None or baseline_v is None:
                diff_str = rel_str = "n/a"
            else:
                diff = v - baseline_v
                rel = (diff / baseline_v * 100) if baseline_v else float("nan")
                diff_str = f"{diff:+.4f}"
                rel_str = f"{rel:+.1f}%"
            row_parts.append(f"{diff_str:>{col_widths['diff']}}")
            row_parts.append(f"{rel_str:>{col_widths['rel']}}")
        print(" ".join(row_parts))


if __name__ == "__main__":
    main()
