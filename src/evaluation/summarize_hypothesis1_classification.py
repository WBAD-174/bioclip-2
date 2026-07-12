#!/usr/bin/env python3
"""
Aggregate all classification.py runs under eval_hypothesis1's log dirs and print a
teacher-vs-student comparison table straight to stdout (paste-friendly -- no file to
cat afterward). Pass --out to also dump the full (all runs, not just latest) data as
CSV.

Usage (on the cluster, where the logs actually live):
    python -m src.evaluation.summarize_hypothesis1_classification \
        --logs-root /fs/scratch/PAS2136/chenxujiang/bioclip-hypothesis1-eval/logs

Only the most recent run per (model, dataset) is used for the printed table -- eval_hypothesis1.sh
gets re-run multiple times across days, and run_name's leading timestamp sorts lexically, so
"most recent" = max(run_name) per (model, dataset).

For each <logs-root>/{teacher,student}/*classification*/ dir, reads params.txt for
data_root/label_filename (to identify the dataset) and out.log for the "Results:"
block (metric: value% lines).
"""
import argparse
import csv
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


def parse_results(out_log_path):
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
                # first non-matching line after the block ends it
                if results:
                    break
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs-root", required=True)
    ap.add_argument(
        "--baseline", default="teacher",
        help="model name to compute diff/rel%% against (must match a subdir name under --logs-root)",
    )
    ap.add_argument("--out", default=None, help="optional: also write all-runs CSV here")
    args = ap.parse_args()

    # Model names are just subdirectories of logs-root (e.g. "teacher", "student",
    # "contrastive_only") -- not hardcoded, so adding a new model to eval_hypothesis1.sh
    # (a new MODEL_NAMES entry / new mkdir under logs-root) is picked up automatically.
    model_names = sorted(
        d for d in os.listdir(args.logs_root)
        if os.path.isdir(os.path.join(args.logs_root, d))
    )

    rows = []
    for model_name in model_names:
        model_dir = os.path.join(args.logs_root, model_name)
        if not os.path.isdir(model_dir):
            continue
        for run_name in sorted(os.listdir(model_dir)):
            if "classification" not in run_name:
                continue
            run_dir = os.path.join(model_dir, run_name)
            params_path = os.path.join(run_dir, "params.txt")
            out_log_path = os.path.join(run_dir, "out.log")
            if not (os.path.isfile(params_path) and os.path.isfile(out_log_path)):
                continue

            params = parse_params(params_path)
            dataset = os.path.basename(params.get("label_filename", "").rstrip("/"))
            data_root = params.get("data_root", "")
            results = parse_results(out_log_path)

            for metric, value in results.items():
                rows.append({
                    "model": model_name,
                    "run_dir": run_name,
                    "dataset": dataset,
                    "data_root": data_root,
                    "metric": metric,
                    "value_pct": value,
                })

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["model", "run_dir", "dataset", "data_root", "metric", "value_pct"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {len(rows)} rows to {args.out}\n")

    # --- keep only the latest run per (model, dataset) ---
    latest_run = {}  # (model, dataset) -> run_name
    for r in rows:
        key = (r["model"], r["dataset"])
        if key not in latest_run or r["run_dir"] > latest_run[key]:
            latest_run[key] = r["run_dir"]

    # --- pivot: (dataset, metric) -> {model_name: value}, one column per discovered model ---
    pivot = {}
    for r in rows:
        if latest_run[(r["model"], r["dataset"])] != r["run_dir"]:
            continue
        pivot.setdefault((r["dataset"], r["metric"]), {})[r["model"]] = r["value_pct"]

    # non-baseline models get a "vs <baseline> diff/rel%" pair of columns each
    other_models = [m for m in model_names if m != args.baseline]

    col_widths = {"dataset": 22, "metric": 20, "model": 9, "diff": 8, "rel": 8}
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
            row_parts.append(f"{(f'{v:.2f}' if v is not None else 'n/a'):>{col_widths['model']}}")
        baseline_v = vals.get(args.baseline)
        for m in other_models:
            v = vals.get(m)
            if v is None or baseline_v is None:
                diff_str = rel_str = "n/a"
            else:
                diff = v - baseline_v
                rel = (diff / baseline_v * 100) if baseline_v else float("nan")
                diff_str = f"{diff:+.2f}"
                rel_str = f"{rel:+.1f}%"
            row_parts.append(f"{diff_str:>{col_widths['diff']}}")
            row_parts.append(f"{rel_str:>{col_widths['rel']}}")
        print(" ".join(row_parts))


if __name__ == "__main__":
    main()
