#!/usr/bin/env python3
import argparse
import csv
import json
import math
from pathlib import Path


METRIC_PATHS = {
    "clinical4_accuracy": ("clinical4", "accuracy"),
    "clinical4_balanced_accuracy": ("clinical4", "balanced_accuracy"),
    "clinical4_macro_f1": ("clinical4", "macro_f1"),
    "clinical4_top2_accuracy": ("clinical4", "top2_accuracy"),
    "binary_accuracy": ("binary_head", "accuracy"),
    "binary_balanced_accuracy": ("binary_head", "balanced_accuracy"),
    "binary_macro_f1": ("binary_head", "macro_f1"),
    "binary_risk_recall": ("binary_head", "risk_workup_recall"),
    "binary_benign_recall": ("binary_head", "benign_like_recall"),
}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def metric(summary, path):
    d = summary.get("oof_metrics", {})
    for key in path:
        d = d.get(key, {})
    return float(d)


def summarize(values):
    n = len(values)
    mean = sum(values) / max(n, 1)
    if n <= 1:
        return {"mean": mean, "std": 0.0, "ci95_low": mean, "ci95_high": mean}
    var = sum((x - mean) ** 2 for x in values) / (n - 1)
    std = math.sqrt(var)
    half = 1.96 * std / math.sqrt(n)
    return {"mean": mean, "std": std, "ci95_low": mean - half, "ci95_high": mean + half}


def main():
    parser = argparse.ArgumentParser(description="Summarize repeated 5-fold CV report directories.")
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    run_dirs = sorted([p for p in args.runs_root.glob("seed_*") if (p / "summary.json").exists()])
    seed_rows = []
    for run_dir in run_dirs:
        summary = read_json(run_dir / "summary.json")
        row = {"run": run_dir.name, "num_rows": summary.get("num_rows", "")}
        for name, path in METRIC_PATHS.items():
            row[name] = metric(summary, path)
        seed_rows.append(row)
    aggregate = {
        name: summarize([float(row[name]) for row in seed_rows])
        for name in METRIC_PATHS
    }
    payload = {"num_runs": len(seed_rows), "runs_root": str(args.runs_root), "seed_metrics": seed_rows, "aggregate": aggregate}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "repeated_cv_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_rows(args.out_dir / "seed_metrics.csv", seed_rows)
    lines = ["# Repeated 5-fold CV Summary", "", f"Runs: {len(seed_rows)}", ""]
    lines.append("| Metric | Mean | 95% CI | Std |")
    lines.append("|---|---:|---:|---:|")
    for name, stats in aggregate.items():
        lines.append(
            f"| {name} | {stats['mean']:.3f} | "
            f"{stats['ci95_low']:.3f}-{stats['ci95_high']:.3f} | {stats['std']:.3f} |"
        )
    (args.out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload["aggregate"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
