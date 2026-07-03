#!/usr/bin/env python3
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = PROJECT_ROOT / "runs"
REPORTS_ROOT = PROJECT_ROOT / "reports"
ASSETS_ROOT = REPORTS_ROOT / "assets"

TASKS = ["5class", "sarcoma", "ppgl", "lymphoma"]
TASK_LABELS = {
    "5class": "Five-class",
    "sarcoma": "Sarcoma vs non",
    "ppgl": "PPGL vs non",
    "lymphoma": "Lymphoma vs non",
}
FEATURES = [
    ("Whole", "features_cache_96slice_resnet18"),
    ("Body crop", "features_cache_body_96slice_resnet18"),
]
FOLDS = range(5)
POOLING = "meanmax"
METRICS = ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "auroc", "average_precision", "positive_recall"]


def read_metrics(task, feature_name, fold, split):
    path = RUNS_ROOT / f"{task}_{feature_name}_fold{fold}_{POOLING}" / f"{split}_metrics.json"
    return json.loads(path.read_text(encoding="utf-8"))


def values_for(task, feature_name, split):
    rows = []
    for fold in FOLDS:
        m = read_metrics(task, feature_name, fold, split)
        item = dict(m)
        item["fold"] = fold
        item["positive_recall"] = m["per_class_recall"][1] if task != "5class" else None
        rows.append(item)
    return rows


def mean_std(items, key):
    vals = [x[key] for x in items if x.get(key) is not None]
    if not vals:
        return None, None
    arr = np.asarray(vals, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=1))


def fmt(mean, std):
    if mean is None:
        return ""
    return f"{mean:.3f} +/- {std:.3f}"


def write_rows(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def build_summary():
    summary = []
    for task in TASKS:
        for input_name, feature_name in FEATURES:
            for split in ["val", "test"]:
                items = values_for(task, feature_name, split)
                row = {
                    "task": task,
                    "task_label": TASK_LABELS[task],
                    "input": input_name,
                    "split": split,
                    "n_folds": len(items),
                }
                for metric in METRICS:
                    mean, std = mean_std(items, metric)
                    row[f"{metric}_mean"] = mean
                    row[f"{metric}_std"] = std
                    row[f"{metric}_mean_std"] = fmt(mean, std)
                summary.append(row)
    return summary


def rows_for_split(summary, split):
    return [r for r in summary if r["split"] == split]


def plot_core_metrics(summary):
    rows = rows_for_split(summary, "test")
    labels = [f"{r['task_label']}\n{r['input']}" for r in rows]
    x = np.arange(len(rows))
    width = 0.38
    bal = [r["balanced_accuracy_mean"] for r in rows]
    bal_std = [r["balanced_accuracy_std"] for r in rows]
    macro = [r["macro_f1_mean"] for r in rows]
    macro_std = [r["macro_f1_std"] for r in rows]

    fig, ax = plt.subplots(figsize=(12, 5.6), dpi=180)
    ax.bar(x - width / 2, bal, width, yerr=bal_std, capsize=3, label="Balanced accuracy", color="#4c78a8")
    ax.bar(x + width / 2, macro, width, yerr=macro_std, capsize=3, label="Macro-F1", color="#f58518")
    ax.set_ylim(0, 0.85)
    ax.set_ylabel("Mean score across 5 folds")
    ax.set_title("Frozen ResNet18 meanmax MIL head: 5-fold test metrics")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(ASSETS_ROOT / "meanmax_5fold_test_core_metrics.png")
    plt.close(fig)


def plot_binary_metrics(summary):
    rows = [r for r in rows_for_split(summary, "test") if r["task"] != "5class"]
    labels = [f"{r['task_label']}\n{r['input']}" for r in rows]
    x = np.arange(len(rows))
    width = 0.38
    recall = [r["positive_recall_mean"] for r in rows]
    recall_std = [r["positive_recall_std"] for r in rows]
    auroc = [r["auroc_mean"] for r in rows]
    auroc_std = [r["auroc_std"] for r in rows]

    fig, ax = plt.subplots(figsize=(10, 5.2), dpi=180)
    ax.bar(x - width / 2, recall, width, yerr=recall_std, capsize=3, label="Positive recall", color="#54a24b")
    ax.bar(x + width / 2, auroc, width, yerr=auroc_std, capsize=3, label="AUROC", color="#b279a2")
    ax.set_ylim(0, 0.85)
    ax.set_ylabel("Mean score across 5 folds")
    ax.set_title("Binary endpoints: 5-fold positive recall and AUROC")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(ASSETS_ROOT / "meanmax_5fold_binary_recall_auroc.png")
    plt.close(fig)


def markdown_table(rows, metrics):
    header = ["Task", "Input"] + metrics
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
    for r in rows:
        vals = [r["task_label"], r["input"]]
        vals += [r[f"{m}_mean_std"] for m in metrics]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_report(summary):
    test_rows = rows_for_split(summary, "test")
    val_rows = rows_for_split(summary, "val")
    binary_rows = [r for r in test_rows if r["task"] != "5class"]
    text = f"""# Frozen-Feature Meanmax MIL 5-Fold Report

This report updates the fold 0 smoke test to a patient-level 5-fold summary. It compares whole-abdomen and simple body-crop inputs using the same de-identified labels and folds.

The model is intentionally small: a frozen ImageNet ResNet18 extracts 96 slice features per case, and a mean+max MIL head trains only a linear classifier on pooled features.

## Test Summary

![5-fold core metrics](assets/meanmax_5fold_test_core_metrics.png)

{markdown_table(test_rows, ["balanced_accuracy", "macro_f1", "weighted_f1", "accuracy", "auroc", "average_precision"])}

## Validation Summary

{markdown_table(val_rows, ["balanced_accuracy", "macro_f1", "weighted_f1", "accuracy", "auroc", "average_precision"])}

## Binary Endpoint Detail

![5-fold binary metrics](assets/meanmax_5fold_binary_recall_auroc.png)

{markdown_table(binary_rows, ["positive_recall", "balanced_accuracy", "macro_f1", "auroc", "average_precision"])}

## Interpretation

- Five-class classification remains weak and should stay exploratory, especially because the GIST class is small.
- The binary endpoints are more interpretable and should be the main reporting line for this stage.
- Body crop is not uniformly better. Its value depends on the endpoint, so the next useful input improvement is a more anatomy-aware retroperitoneal crop or a small lesion-bbox subset.
- These results still do not support jumping to a whole-volume 3D model. The priority remains better localization/crop before larger models.

## Artifacts

- Summary CSV: `reports/frozen_feature_meanmax_5fold_summary.csv`
- Summary JSON: `reports/frozen_feature_meanmax_5fold_summary.json`
- Core metric figure: `reports/assets/meanmax_5fold_test_core_metrics.png`
- Binary metric figure: `reports/assets/meanmax_5fold_binary_recall_auroc.png`
"""
    (REPORTS_ROOT / "frozen_feature_meanmax_5fold_report.md").write_text(text, encoding="utf-8")


def main():
    REPORTS_ROOT.mkdir(parents=True, exist_ok=True)
    ASSETS_ROOT.mkdir(parents=True, exist_ok=True)
    summary = build_summary()
    write_rows(REPORTS_ROOT / "frozen_feature_meanmax_5fold_summary.csv", summary)
    (REPORTS_ROOT / "frozen_feature_meanmax_5fold_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    plot_core_metrics(summary)
    plot_binary_metrics(summary)
    write_report(summary)
    print(REPORTS_ROOT / "frozen_feature_meanmax_5fold_report.md")


if __name__ == "__main__":
    main()
