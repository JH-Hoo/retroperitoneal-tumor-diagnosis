#!/usr/bin/env python3
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = PROJECT_ROOT / "runs" / "5class_groupcv_fold0_resnet18_mil"
FIG_DIR = RUN_DIR / "figures"
REPORT_PATH = PROJECT_ROOT / "reports" / "5class_groupcv_fold0_report.md"
LABELS_DIR = PROJECT_ROOT / "data" / "labels"

CLASS_NAMES = ["肉瘤类", "良性神经源性肿瘤", "PPGL", "淋巴瘤", "胃肠道间质瘤"]
CLASS_NAMES_EN = ["Sarcoma", "Benign neurogenic", "PPGL", "Lymphoma", "GIST"]


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def metric_table(title, metrics):
    lines = [
        f"### {title}",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| accuracy | {metrics['accuracy']:.3f} |",
        f"| balanced accuracy | {metrics['balanced_accuracy']:.3f} |",
        f"| macro-F1 | {metrics['macro_f1']:.3f} |",
        f"| weighted-F1 | {metrics['weighted_f1']:.3f} |",
        "",
        "| Class | Recall |",
        "|---|---:|",
    ]
    for name, recall in zip(CLASS_NAMES, metrics["per_class_recall"]):
        lines.append(f"| {name} | {recall:.3f} |")
    return "\n".join(lines)


def plot_training(log_rows):
    epochs = [int(r["epoch"]) for r in log_rows]
    train_loss = [float(r["train_loss"]) for r in log_rows]
    train_acc = [float(r["train_accuracy"]) for r in log_rows]
    val_f1 = [float(r["macro_f1"]) for r in log_rows]

    fig, ax1 = plt.subplots(figsize=(8, 4.8), dpi=160)
    ax1.plot(epochs, train_loss, marker="o", label="Train loss", color="#1f77b4")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(alpha=0.25)
    ax2 = ax1.twinx()
    ax2.plot(epochs, train_acc, marker="s", label="Train accuracy", color="#2ca02c")
    ax2.plot(epochs, val_f1, marker="^", label="Val macro-F1", color="#d62728")
    ax2.set_ylabel("Score")
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [l.get_label() for l in lines], loc="center right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "training_curve.png")
    plt.close(fig)


def plot_class_distribution():
    rows = read_csv(LABELS_DIR / "fold_label_counts.csv")
    row = next(r for r in rows if r["fold"] == "0")
    counts = [int(row[name]) for name in CLASS_NAMES]
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=160)
    ax.bar(CLASS_NAMES_EN, counts, color="#4c78a8")
    ax.set_ylabel("Cases")
    ax.set_title("Fold 0 test-set class distribution")
    ax.tick_params(axis="x", rotation=18)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "class_distribution.png")
    plt.close(fig)


def plot_confusion(metrics, title, filename):
    cm = np.array(metrics["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(6.6, 5.6), dpi=160)
    image = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(np.arange(len(CLASS_NAMES_EN)))
    ax.set_yticks(np.arange(len(CLASS_NAMES_EN)))
    ax.set_xticklabels(CLASS_NAMES_EN, rotation=35, ha="right")
    ax.set_yticklabels(CLASS_NAMES_EN)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    threshold = cm.max() / 2 if cm.max() else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] > threshold else "black")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(FIG_DIR / filename)
    plt.close(fig)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    train_log = read_csv(RUN_DIR / "train_log.csv")
    val_metrics = json.loads((RUN_DIR / "val_metrics.json").read_text(encoding="utf-8"))
    test_metrics = json.loads((RUN_DIR / "test_metrics.json").read_text(encoding="utf-8"))
    summary = json.loads((LABELS_DIR / "dataset_summary.json").read_text(encoding="utf-8"))

    plot_training(train_log)
    plot_class_distribution()
    plot_confusion(val_metrics, "Validation confusion matrix", "val_confusion_matrix.png")
    plot_confusion(test_metrics, "Test confusion matrix", "test_confusion_matrix.png")

    best = max(train_log, key=lambda r: float(r["macro_f1"]))
    last = train_log[-1]
    report = f"""# 5-Class Group-CV Fold 0 Report

## Summary

This run is a lightweight smoke-test baseline for retroperitoneal tumor CT classification. It uses de-identified patient-level group split data, 96-slice three-window CT tensors, an ImageNet-pretrained ResNet18 backbone, and attention MIL pooling.

The result should be read as a pipeline baseline, not as a stable clinical-performance claim.

## Data

| Item | Value |
|---|---:|
| Cases | {summary['num_cases']} |
| Patients | {summary['num_patients']} |
| Split method | StratifiedGroupKFold by patient hash |
| Fold | 0 |
| Train / val / test | 147 / 50 / 49 |

![Class distribution](../runs/5class_groupcv_fold0_resnet18_mil/figures/class_distribution.png)

## Method

Each CT case is represented as `96 x 3 x 224 x 224`: 96 uniformly sampled axial slices and three CT windows per slice.

| Component | Setting |
|---|---|
| Backbone | ResNet18, ImageNet pretrained |
| Pooling | Attention MIL |
| Classes | 肉瘤类, 良性神经源性肿瘤, PPGL, 淋巴瘤, 胃肠道间质瘤 |
| Training | Freeze backbone for 5 epochs, then unfreeze layer4 |
| BatchNorm | Frozen/eval |
| Loss | Class-weighted cross entropy |

## Training

Best validation macro-F1 occurred at epoch {best['epoch']} with macro-F1 `{float(best['macro_f1']):.3f}`. The final training accuracy reached `{float(last['train_accuracy']):.3f}`, which indicates clear overfitting.

![Training curve](../runs/5class_groupcv_fold0_resnet18_mil/figures/training_curve.png)

## Validation

{metric_table('Validation Metrics', val_metrics)}

![Validation confusion matrix](../runs/5class_groupcv_fold0_resnet18_mil/figures/val_confusion_matrix.png)

## Test

{metric_table('Test Metrics', test_metrics)}

![Test confusion matrix](../runs/5class_groupcv_fold0_resnet18_mil/figures/test_confusion_matrix.png)

## Interpretation

The model mainly recognizes sarcoma and part of benign neurogenic tumors. PPGL and lymphoma remain weak, and GIST recall is still zero. The main bottleneck is likely not model size, but limited sample size, class imbalance, and the lack of lesion localization or body crop.
"""
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(REPORT_PATH)


if __name__ == "__main__":
    main()
