#!/usr/bin/env python3
import csv
import json
import os
import re
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = PROJECT_ROOT / "runs"
RUN_PATTERN = os.environ.get("RUN_PATTERN", "binary_nonbenign_features_cache_96slice_resnet18_fold{fold}_meanmax_age_sex_fusion")
PRED_FILE = os.environ.get("PRED_FILE", "test_predictions.csv")
OUT_NAME = os.environ.get("OUT_NAME", "pooled_" + re.sub(r"[^A-Za-z0-9_.-]+", "_", RUN_PATTERN.replace("{fold}", "foldx")))
OUT_DIR = RUN_ROOT / OUT_NAME
BOOTSTRAP_N = int(os.environ.get("BOOTSTRAP_N", "2000"))
SEED = int(os.environ.get("SEED", "20260704"))
LABEL5_NAMES = ["肉瘤类", "良性神经源性肿瘤", "PPGL", "淋巴瘤", "胃肠道间质瘤"]


def read_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_rows(path, rows):
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def metric_dict(rows):
    y = np.asarray([int(r["true_id"]) for r in rows])
    p = np.asarray([float(r["prob_nonbenign_actionable"]) for r in rows])
    pred = np.asarray([int(r.get("pred_id", float(r["prob_nonbenign_actionable"]) >= 0.5)) for r in rows])
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    out = {
        "n": int(len(rows)),
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "macro_f1": f1_score(y, pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y, pred, average="weighted", zero_division=0),
        "sensitivity": tp / (tp + fn) if tp + fn else 0.0,
        "specificity": tn / (tn + fp) if tn + fp else 0.0,
        "ppv": tp / (tp + fp) if tp + fp else 0.0,
        "npv": tn / (tn + fn) if tn + fn else 0.0,
        "brier": brier_score_loss(y, p),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }
    if len(set(y.tolist())) == 2:
        out["auroc"] = roc_auc_score(y, p)
        out["average_precision"] = average_precision_score(y, p)
        logit = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1 - 1e-6)).reshape(-1, 1)
        clf = LogisticRegression(C=1e6, solver="lbfgs").fit(logit, y)
        out["calibration_intercept"] = float(clf.intercept_[0])
        out["calibration_slope"] = float(clf.coef_[0, 0])
    return out


def subtype_rows(rows):
    out = []
    for label in LABEL5_NAMES:
        sub = [r for r in rows if r["true_label_5"] == label]
        if not sub:
            continue
        target = 0 if label == "良性神经源性肿瘤" else 1
        correct = sum(int(r["pred_id"]) == target for r in sub)
        out.append(
            {
                "label_5": label,
                "n": len(sub),
                "binary_target": target,
                "binary_recall": correct / len(sub),
                "mean_prob_nonbenign": float(np.mean([float(r["prob_nonbenign_actionable"]) for r in sub])),
            }
        )
    return out


def bootstrap_ci(rows):
    rng = np.random.default_rng(SEED)
    metrics = ["auroc", "average_precision", "balanced_accuracy", "sensitivity", "specificity", "ppv", "npv", "brier"]
    values = {m: [] for m in metrics}
    rows = list(rows)
    for _ in range(BOOTSTRAP_N):
        sample = [rows[i] for i in rng.integers(0, len(rows), len(rows))]
        m = metric_dict(sample)
        for key in metrics:
            if key in m:
                values[key].append(float(m[key]))
    return {
        key: {
            "mean": float(np.mean(vals)),
            "ci95_low": float(np.percentile(vals, 2.5)),
            "ci95_high": float(np.percentile(vals, 97.5)),
        }
        for key, vals in values.items()
        if vals
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pooled = []
    fold_summary = []
    for fold in range(5):
        run_dir = RUN_ROOT / RUN_PATTERN.format(fold=fold)
        rows = read_rows(run_dir / PRED_FILE)
        for r in rows:
            r["fold"] = fold
        pooled.extend(rows)
        m = metric_dict(rows)
        fold_summary.append({"fold": fold, **{k: json.dumps(v, ensure_ascii=False) if isinstance(v, list) else v for k, v in m.items()}})

    pooled_metrics = metric_dict(pooled)
    write_rows(OUT_DIR / "pooled_predictions.csv", pooled)
    write_rows(OUT_DIR / "fold_summary.csv", fold_summary)
    write_rows(OUT_DIR / "pooled_subtype_metrics.csv", subtype_rows(pooled))
    (OUT_DIR / "pooled_metrics.json").write_text(json.dumps(pooled_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_DIR / "bootstrap_ci.json").write_text(json.dumps(bootstrap_ci(pooled), ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_DIR / "config.json").write_text(
        json.dumps({"run_pattern": RUN_PATTERN, "pred_file": PRED_FILE, "bootstrap_n": BOOTSTRAP_N, "seed": SEED}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"out_dir": str(OUT_DIR), "metrics": pooled_metrics}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
