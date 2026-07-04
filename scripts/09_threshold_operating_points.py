#!/usr/bin/env python3
import csv
import json
import os
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


PRED_CSV = Path(os.environ["PRED_CSV"])
OUT_DIR = Path(os.environ.get("OUT_DIR", PRED_CSV.parent / "threshold_operating_points"))
TARGET_SENS = [float(x) for x in os.environ.get("TARGET_SENS", "0.95,0.93,0.90,0.85,0.80").split(",")]


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


def metrics(y, p, threshold):
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if tp + fn else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    return {
        "threshold": float(threshold),
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "macro_f1": f1_score(y, pred, average="macro", zero_division=0),
        "sensitivity": sens,
        "specificity": spec,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def candidate_thresholds(p):
    unique = np.sort(np.unique(p))
    mid = (unique[:-1] + unique[1:]) / 2 if len(unique) > 1 else unique
    return np.unique(np.r_[0.0, mid, 1.0])


def main():
    rows = read_rows(PRED_CSV)
    y = np.asarray([int(r["true_id"]) for r in rows])
    p = np.asarray([float(r["prob_nonbenign_actionable"]) for r in rows])
    thresholds = candidate_thresholds(p)
    all_rows = []
    best_bacc, best_youden = None, None
    for threshold in thresholds:
        m = metrics(y, p, threshold)
        all_rows.append(m)
        if best_bacc is None or m["balanced_accuracy"] > best_bacc["balanced_accuracy"]:
            best_bacc = m
        youden = m["sensitivity"] + m["specificity"] - 1
        if best_youden is None or youden > best_youden["youden"]:
            best_youden = {**m, "youden": youden}

    target_rows = []
    for target in TARGET_SENS:
        feasible = [m for m in all_rows if m["sensitivity"] >= target]
        chosen = max(feasible, key=lambda x: (x["specificity"], x["threshold"])) if feasible else max(all_rows, key=lambda x: x["sensitivity"])
        target_rows.append({"target_sensitivity": target, **chosen})

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    flat_all = [{k: json.dumps(v, ensure_ascii=False) if isinstance(v, list) else v for k, v in m.items()} for m in all_rows]
    flat_targets = [{k: json.dumps(v, ensure_ascii=False) if isinstance(v, list) else v for k, v in m.items()} for m in target_rows]
    write_rows(OUT_DIR / "threshold_curve.csv", flat_all)
    write_rows(OUT_DIR / "target_sensitivity_points.csv", flat_targets)
    summary = {
        "pred_csv": str(PRED_CSV),
        "n": int(len(rows)),
        "auroc": roc_auc_score(y, p),
        "average_precision": average_precision_score(y, p),
        "best_balanced_accuracy": best_bacc,
        "best_youden": best_youden,
        "target_sensitivity_points": target_rows,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
