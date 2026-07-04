#!/usr/bin/env python3
import csv
import itertools
import json
import os
from pathlib import Path

import numpy as np
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
FUSION_NAME = os.environ.get("FUSION_NAME", "late_fusion")
RUN_PATTERNS = [x.strip() for x in os.environ.get("FUSION_RUNS", "").split(",") if x.strip()]
OUT_DIR = RUN_ROOT / f"fusion_{FUSION_NAME}"
WEIGHT_GRID = [float(x) for x in os.environ.get("WEIGHT_GRID", "0,0.25,0.5,0.75,1").split(",")]
FUSION_SELECT = os.environ.get("FUSION_SELECT", "auroc")
THRESHOLD_MODE = os.environ.get("THRESHOLD_MODE", "sens90")
MIN_SENSITIVITY = float(os.environ.get("MIN_SENSITIVITY", "0.90"))
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


def prediction_file(run_dir, split):
    fixed = run_dir / f"{split}_predictions_fixed05.csv"
    return fixed if fixed.exists() else run_dir / f"{split}_predictions.csv"


def load_model_preds(pattern, fold, split, model_idx):
    rows = read_rows(prediction_file(RUN_ROOT / pattern.format(fold=fold), split))
    out = {}
    for r in rows:
        out[r["case_id"]] = {
            "case_id": r["case_id"],
            "fold": fold,
            "true_label_5": r["true_label_5"],
            "true_id": int(r["true_id"]),
            f"prob_model{model_idx}": float(r["prob_nonbenign_actionable"]),
        }
    return out


def merge_preds(fold, split):
    merged = None
    for i, pattern in enumerate(RUN_PATTERNS):
        pred = load_model_preds(pattern, fold, split, i)
        if merged is None:
            merged = pred
        else:
            for case_id, row in merged.items():
                row.update(pred[case_id])
    return list(merged.values())


def weight_grid(n):
    for weights in itertools.product(WEIGHT_GRID, repeat=n):
        if abs(sum(weights) - 1.0) < 1e-9:
            yield np.asarray(weights, dtype=np.float64)


def add_fused_prob(rows, weights):
    out = []
    prob_cols = [f"prob_model{i}" for i in range(len(weights))]
    for r in rows:
        rr = dict(r)
        rr["prob_nonbenign_actionable"] = float(sum(weights[i] * float(rr[prob_cols[i]]) for i in range(len(weights))))
        out.append(rr)
    return out


def add_predictions(rows, threshold):
    out = []
    for r in rows:
        rr = dict(r)
        rr["threshold"] = float(threshold)
        rr["pred_id"] = int(float(rr["prob_nonbenign_actionable"]) >= threshold)
        rr["pred_label"] = "nonbenign_actionable" if rr["pred_id"] else "benign_neurogenic"
        out.append(rr)
    return out


def metrics(rows):
    y = np.asarray([int(r["true_id"]) for r in rows])
    p = np.asarray([float(r["prob_nonbenign_actionable"]) for r in rows])
    pred = np.asarray([int(r["pred_id"]) for r in rows])
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
    return out


def choose_threshold(rows, mode, min_sensitivity):
    if mode == "fixed_05":
        return 0.5
    if mode == "sens90":
        mode, min_sensitivity = "screening", 0.90
    if mode == "sens85":
        mode, min_sensitivity = "screening", 0.85
    y = np.asarray([int(r["true_id"]) for r in rows])
    p = np.asarray([float(r["prob_nonbenign_actionable"]) for r in rows])
    unique_p = np.sort(np.unique(p))
    midpoints = (unique_p[:-1] + unique_p[1:]) / 2 if len(unique_p) > 1 else unique_p
    best_t, best_key = 0.5, None
    for t in np.unique(np.r_[0.0, midpoints, 1.0]):
        pred = (p >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        sens = tp / (tp + fn) if tp + fn else 0.0
        spec = tn / (tn + fp) if tn + fp else 0.0
        bacc = 0.5 * (sens + spec)
        if mode == "youden":
            key = (sens + spec - 1.0, t, -abs(t - 0.5))
        elif mode == "balanced_accuracy":
            key = (bacc, t, -abs(t - 0.5))
        elif mode == "screening":
            key = (1, spec, t, -abs(t - 0.5)) if sens >= min_sensitivity else (0, sens, spec, t)
        else:
            raise ValueError(f"unknown THRESHOLD_MODE: {mode}")
        if best_key is None or key > best_key:
            best_t, best_key = float(t), key
    return best_t


def selection_score(rows):
    y = np.asarray([int(r["true_id"]) for r in rows])
    p = np.asarray([float(r["prob_nonbenign_actionable"]) for r in rows])
    if FUSION_SELECT == "auroc":
        return roc_auc_score(y, p)
    if FUSION_SELECT == "average_precision":
        return average_precision_score(y, p)
    if FUSION_SELECT == "rank_score":
        return 0.7 * roc_auc_score(y, p) + 0.3 * average_precision_score(y, p)
    if FUSION_SELECT == "screening":
        pred_rows = add_predictions(rows, choose_threshold(rows, "screening", MIN_SENSITIVITY))
        m = metrics(pred_rows)
        return m["specificity"] if m["sensitivity"] >= MIN_SENSITIVITY else m["sensitivity"] - MIN_SENSITIVITY - 1.0
    raise ValueError(f"unknown FUSION_SELECT: {FUSION_SELECT}")


def subtype_rows(rows):
    out = []
    for label in LABEL5_NAMES:
        sub = [r for r in rows if r["true_label_5"] == label]
        if not sub:
            continue
        target = 0 if label == "良性神经源性肿瘤" else 1
        out.append(
            {
                "label_5": label,
                "n": len(sub),
                "binary_target": target,
                "binary_recall": sum(int(r["pred_id"]) == target for r in sub) / len(sub),
                "mean_prob_nonbenign": float(np.mean([float(r["prob_nonbenign_actionable"]) for r in sub])),
            }
        )
    return out


def main():
    if not RUN_PATTERNS:
        raise SystemExit("Set FUSION_RUNS to comma-separated run patterns containing {fold}.")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_val, all_test, fold_rows, weight_rows = [], [], [], []
    for fold in range(5):
        val_base = merge_preds(fold, "val")
        test_base = merge_preds(fold, "test")
        best_weights, best_score = None, -1e18
        for weights in weight_grid(len(RUN_PATTERNS)):
            val = add_fused_prob(val_base, weights)
            score = selection_score(val)
            if score > best_score:
                best_weights, best_score = weights, score
        val = add_fused_prob(val_base, best_weights)
        test = add_fused_prob(test_base, best_weights)
        threshold = choose_threshold(val, THRESHOLD_MODE, MIN_SENSITIVITY)
        val = add_predictions(val, threshold)
        test = add_predictions(test, threshold)
        all_val.extend(val)
        all_test.extend(test)
        fold_metric = metrics(test)
        fold_rows.append({"fold": fold, **{k: json.dumps(v, ensure_ascii=False) if isinstance(v, list) else v for k, v in fold_metric.items()}})
        weight_rows.append(
            {
                "fold": fold,
                "threshold": threshold,
                "selection_score": best_score,
                **{f"weight_model{i}": float(best_weights[i]) for i in range(len(best_weights))},
            }
        )

    write_rows(OUT_DIR / "val_predictions.csv", all_val)
    write_rows(OUT_DIR / "test_predictions.csv", all_test)
    write_rows(OUT_DIR / "fold_summary.csv", fold_rows)
    write_rows(OUT_DIR / "fusion_weights.csv", weight_rows)
    write_rows(OUT_DIR / "test_subtype_metrics.csv", subtype_rows(all_test))
    (OUT_DIR / "test_metrics.json").write_text(json.dumps(metrics(all_test), ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_DIR / "config.json").write_text(
        json.dumps(
            {
                "fusion_name": FUSION_NAME,
                "run_patterns": RUN_PATTERNS,
                "weight_grid": WEIGHT_GRID,
                "fusion_select": FUSION_SELECT,
                "threshold_mode": THRESHOLD_MODE,
                "min_sensitivity": MIN_SENSITIVITY,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"out_dir": str(OUT_DIR), "metrics": metrics(all_test)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
