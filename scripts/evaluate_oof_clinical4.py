#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
)


CLINICAL4_CLASS_NAMES = ["sarcoma/GIST-like", "lymphoma", "PPGL", "benign neurogenic"]
CLINICAL4_PROB_COLUMNS = ["prob_sarcoma_gist_like", "prob_lymphoma", "prob_ppgl", "prob_benign_neurogenic"]
BINARY_NAMES = ["risk/workup", "benign-like"]
BENIGN_ID = 3


def read_rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields, seen = [], set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def as_float(row, key, default=0.0):
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return float(default)


def clinical4_arrays(rows):
    y = np.asarray([int(r["true_clinical4_id"]) for r in rows], dtype=int)
    probs = np.asarray([[as_float(r, c) for c in CLINICAL4_PROB_COLUMNS] for r in rows], dtype=float)
    return y, probs


def binary_arrays_from_rows(rows, prefix):
    if prefix == "binary_head":
        y = np.asarray([int(r["binary_head_true_binary_id"]) for r in rows], dtype=int)
        probs = np.asarray(
            [[as_float(r, "prob_binary_head_risk_workup"), as_float(r, "prob_binary_head_benign_like")] for r in rows],
            dtype=float,
        )
    elif prefix == "derived":
        y = np.asarray([int(r["derived_true_binary_id"]) for r in rows], dtype=int)
        probs = np.asarray([[as_float(r, "prob_risk_workup"), as_float(r, "prob_benign_like")] for r in rows], dtype=float)
    else:
        raise ValueError(prefix)
    return y, probs


def clinical4_metrics(y, probs):
    pred = probs.argmax(axis=1)
    top2 = np.argsort(probs, axis=1)[:, -2:]
    labels = list(range(len(CLINICAL4_CLASS_NAMES)))
    out = {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y, pred, average="weighted", zero_division=0)),
        "top2_accuracy": float(np.mean([yy in tt for yy, tt in zip(y, top2)])),
        "confusion_matrix": confusion_matrix(y, pred, labels=labels).tolist(),
    }
    per_recall = recall_score(y, pred, labels=labels, average=None, zero_division=0)
    out["per_class_recall"] = {CLINICAL4_CLASS_NAMES[i]: float(per_recall[i]) for i in labels}
    out["per_class_top2_recall"] = {
        CLINICAL4_CLASS_NAMES[i]: float(np.mean([i in tt for yy, tt in zip(y, top2) if yy == i])) if np.any(y == i) else 0.0
        for i in labels
    }
    out["one_vs_rest_auc"] = {}
    out["one_vs_rest_pr_auc"] = {}
    for i, name in enumerate(CLINICAL4_CLASS_NAMES):
        yy = (y == i).astype(int)
        if len(np.unique(yy)) < 2:
            out["one_vs_rest_auc"][name] = None
            out["one_vs_rest_pr_auc"][name] = None
            continue
        out["one_vs_rest_auc"][name] = float(roc_auc_score(yy, probs[:, i]))
        out["one_vs_rest_pr_auc"][name] = float(average_precision_score(yy, probs[:, i]))
    auc_vals = [v for v in out["one_vs_rest_auc"].values() if v is not None]
    pr_vals = [v for v in out["one_vs_rest_pr_auc"].values() if v is not None]
    out["macro_ovr_auc"] = float(np.mean(auc_vals)) if auc_vals else None
    out["macro_ovr_pr_auc"] = float(np.mean(pr_vals)) if pr_vals else None
    return out


def binary_metrics(y, probs, threshold=None):
    if threshold is None:
        pred = probs.argmax(axis=1)
    else:
        pred = (probs[:, 0] < float(threshold)).astype(int)
    recall = recall_score(y, pred, labels=[0, 1], average=None, zero_division=0)
    out = {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "risk_workup_recall": float(recall[0]),
        "benign_like_recall": float(recall[1]),
        "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
    }
    if len(np.unique(y)) == 2:
        benign_prob = probs[:, 1]
        out["benign_like_auc"] = float(roc_auc_score(y, benign_prob))
        out["benign_like_pr_auc"] = float(average_precision_score(y, benign_prob))
        out["brier_benign_like"] = float(brier_score_loss(y, benign_prob))
        out["ece_benign_like"] = float(expected_calibration_error(y, benign_prob))
    return out


def expected_calibration_error(y, prob, bins=10):
    y = np.asarray(y, dtype=float)
    prob = np.asarray(prob, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (prob >= lo) & (prob < hi if hi < 1.0 else prob <= hi)
        if not np.any(mask):
            continue
        ece += float(mask.mean()) * abs(float(y[mask].mean()) - float(prob[mask].mean()))
    return ece


def threshold_curve(y, probs):
    thresholds = np.unique(np.concatenate([np.linspace(0.0, 1.0, 201), probs[:, 0]]))
    rows = []
    for thr in thresholds:
        metrics = binary_metrics(y, probs, threshold=float(thr))
        rows.append(
            {
                "risk_threshold": float(thr),
                "accuracy": metrics["accuracy"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "macro_f1": metrics["macro_f1"],
                "risk_workup_recall": metrics["risk_workup_recall"],
                "benign_like_recall": metrics["benign_like_recall"],
            }
        )
    return rows


def best_threshold_at_min_risk_recall(rows, min_risk_recall):
    valid = [r for r in rows if float(r["risk_workup_recall"]) >= min_risk_recall]
    if not valid:
        return None
    valid.sort(key=lambda r: (float(r["benign_like_recall"]), float(r["balanced_accuracy"]), -float(r["risk_threshold"])), reverse=True)
    return valid[0]


def bootstrap_ci(y, probs, metric_fn, n_boot=2000, seed=20260709):
    rng = np.random.default_rng(seed)
    n = len(y)
    values = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yy = y[idx]
        if len(np.unique(yy)) < 2:
            continue
        try:
            vals = metric_fn(yy, probs[idx])
        except ValueError:
            continue
        values.append(vals)
    if not values:
        return {}
    keys = values[0].keys()
    out = {}
    for key in keys:
        arr = np.asarray([v[key] for v in values if isinstance(v.get(key), (int, float))], dtype=float)
        if len(arr) == 0:
            continue
        out[key] = {
            "mean": float(arr.mean()),
            "ci95_low": float(np.percentile(arr, 2.5)),
            "ci95_high": float(np.percentile(arr, 97.5)),
        }
    return out


def compact_clinical4_metrics(y, probs):
    m = clinical4_metrics(y, probs)
    return {k: m[k] for k in ["accuracy", "balanced_accuracy", "macro_f1", "top2_accuracy"] if k in m}


def compact_binary_metrics(y, probs):
    m = binary_metrics(y, probs)
    return {k: m[k] for k in ["accuracy", "balanced_accuracy", "macro_f1", "risk_workup_recall", "benign_like_recall"] if k in m}


def plot_calibration(path, y, prob, title):
    path = Path(path)
    bins = np.linspace(0.0, 1.0, 11)
    xs, ys, ns = [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (prob >= lo) & (prob < hi if hi < 1.0 else prob <= hi)
        if np.any(mask):
            xs.append(float(prob[mask].mean()))
            ys.append(float(y[mask].mean()))
            ns.append(int(mask.sum()))
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "--", color="#64748b", linewidth=1)
    ax.plot(xs, ys, marker="o", color="#2563eb")
    for x, yy, n in zip(xs, ys, ns):
        ax.text(x, yy, str(n), fontsize=8, ha="center", va="bottom")
    ax.set_xlabel("Predicted benign-like probability")
    ax.set_ylabel("Observed benign-like frequency")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def load_cache_meta(path):
    if not path or not Path(path).exists():
        return {}
    return {row.get("group", ""): row for row in read_rows(path)}


def write_error_review(path, rows, cache_meta):
    out = []
    for row in rows:
        clinical4_error = int(row["true_clinical4_id"] != row["pred_clinical4_id"])
        binary_error = int(row["binary_head_true_binary_id"] != row["binary_head_pred_binary_id"])
        if not clinical4_error and not binary_error:
            continue
        meta = cache_meta.get(row["group"], {})
        out.append(
            {
                "group": row["group"],
                "label_5": row.get("label_5", ""),
                "true_clinical4": row["true_clinical4_label"],
                "pred_clinical4": row["pred_clinical4_label"],
                "top2_clinical4": row["top2_clinical4_label"],
                "clinical4_error": clinical4_error,
                "true_binary": row["binary_head_true_binary_label"],
                "pred_binary": row["binary_head_pred_binary_label"],
                "binary_head_error": binary_error,
                "fold": row.get("fold", ""),
                "top_slice_index_in_bag": row.get("top_slice_index_in_bag", ""),
                "top_slice_attention": row.get("top_slice_attention", ""),
                "tumor_voxels": meta.get("tumor_voxels", ""),
                "sample_status": meta.get("sample_status", ""),
                "crop_status": meta.get("crop_status", ""),
                "prob_sarcoma_gist_like": row.get("prob_sarcoma_gist_like", ""),
                "prob_lymphoma": row.get("prob_lymphoma", ""),
                "prob_ppgl": row.get("prob_ppgl", ""),
                "prob_benign_neurogenic": row.get("prob_benign_neurogenic", ""),
                "prob_binary_head_risk_workup": row.get("prob_binary_head_risk_workup", ""),
                "prob_binary_head_benign_like": row.get("prob_binary_head_benign_like", ""),
            }
        )
    write_rows(path, out)


def main():
    parser = argparse.ArgumentParser(description="Extended evaluation for clinical4 OOF predictions.")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--cache-all-csv", type=Path)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260709)
    parser.add_argument("--min-risk-recall", type=float, default=0.95)
    args = parser.parse_args()

    rows = read_rows(args.predictions)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    y4, p4 = clinical4_arrays(rows)
    y_derived, p_derived = binary_arrays_from_rows(rows, "derived")
    y_bin, p_bin = binary_arrays_from_rows(rows, "binary_head")

    curve_derived = threshold_curve(y_derived, p_derived)
    curve_binary = threshold_curve(y_bin, p_bin)
    write_rows(args.out_dir / "threshold_curve_derived_binary.csv", curve_derived)
    write_rows(args.out_dir / "threshold_curve_binary_head.csv", curve_binary)

    metrics = {
        "num_rows": len(rows),
        "clinical4": clinical4_metrics(y4, p4),
        "derived_binary_argmax": binary_metrics(y_derived, p_derived),
        "binary_head_argmax": binary_metrics(y_bin, p_bin),
        "derived_binary_threshold_at_min_risk_recall": best_threshold_at_min_risk_recall(curve_derived, args.min_risk_recall),
        "binary_head_threshold_at_min_risk_recall": best_threshold_at_min_risk_recall(curve_binary, args.min_risk_recall),
        "bootstrap_ci": {
            "clinical4": bootstrap_ci(y4, p4, compact_clinical4_metrics, args.bootstrap, args.seed),
            "derived_binary": bootstrap_ci(y_derived, p_derived, compact_binary_metrics, args.bootstrap, args.seed + 1),
            "binary_head": bootstrap_ci(y_bin, p_bin, compact_binary_metrics, args.bootstrap, args.seed + 2),
        },
    }
    (args.out_dir / "extended_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    plot_calibration(args.out_dir / "calibration_derived_binary.png", y_derived, p_derived[:, 1], "Derived binary calibration")
    plot_calibration(args.out_dir / "calibration_binary_head.png", y_bin, p_bin[:, 1], "Binary-head calibration")
    write_error_review(args.out_dir / "error_review.csv", rows, load_cache_meta(args.cache_all_csv))
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
