#!/usr/bin/env python3
import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE_ROOT = PROJECT_ROOT / "data" / "champion_flare23_25d_cache_15x224_minvox5000"
DEFAULT_OUT_DIR = PROJECT_ROOT / "models" / "champion_aux_only_clinical4_minvox5000_cv5"

CLINICAL4_CLASS_NAMES = ["sarcoma/GIST-like", "lymphoma", "PPGL", "benign neurogenic"]
DERIVED_BINARY_CLASS_NAMES = ["risk/workup", "benign-like"]
CLINICAL4_PROB_COLUMNS = ["prob_sarcoma_gist_like", "prob_lymphoma", "prob_ppgl", "prob_benign_neurogenic"]
BINARY_PROB_COLUMNS = ["prob_binary_head_risk_workup", "prob_binary_head_benign_like"]
LABEL_5_TO_CLINICAL4 = {
    "肉瘤类": 0,
    "胃肠道间质瘤": 0,
    "淋巴瘤": 1,
    "PPGL": 2,
    "良性神经源性肿瘤": 3,
}
LABEL_5_ID_TO_CLINICAL4 = {0: 0, 4: 0, 3: 1, 2: 2, 1: 3}
CLASS_IDS = list(range(len(CLINICAL4_CLASS_NAMES)))
BENIGN_CLINICAL4_ID = 3
BASE_AUX_COLUMNS = [
    "no_tumor_label14",
    "z_peak_norm",
    "z_centroid_norm",
    "z_std_norm",
    "z_q10_norm",
    "z_q25_norm",
    "z_q50_norm",
    "z_q75_norm",
    "z_q90_norm",
    "tumor_z_slices",
    "tumor_z_extent_norm",
    "tumor_area_max_frac",
    "tumor_area_entropy",
    "tumor_voxels",
    "crop_x",
    "crop_y",
    "spacing_z_mm",
]


def read_rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
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


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def clinical4_id(row):
    if row.get("clinical4_id", "") != "":
        return int(row["clinical4_id"])
    label = row.get("label_5", "")
    if label in LABEL_5_TO_CLINICAL4:
        return LABEL_5_TO_CLINICAL4[label]
    label_5_id = row.get("label_5_id", "")
    if label_5_id != "":
        return LABEL_5_ID_TO_CLINICAL4[int(float(label_5_id))]
    raise ValueError(f"Cannot map row to clinical4 class: {row.get('group', '')}")


def binary_id_from_clinical4(cls):
    return 1 if int(cls) == BENIGN_CLINICAL4_ID else 0


def parse_float(value, default=0.0):
    if value is None or value == "":
        return float(default)
    try:
        return float(value)
    except ValueError:
        return float(default)


def parse_semicolon_floats(text):
    if not text:
        return []
    return [float(x) for x in str(text).split(";") if str(x).strip()]


def parse_hist(row):
    vals = parse_semicolon_floats(row.get("z_hist", ""))
    return np.asarray(vals, dtype=np.float32)


def cosine(a, b):
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-8:
        return 0.0
    return float(np.dot(a, b) / denom)


class AuxFeaturizer:
    def __init__(self, base_columns=None):
        self.base_columns = list(base_columns or BASE_AUX_COLUMNS)
        self.columns = []
        self.prototypes = None

    def fit(self, rows):
        self.columns = [c for c in self.base_columns if c in rows[0]]
        hist_by_class = defaultdict(list)
        for row in rows:
            hist = parse_hist(row)
            if hist.size:
                hist_by_class[clinical4_id(row)].append(hist)
        self.prototypes = {}
        fallback = np.zeros(16, dtype=np.float32)
        if hist_by_class:
            fallback = np.zeros_like(next(iter(hist_by_class.values()))[0])
        for cls in CLASS_IDS:
            hists = hist_by_class.get(cls, [])
            if hists:
                proto = np.stack(hists, axis=0).mean(axis=0)
                proto = proto / max(float(proto.sum()), 1e-8)
            else:
                proto = fallback
            self.prototypes[cls] = proto.astype(np.float32)
        return self

    def transform_row(self, row):
        values = [parse_float(row.get(c, ""), 0.0) for c in self.columns]
        hist = parse_hist(row)
        if hist.size == 0:
            hist = np.zeros_like(self.prototypes[0])
        for cls in CLASS_IDS:
            values.append(cosine(hist, self.prototypes[cls]))
        for cls in CLASS_IDS:
            values.append(float(np.abs(hist - self.prototypes[cls]).sum()))
        return np.asarray(values, dtype=np.float32)

    def transform(self, rows):
        return np.stack([self.transform_row(row) for row in rows], axis=0)

    def to_dict(self):
        cosine_cols = [f"z_cosine_{name}" for name in CLINICAL4_CLASS_NAMES]
        l1_cols = [f"z_l1_{name}" for name in CLINICAL4_CLASS_NAMES]
        return {
            "columns": self.columns + cosine_cols + l1_cols,
            "class_z_prototypes": {CLINICAL4_CLASS_NAMES[k]: v.tolist() for k, v in (self.prototypes or {}).items()},
        }


def stratified_val_split(rows, val_fraction, seed):
    rng = random.Random(seed)
    by_class = defaultdict(list)
    for row in rows:
        by_class[clinical4_id(row)].append(row)
    train_rows, val_rows = [], []
    for cls_rows in by_class.values():
        rng.shuffle(cls_rows)
        n_val = max(1, int(round(len(cls_rows) * val_fraction))) if len(cls_rows) >= 3 else 0
        val_rows.extend(cls_rows[:n_val])
        train_rows.extend(cls_rows[n_val:])
    rng.shuffle(train_rows)
    rng.shuffle(val_rows)
    return train_rows, val_rows


def model_candidates(seed):
    return {
        "logistic_regression": {
            "clinical4": make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs"),
            ),
            "binary": make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs"),
            ),
        },
        "random_forest": {
            "clinical4": RandomForestClassifier(
                n_estimators=500,
                random_state=seed,
                class_weight="balanced_subsample",
                min_samples_leaf=2,
                n_jobs=-1,
            ),
            "binary": RandomForestClassifier(
                n_estimators=500,
                random_state=seed + 17,
                class_weight="balanced_subsample",
                min_samples_leaf=2,
                n_jobs=-1,
            ),
        },
    }


def full_proba(model, x, labels):
    probs = model.predict_proba(x)
    out = np.zeros((len(x), len(labels)), dtype=np.float32)
    for src_col, cls in enumerate(model.classes_):
        if int(cls) in labels:
            out[:, labels.index(int(cls))] = probs[:, src_col]
    row_sum = out.sum(axis=1, keepdims=True)
    row_sum[row_sum <= 1e-8] = 1.0
    return out / row_sum


def metrics_dict(ys, probs):
    pred = np.asarray(probs).argmax(axis=1)
    probs = np.asarray(probs)
    cm = confusion_matrix(ys, pred, labels=CLASS_IDS)
    recall = recall_score(ys, pred, labels=CLASS_IDS, average=None, zero_division=0)
    top2 = np.argsort(probs, axis=1)[:, -2:] if len(probs) else np.empty((0, 2), dtype=int)
    return {
        "accuracy": accuracy_score(ys, pred),
        "balanced_accuracy": balanced_accuracy_score(ys, pred),
        "macro_f1": f1_score(ys, pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(ys, pred, average="weighted", zero_division=0),
        "top2_accuracy": float(np.mean([int(y in top2_i) for y, top2_i in zip(ys, top2)])) if len(top2) else 0.0,
        "per_class_recall": {CLINICAL4_CLASS_NAMES[i]: float(recall[i]) for i in CLASS_IDS},
        "confusion_matrix": cm.tolist(),
    }


def binary_metrics_dict(ys, probs):
    pred = np.asarray(probs).argmax(axis=1)
    cm = confusion_matrix(ys, pred, labels=[0, 1])
    recall = recall_score(ys, pred, labels=[0, 1], average=None, zero_division=0)
    return {
        "accuracy": accuracy_score(ys, pred),
        "balanced_accuracy": balanced_accuracy_score(ys, pred),
        "macro_f1": f1_score(ys, pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(ys, pred, average="weighted", zero_division=0),
        "risk_workup_recall": float(recall[0]),
        "benign_like_recall": float(recall[1]),
        "confusion_matrix": cm.tolist(),
    }


def derived_binary_arrays(ys, probs):
    probs = np.asarray(probs, dtype=np.float32)
    y_bin = [binary_id_from_clinical4(y) for y in ys]
    p_benign = probs[:, BENIGN_CLINICAL4_ID]
    p_risk = 1.0 - p_benign
    return y_bin, np.stack([p_risk, p_benign], axis=1)


def prediction_rows(rows, probs4, probs2, fold):
    out = []
    for row, prob, prob_binary in zip(rows, probs4, probs2):
        true = clinical4_id(row)
        pred = int(prob.argmax())
        order = np.argsort(prob)[::-1]
        p_benign = float(prob[BENIGN_CLINICAL4_ID])
        p_risk = float(1.0 - p_benign)
        true_binary = binary_id_from_clinical4(true)
        binary_pred = int(p_benign >= p_risk)
        binary_head_pred = int(prob_binary.argmax())
        rr = {
            "group": row["group"],
            "label_5": row["label_5"],
            "true_clinical4_label": CLINICAL4_CLASS_NAMES[true],
            "true_clinical4_id": true,
            "pred_clinical4_label": CLINICAL4_CLASS_NAMES[pred],
            "pred_clinical4_id": pred,
            "top1_clinical4_label": CLINICAL4_CLASS_NAMES[int(order[0])],
            "top1_clinical4_prob": float(prob[order[0]]),
            "top2_clinical4_label": CLINICAL4_CLASS_NAMES[int(order[1])],
            "top2_clinical4_prob": float(prob[order[1]]),
            "derived_true_binary_label": DERIVED_BINARY_CLASS_NAMES[true_binary],
            "derived_true_binary_id": true_binary,
            "derived_pred_binary_label": DERIVED_BINARY_CLASS_NAMES[binary_pred],
            "derived_pred_binary_id": binary_pred,
            "prob_risk_workup": p_risk,
            "prob_benign_like": p_benign,
            "binary_head_true_binary_label": DERIVED_BINARY_CLASS_NAMES[true_binary],
            "binary_head_true_binary_id": true_binary,
            "binary_head_pred_binary_label": DERIVED_BINARY_CLASS_NAMES[binary_head_pred],
            "binary_head_pred_binary_id": binary_head_pred,
            "fold": fold,
        }
        for cls_idx, col in enumerate(CLINICAL4_PROB_COLUMNS):
            rr[col] = float(prob[cls_idx])
        for cls_idx, col in enumerate(BINARY_PROB_COLUMNS):
            rr[col] = float(prob_binary[cls_idx])
        out.append(rr)
    return out


def probs_from_prediction_rows(rows):
    y, probs = [], []
    for row in rows:
        y.append(int(row["true_clinical4_id"]))
        probs.append([float(row[col]) for col in CLINICAL4_PROB_COLUMNS])
    return y, probs


def binary_probs_from_prediction_rows(rows):
    y, probs = [], []
    for row in rows:
        y.append(int(row["binary_head_true_binary_id"]))
        probs.append([float(row[col]) for col in BINARY_PROB_COLUMNS])
    return y, probs


def plot_confusion_matrix(cm, out_path, title, class_names):
    fig_w = max(7, 1.7 * len(class_names))
    fig, ax = plt.subplots(figsize=(fig_w, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(class_names)), labels=class_names, rotation=25, ha="right")
    ax.set_yticks(range(len(class_names)), labels=class_names)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(title)
    threshold = np.asarray(cm).max() / 2 if np.asarray(cm).size else 0
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            color = "white" if cm[i][j] > threshold else "#1f2937"
            ax.text(j, i, str(cm[i][j]), ha="center", va="center", color=color, fontsize=16)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Auxiliary-feature-only 4-class clinical-imaging CV baseline.")
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=20260708)
    args = parser.parse_args()

    set_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = [r for r in read_rows(args.cache_root / "all.csv") if r.get("label_5_id", "") != ""]
    rows = [r for r in rows if r.get("cache_status", "ok") == "ok"]
    y_all = np.asarray([clinical4_id(r) for r in rows], dtype=np.int64)
    class_counts = Counter(CLINICAL4_CLASS_NAMES[y] for y in y_all)
    min_class_count = min(class_counts.values()) if class_counts else 0
    if args.folds > min_class_count:
        raise ValueError(f"--folds={args.folds} exceeds smallest class count {min_class_count}")
    print(f"rows={len(rows)} folds={args.folds} class_counts={dict(class_counts)}", flush=True)

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    all_test_predictions, fold_summaries = [], []
    for fold, (train_val_idx, test_idx) in enumerate(skf.split(np.zeros(len(rows)), y_all), 1):
        fold_seed = args.seed + fold
        train_val_rows = [rows[i] for i in train_val_idx]
        test_rows = [rows[i] for i in test_idx]
        train_rows, val_rows = stratified_val_split(train_val_rows, args.val_fraction, fold_seed)
        featurizer = AuxFeaturizer().fit(train_rows)
        x_train = featurizer.transform(train_rows)
        x_val = featurizer.transform(val_rows)
        x_test = featurizer.transform(test_rows)
        y_train = np.asarray([clinical4_id(r) for r in train_rows])
        y_val = np.asarray([clinical4_id(r) for r in val_rows])
        y_test = np.asarray([clinical4_id(r) for r in test_rows])
        y_train_bin = np.asarray([binary_id_from_clinical4(y) for y in y_train])
        y_val_bin = np.asarray([binary_id_from_clinical4(y) for y in y_val])
        y_test_bin = np.asarray([binary_id_from_clinical4(y) for y in y_test])

        best_name, best_score, best = "", -1.0, None
        candidate_metrics = {}
        for name, models in model_candidates(fold_seed).items():
            clf4 = models["clinical4"].fit(x_train, y_train)
            clf2 = models["binary"].fit(x_train, y_train_bin)
            val_probs4 = full_proba(clf4, x_val, CLASS_IDS)
            val_probs2 = full_proba(clf2, x_val, [0, 1])
            val_clinical4 = metrics_dict(y_val, val_probs4)
            val_binary = binary_metrics_dict(y_val_bin, val_probs2)
            score = val_clinical4["macro_f1"] + 0.5 * val_binary["macro_f1"]
            candidate_metrics[name] = {"clinical4": val_clinical4, "binary_head": val_binary, "combined_score": score}
            if score > best_score:
                best_name, best_score, best = name, score, (clf4, clf2)

        probs4 = full_proba(best[0], x_test, CLASS_IDS)
        probs2 = full_proba(best[1], x_test, [0, 1])
        test_pred = prediction_rows(test_rows, probs4, probs2, fold)
        all_test_predictions.extend(test_pred)
        test_metrics = {
            "clinical4": metrics_dict(y_test, probs4),
            "derived_binary": binary_metrics_dict(*derived_binary_arrays(y_test, probs4)),
            "binary_head": binary_metrics_dict(y_test_bin, probs2),
        }
        fold_summary = {
            "fold": fold,
            "selected_model": best_name,
            "selection_score": best_score,
            "splits": {"train": len(train_rows), "val": len(val_rows), "test": len(test_rows)},
            "aux_featurizer": featurizer.to_dict(),
            "candidate_val_metrics": candidate_metrics,
            "metrics": {"test": test_metrics},
        }
        fold_summaries.append(fold_summary)
        print(f"fold {fold} selected={best_name} test {json.dumps(test_metrics, ensure_ascii=False)}", flush=True)

    y_oof, prob_oof = probs_from_prediction_rows(all_test_predictions)
    y_bin_oof, prob_bin_oof = binary_probs_from_prediction_rows(all_test_predictions)
    y_derived_oof, prob_derived_oof = derived_binary_arrays(y_oof, prob_oof)
    oof_metrics = {
        "clinical4": metrics_dict(y_oof, prob_oof),
        "derived_binary": binary_metrics_dict(y_derived_oof, prob_derived_oof),
        "binary_head": binary_metrics_dict(y_bin_oof, prob_bin_oof),
    }
    write_rows(args.out_dir / "oof_predictions.csv", all_test_predictions)
    write_rows(
        args.out_dir / "oof_predictions_derived_binary.csv",
        [
            {
                "group": row["group"],
                "label_5": row["label_5"],
                "true_binary_label": row["derived_true_binary_label"],
                "true_binary_id": row["derived_true_binary_id"],
                "pred_binary_label": row["derived_pred_binary_label"],
                "pred_binary_id": row["derived_pred_binary_id"],
                "prob_risk_workup": row["prob_risk_workup"],
                "prob_benign_like": row["prob_benign_like"],
                "fold": row["fold"],
            }
            for row in all_test_predictions
        ],
    )
    write_rows(
        args.out_dir / "oof_predictions_binary_head.csv",
        [
            {
                "group": row["group"],
                "label_5": row["label_5"],
                "true_binary_label": row["binary_head_true_binary_label"],
                "true_binary_id": row["binary_head_true_binary_id"],
                "pred_binary_label": row["binary_head_pred_binary_label"],
                "pred_binary_id": row["binary_head_pred_binary_id"],
                "prob_binary_head_risk_workup": row["prob_binary_head_risk_workup"],
                "prob_binary_head_benign_like": row["prob_binary_head_benign_like"],
                "fold": row["fold"],
            }
            for row in all_test_predictions
        ],
    )
    summary = {
        "task": "aux_only_clinical4_binary_baseline",
        "architecture": "Structured auxiliary features only; fold-wise selection between balanced logistic regression and balanced random forest.",
        "cache_root": str(args.cache_root),
        "num_rows": len(rows),
        "folds": args.folds,
        "class_counts": dict(class_counts),
        "oof_metrics": oof_metrics,
        "folds_detail": fold_summaries,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    plot_confusion_matrix(
        oof_metrics["clinical4"]["confusion_matrix"],
        args.out_dir / "aux_only_clinical4_oof_confusion_matrix.png",
        f"Aux-only clinical4 5-fold OOF confusion matrix (acc={oof_metrics['clinical4']['accuracy']:.3f})",
        CLINICAL4_CLASS_NAMES,
    )
    plot_confusion_matrix(
        oof_metrics["derived_binary"]["confusion_matrix"],
        args.out_dir / "aux_only_derived_binary_oof_confusion_matrix.png",
        f"Aux-only derived binary OOF confusion matrix (acc={oof_metrics['derived_binary']['accuracy']:.3f})",
        DERIVED_BINARY_CLASS_NAMES,
    )
    plot_confusion_matrix(
        oof_metrics["binary_head"]["confusion_matrix"],
        args.out_dir / "aux_only_binary_head_oof_confusion_matrix.png",
        f"Aux-only binary head OOF confusion matrix (acc={oof_metrics['binary_head']['accuracy']:.3f})",
        DERIVED_BINARY_CLASS_NAMES,
    )
    print(json.dumps({"oof": oof_metrics}, ensure_ascii=False, indent=2), flush=True)
    print(f"outputs: {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
