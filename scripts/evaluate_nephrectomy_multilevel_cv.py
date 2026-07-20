#!/usr/bin/env python3
"""Nested patient-level evaluation of Yang-style and adapted feature fusion."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier


MODEL_GROUPS = {
    "size_only": ("size",),
    "geometry": ("geometry",),
    "paper_radiomics": ("paper_radiomics",),
    "paper_radiomics_voxel": ("paper_radiomics", "voxel"),
    "paper_deep": ("deep",),
    "paper_all": ("paper_radiomics", "voxel", "deep"),
    "adapted_all": ("geometry", "all_radiomics", "voxel", "deep"),
}


def read_csv(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows, fieldnames=None):
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def numeric(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def static_feature_sets(columns):
    size_candidates = {
        "geo_tumor_volume_ml",
        "geo_tumor_bbox_diagonal_mm",
        "rad_tumor_shape_Maximum3DDiameter",
    }
    return {
        "size": [c for c in columns if c in size_candidates],
        "geometry": [c for c in columns if c.startswith("geo_")],
        "paper_radiomics": [
            c for c in columns if c.startswith("rad_tumor_") or c.startswith("rad_kidney_")
        ],
        "all_radiomics": [c for c in columns if c.startswith("rad_")],
        "voxel": [c for c in columns if c.startswith("pca_") or c.startswith("svd_")],
    }


def candidate_configs(raw_feature_count):
    candidates = [
        {"k": 10, "max_depth": 1, "n_estimators": 50, "learning_rate": 0.05, "reg_lambda": 30.0},
        {"k": 10, "max_depth": 2, "n_estimators": 100, "learning_rate": 0.03, "reg_lambda": 30.0},
        {"k": 25, "max_depth": 1, "n_estimators": 100, "learning_rate": 0.05, "reg_lambda": 10.0},
        {"k": 25, "max_depth": 2, "n_estimators": 100, "learning_rate": 0.05, "reg_lambda": 30.0},
        {"k": "all", "max_depth": 1, "n_estimators": 100, "learning_rate": 0.03, "reg_lambda": 30.0},
        {"k": "all", "max_depth": 2, "n_estimators": 50, "learning_rate": 0.05, "reg_lambda": 30.0},
    ]
    return [c for c in candidates if c["k"] == "all" or c["k"] <= raw_feature_count]


def make_pipeline(config, scale_pos_weight, seed):
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("variance", VarianceThreshold()),
            ("select", SelectKBest(score_func=f_classif, k=config["k"])),
            (
                "model",
                XGBClassifier(
                    objective="binary:logistic",
                    eval_metric="logloss",
                    tree_method="hist",
                    random_state=seed,
                    n_jobs=1,
                    max_depth=config["max_depth"],
                    n_estimators=config["n_estimators"],
                    learning_rate=config["learning_rate"],
                    reg_lambda=config["reg_lambda"],
                    reg_alpha=1.0,
                    min_child_weight=3.0,
                    subsample=0.8,
                    colsample_bytree=0.7,
                    scale_pos_weight=scale_pos_weight,
                ),
            ),
        ]
    )


def choose_config(X, y, inner_folds, seed):
    cv = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=seed)
    scale_pos_weight = float((y == 0).sum() / max((y == 1).sum(), 1))
    scores = []
    for index, config in enumerate(candidate_configs(X.shape[1])):
        fold_scores = []
        failed = False
        for train_idx, val_idx in cv.split(X, y):
            pipeline = make_pipeline(config, scale_pos_weight, seed + index)
            try:
                pipeline.fit(X[train_idx], y[train_idx])
                probability = pipeline.predict_proba(X[val_idx])[:, 1]
                fold_scores.append(average_precision_score(y[val_idx], probability))
            except ValueError:
                failed = True
                break
        if not failed:
            scores.append((float(np.mean(fold_scores)), config))
    if not scores:
        fallback = {"k": "all", "max_depth": 1, "n_estimators": 50, "learning_rate": 0.05, "reg_lambda": 30.0}
        return fallback, []
    scores.sort(key=lambda item: item[0], reverse=True)
    return scores[0][1], [{"mean_inner_ap": score, **cfg} for score, cfg in scores]


def choose_threshold(y, probability):
    thresholds = np.unique(np.r_[0.0, probability, 1.0])
    best = (0.5, -np.inf)
    for threshold in thresholds:
        score = balanced_accuracy_score(y, probability >= threshold)
        if score > best[1]:
            best = (float(threshold), float(score))
    return best[0]


def metric_dict(y, probability, threshold):
    prediction = probability >= threshold
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    return {
        "n": int(len(y)),
        "events": int(y.sum()),
        "event_rate": float(y.mean()),
        "roc_auc": float(roc_auc_score(y, probability)),
        "average_precision": float(average_precision_score(y, probability)),
        "brier": float(brier_score_loss(y, probability)),
        "threshold": float(threshold),
        "sensitivity": float(tp / max(tp + fn, 1)),
        "specificity": float(tn / max(tn + fp, 1)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def bootstrap_ci(y, probability, metric, iterations, seed):
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(iterations):
        index = rng.integers(0, len(y), len(y))
        yy, pp = y[index], probability[index]
        if len(np.unique(yy)) < 2:
            continue
        if metric == "roc_auc":
            values.append(roc_auc_score(yy, pp))
        elif metric == "average_precision":
            values.append(average_precision_score(yy, pp))
        elif metric == "brier":
            values.append(brier_score_loss(yy, pp))
        else:
            raise ValueError(metric)
    estimate = {
        "roc_auc": roc_auc_score,
        "average_precision": average_precision_score,
        "brier": brier_score_loss,
    }[metric](y, probability)
    return {
        "estimate": float(estimate),
        "ci_low": float(np.quantile(values, 0.025)),
        "ci_high": float(np.quantile(values, 0.975)),
        "iterations": int(len(values)),
    }


def score_metric(y, probability, metric):
    if metric == "roc_auc":
        return float(roc_auc_score(y, probability))
    if metric == "average_precision":
        return float(average_precision_score(y, probability))
    raise ValueError(metric)


def paired_bootstrap_delta(y, probability, reference_probability, metric, iterations, seed):
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(iterations):
        index = rng.integers(0, len(y), len(y))
        yy = y[index]
        if len(np.unique(yy)) < 2:
            continue
        values.append(
            score_metric(yy, probability[index], metric)
            - score_metric(yy, reference_probability[index], metric)
        )
    estimate = score_metric(y, probability, metric) - score_metric(y, reference_probability, metric)
    return float(estimate), float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def permutation_p_value(y, probability, metric, iterations, seed):
    rng = np.random.default_rng(seed)
    observed = score_metric(y, probability, metric)
    null = [score_metric(rng.permutation(y), probability, metric) for _ in range(iterations)]
    return float((1 + np.sum(np.asarray(null) >= observed)) / (iterations + 1))


def selected_feature_names(pipeline, feature_names):
    names = np.asarray(feature_names, dtype=object)
    names = names[pipeline.named_steps["variance"].get_support()]
    names = names[pipeline.named_steps["select"].get_support()]
    return list(names)


def transform_before_model(pipeline, X):
    X = pipeline.named_steps["imputer"].transform(X)
    X = pipeline.named_steps["variance"].transform(X)
    X = pipeline.named_steps["select"].transform(X)
    return X


def feature_matrix(case_ids, static_by_id, deep_by_id, names):
    rows = []
    for case_id in case_ids:
        source = dict(static_by_id[case_id])
        source.update(deep_by_id.get(case_id, {}))
        rows.append([numeric(source.get(name)) for name in names])
    return np.asarray(rows, dtype=np.float64)


def load_deep(path: Path):
    return {row["case_id"]: row for row in read_csv(path)}


def plot_curves(out_dir, y_by_model, p_by_model):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for model in MODEL_GROUPS:
        y, p = y_by_model[model], p_by_model[model]
        fpr, tpr, _ = roc_curve(y, p)
        precision, recall, _ = precision_recall_curve(y, p)
        axes[0].plot(fpr, tpr, label=f"{model} ({roc_auc_score(y,p):.3f})")
        axes[1].plot(recall, precision, label=f"{model} ({average_precision_score(y,p):.3f})")
    axes[0].plot([0, 1], [0, 1], "--", color="#94A3B8")
    axes[0].set(xlabel="False-positive rate", ylabel="Sensitivity", title="Patient-level OOF ROC")
    prevalence = float(next(iter(y_by_model.values())).mean())
    axes[1].axhline(prevalence, linestyle="--", color="#94A3B8")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Patient-level OOF precision-recall")
    for ax in axes:
        ax.legend(fontsize=7)
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_dir / "oof_curves.png", dpi=180)
    plt.close(fig)


def plot_importance(out_dir, importance_rows, model_name="adapted_all"):
    aggregate = defaultdict(list)
    for row in importance_rows:
        if row["model"] == model_name:
            aggregate[row["feature"]].append(float(row["importance"]))
    ranked = sorted(((np.mean(v), k) for k, v in aggregate.items()), reverse=True)[:20]
    if not ranked:
        return
    values, names = zip(*reversed(ranked))
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(names, values, color="#2563EB")
    ax.set_xlabel("Mean absolute SHAP value on held-out outer folds")
    ax.set_title("Adapted multi-level model: top 20 SHAP features")
    fig.tight_layout()
    fig.savefig(out_dir / "adapted_all_shap_importance.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--deep-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--permutations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260721)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    static_rows = [row for row in read_csv(args.feature_root / "features.csv") if row.get("feature_status") == "ok"]
    static_by_id = {row["case_id"]: row for row in static_rows}
    columns = list(static_rows[0])
    feature_sets = static_feature_sets(columns)
    deep_columns = [f"deep_{task}_{i:02d}" for task in ("pathology", "proximity") for i in range(64)]
    feature_sets["deep"] = deep_columns
    splits = json.loads((args.deep_root / "outer_splits.json").read_text())

    predictions = []
    fold_results = []
    importance_rows = []
    y_by_model = {}
    p_by_model = {}

    for model_name, parts in MODEL_GROUPS.items():
        model_predictions = []
        for split in splits:
            fold = int(split["fold"])
            train_ids = split["train_case_ids"]
            test_ids = split["test_case_ids"]
            train_deep = load_deep(args.deep_root / f"fold{fold}" / "train_deep_features.csv")
            test_deep = load_deep(args.deep_root / f"fold{fold}" / "test_deep_features.csv")
            feature_names = []
            for part in parts:
                feature_names.extend(feature_sets[part])
            feature_names = list(dict.fromkeys(feature_names))
            X_train = feature_matrix(train_ids, static_by_id, train_deep, feature_names)
            X_test = feature_matrix(test_ids, static_by_id, test_deep, feature_names)
            y_train = np.asarray([int(float(static_by_id[x]["nephrectomy"])) for x in train_ids])
            y_test = np.asarray([int(float(static_by_id[x]["nephrectomy"])) for x in test_ids])

            best_config, tuning = choose_config(X_train, y_train, args.inner_folds, args.seed + fold)
            scale_pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
            pipeline = make_pipeline(best_config, scale_pos_weight, args.seed + fold)
            inner_cv = StratifiedKFold(
                n_splits=args.inner_folds, shuffle=True, random_state=args.seed + fold
            )
            train_oof = cross_val_predict(
                pipeline, X_train, y_train, cv=inner_cv, method="predict_proba", n_jobs=1
            )[:, 1]
            threshold = choose_threshold(y_train, train_oof)
            pipeline.fit(X_train, y_train)
            probability = pipeline.predict_proba(X_test)[:, 1]
            selected_names = selected_feature_names(pipeline, feature_names)
            transformed_test = transform_before_model(pipeline, X_test)
            # XGBoost's prediction contributions are exact TreeSHAP values for
            # the fitted booster.  Compute them only on the untouched outer-test
            # patients so the explanation follows the same leakage boundary.
            contribution = pipeline.named_steps["model"].get_booster().predict(
                __import__("xgboost").DMatrix(transformed_test), pred_contribs=True
            )[:, :-1]
            mean_abs_shap = np.mean(np.abs(contribution), axis=0)
            for name, value in zip(selected_names, mean_abs_shap):
                importance_rows.append(
                    {
                        "model": model_name,
                        "fold": fold,
                        "feature": name,
                        "importance": float(value),
                        "importance_type": "mean_absolute_test_SHAP",
                    }
                )
            fold_results.append(
                {
                    "model": model_name,
                    "fold": fold,
                    "train_n": len(train_ids),
                    "test_n": len(test_ids),
                    "test_events": int(y_test.sum()),
                    "raw_feature_count": len(feature_names),
                    "selected_feature_count": len(selected_names),
                    "threshold": threshold,
                    "best_config": best_config,
                    "inner_tuning": tuning,
                    "test_roc_auc": float(roc_auc_score(y_test, probability)),
                    "test_average_precision": float(average_precision_score(y_test, probability)),
                }
            )
            for case_id, truth, prob in zip(test_ids, y_test, probability):
                row = static_by_id[case_id]
                model_predictions.append(
                    {
                        "case_id": case_id,
                        "nephrectomy": int(truth),
                        "model": model_name,
                        "fold": fold,
                        "oof_probability": float(prob),
                        "threshold": threshold,
                        "oof_prediction": int(prob >= threshold),
                        "label_4": row.get("label_4", ""),
                        "pathology_class": row.get("pathology_class", ""),
                        "ct_after_pathology": row.get("ct_after_pathology", ""),
                    }
                )
        predictions.extend(model_predictions)
        y_by_model[model_name] = np.asarray([row["nephrectomy"] for row in model_predictions])
        p_by_model[model_name] = np.asarray([row["oof_probability"] for row in model_predictions])

    summaries = []
    sensitivity = []
    for model_name in MODEL_GROUPS:
        rows = [row for row in predictions if row["model"] == model_name]
        y = np.asarray([row["nephrectomy"] for row in rows])
        p = np.asarray([row["oof_probability"] for row in rows])
        pred = np.asarray([row["oof_prediction"] for row in rows])
        threshold_for_metrics = 0.5
        metrics = metric_dict(y, p, threshold_for_metrics)
        metrics["balanced_accuracy_fold_specific_thresholds"] = float(balanced_accuracy_score(y, pred))
        for metric in ("roc_auc", "average_precision", "brier"):
            metrics[f"{metric}_bootstrap"] = bootstrap_ci(
                y, p, metric, args.bootstrap, args.seed + len(summaries) * 10
            )
        summaries.append({"model": model_name, **metrics})

        for cohort_name, keep in (
            ("exclude_ct_after_pathology", [str(row["ct_after_pathology"]) not in {"1", "1.0"} for row in rows]),
            ("sarcoma_only", [row["label_4"] == "肉瘤类" for row in rows]),
        ):
            keep = np.asarray(keep, dtype=bool)
            yy, pp = y[keep], p[keep]
            if len(yy) and len(np.unique(yy)) == 2:
                sensitivity.append(
                    {
                        "model": model_name,
                        "cohort": cohort_name,
                        "n": int(len(yy)),
                        "events": int(yy.sum()),
                        "roc_auc": float(roc_auc_score(yy, pp)),
                        "average_precision": float(average_precision_score(yy, pp)),
                    }
                )

    statistical_comparisons = []
    reference_probability = p_by_model["geometry"]
    reference_y = y_by_model["geometry"]
    for index, model_name in enumerate(MODEL_GROUPS):
        y = y_by_model[model_name]
        p = p_by_model[model_name]
        if not np.array_equal(y, reference_y):
            raise RuntimeError("model prediction rows are not aligned")
        row = {"model": model_name, "reference": "geometry"}
        for offset, metric in enumerate(("roc_auc", "average_precision")):
            delta, low, high = paired_bootstrap_delta(
                y,
                p,
                reference_probability,
                metric,
                args.bootstrap,
                args.seed + index * 100 + offset,
            )
            row[f"{metric}_delta_vs_geometry"] = delta
            row[f"{metric}_delta_ci_low"] = low
            row[f"{metric}_delta_ci_high"] = high
            row[f"{metric}_permutation_p"] = permutation_p_value(
                y, p, metric, args.permutations, args.seed + index * 1000 + offset
            )
        statistical_comparisons.append(row)

    write_csv(args.out_dir / "oof_predictions.csv", predictions)
    write_csv(args.out_dir / "model_summary.csv", summaries)
    write_csv(args.out_dir / "sensitivity_analysis.csv", sensitivity)
    write_csv(args.out_dir / "feature_importance.csv", importance_rows)
    write_csv(args.out_dir / "statistical_comparisons.csv", statistical_comparisons)
    plot_curves(args.out_dir, y_by_model, p_by_model)
    plot_importance(args.out_dir, importance_rows)

    payload = {
        "analysis": "nested patient-level 5-fold OOF evaluation",
        "primary_metric": "average_precision",
        "unknown_nephrectomy_excluded": True,
        "feature_selection_inside_inner_cv": True,
        "models": summaries,
        "folds": fold_results,
        "sensitivity": sensitivity,
        "statistical_comparisons": statistical_comparisons,
        "feature_group_counts": {key: len(value) for key, value in feature_sets.items()},
    }
    (args.out_dir / "results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"models": summaries, "sensitivity": sensitivity}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
