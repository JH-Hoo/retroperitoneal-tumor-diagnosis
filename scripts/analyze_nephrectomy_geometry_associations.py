#!/usr/bin/env python3
"""Exploratory univariate association audit for tumor-kidney geometry."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score


def read_csv(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def bh_adjust(p_values):
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 1.0
    count = len(values)
    for rank_index in range(count - 1, -1, -1):
        original_index = order[rank_index]
        rank = rank_index + 1
        running = min(running, values[original_index] * count / rank)
        adjusted[original_index] = running
    return np.clip(adjusted, 0.0, 1.0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = [
        row
        for row in read_csv(args.features_csv)
        if row.get("feature_status") == "ok"
        and row.get("nephrectomy") in {"0", "1", "0.0", "1.0"}
    ]
    y = np.asarray([int(float(row["nephrectomy"])) for row in rows], dtype=int)
    geometry = [key for key in rows[0] if key.startswith("geo_")]
    results = []
    for feature in geometry:
        values = np.asarray([float(row[feature]) for row in rows], dtype=float)
        keep = np.isfinite(values)
        yy, xx = y[keep], values[keep]
        positive, negative = xx[yy == 1], xx[yy == 0]
        test = mannwhitneyu(positive, negative, alternative="two-sided", method="asymptotic")
        auc = float(roc_auc_score(yy, xx))
        results.append(
            {
                "feature": feature,
                "n": int(len(xx)),
                "positive_n": int(len(positive)),
                "negative_n": int(len(negative)),
                "positive_median": float(np.median(positive)),
                "negative_median": float(np.median(negative)),
                "median_difference_positive_minus_negative": float(
                    np.median(positive) - np.median(negative)
                ),
                "univariate_auc_higher_predicts_nephrectomy": auc,
                "direction_adjusted_auc": max(auc, 1.0 - auc),
                "rank_biserial_effect": float(
                    2.0 * test.statistic / (len(positive) * len(negative)) - 1.0
                ),
                "p_value": float(test.pvalue),
            }
        )
    adjusted = bh_adjust([row["p_value"] for row in results])
    for row, q_value in zip(results, adjusted):
        row["bh_fdr_q_value"] = float(q_value)
    results.sort(key=lambda row: (row["bh_fdr_q_value"], row["p_value"]))
    write_csv(args.out_dir / "geometry_univariate_associations.csv", results)
    payload = {
        "analysis": "exploratory univariate geometry association; not a predictive validation",
        "cases": len(rows),
        "events": int(y.sum()),
        "features_tested": len(geometry),
        "fdr_significant_0_05": sum(row["bh_fdr_q_value"] < 0.05 for row in results),
        "top_features": results[:10],
    }
    (args.out_dir / "geometry_univariate_associations.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
