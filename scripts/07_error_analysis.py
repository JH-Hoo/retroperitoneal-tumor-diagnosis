#!/usr/bin/env python3
import csv
import os
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRED_CSV = Path(os.environ.get("PRED_CSV", PROJECT_ROOT / "runs" / "fusion_late_fusion" / "test_predictions.csv"))
LABEL_CSV = PROJECT_ROOT / "data" / "labels" / "labels_5class.csv"
OUT_DIR = Path(os.environ.get("OUT_DIR", PRED_CSV.parent / "error_analysis"))


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


def age_bin(age):
    age = float(age)
    if age < 40:
        return "<40"
    if age < 50:
        return "40-49"
    if age < 60:
        return "50-59"
    if age < 70:
        return "60-69"
    return ">=70"


def summarize(rows, key):
    out = []
    for value in sorted({r.get(key, "") for r in rows}):
        sub = [r for r in rows if r.get(key, "") == value]
        fp = sum(int(r["true_id"]) == 0 and int(r["pred_id"]) == 1 for r in sub)
        fn = sum(int(r["true_id"]) == 1 and int(r["pred_id"]) == 0 for r in sub)
        correct = sum(int(r["true_id"]) == int(r["pred_id"]) for r in sub)
        out.append(
            {
                key: value,
                "n": len(sub),
                "correct": correct,
                "error": len(sub) - correct,
                "false_positive_benign": fp,
                "false_negative_nonbenign": fn,
                "mean_prob_nonbenign": float(np.mean([float(r["prob_nonbenign_actionable"]) for r in sub])),
            }
        )
    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    labels = {r["case_id"]: r for r in read_rows(LABEL_CSV)}
    rows = []
    for r in read_rows(PRED_CSV):
        rr = dict(r)
        meta = labels[rr["case_id"]]
        for key in ["age_at_scan", "sex", "label_5", "patient_uid_hash"]:
            rr[key] = meta.get(key, "")
        rr["age_bin"] = age_bin(rr["age_at_scan"])
        rows.append(rr)

    fp = [r for r in rows if int(r["true_id"]) == 0 and int(r["pred_id"]) == 1]
    fn = [r for r in rows if int(r["true_id"]) == 1 and int(r["pred_id"]) == 0]
    write_rows(OUT_DIR / "false_positive_benign.csv", fp)
    write_rows(OUT_DIR / "false_negative_nonbenign.csv", fn)
    write_rows(OUT_DIR / "error_by_label5.csv", summarize(rows, "true_label_5"))
    write_rows(OUT_DIR / "error_by_age_bin.csv", summarize(rows, "age_bin"))
    write_rows(OUT_DIR / "error_by_sex.csv", summarize(rows, "sex"))
    if any("top_slice_index_in_bag" in r and r["top_slice_index_in_bag"] != "" for r in rows):
        top_rows = [
            {
                "case_id": r["case_id"],
                "fold": r.get("fold", ""),
                "true_label_5": r["true_label_5"],
                "true_id": r["true_id"],
                "pred_id": r["pred_id"],
                "prob_nonbenign_actionable": r["prob_nonbenign_actionable"],
                "top_slice_index_in_bag": r.get("top_slice_index_in_bag", ""),
            }
            for r in rows
        ]
        write_rows(OUT_DIR / "top_attention_slices.csv", top_rows)
    print(f"wrote {OUT_DIR} fp={len(fp)} fn={len(fn)}")


if __name__ == "__main__":
    main()
