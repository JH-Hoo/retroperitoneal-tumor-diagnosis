#!/usr/bin/env python3
import argparse
import csv
import hashlib
from pathlib import Path


LABEL_5_TO_CLINICAL4 = {
    "肉瘤类": "sarcoma/GIST-like",
    "胃肠道间质瘤": "sarcoma/GIST-like",
    "淋巴瘤": "lymphoma",
    "PPGL": "PPGL",
    "良性神经源性肿瘤": "benign neurogenic",
}


def read_rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "case_id_hash",
        "clinical4_label",
        "label_5",
        "fold",
        "champion_tumor_voxels",
        "cache_status",
        "sample_status",
        "crop_status",
        "source_z",
        "spacing_x_mm",
        "spacing_y_mm",
        "spacing_z_mm",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def case_hash(group, salt):
    text = f"{salt}:{group}".encode("utf-8")
    return hashlib.sha256(text).hexdigest()[:16]


def main():
    parser = argparse.ArgumentParser(description="Write a de-identified public manifest for cache/report reproducibility.")
    parser.add_argument("--cache-all-csv", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--salt", default="retroperitoneal-tumor-diagnosis")
    args = parser.parse_args()

    folds = {row["group"]: row.get("fold", "") for row in read_rows(args.predictions)}
    rows = []
    for row in read_rows(args.cache_all_csv):
        group = row.get("group", "")
        if not group:
            continue
        rows.append(
            {
                "case_id_hash": case_hash(group, args.salt),
                "clinical4_label": row.get("clinical4_label", "") or LABEL_5_TO_CLINICAL4.get(row.get("label_5", ""), ""),
                "label_5": row.get("label_5", ""),
                "fold": folds.get(group, ""),
                "champion_tumor_voxels": row.get("tumor_voxels", ""),
                "cache_status": row.get("cache_status", ""),
                "sample_status": row.get("sample_status", ""),
                "crop_status": row.get("crop_status", ""),
                "source_z": row.get("source_z", ""),
                "spacing_x_mm": row.get("spacing_x_mm", ""),
                "spacing_y_mm": row.get("spacing_y_mm", ""),
                "spacing_z_mm": row.get("spacing_z_mm", ""),
            }
        )
    write_rows(args.out, rows)
    print(f"wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
