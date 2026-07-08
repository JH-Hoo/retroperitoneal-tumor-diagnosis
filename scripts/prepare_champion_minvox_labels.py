#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LABELS = PROJECT_ROOT / "data" / "labels_5class_holdout" / "all.csv"
DEFAULT_STATS = PROJECT_ROOT / "models" / "flare23_champion_summary" / "champion_label14_stats.csv"
DEFAULT_OUT = PROJECT_ROOT / "data" / "labels" / "champion_minvox5000.csv"


def read_rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError("No rows to write.")
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


def main():
    parser = argparse.ArgumentParser(
        description="Filter supervised rows to cases with sufficient Shenzhen-Yorktal FLARE23 champion label14 voxels."
    )
    parser.add_argument("--labels-csv", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--champion-stats", type=Path, default=DEFAULT_STATS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--min-tumor-voxels", type=int, default=5000)
    args = parser.parse_args()

    stats = {}
    for row in read_rows(args.champion_stats):
        stats[row["case"]] = int(float(row.get("tumor_voxels", 0) or 0))

    rows = []
    skipped = {"missing_stats": 0, "below_minvox": 0, "unlabeled": 0}
    for row in read_rows(args.labels_csv):
        group = row.get("group", "")
        if row.get("label_5_id", "") == "":
            skipped["unlabeled"] += 1
            continue
        if group not in stats:
            skipped["missing_stats"] += 1
            continue
        voxels = stats[group]
        if voxels < args.min_tumor_voxels:
            skipped["below_minvox"] += 1
            continue
        rr = dict(row)
        rr["champion_tumor_voxels"] = voxels
        rr["no_tumor_label14"] = 0
        rows.append(rr)

    write_rows(args.out, rows)
    summary = {
        "source_labels": str(args.labels_csv),
        "champion_stats": str(args.champion_stats),
        "out": str(args.out),
        "min_tumor_voxels": args.min_tumor_voxels,
        "num_rows": len(rows),
        "skipped": skipped,
    }
    args.out.with_suffix(".summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
