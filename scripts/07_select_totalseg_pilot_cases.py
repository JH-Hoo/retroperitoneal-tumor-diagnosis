#!/usr/bin/env python3
import csv
import random
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LABELS_CSV = PROJECT_ROOT / "data" / "labels" / "labels_5class.csv"
OUT_CSV = PROJECT_ROOT / "data" / "annotations" / "totalseg_pilot_30.csv"

CLASS_NAMES = ["肉瘤类", "良性神经源性肿瘤", "PPGL", "淋巴瘤", "胃肠道间质瘤"]
N_PER_CLASS = 6
SEED = 20260704


def read_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    rng = random.Random(SEED)
    by_class = defaultdict(list)
    for row in read_rows(LABELS_CSV):
        by_class[row["label_5"]].append(row)

    selected = []
    for label in CLASS_NAMES:
        rows = sorted(by_class[label], key=lambda r: (r["fold"], r["case_id"]))
        rng.shuffle(rows)
        selected.extend(rows[:N_PER_CLASS])

    out = []
    for row in sorted(selected, key=lambda r: (CLASS_NAMES.index(r["label_5"]), r["case_id"])):
        case_id = row["case_id"]
        out.append(
            {
                "case_id": case_id,
                "label_5": row["label_5"],
                "label_5_id": row["label_5_id"],
                "fold": row["fold"],
                "nifti": f"data_private/standard/images/{case_id}.nii.gz",
                "totalseg_dir": f"data/segmentations/totalseg_pilot/{case_id}",
                "retro_roi_json": f"data/derived/retroperitoneal_roi/{case_id}.json",
                "qc_sheet": f"data/qc/contact_sheets/{case_id}_totalseg_roi.png",
            }
        )

    write_rows(OUT_CSV, out)
    for label in CLASS_NAMES:
        print(label, sum(r["label_5"] == label for r in out))
    print(OUT_CSV)


if __name__ == "__main__":
    main()
