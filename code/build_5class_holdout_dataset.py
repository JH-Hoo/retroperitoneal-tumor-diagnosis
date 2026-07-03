#!/usr/bin/env python3
import csv
import json
from collections import Counter
from pathlib import Path

from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_CSV = PROJECT_ROOT / "data" / "standard" / "all.csv"
OUT_ROOT = PROJECT_ROOT / "data" / "labels_5class_holdout"
SEED = 20260703
TEST_SIZE = 0.10
VAL_SIZE = 0.10

BAD_GROUPS = {"G0122", "G0137", "G0369"}
EXCLUDE_GROUPS = {"G0180", "G0224", "G0296"}
MANUAL = {
    "G0186": ("PPGL", "副神经节瘤", "PGL_manual_match"),
    "G0326": ("肉瘤类", "脂肪肉瘤", "WDLPS_manual_match"),
    "G0365": ("肉瘤类", "脂肪肉瘤", "LPS_manual_match"),
}
CLASS_NAMES = ["肉瘤类", "良性神经源性肿瘤", "PPGL", "淋巴瘤", "胃肠道间质瘤"]


def read_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_rows(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def label_5(row):
    group = row["group"]
    if group in MANUAL:
        return MANUAL[group]
    if row["label_4"] == "副神经节瘤":
        return "PPGL", row["pathology_class"], "from_label_4"
    if row["label_4"] in {"肉瘤类", "良性神经源性肿瘤", "淋巴瘤"}:
        return row["label_4"], row["pathology_class"], "from_label_4"
    if row["pathology_class"] == "嗜铬细胞瘤":
        return "PPGL", row["pathology_class"], "from_pathology_class"
    if row["pathology_class"] == "胃肠道间质瘤":
        return "胃肠道间质瘤", row["pathology_class"], "from_pathology_class"
    return "", row["pathology_class"], "excluded_no_supervised_label"


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "splits").mkdir(exist_ok=True)

    rows = []
    for r in read_rows(SOURCE_CSV):
        label, pathology_class, source = label_5(r)
        if r["group"] in BAD_GROUPS or r["group"] in EXCLUDE_GROUPS or not label:
            continue
        rr = dict(r)
        rr["label_5"] = label
        rr["label_5_id"] = str(CLASS_NAMES.index(label))
        rr["label_5_source"] = source
        rr["pathology_class_5class"] = pathology_class
        rr["tensor"] = f"data/cache_96slice/tensors/{rr['group']}.pt"
        rows.append(rr)

    labels = [r["label_5_id"] for r in rows]
    train_val, test = train_test_split(rows, test_size=TEST_SIZE, stratify=labels, random_state=SEED)
    train_val_labels = [r["label_5_id"] for r in train_val]
    train, val = train_test_split(
        train_val,
        test_size=VAL_SIZE / (1.0 - TEST_SIZE),
        stratify=train_val_labels,
        random_state=SEED,
    )

    for split, split_rows in [("train", train), ("val", val), ("test", test)]:
        for r in split_rows:
            r["split_5class"] = split
        write_rows(OUT_ROOT / "splits" / f"{split}.csv", split_rows)
    all_rows = train + val + test
    write_rows(OUT_ROOT / "all.csv", all_rows)

    summary = {
        "name": "labels_5class_holdout",
        "source_dataset": "data/standard plus data/cache_96slice tensors",
        "num_cases": len(all_rows),
        "class_names": CLASS_NAMES,
        "split_counts": dict(Counter(r["split_5class"] for r in all_rows)),
        "label_5_counts": dict(Counter(r["label_5"] for r in all_rows)),
        "split_label_5_counts": {
            split: dict(Counter(r["label_5"] for r in all_rows if r["split_5class"] == split))
            for split in ["train", "val", "test"]
        },
        "excluded_bad_groups": sorted(BAD_GROUPS),
        "excluded_no_supervised_label": sorted(EXCLUDE_GROUPS),
        "manual_matches": MANUAL,
    }
    (OUT_ROOT / "dataset_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_ROOT / "label_mapping.json").write_text(
        json.dumps({name: i for i, name in enumerate(CLASS_NAMES)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
