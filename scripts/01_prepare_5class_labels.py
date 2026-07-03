#!/usr/bin/env python3
import csv
import hashlib
import json
import os
import secrets
from collections import Counter, defaultdict
from pathlib import Path

from sklearn.model_selection import StratifiedGroupKFold


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRIVATE_ROOT = PROJECT_ROOT / "data_private"
SOURCE_CSV = PRIVATE_ROOT / "standard" / "all.csv"
SALT_PATH = PRIVATE_ROOT / "deid_salt.txt"
OUT_ROOT = PROJECT_ROOT / "data" / "labels"
PRIVATE_AUDIT_ROOT = PRIVATE_ROOT / "audit"

N_SPLITS = 5
SEED = 20260703
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


def write_rows(path, rows, fieldnames=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames or list(rows[0].keys()), lineterminator="\n")
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


def salt():
    PRIVATE_ROOT.mkdir(parents=True, exist_ok=True)
    if not SALT_PATH.exists():
        SALT_PATH.write_text(secrets.token_hex(32), encoding="utf-8")
        os.chmod(SALT_PATH, 0o600)
    return SALT_PATH.read_text(encoding="utf-8").strip()


def digest(s, value):
    return hashlib.sha256(f"{s}|{value}".encode("utf-8")).hexdigest()[:24]


def age_years(value):
    digits = "".join(ch for ch in value if ch.isdigit())
    return str(int(digits)) if digits else ""


def scan_year(value):
    digits = "".join(ch for ch in value if ch.isdigit())
    return digits[:4] if len(digits) >= 4 else ""


def simplified_match_status(value):
    if value.startswith("ambiguous"):
        return "ambiguous"
    if value.startswith("matched"):
        return "matched"
    if value.startswith("manual"):
        return "manual"
    return value or "unknown"


def patient_key(row):
    if row["hospital_no"]:
        return f"hospital_no:{row['hospital_no']}"
    if row["matched_name"] and row["image_birthdate"]:
        return f"name_birth:{row['matched_name']}:{row['image_birthdate']}"
    if row["matched_name"]:
        return f"name:{row['matched_name']}"
    return f"case:{row['group']}"


def pathology_key(row):
    parts = [row.get("hospital_no", ""), row.get("pathology_no", ""), row.get("pathology_excel_row", "")]
    if any(parts):
        return "pathology:" + ":".join(parts)
    return patient_key(row)


def deid_row(row, patient_hash, pathology_hash, fold):
    return {
        "case_id": row["group"],
        "patient_uid_hash": patient_hash,
        "pathology_uid_hash": pathology_hash,
        "sex": row["image_sex"],
        "age_at_scan": age_years(row["image_age"] or row["pathology_age"]),
        "scan_year": scan_year(row["image_acquisition_date"]),
        "phase": row["phase"],
        "layer": row["layer"],
        "match_status_simplified": simplified_match_status(row["match_status"]),
        "label_5": row["label_5"],
        "label_5_id": row["label_5_id"],
        "label_5_source": row["label_5_source"],
        "pathology_class_5class": row["pathology_class_5class"],
        "tensor": f"data/cache_96slice/tensors/{row['group']}.pt",
        "fold": str(fold),
    }


def main():
    s = salt()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    PRIVATE_AUDIT_ROOT.mkdir(parents=True, exist_ok=True)

    source_rows = []
    for row in read_rows(SOURCE_CSV):
        label, pathology_class, label_source = label_5(row)
        if row["group"] in BAD_GROUPS or row["group"] in EXCLUDE_GROUPS or not label:
            continue
        r = dict(row)
        r["label_5"] = label
        r["label_5_id"] = str(CLASS_NAMES.index(label))
        r["label_5_source"] = label_source
        r["pathology_class_5class"] = pathology_class
        source_rows.append(r)

    y = [int(r["label_5_id"]) for r in source_rows]
    groups = [digest(s, "patient|" + patient_key(r)) for r in source_rows]
    splitter = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    fold_by_index = {}
    for fold, (_, test_idx) in enumerate(splitter.split(source_rows, y, groups)):
        for i in test_idx:
            fold_by_index[i] = fold

    rows = []
    private_rows = []
    for i, r in enumerate(source_rows):
        p_hash = groups[i]
        path_hash = digest(s, "pathology|" + pathology_key(r))
        rows.append(deid_row(r, p_hash, path_hash, fold_by_index[i]))
        private_rows.append(
            {
                "case_id": r["group"],
                "patient_uid_hash": p_hash,
                "pathology_uid_hash": path_hash,
                "patient_key_plain": patient_key(r),
                "pathology_key_plain": pathology_key(r),
                "matched_name": r["matched_name"],
                "PatientFolder": r["PatientFolder"],
                "hospital_no": r["hospital_no"],
                "pathology_no": r["pathology_no"],
                "pathology_excel_row": r["pathology_excel_row"],
                "legacy_split": r["split"],
                "fold": str(fold_by_index[i]),
                "label_5": r["label_5"],
            }
        )

    write_rows(OUT_ROOT / "labels_5class.csv", rows)
    write_rows(PRIVATE_AUDIT_ROOT / "linkage_audit_with_phi.csv", private_rows)

    fold_indices = defaultdict(list)
    for r in rows:
        fold_indices[int(r["fold"])].append(r)

    for fold in range(N_SPLITS):
        split_dir = OUT_ROOT / "splits" / f"fold_{fold}"
        test_rows = [r for r in rows if int(r["fold"]) == fold]
        val_rows = [r for r in rows if int(r["fold"]) == (fold + 1) % N_SPLITS]
        train_rows = [r for r in rows if int(r["fold"]) not in {fold, (fold + 1) % N_SPLITS}]
        write_rows(split_dir / "train.csv", train_rows)
        write_rows(split_dir / "val.csv", val_rows)
        write_rows(split_dir / "test.csv", test_rows)

    by_patient = defaultdict(list)
    for r in rows:
        by_patient[r["patient_uid_hash"]].append(r)

    duplicate_rows = []
    legacy_leak_rows = []
    for patient_hash, patient_rows in by_patient.items():
        if len(patient_rows) > 1:
            duplicate_rows.append(
                {
                    "patient_uid_hash": patient_hash,
                    "num_cases": len(patient_rows),
                    "case_ids": ";".join(r["case_id"] for r in patient_rows),
                    "labels": ";".join(sorted(set(r["label_5"] for r in patient_rows))),
                    "folds": ";".join(sorted(set(r["fold"] for r in patient_rows))),
                }
            )
        legacy_splits = sorted(set(pr["legacy_split"] for pr in private_rows if pr["patient_uid_hash"] == patient_hash))
        if len(legacy_splits) > 1:
            legacy_leak_rows.append(
                {
                    "patient_uid_hash": patient_hash,
                    "num_cases": len(patient_rows),
                    "case_ids": ";".join(r["case_id"] for r in patient_rows),
                    "legacy_splits": ";".join(legacy_splits),
                    "new_folds": ";".join(sorted(set(r["fold"] for r in patient_rows))),
                    "labels": ";".join(sorted(set(r["label_5"] for r in patient_rows))),
                }
            )

    if duplicate_rows:
        write_rows(PRIVATE_AUDIT_ROOT / "duplicate_patient_audit_deid.csv", duplicate_rows)
    if legacy_leak_rows:
        write_rows(PRIVATE_AUDIT_ROOT / "legacy_split_leakage_audit_deid.csv", legacy_leak_rows)

    count_rows = []
    for fold in range(N_SPLITS):
        fold_rows = fold_indices[fold]
        counts = Counter(r["label_5"] for r in fold_rows)
        row = {"fold": fold, "num_cases": len(fold_rows)}
        row.update({name: counts.get(name, 0) for name in CLASS_NAMES})
        count_rows.append(row)
    write_rows(OUT_ROOT / "fold_label_counts.csv", count_rows)

    label_mapping = {name: i for i, name in enumerate(CLASS_NAMES)}
    (OUT_ROOT / "label_mapping.json").write_text(json.dumps(label_mapping, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "name": "labels_5class",
        "source": "data_private/standard/all.csv with PHI stripped",
        "num_cases": len(rows),
        "num_patients": len(by_patient),
        "n_splits": N_SPLITS,
        "split_method": "StratifiedGroupKFold by salted patient_uid_hash",
        "class_names": CLASS_NAMES,
        "class_counts": dict(Counter(r["label_5"] for r in rows)),
        "fold_counts": {str(fold): len(fold_indices[fold]) for fold in range(N_SPLITS)},
        "duplicate_patient_count": sum(1 for v in by_patient.values() if len(v) > 1),
        "legacy_cross_split_patient_count": len(legacy_leak_rows),
        "excluded_bad_groups": sorted(BAD_GROUPS),
        "excluded_no_supervised_label": sorted(EXCLUDE_GROUPS),
        "manual_matches": MANUAL,
        "phi_policy": "GitHub keeps only deidentified labels and splits; salt and PHI audit stay in data_private/ and are gitignored.",
    }
    (OUT_ROOT / "dataset_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
