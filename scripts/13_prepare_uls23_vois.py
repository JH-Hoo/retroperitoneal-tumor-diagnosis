#!/usr/bin/env python3
import csv
import json
from pathlib import Path

import nibabel as nib
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLICKS_CSV = PROJECT_ROOT / "data" / "annotations" / "tumor_clicks_pilot_30.csv"
OUT_DIR = PROJECT_ROOT / "data" / "derived" / "uls23_vois"
STATUS_CSV = OUT_DIR / "uls23_voi_status.csv"
REVIEW_CSV = PROJECT_ROOT / "data" / "annotations" / "lesion_mask_review_pilot_30.csv"

VOI_SHAPE = np.array([256, 256, 128], dtype=int)


def read_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_rows(path, rows, fieldnames=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames or list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def has_click(row):
    return row["x_voxel"].strip() and row["y_voxel"].strip() and row["z_voxel"].strip()


def parse_center(row):
    return np.array([round(float(row["x_voxel"])), round(float(row["y_voxel"])), round(float(row["z_voxel"]))], dtype=int)


def crop_with_padding(vol, center):
    start = center - VOI_SHAPE // 2
    end = start + VOI_SHAPE
    src_start = np.maximum(start, 0)
    src_end = np.minimum(end, np.array(vol.shape[:3]))
    dst_start = src_start - start
    dst_end = dst_start + (src_end - src_start)
    out = np.full(tuple(VOI_SHAPE), float(vol.min()), dtype=np.float32)
    out[dst_start[0] : dst_end[0], dst_start[1] : dst_end[1], dst_start[2] : dst_end[2]] = vol[
        src_start[0] : src_end[0], src_start[1] : src_end[1], src_start[2] : src_end[2]
    ]
    return out, start, src_start, src_end


def prepare_case(row):
    case_id = row["case_id"]
    if not has_click(row):
        return {
            "case_id": case_id,
            "label_5": row["label_5"],
            "status": "missing_click",
            "center_voxel": "",
            "voi_path": "",
            "meta_path": "",
            "message": "fill x_voxel,y_voxel,z_voxel first",
        }, None

    img = nib.load(str(PROJECT_ROOT / row["nifti"]))
    vol = np.asarray(img.get_fdata(dtype=np.float32))
    center = parse_center(row)
    if np.any(center < 0) or np.any(center >= np.array(vol.shape[:3])):
        return {
            "case_id": case_id,
            "label_5": row["label_5"],
            "status": "invalid_click",
            "center_voxel": ";".join(map(str, center.tolist())),
            "voi_path": "",
            "meta_path": "",
            "message": f"image shape {vol.shape[:3]}",
        }, None

    voi, start, src_start, src_end = crop_with_padding(vol, center)
    affine = img.affine.copy()
    affine[:3, 3] = nib.affines.apply_affine(img.affine, start)
    voi_path = OUT_DIR / f"{case_id}_candidate_001_voi.nii.gz"
    meta_path = OUT_DIR / f"{case_id}_candidate_001_meta.json"
    nib.save(nib.Nifti1Image(voi, affine, img.header), str(voi_path))
    meta = {
        "case_id": case_id,
        "candidate_id": "001",
        "source_image": row["nifti"],
        "center_voxel_original": center.tolist(),
        "voi_shape": VOI_SHAPE.tolist(),
        "voi_start_voxel_original": start.tolist(),
        "source_copy_start_voxel": src_start.tolist(),
        "source_copy_end_voxel_exclusive": src_end.tolist(),
        "output_voi": str(voi_path.relative_to(PROJECT_ROOT)),
        "intended_next_step": "Run ULS23 or another lesion mask proposer on this lesion-centered VOI.",
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    review = {
        "case_id": case_id,
        "candidate_id": "001",
        "usable": "",
        "hit_status": "",
        "needs_edit": "",
        "reviewer": "",
        "comment": "",
    }
    return {
        "case_id": case_id,
        "label_5": row["label_5"],
        "status": "ready",
        "center_voxel": ";".join(map(str, center.tolist())),
        "voi_path": str(voi_path.relative_to(PROJECT_ROOT)),
        "meta_path": str(meta_path.relative_to(PROJECT_ROOT)),
        "message": "",
    }, review


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    status_rows = []
    review_rows = []
    for row in read_rows(CLICKS_CSV):
        status, review = prepare_case(row)
        status_rows.append(status)
        if review:
            review_rows.append(review)
        print(status["case_id"], status["status"], status["message"], flush=True)

    write_rows(STATUS_CSV, status_rows)
    review_fields = ["case_id", "candidate_id", "usable", "hit_status", "needs_edit", "reviewer", "comment"]
    write_rows(REVIEW_CSV, review_rows, review_fields)
    print(STATUS_CSV)
    print(REVIEW_CSV)


if __name__ == "__main__":
    main()
