#!/usr/bin/env python3
import csv
import json
import os
from concurrent.futures import ProcessPoolExecutor
from math import ceil
from pathlib import Path

import nibabel as nib
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PILOT_CSV = PROJECT_ROOT / "data" / "annotations" / "totalseg_pilot_30.csv"
OUT_DIR = PROJECT_ROOT / "data" / "derived" / "retroperitoneal_roi"
SUMMARY_CSV = OUT_DIR / "summary.csv"
MARGIN_XY_MM = 70.0
MARGIN_Z_MM = 50.0
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "4"))

ANCHOR_MASKS = [
    "kidney_right",
    "kidney_left",
    "adrenal_gland_right",
    "adrenal_gland_left",
    "sacrum",
    "vertebrae_S1",
    "vertebrae_L5",
    "vertebrae_L4",
    "vertebrae_L3",
    "vertebrae_L2",
    "vertebrae_L1",
    "aorta",
    "inferior_vena_cava",
    "iliac_artery_left",
    "iliac_artery_right",
    "iliac_vena_left",
    "iliac_vena_right",
    "iliopsoas_left",
    "iliopsoas_right",
]
Z_ANCHOR_MASKS = [
    "kidney_right",
    "kidney_left",
    "adrenal_gland_right",
    "adrenal_gland_left",
    "sacrum",
    "vertebrae_S1",
    "vertebrae_L5",
    "vertebrae_L4",
    "vertebrae_L3",
    "vertebrae_L2",
    "vertebrae_L1",
]


def read_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def mask_unions(seg_dir, shape):
    xy_union = np.zeros(shape, dtype=bool)
    z_union = np.zeros(shape, dtype=bool)
    found = []
    missing = []
    z_found = []
    z_missing = []
    z_names = set(Z_ANCHOR_MASKS)
    for name in ANCHOR_MASKS:
        path = seg_dir / f"{name}.nii.gz"
        if path.exists():
            mask = np.asanyarray(nib.load(str(path)).dataobj) > 0
            xy_union |= mask
            found.append(name)
            if name in z_names:
                z_union |= mask
                z_found.append(name)
        else:
            missing.append(name)
            if name in z_names:
                z_missing.append(name)
    return xy_union, z_union, found, missing, z_found, z_missing


def padded_bbox(xy_mask, z_mask, shape, spacing):
    xy_coords = np.argwhere(xy_mask)
    z_coords = np.argwhere(z_mask)
    fallback_full = len(xy_coords) == 0 or len(z_coords) == 0
    if fallback_full:
        mins = np.array([0, 0, 0])
        maxs = np.array(shape)
    else:
        xy_mins = xy_coords.min(axis=0)
        xy_maxs = xy_coords.max(axis=0) + 1
        z_mins = z_coords.min(axis=0)
        z_maxs = z_coords.max(axis=0) + 1
        mins = np.array([xy_mins[0], xy_mins[1], z_mins[2]])
        maxs = np.array([xy_maxs[0], xy_maxs[1], z_maxs[2]])
    pad = np.array([ceil(MARGIN_XY_MM / spacing[0]), ceil(MARGIN_XY_MM / spacing[1]), ceil(MARGIN_Z_MM / spacing[2])])
    mins = np.maximum(0, mins - pad)
    maxs = np.minimum(np.array(shape), maxs + pad)
    return [int(mins[0]), int(maxs[0]), int(mins[1]), int(maxs[1]), int(mins[2]), int(maxs[2])], fallback_full


def process_row(row):
    case_id = row["case_id"]
    seg_dir = PROJECT_ROOT / row["totalseg_dir"]
    if not list(seg_dir.glob("*.nii.gz")):
        return None
    image = nib.load(str(PROJECT_ROOT / row["nifti"]))
    shape = image.shape[:3]
    spacing = image.header.get_zooms()[:3]
    xy_union, z_union, found, missing, z_found, z_missing = mask_unions(seg_dir, shape)
    bbox, fallback_full = padded_bbox(xy_union, z_union, shape, spacing)
    voxels = (bbox[1] - bbox[0]) * (bbox[3] - bbox[2]) * (bbox[5] - bbox[4])
    payload = {
        "case_id": case_id,
        "label_5": row["label_5"],
        "image": row["nifti"],
        "totalseg_dir": row["totalseg_dir"],
        "bbox_ijk_exclusive": bbox,
        "image_shape": list(map(int, shape)),
        "spacing_mm": [float(x) for x in spacing],
        "margin_xy_mm": MARGIN_XY_MM,
        "margin_z_mm": MARGIN_Z_MM,
        "volume_fraction": voxels / float(np.prod(shape)),
        "fallback_full_volume": fallback_full,
        "anchor_masks_found": found,
        "anchor_masks_missing": missing,
        "z_anchor_masks_found": z_found,
        "z_anchor_masks_missing": z_missing,
    }
    out_path = OUT_DIR / f"{case_id}.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "case_id": case_id,
        "label_5": row["label_5"],
        "bbox_ijk_exclusive": ";".join(map(str, bbox)),
        "image_shape": "x".join(map(str, shape)),
        "spacing_mm": "x".join(f"{x:.4g}" for x in spacing),
        "volume_fraction": f"{payload['volume_fraction']:.4f}",
        "fallback_full_volume": str(fallback_full),
        "num_anchor_masks_found": len(found),
        "num_anchor_masks_missing": len(missing),
        "roi_json": f"data/derived/retroperitoneal_roi/{case_id}.json",
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        for result in executor.map(process_row, read_rows(PILOT_CSV)):
            if result:
                rows.append(result)
                print(result["case_id"], result["volume_fraction"], result["num_anchor_masks_found"], "masks", flush=True)

    write_rows(SUMMARY_CSV, rows)
    print(SUMMARY_CSV)


if __name__ == "__main__":
    main()
