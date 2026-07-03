#!/usr/bin/env python3
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLICKS_CSV = PROJECT_ROOT / "data" / "annotations" / "tumor_clicks_pilot_30.csv"
STATUS_CSV = PROJECT_ROOT / "data" / "segmentations" / "uls23_candidates" / "uls23_candidate_status.csv"
OUT_DIR = PROJECT_ROOT / "data" / "qc" / "uls23_review_sheets"
REVIEW_CSV = PROJECT_ROOT / "data" / "annotations" / "lesion_mask_review_pilot_30.csv"


def read_rows(path):
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_rows(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def window(x, low=-200.0, high=400.0):
    return np.clip((x - low) / (high - low), 0, 1)


def make_sheet(status, click_by_case):
    case_id = status["case_id"]
    if status["status"] != "ok" or not status["mask_path"]:
        return ""
    click = click_by_case[case_id]
    img = nib.load(str(PROJECT_ROOT / click["nifti"]))
    vol = np.asarray(img.get_fdata(dtype=np.float32))
    mask_img = nib.load(str(PROJECT_ROOT / status["mask_path"]))
    mask = np.asarray(mask_img.get_fdata(dtype=np.float32)) > 0
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return ""
    center_z = int(np.round(coords[:, 2].mean()))
    z_indices = np.linspace(max(0, center_z - 24), min(vol.shape[2] - 1, center_z + 24), 6).round().astype(int)

    fig, axes = plt.subplots(2, 3, figsize=(10.5, 7), dpi=150)
    for ax, z in zip(axes.flat, z_indices):
        ax.imshow(window(vol[:, :, z]).T, cmap="gray", origin="lower")
        ax.contour(mask[:, :, z].T, levels=[0.5], colors=["#ff3b30"], linewidths=1.2)
        ax.set_title(f"z={z}", fontsize=8)
        ax.set_axis_off()
    fig.suptitle(f"{case_id} | candidate 001 | review mask", fontsize=11)
    fig.tight_layout()
    out_path = OUT_DIR / f"{case_id}_candidate_001_review.png"
    fig.savefig(out_path)
    plt.close(fig)
    return str(out_path.relative_to(PROJECT_ROOT))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    click_by_case = {row["case_id"]: row for row in read_rows(CLICKS_CSV)}
    status_rows = read_rows(STATUS_CSV)
    review_rows = []
    for status in status_rows:
        if status["status"] != "ok":
            continue
        sheet = make_sheet(status, click_by_case)
        review_rows.append(
            {
                "case_id": status["case_id"],
                "candidate_id": status["candidate_id"],
                "usable": "",
                "hit_status": "",
                "needs_edit": "",
                "reviewer": "",
                "comment": "",
                "review_sheet": sheet,
                "mask_path": status["mask_path"],
            }
        )
        print(status["case_id"], sheet, flush=True)

    fields = ["case_id", "candidate_id", "usable", "hit_status", "needs_edit", "reviewer", "comment", "review_sheet", "mask_path"]
    write_rows(REVIEW_CSV, review_rows, fields)
    print(REVIEW_CSV)


if __name__ == "__main__":
    main()
