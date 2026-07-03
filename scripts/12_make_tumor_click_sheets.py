#!/usr/bin/env python3
import csv
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PILOT_CSV = PROJECT_ROOT / "data" / "annotations" / "totalseg_pilot_30.csv"
ROI_DIR = PROJECT_ROOT / "data" / "derived" / "retroperitoneal_roi"
OUT_DIR = PROJECT_ROOT / "data" / "qc" / "tumor_click_sheets"
OUT_CSV = PROJECT_ROOT / "data" / "annotations" / "tumor_clicks_pilot_30.csv"
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "4"))

LABEL_EN = {
    "肉瘤类": "Sarcoma",
    "良性神经源性肿瘤": "Benign neurogenic",
    "PPGL": "PPGL",
    "淋巴瘤": "Lymphoma",
    "胃肠道间质瘤": "GIST",
}


def read_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def window(x, low=-200.0, high=400.0):
    return np.clip((x - low) / (high - low), 0, 1)


def make_sheet(row):
    case_id = row["case_id"]
    roi_path = ROI_DIR / f"{case_id}.json"
    if not roi_path.exists():
        return None

    roi = json.loads(roi_path.read_text(encoding="utf-8"))
    x0, x1, y0, y1, z0, z1 = roi["bbox_ijk_exclusive"]
    image = nib.load(str(PROJECT_ROOT / row["nifti"]))
    vol = np.asarray(image.get_fdata(dtype=np.float32))
    z_indices = np.linspace(z0, max(z0, z1 - 1), 12).round().astype(int)
    label = LABEL_EN.get(row["label_5"], row["label_5"])

    fig, axes = plt.subplots(3, 4, figsize=(13.2, 9.6), dpi=150)
    for ax, z in zip(axes.flat, z_indices):
        ax.imshow(window(vol[:, :, z]).T, cmap="gray", origin="lower")
        rect = patches.Rectangle((x0, y0), x1 - x0, y1 - y0, linewidth=1.2, edgecolor="#ff3b30", facecolor="none")
        ax.add_patch(rect)
        ax.set_title(f"z={z}", fontsize=8)
        ax.set_xlim(max(0, x0 - 24), min(vol.shape[0], x1 + 24))
        ax.set_ylim(max(0, y0 - 24), min(vol.shape[1], y1 + 24))
        ax.tick_params(labelsize=6)
    fig.suptitle(f"{case_id} | {label} | choose tumor center in original voxel coordinates", fontsize=11)
    fig.tight_layout()
    out_path = OUT_DIR / f"{case_id}_tumor_click_sheet.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = [row for row in read_rows(PILOT_CSV) if (ROI_DIR / f"{row['case_id']}.json").exists()]
    click_rows = []
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        for row, out_path in zip(rows, executor.map(make_sheet, rows)):
            if not out_path:
                continue
            rel_sheet = str(out_path.relative_to(PROJECT_ROOT))
            click_rows.append(
                {
                    "case_id": row["case_id"],
                    "label_5": row["label_5"],
                    "x_voxel": "",
                    "y_voxel": "",
                    "z_voxel": "",
                    "source": "",
                    "reviewer": "",
                    "comment": "",
                    "nifti": row["nifti"],
                    "retro_roi_json": row["retro_roi_json"],
                    "click_sheet": rel_sheet,
                }
            )
            print(out_path, flush=True)

    write_rows(OUT_CSV, click_rows)
    print(OUT_CSV)


if __name__ == "__main__":
    main()
