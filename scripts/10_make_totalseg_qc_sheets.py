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
OUT_DIR = PROJECT_ROOT / "data" / "qc" / "contact_sheets"
OUT_CSV = PROJECT_ROOT / "data" / "qc" / "totalseg_contact_sheets.csv"
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
    image = nib.load(str(PROJECT_ROOT / row["nifti"]))
    vol = np.asarray(image.get_fdata(dtype=np.float32))
    roi = json.loads((ROI_DIR / f"{case_id}.json").read_text(encoding="utf-8"))
    x0, x1, y0, y1, z0, z1 = roi["bbox_ijk_exclusive"]
    z_indices = np.linspace(z0, max(z0, z1 - 1), 6).round().astype(int)

    fig, axes = plt.subplots(2, 3, figsize=(10.5, 7), dpi=150)
    for ax, z in zip(axes.flat, z_indices):
        ax.imshow(window(vol[:, :, z]).T, cmap="gray", origin="lower")
        rect = patches.Rectangle((x0, y0), x1 - x0, y1 - y0, linewidth=1.4, edgecolor="#ff3b30", facecolor="none")
        ax.add_patch(rect)
        ax.set_title(f"z={z}", fontsize=9)
        ax.set_axis_off()
    fig.suptitle(f"{case_id} | {LABEL_EN.get(row['label_5'], row['label_5'])} | TotalSeg retroperitoneal ROI", fontsize=11)
    fig.tight_layout()
    out_path = OUT_DIR / f"{case_id}_totalseg_roi.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    todo = [row for row in read_rows(PILOT_CSV) if (ROI_DIR / f"{row['case_id']}.json").exists()]
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        for row, out_path in zip(todo, executor.map(make_sheet, todo)):
            rows.append({"case_id": row["case_id"], "label_5": row["label_5"], "qc_sheet": str(out_path.relative_to(PROJECT_ROOT))})
            print(out_path, flush=True)
    write_rows(OUT_CSV, rows)
    print(OUT_CSV)


if __name__ == "__main__":
    main()
