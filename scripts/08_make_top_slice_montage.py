#!/usr/bin/env python3
import csv
import os
from pathlib import Path

import matplotlib.pyplot as plt
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRED_CSV = Path(os.environ.get("PRED_CSV", PROJECT_ROOT / "runs" / "fusion_late_fusion" / "test_predictions.csv"))
CACHE_NAME = os.environ.get("CACHE_NAME", "cache_96slice")
TENSOR_DIR = PROJECT_ROOT / "data" / CACHE_NAME / "tensors"
OUT_DIR = Path(os.environ.get("OUT_DIR", PRED_CSV.parent / "top_slice_montage"))
MAX_CASES = int(os.environ.get("MAX_CASES", "80"))
OFFSETS = [int(x) for x in os.environ.get("OFFSETS", "-6,-3,0,3,6").split(",")]
VIEW_ID = int(os.environ.get("VIEW_ID", "0"))


def read_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_rows(path, rows):
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def tensor_path(case_id):
    view_path = TENSOR_DIR / f"{case_id}_view{VIEW_ID}.pt"
    if view_path.exists():
        return view_path
    return TENSOR_DIR / f"{case_id}.pt"


def make_montage(row):
    case_id = row["case_id"]
    top = row.get("top_slice_index_in_bag", "")
    center = int(top) if top != "" else 48
    tensor = torch.load(tensor_path(case_id), map_location="cpu", weights_only=False)
    indices = [min(max(center + offset, 0), tensor.shape[0] - 1) for offset in OFFSETS]
    fig, axes = plt.subplots(1, len(indices), figsize=(2.2 * len(indices), 2.4), dpi=150)
    if len(indices) == 1:
        axes = [axes]
    for ax, idx in zip(axes, indices):
        ax.imshow(tensor[idx, 0].numpy(), cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"z={idx}", fontsize=8)
        ax.axis("off")
    title = f"{case_id} true={row['true_label_5']} pred={row.get('pred_label', row.get('pred_id', ''))} p={float(row['prob_nonbenign_actionable']):.3f}"
    fig.suptitle(title, fontsize=9)
    out_path = OUT_DIR / f"{case_id}.png"
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return {
        "case_id": case_id,
        "true_label_5": row["true_label_5"],
        "true_id": row["true_id"],
        "pred_id": row.get("pred_id", ""),
        "prob_nonbenign_actionable": row["prob_nonbenign_actionable"],
        "top_slice_index_in_bag": top,
        "montage": str(out_path.relative_to(OUT_DIR)),
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = read_rows(PRED_CSV)
    error_first = sorted(rows, key=lambda r: int(r.get("true_id", "0")) == int(r.get("pred_id", "0")))
    manifest = [make_montage(r) for r in error_first[:MAX_CASES]]
    write_rows(OUT_DIR / "montage_manifest.csv", manifest)
    print(f"wrote {len(manifest)} montages to {OUT_DIR}")


if __name__ == "__main__":
    main()
