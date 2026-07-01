#!/usr/bin/env python3
import csv
import hashlib
import json
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = PROJECT_ROOT / "dataset_standard_v0"
OUT_ROOT = PROJECT_ROOT / "dataset_96slice_v0"
TENSOR_DIR = OUT_ROOT / "tensors"

NUM_SLICES = 96
IMAGE_SIZE = 224
WINDOWS = [
    (-160.0, 240.0),
    (-200.0, 100.0),
    (-200.0, 400.0),
]


def window_channel(x, low, high):
    x = np.clip(x, low, high)
    return (x - low) / (high - low)


def make_tensor(nifti_path):
    img = nib.load(str(nifti_path))
    vol = np.asarray(img.get_fdata(dtype=np.float32))
    z = vol.shape[2]
    idx = np.linspace(0, z - 1, NUM_SLICES).round().astype(int)
    slices = vol[:, :, idx].transpose(2, 0, 1)
    channels = [window_channel(slices, low, high) for low, high in WINDOWS]
    x = torch.from_numpy(np.stack(channels, axis=1).astype(np.float32))
    x = F.interpolate(x, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False)
    return (x.clamp(0, 1).mul(255).round()).to(torch.uint8)


def read_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_rows(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def with_tensor_paths(rows):
    out = []
    for r in rows:
        case_id = r["group"]
        rr = dict(r)
        rr["source_image"] = rr["image"]
        rr["tensor"] = f"tensors/{case_id}.pt"
        out.append(rr)
    return out


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    TENSOR_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "splits").mkdir(exist_ok=True)

    rows = with_tensor_paths(read_rows(RAW_ROOT / "all.csv"))
    checksum_rows = []
    for i, r in enumerate(rows, 1):
        out_path = OUT_ROOT / r["tensor"]
        tensor = make_tensor(RAW_ROOT / r["source_image"])
        torch.save(tensor, out_path)
        checksum_rows.append(
            {
                "group": r["group"],
                "tensor": r["tensor"],
                "shape": "96,3,224,224",
                "dtype": "uint8",
                "bytes": out_path.stat().st_size,
                "sha256": sha256(out_path),
            }
        )
        print(f"{i:02d}/{len(rows)} {r['group']} {tuple(tensor.shape)} {out_path.stat().st_size}")

    write_rows(OUT_ROOT / "all.csv", rows)
    write_rows(OUT_ROOT / "splits" / "train.csv", [r for r in rows if r["split"] == "train"])
    write_rows(OUT_ROOT / "splits" / "test.csv", [r for r in rows if r["split"] == "test"])
    write_rows(OUT_ROOT / "tensors_sha256.csv", checksum_rows)

    shutil.copy2(RAW_ROOT / "label_mapping.json", OUT_ROOT / "label_mapping.json")
    for name in ["labels.csv", "metadata.csv"]:
        shutil.copy2(RAW_ROOT / name, OUT_ROOT / name)

    summary = {
        "name": "dataset_96slice_v0",
        "source_dataset": "dataset_standard_v0",
        "num_cases": len(rows),
        "num_train": sum(r["split"] == "train" for r in rows),
        "num_test": sum(r["split"] == "test" for r in rows),
        "tensor_shape": [NUM_SLICES, 3, IMAGE_SIZE, IMAGE_SIZE],
        "tensor_dtype": "uint8",
        "windows": WINDOWS,
        "normalization": "stored as 0-255 windowed images; divide by 255 and apply ImageNet mean/std at training time",
    }
    (OUT_ROOT / "dataset_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_ROOT / "README.md").write_text(
        "# Retroperitoneal Tumor CT Dataset 96-slice v0\n\n"
        "Offline preprocessed cache derived from `dataset_standard_v0`.\n\n"
        "Each case is stored as one PyTorch tensor in `tensors/` with shape `96 x 3 x 224 x 224` and dtype `uint8`.\n"
        "The three channels are fixed CT windows: soft tissue `[-160, 240]`, fat-sensitive `[-200, 100]`, and wide abdomen `[-200, 400]`.\n"
        "Training code should convert tensors to float, divide by 255, then apply ImageNet mean/std normalization.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
