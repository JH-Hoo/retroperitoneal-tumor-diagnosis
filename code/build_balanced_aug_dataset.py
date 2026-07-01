#!/usr/bin/env python3
import csv
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = PROJECT_ROOT / "dataset_standard_v0"
OUT_ROOT = PROJECT_ROOT / "dataset_96slice_balanced_aug_v0"
TENSOR_DIR = OUT_ROOT / "tensors"

NUM_SLICES = 96
IMAGE_SIZE = 224
SEED = 20260702
TRAIN_MULTIPLIERS = {
    "肉瘤类": 14,
    "良性神经源性肿瘤": 14,
    "副神经节瘤": 6,
    "淋巴瘤": 1,
}
WINDOWS = [
    (-160.0, 240.0),
    (-200.0, 100.0),
    (-200.0, 400.0),
]


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


def window_channel(x, low, high):
    x = np.clip(x, low, high)
    return (x - low) / (high - low)


def slice_indices(z, rng):
    edges = np.linspace(0, z, NUM_SLICES + 1)
    idx = []
    for a, b in zip(edges[:-1], edges[1:]):
        lo = int(np.floor(a))
        hi = max(lo + 1, int(np.ceil(b)))
        idx.append(rng.integers(lo, min(hi, z)))
    return np.array(idx, dtype=int)


def make_tensor(nifti_path, rng):
    img = nib.load(str(nifti_path))
    vol = np.asarray(img.get_fdata(dtype=np.float32))
    idx = slice_indices(vol.shape[2], rng)
    slices = vol[:, :, idx].transpose(2, 0, 1)
    channels = [window_channel(slices, low, high) for low, high in WINDOWS]
    x = torch.from_numpy(np.stack(channels, axis=1).astype(np.float32))
    x = F.interpolate(x, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False)
    return (x.clamp(0, 1).mul(255).round()).to(torch.uint8)


def expanded_rows(raw_rows):
    rows = []
    for r in raw_rows:
        n = TRAIN_MULTIPLIERS[r["label_4"]] if r["split"] == "train" else 1
        for aug_id in range(n):
            rr = dict(r)
            rr["source_group"] = r["group"]
            rr["source_image"] = r["image"]
            rr["aug_id"] = f"aug{aug_id:02d}" if r["split"] == "train" else "test00"
            rr["case_aug_id"] = f"{r['group']}_{rr['aug_id']}"
            rr["tensor"] = f"tensors/{rr['case_aug_id']}.pt"
            rows.append(rr)
    return rows


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    TENSOR_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "splits").mkdir(exist_ok=True)

    raw_rows = read_rows(RAW_ROOT / "all.csv")
    rows = expanded_rows(raw_rows)
    checksum_rows = []
    for i, r in enumerate(rows, 1):
        seed = SEED + int(r["source_group"][1:]) * 100 + int(r["aug_id"][-2:])
        rng = np.random.default_rng(seed)
        out_path = OUT_ROOT / r["tensor"]
        tensor = make_tensor(RAW_ROOT / r["source_image"], rng)
        torch.save(tensor, out_path)
        checksum_rows.append(
            {
                "case_aug_id": r["case_aug_id"],
                "source_group": r["source_group"],
                "split": r["split"],
                "label_4": r["label_4"],
                "tensor": r["tensor"],
                "shape": "96,3,224,224",
                "dtype": "uint8",
                "bytes": out_path.stat().st_size,
                "sha256": sha256(out_path),
            }
        )
        print(f"{i:03d}/{len(rows)} {r['case_aug_id']} {r['label_4']} {tuple(tensor.shape)}")

    write_rows(OUT_ROOT / "all.csv", rows)
    write_rows(OUT_ROOT / "splits" / "train.csv", [r for r in rows if r["split"] == "train"])
    write_rows(OUT_ROOT / "splits" / "test.csv", [r for r in rows if r["split"] == "test"])
    write_rows(OUT_ROOT / "tensors_sha256.csv", checksum_rows)

    shutil.copy2(RAW_ROOT / "label_mapping.json", OUT_ROOT / "label_mapping.json")
    for name in ["labels.csv", "metadata.csv"]:
        shutil.copy2(RAW_ROOT / name, OUT_ROOT / name)

    train_counts = Counter(r["label_4"] for r in rows if r["split"] == "train")
    test_counts = Counter(r["label_4"] for r in rows if r["split"] == "test")
    summary = {
        "name": "dataset_96slice_balanced_aug_v0",
        "source_dataset": "dataset_standard_v0",
        "purpose": "experimental class-balanced training-time augmentation cache",
        "num_source_cases": len(raw_rows),
        "num_cached_bags": len(rows),
        "num_train_bags": sum(r["split"] == "train" for r in rows),
        "num_test_bags": sum(r["split"] == "test" for r in rows),
        "train_counts": dict(train_counts),
        "test_counts": dict(test_counts),
        "train_multipliers": TRAIN_MULTIPLIERS,
        "tensor_shape": [NUM_SLICES, 3, IMAGE_SIZE, IMAGE_SIZE],
        "tensor_dtype": "uint8",
        "sampling": "stratified random z sampling; split z axis into 96 bins and sample one slice per bin",
        "windows": WINDOWS,
        "normalization": "stored as 0-255 windowed images; divide by 255 and apply ImageNet mean/std at training time",
    }
    (OUT_ROOT / "dataset_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_ROOT / "README.md").write_text(
        "# Retroperitoneal Tumor CT Dataset 96-slice Balanced Aug v0\n\n"
        "Experimental cache derived from `dataset_standard_v0` for class-balanced MIL smoke tests.\n\n"
        "Only the training split is augmented. Test cases remain one fixed cached bag per source case.\n"
        "Each cached bag is a PyTorch tensor with shape `96 x 3 x 224 x 224` and dtype `uint8`.\n"
        "Training multipliers: 肉瘤类 14x, 良性神经源性肿瘤 14x, 副神经节瘤 6x, 淋巴瘤 1x.\n"
        "This dataset must be treated as augmentation/oversampling, not as additional independent patients.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
