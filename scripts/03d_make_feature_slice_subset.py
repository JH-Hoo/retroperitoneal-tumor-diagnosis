#!/usr/bin/env python3
import csv
import hashlib
import json
import os
from statistics import NormalDist
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LABEL_CSV = PROJECT_ROOT / "data" / "labels" / "labels_5class.csv"
SOURCE_FEATURE_NAME = os.environ.get("SOURCE_FEATURE_NAME", "features_cache_96slice_resnet18")
FEATURE_NAME = os.environ.get("FEATURE_NAME", "features_cache_96slice_resnet18_subset")
SUBSET_MODE = os.environ.get("SUBSET_MODE", "uniform")
NUM_SLICES = int(os.environ.get("NUM_SLICES", "64"))
GAUSS_MU = float(os.environ.get("GAUSS_MU", "0.5"))
GAUSS_SIGMA = float(os.environ.get("GAUSS_SIGMA", "0.22"))

SOURCE_DIR = PROJECT_ROOT / "data" / SOURCE_FEATURE_NAME / "features"
OUT_DIR = PROJECT_ROOT / "data" / FEATURE_NAME
FEATURE_DIR = OUT_DIR / "features"


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


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def indices(n):
    if SUBSET_MODE == "uniform":
        return np.linspace(0, n - 1, NUM_SLICES).round().astype(int)
    if SUBSET_MODE == "center":
        start = max(0, (n - NUM_SLICES) // 2)
        return np.arange(start, start + NUM_SLICES).clip(0, n - 1)
    if SUBSET_MODE == "upper":
        return np.linspace(0, int(round((n - 1) * 0.75)), NUM_SLICES).round().astype(int)
    if SUBSET_MODE == "lower":
        return np.linspace(int(round((n - 1) * 0.25)), n - 1, NUM_SLICES).round().astype(int)
    if SUBSET_MODE == "gauss":
        dist = NormalDist(mu=GAUSS_MU, sigma=GAUSS_SIGMA)
        eps = 1.0 / (NUM_SLICES * 4)
        qs = np.linspace(eps, 1.0 - eps, NUM_SLICES)
        pos = np.asarray([dist.inv_cdf(float(q)) for q in qs])
        pos = np.clip(pos, 0.0, 1.0)
        return np.rint(pos * (n - 1)).astype(int)
    raise ValueError(f"unknown SUBSET_MODE: {SUBSET_MODE}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    rows = read_rows(LABEL_CSV)
    checksum_rows = []
    idx_text = None
    for i, row in enumerate(rows, 1):
        case_id = row["case_id"]
        feat = torch.load(SOURCE_DIR / f"{case_id}.pt", map_location="cpu", weights_only=False)
        idx = indices(feat.shape[0])
        out = feat[idx]
        out_path = FEATURE_DIR / f"{case_id}.pt"
        torch.save(out, out_path)
        idx_text = ";".join(map(str, idx.tolist()))
        checksum_rows.append(
            {
                "case_id": case_id,
                "feature": f"features/{case_id}.pt",
                "shape": ",".join(map(str, out.shape)),
                "dtype": str(out.dtype).replace("torch.", ""),
                "bytes": out_path.stat().st_size,
                "sha256": sha256(out_path),
            }
        )
        print(f"{i:03d}/{len(rows)} {case_id} {tuple(out.shape)}")

    write_rows(OUT_DIR / "features_sha256.csv", checksum_rows)
    summary = {
        "name": FEATURE_NAME,
        "source_feature_name": SOURCE_FEATURE_NAME,
        "subset_mode": SUBSET_MODE,
        "num_slices": NUM_SLICES,
        "gauss_mu": GAUSS_MU if SUBSET_MODE == "gauss" else None,
        "gauss_sigma": GAUSS_SIGMA if SUBSET_MODE == "gauss" else None,
        "selected_feature_indices": idx_text,
        "feature_shape": [NUM_SLICES, 512],
        "feature_dtype": "float16",
    }
    (OUT_DIR / "dataset_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_DIR / "README.md").write_text(
        f"# {FEATURE_NAME}\n\nFeature subset cache derived from `{SOURCE_FEATURE_NAME}`.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
