#!/usr/bin/env python3
import csv
import hashlib
import json
import os
from pathlib import Path

import torch
from torchvision.models import ResNet18_Weights, resnet18


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_NAME = os.environ.get("CACHE_NAME", "cache_96slice")
FEATURE_NAME = os.environ.get("FEATURE_NAME", f"features_{CACHE_NAME}_resnet18")
LABEL_CSV = PROJECT_ROOT / "data" / "labels" / "labels_5class.csv"
CACHE_DIR = PROJECT_ROOT / "data" / CACHE_NAME / "tensors"
OUT_DIR = PROJECT_ROOT / "data" / FEATURE_NAME
FEATURE_DIR = OUT_DIR / "features"

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def read_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_rows(path, rows):
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


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    dev = device()
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = torch.nn.Identity()
    model.to(dev).eval()
    for p in model.parameters():
        p.requires_grad = False

    rows = read_rows(LABEL_CSV)
    checksum_rows = []
    with torch.no_grad():
        for i, row in enumerate(rows, 1):
            case_id = row["case_id"]
            x = torch.load(CACHE_DIR / f"{case_id}.pt", map_location="cpu").float().div(255.0)
            x = ((x - IMAGENET_MEAN) / IMAGENET_STD).to(dev)
            feat = model(x).cpu().to(torch.float16)
            out_path = FEATURE_DIR / f"{case_id}.pt"
            torch.save(feat, out_path)
            checksum_rows.append(
                {
                    "case_id": case_id,
                    "feature": f"features/{case_id}.pt",
                    "shape": "96,512",
                    "dtype": "float16",
                    "bytes": out_path.stat().st_size,
                    "sha256": sha256(out_path),
                }
            )
            print(f"{i:03d}/{len(rows)} {case_id} {tuple(feat.shape)}")

    write_rows(OUT_DIR / "features_sha256.csv", checksum_rows)
    summary = {
        "name": FEATURE_NAME,
        "source_cache": CACHE_NAME,
        "backbone": "resnet18",
        "pretrained": "ImageNet",
        "num_cases": len(rows),
        "feature_shape": [96, 512],
        "feature_dtype": "float16",
    }
    (OUT_DIR / "dataset_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_DIR / "README.md").write_text(
        f"# {FEATURE_NAME}\n\nFrozen ResNet18 slice features extracted from `{CACHE_NAME}`.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
