#!/usr/bin/env python3
import csv
import hashlib
import json
import os
from pathlib import Path

import torch
from torchvision.models import (
    DenseNet121_Weights,
    ResNet18_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
    densenet121,
    resnet18,
    resnet34,
    resnet50,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_NAME = os.environ.get("CACHE_NAME", "cache_96slice_aug5")
BACKBONE = os.environ.get("BACKBONE", "resnet18")
FEATURE_NAME = os.environ.get("FEATURE_NAME", f"features_{CACHE_NAME}_{BACKBONE}")
NUM_VIEWS = int(os.environ.get("NUM_VIEWS", "5"))
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


def make_model():
    if BACKBONE == "resnet18":
        model = resnet18(weights=ResNet18_Weights.DEFAULT)
        model.fc = torch.nn.Identity()
    elif BACKBONE == "resnet34":
        model = resnet34(weights=ResNet34_Weights.DEFAULT)
        model.fc = torch.nn.Identity()
    elif BACKBONE == "resnet50":
        model = resnet50(weights=ResNet50_Weights.DEFAULT)
        model.fc = torch.nn.Identity()
    elif BACKBONE == "densenet121":
        model = densenet121(weights=DenseNet121_Weights.DEFAULT)
        model.classifier = torch.nn.Identity()
    else:
        raise ValueError(f"unknown BACKBONE: {BACKBONE}")
    return model


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


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    dev = device()
    model = make_model().to(dev).eval()
    for p in model.parameters():
        p.requires_grad = False

    rows = read_rows(LABEL_CSV)
    checksum_rows = []
    feature_shape = None
    with torch.no_grad():
        for i, row in enumerate(rows, 1):
            case_id = row["case_id"]
            for view_id in range(NUM_VIEWS):
                x = torch.load(CACHE_DIR / f"{case_id}_view{view_id}.pt", map_location="cpu", weights_only=False).float().div(255.0)
                x = ((x - IMAGENET_MEAN) / IMAGENET_STD).to(dev)
                feat = model(x).flatten(1).cpu().to(torch.float16)
                feature_shape = list(feat.shape)
                rel = f"features/{case_id}_view{view_id}.pt"
                out_path = OUT_DIR / rel
                torch.save(feat, out_path)
                checksum_rows.append(
                    {
                        "case_id": case_id,
                        "view_id": view_id,
                        "feature": rel,
                        "shape": ",".join(map(str, feat.shape)),
                        "dtype": "float16",
                        "bytes": out_path.stat().st_size,
                        "sha256": sha256(out_path),
                    }
                )
            print(f"{i:03d}/{len(rows)} {case_id} views={NUM_VIEWS}")

    write_rows(OUT_DIR / "features_sha256.csv", checksum_rows)
    summary = {
        "name": FEATURE_NAME,
        "source_cache": CACHE_NAME,
        "backbone": BACKBONE,
        "pretrained": "ImageNet",
        "num_cases": len(rows),
        "num_views": NUM_VIEWS,
        "feature_shape_per_view": feature_shape,
        "feature_dtype": "float16",
    }
    (OUT_DIR / "dataset_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_DIR / "README.md").write_text(
        f"# {FEATURE_NAME}\n\nFrozen {BACKBONE} slice features extracted from `{CACHE_NAME}`.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
