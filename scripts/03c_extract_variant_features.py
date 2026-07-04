#!/usr/bin/env python3
import csv
import hashlib
import json
import os
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, resnet18


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRIVATE_ROOT = PROJECT_ROOT / "data_private"
RAW_ROOT = PRIVATE_ROOT / "standard" / "images"
LABEL_CSV = PROJECT_ROOT / "data" / "labels" / "labels_5class.csv"

FEATURE_NAME = os.environ.get("FEATURE_NAME", "features_variant_resnet18")
OUT_DIR = PROJECT_ROOT / "data" / FEATURE_NAME
FEATURE_DIR = OUT_DIR / "features"
AUDIT_ROOT = PRIVATE_ROOT / "audit"

NUM_SLICES = int(os.environ.get("NUM_SLICES", "96"))
IMAGE_SIZE = int(os.environ.get("IMAGE_SIZE", "224"))
SLICE_MODE = os.environ.get("SLICE_MODE", "uniform")
CROP_MODE = os.environ.get("CROP_MODE", "whole")
FEATURE_BATCH = int(os.environ.get("FEATURE_BATCH", "64"))
BODY_THRESHOLD = float(os.environ.get("BODY_THRESHOLD", "-500"))
BODY_Z_AREA_FRAC = float(os.environ.get("BODY_Z_AREA_FRAC", "0.01"))
BODY_Z_MARGIN_FRAC = float(os.environ.get("BODY_Z_MARGIN_FRAC", "0.05"))
BODY_XY_PAD = int(os.environ.get("BODY_XY_PAD", "16"))
FORCE = int(os.environ.get("FORCE", "0"))

DEFAULT_WINDOWS = [(-160.0, 240.0), (-200.0, 100.0), (-200.0, 400.0)]
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def parse_windows(text):
    if not text:
        return DEFAULT_WINDOWS
    out = []
    for part in text.split(";"):
        low, high = part.split(",")
        out.append((float(low), float(high)))
    return out


WINDOWS = parse_windows(os.environ.get("WINDOWS", ""))


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


def uniform_indices(z):
    return np.linspace(0, z - 1, NUM_SLICES).round().astype(int), {"slice_start": 0, "slice_end": z - 1}


def center_fraction_indices(z, frac=0.80):
    pad = (1.0 - frac) / 2.0
    start = int(round((z - 1) * pad))
    end = int(round((z - 1) * (1.0 - pad)))
    return np.linspace(start, end, NUM_SLICES).round().astype(int), {"slice_start": start, "slice_end": end}


def body_z_indices(vol):
    z = vol.shape[2]
    body = vol > BODY_THRESHOLD
    areas = body.reshape(-1, z).sum(axis=0)
    min_area = vol.shape[0] * vol.shape[1] * BODY_Z_AREA_FRAC
    valid = np.where(areas > min_area)[0]
    if len(valid) == 0:
        return uniform_indices(z)
    margin = int(round((valid[-1] - valid[0] + 1) * BODY_Z_MARGIN_FRAC))
    start = max(0, int(valid[0]) - margin)
    end = min(z - 1, int(valid[-1]) + margin)
    return np.linspace(start, end, NUM_SLICES).round().astype(int), {"slice_start": start, "slice_end": end}


def select_indices(vol):
    if SLICE_MODE == "uniform":
        return uniform_indices(vol.shape[2])
    if SLICE_MODE == "center80":
        return center_fraction_indices(vol.shape[2], 0.80)
    if SLICE_MODE == "center60":
        return center_fraction_indices(vol.shape[2], 0.60)
    if SLICE_MODE == "body_z":
        return body_z_indices(vol)
    raise ValueError(f"unknown SLICE_MODE: {SLICE_MODE}")


def body_xy_bbox(vol):
    body = (vol > BODY_THRESHOLD).any(axis=2)
    xs, ys = np.where(body)
    if len(xs) == 0:
        return 0, vol.shape[0], 0, vol.shape[1], True
    x0 = max(0, int(xs.min()) - BODY_XY_PAD)
    x1 = min(vol.shape[0], int(xs.max()) + BODY_XY_PAD + 1)
    y0 = max(0, int(ys.min()) - BODY_XY_PAD)
    y1 = min(vol.shape[1], int(ys.max()) + BODY_XY_PAD + 1)
    return x0, x1, y0, y1, False


def window_channel(x, low, high):
    x = np.clip(x, low, high)
    return (x - low) / (high - low)


def make_input(case_id):
    raw_img = nib.load(str(RAW_ROOT / f"{case_id}.nii.gz"))
    img = nib.as_closest_canonical(raw_img)
    vol = np.asarray(img.get_fdata(dtype=np.float32))
    idx, slice_meta = select_indices(vol)
    slices = vol[:, :, idx].transpose(2, 0, 1)
    x0, x1, y0, y1, crop_failed = 0, vol.shape[0], 0, vol.shape[1], False
    if CROP_MODE == "body_xy":
        x0, x1, y0, y1, crop_failed = body_xy_bbox(vol)
        slices = slices[:, x0:x1, y0:y1]
    elif CROP_MODE != "whole":
        raise ValueError(f"unknown CROP_MODE: {CROP_MODE}")
    channels = [window_channel(slices, low, high) for low, high in WINDOWS]
    x = torch.from_numpy(np.stack(channels, axis=1).astype(np.float32))
    x = F.interpolate(x, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False)
    audit = {
        "case_id": case_id,
        "shape": "x".join(map(str, vol.shape)),
        "spacing": "x".join(f"{v:.6g}" for v in img.header.get_zooms()[:3]),
        "orientation_original": "".join(nib.aff2axcodes(raw_img.affine)),
        "orientation_canonical": "".join(nib.aff2axcodes(img.affine)),
        "slice_mode": SLICE_MODE,
        "selected_slice_indices": ";".join(map(str, idx.tolist())),
        "slice_start": slice_meta["slice_start"],
        "slice_end": slice_meta["slice_end"],
        "windows": ";".join(f"{low:.2f},{high:.2f}" for low, high in WINDOWS),
        "crop_mode": CROP_MODE,
        "crop_bbox_xy": f"{x0},{x1},{y0},{y1}",
        "crop_failed": int(crop_failed),
    }
    return x.clamp(0, 1), audit


def extract_features(model, x, dev):
    feats = []
    with torch.no_grad():
        for start in range(0, x.shape[0], FEATURE_BATCH):
            batch = x[start : start + FEATURE_BATCH]
            batch = ((batch - IMAGENET_MEAN) / IMAGENET_STD).to(dev)
            feats.append(model(batch).flatten(1).cpu())
    return torch.cat(feats, dim=0).to(torch.float16)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    AUDIT_ROOT.mkdir(parents=True, exist_ok=True)
    dev = device()
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = torch.nn.Identity()
    model.to(dev).eval()
    for p in model.parameters():
        p.requires_grad = False

    checksum_rows, audit_rows = [], []
    rows = read_rows(LABEL_CSV)
    for i, row in enumerate(rows, 1):
        case_id = row["case_id"]
        out_path = FEATURE_DIR / f"{case_id}.pt"
        if out_path.exists() and not FORCE:
            feat = torch.load(out_path, map_location="cpu", weights_only=False)
            audit = {"case_id": case_id, "slice_mode": SLICE_MODE, "windows": ";".join(f"{a:.2f},{b:.2f}" for a, b in WINDOWS), "crop_mode": CROP_MODE}
        else:
            x, audit = make_input(case_id)
            feat = extract_features(model, x, dev)
            torch.save(feat, out_path)
        checksum_rows.append(
            {
                "case_id": case_id,
                "feature": f"features/{case_id}.pt",
                "shape": ",".join(map(str, feat.shape)),
                "dtype": "float16",
                "bytes": out_path.stat().st_size,
                "sha256": sha256(out_path),
            }
        )
        audit_rows.append(audit)
        print(f"{i:03d}/{len(rows)} {case_id} {tuple(feat.shape)}")

    write_rows(OUT_DIR / "features_sha256.csv", checksum_rows)
    write_rows(AUDIT_ROOT / f"feature_audit_{FEATURE_NAME}.csv", audit_rows)
    summary = {
        "name": FEATURE_NAME,
        "source": "data_private/standard/images",
        "backbone": "resnet18",
        "pretrained": "ImageNet",
        "num_cases": len(rows),
        "feature_shape": [NUM_SLICES, 512],
        "feature_dtype": "float16",
        "num_slices": NUM_SLICES,
        "slice_mode": SLICE_MODE,
        "crop_mode": CROP_MODE,
        "windows": WINDOWS,
    }
    (OUT_DIR / "dataset_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_DIR / "README.md").write_text(
        f"# {FEATURE_NAME}\n\nDirect NIfTI-to-ResNet18 feature cache for window/slice/crop experiments.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
