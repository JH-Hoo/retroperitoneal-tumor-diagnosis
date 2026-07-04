#!/usr/bin/env python3
import csv
import hashlib
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRIVATE_ROOT = PROJECT_ROOT / "data_private"
RAW_ROOT = PRIVATE_ROOT / "standard"
CACHE_NAME = os.environ.get("CACHE_NAME", "cache_96slice_aug5")
OUT_ROOT = PROJECT_ROOT / "data" / CACHE_NAME
TENSOR_DIR = OUT_ROOT / "tensors"
AUDIT_ROOT = PRIVATE_ROOT / "audit"

NUM_SLICES = int(os.environ.get("NUM_SLICES", "96"))
IMAGE_SIZE = int(os.environ.get("IMAGE_SIZE", "224"))
NUM_VIEWS = int(os.environ.get("NUM_VIEWS", "5"))
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "8"))
SLICE_MODE = os.environ.get("SLICE_MODE", "uniform")
BODY_THRESHOLD = float(os.environ.get("BODY_THRESHOLD", "-500"))
BODY_Z_AREA_FRAC = float(os.environ.get("BODY_Z_AREA_FRAC", "0.01"))
BODY_Z_MARGIN_FRAC = float(os.environ.get("BODY_Z_MARGIN_FRAC", "0.05"))
SEED = int(os.environ.get("SEED", "20260704"))
DEFAULT_WINDOWS = [
    (-160.0, 240.0),
    (-200.0, 100.0),
    (-200.0, 400.0),
]
BAD_GROUPS = {"G0122", "G0137", "G0369"}


def parse_windows(text):
    if not text:
        return DEFAULT_WINDOWS
    out = []
    for part in text.split(";"):
        low, high = part.split(",")
        out.append((float(low), float(high)))
    return out


WINDOWS = parse_windows(os.environ.get("WINDOWS", ""))


def window_channel(x, low, high):
    x = np.clip(x, low, high)
    return (x - low) / (high - low)


def jitter_window(low, high, rng, center_delta=20.0, width_scale=0.10):
    center = 0.5 * (low + high) + rng.uniform(-center_delta, center_delta)
    width = (high - low) * rng.uniform(1 - width_scale, 1 + width_scale)
    return center - width / 2, center + width / 2


def uniform_indices(z):
    return np.linspace(0, z - 1, NUM_SLICES).round().astype(int)


def center_fraction_indices(z, frac):
    pad = (1.0 - frac) / 2.0
    start = int(round((z - 1) * pad))
    end = int(round((z - 1) * (1.0 - pad)))
    return np.linspace(start, end, NUM_SLICES).round().astype(int)


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
    return np.linspace(start, end, NUM_SLICES).round().astype(int)


def base_indices(vol):
    if SLICE_MODE == "uniform":
        return uniform_indices(vol.shape[2])
    if SLICE_MODE == "center80":
        return center_fraction_indices(vol.shape[2], 0.80)
    if SLICE_MODE == "center60":
        return center_fraction_indices(vol.shape[2], 0.60)
    if SLICE_MODE == "body_z":
        return body_z_indices(vol)
    raise ValueError(f"unknown SLICE_MODE: {SLICE_MODE}")


def jitter_indices(z, rng):
    edges = np.linspace(0, z, NUM_SLICES + 1)
    idx = []
    for a, b in zip(edges[:-1], edges[1:]):
        lo = int(np.floor(a))
        hi = max(lo + 1, int(np.ceil(b)))
        idx.append(rng.integers(lo, min(hi, z)))
    return np.asarray(idx, dtype=np.int64)


def apply_mild_affine(x, rng):
    angle = math.radians(rng.uniform(-5.0, 5.0))
    scale = rng.uniform(0.97, 1.03)
    tx = rng.uniform(-0.03, 0.03)
    ty = rng.uniform(-0.03, 0.03)
    c, s = math.cos(angle) * scale, math.sin(angle) * scale
    theta = torch.tensor([[c, -s, tx], [s, c, ty]], dtype=torch.float32).unsqueeze(0).repeat(x.shape[0], 1, 1)
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)


def view_config(view_id, rng):
    if view_id == 0:
        return f"{SLICE_MODE}_fixed", "base", WINDOWS, False, False
    windows = [jitter_window(low, high, rng) for low, high in WINDOWS]
    if view_id == 1:
        return "z_jitter_window_jitter", "jitter", windows, False, False
    if view_id == 2:
        return "z_jitter_affine", "jitter", WINDOWS, True, False
    if view_id == 3:
        return "z_jitter_window_jitter_noise", "jitter", windows, False, True
    return "z_jitter_window_jitter_seeded", "jitter", windows, False, False


def view_metadata(case_id, raw_img, img, vol, view_id, case_seed):
    rng = np.random.default_rng(case_seed + view_id)
    mode, index_mode, windows, do_affine, do_noise = view_config(view_id, rng)
    idx = jitter_indices(vol.shape[2], rng) if index_mode == "jitter" else base_indices(vol)
    audit = {
        "case_id": case_id,
        "view_id": view_id,
        "view_mode": mode,
        "shape": "x".join(map(str, vol.shape)),
        "spacing": "x".join(f"{v:.6g}" for v in img.header.get_zooms()[:3]),
        "orientation_original": "".join(nib.aff2axcodes(raw_img.affine)),
        "orientation_canonical": "".join(nib.aff2axcodes(img.affine)),
        "selected_slice_indices": ";".join(map(str, idx.tolist())),
        "slice_mode": SLICE_MODE,
        "windows": ";".join(f"{low:.2f},{high:.2f}" for low, high in windows),
        "affine": int(do_affine),
        "noise": int(do_noise),
        "crop": "whole",
    }
    return rng, idx, windows, do_affine, do_noise, audit


def make_tensor_from_volume(vol, idx, windows, do_affine, do_noise, rng, torch_seed):
    slices = vol[:, :, idx].transpose(2, 0, 1)
    channels = [window_channel(slices, low, high) for low, high in windows]
    x = torch.from_numpy(np.stack(channels, axis=1).astype(np.float32))
    x = F.interpolate(x, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False)
    if do_affine:
        x = apply_mild_affine(x, rng)
    if do_noise:
        gen = torch.Generator()
        gen.manual_seed(torch_seed)
        x = x + torch.randn(x.shape, generator=gen) * 0.01
    return (x.clamp(0, 1).mul(255).round()).to(torch.uint8)


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


def process_case(i, total, r):
    torch.set_num_threads(1)
    case_id = r["group"]
    case_seed = SEED + i * 1000
    raw_img = nib.load(str(RAW_ROOT / r["image"]))
    img = nib.as_closest_canonical(raw_img)
    vol = np.asarray(img.get_fdata(dtype=np.float32))
    checksum_rows, audit_rows = [], []
    generated, skipped = 0, 0
    for view_id in range(NUM_VIEWS):
        rng, idx, windows, do_affine, do_noise, audit = view_metadata(case_id, raw_img, img, vol, view_id, case_seed)
        rel = f"tensors/{case_id}_view{view_id}.pt"
        out_path = OUT_ROOT / rel
        if out_path.exists():
            skipped += 1
        else:
            tensor = make_tensor_from_volume(vol, idx, windows, do_affine, do_noise, rng, case_seed + view_id)
            torch.save(tensor, out_path)
            generated += 1
        audit_rows.append(audit)
        checksum_rows.append(
            {
                "case_id": case_id,
                "view_id": view_id,
                "tensor": rel,
                "shape": f"{NUM_SLICES},3,{IMAGE_SIZE},{IMAGE_SIZE}",
                "dtype": "uint8",
                "bytes": out_path.stat().st_size,
                "sha256": sha256(out_path),
            }
        )
    return f"{i:03d}/{total} {case_id} generated={generated} skipped={skipped}", checksum_rows, audit_rows


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    TENSOR_DIR.mkdir(parents=True, exist_ok=True)
    AUDIT_ROOT.mkdir(parents=True, exist_ok=True)
    rows = [r for r in read_rows(RAW_ROOT / "all.csv") if r["group"] not in BAD_GROUPS]
    checksum_rows, audit_rows = [], []
    if NUM_WORKERS <= 1:
        results = [process_case(i, len(rows), r) for i, r in enumerate(rows, 1)]
    else:
        with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
            futures = [executor.submit(process_case, i, len(rows), r) for i, r in enumerate(rows, 1)]
            results = []
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                print(result[0], flush=True)

    if NUM_WORKERS <= 1:
        for result in results:
            print(result[0], flush=True)

    for _, case_checksum_rows, case_audit_rows in results:
        checksum_rows.extend(case_checksum_rows)
        audit_rows.extend(case_audit_rows)

    checksum_rows.sort(key=lambda x: (x["case_id"], int(x["view_id"])))
    audit_rows.sort(key=lambda x: (x["case_id"], int(x["view_id"])))

    write_rows(OUT_ROOT / "tensors_sha256.csv", checksum_rows)
    write_rows(AUDIT_ROOT / f"header_audit_{CACHE_NAME}.csv", audit_rows)
    summary = {
        "name": CACHE_NAME,
        "source_dataset": "data_private/standard",
        "num_cases": len(rows),
        "num_views": NUM_VIEWS,
        "tensor_shape": [NUM_SLICES, 3, IMAGE_SIZE, IMAGE_SIZE],
        "tensor_dtype": "uint8",
        "slice_mode": SLICE_MODE,
        "windows": WINDOWS,
        "view0": f"{SLICE_MODE} {NUM_SLICES} slices with fixed windows",
        "augmentation": "z-jitter, window jitter, mild affine, mild Gaussian noise",
    }
    (OUT_ROOT / "dataset_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_ROOT / "README.md").write_text(
        f"# {CACHE_NAME}\n\n"
        "Multiview CT tensor cache. Each case-view tensor has shape "
        f"`{NUM_SLICES} x 3 x {IMAGE_SIZE} x {IMAGE_SIZE}` and dtype `uint8`.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
