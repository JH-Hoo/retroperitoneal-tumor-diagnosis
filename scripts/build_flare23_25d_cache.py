#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import math
import shutil
from collections import defaultdict
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LABELS = PROJECT_ROOT / "data" / "labels" / "champion_minvox5000.csv"
DEFAULT_IMAGE_ROOT = PROJECT_ROOT / "dataset_standard_v0"
DEFAULT_MASK_DIR = Path("/root/autodl-tmp/flare23_champion_outputs")
DEFAULT_OUT_ROOT = PROJECT_ROOT / "data" / "champion_flare23_25d_cache_15x224_minvox5000"

CLASS_NAMES = ["肉瘤类", "良性神经源性肿瘤", "PPGL", "淋巴瘤", "胃肠道间质瘤"]
CHANNELS = ["ct_soft", "ct_fat", "tumor_mask", "tumor_shell_2d", "organ_union"]
WINDOWS = {"ct_soft": [-150.0, 250.0], "ct_fat": [-250.0, 150.0]}
REUSE_META_COLUMNS = [
    "cache_status",
    "sample_status",
    "crop_status",
    "spacing_x_mm",
    "spacing_y_mm",
    "spacing_z_mm",
    "crop_x",
    "crop_y",
    "source_z",
    "selected_z_indices",
    "selected_z_norm",
    "z_hist",
    "tumor_voxels",
    "no_tumor_label14",
    "z_peak_norm",
    "z_centroid_norm",
    "z_std_norm",
    "z_q10_norm",
    "z_q25_norm",
    "z_q50_norm",
    "z_q75_norm",
    "z_q90_norm",
    "tumor_z_slices",
    "tumor_z_extent_norm",
    "tumor_area_max_frac",
    "tumor_area_entropy",
]


def read_rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields, seen = [], set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def effective_split(row):
    return row.get("split_5class") or row.get("split") or ""


def parse_quantiles(text):
    vals = [float(x) for x in text.replace(";", ",").split(",") if x.strip()]
    if not vals:
        raise argparse.ArgumentTypeError("at least one quantile is required")
    if min(vals) < 0 or max(vals) > 1:
        raise argparse.ArgumentTypeError("quantiles must be in [0, 1]")
    return tuple(vals)


def window_channel(x, low, high):
    x = np.clip(x, low, high)
    return (x - low) / (high - low)


def largest_component(mask):
    if not mask.any():
        return mask
    labeled, n = ndimage.label(mask)
    if n <= 1:
        return mask
    counts = np.bincount(labeled.ravel())
    counts[0] = 0
    return labeled == counts.argmax()


def crop_xy_slices(mask, spacing_xy, margin_mm, shape_xy):
    coords = np.argwhere(mask)
    if coords.size == 0:
        return slice(0, shape_xy[0]), slice(0, shape_xy[1])
    spacing_xy = np.asarray(spacing_xy, dtype=np.float32)
    margin = np.ceil(float(margin_mm) / spacing_xy).astype(int) + 1
    lo = np.maximum(coords.min(axis=0) - margin, 0)
    hi = np.minimum(coords.max(axis=0) + margin + 1, np.asarray(shape_xy))
    return slice(int(lo[0]), int(hi[0])), slice(int(lo[1]), int(hi[1]))


def center_crop_xy(shape_xy):
    lo = [int(s * 0.1) for s in shape_xy]
    hi = [max(lo[i] + 1, int(shape_xy[i] * 0.9)) for i in range(2)]
    return slice(lo[0], hi[0]), slice(lo[1], hi[1])


def resize_2d(arr_hw, image_size, mode):
    t = torch.from_numpy(arr_hw.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    kwargs = {"size": (image_size, image_size), "mode": mode}
    if mode == "bilinear":
        kwargs["align_corners"] = False
    out = F.interpolate(t, **kwargs)
    return out.squeeze(0).squeeze(0).numpy()


def tumor_z_distribution(tumor_mask):
    area = tumor_mask.sum(axis=(0, 1)).astype(np.float64)
    if area.sum() <= 0:
        return area, np.zeros_like(area)
    return area, area / area.sum()


def weighted_quantile_indices(prob_z, quantiles):
    if prob_z.sum() <= 0:
        return []
    cdf = np.cumsum(prob_z)
    out = []
    for q in quantiles:
        out.append(int(np.searchsorted(cdf, q, side="left")))
    return out


def unique_sorted_indices(indices, z_size):
    return sorted({int(np.clip(i, 0, z_size - 1)) for i in indices})


def pad_or_trim_indices(indices, z_size, num_slices, peak_idx=None):
    indices = unique_sorted_indices(indices, z_size)
    if len(indices) < num_slices:
        center = int(peak_idx if peak_idx is not None else (indices[len(indices) // 2] if indices else z_size // 2))
        radius = 1
        while len(indices) < num_slices and radius <= z_size:
            indices.extend([center - radius, center + radius])
            indices = unique_sorted_indices(indices, z_size)
            radius += 1
    if len(indices) < num_slices:
        indices.extend(np.linspace(0, z_size - 1, num_slices).round().astype(int).tolist())
        indices = unique_sorted_indices(indices, z_size)
    if len(indices) > num_slices:
        keep_pos = np.linspace(0, len(indices) - 1, num_slices).round().astype(int)
        indices = [indices[i] for i in keep_pos]
    return indices


def select_tumor_guided_slices(tumor_mask, organ_mask, num_slices, quantiles):
    z_size = tumor_mask.shape[2]
    area_z, prob_z = tumor_z_distribution(tumor_mask)
    if prob_z.sum() > 0:
        peak = int(area_z.argmax())
        indices = weighted_quantile_indices(prob_z, quantiles)
        indices.extend([peak - 3, peak - 2, peak - 1, peak, peak + 1, peak + 2, peak + 3])
        return pad_or_trim_indices(indices, z_size, num_slices, peak), "tumor_z_distribution", area_z, prob_z

    organ_area_z = organ_mask.sum(axis=(0, 1)).astype(np.float64)
    if organ_area_z.sum() > 0:
        peak = int(organ_area_z.argmax())
        prob = organ_area_z / organ_area_z.sum()
        indices = weighted_quantile_indices(prob, quantiles)
        indices.extend([peak - 3, peak - 2, peak - 1, peak, peak + 1, peak + 2, peak + 3])
        return pad_or_trim_indices(indices, z_size, num_slices, peak), "no_tumor_label14_organ_z_distribution", area_z, prob_z

    indices = np.linspace(0, z_size - 1, num_slices).round().astype(int).tolist()
    return pad_or_trim_indices(indices, z_size, num_slices), "no_tumor_label14_even_z", area_z, prob_z


def hist_on_normalized_z(area_z, bins):
    if area_z.sum() <= 0:
        return np.zeros(bins, dtype=np.float32)
    z_size = len(area_z)
    bin_idx = np.floor((np.arange(z_size) + 0.5) / z_size * bins).astype(int)
    bin_idx = np.clip(bin_idx, 0, bins - 1)
    hist = np.zeros(bins, dtype=np.float64)
    np.add.at(hist, bin_idx, area_z)
    hist = hist / hist.sum()
    return hist.astype(np.float32)


def z_summary(area_z):
    z_size = len(area_z)
    if area_z.sum() <= 0:
        return {
            "z_peak_norm": "",
            "z_centroid_norm": "",
            "z_std_norm": "",
            "z_q10_norm": "",
            "z_q25_norm": "",
            "z_q50_norm": "",
            "z_q75_norm": "",
            "z_q90_norm": "",
            "tumor_z_slices": 0,
            "tumor_z_extent_norm": "",
            "tumor_area_max_frac": "",
            "tumor_area_entropy": "",
        }
    z = np.arange(z_size, dtype=np.float64)
    prob = area_z / area_z.sum()
    denom = max(z_size - 1, 1)
    cdf = np.cumsum(prob)
    nz = np.flatnonzero(area_z > 0)
    entropy = -float(np.sum(prob[prob > 0] * np.log(prob[prob > 0]))) / math.log(max(len(nz), 2))
    return {
        "z_peak_norm": float(area_z.argmax() / denom),
        "z_centroid_norm": float((z * prob).sum() / denom),
        "z_std_norm": float(np.sqrt(((z - (z * prob).sum()) ** 2 * prob).sum()) / denom),
        "z_q10_norm": float(np.searchsorted(cdf, 0.10, side="left") / denom),
        "z_q25_norm": float(np.searchsorted(cdf, 0.25, side="left") / denom),
        "z_q50_norm": float(np.searchsorted(cdf, 0.50, side="left") / denom),
        "z_q75_norm": float(np.searchsorted(cdf, 0.75, side="left") / denom),
        "z_q90_norm": float(np.searchsorted(cdf, 0.90, side="left") / denom),
        "tumor_z_slices": int(len(nz)),
        "tumor_z_extent_norm": float((nz.max() - nz.min() + 1) / max(z_size, 1)),
        "tumor_area_max_frac": float(area_z.max() / area_z.sum()),
        "tumor_area_entropy": entropy,
    }


def make_shell_2d(tumor_slice, spacing_xy, shell_mm):
    if not tumor_slice.any():
        return np.zeros_like(tumor_slice, dtype=bool)
    dist = ndimage.distance_transform_edt(~tumor_slice, sampling=spacing_xy)
    return (dist > 0) & (dist <= float(shell_mm))


def make_case_tensor(image_path, mask_path, image_size, num_slices, margin_mm, shell_mm, quantiles, z_hist_bins):
    img_nii = nib.load(str(image_path))
    seg_nii = nib.load(str(mask_path))
    if img_nii.shape != seg_nii.shape:
        raise ValueError(f"shape_mismatch image={img_nii.shape} mask={seg_nii.shape}")

    spacing = np.asarray(img_nii.header.get_zooms()[:3], dtype=np.float32)
    image = np.asarray(img_nii.dataobj, dtype=np.float32)
    seg = np.asarray(seg_nii.dataobj, dtype=np.uint8)

    tumor_all = seg == 14
    tumor_main = largest_component(tumor_all)
    organ_union_all = (seg >= 1) & (seg <= 13)

    indices, sample_status, area_z, prob_z = select_tumor_guided_slices(
        tumor_all,
        organ_union_all,
        num_slices,
        quantiles,
    )
    if tumor_main.any():
        xy_crop = crop_xy_slices(tumor_main.any(axis=2), spacing[:2], margin_mm, image.shape[:2])
        crop_status = "tumor_xy_bbox"
    elif organ_union_all.any():
        xy_crop = crop_xy_slices(organ_union_all.any(axis=2), spacing[:2], margin_mm, image.shape[:2])
        crop_status = "no_tumor_label14_organ_xy_bbox"
    else:
        xy_crop = center_crop_xy(image.shape[:2])
        crop_status = "no_tumor_label14_center_xy_crop"

    slices = []
    for z in indices:
        img_slice = image[xy_crop[0], xy_crop[1], z]
        seg_slice = seg[xy_crop[0], xy_crop[1], z]
        tumor_slice = seg_slice == 14
        shell_slice = make_shell_2d(tumor_slice, spacing[:2], shell_mm)
        organ_slice = (seg_slice >= 1) & (seg_slice <= 13)
        ct_soft = resize_2d(window_channel(img_slice, *WINDOWS["ct_soft"]), image_size, "bilinear")
        ct_fat = resize_2d(window_channel(img_slice, *WINDOWS["ct_fat"]), image_size, "bilinear")
        masks = [
            resize_2d(tumor_slice.astype(np.float32), image_size, "nearest"),
            resize_2d(shell_slice.astype(np.float32), image_size, "nearest"),
            resize_2d(organ_slice.astype(np.float32), image_size, "nearest"),
        ]
        slices.append(np.stack([ct_soft, ct_fat, *masks], axis=0))

    tensor = torch.from_numpy(np.stack(slices, axis=0).clip(0, 1)).mul(255).round().to(torch.uint8)
    hist = hist_on_normalized_z(area_z, z_hist_bins)
    summary = z_summary(area_z)
    selected_z_norm = [float(z / max(image.shape[2] - 1, 1)) for z in indices]
    meta = {
        "cache_status": "ok",
        "sample_status": sample_status,
        "crop_status": crop_status,
        "spacing_x_mm": float(spacing[0]),
        "spacing_y_mm": float(spacing[1]),
        "spacing_z_mm": float(spacing[2]),
        "crop_x": xy_crop[0].stop - xy_crop[0].start,
        "crop_y": xy_crop[1].stop - xy_crop[1].start,
        "source_z": image.shape[2],
        "selected_z_indices": ";".join(map(str, indices)),
        "selected_z_norm": ";".join(f"{x:.6f}" for x in selected_z_norm),
        "z_hist": ";".join(f"{float(x):.8f}" for x in hist),
        "tumor_voxels": int(tumor_all.sum()),
        "no_tumor_label14": int(not tumor_all.any()),
        **summary,
    }
    return tensor, meta


def select_rows(rows, image_root, mask_dir, max_cases, max_per_class, include_unlabeled):
    selected, per_class = [], defaultdict(int)
    for row in rows:
        split = effective_split(row)
        if split not in {"train", "val", "test"} and not include_unlabeled:
            continue
        if row.get("label_5_id", "") == "" and not include_unlabeled:
            continue
        if not (image_root / row["image"]).exists():
            continue
        if not (mask_dir / f"{row['group']}.nii.gz").exists():
            continue
        label = row.get("label_5", "unlabeled")
        if max_per_class and per_class[label] >= max_per_class:
            continue
        selected.append(row)
        per_class[label] += 1
        if max_cases and len(selected) >= max_cases:
            break
    return selected


def load_reuse_rows(cache_roots):
    out = {}
    for root in cache_roots or []:
        root = Path(root)
        csv_path = root / "all.csv"
        if not csv_path.exists():
            continue
        for row in read_rows(csv_path):
            group = row.get("group", "")
            tensor = row.get("tensor", "")
            tensor_path = root / tensor
            if group and tensor and tensor_path.exists() and row.get("cache_status") == "ok":
                out[group] = {"root": root, "row": row, "tensor_path": tensor_path}
    return out


def main():
    parser = argparse.ArgumentParser(description="Build FLARE23 label14-guided 2.5D slice cache.")
    parser.add_argument("--labels-csv", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--mask-dir", type=Path, default=DEFAULT_MASK_DIR)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-slices", type=int, default=15)
    parser.add_argument("--quantiles", type=parse_quantiles, default=parse_quantiles("0.05,0.10,0.25,0.40,0.50,0.60,0.75,0.90,0.95"))
    parser.add_argument("--z-hist-bins", type=int, default=16)
    parser.add_argument("--margin-mm", type=float, default=30.0)
    parser.add_argument("--shell-mm", type=float, default=10.0)
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--max-per-class", type=int, default=0)
    parser.add_argument("--include-unlabeled", action="store_true")
    parser.add_argument("--reuse-cache-root", type=Path, action="append", default=[])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    tensor_dir = args.out_root / "tensors"
    split_dir = args.out_root / "splits"
    tensor_dir.mkdir(parents=True, exist_ok=True)
    split_dir.mkdir(parents=True, exist_ok=True)

    rows = select_rows(
        read_rows(args.labels_csv),
        args.image_root,
        args.mask_dir,
        args.max_cases,
        args.max_per_class,
        args.include_unlabeled,
    )
    reuse_rows = load_reuse_rows(args.reuse_cache_root)
    out_rows, checksum_rows, status_counts = [], [], defaultdict(int)
    for i, row in enumerate(rows, 1):
        group = row["group"]
        out_path = tensor_dir / f"{group}.pt"
        rr = dict(row)
        rr["source_split"] = row.get("split", "")
        rr["split"] = effective_split(row)
        rr["source_image"] = row["image"]
        rr["source_mask"] = str(args.mask_dir / f"{group}.nii.gz")
        rr["tensor"] = f"tensors/{group}.pt"
        try:
            reuse = None if args.overwrite else reuse_rows.get(group)
            if reuse is not None:
                if not out_path.exists():
                    shutil.copy2(reuse["tensor_path"], out_path)
                tensor = torch.load(out_path, map_location="cpu")
                meta = {k: reuse["row"].get(k, "") for k in REUSE_META_COLUMNS if k in reuse["row"]}
                meta["sample_status"] = f"reused:{reuse['root'].name}:{meta.get('sample_status', '')}"
                meta["crop_status"] = f"reused:{reuse['root'].name}:{meta.get('crop_status', '')}"
            else:
                tensor, meta = make_case_tensor(
                    args.image_root / row["image"],
                    args.mask_dir / f"{group}.nii.gz",
                    args.image_size,
                    args.num_slices,
                    args.margin_mm,
                    args.shell_mm,
                    args.quantiles,
                    args.z_hist_bins,
                )
                torch.save(tensor, out_path)
            rr.update(meta)
            rr["cache_status"] = "ok"
            rr["error"] = ""
            checksum_rows.append(
                {
                    "group": group,
                    "tensor": rr["tensor"],
                    "shape": ",".join(map(str, tuple(tensor.shape))),
                    "dtype": str(tensor.dtype).replace("torch.", ""),
                    "bytes": out_path.stat().st_size,
                    "sha256": sha256(out_path),
                }
            )
        except Exception as exc:
            rr["cache_status"] = "error"
            rr["error"] = repr(exc)
        status_counts[rr["cache_status"]] += 1
        out_rows.append(rr)
        print(f"{i}/{len(rows)} {group} {rr['cache_status']} {rr.get('sample_status', '')}", flush=True)

    ok_rows = [r for r in out_rows if r.get("cache_status") == "ok"]
    write_rows(args.out_root / "all.csv", ok_rows)
    for split in ["train", "val", "test"]:
        write_rows(split_dir / f"{split}.csv", [r for r in ok_rows if r.get("split") == split])
    write_rows(args.out_root / "build_manifest.csv", out_rows)
    write_rows(args.out_root / "tensors_sha256.csv", checksum_rows)
    if (args.labels_csv.parent / "label_mapping.json").exists():
        shutil.copy2(args.labels_csv.parent / "label_mapping.json", args.out_root / "label_mapping.json")

    summary = {
        "name": args.out_root.name,
        "source_labels": str(args.labels_csv),
        "source_images": str(args.image_root),
        "source_masks": str(args.mask_dir),
        "num_rows": len(out_rows),
        "num_ok": len(ok_rows),
        "status_counts": dict(status_counts),
        "tensor_shape": [args.num_slices, len(CHANNELS), args.image_size, args.image_size],
        "tensor_dtype": "uint8",
        "channels": CHANNELS,
        "ct_windows": WINDOWS,
        "slice_sampling": {
            "primary": "per-case FLARE label14 z distribution",
            "quantiles": list(args.quantiles),
            "peak_offsets": [-3, -2, -1, 0, 1, 2, 3],
            "fallback": "organ z distribution, then even z",
        },
        "z_hist_bins": args.z_hist_bins,
            "margin_mm": args.margin_mm,
            "shell_mm": args.shell_mm,
            "reuse_cache_roots": [str(p) for p in args.reuse_cache_root],
            "class_names": CLASS_NAMES,
        }
    (args.out_root / "dataset_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {args.out_root}", flush=True)


if __name__ == "__main__":
    main()
