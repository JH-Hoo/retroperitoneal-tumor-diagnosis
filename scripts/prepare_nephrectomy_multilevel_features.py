#!/usr/bin/env python3
"""Prepare Yang-style multi-level CT features for nephrectomy prediction.

The paper ROI is adapted from an intrarenal RCC setting to retroperitoneal
tumors.  Each case is resampled, the largest FLARE23 label-14 component is
paired with the nearest kidney, and a fixed 64^3 crop is created.  The script
extracts:

* explicit tumor-kidney geometry;
* PyRadiomics features from tumor, nearest kidney, and their 5-mm interface;
* the paper's per-case 256 PCA and 64 singular-value voxel descriptors;
* a compact NPZ ROI used by the task-oriented 3-D feature extractor.

No nephrectomy label is used to construct any image feature.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy import ndimage
from sklearn.decomposition import PCA


TUMOR_LABEL = 14
KIDNEY_LABELS = {"right": 2, "left": 13}
HU_CLIP = (-250.0, 400.0)
DISTANCE_THRESHOLDS_MM = (1.0, 3.0, 5.0, 10.0)


def parse_triplet(text: str, cast=float):
    values = tuple(cast(x.strip()) for x in text.split(","))
    if len(values) != 3:
        raise argparse.ArgumentTypeError("expected three comma-separated values")
    return values


def largest_component(mask: np.ndarray) -> np.ndarray:
    labels, count = ndimage.label(mask, structure=ndimage.generate_binary_structure(3, 3))
    if count == 0:
        return np.zeros_like(mask, dtype=bool)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    return labels == int(sizes.argmax())


def resample_pair(image_path: Path, mask_path: Path, spacing_xyz):
    image = sitk.ReadImage(str(image_path), sitk.sitkFloat32)
    mask = sitk.ReadImage(str(mask_path), sitk.sitkUInt8)

    target_spacing = tuple(float(x) for x in spacing_xyz)
    old_size = np.asarray(image.GetSize(), dtype=np.int64)
    old_spacing = np.asarray(image.GetSpacing(), dtype=np.float64)
    new_size = np.maximum(1, np.rint(old_size * old_spacing / target_spacing)).astype(int)

    reference = sitk.Image([int(x) for x in new_size], sitk.sitkFloat32)
    reference.SetSpacing(target_spacing)
    reference.SetOrigin(image.GetOrigin())
    reference.SetDirection(image.GetDirection())

    image_r = sitk.Resample(
        image,
        reference,
        sitk.Transform(),
        sitk.sitkBSpline,
        float(HU_CLIP[0]),
        sitk.sitkFloat32,
    )
    mask_r = sitk.Resample(
        mask,
        reference,
        sitk.Transform(),
        sitk.sitkNearestNeighbor,
        0,
        sitk.sitkUInt8,
    )
    return image_r, mask_r


def surface(mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)
    return mask & ~ndimage.binary_erosion(mask, structure=ndimage.generate_binary_structure(3, 1))


def crop_with_padding(array: np.ndarray, center_zyx, crop_size_zyx, fill_value):
    crop_size = np.asarray(crop_size_zyx, dtype=int)
    center = np.asarray(center_zyx, dtype=float)
    start = np.floor(center - crop_size / 2.0).astype(int)
    stop = start + crop_size
    src_start = np.maximum(start, 0)
    src_stop = np.minimum(stop, np.asarray(array.shape))
    dst_start = src_start - start
    dst_stop = dst_start + (src_stop - src_start)

    out = np.full(tuple(crop_size), fill_value, dtype=array.dtype)
    src = tuple(slice(int(a), int(b)) for a, b in zip(src_start, src_stop))
    dst = tuple(slice(int(a), int(b)) for a, b in zip(dst_start, dst_stop))
    out[dst] = array[src]
    return out


def bbox_features(mask: np.ndarray, spacing_zyx, prefix: str):
    coords = np.argwhere(mask)
    if not len(coords):
        return {}
    extent_vox = coords.max(axis=0) - coords.min(axis=0) + 1
    extent_mm = extent_vox * np.asarray(spacing_zyx)
    centroid_mm = coords.mean(axis=0) * np.asarray(spacing_zyx)
    return {
        f"geo_{prefix}_bbox_z_mm": float(extent_mm[0]),
        f"geo_{prefix}_bbox_y_mm": float(extent_mm[1]),
        f"geo_{prefix}_bbox_x_mm": float(extent_mm[2]),
        f"geo_{prefix}_bbox_diagonal_mm": float(np.linalg.norm(extent_mm)),
        f"geo_{prefix}_centroid_z_mm": float(centroid_mm[0]),
        f"geo_{prefix}_centroid_y_mm": float(centroid_mm[1]),
        f"geo_{prefix}_centroid_x_mm": float(centroid_mm[2]),
    }


def geometry_features(tumor: np.ndarray, kidney: np.ndarray, spacing_zyx):
    spacing = np.asarray(spacing_zyx, dtype=np.float64)
    voxel_volume = float(np.prod(spacing))
    tumor_surface = surface(tumor)
    kidney_surface = surface(kidney)
    distance_to_kidney = ndimage.distance_transform_edt(~kidney_surface, sampling=spacing)
    distance_to_tumor = ndimage.distance_transform_edt(~tumor_surface, sampling=spacing)
    surface_distances = distance_to_kidney[tumor_surface]
    min_distance = float(surface_distances.min()) if surface_distances.size else math.nan

    result = {
        "geo_tumor_volume_ml": float(tumor.sum() * voxel_volume / 1000.0),
        "geo_kidney_volume_ml": float(kidney.sum() * voxel_volume / 1000.0),
        "geo_tumor_kidney_volume_ratio": float(tumor.sum() / max(kidney.sum(), 1)),
        "geo_min_surface_distance_mm": min_distance,
        "geo_mean_surface_distance_mm": float(surface_distances.mean()) if surface_distances.size else math.nan,
        "geo_median_surface_distance_mm": float(np.median(surface_distances)) if surface_distances.size else math.nan,
        "geo_p10_surface_distance_mm": float(np.quantile(surface_distances, 0.1)) if surface_distances.size else math.nan,
        "geo_p90_surface_distance_mm": float(np.quantile(surface_distances, 0.9)) if surface_distances.size else math.nan,
        "geo_direct_contact": int(bool(surface_distances.size and min_distance <= 1e-6)),
    }
    tumor_surface_count = max(int(tumor_surface.sum()), 1)
    for threshold in DISTANCE_THRESHOLDS_MM:
        near = surface_distances <= threshold
        overlap = (distance_to_tumor <= threshold) & kidney
        result[f"geo_tumor_surface_within_{threshold:g}mm_fraction"] = float(near.sum() / tumor_surface_count)
        result[f"geo_dilated_tumor_overlap_kidney_{threshold:g}mm_ml"] = float(overlap.sum() * voxel_volume / 1000.0)

    tumor_coords = np.argwhere(tumor)
    kidney_coords = np.argwhere(kidney)
    delta = (tumor_coords.mean(axis=0) - kidney_coords.mean(axis=0)) * spacing
    result.update(
        {
            "geo_centroid_delta_z_mm": float(delta[0]),
            "geo_centroid_delta_y_mm": float(delta[1]),
            "geo_centroid_delta_x_mm": float(delta[2]),
            "geo_centroid_distance_mm": float(np.linalg.norm(delta)),
        }
    )
    result.update(bbox_features(tumor, spacing, "tumor"))
    result.update(bbox_features(kidney, spacing, "kidney"))
    return result, distance_to_tumor, distance_to_kidney


def make_interface_mask(tumor, kidney, distance_to_tumor, distance_to_kidney, width_mm=5.0):
    interface = (distance_to_tumor <= width_mm) & (distance_to_kidney <= width_mm)
    # Preserve the local fat plane as well as the two organ boundaries.
    return largest_component(interface)


def nearest_interface_center(tumor: np.ndarray, kidney: np.ndarray, spacing_zyx, crop_size_zyx):
    """Return a crop-safe center near the closest tumor-kidney interface.

    The unconstrained midpoint is ideal when both structures fit.  When they
    are farther apart than the fixed paper ROI, constrain the offset so the
    closest tumor voxel is always retained; absence of kidney in that crop then
    truthfully represents a very large separation instead of an empty ROI.
    """
    distance, indices = ndimage.distance_transform_edt(
        ~kidney, sampling=spacing_zyx, return_indices=True
    )
    tumor_coords = np.argwhere(tumor)
    nearest_tumor = tumor_coords[int(np.argmin(distance[tumor]))]
    nearest_kidney = indices[(slice(None), *nearest_tumor)].astype(np.float64)
    half_extent = np.asarray(crop_size_zyx, dtype=np.float64) / 2.0 - 2.0
    midpoint_offset = (nearest_kidney - nearest_tumor) / 2.0
    safe_offset = np.clip(midpoint_offset, -half_extent, half_extent)
    return nearest_tumor.astype(np.float64) + safe_offset


def radiomics_extractor():
    from radiomics import featureextractor

    extractor = featureextractor.RadiomicsFeatureExtractor(
        binWidth=25,
        interpolator=sitk.sitkBSpline,
        correctMask=True,
        minimumROISize=16,
    )
    extractor.disableAllImageTypes()
    extractor.enableImageTypeByName("Original")
    extractor.disableAllFeatures()
    for name in ("shape", "firstorder", "glcm", "glrlm", "glszm", "gldm", "ngtdm"):
        extractor.enableFeatureClassByName(name)
    return extractor


def extract_radiomics(image_sitk, masks, prefixes):
    extractor = radiomics_extractor()
    features = {}
    for mask, prefix in zip(masks, prefixes):
        if int(mask.sum()) < 16:
            continue
        mask_img = sitk.GetImageFromArray(mask.astype(np.uint8))
        mask_img.CopyInformation(image_sitk)
        try:
            result = extractor.execute(image_sitk, mask_img, label=1)
        except Exception as exc:
            logging.warning("radiomics %s failed: %s", prefix, exc)
            continue
        for key, value in result.items():
            if not key.startswith("original_"):
                continue
            try:
                features[f"rad_{prefix}_{key.removeprefix('original_')}"] = float(value)
            except (TypeError, ValueError):
                pass
    return features


def paper_voxel_features(masked_ct_roi: np.ndarray):
    matrix = np.asarray(masked_ct_roi, dtype=np.float64).reshape(64, -1)
    pca_values = PCA(n_components=4, svd_solver="full").fit_transform(matrix).reshape(-1)
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    result = {f"pca_{i:03d}": float(v) for i, v in enumerate(pca_values)}
    result.update({f"svd_{i:02d}": float(v) for i, v in enumerate(singular_values)})
    return result


def normalize_ct(ct: np.ndarray, foreground: np.ndarray):
    clipped = np.clip(ct.astype(np.float32), *HU_CLIP)
    values = clipped[foreground]
    mean = float(values.mean()) if values.size else float(clipped.mean())
    std = float(values.std()) if values.size else float(clipped.std())
    std = max(std, 1.0)
    return ((clipped - mean) / std).astype(np.float32), mean, std


def process_case(record, args_dict):
    case_id = record["case_id"]
    image_path = Path(args_dict["image_dir"]) / f"{case_id}.nii.gz"
    mask_path = Path(args_dict["mask_dir"]) / f"{case_id}.nii.gz"
    base = {
        "case_id": case_id,
        "nephrectomy_status": record.get("nephrectomy_status", "unknown"),
        "nephrectomy": record.get("nephrectomy"),
        "patient_group": record.get("patient_group", case_id),
        "label_4": record.get("label_4", ""),
        "pathology_class": record.get("pathology_class", ""),
        "image_acquisition_date": record.get("image_acquisition_date", ""),
        "pathology_received_date": record.get("pathology_received_date", ""),
        "ct_after_pathology": record.get("ct_after_pathology", ""),
        "feature_status": "error",
        "feature_error": "",
    }
    if not image_path.exists():
        return {**base, "feature_status": "missing_ct", "feature_error": str(image_path)}
    if not mask_path.exists():
        return {**base, "feature_status": "missing_mask", "feature_error": str(mask_path)}

    try:
        image_sitk, mask_sitk = resample_pair(image_path, mask_path, args_dict["spacing_xyz"])
        ct = sitk.GetArrayFromImage(image_sitk).astype(np.float32)
        seg = sitk.GetArrayFromImage(mask_sitk).astype(np.uint8)
        spacing_zyx = tuple(reversed(image_sitk.GetSpacing()))

        tumor = largest_component(seg == int(args_dict["tumor_label"]))
        if not tumor.any():
            return {**base, "feature_status": "no_tumor_label14"}

        kidneys = {
            side: largest_component(seg == int(label))
            for side, label in args_dict["kidney_labels"].items()
        }
        kidneys = {side: mask for side, mask in kidneys.items() if mask.any()}
        if not kidneys:
            return {**base, "feature_status": "no_kidney_label"}

        nearest_side = min(
            kidneys,
            key=lambda side: float(
                ndimage.distance_transform_edt(~kidneys[side], sampling=spacing_zyx)[tumor].min()
            ),
        )
        kidney = kidneys[nearest_side]
        geo, distance_to_tumor, distance_to_kidney = geometry_features(tumor, kidney, spacing_zyx)
        interface = make_interface_mask(tumor, kidney, distance_to_tumor, distance_to_kidney)
        union = tumor | kidney
        # The paper used a compact kidney-tumor ROI.  Retroperitoneal tumors can
        # be much larger, so center the fixed crop on the clinically relevant
        # nearest interface and audit how much of each structure is retained.
        crop_size = tuple(int(x) for x in args_dict["crop_size_zyx"])
        center = nearest_interface_center(tumor, kidney, spacing_zyx, crop_size)

        normalized, norm_mean, norm_std = normalize_ct(ct, union)
        local_context = distance_to_tumor <= float(args_dict["context_mm"])
        local_context |= distance_to_kidney <= float(args_dict["context_mm"])
        paper_masked = np.full_like(normalized, float(normalized.min()))
        paper_masked[union] = normalized[union]
        context_masked = np.full_like(normalized, float(normalized.min()))
        context_masked[local_context] = normalized[local_context]

        paper_roi = crop_with_padding(paper_masked, center, crop_size, float(normalized.min()))
        context_roi = crop_with_padding(context_masked, center, crop_size, float(normalized.min()))
        tumor_roi = crop_with_padding(tumor.astype(np.uint8), center, crop_size, 0)
        kidney_roi = crop_with_padding(kidney.astype(np.uint8), center, crop_size, 0)
        interface_roi = crop_with_padding(interface.astype(np.uint8), center, crop_size, 0)

        tumor_retained = float(tumor_roi.sum() / max(tumor.sum(), 1))
        kidney_retained = float(kidney_roi.sum() / max(kidney.sum(), 1))
        interface_retained = float(interface_roi.sum() / max(interface.sum(), 1))

        roi_dir = Path(args_dict["out_dir"]) / "roi"
        roi_dir.mkdir(parents=True, exist_ok=True)
        roi_path = roi_dir / f"{case_id}.npz"
        np.savez_compressed(
            roi_path,
            ct=context_roi.astype(np.float16),
            paper_ct=paper_roi.astype(np.float16),
            tumor=tumor_roi.astype(np.uint8),
            kidney=kidney_roi.astype(np.uint8),
            interface=interface_roi.astype(np.uint8),
        )

        features = {
            **base,
            "feature_status": "ok",
            "feature_error": "",
            "nearest_kidney": nearest_side,
            "roi_path": str(roi_path.relative_to(Path(args_dict["out_dir"]))),
            "spacing_x_mm": float(image_sitk.GetSpacing()[0]),
            "spacing_y_mm": float(image_sitk.GetSpacing()[1]),
            "spacing_z_mm": float(image_sitk.GetSpacing()[2]),
            "norm_mean_hu": norm_mean,
            "norm_std_hu": norm_std,
            "interface_voxels": int(interface.sum()),
            "roi_tumor_retained_fraction": tumor_retained,
            "roi_kidney_retained_fraction": kidney_retained,
            "roi_interface_retained_fraction": interface_retained,
            **geo,
            **paper_voxel_features(paper_roi),
        }
        if args_dict["radiomics"]:
            features.update(
                extract_radiomics(
                    image_sitk,
                    (tumor, kidney, interface),
                    ("tumor", "kidney", "interface"),
                )
            )
        return features
    except Exception as exc:
        logging.exception("case %s failed", case_id)
        return {**base, "feature_status": "error", "feature_error": f"{type(exc).__name__}: {exc}"}


def read_csv(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def prepare_records(labels_json: Path, metadata_csv: Path):
    label_payload = json.loads(labels_json.read_text(encoding="utf-8"))
    metadata = {row["group"]: row for row in read_csv(metadata_csv)}
    records = []
    for item in label_payload["cases"]:
        row = dict(item)
        meta = metadata.get(item["case_id"], {})
        row["image_acquisition_date"] = meta.get("image_acquisition_date", "")
        row["pathology_received_date"] = meta.get("pathology_received_date", "")[:10]
        row["ct_after_pathology"] = ""
        try:
            acquisition = row["image_acquisition_date"]
            pathology = row["pathology_received_date"].replace("-", "")
            if acquisition and pathology:
                row["ct_after_pathology"] = int(acquisition > pathology)
        except Exception:
            pass
        records.append(row)
    return records, label_payload


def write_csv(path: Path, rows):
    keys = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                keys.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-json", type=Path, required=True)
    parser.add_argument("--metadata-csv", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--mask-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--spacing-xyz", type=lambda x: parse_triplet(x, float), default=(1.5, 1.5, 2.0))
    parser.add_argument("--crop-size-zyx", type=lambda x: parse_triplet(x, int), default=(64, 64, 64))
    parser.add_argument("--context-mm", type=float, default=5.0)
    parser.add_argument("--tumor-label", type=int, default=TUMOR_LABEL)
    parser.add_argument("--right-kidney-label", type=int, default=KIDNEY_LABELS["right"])
    parser.add_argument("--left-kidney-label", type=int, default=KIDNEY_LABELS["left"])
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument(
        "--case-ids",
        default="",
        help="optional comma-separated case IDs for targeted recomputation",
    )
    parser.add_argument("--radiomics", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    records, label_payload = prepare_records(args.labels_json, args.metadata_csv)
    if args.case_ids:
        requested = {value.strip() for value in args.case_ids.split(",") if value.strip()}
        records = [record for record in records if record["case_id"] in requested]
    if args.max_cases:
        records = records[: args.max_cases]
    args_dict = {
        "image_dir": str(args.image_dir),
        "mask_dir": str(args.mask_dir),
        "out_dir": str(args.out_dir),
        "spacing_xyz": args.spacing_xyz,
        "crop_size_zyx": args.crop_size_zyx,
        "context_mm": args.context_mm,
        "tumor_label": args.tumor_label,
        "kidney_labels": {"right": args.right_kidney_label, "left": args.left_kidney_label},
        "radiomics": args.radiomics,
    }

    rows = []
    if args.jobs <= 1:
        for index, record in enumerate(records, 1):
            print(f"[{index}/{len(records)}] {record['case_id']}", flush=True)
            rows.append(process_case(record, args_dict))
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            futures = {pool.submit(process_case, record, args_dict): record["case_id"] for record in records}
            for index, future in enumerate(as_completed(futures), 1):
                case_id = futures[future]
                print(f"[{index}/{len(records)}] {case_id}", flush=True)
                rows.append(future.result())
        rows.sort(key=lambda row: row["case_id"])

    write_csv(args.out_dir / "features.csv", rows)
    statuses = {}
    for row in rows:
        statuses[row["feature_status"]] = statuses.get(row["feature_status"], 0) + 1
    known_ok = [row for row in rows if row.get("nephrectomy") in (0, 1) and row["feature_status"] == "ok"]
    summary = {
        "method": "Yang et al. multi-level anatomical features adapted to retroperitoneal tumor nephrectomy",
        "source_label_rule": label_payload.get("rule", ""),
        "processed_cases": len(rows),
        "status_counts": statuses,
        "known_feature_ready": len(known_ok),
        "known_positive_feature_ready": sum(int(row["nephrectomy"]) for row in known_ok),
        "known_negative_feature_ready": sum(1 - int(row["nephrectomy"]) for row in known_ok),
        "spacing_xyz_mm": args.spacing_xyz,
        "crop_size_zyx": args.crop_size_zyx,
        "radiomics_enabled": bool(args.radiomics),
    }
    (args.out_dir / "feature_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
