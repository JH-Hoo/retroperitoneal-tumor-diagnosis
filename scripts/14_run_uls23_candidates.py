#!/usr/bin/env python3
import csv
import json
import os
import shlex
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VOI_STATUS_CSV = PROJECT_ROOT / "data" / "derived" / "uls23_vois" / "uls23_voi_status.csv"
OUT_DIR = PROJECT_ROOT / "data" / "segmentations" / "uls23_candidates"
RUN_STATUS_CSV = OUT_DIR / "uls23_candidate_status.csv"

ULS23_CMD_TEMPLATE = os.environ.get("ULS23_CMD_TEMPLATE", "")


def read_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def run_candidate(row):
    case_id = row["case_id"]
    if row["status"] != "ready":
        return {
            "case_id": case_id,
            "candidate_id": "001",
            "status": "missing_voi",
            "voi_path": row.get("voi_path", ""),
            "mask_path": "",
            "meta_path": "",
            "message": f"VOI status is {row['status']}",
        }

    if not ULS23_CMD_TEMPLATE:
        return {
            "case_id": case_id,
            "candidate_id": "001",
            "status": "missing_uls23_command",
            "voi_path": row["voi_path"],
            "mask_path": "",
            "meta_path": "",
            "message": "set ULS23_CMD_TEMPLATE before running candidate segmentation",
        }

    case_dir = OUT_DIR / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    mask_path = case_dir / "candidate_001_mask.nii.gz"
    meta_path = case_dir / "candidate_001_meta.json"
    cmd = ULS23_CMD_TEMPLATE.format(
        input=str(PROJECT_ROOT / row["voi_path"]),
        output=str(mask_path),
        case_id=case_id,
    )
    result = subprocess.run(shlex.split(cmd), text=True, capture_output=True)
    status = "ok" if result.returncode == 0 and mask_path.exists() else "failed"
    message = (result.stderr or result.stdout).strip().splitlines()[-1:] or [""]
    meta = {
        "case_id": case_id,
        "candidate_id": "001",
        "voi_path": row["voi_path"],
        "mask_path": str(mask_path.relative_to(PROJECT_ROOT)) if mask_path.exists() else "",
        "command_template": ULS23_CMD_TEMPLATE,
        "returncode": result.returncode,
        "message": message[0],
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "case_id": case_id,
        "candidate_id": "001",
        "status": status,
        "voi_path": row["voi_path"],
        "mask_path": str(mask_path.relative_to(PROJECT_ROOT)) if mask_path.exists() else "",
        "meta_path": str(meta_path.relative_to(PROJECT_ROOT)),
        "message": message[0],
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = [run_candidate(row) for row in read_rows(VOI_STATUS_CSV)]
    write_rows(RUN_STATUS_CSV, rows)
    for row in rows:
        print(row["case_id"], row["status"], row["message"], flush=True)
    print(RUN_STATUS_CSV)


if __name__ == "__main__":
    main()
