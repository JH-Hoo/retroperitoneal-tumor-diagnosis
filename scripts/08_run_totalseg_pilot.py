#!/usr/bin/env python3
import csv
import os
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PILOT_CSV = PROJECT_ROOT / "data" / "annotations" / "totalseg_pilot_30.csv"
RUN_LOG = PROJECT_ROOT / "data" / "segmentations" / "totalseg_pilot_run_log.csv"

TOTALSEG_CMD = os.environ.get("TOTALSEG_CMD", str(Path(sys.executable).with_name("TotalSegmentator")))
DEVICE = os.environ.get("TOTALSEG_DEVICE", "gpu")
MAX_CASES = int(os.environ.get("MAX_CASES", "0"))
RERUN = os.environ.get("RERUN", "0") == "1"
os.environ.setdefault("TOTALSEG_HOME_DIR", str(PROJECT_ROOT / "data_private" / "totalsegmentator_home"))

ROI_SUBSET = [
    "kidney_right",
    "kidney_left",
    "adrenal_gland_right",
    "adrenal_gland_left",
    "sacrum",
    "vertebrae_S1",
    "vertebrae_L5",
    "vertebrae_L4",
    "vertebrae_L3",
    "vertebrae_L2",
    "vertebrae_L1",
    "aorta",
    "inferior_vena_cava",
    "iliac_artery_left",
    "iliac_artery_right",
    "iliac_vena_left",
    "iliac_vena_right",
    "iliopsoas_left",
    "iliopsoas_right",
]


def read_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def run_case(row):
    image = PROJECT_ROOT / row["nifti"]
    out_dir = PROJECT_ROOT / row["totalseg_dir"]
    done_mask = out_dir / "aorta.nii.gz"
    if done_mask.exists() and not RERUN:
        return "skipped_existing", 0.0, ""

    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        TOTALSEG_CMD,
        "-i",
        str(image),
        "-o",
        str(out_dir),
        "-rs",
        *ROI_SUBSET,
        "-f",
        "-d",
        DEVICE,
        "-ns",
        "1",
    ]
    t0 = time.time()
    result = subprocess.run(cmd, text=True, capture_output=True)
    seconds = time.time() - t0
    status = "ok" if result.returncode == 0 else "failed"
    message = (result.stderr or result.stdout).strip().splitlines()[-1:] or [""]
    return status, seconds, message[0]


def main():
    rows = read_rows(PILOT_CSV)
    if MAX_CASES:
        rows = rows[:MAX_CASES]

    log_rows = []
    for i, row in enumerate(rows, 1):
        status, seconds, message = run_case(row)
        log = {
            "case_id": row["case_id"],
            "label_5": row["label_5"],
            "status": status,
            "seconds": f"{seconds:.1f}",
            "message": message,
            "totalseg_dir": row["totalseg_dir"],
        }
        log_rows.append(log)
        print(f"{i}/{len(rows)} {row['case_id']} {status} {seconds:.1f}s", flush=True)

    write_rows(RUN_LOG, log_rows)
    print(RUN_LOG)


if __name__ == "__main__":
    main()
