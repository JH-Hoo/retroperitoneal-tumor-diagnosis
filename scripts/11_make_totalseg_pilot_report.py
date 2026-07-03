#!/usr/bin/env python3
import csv
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PILOT_CSV = PROJECT_ROOT / "data" / "annotations" / "totalseg_pilot_30.csv"
RUN_LOG = PROJECT_ROOT / "data" / "segmentations" / "totalseg_pilot_run_log.csv"
ROI_SUMMARY = PROJECT_ROOT / "data" / "derived" / "retroperitoneal_roi" / "summary.csv"
QC_CSV = PROJECT_ROOT / "data" / "qc" / "totalseg_contact_sheets.csv"
REPORT_PATH = PROJECT_ROOT / "reports" / "totalseg_uls23_pilot_report.md"


def read_rows(path):
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def table(rows, fields):
    lines = ["|" + "|".join(fields) + "|", "|" + "|".join(["---"] * len(fields)) + "|"]
    for row in rows:
        lines.append("|" + "|".join(str(row.get(f, "")) for f in fields) + "|")
    return "\n".join(lines)


def main():
    pilot = read_rows(PILOT_CSV)
    run_log = read_rows(RUN_LOG)
    roi_rows = read_rows(ROI_SUMMARY)
    qc_rows = read_rows(QC_CSV)
    class_counts = Counter(r["label_5"] for r in pilot)
    status_counts = Counter(r["status"] for r in run_log)
    ok_seconds = [float(r["seconds"]) for r in run_log if r["status"] == "ok" and r["seconds"]]

    roi_payload = []
    roi_by_class = []
    if roi_rows:
        fractions = [float(r["volume_fraction"]) for r in roi_rows]
        roi_payload = [
            {"metric": "ROI cases", "value": len(roi_rows)},
            {"metric": "Mean ROI volume fraction", "value": f"{sum(fractions) / len(fractions):.3f}"},
            {"metric": "Min ROI volume fraction", "value": f"{min(fractions):.3f}"},
            {"metric": "Max ROI volume fraction", "value": f"{max(fractions):.3f}"},
            {"metric": "Fallback full-volume cases", "value": sum(r["fallback_full_volume"] == "True" for r in roi_rows)},
        ]
        by_class = defaultdict(list)
        for row in roi_rows:
            by_class[row["label_5"]].append(float(row["volume_fraction"]))
        for label, values in by_class.items():
            roi_by_class.append(
                {
                    "class": label,
                    "cases": len(values),
                    "mean_fraction": f"{sum(values) / len(values):.3f}",
                    "min_fraction": f"{min(values):.3f}",
                    "max_fraction": f"{max(values):.3f}",
                }
            )

    examples = "\n".join(f"![{r['case_id']}](../{r['qc_sheet']})" for r in qc_rows[:3])
    status_table = table([{"status": k, "cases": v} for k, v in status_counts.items()], ["status", "cases"]) if run_log else "TotalSegmentator has not been run yet on this branch."
    if ok_seconds:
        status_table += "\n\n" + table(
            [
                {"metric": "Mean successful runtime seconds", "value": f"{sum(ok_seconds) / len(ok_seconds):.1f}"},
                {"metric": "Min successful runtime seconds", "value": f"{min(ok_seconds):.1f}"},
                {"metric": "Max successful runtime seconds", "value": f"{max(ok_seconds):.1f}"},
            ],
            ["metric", "value"],
        )
    roi_table = table(roi_payload, ["metric", "value"]) if roi_payload else "ROI JSON files have not been generated yet."

    report = f"""# TotalSegmentator + ULS23 Pilot Branch

## Summary

This branch tests the first practical step of the proposed anatomy-prior pipeline: use TotalSegmentator to segment retroperitoneal anchor anatomy, then convert those structures into a coarse retroperitoneal ROI. It does not claim tumor segmentation.

ULS23-style lesion segmentation is intentionally not run yet, because it needs a lesion-centered click or VOI. The placeholder input format is `data/annotations/tumor_clicks_template.csv`.

## Pilot Cohort

| Item | Value |
|---|---:|
| Cases | {len(pilot)} |
| Sampling | 6 cases per 5-class label |
| Source labels | `data/labels/labels_5class.csv` |
| Raw images | `data_private/standard/images/*.nii.gz` |

{table([{'class': k, 'cases': v} for k, v in class_counts.items()], ['class', 'cases'])}

## Method

The TotalSegmentator ROI subset contains kidneys, adrenal glands, lumbar/sacral vertebrae, sacrum, aorta, IVC, iliac vessels, and iliopsoas muscles. The x/y range uses all available anchor masks with a 70 mm margin. The z range uses kidneys, adrenal glands, lumbar/sacral vertebrae, and sacrum with a 50 mm margin, because long vessels can otherwise make the z crop nearly full-volume.

Outputs:

| Output | Path |
|---|---|
| Pilot manifest | `data/annotations/totalseg_pilot_30.csv` |
| TotalSegmentator masks | `data/segmentations/totalseg_pilot/` |
| TotalSegmentator weights | `data_private/totalsegmentator_home/` |
| ROI bbox JSON | `data/derived/retroperitoneal_roi/*.json` |
| ROI summary | `data/derived/retroperitoneal_roi/summary.csv` |
| QC contact sheets | `data/qc/contact_sheets/*.png` |

## Run Status

{status_table}

## ROI Summary

{roi_table}

{table(roi_by_class, ['class', 'cases', 'mean_fraction', 'min_fraction', 'max_fraction']) if roi_by_class else ''}

## QC Examples

{examples if examples else 'QC sheets have not been generated yet.'}

## Interpretation

This is an anatomy-prior experiment. It can answer whether a TotalSegmentator-derived retroperitoneal crop is technically usable and visually sane. It cannot answer whether the tumor itself is segmented, because the target lesion is not part of TotalSegmentator's anatomy labels.

If the contact sheets show that the red ROI consistently covers the retroperitoneal tumor-bearing region, the next step is to build a 96-slice cache from this ROI and compare it against whole/body crops. If not, the margin or anchor set should be enlarged before trying ULS23.

## References

- TotalSegmentator official repository: https://github.com/wasserth/TotalSegmentator
- ULS23 challenge repository: https://github.com/DIAGNijmegen/ULS23
"""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(REPORT_PATH)


if __name__ == "__main__":
    main()
