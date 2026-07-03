# TotalSegmentator + ULS23 Pilot Branch

## Summary

This branch tests the first practical step of the proposed anatomy-prior pipeline: use TotalSegmentator to segment retroperitoneal anchor anatomy, then convert those structures into a coarse retroperitoneal ROI. It does not claim tumor segmentation.

ULS23-style lesion segmentation is intentionally not run yet, because it needs a lesion-centered click or VOI. The placeholder input format is `data/annotations/tumor_clicks_template.csv`.

## Pilot Cohort

| Item | Value |
|---|---:|
| Cases | 30 |
| Sampling | 6 cases per 5-class label |
| Source labels | `data/labels/labels_5class.csv` |
| Raw images | `data_private/standard/images/*.nii.gz` |

|class|cases|
|---|---|
|肉瘤类|6|
|良性神经源性肿瘤|6|
|PPGL|6|
|淋巴瘤|6|
|胃肠道间质瘤|6|

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
| Tumor click working table | `data/annotations/tumor_clicks_pilot_30.csv` |
| Tumor click reference sheets | `data/qc/tumor_click_sheets/*.png` |
| ULS23 VOI status | `data/derived/uls23_vois/uls23_voi_status.csv` |
| ULS23 candidate status | `data/segmentations/uls23_candidates/uls23_candidate_status.csv` |

## Run Status

|status|cases|
|---|---|
|skipped_existing|2|
|ok|28|

|metric|value|
|---|---|
|Mean successful runtime seconds|104.1|
|Min successful runtime seconds|52.5|
|Max successful runtime seconds|131.7|

## ROI Summary

|metric|value|
|---|---|
|ROI cases|30|
|Mean ROI volume fraction|0.597|
|Min ROI volume fraction|0.425|
|Max ROI volume fraction|0.765|
|Fallback full-volume cases|0|

|class|cases|mean_fraction|min_fraction|max_fraction|
|---|---|---|---|---|
|肉瘤类|6|0.614|0.462|0.745|
|良性神经源性肿瘤|6|0.626|0.472|0.728|
|PPGL|6|0.569|0.425|0.688|
|淋巴瘤|6|0.628|0.533|0.765|
|胃肠道间质瘤|6|0.548|0.453|0.651|

## QC Examples

![G0097](../data/qc/contact_sheets/G0097_totalseg_roi.png)
![G0121](../data/qc/contact_sheets/G0121_totalseg_roi.png)
![G0286](../data/qc/contact_sheets/G0286_totalseg_roi.png)

## Interpretation

This is an anatomy-prior experiment. It can answer whether a TotalSegmentator-derived retroperitoneal crop is technically usable and visually sane. It cannot answer whether the tumor itself is segmented, because the target lesion is not part of TotalSegmentator's anatomy labels.

If the contact sheets show that the red ROI consistently covers the retroperitoneal tumor-bearing region, the next step is to build a 96-slice cache from this ROI and compare it against whole/body crops. If not, the margin or anchor set should be enlarged before trying ULS23.

## ULS23 Readiness

The ULS23-style stage is not run yet because it requires lesion-centered input. A working table has been prepared with 30 rows. Fill `x_voxel`, `y_voxel`, and `z_voxel` in original NIfTI voxel coordinates, then the next step is to crop lesion-centered VOIs and run the candidate mask proposer.

Current VOI preparation status:

|status|cases|
|---|---|
|missing_click|30|

Current candidate segmentation status:

|status|cases|
|---|---|
|missing_voi|30|

## References

- TotalSegmentator official repository: https://github.com/wasserth/TotalSegmentator
- ULS23 challenge repository: https://github.com/DIAGNijmegen/ULS23
