# Nephrectomy Multi-level CT Experiment (2026-07-21)

This is the compact, non-patient-level result of the Yang-style method adapted
to the retroperitoneal-tumor cohort.

## Cohort

- 252 CT studies were considered.
- 224 had usable CT plus FLARE23 tumor/kidney segmentation.
- 192 had a known endpoint and usable features: 16 nephrectomy, 176 no
  nephrectomy.
- 25 were excluded for no FLARE23 label-14 tumor and 3 for missing masks.

## Patient-level nested fivefold OOF result

The positive prevalence, and therefore the no-skill average precision, was
0.0833.

| Feature/model group | AUROC | Average precision |
|---|---:|---:|
| size only | 0.450 | 0.079 |
| geometry | 0.580 | 0.129 |
| paper radiomics | 0.534 | 0.103 |
| paper radiomics + voxel | 0.519 | 0.110 |
| task-oriented deep | 0.536 | 0.163 |
| paper-style full fusion | 0.473 | 0.108 |
| adapted full fusion | 0.498 | 0.114 |

The task-oriented deep model had the largest average precision, but its 95%
bootstrap interval was 0.063-0.378 and its one-sided label-permutation p-value
was 0.0717. Its paired average-precision difference versus geometry was 0.034
(95% bootstrap interval -0.127 to 0.229). This is weak, inconclusive signal.

The 35 explicit geometry variables had no BH-FDR-significant univariate
association. The two strongest nominal trends were kidney x-axis bounding-box
width (raw p=0.016, q=0.357) and tumor-surface-distance p90 (raw p=0.020,
q=0.357).

## Conclusion

The published high discrimination did not transfer to this cohort. The current
models are not suitable for clinical decision-making or automatic nephrectomy
recommendations. The main limitations are only 16 usable positive outcomes, no
external validation cohort, segmentation exclusions, and important non-image
drivers of surgical choice.

The experimental branch should not be merged to main as a production model.
Private full artifacts, including OOF predictions, feature matrices, weights,
SHAP values, and the validated workbook, are stored under:

```text
/Volumes/My_Drive/腹膜后肿瘤诊断/reports/nephrectomy_multilevel/outputs/20260721/
```
