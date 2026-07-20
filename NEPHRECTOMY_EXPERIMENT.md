# Nephrectomy Prediction Experiment

This branch adapts the multi-level 3-D CT framework of Yang et al. to the
retroperitoneal-tumor cohort. The input is CT plus FLARE23 segmentation; the
binary endpoint is known nephrectomy status. Unknown outcomes are excluded from
the endpoint model.

## Adaptation

- Fixed 64 x 64 x 64 kidney-tumor ROI at 2.0 x 1.5 x 1.5 mm (z/y/x), centered
  on the nearest tumor-kidney interface.
- Paper-style radiomics from tumor and nearest kidney, plus interface radiomics.
- Paper-style per-case 256 PCA and 64 singular-value descriptors.
- Two 64-dimensional task-oriented 3-D encoders. Because this cohort has no
  reliable stage/grade labels, the auxiliary targets are pathology phenotype
  and tumor-kidney proximity. Neither uses the nephrectomy endpoint.
- Explicit tumor-kidney geometry is added for the adapted model.
- Patient-level nested fivefold OOF evaluation. Feature selection, imputation,
  hyperparameter choice, and decision-threshold choice occur inside each outer
  training fold. The primary metric is average precision because the endpoint
  is highly imbalanced.
- Exact TreeSHAP contributions are calculated on held-out outer-fold patients.

The comparison includes size-only, geometry-only, paper radiomics, paper
radiomics plus voxel descriptors, deep features, the paper-style full fusion,
and the adapted full fusion.

## Remote run

Private labels must exist at
`data_private/nephrectomy_cohort_labels.json`. Then run:

```bash
bash scripts/run_nephrectomy_multilevel_remote.sh
```

Large ROIs, weights, and patient-level results are written outside Git to
`/root/autodl-tmp/nephrectomy_multilevel` by default.
