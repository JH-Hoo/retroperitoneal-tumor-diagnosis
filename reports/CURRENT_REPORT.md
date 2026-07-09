# Current Experiment Report

Updated: 2026-07-09

## Bottom Line

The current best path is the champion FLARE23 label14-guided 2.5D ResNet MIL
pipeline. It is useful as a candidate-ranking and risk-triage baseline, but it
is not yet a stable standalone diagnostic model.

The strongest signal is clinical4 top-2 accuracy, which stays close to 0.80 in
repeated CV. Top-1 clinical4 accuracy and benign-like recall remain the main
weak points.

## Primary Model

Model: ImageNet-pretrained ResNet18, 15-slice 2.5D bag, CT soft/fat windows,
FLARE label14 tumor mask, peritumor shell, organ union, gated multi-head
attention, mean/max/logsumexp pooling, auxiliary z/volume features, clinical4
head plus binary head.

Single 5-fold OOF on the `minvox5000` cohort:

| Metric | Result |
|---|---:|
| cases | 179 |
| clinical4 accuracy | 0.592 |
| clinical4 balanced accuracy | 0.532 |
| clinical4 macro-F1 | 0.529 |
| clinical4 top-2 accuracy | 0.816 |
| binary-head accuracy | 0.849 |
| binary-head balanced accuracy | 0.725 |
| binary-head benign-like recall | 0.531 |

Repeated 5-fold CV across 5 seeds:

| Metric | Mean | 95% CI |
|---|---:|---:|
| clinical4 accuracy | 0.573 | 0.545-0.602 |
| clinical4 balanced accuracy | 0.505 | 0.469-0.541 |
| clinical4 macro-F1 | 0.502 | 0.466-0.537 |
| clinical4 top-2 accuracy | 0.801 | 0.789-0.813 |
| binary-head accuracy | 0.810 | 0.787-0.833 |
| binary-head balanced accuracy | 0.667 | 0.627-0.707 |
| binary-head benign-like recall | 0.444 | 0.373-0.515 |

## Signal-Source Ablations

| Run | N | Clinical4 Acc | Macro-F1 | Top-2 | Binary Benign Recall |
|---|---:|---:|---:|---:|---:|
| full aux, minvox5000 | 179 | 0.592 | 0.529 | 0.816 | 0.531 |
| aux-only | 179 | 0.363 | 0.291 | 0.637 | 0.344 |
| no aux, all image channels | 179 | 0.547 | 0.485 | 0.821 | 0.531 |
| CT-only, no aux | 179 | 0.547 | 0.501 | 0.821 | 0.500 |
| CT + tumor mask, no aux | 179 | 0.581 | 0.533 | 0.771 | 0.625 |
| CT + tumor + shell, no aux | 179 | 0.564 | 0.520 | 0.804 | 0.594 |
| full aux, minvox1000 | 207 | 0.527 | 0.456 | 0.807 | 0.375 |
| full aux, minvox0 | 246 | 0.455 | 0.410 | 0.736 | 0.600 |

Interpretation:

- Aux-only is weak, so the model is not only using z/volume shortcut features.
- CT appearance carries real signal.
- FLARE label14 tumor mask helps; CT + tumor mask is the strongest no-aux run.
- Lowering the label14 volume threshold adds cases but reduces clinical4
  performance, so segmentation quality matters.

## Extended Evaluation

| Item | Result |
|---|---:|
| clinical4 macro one-vs-rest ROC-AUC | 0.782 |
| clinical4 macro one-vs-rest PR-AUC | 0.558 |
| derived binary benign recall at risk recall >= 0.95 | 0.531 |
| binary-head benign recall at risk recall >= 0.95 | 0.438 |

The probability scores are usable for thresholded risk triage, but benign-like
recall is still not high enough for confident rule-out.

## Segmentation-As-Classification Baseline

The REMIND-like pseudo-segmentation baseline assigns each FLARE label14 pixel to
the case-level clinical4 label and trains a small 2D U-Net.

| Metric | Result |
|---|---:|
| clinical4 accuracy | 0.408 |
| clinical4 balanced accuracy | 0.465 |
| clinical4 macro-F1 | 0.411 |
| clinical4 top-2 accuracy | 0.665 |
| derived binary accuracy | 0.715 |
| derived binary benign-like recall | 0.281 |

Interpretation: this baseline is weaker than the MIL classifier. Pseudo label14
is not enough to reproduce the REMIND class-aware segmentation strategy without
cleaner tumor annotations.

## Error Hotspots

Clinical4 error rate by original label:

| Original label | N | Clinical4 error rate | Binary-head error rate |
|---|---:|---:|---:|
| GIST | 10 | 0.200 | 0.000 |
| sarcoma | 80 | 0.300 | 0.038 |
| benign neurogenic | 32 | 0.375 | 0.469 |
| PPGL | 25 | 0.520 | 0.080 |
| lymphoma | 32 | 0.688 | 0.219 |

The hardest class is lymphoma. PPGL is also unstable in 4-class prediction.
For binary triage, benign neurogenic recall is the main limitation.

## Artifacts

- Primary report: `reports/champion_resnet25d_clinical4_minvox5000/`
- Extended evaluation: `reports/champion_resnet25d_clinical4_minvox5000/extended_eval/`
- P0 ablations: `reports/ablations/`
- P2 pseudo-segmentation baseline: `reports/pseudo_seg25d_clinical4_minvox5000/`
- P3 repeated CV: `reports/repeated_cv/champion_resnet25d_clinical4_minvox5000/`
- Public manifest: `data/champion_flare23_25d_cache_15x224_minvox5000/manifest_public.csv`
