# Clinical4 P0 Signal-Source Ablations

This folder contains the P0 ablations for the champion FLARE23 2.5D ResNet
clinical4 pipeline. The goal is to test where the current score comes from:
structured auxiliary features, CT appearance, FLARE label14 masks, mask-channel
initialization, and the `minvox` filter.

Unless noted otherwise, runs use 5-fold OOF evaluation, ImageNet-pretrained
ResNet18, gated multi-head attention, mean/max/logsumexp pooling, joint
clinical4 + binary-head training, and the `minvox5000` champion label14 cohort.

| Run | N | Clinical4 Acc | Bal Acc | Macro F1 | Top-2 | Binary Acc | Binary Bal Acc | Binary Macro F1 | Risk Recall | Benign Recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| full_aux_minvox5000 | 179 | 0.592 | 0.532 | 0.529 | 0.816 | 0.849 | 0.725 | 0.733 | 0.918 | 0.531 |
| aux_only_minvox5000 | 179 | 0.363 | 0.292 | 0.291 | 0.637 | 0.726 | 0.577 | 0.570 | 0.810 | 0.344 |
| no_aux_minvox5000 | 179 | 0.547 | 0.500 | 0.485 | 0.821 | 0.810 | 0.701 | 0.691 | 0.871 | 0.531 |
| ct_only_noaux_minvox5000 | 179 | 0.547 | 0.539 | 0.501 | 0.821 | 0.804 | 0.685 | 0.679 | 0.871 | 0.500 |
| ct_tumor_noaux_minvox5000 | 179 | 0.581 | 0.547 | 0.533 | 0.771 | 0.810 | 0.738 | 0.710 | 0.850 | 0.625 |
| ct_tumor_shell_noaux_minvox5000 | 179 | 0.564 | 0.541 | 0.520 | 0.804 | 0.793 | 0.715 | 0.688 | 0.837 | 0.594 |
| maskinit_small_minvox5000 | 179 | 0.559 | 0.523 | 0.512 | 0.771 | 0.832 | 0.653 | 0.673 | 0.932 | 0.375 |
| maskinit_mean_minvox5000 | 179 | 0.559 | 0.488 | 0.491 | 0.793 | 0.804 | 0.698 | 0.686 | 0.864 | 0.531 |
| full_aux_minvox1000 | 207 | 0.527 | 0.462 | 0.456 | 0.807 | 0.773 | 0.622 | 0.625 | 0.868 | 0.375 |
| full_aux_minvox0 | 246 | 0.455 | 0.417 | 0.410 | 0.736 | 0.744 | 0.693 | 0.669 | 0.785 | 0.600 |

Interpretation:

- Aux-only is weak (`clinical4 macro-F1=0.291`), so the main result is not just
  a shortcut through z/volume/spacing/crop features.
- Removing aux still keeps useful signal (`no_aux clinical4 top-2=0.821`), but
  full aux gives the best primary full model on `minvox5000`.
- Among no-aux channel runs, CT + tumor mask is strongest (`clinical4 acc=0.581`,
  `macro-F1=0.533`). Adding shell/organ does not improve this seed.
- Mask-channel initialization with zero remains the best current default. Small
  and mean initialization both underperform the full zero-init run.
- Lowering the FLARE label14 threshold hurts clinical4 performance:
  `minvox5000 > minvox1000 > minvox0`. The broader cohorts add real cases but
  also add lower-quality or harder masks, so results must be reported as
  threshold-specific.

Coverage of the requested P0 checklist:

| P0 item | Artifact |
|---|---|
| aux-only baseline | `aux_only_minvox5000/summary.json` |
| image-only without aux | `no_aux_minvox5000/summary.json` |
| CT-only | `ct_only_noaux_minvox5000/summary.json` |
| CT + tumor mask | `ct_tumor_noaux_minvox5000/summary.json` |
| CT + tumor mask + shell | `ct_tumor_shell_noaux_minvox5000/summary.json` |
| CT + tumor mask + shell + organ | `no_aux_minvox5000/summary.json` |
| mask_channel_init zero/small/mean | primary report, `maskinit_small_minvox5000/`, `maskinit_mean_minvox5000/` |
| minvox 0/1000/5000 | `full_aux_minvox0/`, `full_aux_minvox1000/`, primary report |

Each run folder contains:

- `summary.json`
- `oof_predictions.csv`
- `oof_predictions_derived_binary.csv`
- `oof_predictions_binary_head.csv`
- OOF confusion-matrix PNGs for clinical4, derived binary, and binary head.

Primary full-aux `minvox5000` report:

- `../champion_resnet25d_clinical4_minvox5000/summary.json`
- `../champion_resnet25d_clinical4_minvox5000/resnet25d_clinical4_oof_confusion_matrix.png`
- `../champion_resnet25d_clinical4_minvox5000/resnet25d_binary_head_oof_confusion_matrix.png`
