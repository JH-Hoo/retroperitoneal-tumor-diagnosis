# Retroperitoneal Tumor CT Diagnosis

## What This Project Does

This project trains weakly supervised 2.5D MIL models for retroperitoneal tumor diagnosis from contrast-enhanced CT.

The current baseline does not use tumor segmentation or manual lesion boxes. Each CT case is represented by 96 axial slices with three CT windows, then aggregated into a case-level prediction. The latest small-data experiment freezes a ResNet18 slice feature extractor and trains only a lightweight MIL head.

## Dataset

- Original CT NIfTI files: 252
- Skipped corrupted NIfTI files: 3
- 96-slice cache tensors: 249
- Supervised five-class cases: 246
- Split: 5-fold `StratifiedGroupKFold` by salted `patient_uid_hash`
- Classes: 肉瘤类, 良性神经源性肿瘤, PPGL, 淋巴瘤, 胃肠道间质瘤

Raw NIfTI files, tensor cache files, source Excel sheets, linkage tables, hash salt, and model weights are not included in GitHub.

## Method

- Input tensor per case: `96 x 3 x 224 x 224`
- CT windows:
  - soft tissue: `[-160, 240]`
  - fat-sensitive: `[-200, 100]`
  - wide abdomen: `[-200, 400]`
- Cache variants: whole-abdomen and simple body crop
- Backbone: ImageNet-pretrained ResNet18
- Current preferred baseline: frozen slice features + mean/max MIL head
- Earlier baseline: partial fine-tuning with attention MIL
- Loss: class-weighted cross entropy

## Current Experiment

Current frozen-feature 5-fold runs:

```text
runs/*features*_fold0_meanmax/
runs/*features*_fold1_meanmax/
runs/*features*_fold2_meanmax/
runs/*features*_fold3_meanmax/
runs/*features*_fold4_meanmax/
```

5-fold test results are reported as mean +/- standard deviation:

| Task | Input | Balanced Accuracy | Macro-F1 | AUROC |
|---|---|---:|---:|---:|
| Five-class | Whole | 0.252 +/- 0.071 | 0.234 +/- 0.060 |  |
| Five-class | Body crop | 0.236 +/- 0.106 | 0.217 +/- 0.094 |  |
| Sarcoma vs non | Whole | 0.571 +/- 0.108 | 0.568 +/- 0.110 | 0.591 +/- 0.119 |
| Sarcoma vs non | Body crop | 0.581 +/- 0.066 | 0.561 +/- 0.087 | 0.600 +/- 0.048 |
| PPGL vs non | Whole | 0.534 +/- 0.116 | 0.501 +/- 0.085 | 0.568 +/- 0.136 |
| PPGL vs non | Body crop | 0.575 +/- 0.136 | 0.547 +/- 0.111 | 0.622 +/- 0.137 |
| Lymphoma vs non | Whole | 0.487 +/- 0.112 | 0.473 +/- 0.107 | 0.508 +/- 0.117 |
| Lymphoma vs non | Body crop | 0.498 +/- 0.103 | 0.480 +/- 0.091 | 0.510 +/- 0.086 |

These numbers remain exploratory, but they are now patient-level 5-fold estimates instead of a single-fold smoke test.

Report:

```text
reports/frozen_feature_meanmax_5fold_report.md
```

## How To Run

Prepare de-identified labels and patient-level folds:

```bash
python scripts/01_prepare_5class_labels.py
```

Build the 96-slice cache from private NIfTI files:

```bash
python scripts/02_build_96slice_cache.py
CACHE_NAME=cache_body_96slice BODY_CROP=1 python scripts/02_build_96slice_cache.py
```

Train the earlier partial fine-tuning fold 0 baseline:

```bash
python scripts/03_train_mil.py
```

Extract frozen ResNet18 features:

```bash
CACHE_NAME=cache_96slice python scripts/03_extract_slice_features.py
CACHE_NAME=cache_body_96slice python scripts/03_extract_slice_features.py
```

Train frozen-feature MIL heads:

```bash
FEATURE_NAME=features_cache_96slice_resnet18 TASK=5class POOLING=meanmax python scripts/04_train_mil_head.py
FEATURE_NAME=features_cache_body_96slice_resnet18 TASK=ppgl POOLING=meanmax python scripts/04_train_mil_head.py
```

Regenerate the earlier fine-tuning report:

```bash
python scripts/05_make_report.py
```

Summarize frozen-feature 5-fold runs:

```bash
python scripts/06_summarize_frozen_5fold.py
```

## Repository Structure

```text
configs/   One small YAML file for the current run
scripts/   Ordered scripts: prepare labels, build cache, train, report
data/      De-identified labels and cache metadata
runs/      Experiment outputs, metrics, predictions, figures
reports/   Markdown report
envs/      CUDA/PyTorch environment files
```

Private local-only content lives under `data_private/` and is ignored by Git.

## Notes

The current result is a lightweight baseline. It shows that the pipeline works, but it is not strong enough to claim stable clinical performance. The next useful step is retroperitoneal or lesion coarse crop, not a larger whole-volume model.
