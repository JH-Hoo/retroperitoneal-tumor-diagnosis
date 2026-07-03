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

Current frozen-feature fold 0 runs:

```text
runs/*features*_fold0_meanmax/
```

Fold 0 test results:

| Task | Input | Accuracy | Balanced Accuracy | Macro-F1 | AUROC |
|---|---|---:|---:|---:|---:|
| Five-class | Whole | 0.367 | 0.258 | 0.218 |  |
| Five-class | Body crop | 0.286 | 0.212 | 0.204 |  |
| Sarcoma vs non | Whole | 0.776 | 0.741 | 0.749 | 0.751 |
| Sarcoma vs non | Body crop | 0.694 | 0.607 | 0.597 | 0.659 |
| PPGL vs non | Whole | 0.735 | 0.562 | 0.537 | 0.578 |
| PPGL vs non | Body crop | 0.714 | 0.622 | 0.560 | 0.624 |
| Lymphoma vs non | Whole | 0.653 | 0.518 | 0.517 | 0.462 |
| Lymphoma vs non | Body crop | 0.531 | 0.558 | 0.510 | 0.521 |

These fold 0 numbers are a smoke-test comparison, not a stable performance estimate.

Report:

```text
reports/fold0_frozen_feature_bodycrop_report.md
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

The current result is a lightweight baseline. It shows that the pipeline works, but it is not strong enough to claim stable clinical performance. The next useful step is full 5-fold frozen-feature evaluation, then retroperitoneal or lesion coarse crop, not a larger whole-volume model.
