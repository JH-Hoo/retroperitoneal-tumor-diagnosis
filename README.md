# Retroperitoneal Tumor CT Diagnosis

## What This Project Does

This project trains a weakly supervised 2.5D attention MIL model for five-class retroperitoneal tumor diagnosis from contrast-enhanced CT.

The current baseline does not use tumor segmentation or manual lesion boxes. Each CT case is represented by 96 axial slices with three CT windows, then aggregated into a case-level prediction.

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
- Backbone: ImageNet-pretrained ResNet18
- Pooling: attention MIL
- Training: freeze backbone for 5 epochs, then unfreeze `layer4`
- BatchNorm: frozen/eval during training
- Loss: class-weighted cross entropy

## Current Experiment

Current run:

```text
runs/5class_groupcv_fold0_resnet18_mil/
```

Fold 0 results:

| Split | Accuracy | Balanced Accuracy | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|---:|
| Validation | 0.300 | 0.238 | 0.230 | 0.282 |
| Test | 0.429 | 0.313 | 0.276 | 0.372 |

Report:

```text
reports/5class_groupcv_fold0_report.md
```

## How To Run

Prepare de-identified labels and patient-level folds:

```bash
python scripts/01_prepare_5class_labels.py
```

Build the 96-slice cache from private NIfTI files:

```bash
python scripts/02_build_96slice_cache.py
```

Train fold 0:

```bash
python scripts/03_train_mil.py
```

Regenerate figures and report:

```bash
python scripts/04_make_report.py
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

The current result is a lightweight baseline. It shows that the pipeline works, but it is not strong enough to claim stable clinical performance. The next useful step is body crop or lesion coarse crop, not a larger backbone.
