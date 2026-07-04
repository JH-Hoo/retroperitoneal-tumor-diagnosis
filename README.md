# Retroperitoneal Tumor CT Binary Screening

This project trains a lightweight CT model for:

> benign neurogenic tumors vs non-benign/actionable retroperitoneal tumors

The task is intentionally not phrased as strict benign-vs-malignant diagnosis. PPGL cases are included in the non-benign/actionable group because their clinical management should not be mixed with ordinary benign neurogenic tumors.

## Dataset

- Supervised CT cases: 246
- Split: 5-fold patient-level split by salted `patient_uid_hash`
- Negative class: `良性神经源性肿瘤`, 55 cases
- Positive class: `肉瘤类`, `淋巴瘤`, `PPGL`, `胃肠道间质瘤`, 191 cases

Raw NIfTI files, tensor files, source Excel sheets, linkage tables, hash salt, and model weights are not included in Git.

## Method

- Input per case: 96 axial CT slices
- Slice size: `224 x 224`
- CT windows:
  - soft tissue: `[-160, 240]`
  - fat-sensitive: `[-200, 100]`
  - wide abdomen: `[-200, 400]`
- Feature extractor: ImageNet-pretrained ResNet18
- Aggregation: mean/max pooling over 96 slice features
- Optional tabular fusion: age and sex
- Loss: class-weighted cross entropy

The current preferred model uses whole-image 96-slice ResNet18 features plus age/sex fusion.

## Current Result

Pooled 5-fold test result for the preferred setting:

| Model | Accuracy | Balanced Accuracy | Macro-F1 | Sensitivity | Specificity | AUROC |
|---|---:|---:|---:|---:|---:|---:|
| Whole-image ResNet18 features + age/sex | 0.789 | 0.650 | 0.664 | 0.901 | 0.400 | 0.698 |

The model behaves like a screening baseline: sensitivity is relatively high, but specificity is limited. It should not be described as a clinically validated diagnostic model.

Detailed report:

```text
reports/binary_benign_malignant_trial_report.md
```

## How To Run

Train one fold with default settings:

```bash
python scripts/20_train_binary_feature_fusion.py
```

Train all folds with whole-image features and age/sex fusion:

```bash
for f in 0 1 2 3 4; do
  FOLD=$f FUSION=1 FEATURE_NAME=features_cache_96slice_resnet18 \
  RUN_NAME=binary_nonbenign_features_cache_96slice_resnet18_fold${f}_meanmax_age_sex_fusion \
  python scripts/20_train_binary_feature_fusion.py
done
```

Useful knobs:

```text
FOLD=0..4
FEATURE_NAME=features_cache_96slice_resnet18
POOLING=meanmax
FUSION=0 or 1
EPOCHS=80
BATCH_SIZE=16
LR=0.001
```

## Repository Structure

```text
data/labels/                         de-identified labels and patient-level folds
data/cache_96slice/                  96-slice cache metadata; tensors ignored by Git
data/features_cache_96slice_resnet18/ ResNet18 feature metadata; features ignored by Git
scripts/                             binary training entrypoint and preprocessing helpers
reports/                             binary task report and figures
envs/                                PyTorch/CUDA environment notes
```

Private local-only content lives under `data_private/` and is ignored by Git.
