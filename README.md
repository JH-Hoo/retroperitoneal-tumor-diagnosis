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
- Aggregation options: mean/max pooling, top-k MIL, gated-attention MIL
- Optional tabular branch: age and sex
- Optional samplers: natural, binary-balanced, subtype-balanced, binary50-subtype50
- Threshold selection: fixed 0.5, Youden, or validation-selected screening thresholds
- Model selection: macro-F1, balanced accuracy, AUROC, AP, or rank score
- Optional multiview features: z-jitter, window jitter, mild affine/noise, test-time view averaging

The current preferred setting is still deliberately small: whole-image 96-slice
ResNet18 features plus age/sex, without ROI, segmentation, TotalSeg, or tumor
center clicks.

The next engineering direction is S5-like late fusion: keep metadata-only as a
required control, train image-only MIL models, then select late-fusion weights
and screening thresholds on validation folds.

## Current Result

Pooled 5-fold test results. Each case appears once in a fold test set, so the
pooled confusion matrix covers all 246 cases.

| Setting | Accuracy | Balanced Accuracy | Macro-F1 | Sensitivity | Specificity | AUROC | Confusion Matrix |
|---|---:|---:|---:|---:|---:|---:|---|
| Previous ResNet18 + age/sex baseline | 0.789 | 0.650 | 0.664 | 0.901 | 0.400 | 0.698 | `[[22,33],[19,172]]` |
| High-sensitivity ensemble: metadata + gated MIL fusion | 0.813 | 0.679 | 0.698 | 0.921 | 0.436 | 0.642 | `[[24,31],[15,176]]` |
| Balanced late fusion: metadata + image-only mean/max MIL | 0.805 | 0.680 | 0.694 | 0.906 | 0.455 | 0.712 | `[[25,30],[18,173]]` |

The first optimized setting is the current screening-favored result because it
reduces missed non-benign/actionable cases from 19 to 15. The late-fusion setting
is a cleaner image-plus-tabular control and gives better AUROC/specificity while
keeping sensitivity above 0.90.

These numbers are still exploratory cross-validation results, not clinical
validation.

Detailed report:

```text
reports/binary_benign_malignant_trial_report.md
reports/腹膜后肿瘤CT二分类筛查模型技术报告.md
reports/腹膜后肿瘤CT二分类筛查模型技术报告.pdf
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
SAMPLER=natural|balanced|subtype_balanced|binary50_subtype50
LOSS=weighted_ce|ce|focal
SELECT_METRIC=macro_f1|balanced_accuracy|youden|screening|auroc|average_precision|rank_score
THRESHOLD_MODE=fixed_05|youden|sens90|sens85
NUM_VIEWS=1
TRAIN_VIEW_MODE=random
TEST_VIEW_MODE=mean
SCHEDULER=none|cosine
CLIP_NORM=0 or 5
SHUFFLE_METADATA=0 or 1
EPOCHS=80
BATCH_SIZE=16
LR=0.001
```

Pool 5-fold predictions and compute bootstrap confidence intervals:

```bash
RUN_PATTERN='binary_nonbenign_features_cache_96slice_resnet18_fold{fold}_meanmax_age_sex_fusion' \
python scripts/05_pool_cv_predictions.py
```

Run validation-selected late fusion:

```bash
FUSION_RUNS='metadata_only_fold{fold},image_meanmax_fold{fold},image_gated_fold{fold}' \
FUSION_SELECT=auroc \
THRESHOLD_MODE=sens90 \
FUSION_NAME=metadata_image_late_fusion \
python scripts/06_late_fusion.py
```

Build multiview cache and frozen features:

```bash
CACHE_NAME=cache_96slice_aug5 NUM_VIEWS=5 python scripts/02b_build_multiview_tensor_cache.py
CACHE_NAME=cache_96slice_aug5 FEATURE_NAME=features_cache_96slice_aug5_resnet18 NUM_VIEWS=5 python scripts/03b_extract_multiview_features.py
```

Generate error analysis tables and top-slice montages:

```bash
PRED_CSV=runs/fusion_metadata_image_late_fusion/test_predictions.csv python scripts/07_error_analysis.py
PRED_CSV=runs/fusion_metadata_image_late_fusion/test_predictions.csv python scripts/08_make_top_slice_montage.py
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
