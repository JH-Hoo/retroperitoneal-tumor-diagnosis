# Retroperitoneal Tumor CT Diagnosis

This repository is intentionally narrowed to the current working pipeline:

1. Generate organ and tumor-candidate masks with the Shenzhen-Yorktal FLARE23 champion project.
2. Use FLARE label `14` to sample lesion-guided 2.5D CT slices.
3. Train an ImageNet-pretrained ResNet18 attention-MIL model for binary diagnosis.

The binary task is:

- benign: `良性神经源性肿瘤`
- malignant/risk: `肉瘤类 + PPGL + 淋巴瘤 + 胃肠道间质瘤`

This branch contains only the champion-mask 2.5D ResNet pipeline and its current binary CV summary.

## External Segmentation

The champion FLARE23 implementation is an external dependency, not vendored code.

See [external/flare23_champion/README.md](external/flare23_champion/README.md).

Remote inference helper:

```bash
bash scripts/monitor_and_run_flare23_champion.sh
```

Expected champion outputs:

```text
/root/autodl-tmp/flare23_champion_outputs/Gxxxx.nii.gz
```

## 2.5D Input

Each case is converted into a tensor:

```text
15 slices x 5 channels x 224 x 224
```

Channels:

1. soft-tissue CT window
2. fat-sensitive CT window
3. tumor mask, FLARE label `14`
4. 2D peritumor shell
5. organ union, FLARE labels `1-13`

The current default keeps cases with at least `5000` champion label14 voxels.

## Run On The 4090 Machine

After champion FLARE23 inference and label14 statistics are available:

```bash
bash scripts/run_champion_resnet25d_binary_remote.sh
```

The script runs:

1. `scripts/prepare_champion_minvox_labels.py`
2. `scripts/build_flare23_25d_cache.py`
3. `scripts/train_resnet25d_binary_cv.py`

Private NIfTI files, Excel sheets, tensor caches, and model weights are ignored by Git.

## Current Result

Current champion-mask binary 5-fold OOF result, using `minvox5000`:

| Model | Cases | Accuracy | Balanced Accuracy | Macro F1 | Benign Recall | Risk Recall |
|---|---:|---:|---:|---:|---:|---:|
| Champion FLARE23 + 2.5D ResNet | 179 | 0.838 | 0.718 | 0.721 | 0.531 | 0.905 |

Confusion matrix:

![OOF confusion matrix](reports/champion_resnet25d_binary_minvox5000/confusion_matrix.png)

Full summary:

```text
reports/champion_resnet25d_binary_minvox5000/summary.json
```

## Repository Layout

```text
scripts/
  monitor_and_run_flare23_champion.sh
  prepare_champion_minvox_labels.py
  build_flare23_25d_cache.py
  train_resnet25d_binary_cv.py
  run_champion_resnet25d_binary_remote.sh

external/flare23_champion/
  README.md

reports/champion_resnet25d_binary_minvox5000/
  summary.json
  oof_predictions.csv
  confusion_matrix.png

data/champion_flare23_25d_cache_15x224_minvox5000/
  dataset_summary.json
  tensors_sha256.csv
```
