# Retroperitoneal Tumor CT Diagnosis

This repository is intentionally narrowed to the current working pipeline:

1. Generate organ and tumor-candidate masks with the Shenzhen-Yorktal FLARE23 champion project.
2. Use FLARE label `14` to sample lesion-guided 2.5D CT slices.
3. Train an ImageNet-pretrained ResNet18 multi-task gated-MIL model on 4 clinical-imaging groups.

The primary training task is:

| ID | Clinical-imaging group | Source labels |
|---:|---|---|
| 0 | sarcoma/GIST-like | `肉瘤类 + 胃肠道间质瘤` |
| 1 | lymphoma | `淋巴瘤` |
| 2 | PPGL | `PPGL` |
| 3 | benign neurogenic | `良性神经源性肿瘤` |

The derived binary output is computed from the 4-class probabilities:

```text
risk/workup = P(sarcoma/GIST-like) + P(lymphoma) + P(PPGL)
benign-like = P(benign neurogenic)
```

The model also trains an explicit binary head for `risk/workup` vs `benign-like`.
This branch contains the champion-mask 2.5D ResNet pipeline, with 4-class training as the main task.

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

## Run On The Remote GPU Machine

After champion FLARE23 inference and label14 statistics are available:

```bash
bash scripts/run_champion_resnet25d_clinical4_remote.sh
```

The script runs:

1. `scripts/prepare_champion_minvox_labels.py`
2. `scripts/build_flare23_25d_cache.py`
3. `scripts/train_resnet25d_clinical4_cv.py`

Private NIfTI files, Excel sheets, tensor caches, and model weights are ignored by Git.

## Current Result

Current champion-mask 4-class 5-fold OOF result, using `minvox5000`:

| Model | Cases | Accuracy | Balanced Accuracy | Macro F1 | Top-2 Accuracy |
|---|---:|---:|---:|---:|---:|
| Champion FLARE23 + 2.5D ResNet clinical4 multitask | 179 | 0.592 | 0.532 | 0.529 | 0.816 |

Per-class recall:

| Class | Recall |
|---|---:|
| sarcoma/GIST-like | 0.711 |
| lymphoma | 0.312 |
| PPGL | 0.480 |
| benign neurogenic | 0.625 |

Derived binary result from the same 4-class probabilities:

| Output | Accuracy | Balanced Accuracy | Macro F1 | Risk/Workup Recall | Benign-Like Recall |
|---|---:|---:|---:|---:|---:|
| derived from clinical4 probabilities | 0.866 | 0.711 | 0.738 | 0.952 | 0.469 |
| explicit binary head | 0.849 | 0.725 | 0.733 | 0.918 | 0.531 |

Clinical4 confusion matrix:

![Clinical4 OOF confusion matrix](reports/champion_resnet25d_clinical4_minvox5000/resnet25d_clinical4_oof_confusion_matrix.png)

Derived binary confusion matrix:

![Derived binary OOF confusion matrix](reports/champion_resnet25d_clinical4_minvox5000/resnet25d_derived_binary_oof_confusion_matrix.png)

Binary head confusion matrix:

![Binary head OOF confusion matrix](reports/champion_resnet25d_clinical4_minvox5000/resnet25d_binary_head_oof_confusion_matrix.png)

Full summary:

```text
reports/champion_resnet25d_clinical4_minvox5000/summary.json
```

## Current Ablation

The first P0 ablation repeats the same 179-case 5-fold OOF experiment with
structured auxiliary features disabled (`--no-aux`):

| Run | Clinical4 Accuracy | Clinical4 Balanced Accuracy | Clinical4 Macro F1 | Clinical4 Top-2 Accuracy | Binary Head Accuracy | Binary Head Balanced Accuracy |
|---|---:|---:|---:|---:|---:|---:|
| full aux | 0.592 | 0.532 | 0.529 | 0.816 | 0.849 | 0.725 |
| no aux | 0.547 | 0.500 | 0.485 | 0.821 | 0.810 | 0.701 |

This suggests auxiliary features help the top-1 and binary-head metrics, but the
CT/mask slice bag still carries useful signal. See:

```text
reports/ablations/README.md
```

## Repository Layout

```text
scripts/
  monitor_and_run_flare23_champion.sh
  prepare_champion_minvox_labels.py
  build_flare23_25d_cache.py
  train_resnet25d_clinical4_cv.py
  run_champion_resnet25d_clinical4_remote.sh

external/flare23_champion/
  README.md

reports/champion_resnet25d_clinical4_minvox5000/
  summary.json
  oof_predictions.csv
  oof_predictions_derived_binary.csv
  oof_predictions_binary_head.csv
  resnet25d_clinical4_oof_confusion_matrix.png
  resnet25d_derived_binary_oof_confusion_matrix.png
  resnet25d_binary_head_oof_confusion_matrix.png

reports/ablations/
  README.md
  no_aux_minvox5000/
    summary.json
    oof_predictions.csv
    oof_predictions_derived_binary.csv
    oof_predictions_binary_head.csv

data/champion_flare23_25d_cache_15x224_minvox5000/
  dataset_summary.json
  tensors_sha256.csv
```
