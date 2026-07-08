# No-Aux Clinical4 2.5D ResNet Ablation

This run repeats the primary 179-case `minvox5000` 5-fold OOF experiment with
the same CT/mask inputs and model architecture, but disables structured
auxiliary features with `--no-aux`.

The disabled auxiliary features are the z-prior and mask-derived structured
features fitted by `ZPriorScaler`. The model still receives:

1. soft-tissue CT window
2. fat-sensitive CT window
3. FLARE label14 tumor mask
4. peritumor shell
5. organ-union mask
6. per-slice z-position embedding

Clinical4 OOF metrics:

| Metric | Value |
|---|---:|
| Accuracy | 0.547 |
| Balanced accuracy | 0.500 |
| Macro F1 | 0.485 |
| Top-2 accuracy | 0.821 |

Clinical4 confusion matrix, rows=true and columns=predicted:

```text
[[58,  8, 13, 11],
 [10, 10,  1, 11],
 [ 9,  0, 12,  4],
 [ 3,  6,  5, 18]]
```

Binary head OOF metrics:

| Metric | Value |
|---|---:|
| Accuracy | 0.810 |
| Balanced accuracy | 0.701 |
| Macro F1 | 0.691 |
| Risk/workup recall | 0.871 |
| Benign-like recall | 0.531 |

Binary head confusion matrix, rows=true and columns=predicted:

```text
[[128, 19],
 [ 15, 17]]
```

Artifacts:

- `summary.json`: training settings, fold metrics, and OOF metrics.
- `oof_predictions.csv`: per-case 4-class probabilities and binary outputs.
- `oof_predictions_derived_binary.csv`: compact derived binary output.
- `oof_predictions_binary_head.csv`: compact explicit binary-head output.
- `resnet25d_clinical4_oof_confusion_matrix.png`: clinical4 confusion matrix.
- `resnet25d_derived_binary_oof_confusion_matrix.png`: derived binary confusion matrix.
- `resnet25d_binary_head_oof_confusion_matrix.png`: binary-head confusion matrix.

