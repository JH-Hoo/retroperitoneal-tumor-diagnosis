# Clinical4 Signal-Source Ablations

This folder tracks P0 ablations for the champion FLARE23 2.5D ResNet pipeline.
The first comparison asks whether the current score depends mainly on structured
auxiliary features, or whether the CT/mask slice bag itself carries signal.

Both runs use the same 179-case `minvox5000` cohort, 5-fold OOF evaluation,
ImageNet-pretrained ResNet18 slice encoder, zero-initialized extra mask channels,
gated multi-head attention, mean/max/logsumexp pooling, and joint clinical4 +
binary-head training.

| Run | Aux features | Clinical4 Acc | Clinical4 Bal Acc | Clinical4 Macro F1 | Clinical4 Top-2 | Binary Head Acc | Binary Head Bal Acc | Binary Head Macro F1 | Risk Recall | Benign Recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| full_aux_minvox5000 | yes | 0.592 | 0.532 | 0.529 | 0.816 | 0.849 | 0.725 | 0.733 | 0.918 | 0.531 |
| no_aux_minvox5000 | no | 0.547 | 0.500 | 0.485 | 0.821 | 0.810 | 0.701 | 0.691 | 0.871 | 0.531 |

Interpretation:

- Removing auxiliary z/volume/spacing features hurts clinical4 top-1 metrics and
  the explicit binary head, so those structured features are helping.
- The no-aux model still reaches clinical4 top-2 accuracy of 0.821 and keeps the
  binary-head benign recall at 0.531, so the result is not purely an auxiliary
  feature shortcut.
- The training curves show clear train/validation separation in both settings,
  so the next ablations should test CT-only and mask-channel variants before
  making the model larger.

Primary full-aux report:

- `../champion_resnet25d_clinical4_minvox5000/summary.json`
- `../champion_resnet25d_clinical4_minvox5000/resnet25d_clinical4_oof_confusion_matrix.png`
- `../champion_resnet25d_clinical4_minvox5000/resnet25d_binary_head_oof_confusion_matrix.png`

No-aux report:

- `no_aux_minvox5000/summary.json`
- `no_aux_minvox5000/resnet25d_clinical4_oof_confusion_matrix.png`
- `no_aux_minvox5000/resnet25d_binary_head_oof_confusion_matrix.png`

