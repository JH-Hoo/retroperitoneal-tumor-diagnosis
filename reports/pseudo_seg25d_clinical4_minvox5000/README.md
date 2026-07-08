# Pseudo-Segmentation Clinical4 Baseline

REMIND-like segmentation-as-classification baseline using champion FLARE label14
as a pseudo tumor ROI. In each cached 2.5D crop, label14 pixels are assigned to
the case-level clinical4 class. A small 2D U-Net is trained on CT soft window,
CT fat window, organ union, and z-position channels. At inference, predicted
class-specific foreground pixels are summed per case to produce clinical4 top-1
and top-2 predictions.

OOF result on the same 179-case `minvox5000` cohort:

| Metric | Result |
|---|---:|
| clinical4 accuracy | 0.408 |
| clinical4 balanced accuracy | 0.465 |
| clinical4 macro-F1 | 0.411 |
| clinical4 top-2 accuracy | 0.665 |
| derived binary accuracy | 0.715 |
| derived binary balanced accuracy | 0.545 |
| derived binary benign-like recall | 0.281 |

Interpretation: this pseudo-mask REMIND-like baseline is weaker than the 2.5D
ResNet MIL model. It is useful as a negative/diagnostic control: simply turning
FLARE label14 into class-aware pseudo segmentation is not enough to replace the
MIL classifier without cleaner class-aware tumor annotations.

Files:

- `summary.json`
- `oof_predictions.csv`
- `pseudo_seg25d_clinical4_oof_confusion_matrix.png`
- `pseudo_seg25d_derived_binary_oof_confusion_matrix.png`
