# Champion FLARE23 2.5D ResNet Clinical4 Report

This report is the 5-fold OOF result for the primary 4-class clinical-imaging task:

| ID | Class |
|---:|---|
| 0 | sarcoma/GIST-like |
| 1 | lymphoma |
| 2 | PPGL |
| 3 | benign neurogenic |

The derived binary output is computed from the 4-class probabilities:

```text
risk/workup = P(class 0) + P(class 1) + P(class 2)
benign-like = P(class 3)
```

Artifacts:

- `summary.json`: training settings, fold metrics, OOF metrics.
- `oof_predictions.csv`: per-case 4-class probabilities, top-1/top-2, and derived binary output.
- `oof_predictions_derived_binary.csv`: compact per-case derived binary output.
- `resnet25d_clinical4_oof_confusion_matrix.png`: 4-class confusion matrix.
- `resnet25d_derived_binary_oof_confusion_matrix.png`: derived binary confusion matrix.
