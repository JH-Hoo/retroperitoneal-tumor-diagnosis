# Extended Evaluation

Extended OOF evaluation for the primary `minvox5000` clinical4 model.

Key results:

| Item | Result |
|---|---:|
| clinical4 macro one-vs-rest ROC-AUC | 0.782 |
| clinical4 macro one-vs-rest PR-AUC | 0.558 |
| derived binary threshold for risk recall >= 0.95 | risk threshold 0.515 |
| derived binary benign recall at that threshold | 0.531 |
| binary-head threshold for risk recall >= 0.95 | risk threshold 0.400 |
| binary-head benign recall at that threshold | 0.438 |

Files:

- `extended_metrics.json`: AUC/PR-AUC, calibration, threshold tuning, and bootstrap CIs.
- `threshold_curve_derived_binary.csv`: derived binary threshold sweep.
- `threshold_curve_binary_head.csv`: explicit binary-head threshold sweep.
- `error_review.csv`: clinical4 or binary-head error cases with tumor-volume and cache status metadata.
- `calibration_derived_binary.png`: derived binary benign-like calibration plot.
- `calibration_binary_head.png`: binary-head benign-like calibration plot.
