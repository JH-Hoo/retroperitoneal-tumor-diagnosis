# Binary Label Definition

The active task is:

> benign neurogenic tumors vs non-benign/actionable retroperitoneal tumors

This is a screening-style binary task, not a strict pathology-level benign-vs-malignant task.

## Mapping

| Binary class | Class id | Source `label_5` values | Count |
|---|---:|---|---:|
| `benign_neurogenic` | 0 | `良性神经源性肿瘤` | 55 |
| `nonbenign_actionable` | 1 | `肉瘤类`, `淋巴瘤`, `PPGL`, `胃肠道间质瘤` | 191 |

Total supervised cases: 246.

PPGL is grouped with the positive class because these tumors are clinically actionable and should not be treated as ordinary benign neurogenic tumors in this screening baseline.

## Files

- `labels_5class.csv`: de-identified case table and source five-class labels.
- `splits/fold_*/`: patient-level train/val/test splits reused by the binary task.
- `binary_label_mapping.json`: machine-readable binary mapping.
- `binary_fold_counts.csv`: binary class counts for each fold.
