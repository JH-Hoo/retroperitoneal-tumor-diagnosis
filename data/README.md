# Data

This directory contains de-identified labels, patient-level split files, and cache metadata for the binary screening task.

Private source files are kept outside Git under `data_private/`, including:

- source Excel sheets
- raw NIfTI metadata with PHI
- linkage audit files
- hash salt
- raw NIfTI images
- tensor cache files

Canonical case metadata are in `data/labels/labels_5class.csv`. The active training script maps those source labels into the binary task:

- `良性神经源性肿瘤` -> negative class
- `肉瘤类`, `淋巴瘤`, `PPGL`, `胃肠道间质瘤` -> positive class
