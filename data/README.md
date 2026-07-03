# Data

This directory contains only de-identified labels, patient-level split files, and cache metadata.

Private source files are kept outside Git under `data_private/`, including:

- source Excel sheets
- raw NIfTI metadata with PHI
- linkage audit files
- hash salt
- raw NIfTI images
- tensor cache files

Canonical training labels are in `data/labels/labels_5class.csv`.
