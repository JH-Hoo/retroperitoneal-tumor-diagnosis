# Data Directory

This repository keeps only lightweight metadata for the current champion FLARE23 + 2.5D ResNet pipeline.

Private or generated files are intentionally ignored:

- original CT NIfTI files
- source Excel sheets
- patient linkage tables
- generated label CSV files
- tensor caches
- model weights

The committed `champion_flare23_25d_cache_15x224_minvox5000/dataset_summary.json` records the expected tensor shape and channel definition for the current run.
