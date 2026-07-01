# Retroperitoneal Tumor CT Dataset Standard v0

This is a minimal raw-image dataset for the pilot four-class retroperitoneal tumor classification workflow.

## Contents

- `images/`: raw venous/portal-phase CT NIfTI files, one file per case, named by anonymized group ID.
- `labels.csv`: case-level pathology labels.
- `metadata.csv`: clinical/image metadata and source traceability fields.
- `all.csv`: combined labels, metadata, and split assignment.
- `splits/train.csv`: 37 training cases.
- `splits/test.csv`: 4 fixed test cases, one per four-class label.
- `label_mapping.json`: numeric label mappings.
- `checksums_sha256.csv`: file size and SHA256 checksum for each image.
- `dataset_summary.json`: compact machine-readable dataset summary.

## Label Definition

Four-class labels:

- `肉瘤类`
- `良性神经源性肿瘤`
- `副神经节瘤`
- `淋巴瘤`

Fine labels currently present:

- `LMS`
- `GN`
- `SWN`
- `PGL`
- `LYM`

This v0 dataset contains no liposarcoma cases yet. The `肉瘤类` cases are currently all `LMS`.

## Split

The test set is a fixed 4-case holdout with one case per four-class label. No separate validation set is included because the current sample size is small. Model selection should use internal cross-validation or fixed training settings on the training set.

## Preprocessing Policy

Images are stored as raw NIfTI CT volumes. HU clipping, multi-window three-channel conversion, body crop, slice sampling, resizing, and normalization are not baked into this dataset. Those transforms should live in training or preprocessing code so that experiments remain reproducible and adjustable.

Recommended first-pass model input:

- venous/portal-phase CT only
- 2D or 2.5D slice-level MIL
- dynamic multi-window slice transform during training

