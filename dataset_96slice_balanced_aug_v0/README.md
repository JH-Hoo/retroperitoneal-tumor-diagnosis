# Retroperitoneal Tumor CT Dataset 96-slice Balanced Aug v0

Experimental cache derived from `dataset_standard_v0` for class-balanced MIL smoke tests.

Only the training split is augmented. Test cases remain one fixed cached bag per source case.
Each cached bag is a PyTorch tensor with shape `96 x 3 x 224 x 224` and dtype `uint8`.
Training multipliers: 肉瘤类 14x, 良性神经源性肿瘤 14x, 副神经节瘤 6x, 淋巴瘤 1x.
This dataset must be treated as augmentation/oversampling, not as additional independent patients.
