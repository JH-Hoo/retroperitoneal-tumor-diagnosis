# Retroperitoneal Tumor CT Dataset cache_body_96slice

Offline preprocessed cache derived from private NIfTI files under `data_private/standard`.

Each case is stored as one PyTorch tensor in `tensors/` with shape `96 x 3 x 224 x 224` and dtype `uint8`.
The three channels are fixed CT windows: soft tissue `[-160, 240]`, fat-sensitive `[-200, 100]`, and wide abdomen `[-200, 400]`.
Body crop: `True`.
Training code should convert tensors to float, divide by 255, then apply ImageNet mean/std normalization.
