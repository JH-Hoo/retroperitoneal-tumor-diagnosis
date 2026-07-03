# Retroperitoneal Tumor CT Dataset 96-slice v0

Offline preprocessed cache derived from `dataset_standard_v0`.

Each case is stored as one PyTorch tensor in `tensors/` with shape `96 x 3 x 224 x 224` and dtype `uint8`.
The three channels are fixed CT windows: soft tissue `[-160, 240]`, fat-sensitive `[-200, 100]`, and wide abdomen `[-200, 400]`.
Training code should convert tensors to float, divide by 255, then apply ImageNet mean/std normalization.
