# FLARE23 Champion Segmentation Dependency

This repository uses the champion FLARE23 implementation only as an external tumor and organ mask generator.

- Upstream project: <https://github.com/Shenzhen-Yorktal/flare23>
- Required weights in the upstream `models/` directory:
  - `model_roi.pt`
  - `model_fine.pt`
  - `model_fine2.pt`
- Expected output masks: one `.nii.gz` segmentation per case, with organ labels `1-13` and tumor candidate label `14`.

The upstream code is not vendored here. Clone it separately on the GPU machine and run `scripts/monitor_and_run_flare23_champion.sh` from this repository to stage weights, prepare nnU-Net style inputs, and launch inference.

For PyTorch versions that default to `weights_only=True` in `torch.load`, the upstream inference script may need `torch.load(..., weights_only=False)` when loading the serialized full-model checkpoints.
