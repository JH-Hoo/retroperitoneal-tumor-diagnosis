# ULS23 VOI Preparation

`uls23_voi_status.csv` records whether each pilot case has enough lesion-center information to crop a ULS23-style VOI.

When `data/annotations/tumor_clicks_pilot_30.csv` has `x_voxel`, `y_voxel`, and `z_voxel` filled, `scripts/13_prepare_uls23_vois.py` writes lesion-centered `256 x 256 x 128` VOIs here. The NIfTI VOI files are ignored by Git.
