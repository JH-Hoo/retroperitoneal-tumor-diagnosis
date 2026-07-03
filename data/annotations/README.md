# Annotations

This folder stores lightweight de-identified annotation tables for experimental branches.

`totalseg_pilot_30.csv` is a balanced 30-case pilot manifest for running TotalSegmentator anatomy priors.

`tumor_clicks_template.csv` is the expected manual input format for a later lesion-prompt experiment. ULS23-style lesion segmentation needs at least a lesion-centered point or VOI; TotalSegmentator alone is only used here as an anatomy prior.

`tumor_clicks_pilot_30.csv` is the working table to fill before running a ULS23-style lesion VOI experiment. Coordinates should be original NIfTI voxel indices.
