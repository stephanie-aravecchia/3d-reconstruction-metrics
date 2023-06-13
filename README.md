# 3d-reconstruction-metrics

This repository contains the code to implement the 3D-reconstruction metrics described in:

> "Comparing Metrics for Evaluating 3D Reconstruction Quality in Large-Scale Environment Mappinge", Stéphanie Aravecchia, Marianne Clausel, Cédric Pradalier, 2023, currently under review.

## The main packages are:
* `mesh_slicer` to slice a mesh into images
* `pcd_slicer` to slice a point-cloud into images
* `slice_octomap` to slice an octomap octree into images
* `comparator` to compare the dataset reconstruction and ground-truth obtained from the two        previous slicers



### The comparaison is implemented on a specific folder hierarchy, please follow the guidelines   below to run them as they are provided
in `base_folder/` each xp is a subfolder `xp_id` as follows:
* `ground_truth_stl/stl/xp_id`  the stl and off files (the slicer input is a .off file)
* `ground_truth_stl/img/xp_id`  the sliced images from mesh_slicer are to be put here
* `3d_grid/ot/xp_id`  the octomap xp_id.ot
* `3d_grid/img/xp_id` slice_octomap will output the images here
* `3d_grid/res/obs/xp_id` the comparator wil read in the two img folders, and will output the      result here

## The simulated environments can be found here:
https://github.com/stephanie-aravecchia/unstructured-env-simulator.git

<img title="Simu" src="https://github.com/stephanie-aravecchia/unstructured-env-simulator/blob/main/pics/simu-environment.png" alt="Simu" width="600">



