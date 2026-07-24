# Stereo Reconstruction and Epipolar Geometry

Three-part implementation of the camera geometry pipeline: projection matrix estimation, fundamental matrix and epipolar geometry, and 3D point triangulation from stereo pairs.

## Tasks

### Task 1 — Projection Matrix Estimation
Given a set of 3D-to-2D point correspondences, estimates the 3×4 projection matrix M using the Direct Linear Transform (DLT). Reprojection error on the test data:

```
Average distance error: 0.002227 pixels
```

That's effectively exact — the correspondences are consistent with a single linear camera model.

### Task 2 — Fundamental Matrix and Epipolar Geometry
Estimates F using the 8-point algorithm (normalized coordinates) for three stereo datasets: Temple, Z-translation, and X-translation. Epipoles are extracted from the null space of F and verified geometrically. The resulting epipolar lines are visualized overlaid on both images of each pair.

### Task 3 — 3D Triangulation
Recovers the Essential matrix E from F and the calibration matrix K, decomposes it into the 4 possible (R, t) configurations, and selects the correct one by checking that reconstructed points lie in front of both cameras. Final output is a 3D point cloud of the scene (880 points for the ReallyInwards dataset).

## How to run

```bash
pip install numpy matplotlib opencv-python scipy

# Task 1: Projection matrix
python task1.py

# Tasks 2 and 3: Fundamental matrix, epipolar geometry, triangulation
python task23.py
```

Results (epipolar visualizations, 3D scatter plots) are saved to `results/`.

## What I'd improve

The 8-point algorithm is sensitive to noise and the choice of normalization. RANSAC-based estimation (like in OpenCV's `findFundamentalMat`) would give more robust results on real image pairs with outlier correspondences.
