<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 159. Scoring a focal length by epipolar inliers picks the shortest lens

Structure from motion on photographs with no EXIF has to guess the focal
length, and the guess matters more than it looks: a seed pair reconstructed
through a focal length 10% off is a projectively distorted scene (~100 px at
the edge of a 2048 px frame), later views stop registering onto it, and
bundle adjustment creeps out of the distorted basin by about 1% per round
instead of leaving it (`crates/sfm`, measured 921 -> 1363 px over 10
re-triangulation rounds).

The obvious estimator - for each candidate f, count essential-matrix inliers
over the best-matched pairs and keep the maximum - is biased toward SHORT
focal lengths. On a 16-photo phone capture it chose 0.45x the long side, the
bottom of its scan range. Solving the whole capture under each candidate held
fixed and keeping the one that registers the most views at the lowest
reprojection error chose 0.69x (1408 px: 16/16 registered at 0.88 px, against
1.04-1.15 px either side), and the free refinement that follows holds it
(1410 px).

**Rule:** choose a calibration parameter by the quality of the reconstruction
it produces, not by a two-view consistency count. The solves are independent
and run in parallel; the features and matches are shared.
