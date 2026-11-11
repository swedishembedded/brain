<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Structure from Motion and Multi-View Stereo

*How can we recover a 3D scene and the camera positions from ordinary photographs?*

Two cameras with a focal length of 500 pixels stand side by side, their axes parallel and their centres 0.6 metres apart. A fence post that appears in the left photograph appears $d = fB/Z$ pixels further left in the right one, where $B = 0.6$ m is the gap between the cameras and $Z$ the distance of the post: this is the disparity of [Volume VII, Chapter 5](../v07-vision/05-dense-prediction.md), and it is the whole of stereo vision. Read the other way, it is a ruler: $Z = fB/d$. A post that moved 60 pixels is 5 metres away.

<!-- worked: structure_from_motion.disparity_depth -->
Two cameras with f = 500 pixels, parallel axes and centres 0.6 m apart along x. A point at depth Z appears in the second image d = f B / Z pixels to the left of where it appears in the first (the disparity). The depth read back from a disparity that is wrong by half a pixel, and the first-order error Z^2 (0.5) / (f B):

| Z (m) | disparity d (px) | depth read from d + 0.5 | depth read from d - 0.5 | first-order error for 0.5 px |
|---:|---:|---:|---:|---:|
| 2 | 150.00 | 1.9934 | 2.0067 | 0.0067 |
| 5 | 60.00 | 4.9587 | 5.0420 | 0.0417 |
| 10 | 30.00 | 9.8361 | 10.1695 | 0.1667 |
| 20 | 15.00 | 19.3548 | 20.6897 | 0.6667 |
| 40 | 7.50 | 37.5000 | 42.8571 | 2.6667 |
<!-- /worked -->

The ruler is cruel to far things. Half a pixel of error in $d$ costs 4 centimetres at 5 metres and 2.7 metres at 40, and the first-order error $Z^2\,\delta d/(fB)$ of the last column grows with the square of the distance. That is a fact about every method in this chapter, and it returns in three places. But the formula assumed three things that nobody has when the photographs are a folder of holiday pictures of a statue: that we know which pixel of one photograph is the same point as which pixel of the other, that the two cameras are parallel and a known distance apart, and that the focal length is known. None of the three is given. The photographs come from places nobody wrote down, facing directions nobody measured, through a lens whose number may be missing from the file. This chapter is about recovering all of it from the pictures alone: first the matches, then the geometry that two matched pictures impose, then every camera and every point at once, and last the dense surface between the points. The result is called **structure from motion** when it is cameras and sparse points, and multi-view stereo when it is the surface, and it is the way a model of a real scene is made without a model.

The chapter follows one small synthetic scene from end to end, because a reconstruction that is wrong can look right for a long time, and the only way to see that it is wrong is to know the answer. Four things from earlier chapters will be used before they are explained again; try them first, the answers are at the end.

1. (Volume X, [Chapter 3](03-camera-geometry.md), the pinhole camera.) For $f_x = f_y = 500$ and principal point $(320, 240)$, where does the camera-frame point $(0.8, -0.4, 2)$ land, and which line of points lands on the pixel $(400, 300)$?
2. (Volume X, [Chapter 3](03-camera-geometry.md), camera extrinsics.) A camera has $\mathbf{R} = \mathbf{I}$ and $\mathbf{t} = (-0.6, 0, 0)$. Where is its centre in world coordinates, and which way does a program that reads the pair as camera to world put it?
3. (Volume II, [Chapter 2](../v02-learning/02-linear-regression.md), least squares.) A system $\mathbf{A}\mathbf{x} \approx \mathbf{b}$ has more equations than unknowns. Which $\mathbf{x}$ minimizes $\lVert\mathbf{A}\mathbf{x} - \mathbf{b}\rVert^2$, and what is the residual orthogonal to?
4. (Volume II, [Chapter 2](../v02-learning/02-linear-regression.md), gradient descent.) A loss is a bowl whose curvature is 1 in one direction and 100 in the other. What is the largest step size gradient descent can take without diverging, and by what factor does the error along the flat direction shrink per step at that size?
5. (Volume X, [Chapter 3](03-camera-geometry.md), the direct linear transform.) The homography of a plane was found from correspondences as the unit vector that makes $\lVert\mathbf{A}\mathbf{h}\rVert$ smallest. Which singular vector is it, and why was the point set normalized before the fit?

## A scene small enough to know

The scene has 60 points in a box in front of four cameras. Camera 0 defines the world: its pose is the identity. The other three stand to its right, 0.6, 1.3 and 2.0 metres along, all looking at one point in the middle of the scene, and every pixel is seen with half a pixel of Gaussian noise. Everything below is computed from the seeds printed here, so that a reader who reimplements the draws gets the same cloud.

<!-- worked: structure_from_motion.scene_setup -->
Every camera has f = 500 pixels, principal point (320, 240) and a 640 by 480 image, with no lens distortion. The 60 scene points come from `bookrng.generator("structure_from_motion.scene")` (seed 18033376832500173267): for each point in turn, three `uniform()` draws give X = 3.6 u - 1.8, Y = 2.4 u - 1.2 and Z = 4 + 3 u (metres, in the frame of camera 0, x right, y down, z forward). Camera 0 is the world frame; cameras 1 to 3 are placed with the `look_at` of the previous chapter at the eye positions below, looking at one target, with up = (0, -1, 0). The observed pixels are the projections plus noise from `bookrng.generator("structure_from_motion.noise")` (seed 8282942143422244089), standard deviation 0.5 pixel per coordinate: for view 0 to 3 in turn and point 0 to 59 in turn, `normal()` for u and then `normal()` for v.

| camera | eye C (metres) | looks at | u of the 60 points | v of the 60 points |
|---|---|---|---|---|
| 0 | (0, 0, 0) | straight ahead (R = I) | 99.7 .. 503.2 | 105.7 .. 364.8 |
| 1 | (0.6, 0.08, 0.05) | (0.6, 0, 5.5) | 21.6 .. 443.0 | 102.2 .. 364.5 |
| 2 | (1.3, -0.05, 0.15) | (0.6, 0, 5.5) | 18.9 .. 438.2 | 106.0 .. 365.4 |
| 3 | (2, 0.06, 0.1) | (0.6, 0, 5.5) | 38.3 .. 425.9 | 112.4 .. 354.7 |

First three points (X, Y, Z) and their noisy pixels in views 0 and 3:

| point | (X, Y, Z) | view 0 (u, v) | view 3 (u, v) |
|---|---:|---:|---:|
| 0 | (-0.9539, 1.103, 4.542) | (214.54, 361.21) | (146.95, 349.40) |
| 1 | (0.2828, 0.1954, 5.192) | (346.75, 258.51) | (283.43, 257.85) |
| 2 | (1.624, -0.823, 5.169) | (476.79, 160.33) | (410.66, 157.00) |
<!-- /worked -->

Nothing in the data that follows is secret from the code that generates it, and nothing is allowed to leak into the code that estimates: the estimators see pixels, descriptors and a focal length, and the truth is used only to score them.

## Where two rays meet

The easiest stage comes first, because everything else exists to make it possible. Suppose the cameras are known: poses $(\mathbf{R}_i, \mathbf{t}_i)$ and intrinsics. A matched pixel in each of two images gives two rays (chapter 3), and the point is where they cross. With noise they do not cross, since two lines in space generally miss each other, so one asks for the point that fits both best. This is **triangulation**.

Write the projection of a world point in homogeneous form, $\lambda\,\mathbf{x} = \mathbf{P}\tilde{\mathbf{X}}$, where $\mathbf{x} = (x, y, 1)$ is the normalized image point $\mathbf{K}^{-1}(u, v, 1)$ and $\mathbf{P} = [\mathbf{R} \mid \mathbf{t}]$ has rows $\mathbf{p}_1, \mathbf{p}_2, \mathbf{p}_3$. The unknown factor $\lambda$ equals the third entry $\mathbf{p}_3\cdot\tilde{\mathbf{X}}$, so $x\,(\mathbf{p}_3\cdot\tilde{\mathbf{X}}) = \mathbf{p}_1\cdot\tilde{\mathbf{X}}$ and the same with $y$ and $\mathbf{p}_2$. These are linear in $\tilde{\mathbf{X}}$:

$$
(x\,\mathbf{p}_3 - \mathbf{p}_1)\cdot\tilde{\mathbf{X}} = 0, \qquad (y\,\mathbf{p}_3 - \mathbf{p}_2)\cdot\tilde{\mathbf{X}} = 0 . \tag{1}
$$

Each view contributes two rows, $n$ views a matrix with $2n$ rows and 4 columns, and $\tilde{\mathbf{X}}$ is its null vector (the right singular vector of the smallest singular value, as for the homography of chapter 3); dividing by the fourth entry gives the point. Two views already give an overdetermined system, four equations for three unknowns. The method minimizes an algebraic quantity and not a distance in pixels, and the difference matters when the data are poor and not when they are good.

How good is the answer, as a function of where the cameras stand? The disparity argument of the opening answers it. Two noisy pixel coordinates give a disparity with noise $\sqrt{2}\,\sigma$, and the depth error is that noise times $Z^2/(fB)$. The simulation below puts a point 5 metres away in front of two cameras at five baselines and adds the half pixel of noise 200 times each.

<!-- worked: structure_from_motion.triangulation_error -->
A point at (0, 0, 5) seen by a camera at the origin and one B metres to the right (same orientation, f = 500); 200 trials per baseline, each adding Gaussian noise of standard deviation 0.5 pixel to the four pixel coordinates in the order u and v of camera 0, u and v of camera 1 (`bookrng.generator("structure_from_motion.triangulation")`, seed 15221094888374321603, one stream through all five baselines in the order printed, B = 0.05 first). Root mean square error of the estimated depth, in metres, for the linear method, for the linear method followed by three Gauss-Newton steps on the reprojection error, and the first-order prediction sqrt(2) sigma Z^2 / (f B):

| baseline B (m) | angle between the rays (deg) | linear | linear + 3 Gauss-Newton steps | prediction |
|---:|---:|---:|---:|---:|
| 0.05 | 0.57 | 0.7041 | 0.7042 | 0.7071 |
| 0.1 | 1.15 | 0.3374 | 0.3375 | 0.3536 |
| 0.25 | 2.86 | 0.1449 | 0.1448 | 0.1414 |
| 0.6 | 6.87 | 0.0582 | 0.0582 | 0.0589 |
| 1.5 | 17.06 | 0.0237 | 0.0237 | 0.0236 |
<!-- /worked -->

![Root mean square depth error against the angle between the two rays, both axes logarithmic: the measured values of the linear method fall on a straight line of slope minus one, which the prediction sqrt(2) sigma Z^2 / (f B) follows from 0.70 metre at 0.57 degrees to 0.024 metre at 17 degrees.](img/triangulation-error.svg)

The measured errors follow the prediction within the scatter of 200 trials, and the figure shows why a baseline is the first thing a reconstruction needs: halving the angle between the rays doubles the error of the depth. At half a degree the depth of a point 5 metres away is wrong by 0.7 metre, 14 percent; at 17 degrees by 2 centimetres. Three thresholds of `crates/sfm` and `crates/mvs` are this figure in code. A new point is triangulated only if some pair of its views sees it under at least `min_angle_deg`, 1.5 degrees by default (`SfmCfg` in `crates/sfm/src/incremental.rs`); a seed pair is accepted only if the median angle of its triangulated points is at least 3 degrees (`Solver::seed`); and the multi-view stereo that follows gives full weight to a source view whose angle to the reference lies between 5 and 60 degrees and a quadratic penalty outside (`angle_weight` in `crates/mvs/src/select.rs`: below, depth is ill-conditioned, above, the two views see different surfaces).

The table's two columns for the linear method and for the linear method followed by three Gauss-Newton steps on the reprojection error agree to the fourth digit, because with errors this small the algebraic and the geometric cost have the same minimum to first order. The triangulation of `crates/sfm/src/twoview.rs` is a refinement of equation (1) in another direction: it works on unit bearings instead of normalized coordinates, so that rays more than 90 degrees off the axis of a fisheye are usable, takes all three rows of the cross product $\mathbf{f}\times(\mathbf{R}\mathbf{X} + \mathbf{t}) = 0$ of each view, solves once, and then solves again with each view's rows divided by the point's range from it, so that each row measures the sine of an angular error and not an error weighted by how far away the camera is.

## Which pixel is the same pixel

Triangulation needs the pairs. A photograph is a million pixels and almost none of them is distinctive: a pixel on a white wall looks like every other. The pixels that can be found again are the corners and blobs of texture, and the standard way to find them is to detect **keypoints** (places where the picture has structure at a chosen scale) and to describe each by a **descriptor**, a vector summarizing the gradients around it, built to change little under a change of viewpoint, of scale, and of light. `crates/sfm/src/sift.rs` implements the scale-invariant feature transform of Lowe (2004): keypoints are extrema of a difference of Gaussians over scale space, refined to sub-pixel position, and each gets a 128-number descriptor made of orientation histograms of its neighbourhood, square-rooted and normalized to unit length so that the Euclidean distance between two descriptors is $\sqrt{2 - 2\,\mathbf{a}\cdot\mathbf{b}}$ and the nearest descriptor is the one with the largest dot product (`crates/sfm/src/matching.rs`).

**Feature matching** is the step that pairs keypoints of two images by their descriptors, and the obvious rule fails in an instructive way. Take for each descriptor of image A its nearest neighbour in image B and call the pair a match. Every keypoint then has a match, including those with no counterpart in the other image, and a surface with repeated structure (the windows of a facade, the tiles of a floor) gives descriptors so alike that the nearest one is a coin toss. Two cheap tests remove most of the damage. The **ratio test** keeps a match only if the nearest neighbour is much nearer than the second nearest, so that the descriptor is not nearly as close to something else, with the distance ratio below a threshold (0.8 by default in `SfmCfg::ratio`). The **mutual test** keeps it only if the nearest neighbour of the match, searched back in image A, is the keypoint we started from.

The toy features of the next table reproduce the situation. Each of the 60 scene points has a 32-number descriptor, the same in every view up to noise; ten pairs of points are made near-copies of each other, a repeated texture; and each image has 40 clutter keypoints that belong to no scene point. The recall column divides by the 60 points that both images show.

<!-- worked: structure_from_motion.feature_matching -->
Toy features for the four views: 32-number descriptors with unit length, from `bookrng.generator("structure_from_motion.features")` (seed 7678793205867606946). Draw order: the base descriptors of points 0 to 59 (32 `normal()` draws each, divided by the length); then, for k = 0 .. 9, point 10 + k gets the descriptor of point k plus 0.04 `normal()` on every entry, renormalized (a repeated texture, so that ten pairs of points look alike); then for view 0, 1, 2, 3 in turn, the 60 keypoints of the scene points in point order (their pixels from the noisy observations; the descriptor is the base descriptor plus 0.05 `normal()` on every entry, renormalized), then 40 clutter keypoints (u = 640 `uniform()`, v = 480 `uniform()`, then a fresh unit descriptor from 32 `normal()` draws), and finally `shuffle` of that list of 100 keypoints.

Matching view 0 against view 1, 60 scene points have a keypoint in both. The nearest neighbour of a descriptor is the one with the largest dot product (unit descriptors, so the distance is sqrt(2 - 2 dot)); the ratio test keeps a match when its distance is below the ratio times the distance of the second-nearest; the mutual test keeps it when the nearest neighbour of the match in the other image is the descriptor we started from. A match is correct when both keypoints belong to the same scene point.

| rule | matches | correct | wrong | precision | recall | correct matches among points 0 to 19 |
|---|---:|---:|---:|---:|---:|---:|
| nearest neighbour only | 100 | 54 | 46 | 54% | 90% | 14 |
| mutual only | 64 | 52 | 12 | 81% | 87% | 12 |
| ratio 0.9 | 55 | 49 | 6 | 89% | 82% | 9 |
| ratio 0.9 and mutual | 50 | 48 | 2 | 96% | 80% | 8 |
| ratio 0.8 | 45 | 45 | 0 | 100% | 75% | 5 |
| ratio 0.8 and mutual | 45 | 45 | 0 | 100% | 75% | 5 |
| ratio 0.7 and mutual | 41 | 41 | 0 | 100% | 68% | 1 |
<!-- /worked -->

The plain nearest-neighbour rule returns 100 matches of which 54 are right, a precision of 54 percent. The mutual test alone cuts the wrong matches from 46 to 12, since a clutter descriptor is rarely anyone's nearest neighbour in return: 64 matches. The ratio test at 0.8 removes the rest, 45 of 45 correct, and the last column shows what it costs: of the 20 points that were made twins, the plain rule matches 14 and the ratio test 5, and at 0.7 only 1. The ratio test refuses exactly the matches that cannot be trusted, and its price is the points it cannot tell apart from their neighbours. There is no setting that is right for every scene, which is why what follows must survive a matcher that is wrong a fifth of the time or more.

A matcher compares two images; a reconstruction wants a point followed through all of them. Merging the matches of every pair of views gives **tracks**: a track is a connected component of the graph whose nodes are (image, keypoint) pairs and whose edges are the matches, found with a union-find structure, one scene point per track. A track that contains two keypoints of one image is inconsistent (a point is in one place in one picture), and a wrong match is the usual cause, since it welds two tracks of different points; such a component is dropped. The same four views, matched at three rules:

<!-- worked: structure_from_motion.track_building -->
The matches between each of the six pairs of views under three rules are merged into tracks: the nodes are (view, keypoint) pairs, the edges are the matches, a track is a connected component of two or more nodes (found by union-find), and a component that holds two keypoints of one image is dropped. A track is pure when all its keypoints belong to one scene point.

| rule | matches (six pairs) | correct | components | dropped: two keypoints in one image | tracks kept | pure | pure, seen in all four views |
|---|---:|---:|---:|---:|---:|---:|---:|
| ratio 0.8 and mutual | 276 | 274 | 59 | 0 | 59 | 57 | 44 |
| ratio 0.9 and mutual | 313 | 298 | 68 | 3 | 65 | 54 | 47 |
| mutual only | 382 | 322 | 85 | 8 | 77 | 50 | 50 |
<!-- /worked -->

A looser matcher produces more, longer and more mixed tracks, and the last two columns say how many of them are pure. No rule gives all 60: the point of a pipeline is not to have a clean input but to have stages that each catch what the stage before let through.

The code decides which pairs to match at all, since a capture of $n$ photographs has $n(n-1)/2$ pairs and most share nothing. For up to 24 photographs every pair is matched; beyond that a global descriptor per image (VLAD, learned by k-means on the capture's own descriptors) ranks the others, and the best 12 and the neighbours in capture order are matched (`PairSelection` in `crates/sfm/src/retrieval.rs`). The nearest-neighbour search sits behind a trait (`NearestNeighbours`), so a device can supply it, which `recon::photogrammetry::DeviceMatcher` does when there is a device.

## One line instead of a plane

Take a verified pair of matches and forget the others for a moment: what does one image say about where a point must be in the other? A pixel of image 1 is a ray, the set of all points that could have produced it. Seen from the second camera, that ray is a line in the image, because a line of the world through a point projects to a line. The point's partner must lie on it: the search for a match is not over the image but along a line. This is **epipolar geometry**, the geometry that two views impose on each other, and its algebra is a single equation.

Let $\mathbf{x}_1$ and $\mathbf{x}_2$ be the normalized homogeneous images of one point in two cameras, with camera 1 at the origin and camera 2 related to it by $\mathbf{X}_2 = \mathbf{R}\mathbf{X}_1 + \mathbf{t}$. The point seen by camera 2 is $\mathbf{X}_2 \propto \mathbf{x}_2$, and $\mathbf{X}_1 \propto \mathbf{x}_1$. Three vectors lie in one plane, the plane through the two camera centres and the point: the translation $\mathbf{t}$ (the baseline), the ray $\mathbf{R}\mathbf{x}_1$ of camera 1 turned into camera 2's frame, and the ray $\mathbf{x}_2$. Three coplanar vectors have a vanishing triple product, so $\mathbf{x}_2\cdot(\mathbf{t}\times\mathbf{R}\mathbf{x}_1) = 0$. Writing the cross product with $\mathbf{t}$ as a matrix, $[\mathbf{t}]_\times\mathbf{y} = \mathbf{t}\times\mathbf{y}$,

$$
\mathbf{x}_2^\top\,\mathbf{E}\,\mathbf{x}_1 = 0, \qquad \mathbf{E} = [\mathbf{t}]_\times\,\mathbf{R}, \qquad [\mathbf{t}]_\times = \begin{pmatrix} 0 & -t_3 & t_2 \\ t_3 & 0 & -t_1 \\ -t_2 & t_1 & 0 \end{pmatrix}. \tag{2}
$$

The matrix $\mathbf{E}$ is the **essential matrix** of the pair. It has rank 2 because $[\mathbf{t}]_\times$ does (it sends $\mathbf{t}$ to zero), and since $\mathbf{R}$ preserves lengths its two nonzero singular values are equal; the converse is also true, so any $3\times3$ matrix with singular values $(\sigma, \sigma, 0)$ is an essential matrix. It has five degrees of freedom: three for the rotation and two for the direction of $\mathbf{t}$, since scaling $\mathbf{t}$ scales $\mathbf{E}$ and leaves equation (2) true. The scale of the baseline has dropped out of the constraint, which is the first place where the scale of the scene is lost, and the problem of the next sections is to live without it. With pixel coordinates $\mathbf{u}_i = \mathbf{K}\mathbf{x}_i$ the same constraint reads $\mathbf{u}_2^\top\mathbf{F}\mathbf{u}_1 = 0$ with the **fundamental matrix** $\mathbf{F} = \mathbf{K}^{-\top}\mathbf{E}\mathbf{K}^{-1}$, and $\mathbf{l} = \mathbf{F}\mathbf{u}_1$ is the **epipolar line** in image 2 (the points $\mathbf{u}$ with $\mathbf{l}\cdot\mathbf{u} = 0$). The images of each camera's centre in the other image are the epipoles; every epipolar line passes through the epipole.

For the pair of cameras 0 and 1 of the scene:

<!-- worked: structure_from_motion.epipolar_constraint -->
The pair (camera 0, camera 1): the relative rotation R = R_1 R_0^T and translation t = t_1 - R t_0 take a point of camera 0's frame to camera 1's, and E = [t]x R with t scaled to unit length.

R = [1, 0, 0; 0, 0.999892, 0.0146773; 0, -0.0146773, 0.999892]
t = (-0.6, -0.0807252, -0.0488204), |t| = 0.607371 (the baseline, in metres)
E = [0, 0.082322, -0.131715; -0.0803799, -0.0144992, 0.987757; 0.132909, -0.987757, -0.0144992]
singular values of E: 1, 1, 0; det E = 0.0e+00; the epipole of camera 1 in image 0 is at (6320.0, 1040.0) and that of camera 0 in image 1 at (6465.0, 1066.8).

| pair of keypoints | exact pixels: |b^T E a| | noisy pixels: Sampson distance (px) | same scene point |
|---|---:|---:|---|
| point 0 against point 0 | 1.3e-16 | 0.60 | yes |
| point 1 against point 1 | 4.9e-17 | 0.30 | yes |
| point 2 against point 2 | 3.8e-17 | 0.04 | yes |
| point 0 against point 5 | 7.1e-02 | 25.39 | no |
| point 1 against point 6 | 1.4e-01 | 49.59 | no |
| point 2 against point 7 | 3.7e-01 | 132.37 | no |

The epipolar line of the noisy view-0 pixel of point 0 in image 1 is l = F x with F = K^-T E K^-1: l = (-0.000224, 0.002002, -0.691), and its noisy view-1 pixel (146.29, 362.30) lies 0.86 pixels from it.
<!-- /worked -->

The constraint holds to rounding on exact pixels, $10^{-16}$, and fails by 0.3 to 0.6 pixel on noisy ones, which is the noise itself; a pairing of a point with another point violates it by 25 to 132 pixels. The epipoles are thousands of pixels from the image because the cameras moved sideways, which makes every epipolar line nearly horizontal, the case that rectified stereo assumes and the general case does not need.

The residual $\mathbf{x}_2^\top\mathbf{E}\mathbf{x}_1$ is not a distance. Dividing it by the length of the gradient of $\mathbf{x}_2^\top\mathbf{E}\mathbf{x}_1$ with respect to the four image coordinates, which is the first-order step to the nearest point of the constraint's surface, gives the **Sampson distance**,

$$
d_S = \frac{\lvert\mathbf{x}_2^\top\mathbf{E}\mathbf{x}_1\rvert}{\sqrt{(\mathbf{E}\mathbf{x}_1)_1^2 + (\mathbf{E}\mathbf{x}_1)_2^2 + (\mathbf{E}^\top\mathbf{x}_2)_1^2 + (\mathbf{E}^\top\mathbf{x}_2)_2^2}}, \tag{3}
$$

in normalized units, or $f d_S$ in pixels, where the subscripts are the first two entries of the vectors. This is the number the pair is scored with in the table. `sampson` in `crates/sfm/src/twoview.rs` takes the same quantity on the unit sphere of bearings instead of the image plane, so its value is an angle, and a pixel threshold becomes an angular one by dividing by the focal length.

## Eight points, and why they are not enough

Given matches, equation (2) is linear in the nine entries of $\mathbf{E}$. Each correspondence contributes one row, $b_r a_c$ over the entries $(r, c)$ of $\mathbf{E}$ in row-major order, for $\mathbf{a}$ the point of image 1 and $\mathbf{b}$ that of image 2; eight rows determine $\mathbf{E}$ up to scale as the null vector of the $8\times9$ matrix. This is the **eight-point algorithm**. With more points the null vector is the least-squares solution, and the conditioning rule of chapter 3 applies: move each point set so that its centroid is at the origin and the mean distance from it is $\sqrt{2}$ (Hartley's normalization), solve, and undo the two similarities, $\mathbf{M} = \mathbf{T}_b^\top\mathbf{E}'\mathbf{T}_a$. The result is not an essential matrix: noise breaks the equality of its two nonzero singular values and gives it a third. The nearest essential matrix, in the sense of the Frobenius norm, keeps the directions and sets the singular values to $(1, 1, 0)$.

```text
function eight_point(a: list<array<real>[3]>, b: list<array<real>[3]>, essential: bool)
        -> array<real>[3, 3]:
    require len(a) >= 8 and len(a) == len(b)
    Ta = hartley([[p[0], p[1]] for p in a])
    Tb = hartley([[p[0], p[1]] for p in b])
    rows = []
    for i in 0 .. len(a):
        x = matmul(Ta, a[i])
        y = matmul(Tb, b[i])
        append [y[r] * x[c] for r in 0 .. 3 for c in 0 .. 3] to rows
    v = null_vector(rows)
    M = matmul(transpose(Tb), matmul([v[0:3], v[3:6], v[6:9]], Ta))
    (U, sg, V) = svd(M)
    if essential:
        return matmul(U[:, 0:2], transpose(V[:, 0:2]))
    S2 = zeros(real, [2, 2])
    S2[0, 0] = sg[0]
    S2[1, 1] = sg[1]
    return matmul(U[:, 0:2], matmul(S2, transpose(V[:, 0:2])))
```

Here `hartley` is the routine of chapter 3, `null_vector` and `matmul` as there, and `svd(M)` returns the factors $\mathbf{U}$, $\boldsymbol{\sigma}$, $\mathbf{V}$ of Volume I, Chapter 6, with $\boldsymbol{\sigma}$ descending. With `essential` false the function returns the nearest matrix of rank two, which keeps the two largest singular values, a variant needed in the next section.

Eight is the number of unknowns of a general matrix, not of an essential one. An essential matrix has five parameters, and with eight points nothing in the linear solve knows that, so a noisy minimal sample produces a matrix that fits its own eight points and then loses most of its fit when it is forced onto the essential manifold. The effect is large enough to decide what the algorithm can be used for. From the 54 correct matches of the pair, take 100 random samples of 8 and count how many of the 54 each sample's hypothesis explains:

<!-- worked: structure_from_motion.minimal_samples -->
The 54 correct matches of the pair (0, 1) with their noisy pixels, and 100 samples of 8 of them from `bookrng.generator("structure_from_motion.minimal")` (seed 15877171000727059439; the same `integer` draws with repeats discarded). Each sample gives two hypotheses by the eight-point algorithm; the table counts, for each kind, the matches within 1.5 pixel of Sampson distance of it (all 54 are correct, so a perfect hypothesis scores 54).

| hypothesis | median inliers | largest |
|---|---:|---:|
| nearest essential matrix (singular values 1, 1, 0) | 1.0 | 30 |
| nearest rank-two matrix | 31.5 | 54 |
<!-- /worked -->

The nearest essential matrix of a minimal sample explains a median of 1 of the 54 matches, a rank-two matrix 31.5 of them. For the robust search that follows, a hypothesis is therefore the rank-two matrix, and the essential constraint is imposed only on the final fit, where dozens of inliers have averaged the noise. The minimal solver for the calibrated problem, which uses the five parameters directly, is the five-point algorithm of Nistér (2004), and it is what `crates/sfm/src/twoview.rs` uses (`essential_5pt`): the null space of the five linear equations is four-dimensional, $\mathbf{E} = x\mathbf{X} + y\mathbf{Y} + z\mathbf{Z} + \mathbf{W}$, and the ten cubic equations $\det\mathbf{E} = 0$ and $2\mathbf{E}\mathbf{E}^\top\mathbf{E} - \mathrm{tr}(\mathbf{E}\mathbf{E}^\top)\mathbf{E} = 0$ in $(x, y, z)$, reduced by Gauss-Jordan elimination to a polynomial of degree ten in one unknown, give up to ten real solutions. Why the smaller sample is worth its algebra is the next section.

## Voting for a model

The matcher gives matches of which an unknown share is wrong, and a least-squares fit to all of them is wrecked by the wrong ones: the eight-point algorithm on all 100 matches of the pair gives a rotation 35 degrees from the truth. The remedy of Fischler and Bolles (1981), **random sample consensus** or RANSAC, inverts the logic: do not fit the data and then look for outliers; fit a small sample, count how many data points agree with the fit, and repeat. A sample made only of correct matches gives a model that most correct matches agree with, and one wrong match in the sample gives a model that almost nothing agrees with. Take the best.

How many samples? If a fraction $w$ of the data are inliers, a sample of $s$ is all-inlier with probability $w^s$, and the chance that none of $N$ samples is, is $(1 - w^s)^N$. To make that chance at most 0.001, a confidence of 99.9 percent,

$$
N \ge \frac{\ln 0.001}{\ln(1 - w^s)} . \tag{4}
$$

The bound depends on the sample size in the exponent, and the table of the worked run shows what that means: at a half inliers an eight-point sample needs 1765 draws and a five-point sample 218; at 30 percent, 105,282 and 2840. The inlier fraction is not known in advance, so the bound is recomputed from the best model so far and only ever lowers the number of iterations left. Comparing a model's agreement by a plain count throws away how well the model agrees; the score that `crates/sfm/src/twoview.rs` and the example below use instead is the truncated quadratic one (MSAC, Torr and Zisserman 2000): sum over all data of $\min(d^2, \tau^2)$, with $d$ the Sampson distance and $\tau$ the inlier threshold, lower being better. A model with a few inliers fitted very well can beat one with a few more fitted loosely.

```text
function ransac_iterations(w: real, s: int, cap: int) -> int:
    p = w ^ s
    if p >= 1 - 1e-12:
        return 1
    if p <= 0:
        return cap
    n = ceil(log(0.001) / log1p(-p))
    return max(1, min(cap, n))

function ransac_essential(a: list<array<real>[3]>, b: list<array<real>[3]>, thresh: real,
                          cap: int, rng: Generator) -> list<int>:
    require len(a) >= 8 and len(a) == len(b)
    n = len(a)
    best_score = infinity
    best = []
    need = cap
    it = 0
    while it < need:
        it = it + 1
        idx = []
        while len(idx) < 8:
            c = rng.integer(n)
            if not any([c == j for j in idx]):
                append c to idx
        M = eight_point([a[i] for i in idx], [b[i] for i in idx], false)
        score = 0
        inliers = []
        for i in 0 .. n:
            d = sampson(M, a[i], b[i])
            score = score + min(d ^ 2, thresh ^ 2)
            if d < thresh:
                append i to inliers
        if score < best_score:
            best_score = score
            best = inliers
            need = min(need, ransac_iterations(len(inliers) / n, 8, cap))
    return best
```

`sampson(M, a, b)` is equation (3) in the units of the points, `log1p` and the draw `rng.integer(n)` are those of the pseudocode page, and `any` of an empty list is false. The function returns the inliers of the best hypothesis; the final fit on them is the subject of the next section. The two guards in `ransac_iterations` have a history. The Rust version of this function (`ransac_iterations` in `crates/sfm/src/lib.rs`) once computed $\ln(1 - w^s)$ by forming $1 - w^s$ first. For a model with nearly no inliers, $w^s$ is smaller than half the gap between 1 and the 64-bit number just below it, so $1 - w^s$ rounds to exactly 1, the logarithm is zero, the quotient is minus infinity, and a count clamped to at least one made the bound a single iteration: a first sample that was degenerate ended the search, and a pair whose true matches were never examined was rejected. Commit 5d1e53d2a records 264 putative matches of adjacent views in the pipeline test verified with zero inliers by this route. The fix is `log1p`, which never forms the difference, and a return of the cap for a model with no support:

<!-- worked: structure_from_motion.iteration_bound_rounding -->
The 99.9% bound for samples of 8 correspondences at very low inlier fractions w, computed two ways: ln(0.001) / ln(1 - w^8), where 1 - w^8 is first rounded to a 64-bit number, and ln(0.001) / log1p(-w^8), which never forms 1 - w^8.

| w | p = w^8 | 1 - p as stored | bound from ln(1 - p) | bound from log1p(-p) |
|---:|---:|---:|---:|---:|
| 0.5 | 0.00391 | 0.99609375 | 1765 | 1765 |
| 0.1 | 1e-08 | 0.99999998999999995 | 6.908e+08 | 6.908e+08 |
| 0.03 | 6.56e-13 | 0.99999999999934386 | 1.053e+13 | 1.053e+13 |
| 0.01 | 1e-16 | 0.99999999999999989 | 6.222e+16 | 6.908e+16 |
| 0.003 | 6.56e-21 | 1 | division by zero (ln 1 = 0) | 1.053e+21 |
| 1e-09 | 1e-72 | 1 | division by zero (ln 1 = 0) | 6.908e+72 |
<!-- /worked -->

At $w = 0.01$ the stored value of $1 - w^8$ has a deficit from 1 that is 11 percent too large, and from $w = 0.003$ down it is exactly 1 (the stored value is 1 once $w^8$ falls below $2^{-54}$, $w$ below about 0.0093); the test of that commit (`rare_inliers_keep_the_search_going`) asserts the cap for $w$ of 0, 0.05 and 0.02 at sample sizes 8 and 3, and the textbook 1765 at a half. A sibling comment in `crates/sfm/src/pnp.rs` shows how easily such arithmetic goes wrong in prose: it says that a three-point sample needs "27x" fewer samples than a six-point one at 30 percent inliers, and the ratio is $1/0.3^3 = 37$ (the draws per all-inlier sample are 37 against 1,372).

Now the run. The input is the 100 matches of the plain nearest-neighbour rule between views 0 and 1, 54 of them correct. The threshold is 1.5 pixels of Sampson distance, three standard deviations of the noise on one coordinate.

<!-- worked: structure_from_motion.two_view_ransac -->
The 100 matches of the nearest-neighbour-only rule between views 0 and 1 (the first row of the matching table) are the input; 54 are correct. The sample is 8 distinct matches, drawn with `bookrng.generator("structure_from_motion.ransac")` (seed 13165615263885262654) as `integer(100)` draws, a repeated index discarded and drawn again; a hypothesis is the rank-two matrix of the eight-point algorithm on the sample. The inlier threshold is 1.5 pixel of Sampson distance, the score is the sum over matches of min(d^2, 1.5^2), and the iteration bound is the 99.9% formula of the chapter with the inlier fraction of the best hypothesis so far (cap 5000).

Each time a sample beats the best score so far, the bound is recomputed:

| iteration | inliers of the best hypothesis | bound on the iterations still needed |
|---|---:|---:|
| 1 | 6 | 5000 |
| 3 | 8 | 5000 |
| 7 | 12 | 5000 |
| 23 | 12 | 5000 |
| 33 | 13 | 5000 |
| 45 | 41 | 5000 |
| 255 | 45 | 4105 |
| 329 | 42 | 4105 |
| 934 | 46 | 3443 |
| 1705 | 49 | 2076 |

The loop ended after 2076 iterations with 49 inliers. Their eight-point essential matrix, its pose chosen by the four-fold test and refined by Levenberg-Marquardt on the Sampson distances of the inliers, twice (the second time on the inliers of the first result), leaves 54 inliers, of which 51 are correct matches and 3 are wrong matches that lie within 1.5 pixel of an epipolar line; 3 correct matches were rejected.

| round | inliers | sum of squared Sampson distances before (px^2) | after (px^2) |
|---:|---:|---:|---:|
| 1 | 49 | 338.94 | 14.52 |
| 2 | 54 | 18.42 | 17.75 |

| estimate | rotation error (degrees) | translation direction error (degrees) |
|---|---:|---:|
| eight-point on all 100 matches | 34.83 | 144.47 |
| eight-point on the 49 RANSAC inliers | 1.79 | 6.12 |
| after refinement (the 54 inliers) | 2.52 | 5.24 |
| the same procedure on the 51 correct inliers only (needs the truth) | 0.76 | 1.31 |

Samples needed for 99.9% confidence of drawing at least one all-inlier sample, ceil(ln(0.001) / ln(1 - w^s)) for inlier fraction w and sample size s:

| w | s = 3 | s = 5 | s = 6 | s = 8 |
|---|---:|---:|---:|---:|
| 0.9 | 6 | 8 | 10 | 13 |
| 0.7 | 17 | 38 | 56 | 117 |
| 0.5 | 52 | 218 | 439 | 1765 |
| 0.3 | 253 | 2840 | 9473 | 105282 |
<!-- /worked -->

Read the table from the top. The first hypotheses explain 6 to 13 of the 100 matches; the 45th explains 41, and from there the bound starts to fall, since the best inlier fraction is about 0.45 to 0.5. The row at iteration 329 has fewer inliers (42) than the one before it (45) and is still recorded, because its truncated score is lower: it fits its inliers better. The loop finishes after 2076 iterations with 49 inliers, an inlier fraction of 0.49 and a bound of 2076, against the cap of 5000; the five-point solver, at the same fraction, would have needed $0.49^5$ in the denominator of equation (4), 242 samples. It is worth looking at what the vote found.

![The 640 by 480 image of view 1 with ten slightly sloping lines, the epipolar lines of ten keypoints of view 0, six blue for correct matches and four green for wrong ones; each blue dot, the matched keypoint, lies on its blue line, and each green dot is well away from its green line.](img/epipolar-lines.svg)

A correct match lies on its epipolar line and a wrong match is somewhere else. The figure is the epipolar constraint turned into a picture, and it also shows the limit of the test: the constraint is one-dimensional, so a wrong match whose keypoint happens to lie within a pixel or two of the line passes it. The table says three of them did.

## Refining a model that cannot be solved for

The linear estimate on 49 inliers, the second row of the table of estimates, is 1.8 degrees from the true rotation and 6.1 degrees from the true direction of translation: good enough to start, not good enough to keep. Two things are wrong with it. The eight-point algorithm minimizes an algebraic quantity, and the projection to the essential matrices is a crude step in the Frobenius norm. What the data call for is the pose that minimizes the sum of squared Sampson distances, a sum of squares of nonlinear functions of five parameters. This is a **nonlinear least-squares** problem, and it is the first of three in this chapter; the method that solves one is the same each time, so it is worth deriving once.

Let $\mathbf{r}(\mathbf{x})$ be a vector of $k$ residuals that depend on $m$ parameters $\mathbf{x}$, and let the cost be $\lVert\mathbf{r}(\mathbf{x})\rVert^2$. Near a point $\mathbf{x}$ the residual is, to first order (the Taylor expansion of Volume I, Chapter 8), $\mathbf{r}(\mathbf{x} + \boldsymbol{\delta}) \approx \mathbf{r} + \mathbf{J}\boldsymbol{\delta}$, with $\mathbf{J}$ the $k\times m$ Jacobian of the residuals. Replacing the residual by this line turns the cost into $\lVert\mathbf{r} + \mathbf{J}\boldsymbol{\delta}\rVert^2$, a linear least-squares problem in $\boldsymbol{\delta}$ that question 3 of the opening solves by the normal equations,

$$
\mathbf{J}^\top\mathbf{J}\,\boldsymbol{\delta} = -\mathbf{J}^\top\mathbf{r} . \tag{5}
$$

Solving this and stepping is the **Gauss-Newton** method. It is Newton's method of Volume I, Chapter 7 with one economy: the second derivatives of the cost contain a term in the second derivatives of the residuals, which Gauss-Newton drops, because when the residuals are small near the solution the term is too. It converges fast when the linearization is good and fails when it is not: a step computed from a line may be far too long for the curve, and the cost can rise. Gradient descent (question 4) has the opposite character. It steps down the gradient $\mathbf{J}^\top\mathbf{r}$ with a size that must stay below $2/\lambda_{\max}$ of $\mathbf{J}^\top\mathbf{J}$ to be stable, and is slow along the flat directions. The method of Levenberg and Marquardt interpolates: add $\lambda$ times the diagonal of $\mathbf{J}^\top\mathbf{J}$ to the matrix,

$$
\bigl(\mathbf{J}^\top\mathbf{J} + \lambda\,\mathrm{diag}(\mathbf{J}^\top\mathbf{J})\bigr)\,\boldsymbol{\delta} = -\mathbf{J}^\top\mathbf{r} . \tag{6}
$$

For $\lambda = 0$ this is Gauss-Newton; for large $\lambda$ the matrix is nearly diagonal and the step is a gradient step with a per-parameter step size $1/(\lambda\,[\mathbf{J}^\top\mathbf{J}]_{ii})$, scaled to the curvature of each parameter. The damping is adjusted by what the step does: a step that lowers the cost is accepted and $\lambda$ is divided by 3, so that the next step trusts the line more; a step that does not is thrown away and $\lambda$ is multiplied by 5, which both shortens the step and turns it toward the gradient, and the step is recomputed from the same $\mathbf{J}$.

```text
function levenberg_marquardt(x0: array<real>[m], residual: function(array<real>[m]) -> array<real>[k],
                             jacobian: function(array<real>[m]) -> array<real>[k, m],
                             iters: int, lam0: real) -> array<real>[m]:
    x = copy(x0)
    lam = lam0
    r = residual(x)
    cost = dot(r, r)
    for it in 0 .. iters:
        J = jacobian(x)
        H = matmul(transpose(J), J)
        g = matmul(transpose(J), r)
        accepted = false
        for attempt in 0 .. 8:
            D = copy(H)
            for i in 0 .. m:
                D[i, i] = D[i, i] + lam * max(H[i, i], 1e-9)
            step = solve(D, -g, true)
            r_new = residual(x + step)
            cost_new = dot(r_new, r_new)
            if cost_new < cost:
                x = x + step
                r = r_new
                cost = cost_new
                lam = max(lam / 3, 1e-9)
                accepted = true
                break
            lam = lam * 5
        if not accepted:
            return x
    return x
```

For the essential matrix the parameters are not the nine entries of $\mathbf{E}$, since a step in those leaves the manifold of essential matrices at once, as the rotation matrix of chapter 3 does under gradient descent. They are an increment of the pose: a rotation vector $\boldsymbol{\omega}$ applied as $\mathbf{R}\leftarrow\exp(\boldsymbol\omega)\mathbf{R}$ (Rodrigues' formula, equation (4) of chapter 3) and two steps of the unit translation in the plane across it, $\mathbf{t}\leftarrow(\mathbf{t} + \delta_1\mathbf{e}_1 + \delta_2\mathbf{e}_2)/\lVert\cdot\rVert$. Five parameters, and every iterate an exact essential matrix. `refine_essential` in `crates/sfm/src/twoview.rs` is this, with a Jacobian by central differences at a step of $10^{-7}$, damping starting at $10^{-4}$, and the schedule above of 0.3 and 10; the loop is run twice in the example, the second time on the inliers of the first result.

In the table of the two rounds the sum of squared Sampson distances falls from 338.94 to 14.52 pixels squared over the 49 inliers of the first round, and from 18.42 to 17.75 over the 54 inliers of the second, where the refit found five more. The estimates table gives the rotation error after refinement as 2.52 degrees and the direction error as 5.24, worse in rotation than the linear estimate on the 49. The reason is in the inlier count: three of the 54 are the wrong matches that sat on the epipolar lines. The same procedure on the 51 correct inliers alone, which only a scene with a known answer can supply, gives 0.76 and 1.31 degrees. Three bad matches out of 54, each less than 1.5 pixels from a line they should not be on, tilt the rotation by about 2 degrees and the direction of translation by about 4, and no test that uses two images can see them. The remedy is a third image, and it is what the rest of the chapter is.

## Four answers and one that is in front

The pair's pose is hidden in $\mathbf{E}$, and it comes out of its singular value decomposition $\mathbf{E} = \mathbf{U}\,\mathrm{diag}(1, 1, 0)\,\mathbf{V}^\top$, with the signs of $\mathbf{U}$ and $\mathbf{V}$ chosen so that both have determinant $+1$. With the rotation by a quarter turn $\mathbf{W} = \begin{pmatrix}0&-1&0\\1&0&0\\0&0&1\end{pmatrix}$ the solutions are

$$
\mathbf{R} \in \{\mathbf{U}\mathbf{W}\mathbf{V}^\top,\ \mathbf{U}\mathbf{W}^\top\mathbf{V}^\top\}, \qquad \mathbf{t} = \pm\mathbf{u}_3 , \tag{7}
$$

where $\mathbf{u}_3$ is the last column of $\mathbf{U}$, the direction that $[\mathbf{t}]_\times$ annihilates. The two rotations differ by a half turn about the baseline, and the two signs of $\mathbf{t}$ put the second camera on either side of the first: four candidates, all satisfying equation (2) for every correspondence, because $\mathbf{E}$ and $-\mathbf{E}$ are the same constraint. They cannot be told apart by any number of epipolar residuals. What separates them is that a point has to be in front of both cameras. Triangulating the inliers under each candidate and counting those with positive depth in both:

<!-- worked: structure_from_motion.pose_ambiguity -->
The 49 RANSAC inliers triangulated under each of the four decompositions of their eight-point essential matrix: how many land in front of both cameras, and how far each candidate is from the true rotation and translation direction (degrees).

| candidate (R, t) | rotation error | direction error | in front of both | behind one or both |  |
|---|---:|---:|---:|---:|---|
| U W V^T, +u3 | 1.79 | 6.12 | 47 | 2 | chosen |
| U W V^T, -u3 | 1.79 | 173.88 | 2 | 47 |  |
| U W^T V^T, +u3 | 179.81 | 6.12 | 0 | 49 |  |
| U W^T V^T, -u3 | 179.81 | 173.88 | 0 | 49 |  |
<!-- /worked -->

![Four schematic top views, two by two. In each, camera 0 and camera 1 are drawn as small boxes with an arrow for the optical axis and dashed lines to a triangulated point. In the first panel both arrows point up toward the point; the other three show camera 1 on the other side, or facing away, or the point behind one or both cameras.](img/four-poses.svg)

One candidate puts 47 of the 49 points in front of both cameras, the others 2, 0 and 0. The figure draws the four from the signs of the actual decompositions: which side of camera 0 the second camera stands on, which way it faces, and where the points land. It is a schematic and not to scale. The test is called the cheirality or chirality test, and `relative_pose` in `crates/sfm/src/twoview.rs` applies it to the four decompositions, keeping the one with the most points ahead of both. The two that fail in the winning row are points whose matches are wrong or whose triangulated depth is too noisy to be positive.

What the pose does not contain is a length. The translation comes out with $\lVert\mathbf{t}\rVert = 1$, and the points triangulated with it are in units of the baseline: the scene could be twice as large and the cameras twice as far apart without changing a pixel. This is the freedom that equation (2) already hid. The section on the gauge returns to it, and the first consequence is a choice: the first pair fixes the unit.

## A third camera

Two views give a pose up to scale; a third gives something better, since its relation to the points is fully metric (in the scale of the pair). The 3D points triangulated from the first pair are known; the new photograph shows some of them. Finding where the camera must have stood to see them there is the **absolute pose** problem, or perspective-n-point: given 3D points $\mathbf{X}_i$ and their normalized images $\mathbf{x}_i$, find $(\mathbf{R}, \mathbf{t})$. Equation (1) is the same equation read for a different unknown. For the camera matrix $\mathbf{P}$ of unknown rows, $x_i\,\mathbf{p}_3\cdot\tilde{\mathbf{X}}_i - \mathbf{p}_1\cdot\tilde{\mathbf{X}}_i = 0$ and its twin in $y_i$ are linear in the twelve entries of $\mathbf{P}$, two rows per point, the null vector of a $2n\times12$ matrix. Six points suffice for the eleven degrees of freedom. The left $3\times3$ block of the solution is a scaled rotation, which the nearest-rotation step of chapter 3 makes exact, and the translation is divided by the same scale. This is the direct linear transform of chapter 3 once more, with the 3D points normalized by the analogue of Hartley's similarity.

Where do the 2D-3D correspondences come from? From matching, again, and again with errors. In the example, each inlier match of the first pair gives a 3D point, and its view-0 descriptor is matched to the keypoints of the new view by plain nearest neighbour, so that about one in ten is wrong. The same RANSAC wraps the six-point solver, with the reprojection error in pixels in place of the Sampson distance:

<!-- worked: structure_from_motion.absolute_pose -->
For camera v = 2 and 3, each of the inlier matches of the pair (0, 1) gives a 3D point (triangulated with the refined pose, baseline 1) and a 2D keypoint: the nearest neighbour, among the keypoints of view v, of the match's view-0 descriptor. The RANSAC sample is 6 distinct correspondences from `integer(n)` draws on `bookrng.generator("structure_from_motion.registration.v")` (v = 2: seed 14707663108737989869), a hypothesis is the DLT pose of the sample, and the threshold is 4 pixels of reprojection error with the same truncated score and adaptive bound as before (cap 2000).

| camera | correspondences | correct | bound at the true fraction | iterations run | inliers | correct inliers | rotation error (degrees) | direction error (degrees) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 54 | 49 | 9 | 612 | 42 | 42 | 4.90 | 3.42 |
| 3 | 54 | 48 | 11 | 1081 | 26 | 26 | 8.00 | 3.97 |

Smallest singular values of the 16 by 12 matrix of the linear absolute-pose system, 8 points, last three:

| the 8 points are | sigma_12 | sigma_11 | sigma_10 |
|---|---:|---:|---:|
| 8 points of the scene | 1.1e-16 | 1.2e-01 | 3.4e-01 |
| the same 8 pixels' rays cut by the plane Z = 5.5 | 0.0e+00 | 0.0e+00 | 0.0e+00 |
<!-- /worked -->

The loops ran 612 and 1081 iterations against bounds of 9 and 11 that the true inlier fractions would allow: the minimal six-point DLT hypotheses are poor, because the 3D points they are fitted to carry the error of the pair, and the best of them explains 42 and 26 of the 54 at a threshold of 4 pixels, so the bound never gets short. The RANSAC poses are off by 4.9 and 8.0 degrees in rotation, and refining each against its inlier points by Levenberg-Marquardt, as the pipeline does, leaves 5.2 and 7.8. That is not an error of the algorithm: it is the structure of the pair, distorted by the 2.5-degree error of the last section, being registered faithfully. This is what bundle adjustment is for.

The second table is a different kind of fact. A linear solve for $\mathbf{P}$ needs the points not to lie on a plane: for coplanar points the last three singular values of the matrix are all zero and the solution is a family, not a point. The singular values of the $16\times12$ matrix for eight scene points in general position are $10^{-16}$, 0.12, 0.34; for the same eight rays cut by a plane they are 0, 0, 0. A capture of an object that stands on a floor or a table has most of its points on that plane, so the six-point solver is the wrong default for it. Commit aa05bdda7 (September 2026) records the failure on a real capture of sixteen photographs of an object on a wooden deck: the first version registered 11 of the 16 views and then, with a different focal length guess, 2; one of its three causes was the six-point DLT, "DEGENERATE for coplanar points", and its other cost was the count, a six-point sample at 30 percent inliers being clean once in 1,372 draws. It was replaced by P3P, the three-point solution of 1841: the law of cosines on the three triangles formed by the camera centre and each pair of points gives three equations in the three unknown depths along the rays, which reduce to one in a ratio of depths (`p3p` in `crates/sfm/src/pnp.rs`). A sample of three needs fewer draws, tolerates a plane, and returns up to four candidate poses.

Registration is a loop (`Solver::register_all` in `crates/sfm/src/incremental.rs`): choose the unregistered photograph that sees the most already triangulated points, register it by RANSAC and P3P, give each of its inlier keypoints to its point, triangulate the tracks that two registered views now see with enough angle, adjust everything, and start again, until no photograph has the 20 correspondences (`min_register`) it needs. A photograph that fails is retried after the next success, since more of the scene may exist by then. The beginning is the seed pair: the 30 pairs with the most verified inliers are each solved by RANSAC, and the one that maximizes the number of points times the smaller of 1 and the median triangulation angle over 10 degrees wins, provided that angle is at least 3 degrees. The same triangulation figure is why.

## Everything at once

The poses of the third and fourth cameras, 5.2 and 7.8 degrees off, and the three wrong points all trace to the same cause. Each stage fitted what it was given as if that were exact, and passed on an error that the next stage took for truth. The antidote is to stop treating the earlier answers as data and to ask the question that all of them were approximating: which cameras and which points make the observed pixels most probable?

If the noise on each pixel is Gaussian with the same variance, the most probable reconstruction minimizes the squared **reprojection error**, the sum over all observations of the squared distance in pixels between where a point projects and where it was seen, over every camera pose and every point at once. This is **bundle adjustment**, named for the bundles of rays through every camera centre that it adjusts together:

$$
\min_{\{\mathbf{R}_i, \mathbf{t}_i\},\,\{\mathbf{X}_j\}}\ \sum_{(i, j)\in\mathcal{O}}\rho\Bigl(\bigl\lVert\pi(\mathbf{R}_i\mathbf{X}_j + \mathbf{t}_i) - \mathbf{u}_{ij}\bigr\rVert\Bigr), \tag{8}
$$

where $\mathcal{O}$ is the set of observations, $\pi$ the projection through the camera's intrinsics, $\mathbf{u}_{ij}$ the observed pixel, and $\rho$ a loss on the distance. For $\rho(e) = e^2$ it is the least squares of the opening questions; a few wrong observations then pull with a force that grows with their error, and the Huber loss $\rho(e) = e^2$ for $e \le \delta$ and $2\delta e - \delta^2$ above limits that force to a constant. `crates/sfm/src/ba.rs` uses it with $\delta = 2$ pixels by iteratively reweighted least squares: each observation's contribution to the normal equations is multiplied by $\delta/e$ when $e > \delta$. The unknowns are what the section on Levenberg and Marquardt called $\mathbf{x}$, and the residuals are the two pixel differences of each observation; what is new is the size, and so the structure.

For an observation, write the camera-frame point $\mathbf{c} = \mathbf{R}\mathbf{X} + \mathbf{t} = (c_1, c_2, c_3)$. The projection $\pi(\mathbf{c}) = (f c_1/c_3 + c_x,\ f c_2/c_3 + c_y)$ has the $2\times3$ Jacobian

$$
\mathbf{J}_\pi = \begin{pmatrix} f/c_3 & 0 & -f c_1/c_3^2 \\ 0 & f/c_3 & -f c_2/c_3^2 \end{pmatrix}, \tag{9}
$$

and the chain rule gives the rest. With respect to the point it is $\mathbf{J}_\pi\mathbf{R}$; with respect to a translation increment $\boldsymbol\tau$ it is $\mathbf{J}_\pi$; with respect to a left rotation increment $\boldsymbol\omega$ ($\mathbf{R}\leftarrow\exp(\boldsymbol\omega)\mathbf{R}$) the camera point moves by $\boldsymbol\omega\times(\mathbf{c} - \mathbf{t})$, so the block is $\mathbf{J}_\pi\,(-[\mathbf{c} - \mathbf{t}]_\times)$; and with respect to a focal length shared by all cameras, $(c_1/c_3, c_2/c_3)$. An observation touches one camera and one point, nothing else: its two rows of the Jacobian have nonzeros in six camera columns and three point columns (plus one focal column, if the focal length is free) and zeros elsewhere.

<!-- worked: structure_from_motion.bundle_adjustment -->
The problem after registration: 52 points and 172 observations in 4 cameras, of which 2 are wrong associations; camera 0 is the anchor (held), camera 1 the scale camera (5 columns, baseline 1), cameras 2 and 3 free (6 columns each), so the camera system has 17 unknowns and the points 156. Huber threshold delta = 2 pixels, lambda starts at 1e-3; an iteration tries the step, accepts it when the robust cost falls (lambda divided by 3) and otherwise multiplies lambda by 5.

Before the adjustment the root mean square reprojection error is 0.6255 px and the robust cost 67.2918. The first damped step computed from the dense system of 173 unknowns and from the Schur complement of the 17 camera unknowns differ by at most 2.4e-14 in any entry (the largest camera step is 0.066).

| iteration | tries | lambda after | robust cost (px^2) | RMS reprojection (px) |
|---:|---:|---:|---:|---:|
| 0 |  | 1e-03 | 67.2918 | 0.6255 |
| 1 | 1 | 3.33e-04 | 59.9579 | 0.5904 |
| 2 | 1 | 1.11e-04 | 58.4517 | 0.5830 |
| 3 | 2 | 1.85e-04 | 57.3580 | 0.5775 |
| 4 | 1 | 6.17e-05 | 56.0122 | 0.5707 |
| 5 | 2 | 1.03e-04 | 53.5442 | 0.5579 |
| 6 | 2 | 1.71e-04 | 52.5885 | 0.5529 |
| 7 | 1 | 5.72e-05 | 51.4651 | 0.5470 |
| 28 | 1 | 5.34e-08 | 38.1221 | 0.4708 |
(iterations 8 to 27 omitted; the loop stops at iteration 28, when the gain is below 1e-10 of the cost.)

The noise floor: with sigma = 0.5 pixel per coordinate, 172 observations and 173 parameters the expected RMS error of the fit is sigma sqrt(2) sqrt((2 172 - 173) / (2 172)) = 0.4985 px; the adjustment reaches 0.4708.

| camera | rotation error before (deg) | after | centre error before (m, scaled to the true baseline) | after |
|---:|---:|---:|---:|---:|
| 1 | 2.52 | 0.52 | 0.034 | 0.025 |
| 2 | 5.16 | 0.60 | 0.083 | 0.047 |
| 3 | 7.84 | 0.78 | 0.407 | 0.083 |

Started at the true poses and points instead, with the same observations, the same loop reaches the cost 38.1221 against 38.1221, and the largest difference between any pose entry of the two solutions is 5.7e-08.
<!-- /worked -->

The scene's 52 points and 172 observations (two are wrong associations that survived registration) give 344 residuals and 173 unknowns, and the Jacobian has 2368 nonzero entries of 59,512, 4.0 percent. This is the **sparse structure** of bundle adjustment: a matrix whose rows are observations and whose columns are camera blocks followed by point blocks, with two blocks per row. The matrix that the normal equations (5) and (6) factor, $\mathbf{J}^\top\mathbf{J}$, inherits the pattern.

![Three matrix patterns for three cameras and four points with ten observations. The Jacobian is 20 rows by 30 columns with one 2-row camera block and one 2-row point block per observation. The normal-equations matrix has six-by-six blocks on the diagonal for the cameras, three-by-three blocks on the diagonal for the points, and off-diagonal blocks only where a camera sees a point. The reduced camera system is a full 18 by 18 matrix.](img/ba-structure.svg)

The camera blocks do not couple to one another directly, nor the point blocks: two cameras are linked only through the points that both see, a point and a camera only if it is observed. Order the unknowns cameras first and points last and the system is

$$
\begin{pmatrix}\mathbf{U} & \mathbf{W} \\ \mathbf{W}^\top & \mathbf{V}\end{pmatrix}\begin{pmatrix}\boldsymbol\delta_c \\ \boldsymbol\delta_p\end{pmatrix} = -\begin{pmatrix}\mathbf{g}_c \\ \mathbf{g}_p\end{pmatrix}, \tag{10}
$$

with $\mathbf{U}$ block diagonal over cameras, $\mathbf{V}$ block diagonal over points, $3\times3$ blocks $\mathbf{V}_j$, and $\mathbf{W}$ nonzero where a camera sees a point. The damping adds $\lambda$ times the diagonal to $\mathbf{U}$ and to each $\mathbf{V}_j$. The point blocks are invertible one at a time, three by three, at trivial cost, so eliminate the points: the second block row gives $\boldsymbol\delta_p = -\mathbf{V}^{-1}(\mathbf{g}_p + \mathbf{W}^\top\boldsymbol\delta_c)$, and substituting into the first,

$$
\bigl(\mathbf{U} - \mathbf{W}\mathbf{V}^{-1}\mathbf{W}^\top\bigr)\,\boldsymbol\delta_c = -\mathbf{g}_c + \mathbf{W}\mathbf{V}^{-1}\mathbf{g}_p . \tag{11}
$$

The matrix on the left is the **Schur complement** of $\mathbf{V}$, the third panel of the figure: dense, since two cameras that share a point are linked by it, but only as large as the cameras. Solve it, then recover each point's step from its own $3\times3$ block. The cost is exactly the cost of a system the size of the cameras plus work per point proportional to the square of the number of its observations: to form $\mathbf{V}_j^{-1}\mathbf{W}_{aj}^\top$ for each of $k$ observations of a point costs 54 multiply-adds (a $3\times3$ block times a $3\times6$) and the update of the reduced matrix 108 per pair of its observations.

```text
function schur_step(U: array<real>[c, c], gc: array<real>[c], V: list<array<real>[3, 3]>,
                    gp: list<array<real>[3]>, W: list<array<real>[c, 3]>, lam: real)
        -> (array<real>[c], list<array<real>[3]>):
    S = copy(U)
    rhs = -gc
    for a in 0 .. c:
        S[a, a] = S[a, a] + lam * max(U[a, a], 1e-9)
    Vinv = []
    for j in 0 .. len(V):
        D = copy(V[j])
        for a in 0 .. 3:
            D[a, a] = D[a, a] + lam * max(V[j][a, a], 1e-9)
        append inverse(D) to Vinv
        S = S - matmul(W[j], matmul(Vinv[j], transpose(W[j])))
        rhs = rhs + matmul(W[j], matmul(Vinv[j], gp[j]))
    dc = solve(S, rhs, true)
    dp = [-matmul(Vinv[j], gp[j] + matmul(transpose(W[j]), dc)) for j in 0 .. len(V)]
    return (dc, dp)
```

Here `W[j]` is the $c\times3$ coupling of point $j$ to all camera columns, the sum of the couplings of its observations placed in the columns of their cameras; `U`, `gc`, `V`, `gp` are the blocks of equation (10) summed over observations, and `inverse` and `solve` are those of Volume I, Chapter 4. The step the function returns is the step that the full system would give, which the table checks: the first damped step from the dense system of 173 unknowns and from the 17-unknown Schur complement differ by at most $2.4\times10^{-14}$ in any entry.

The loop on this problem starts at an error of 0.6255 pixel (the points were triangulated from all their views with the poses of the previous section, so the cost is small and the poses are wrong) and ends, after 28 iterations in which the damping falls from $10^{-3}$ to $5\times10^{-8}$, at 0.4708. The noise floor tells what to expect: with 172 observations and 173 parameters, a fit of noise of 0.5 pixel per coordinate has an RMS error of $\sigma\sqrt2\sqrt{(2n - p)/(2n)} = 0.4985$ pixel, and the adjustment lands a little below it, as a fit that has absorbed some noise should. The three camera poses move from 2.5, 5.2 and 7.8 degrees of rotation error to 0.5, 0.6 and 0.8, and the camera centres, scaled to the true baseline, from 0.034, 0.083 and 0.407 metres to 0.025, 0.047 and 0.083. The fit is at the noise floor, and what remains of the pose error is what half a pixel of noise leaves through a 0.6-metre baseline. Whether it is the optimum or only a good stopping point is checkable: started at the true poses and points, the same loop reaches a cost of 38.1221, the same as from the incremental start, and the two solutions differ in no pose entry by more than $6\times10^{-8}$. There is one minimum here, and the pipeline found it.

The work counts are the argument for the elimination. A dense factorization of $n$ unknowns costs about $n^3/6$ multiply-adds and $8n^2$ bytes; the reduced system costs the same for its $c$ camera unknowns plus the per-point work above:

<!-- worked: structure_from_motion.ba_structure -->
Structure of the problem of the previous table. 172 observations give 344 residuals; the unknowns are the 17 camera unknowns and 156 point coordinates, 173 columns. Each observation touches the columns of its camera (0 for camera 0, 5 for camera 1, 6 for cameras 2 and 3) and the 3 of its point, so the Jacobian has 2368 nonzero entries of 59512, 4.0%. The numbers of observations of cameras 0 to 3 are 52, 52, 42, 26.

Singular values of the residual Jacobian at the true parameters, noise-free observations of the first 10 points in all 4 cameras (80 residuals): the eight smallest, and the rank (singular values above 1e-9 times the largest):

| parametrization | columns | rank | missing | eight smallest singular values |
|---|---:|---:|---:|---|
| all four cameras free (6 columns each) | 54 | 47 | 7 | 4.6e+00, 7.6e-14, 6.4e-14, 5.8e-14, 3.2e-14, 2.7e-14, 2.5e-14, 1.9e-14 |
| camera 0 held, camera 1 across the baseline | 47 | 47 | 0 | 2.9e+01, 2.7e+01, 2.3e+01, 1.9e+01, 1.7e+01, 1.6e+01, 7.2e+00, 7.1e-01 |

Work of one step, in multiply-adds (dense: Cholesky of all n unknowns, n^3 / 6; Schur: 54 k + 108 k^2 per point with k observations, here k = 3, plus the Cholesky of the nc camera unknowns), and the bytes of the matrix that is factored (8 bytes per entry; the Schur column adds the 3 by 3 point blocks):

| problem | cameras | points | n | nc | dense multiply-adds | Schur multiply-adds | ratio | dense matrix bytes | Schur bytes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| this chapter's problem | 4 | 52 | 173 | 17 | 8.63e+05 | 5.98e+04 | 14 | 2.39e+05 | 6.06e+03 |
| 16 photographs, 8,792 points | 16 | 8792 | 26465 | 89 | 3.09e+12 | 1.01e+07 | 306250 | 5.6e+09 | 6.96e+05 |
<!-- /worked -->

For the chapter's four cameras the elimination saves a factor of 14. For the capture of commit aa05bdda7, sixteen photographs and 8,792 points (and an assumption of three observations a point), the dense system has 26,465 unknowns, 5.6 gigabytes for the matrix and about $3\times10^{12}$ multiply-adds a step; the reduced one has 89 camera unknowns, 0.7 megabyte, and $10^7$ multiply-adds, a factor of 306,250. `bundle_adjust` in `crates/sfm/src/ba.rs` solves exactly this system: each camera owns its rows of the reduced matrix and the rows for the shared calibration are summed in chunks in a fixed order, and the dense Cholesky factorization is split into panels for the CPU pool once the system is large (`cholesky_solve` in `crates/sfm/src/linalg.rs`). The change that put the reduced system on the pool, commit 16a7847cd, records that a capture of 251 photographs spent most of its structure-from-motion time in a chain of bundle adjustments whose linearization, Schur complement and factorization ran on one thread; knowledge #430 has the profile.

## Seven directions the photographs cannot see

Something in the Jacobian rank test of the table above was easy to miss: with all four cameras free, a problem of 54 unknowns has rank 47. Seven directions in the space of parameters do not change a single reprojection. This is the **gauge freedom** of structure from motion, and its dimension is the dimension of the group of similarities of three-dimensional space: three for a translation, three for a rotation, one for the scale. Moving the whole scene by a rigid motion and scaling it, and moving every camera with it, changes no pixel.

<!-- worked: structure_from_motion.gauge_freedom -->
The solution of the previous section carried into another frame: every point X becomes 2.5 R_g X + t_g, every camera centre likewise and every rotation R becomes R R_g^T, with R_g the rotation vector (0.3, -0.5, 0.2) (radians) and t_g = (1, -2, 3).

| quantity | solution | same solution in the new gauge |
|---|---:|---:|
| RMS reprojection (px) | 0.4707869755 | 0.4707869755 |
| largest change of any residual component (px) |  | 4.0e-13 |
| distance between camera centres 0 and 1 | 1.000000 | 2.500000 |
| camera 0 pose | identity, centre (0, 0, 0) | centre (1, -2, 3) |
<!-- /worked -->

The adjusted reconstruction carried into a frame with the scale 2.5, a rotation and a translation has the same RMS error to ten digits and no residual component changes by more than $4\times10^{-13}$ pixel. Equation (2) already lost the scale of the baseline for a pair, and a reconstruction of any number of photographs loses that scale and the six numbers of the world frame, seven in all, never more and never fewer: no amount of data from photographs alone recovers them. The rank table gives the count, 47 of 54 with the six pose columns of every camera and 47 of 47 once seven columns are gone. A rank-deficient normal-equations matrix is not a disaster for Levenberg-Marquardt, since the damping makes the system solvable, but the solution then drifts along the directions that it cannot see, an accumulation of arbitrary motion that looks like an error and is not one.

The repair is to remove the seven directions, not to let the damping hide them. In `crates/sfm/src/ba.rs` the anchor camera's pose is not a parameter at all (six directions), and the scale camera's centre is allowed to move only across the line from the anchor's centre (five columns instead of six), and after each accepted step the whole reconstruction is rescaled about the anchor so that the distance between the two centres is exactly what it was: the tangent step keeps it to first order, and the rescale, a similarity that no reprojection can see, keeps it to rounding. The test `the_gauge_holds_the_anchor_and_the_scale` asserts that the anchor is bitwise unchanged and the distance holds to $10^{-9}$ of itself. The worked problem used exactly this choice, with the pair's first camera as the world and the baseline of the pair as the unit. Photographs with satellite positions in their metadata can name the gauge by the world instead: `Gauge::Enu` in `crates/sfm/src/incremental.rs` is the gauge of metres east, north and up of a named origin, fitted by a robust similarity on the camera centres (`crates/sfm/src/georef.rs`) and refused when the fixes are as coarse as the capture; `Gauge::Seed` is the pair's.

A gauge is not only a nuisance inside one problem. Whenever two reconstructions of the same place must be put together, they are related by exactly the freedom that was fixed, a similarity. `crates/recon` meets this when one capture is too long for one pass of a learned model and is cut into overlapping chunks, each returned in its own world (the next chapter). The shared cameras register the chunks: `sim3_from_cameras` in `crates/splat/src/align.rs` finds the rotation from the cameras' axes (each shared camera gives three axis pairs, which fix the rotation even when the camera centres are nearly collinear), the scale from the median of the ratios of distances between pairs of shared centres, and the translation from the centroids. The camera residual, relative to the spread of the shared centres, is gated at 0.1 in `ReconstructOpts::max_relative_residual`, because a similarity is solved whether or not it fits: the failure is a scene with two displaced copies of one wall, which nothing downstream can tell from a wall that is there twice (commit 84521a0f2).

<!-- worked: structure_from_motion.chunk_registration -->
Chunk A holds the adjusted cameras 0 to 2 in their own world; chunk B holds cameras 1 to 3 carried into another gauge (scale 2.5, rotation vector (0.3, -0.5, 0.2), translation (1, -2, 3)). They share cameras 1 and 2. The similarity from B to A is the rotation from the cameras' axes, the scale from the median ratio of their centre distances and the translation from the centroids; the residual is the largest distance between a shared centre in A and the registered one from B, relative to the distance between the shared centres (the span); the gate is 0.1.

| chunk B | scale found (true: 0.4) | largest centre residual (A's units) | relative to the span | accepted (gate 0.1) |
|---|---:|---:|---:|---|
| chunk B as solved | 0.4000 | 6.52e-16 | 0.0000 | yes |
| camera 2 of chunk B displaced by 15% of the span | 0.3956 | 9.26e-02 | 0.0744 | yes |
| camera 2 of chunk B displaced by 40% of the span | 0.3714 | 2.36e-01 | 0.1891 | no, refused |
<!-- /worked -->

Chunk B was solved in another gauge, and the similarity recovers the scale 1/2.5 and registers its two shared cameras to $10^{-16}$. Displace one of the shared cameras by 15 percent of the span and the relative residual is 0.074, below the gate; by 40 percent, 0.189, and the chunk is refused. Two shared cameras always fit their centres exactly by a similarity, so what the residual here measures is the disagreement between the cameras' centres and their headings, and a third shared camera would measure more.

## A focal length nobody wrote down

The previous chapter ended on a worry, that a focal length wrong by ten percent produces a projectively distorted scene onto which later photographs stop registering, and that bundle adjustment creeps out of the distortion by about one percent a round (knowledge #159). The synthetic scene can say how true that is. The inlier matches are turned into rays through a wrong focal length, the pair is estimated and triangulated, and the third and fourth cameras are registered against it, each through the same wrong focal length.

<!-- worked: structure_from_motion.focal_length_effect -->
The inlier matches that are correct (51 of the pair's 54) are turned into rays through a focal length f_given, the pair is estimated and triangulated as before, and cameras 2 and 3 are registered against the resulting points (DLT, then Levenberg-Marquardt on the pose alone, projecting through f_given). RMS columns are in pixels.

| f_given | pair: RMS Sampson distance | rotation error (deg) | direction error (deg) | camera 2: RMS reprojection | camera 3: RMS reprojection |
|---:|---:|---:|---:|---:|---:|
| 400 | 0.458 | 0.54 | 1.56 | 2.07 | 3.62 |
| 450 | 0.458 | 0.65 | 1.31 | 1.61 | 2.47 |
| 500 | 0.459 | 0.76 | 1.31 | 1.52 | 2.11 |
| 550 | 0.460 | 0.87 | 1.58 | 1.63 | 2.32 |
| 600 | 0.461 | 0.98 | 2.05 | 1.80 | 2.72 |

The adjustment of the previous sections with the focal length as one more unknown (a column shared by every observation), started at the wrong value:

| f_given | RMS before | RMS after | focal length found | iterations |
|---:|---:|---:|---:|---:|
| 450 | 11.669 | 0.4708 | 499.98 | 29 |
| 550 | 11.665 | 0.4708 | 499.98 | 29 |

And the adjustment with the focal length held at each value (all poses and points free): the robust cost and the RMS error of the best reconstruction through it:

| f held | robust cost (px^2) | RMS (px) |
|---:|---:|---:|
| 400 | 51.716 | 0.5483 |
| 450 | 40.570 | 0.4857 |
| 480 | 38.450 | 0.4728 |
| 500 | 38.122 | 0.4708 |
| 520 | 38.387 | 0.4724 |
| 550 | 39.543 | 0.4795 |
| 600 | 42.633 | 0.4979 |
<!-- /worked -->

The first table is the finding. The pair cannot tell: its RMS Sampson distance is 0.458 pixel at 400, 0.459 at 500 and 0.461 at 600, a thousandth of a pixel, because two views always have a pose that explains them through any focal length in this range. The later views can. Camera 3 registered against the structure built at the true 500 has an RMS reprojection error of 2.11 pixels; at 400 it is 3.62, at 450 2.47, at 550 2.32. It is a mild effect in this scene (the cameras are close together and the depth range is narrow), and in a capture with a wide baseline it grows. The second table shows the way out, which is the sense of the phrase "self-calibration". Release the focal length as one more unknown of the bundle adjustment, a column that every observation touches, and the adjustment started at 450 or at 550 reaches an RMS error of 0.4708 and a focal length of 499.98 in 29 iterations. The third table is the profile of the cost along the focal length, with everything else optimal at each value: 38.12 at 500, 38.45 at 480, 38.39 at 520, 40.57 at 450 and 42.63 at 600. It is a shallow valley, a rise of about one robust unit for a change of 5 percent, which is why a focal length is among the first quantities a method must hold with a prior when the evidence is thin.

`crates/sfm/src/incremental.rs` does what the tables suggest, in stages. A sensor with a focal length in its metadata starts from it and bundle adjustment holds it with a soft Gaussian prior of 5 percent (`focal_prior`), together with a prior that keeps the principal point within 2 percent of the longer side of the image centre (`pp_prior`). A sensor with no metadata is calibrated by a sweep: the whole capture is solved under each of 19 candidate focal lengths, spaced geometrically from 0.3 to 1.3 times the longer side of the image, each held fixed, and the one that registers the most views at the lowest reprojection error is kept. The sweep is the lesson of knowledge #159 in code: scoring the candidates by epipolar inliers over the best pairs, the first idea, chose the bottom of its range on a real capture, 0.45 of the long side, while solving the whole capture under each candidate chose 0.69, with all 16 views registered. Knowledge #163 is the second lesson, that a search whose winner is at the end of its range has found a bound and not a value: the range then was 0.5 to 1.3 times the long side, and an ultra-wide phone camera chose its lower edge exactly and was walked outside it by the refinement; the range now reaches 0.3, and a winner on either edge is logged as such (`AT THE EDGE of the sweep`). After the sweep, bundle adjustment frees the calibration in stages: poses and points alone first, the focal length and the first two lens coefficients once a sensor has 3 views, and the principal point and the whole lens once it has 6 (`stage_for` in `crates/sfm/src/lens.rs`, `full_model_views`). At the end the lens is chosen: a pinhole, a radial model, the full Brown model and the Kannala-Brandt fisheye are each fitted to the finished reconstruction and the one with the lowest Bayesian information criterion, $n\ln(C/n) + k\ln n$ with $n$ the number of residuals, $C$ the robust cost and $k$ the number of calibration parameters, wins, so that a coefficient has to pay for itself in cost.

## The surface between the points

The reconstruction so far is a sparse cloud, the points that were distinctive enough to match, and the cameras. A wall between them has no points, and neither has a smooth table. Yet every pixel of every photograph is a measurement of the surface behind it, and with calibrated cameras one can ask of each pixel what a feature matcher could not: at which distance along this pixel's ray does the surface lie? **Patch-match stereo** answers it for every pixel by two ideas, a cost for a hypothesis and a way to search the hypotheses that does not try them all.

A single depth for a pixel is not enough: the cost of a hypothesis compares a patch, not a pixel, and a patch of a surface that is tilted is not a patch of a fronto-parallel surface. So the hypothesis for the pixel $\mathbf{p}$ of the reference view is a range $r$ along its unit ray $\mathbf{d}_\mathbf{p}$ and a unit normal $\mathbf{n}$ facing the camera: the plane through $r\,\mathbf{d}_\mathbf{p}$ with normal $\mathbf{n}$. Another pixel $\mathbf{q}$ of the patch has the ray $\mathbf{d}_\mathbf{q}$, which meets the plane at the range

$$
r_\mathbf{q} = \frac{\mathbf{n}\cdot(r\,\mathbf{d}_\mathbf{p})}{\mathbf{n}\cdot\mathbf{d}_\mathbf{q}}, \tag{12}
$$

(valid when $\mathbf{n}\cdot\mathbf{d}_\mathbf{q}$ is clearly negative), and that 3D point is carried into the source view with its pose and projected through its own lens, where the grey value is read. No rectification and no homography: the warp is the geometry of equation (12) and the projection of chapter 3, one sample at a time, for any lens, which is how `crates/mvs` handles a fisheye (`lens_project` in the shader library; commit 01cba670e states the design). Ranges, not depths, because for a wide lens a depth along the axis does not exist off the axis.

Compare the reference patch with the warped source patch by their weighted normalized cross-correlation, the correlation coefficient of the two patches after subtracting their means and dividing by their standard deviations, which is blind to a change of brightness and contrast between photographs: for the reference grey values $g_q$ and the source values $s_q$, with weights $w_q$,

$$
\mathrm{NCC} = \frac{\sum_q w_q\,g_q s_q/\sum_q w_q - \bar g\,\bar s}{\sqrt{\sigma_g^2\,\sigma_s^2}}, \qquad \text{cost} = 1 - \mathrm{NCC}, \tag{13}
$$

with the weighted means $\bar g, \bar s$ and variances. The weights are bilateral, $w_q = \exp\bigl(-(g_q - g_\mathbf{p})^2/2\sigma_c^2 - \lVert\mathbf{q} - \mathbf{p}\rVert^2/2\sigma_s^2\bigr)$, from the reference alone: a sample whose colour differs from the centre's, probably across a depth edge, counts for little. A patch whose weighted variance is below a floor carries no evidence and costs 2, the maximum. The window of the example and of `crates/mvs` is 11 by 11 pixels sampled every second pixel, 36 samples, with colour sigma 0.03 and spatial sigma 5. With several source views the cost of a hypothesis is the average over them (the code's weighting is below).

Three small images make it concrete. They show one tilted textured plane; the reference camera is at the origin and two sources stand 0.5 metre to the right and 0.4 metre to the left, all with parallel axes and a focal length of 60 pixels:

<!-- worked: structure_from_motion.plane_sweep -->
Three 48 by 36 gray images from cameras with f = 60 pixels and principal point (24, 18) at the eyes [0.0, 0.0, 0.0], [0.5, 0.0, 0.0] and [-0.4, 0.1, 0.0] (reference, right source, left source), all looking along +z. The scene is the plane through (0, 0, 4) with unit normal (-0.572982, 0.0409273, -0.818546), tilted about the vertical axis. Its texture is a lattice of 37 by 25 uniform values 0.25 m apart (`bookrng.generator("structure_from_motion.texture")`, seed 4428204777391956342, row by row, values in the order of increasing a within a row), interpolated bilinearly over the plane coordinates (a, b) along e1 = unit(n x (0, 1, 0)) and e2 = n x e1 measured from the plane point. Each pixel centre's ray meets the plane and reads the texture; Gaussian noise of standard deviation 0.01 is added (`bookrng.generator("structure_from_motion.images")`, seed 2593629484251715123; views in order, rows top to bottom, left to right, one `normal()` each).

The cost of a hypothesis for the reference pixel is the mean over the two sources of 1 - NCC, the bilaterally weighted normalized cross-correlation of the 6 by 6 samples of an 11 by 11 window (spacing 2), color sigma 0.03, spatial sigma 5. A plane sweep tries 64 ranges, uniform in inverse range from 2.5 to 8:

| pixel | true range | cost at the true plane | cost at the true range, fronto-parallel normal | best range of the fronto-parallel sweep | its cost | best range with the true normal | its cost |
|---|---:|---:|---:|---:|---:|---:|---:|
| (24, 18) | 3.9787 | 0.0912 | 0.1373 | 4.0449 | 0.1331 | 3.9748 | 0.0904 |
| (12, 10) | 4.7053 | 0.2200 | 0.4668 | 4.6154 | 0.4408 | 4.7103 | 0.2250 |
| (36, 26) | 3.6224 | 0.0151 | 0.1909 | 3.7781 | 0.0124 | 3.6000 | 0.0177 |
<!-- /worked -->

The **plane sweep** of the last columns tries one hypothesis normal (fronto-parallel) over 64 ranges and takes the one with the least cost, the oldest way to do stereo. It finds nearly the true range for the central pixel (4.04 against 3.98, cost 0.133 against 0.091 for the true plane). The third pixel shows the price of the fronto-parallel assumption: the best fronto-parallel range is 3.78 where the truth is 3.62, 4 percent too far, at a cost of 0.012 that is lower than the true plane's 0.015, because a window that crosses a tilted surface sees a range of depths and the sweep picks the depth that best fits the middle of it. With the true normal in the hypothesis the best range is 3.60.

![Cost 1 minus normalized cross-correlation against the hypothesised range for the reference pixel (36, 26), with a fronto-parallel normal and with the true normal. Both have a minimum near 3.6 metres; the fronto-parallel curve has its minimum at 3.78 and is about 0.2 high at the true range, while the true-normal curve is lowest at the true range, 3.62.](img/stereo-cost.svg)

A sweep over ranges is already a search over one dimension per pixel; adding the normal makes it three (a range and two angles), and the search space per pixel is the product. For 1 percent steps of the inverse range between 2.5 and 8 metres there are 110 ranges, and normals on a hemisphere to 10-degree cells are $2\pi/(10^\circ)^2 \approx 206$, so about 22,660 hypotheses per pixel, each a patch against each source view. The idea of **PatchMatch** (Barnes and others, 2009; for stereo, Bleyer, Rhemann and Rother, 2011) is that most of them need not be tried, because the answer is spatially coherent: a good plane for a pixel is probably a good plane for its neighbours.

Start every pixel from a random plane (a random range uniform in the inverse range, a random normal facing the camera). Then, repeatedly, let each pixel try the planes of its neighbours, re-intersected with its own ray (equation (12) gives the range), and keep the best; then perturb the winner a little and try a few fresh random planes, which lets good hypotheses refine and rescues a pixel whose neighbours are all wrong. A good plane spreads across a region in a few passes, the way a rumour does, and a pixel that was lucky with a random draw seeds its neighbourhood. The refinement perturbation shrinks with the iteration. To run it in parallel the pixels are coloured as a checkerboard and one colour is updated at a time, so that no pixel reads one that is being written: red-black propagation, which `crates/kernels/wgsl/mvs_pm.wgsl` runs on a GPU with one thread per pixel.

```text
function patchmatch_half_step(r: inout array<real>[H, W], n: inout array<real>[H, W, 3],
                              c: inout array<real>[H, W], rays: array<real>[H, W, 3],
                              colour: int, dr: real, dn: real, rmin: real, rmax: real,
                              cost: function(int, int, real, array<real>[3]) -> real,
                              rng: Generator) -> none:
    ox = [-1, 1, 0, 0]
    oy = [0, 0, -1, 1]
    for y in 0 .. H:
        for x in 0 .. W:
            if (x + y) mod 2 != colour:
                continue
            d = rays[y, x, :]
            best_r = r[y, x]
            best_n = n[y, x, :]
            best_c = c[y, x]
            for k in 0 .. 4:
                nx = x + ox[k]
                ny = y + oy[k]
                if 0 <= nx < W and 0 <= ny < H:
                    m = n[ny, nx, :]
                    den = dot(m, d)
                    if den < -1e-4:
                        rr = dot(m, r[ny, nx] * rays[ny, nx, :]) / den
                        if 0 < rr < 4 * rmax:
                            cc = cost(x, y, rr, m)
                            if cc < best_c:
                                best_r = rr
                                best_n = m
                                best_c = cc
            r0 = best_r
            n0 = best_n
            r_pert = r0 * exp(dr * (2 * rng.uniform() - 1))
            n_pert = unit(n0 + dn * rand_dir(rng))
            r_rand = 1 / (1 / rmax + (1 / rmin - 1 / rmax) * rng.uniform())
            n_rand = face(rand_dir(rng), d)
            rs = [r_pert, r0, r_pert, r_rand, r0, r_rand]
            ns = [n0, n_pert, n_pert, n0, n_rand, n_rand]
            for k in 0 .. 6:
                if dot(ns[k], d) > -0.05:
                    continue
                cc = cost(x, y, rs[k], ns[k])
                if cc < best_c:
                    best_r = rs[k]
                    best_n = ns[k]
                    best_c = cc
            r[y, x] = best_r
            n[y, x, :] = best_n
            c[y, x] = best_c
```

The helpers are small: `rand_dir(rng)` draws $z = 2u - 1$ and $\varphi = 2\pi u'$ with two `rng.uniform()` calls (in that order) and returns $(\sqrt{1 - z^2}\cos\varphi,\ \sqrt{1 - z^2}\sin\varphi,\ z)$, and `face(n, d)` returns $-\mathbf{n}$ if $\mathbf{n}\cdot\mathbf{d} > 0$ and $\mathbf{n}$ otherwise. The six candidates after propagation are the perturbed range, the perturbed normal, both, a fresh range, a fresh normal and both fresh, scored with the same cost. In iteration $t$ the range is perturbed by a relative factor of up to $\exp(0.1\cdot0.5^t)$ and the normal by a random vector of length up to $0.5\cdot0.5^t$ (with floors of 0.002 and 0.02), and an iteration is two half-steps, colour 0 and then colour 1. The example runs six iterations on the three images:

<!-- worked: structure_from_motion.patchmatch_table -->
PatchMatch on the reference image with the two sources, window and weights of the plane sweep. Generator `bookrng.generator("structure_from_motion.patchmatch")` (seed 11193093832229907313). Start: for every pixel in row-major order, three draws: the range (1 / (1/8 + (1/2.5 - 1/8) u)), then the two draws of a random direction (z = 2u - 1, phi = 2 pi u), the normal turned to face the camera. Then 6 iterations of two half-iterations, colour (x + y) mod 2 = 0 and then 1, pixels of a colour in row-major order, each in turn: take the planes of its four neighbours (which have the other colour and are final for this half-iteration), re-intersect each with this pixel's ray, and keep the cheapest of these and the current hypothesis; then draw six uniform numbers (the range perturbation u, the two numbers of the direction added to the normal, the fresh range, the two numbers of a fresh normal) and score the six combinations (perturbed range, perturbed normal, both, fresh range, fresh normal, both fresh), skipping a hypothesis with n.d > -0.05, keeping the cheapest overall. In iteration t (from 0) the perturbation of the range is a relative factor exp(0.1 * 0.5^t (2u - 1)) and the added direction is scaled by 0.5 * 0.5^t (floors 0.002 and 0.02). Every number below is printed with only the digits that are the same in this run and in four copies of it whose texture value under the middle pixel is changed by +1e-12, -1e-12, +3e-12 and -3e-12 (the stable-digits rule of the book): a decision of the search that depended on rounding would show as a digit that differs.

| after | cost evaluations so far | median relative range error | within 1% of the true range | median normal error (deg) |
|---|---:|---:|---:|---:|
| random start | 1728 | 0.2709 | 2.1% | 62.4 |
| iteration 1 | 18245 | 0.0763 | 9.4% | 48.8 |
| iteration 2 | 34956 | 0.0232 | 24.7% | 29.4 |
| iteration 3 | 51749 | 0.0142 | 39.6% | 19.1 |
| iteration 4 | 68546 | 0.0111 | 47.4% | 14.8 |
| iteration 5 | 85391 | 0.0100 | 50.1% | 13.8 |
| iteration 6 | 102232 | 0.0100 | 50.1% | 13.0 |

A cost evaluation is one hypothesis scored against both sources, 36 window samples each. The run used 102232 of them for 1728 pixels, 59 per pixel. An exhaustive search at the same resolution needs 110 ranges (steps of 1% of the inverse range at its middle value) times 206 normals (cells of 10 degrees on a hemisphere, 2 pi / (10 degrees in radians)^2) = 22660 evaluations per pixel, 383 times as many.
<!-- /worked -->

From a random start the median error of the range is 27 percent and 2.1 percent of the pixels are within 1 percent of the truth. After four iterations the median is 1.1 percent; after five it stops improving at 1.0 percent, with half of the pixels within 1 percent of the true range. The normals converge more slowly and less far, to a median of 13 degrees: the data constrain the range to a patch's worth of precision, and the tilt only through a difference of ranges across the window. The work is counted in cost evaluations, the unit of the algorithm (each is 36 window samples times two source views): 59 per pixel for the whole run, against 22,660 for the exhaustive search at this resolution, 383 times as many. Whether the surface of the example is well recovered is not the claim; the claim is that a search that tries a thousandth of the hypotheses finds in a few passes what the full search would, because the neighbours tell it where to look. The digits of this table pass the stable-digit check: the same run with the texture value under the middle pixel changed by $\pm10^{-12}$ and $\pm3\times10^{-12}$ gives the same numbers, so the random process is deterministic given its draws and not chaotic.

## What the real pipeline adds

The example has one reference and two sources and one level. `crates/mvs` is the same algorithm made to work on photographs, and each of its additions answers a failure.

Which sources? The right ones: `select_sources` in `crates/mvs/src/select.rs` scores every candidate by the tracks it shares with the reference, each weighted by the triangulation angle it is seen under (the weight of the section on triangulation: 1 between 5 and 60 degrees, falling quadratically outside) and by how similar the two views' footprints of the point are, times the fraction of an 8 by 8 grid of the reference that the shared tracks cover (a view that shares many tracks in one corner helps one corner), and keeps the best eight with at least 12 shared tracks. The depth range comes from the sparse points too: the ranges of the tracks a view sees, between the 2 and 98 percent quantiles, widened by 30 percent, bound the random initialization. No pixel has been looked at.

Which views, per pixel? A source that cannot see the surface (it is occluded) matches nothing, and averaging it in would punish the true hypothesis. The kernel scores the current hypothesis and the eight candidates against every source and trusts a view when at least two hypotheses match it well (cost below 0.8) and at most three badly (above 1.2): one lucky match is not evidence of visibility. A trusted view weighs $\exp(-c^2/2\sigma^2)$ of its best good cost with $\sigma = 0.3$; if no view is trusted, every view is weighted by its best cost alone. The propagation is richer than the four neighbours of the example: along each of the four image axes a near V-shaped region and a far strip of 11 pixels reaching 23 pixels out, each contributing the plane of its cheapest pixel.

What of weakly textured surfaces, where the cost has no minimum? The photometric cost needs texture. A coarse-to-fine pyramid (exact box halvings of the photographs; three levels by default, six photometric iterations at the coarsest) lets a plane found where the image is still textured at a coarse scale be upsampled to a finer one; and after the photometric iterations of the coarsest level, every level also runs iterations with a **geometric consistency** term that adds to the cost the distance, in pixels, between the reference pixel and the point where the hypothesis, imaged into a source, read from that source's own current range map, and carried back, lands, with weight 0.2 and a truncation at 3 pixels: it is zero exactly when the two maps agree, and it fills regions that the photometric cost alone leaves to chance.

What is kept? Not everything. `mvs_filter.wgsl` keeps a pixel's final hypothesis only if at least two sources agree with it (their range maps, read at the pixel it projects to and carried back, land within 1 pixel and 1 percent of the range, and the two rays meet at an angle of at least 2 degrees, since a view on the same line of sight agrees with anything), its normal faces the camera within 85 degrees, and its cost is at most 1; the confidence is $\min(1, \text{agreeing}/4)(1 - \text{cost}/\text{max cost})$. A patch with no texture costs the maximum and is never kept, however consistent its plane: a plane propagated out over a blank wall is consistent in every view by construction. Then fusion: the maps are merged into one point per surface sample, `fuse` in `crates/mvs/src/fuse.rs`, as a gather over each pair of views with no atomics, so that the cloud is the same in any order; every pixel looks up the other view's range at the pixel its point projects to and accepts it if the two agree within 2 pixels, 1 percent in range and 20 degrees in normal, and the point is emitted once, by the view that samples it most finely, with the number of agreeing views as support (at least 3).

The tests of the crate (`crates/mvs/tests/it/stereo.rs`) are ray-cast scenes through a pinhole and a Kannala-Brandt fisheye with known planes, held to coverage above 60 percent of the surface, more than 80 percent of measured pixels within 0.5 percent of the true range, fewer than 2.5 percent outliers and a median normal error below 3 degrees; the first version of the crate is commit 01cba670e. The fused cloud is thinned when a budget asks (`Fused::thinned`, which keeps a uniform subset and widens each disc by $\sqrt{n/\text{keep}}$ so the kept discs still cover the surface; commit b3e0c6334 records why: a dense start of 3.1 million gaussians had silently ignored a smaller budget) and becomes a starting scene of thin gaussians lying in the surface, with the normal as the shortest axis (`to_splats` in `crates/mvs/src/init.rs`), the subject of two chapters ahead.

## From photographs to a starting scene

![Two rows of boxes. The top row, crates/sfm, runs from features through matching, tracks and initial cameras to bundle adjustment and lens choice; an arrow labelled cameras, calibration and sparse tracks leads to the bottom row, crates/mvs, which runs from source views and range bounds through the PatchMatch pyramid, the consistency filter and fusion to a splat start of thin gaussians.](img/sfm-mvs-pipeline.svg)

`crates/recon` joins the stages (`photogrammetry` in `crates/recon/src/photogrammetry.rs`): `training_set` runs structure from motion on a folder of photographs and turns each registered one into a target for a splat fit with its own camera, `TrainingSet::densify` runs multi-view stereo through the same lenses, attaches each view's range, normal and confidence to its target as priors, and replaces the sparse start by the fused cloud, and `TrainingSet::upright` lands the capture with up pointing up, along the direction of gravity that the reconstruction knows (`Reconstruction::up`, estimated from how the cameras were held, or the vertical of the satellite frame) and from the orbit of the cameras when it knows none. The figure is the path.

A sentence of this chapter's own history is worth the care. The first commit of the path, e6d9f25cd, described `recon::photogrammetry` as "resampling every photograph to an ideal pinhole camera at the training width (lens distortion removed from the targets...)". The module today says the opposite. "Nothing is resampled into an idealized camera": the renderer evaluates every gaussian along each pixel's own ray through the lens model, so undistorting a photograph could only lose what it recorded (interpolation blurs exactly the detail a fit is trying to recover, and the corners that a pinhole frame cannot reach are thrown away), and a lower resolution is reached by exact halvings, never by resampling at an arbitrary factor. The turn happened in commit 2e838f2a6, whose message reads "recon no longer undistorts: every photograph is a target as recorded, masked where the lens has no ray". The first message was true of the code when it was written and is not true of the code now. A commit message is a claim about one day, and the module comment, which a reader of the code meets, is the one to keep true.

A capture that does not fit one forward pass of a learned model is the other thing `crates/recon` does (commit 84521a0f2): it plans overlapping chunks, asks the model how many frames it takes, reconstructs each, registers it onto the world by the similarity of the gauge section and refuses the ones that do not fit. That is the subject of the next chapter, since it presupposes the model, and the two routes meet in what the volume's last chapters need: calibrated cameras, and geometry, however obtained.

## Check your understanding

1. A student says that RANSAC with a larger threshold must find the right model more often, since more matches count as inliers. Give two reasons the larger threshold can make the result worse, one about the model and one about the iteration bound.
2. The epipolar test of the first example passes a wrong match that is 1 pixel from the epipolar line. Explain why no number of further tests of the same pair can detect it, and what a third view adds.
3. A pipeline estimates the pose of a second camera from an essential matrix and finds that the triangulated points are in front of the first camera and behind the second. Which of the four candidates has it chosen, and which change turns it into the right one?
4. Bundle adjustment of 4 cameras and 52 points is run once with the first camera held and the baseline of the first pair fixed, and once with nothing held and only damping. Both reach the same reprojection error. How do the two results differ, and why is neither wrong?
5. A reconstruction is triangulated through a focal length 10 percent too long, and the first pair gives a reprojection error indistinguishable from the correct one. What does that show about the pair, and which later computation exposes the error?
6. In PatchMatch, why must a pixel read only the planes of pixels of the other colour in the same half-iteration, and what goes wrong on a GPU if it reads one of its own colour?
7. A plane sweep over fronto-parallel planes finds a range 4 percent too far for a pixel on a tilted wall, with a lower cost than the true plane. Explain the bias with the shape of the window.

## Summary

A scene can be recovered from photographs by a chain of stages, each of which repairs the last. Keypoint descriptors give matches, a fifth to a half of them wrong; the epipolar constraint $\mathbf{x}_2^\top\mathbf{E}\mathbf{x}_1 = 0$ turns two images into a model with five parameters; RANSAC finds it by voting among minimal samples, with an iteration count that follows from the inlier fraction and the sample size and that fails numerically when written naively; a refinement by Levenberg and Marquardt polishes it; the four-fold ambiguity of the pose is resolved by cheirality; further cameras register through points already triangulated; and a bundle adjustment of every pose and point against the pixels, with the points eliminated by the Schur complement, removes what the stages passed on. The photographs fix the answer only up to a similarity, seven numbers that the adjustment must be told, and a pair of views cannot distinguish a wrong focal length that later views expose. Between the sparse points, plane hypotheses scored by correlation across views, searched by PatchMatch, filtered by multi-view consistency and fused, give a surface.

What the chapter established with evidence: the depth error of a triangulated point falls as the angle between the rays grows and follows $\sqrt{2}\,\sigma Z^2/(fB)$; a matcher that returns 54 percent correct matches can still give a pose after 2076 iterations of a robust search; a bundle adjustment from a start 8 degrees off reaches the same minimum as one from the truth; the Schur complement gives the dense step to $10^{-14}$ at a fraction of the work; and PatchMatch reaches 1 percent of the true range in a few iterations with 59 evaluations a pixel. The scene was static and textured, and the pipeline is a set of thresholds each earned against a failure; a capture with moving objects or blank surfaces needs them rechecked. The next chapter asks whether a network can look at a few images and produce the geometry in one pass.

## References

- Lowe, *Distinctive Image Features from Scale-Invariant Keypoints*, 2004.
- Fischler and Bolles, *Random Sample Consensus*, 1981.
- Torr and Zisserman, *MLESAC: A New Robust Estimator with Application to Estimating Image Geometry*, 2000.
- Nistér, *An Efficient Solution to the Five-Point Relative Pose Problem*, 2004.
- Hartley, *In Defense of the Eight-Point Algorithm*, 1997.
- Hartley and Zisserman, *Multiple View Geometry in Computer Vision*, 2003.
- Triggs, McLauchlan, Hartley and Fitzgibbon, *Bundle Adjustment: A Modern Synthesis*, 1999.
- Levenberg, *A Method for the Solution of Certain Non-Linear Problems in Least Squares*, 1944; Marquardt, *An Algorithm for Least-Squares Estimation of Nonlinear Parameters*, 1963.
- Schönberger and Frahm, *Structure-from-Motion Revisited*, 2016.
- Barnes, Shechtman, Finkelstein and Goldman, *PatchMatch: A Randomized Correspondence Algorithm for Structural Image Editing*, 2009.
- Bleyer, Rhemann and Rother, *PatchMatch Stereo: Stereo Matching with Slanted Support Windows*, 2011.
- Schönberger, Zheng, Pollefeys and Frahm, *Pixelwise View Selection for Unstructured Multi-View Stereo*, 2016.
- Xu and Tao, *Multi-Scale Geometric Consistency Guided Multi-View Stereo*, 2019.
- Goesele, Snavely, Curless, Hoppe and Seitz, *Multi-View Stereo for Community Photo Collections*, 2007.
- Umeyama, *Least-Squares Estimation of Transformation Parameters Between Two Point Patterns*, 1991.

## Exercises

1. **[Calculate]** Two parallel cameras with $f = 800$ pixels are 0.3 metre apart. A point has disparity 24 pixels. Find its depth, and the depth if the disparity is 24.5 and 23.5; then the first-order error $Z^2\,\delta d/(fB)$ for $\delta d = 0.5$. Check against `structure_from_motion.disparity_depth`.
2. **[Implement]** Reproduce `structure_from_motion.triangulation_error`: implement the linear triangulation of equation (1) and the Gauss-Newton refinement, draw the noise in the stated order from the stream and seed of the chapter, and print the RMS depth error and the prediction for the five baselines.
3. **[Calculate]** With 40 percent inliers, find the number of samples that 99.9 percent confidence needs for sample sizes 5, 6 and 8, by equation (4). Then do the same for 10 percent and 0.1 percent inliers with the numerically safe form, and say at which fraction the naive form fails in 64-bit arithmetic for $s = 8$.
4. **[Implement]** Reproduce `structure_from_motion.feature_matching` and `structure_from_motion.track_building`: generate the toy descriptors in the stated draw order, implement the nearest neighbour, ratio and mutual rules and the union-find, and print the tables.
5. **[Implement]** Reproduce `structure_from_motion.two_view_ransac` and `structure_from_motion.pose_ambiguity`: the eight-point algorithm with Hartley normalization, the rank-two hypotheses, the truncated score with the adaptive bound, the refit and the Levenberg-Marquardt refinement of the five-parameter pose, and the four-fold cheirality test.
6. **[Calculate]** For the essential matrix $\mathbf{E} = [\mathbf{t}]_\times$ of $\mathbf{R} = \mathbf{I}$ and $\mathbf{t} = (1, 0, 0)$, write $\mathbf{E}$, find the epipolar line in image 2 of the point $(0.2, 0.3, 1)$ of image 1, and say what the four decompositions are.
7. **[Calculate]** Derive the Jacobian block of the projection with respect to a left rotation increment, $\mathbf{J}_\pi(-[\mathbf{c} - \mathbf{t}]_\times)$, from $\mathbf{c}' = \exp(\boldsymbol\omega)\mathbf{R}\mathbf{X} + \mathbf{t}$ and the first-order expansion $\exp(\boldsymbol\omega)\approx\mathbf{I} + [\boldsymbol\omega]_\times$, and check it numerically on one observation by central differences.
8. **[Implement]** Reproduce `structure_from_motion.bundle_adjustment` and the first lines of `structure_from_motion.ba_structure`: the camera slots (held, scale camera with five columns, free), the Huber weights, the Schur step against the dense step, the damping schedule, and the rescale after each accepted step.
9. **[Calculate]** Using the counts of the chapter, compute for 100 cameras, 20,000 points and 4 observations per point the dense multiply-adds ($n^3/6$ with all unknowns but the seven gauge directions removed, taking $n$ as $6\cdot100 + 3\cdot20{,}000 - 7$), the Schur multiply-adds ($54k + 108k^2$ per point plus $n_c^3/6$) and the bytes of both matrices.
10. **[Investigate]** Reproduce `structure_from_motion.focal_length_effect` and repeat it for a scene whose cameras stand 5 metres apart instead of 0.6 (keep the cameras looking at the same target). How do the RMS errors of the registered cameras at the wrong focal lengths change, and does the shallow valley of the profile steepen?
11. **[Implement]** Reproduce `structure_from_motion.plane_sweep` and `structure_from_motion.patchmatch_table` for the three images: render them from the texture lattice and the noise stream, implement the weighted NCC cost, the sweep, and the red-black PatchMatch with the six-candidate refinement. Compare the number of cost evaluations with the exhaustive count.
12. **[Investigate]** Add a second plane to the PatchMatch scene, a step in depth across the middle of the image, so that part of the reference is seen by only one of the two sources. Run the average of two sources and then a per-view selection rule that drops a source whose best cost is above 0.8. Where do the errors concentrate, and what does each rule do there?

## Key terms

- **feature matching**: pairing the keypoints of two images by the similarity of their descriptors, usually by the nearest neighbour under a ratio test and a mutual test, so that the pairs are views of the same scene point. <!-- concept: feature-matching -->
- **epipolar geometry**: the constraint that two views of one scene impose on each other: the image of a point in one view lies on a line, the epipolar line, in the other, expressed for calibrated cameras by $\mathbf{x}_2^\top\mathbf{E}\mathbf{x}_1 = 0$ with the essential matrix $\mathbf{E} = [\mathbf{t}]_\times\mathbf{R}$. <!-- concept: epipolar-geometry -->
- **triangulation**: finding a 3D point from its images in two or more cameras of known pose, as the point where the rays meet or, with noise, the one that fits them best, whose depth error falls as the angle between the rays grows. <!-- concept: triangulation -->
- **bundle adjustment**: the joint nonlinear least-squares refinement of every camera pose, every 3D point and optionally the calibration against the observed pixels, solved by Levenberg-Marquardt with the points eliminated by the Schur complement, and fixing the gauge. <!-- concept: bundle-adjustment -->
- **patch-match stereo**: dense depth estimation in which every pixel holds a plane hypothesis (range and normal), scored by correlation of its patch across views, and a randomized search propagates good hypotheses between neighbouring pixels and refines them, so that a small fraction of the hypothesis space is tried. <!-- concept: patch-match-stereo -->

## Answers

**Recall questions.**

1. $(520, 140)$: $320 + 500\cdot0.8/2$ and $240 + 500\cdot(-0.4)/2$. The pixel $(400, 300)$ is the image of every point $Z\,(0.16, 0.12, 1)$, $Z > 0$.
2. $\mathbf{C} = -\mathbf{R}^\top\mathbf{t} = (0.6, 0, 0)$. A program that reads the pair as camera to world puts the eye at $\mathbf{t} = (-0.6, 0, 0)$, on the other side of the origin.
3. $\mathbf{x} = (\mathbf{A}^\top\mathbf{A})^{-1}\mathbf{A}^\top\mathbf{b}$, the solution of the normal equations; the residual $\mathbf{A}\mathbf{x} - \mathbf{b}$ is orthogonal to every column of $\mathbf{A}$.
4. The step must stay below $2/\lambda_{\max} = 2/100 = 0.02$. At that size the error along the flat direction (curvature 1) shrinks by $1 - 0.02\cdot1 = 0.98$ per step, which is why plain descent is slow in a narrow valley and why damped Gauss-Newton scales each parameter by its own curvature.
5. The right singular vector of the smallest singular value, since $\lVert\mathbf{A}\mathbf{h}\rVert$ over unit vectors is smallest there, equal to that singular value. The points were normalized because the columns of $\mathbf{A}$ differed by five orders of magnitude, which made the smallest singular value indistinct from the next.

**Check your understanding.**

1. About the model: a threshold above the noise admits wrong matches near the epipolar line as inliers, which then enter the refit (the 3 matches of the example tilted the pose by 2.5 degrees). About the bound: the best inlier fraction rises with the threshold, so the bound of equation (4) falls and the search ends sooner, with less chance to find a sample that is also accurate, since many mediocre samples then reach the same inlier count.
2. The constraint says only that the match is on a line; every point along the line satisfies it equally, and no further statistic of the same pair distinguishes the position along it. A third view, in which the point must lie on a second epipolar line from each of the first two, intersects the lines and so fixes the position; a wrong match then fails there.
3. Of the four candidates exactly one puts the points in front of the first camera and behind the second: the one with the right sign of $\mathbf{t}$ and the rotation that differs from the right one by the half turn about the baseline (the third panel of the figure, where camera 1 faces away). Switching $\mathbf{R}$ to the other rotation, $\mathbf{U}\mathbf{W}\mathbf{V}^\top$ for $\mathbf{U}\mathbf{W}^\top\mathbf{V}^\top$ or the reverse, and keeping $\mathbf{t}$, gives the candidate with the points in front of both.
4. The held version has camera 0 at the identity and the baseline of the first pair equal to 1; the free version has some other frame and scale, a similarity of the first. Both reach the same cost because the cost is invariant under the similarity, so neither is wrong; they differ only by the 7-parameter gauge, and with damping alone the free one also drifts along it by an amount that depends on the starting point.
5. It shows that two views cannot see a focal length in this range, since any two images have a pose that explains them through it. A third or fourth camera registered against the points exposes it: its reprojection error rises (2.11 to 3.62 pixels at 400 in the example), as does the cost of a bundle adjustment with the focal length held at the wrong value.
6. The cost of a pixel depends on the current planes of its neighbours; if a neighbour were updated in the same half-iteration the result would depend on the order the threads ran, and the run would not be reproducible or race-free. The checkerboard colouring means all the neighbours read are of the other colour and are not written in this pass. A pixel that read its own colour would read values another thread may be writing at that moment.
7. A window of 11 pixels at a depth of 4 m is 0.73 m wide, and on a wall tilted by 35 degrees it spans a range of depths about 0.5 m across, and the fronto-parallel hypothesis is one depth for the whole window. The depth that maximizes the correlation over the window is the one that best matches its heaviest-weighted samples, which is not the centre pixel's true range, so the minimum moves along the tilt, here away from the camera; the true normal removes the bias.

**Exercises.**

1. $Z = fB/d = 800\cdot0.3/24 = 10$ m. For $d = 24.5$: $9.796$; for $23.5$: $10.213$. The first-order error is $100\cdot0.5/240 = 0.2083$ m. The table of the worked example follows the same arithmetic for $f = 500$, $B = 0.6$.
2. Table of `structure_from_motion.triangulation_error`: the linear and refined errors agree to four digits and follow the prediction within the sampling scatter of 200 trials.
3. At 40 percent: $s = 5$: $\ln(0.001)/\ln(1 - 0.4^5) = 6.9078/0.010293 = 671.1$, so 672; $s = 6$: $0.4^6 = 0.004096$, 1684; $s = 8$: $0.4^8 = 0.000655$, 10,537. At 10 percent and $s = 8$: $p = 10^{-8}$, $6.9\times10^8$ with `log1p`; at 0.1 percent: $p = 10^{-24}$, $6.9\times10^{24}$. The naive form fails when $1 - p$ rounds to 1: for $s = 8$ when $p < 2^{-54}$, that is $w$ below about 0.0093; in the table, $w = 0.01$ is already 11 percent off in the deficit and $w = 0.003$ divides by zero.
4. The tables of the two blocks: nearest neighbour only gives 100 matches of which 54 are correct, the strict rules give 45 of 45; the track table gives 59 tracks, 44 pure in all four views, for ratio 0.8 and mutual.
5. The tables of the blocks: 2076 iterations, 49 inliers of the best hypothesis, 54 after the refinement, the pose 2.52 degrees and 5.24 degrees from the truth, and the four candidates with 47, 2, 0 and 0 points in front of both cameras.
6. $[\mathbf{t}]_\times$ for $\mathbf{t} = (1, 0, 0)$ is $\begin{pmatrix}0&0&0\\0&0&-1\\0&1&0\end{pmatrix}$, and with $\mathbf{R} = \mathbf{I}$ that is $\mathbf{E}$. The line of $(0.2, 0.3, 1)$ is $\mathbf{E}\mathbf{x}_1 = (0, -1, 0.3)$: $-y + 0.3 = 0$, the horizontal line $y = 0.3$ (a sideways translation keeps every point on its own row). The decompositions are $\mathbf{R} \in \{\mathbf{I}, \text{a half turn about the } x \text{ axis}\}$ with $\mathbf{t} = \pm(1, 0, 0)$.
7. $\mathbf{c}' \approx \mathbf{c} + \boldsymbol\omega\times(\mathbf{c} - \mathbf{t})$ since $\exp(\boldsymbol\omega)\mathbf{R}\mathbf{X} = \mathbf{R}\mathbf{X} + \boldsymbol\omega\times\mathbf{R}\mathbf{X}$ and $\mathbf{R}\mathbf{X} = \mathbf{c} - \mathbf{t}$; and $\boldsymbol\omega\times\mathbf{q} = -\mathbf{q}\times\boldsymbol\omega = -[\mathbf{q}]_\times\boldsymbol\omega$, so $\partial\mathbf{c}/\partial\boldsymbol\omega = -[\mathbf{c} - \mathbf{t}]_\times$ and the block is $\mathbf{J}_\pi$ times it.
8. The tables of the two blocks: 52 points, 172 observations, 28 iterations, 0.6255 to 0.4708 pixel, Schur and dense steps agreeing to $2.4\times10^{-14}$, 2368 nonzeros of 59,512.
9. $n = 600 + 60{,}000 - 7 = 60{,}593$; dense $n^3/6 = 3.71\times10^{13}$ multiply-adds and $8n^2 = 2.94\times10^{10}$ bytes; Schur $n_c = 593$, $20{,}000\cdot(216 + 1728) = 3.89\times10^7$ plus $593^3/6 = 3.5\times10^7$, about $7.4\times10^7$ multiply-adds and $8\cdot(593^2 + 9\cdot20{,}000) = 4.3\times10^6$ bytes.
10. With a baseline of 5 metres the angle between rays at 5 metres of depth is 53 degrees, against 7. The expectation to test is that the registered cameras' error rises more steeply with the wrong focal length and the profile of the cost has a steeper valley, since a wide baseline separates a focal length from a distance; the numbers depend on the scene you build, and the exercise is to measure how much.
11. The tables of the two blocks: the cost at the true plane 0.0151 and the best fronto-parallel range 3.78 at pixel (36, 26); PatchMatch from a median range error of 0.2709 to 0.0100 in 5 iterations with 59 cost evaluations per pixel against 22,660 for the exhaustive count.
12. Pixels seen by one source only have a matching cost that is a mean of one good and one bad cost under the average rule, which prefers a wrong plane that is moderately good in both; the per-view rule that drops the bad source scores the good one alone and fixes the pixel, while at pixels seen by both it changes little.
