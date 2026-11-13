<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Project: Reconstructing and Predicting a Scene

*Can you reconstruct a scene from images and predict how it changes?*

At frame 12 a yellow ball reaches the edge of a table. It has been sliding to the right for twelve frames, 0.15 metres each, and a model that was shown those twelve frames and nothing else has to say where the ball will be in frame 12, in frame 13 and in the four after them. It has never seen an edge. It has seen a ball that moves in a straight line at constant speed, which is exactly what a linear model of the ball describes, and it will carry the ball on across the edge and out into the air at the same speed. By the last of the six frames it will be three quarters of a metre beyond the edge, and the picture it renders will show the ball in a place where the world has none.

Here is that model scored in seven ways, together with three mistakes of the kind that a careful review passes. The scene is built by the module that wrote every number of this chapter: a wall and a table filmed by a camera that slides past them, three balls on the table, twelve observed frames and six that were held back. Each row predicts the six held-out frames from the twelve observed ones.

<!-- worked: project_dynamic_scene.opening -->
Six held-out frames (frames 12 .. 17) of the synthetic scene of milestone 1, predicted in seven ways, scored by the mean over the six of the squared error per pixel and channel against the clean true frame. 'Training loss' is the loss of the fit after 120 updates on the 12 observed frames; the last column is the largest distance, over the three balls, of the predicted floor position from the true one at frame 17. Every trained number carries the digits that the run and its four nudged copies share.

| how the frames were predicted | training loss | held-out error | worst ball error at frame 17 (m) |
|---|---:|---:|---:|
| the pipeline as specified | 4.89e-04 | 4.23e-04 | 0.407 |
| the same without the clamp on the rollout | 4.89e-04 | 1.02e-03 | 0.749 |
| a principal point by the centre rule (half a pixel) | 4.70e-04 | 5.34e-04 | 0.407 |
| cameras exported with R transposed | 9.97e-04 | 2.42e-02 | 1.750 |
| balls held at their last detection | 4.89e-04 | 1.03e-03 | 1.305 |
| balls at their true positions (reference) | 4.89e-04 | 3.45e-04 | 0.000 |
| the last observed frame, copied | - | 1.29e-03 | - |
<!-- /worked -->

The first row is the pipeline this chapter builds. Its error on the held-out frames is 4.23e-04, a third of the 1.29e-03 of the lazy answer (the last observed frame, copied) and 1.23 times the 3.45e-04 of the reference that has the balls in their true places and so measures only what the Gaussians cost. The second row is the same pipeline with one line removed, the clip that keeps a ball on the table, and its error is 2.4 times larger: no better than leaving every ball where it was last seen. The third row changes the principal point by half a pixel and leaves the training loss where it was, even a little lower; its held-out error is 26 percent higher. The fourth exports the rotation of every camera transposed, and trains to a loss only twice the correct one, 9.97e-04 against 4.89e-04, while its held-out error is 57 times the correct one. Nothing in the training loss gives the second and third rows away, and the fourth only by a factor of two.

The numbers on those rows are not what this chapter is about. Anyone can compute a score. What takes work is a loop in which every piece has a number of its own that says it is right, so that a wrong piece cannot hide inside a good total. The twelve frames come from a world whose answers are known to the metre, and the chapter builds the loop out of the pieces of the volume in the order a scene is made: a camera, a path and a sparse structure from the photographs, a set of Gaussians fitted to the frames, a dynamics model of the moving things fitted to what the reconstruction says about them, and a render of the prediction scored against frames that no step saw.

![A two-row diagram. Upper row, left to right: twelve observed frames with 43 feature tracks, structure from motion giving cameras, forty points, a floor and ball detections, the export of the cameras to the renderer (marked as the place of mistakes 1 and 2), and the fit of ten static Gaussians and three balls to the twelve frames. A path leads from the fit down to the lower row, labelled the ball positions of the twelve frames. Lower row, left to right: a dynamics model of each ball, a free run of six frames clipped to the clamp box (marked as the place of mistake 3), a render of the Gaussians at each held-out camera, and the squared error against the held-out frame.](img/scene-loop.svg)

The boxes are the pieces of this volume. [Chapter 3](03-camera-geometry.md) gave the lens and the look-at camera, [Chapter 4](04-structure-from-motion.md) the essential matrix, the registration of a camera against known points and the adjustment of everything at once, [Chapter 6](06-gaussian-splatting.md) the Gaussian and a renderer that can be run backwards, and [Chapter 7](07-world-models.md) the learned linear model of a cart that does not know about walls. `scripts/book/worked/project_dynamic_scene.py`, the module that produced every number below, imports the worked modules of those chapters instead of copying them: the lens and the look-at of Chapter 3 (`camera_geometry`), the eight-point algorithm with RANSAC, the triangulation, the DLT absolute pose and the adjustment of Chapter 4 (`structure_from_motion`), the Gaussian, the renderer and its analytic gradient of Chapter 6 (`gaussian_splatting`), and the elimination of Chapter 7 (`world_models`), with the book's generator and the stable-digits rule. What is written here is the glue: how a ball is turned into a position, how the structure turns into a first set of Gaussians, which numbers the fit may change, what the dynamics model is, and what the planted mistakes are.

You build it in any language you like. You need double-precision arithmetic with a correctly rounded `sqrt`, `exp`, `log`, `sin` and `cos`, the book's generator ([Volume I, Chapter 17](../v01-mathematics/17-randomness.md)), and the routines of the four chapters above, which this chapter names and does not teach again. Six questions bring back what the milestones lean on; the answers are at the end.

1. (Chapter 3, the pose.) A camera is stored as the pair $(\mathbf{R}, \mathbf{t})$ with $\mathbf{x}_{\mathrm{c}} = \mathbf{R}\mathbf{x}_{\mathrm{w}} + \mathbf{t}$. Where is its centre? What does a program that uses $\mathbf{R}^\top$ where $\mathbf{R}$ belongs do to the rotation, and which world point projects correctly anyway?
2. (Chapter 3, which half pixel.) A lens of 640 by 480 pixels has its principal point at $(320, 240)$ when pixel $i$ covers $[i, i + 1)$. What is the principal point of the same lens at 32 by 24 pixels? What does the rule $c' = s(c + \tfrac12) - \tfrac12$ give instead, and by how many pixels is it wrong?
3. (Chapter 4, the gauge.) A set of $N_{\mathrm{c}}$ photographs and $P$ points is adjusted with one camera held and one camera kept at a fixed distance from it. How many camera unknowns does the adjustment have, and how many unknowns in all, for 12 cameras and 40 points?
4. (Chapter 6, the Gaussian and its gradient.) How many numbers describe one Gaussian, which of them are stored as logarithms or logits, and why is the gradient of a pixel loss with respect to all of them not exact at the cut-offs of the renderer?
5. (Chapter 7, the cart and the wall.) A linear model of a cart on a track is fitted to one-step transitions and then run on its own output. What does the error do as the run grows, and what does the model not know about the track?
6. (Volume II, [Chapter 2](../v02-learning/02-linear-regression.md) and [Chapter 4](../v02-learning/04-likelihood.md), least squares and its penalty.) What are the normal equations of a least-squares fit, what goes wrong with them when two features are linear functions of each other, and what does adding $\lambda$ to a diagonal entry change?

## The rules every number follows

Every random input comes from the book's SplitMix64 at the seed `seed_of(name)` printed beside its table, taken in the order the text states. Arithmetic is in 64-bit floating point. A trained number is printed only to the digits that survive the nudge rule of the book: the fit is run again from four starting points whose first mean number differs by $+10^{-12}$, $-10^{-12}$, $+3 \times 10^{-12}$ and $-3 \times 10^{-12}$, and a digit is printed only if all five runs agree on it. The fit here is not chaotic (all five runs agree on every digit the tables show), and the rule is what lets the tables say so. The numbers of structure from motion and of the dynamics are not trained: they come from RANSAC with a named seed, a Levenberg-Marquardt iteration that stops at its own criterion, and a direct solve, and they are printed to the digits another correct implementation will also get. Where a number is a count of operations or bytes it is exact.

Distances are in metres. The world frame is the frame of camera 0: $x$ to the right, $y$ down, $z$ forward, as in Chapter 3.

## Building the dynamic scene in eight steps

The pseudocode follows [Pseudocode Conventions](../pseudocode.md). Build and check each milestone before the next: a disagreement at milestone 6 that follows agreement at 1 to 5 points at the prediction and not at the reconstruction.

### Milestone 1: a world that is filmed

The world has to be built before it can be reconstructed, and it has to be built so that the answers are known. The block below states the whole of it: the lens, the 18 cameras, the three balls, the true scene of Gaussians that the frames are renders of, the noise of the frames, and the feature tracks.

A few decisions deserve their reasons. The frames are 32 by 24 pixels, the 640 by 480 lens divided by 20, because a render of 13 Gaussians has to be cheap enough to run, with its backward pass, 120 times over 12 frames in the reader's own implementation; the tracks, which are what structure from motion reads, are at the full 640 by 480 and accurate to a quarter of a pixel, so that they resolve the scene far more finely than the frames do. The cameras slide along a line that sinks a little and moves forward a little (the $y$ axis points down), all looking at one point on the table, so that camera 0 is the identity and the world is its frame, and the baseline between two cameras four frames apart is 0.59 metres, less than a tenth of the depth of the wall: enough for structure from motion, and a baseline that makes depth the weak coordinate, as it is in life. And the balls are three settings of one recurrence, which is what makes the dynamics of milestone 5 a single model:

$$
v' = \eta\,\bigl(v - k_{\mathrm{s}}(x - x_{\mathrm{c}})\bigr), \qquad x' = x + v',
$$

with a damping factor $\eta$, a stiffness $k_{\mathrm{s}}$ and an anchor $x_{\mathrm{c}}$, taken once for each axis. Ball A is a spring (in $x$, $\eta = 0.995$, $k_{\mathrm{s}} = 0.12$, $x_{\mathrm{c}} = 0$; in $z$, $k_{\mathrm{s}} = 0.08$ and $x_{\mathrm{c}} = 5.2$). Ball B is a glider, $k_{\mathrm{s}} = 0$ and $\eta = 0.9$: it slows by a tenth each frame. Ball C is a slider, $k_{\mathrm{s}} = 0$ and $\eta = 1$, at 0.15 metres a frame, with one more rule that the recurrence does not contain: the table has an edge, and the centre of a ball of radius 0.3 cannot pass $x = 0.9$. It reaches it at frame 12.

<!-- worked: project_dynamic_scene.world_setup -->
Lens: f = 500 pixels, principal point (320, 240), image 640 by 480 pixels, pixel i covering [i, i + 1) (the corner convention of Chapter 3), no distortion. The splat images are the same lens divided by 20 with the corner rule c' = s c: 32 by 24 pixels, f = 25, principal point (16, 12). Camera i, i = 0 .. 17, has its centre at (0.14 i, 0.02 i, 0.04 i) metres, looks at (0, 0, 5.2) with up (0, -1, 0) (the `look_at` of Chapter 3), so camera 0 is the identity and the frame of camera 0 is the world (x right, y down, z forward). The first 12 frames are observed, the last 6 are held out.

| frame | centre C (metres) | angle of R (degrees) | role |
|---:|---:|---:|---|
| 0 | (0, 0, 0) | 0 | observed |
| 4 | (0.56, 0.08, 0.16) | 6.404 | observed |
| 11 | (1.54, 0.22, 0.44) | 18.1 | observed |
| 12 | (1.68, 0.24, 0.48) | 19.78 | held out |
| 17 | (2.38, 0.34, 0.68) | 28.02 | held out |

The table is the rectangle x in [-1.2, 1.2], z in [4, 6.4] at height y = 1.5; a ball of radius 0.3 rests on it, so its centre is at y = 1.2. Floor positions (x, z) of the three balls, from the recurrences (the recurrence v' = eta (v - k (x - c)), x' = x + v' of the chapter, with the settings below, and the stop of C at x = 0.9):

| frame | A: spring | B: glider | C: slider |
|---:|---|---|---|
| 0 | (-0.7, 4.6) | (-0.9, 4.5) | (-0.9, 6) |
| 6 | (0.742, 5.529) | (-0.4783, 4.964) | (0, 6) |
| 11 | (0.24, 5.79) | (-0.2824, 5.179) | (0.75, 6) |
| 12 | (-0.04135, 5.698) | (-0.2542, 5.21) | (0.9, 6) |
| 17 | (-0.7883, 4.893) | (-0.1501, 5.325) | (0.9, 6) |

Ball A: x from x_0 = -0.7, v_0 = 0.12, k = 0.12, eta = 0.995, c = 0; z from z_0 = 4.6, v_0 = 0.05, k = 0.08, eta = 0.995, c = 5.2. Ball B (k = 0): x from -0.9 with v_0 = 0.10 and z from 4.5 with v_0 = 0.11, both with eta = 0.9. Ball C (k = 0, eta = 1): x = min(-0.9 + 0.15 t, 0.9), z = 6.0. A frame is the render (`gaussian_splatting.render`, background (0.55, 0.65, 0.8)) of the true scene: 14 Gaussians for the wall and floor (`project_dynamic_scene.true_scene`, seed 1689292279692065492: for each Gaussian in turn, wall (y = -0.8 then 0.6, x = -1.4, -0.1, 1.2, 2.5) then floor (z = 4.8 then 5.8, x = -0.8, 0, 0.8), three `uniform()` draws for the colour, 0.15 + 0.7 u on the wall and 0.3 + 0.4 u on the floor), scales (0.9, 0.8, 0.1) on the wall at z = 8 and (0.5, 0.04, 0.5) on the floor at y = 1.5, opacity 0.95, identity quaternion, and three balls of scale 0.3 on all axes, opacity 0.95 and colours (0.9, 0.2, 0.2), (0.2, 0.8, 0.3), (0.9, 0.8, 0.2).

| Gaussian | part | mean | scales | colour |
|---:|---|---|---|---|
| 0 | wall | (-1.4, -0.8, 8) | (0.9, 0.8, 0.1) | (0.2629, 0.557, 0.3296) |
| 1 | wall | (-0.1, -0.8, 8) | (0.9, 0.8, 0.1) | (0.3275, 0.3384, 0.7744) |
| 7 | wall | (2.5, 0.6, 8) | (0.9, 0.8, 0.1) | (0.336, 0.6404, 0.3056) |
| 8 | floor | (-0.8, 1.5, 4.8) | (0.5, 0.04, 0.5) | (0.6836, 0.6669, 0.4081) |
| 13 | floor | (0.8, 1.5, 5.8) | (0.5, 0.04, 0.5) | (0.3512, 0.5839, 0.5004) |

The observed frames carry noise: `project_dynamic_scene.image_noise` (seed 17931157441936962617), frame by frame, row by row, column by column, channel by channel, `normal()` times 0.01 added to the clean value. The sum of the 2,304 values of clean frame 0 is 1389.157309, of clean frame 11 1385.041891, of observed frame 0 1389.728264.

The feature tracks: 40 landmarks and the 3 ball centres (the points (x_t, 1.2, z_t)). The landmarks come from `project_dynamic_scene.landmarks` (seed 2783470472750872843): 20 wall points, three `uniform()` draws each in the order x = -1.8 + 4.2 u, y = -1.6 + 2.6 u, z = 7.8 + 0.4 u; then the four corners of the table in the order (xmin, zmin), (xmin, zmax), (xmax, zmin), (xmax, zmax); then 16 floor points, two draws each, x = -1.2 + 2.4 u, z = 4.0 + 2.4 u. The pixels are the projections plus noise from `project_dynamic_scene.track_noise` (seed 9793457138374083996), 0.25 pixel per coordinate, view by view, track by track, `normal()` for u then `normal()` for v. Landmark 0 is at (-0.2307, -0.70093, 8.1902); its pixel in view 0 is (305.6813, 197.2169) and in view 11 (364.2223, 203.9184); the pixel of ball A in view 0 is (243.9071, 370.1808).
<!-- /worked -->

**Milestone 1 ends** when your 18 cameras, your ball paths and your renders reproduce the sums and the pixels of the first rows of the acceptance table (sums of frame 0 to six decimals, the pixel of landmark 0 to four). Nothing that follows can be checked if the frames are not the same frames.

### Milestone 2: cameras and a sparse structure from twelve frames

Now forget the world. What the reader's pipeline has is twelve frames and the tracks, and it must recover from them the cameras, a set of points and the one plane the dynamics will need, the floor. The recipe is that of Chapter 4, with every choice fixed.

The pair is frames 0 and 4, a baseline wide enough to triangulate and close enough that every track is seen in both. An essential matrix is fitted by RANSAC to the 43 normalized correspondences (the eight-point algorithm inside, a Sampson threshold of 1.5 pixels, at most 500 hypotheses, the adaptive bound of Chapter 4), the pose is chosen by the four-way test, and the inlier tracks are triangulated. The tracks that the model rejects should be the three balls, which is the first thing to check: a rigid-scene model does not explain a point that moves, and the pair's inliers are the static scene. (That is what the geometry gives in this world, not a guarantee. A point that moves along the direction of the camera's own motion can stay on the epipolar line of a static one, and then the pair cannot tell it from a point at rest.) The other ten frames are registered one at a time: a 6-point DLT inside RANSAC against the pair's points (threshold 4 pixels, at most 2000 hypotheses), then 30 iterations of Levenberg-Marquardt on the inliers with the points held. Then everything at once, with the gauge fixed in the way Chapter 4 fixed it: frame 0 is held, frame 4 is kept at its unit baseline from it, the ten other cameras are free, and the Huber threshold is 2 pixels. Finally a gate on what the pair let through: any track whose median residual after the adjustment is above 2 pixels is dropped, and the problem is solved again. It removes nothing in this world, since all 40 tracks pass; it is there because a moving point that the pair had accepted would pull the cameras toward itself, and nothing in this scene tests whether a threshold of 2 pixels would catch one.

The adjustment leaves the cameras in a world whose only length is the unit baseline. The one length the reader is told, the 0.5879 metres between frames 0 and 4 (as if from the odometer of the vehicle the camera is on), multiplies every translation and every point by that number, which is a similarity and moves no reprojection.

<!-- worked: project_dynamic_scene.sparse_structure -->
Frames 0 and 4 give the pair (the distance of their centres, 0.5879 metres, is the one length the reader is told). `structure_from_motion.ransac_essential` on the 43 normalized correspondences, threshold 1.5 pixel, cap 500, generator `project_dynamic_scene.ransac` (seed 1397186363808094706): 9 iterations (the adaptive bound at the end is 9), 40 inliers of 43, the tracks outside the model are [40, 41, 42] (tracks 40, 41, 42 are the balls), the pose has the rotation angle 6.200 degrees (true 6.404) and the translation direction 1.899 degrees from the true one. The inlier tracks are triangulated (`structure_from_motion.triangulate`, linear) from the pair and kept if in front of both cameras. The sampling is exactly this. A sample is 8 distinct track indices: `integer(43)` draws are taken from the generator one after another and a draw equal to an index already in the sample is discarded and drawn again, until 8 are held, in draw order. The eight-point algorithm (without refinement) gives a matrix from the sample; every one of the 43 correspondences is scored by its Sampson distance in pixels d, and the hypothesis scores the sum of min(d^2, 1.5^2) (lower is better); a hypothesis replaces the best only when its score is strictly lower, and each time it does the cap is lowered to min(cap, the adaptive bound of the inlier fraction of the new best, 8-point samples), with the loop ending when the number of hypotheses tried reaches the cap. The inliers of the best (Sampson distance below the threshold) are fitted again by the eight-point algorithm; the pose of that essential matrix is the one of the four that puts the most inlier points in front of both cameras; and two rounds of Levenberg-Marquardt on the inliers' Sampson distances refine it, the inliers being recomputed from the refined pose after each round (the second round's inliers are the mask above).

The other ten frames are registered one at a time by `structure_from_motion.ransac_pnp` on those points (6-point DLT, threshold 4 pixels, cap 2000, generator `project_dynamic_scene.ransac.pnp.<v>` for frame v), each refined by 30 Levenberg-Marquardt iterations on its inliers with the points held. Frames are registered in increasing order, 1, 2, 3, 5, ..., 11 (frames 0 and 4 are the pair), each from a fresh generator of its own name; a sample is 6 distinct indices of the pair's points drawn as above with `integer(n)`, n the number of triangulated points; the 6-point DLT gives a pose; the score is the sum of min(d^2, 4^2) of the reprojection distances d in pixels, replaced only by a strictly lower one, with the cap lowered as above (sample size 6); the best pose's inliers (reprojection below the threshold) are the points of the refinement:

| frame | iterations | inliers |
|---:|---:|---|
| 1 | 12 | 39 of 40 |
| 2 | 7 | 40 of 40 |
| 3 | 3 | 40 of 40 |
| 5 | 18 | 40 of 40 |
| 6 | 26 | 36 of 40 |
| 7 | 41 | 35 of 40 |
| 8 | 10 | 36 of 40 |
| 9 | 142 | 39 of 40 |
| 10 | 29 | 31 of 40 |
| 11 | 55 | 33 of 40 |

The adjustment (`structure_from_motion.Problem`: frame 0 held, frame 4 the scale camera, the other ten free, Huber threshold 2 pixels, lambda starting at 1e-3). Each inlier track is triangulated linearly from the observations of the frames whose reprojection under the registered pose is within 8 pixels. Then 9 iterations (RMS reprojection error 0.375 to 0.336 pixels); the median residual of every track is then below 2 pixels (the gate of this milestone), so none is dropped and the second solve has 1 accepted iteration(s). 40 points, 480 observations, 65 camera unknowns and 120 point coordinates; the expected RMS of a fit with the noise 0.25 px is sigma sqrt(2) sqrt((2 n - p) / (2 n)) = 0.318, the adjustment reaches 0.336.

The baseline of the pair is set to its length, so every translation and every point is multiplied by it. Against the truth:

| frame | rotation error (degrees) | centre error (metres) |
|---:|---:|---:|
| 1 | 0.072 | 0.0060 |
| 2 | 0.125 | 0.0119 |
| 3 | 0.094 | 0.0038 |
| 4 | 0.085 | 0.0059 |
| 5 | 0.168 | 0.0059 |
| 6 | 0.186 | 0.0057 |
| 7 | 0.171 | 0.0055 |
| 8 | 0.258 | 0.0084 |
| 9 | 0.236 | 0.0075 |
| 10 | 0.233 | 0.0031 |
| 11 | 0.215 | 0.0056 |

The depth of the points, recovered over true, has mean 0.9842 (range 0.9776 to 0.9916): the structure is about 1.6% too near while the camera centres are within 0.012 metres: the one measured baseline fixes the scale of the cameras, not the depth of the points. The floor plane (`floor_plane`: for each point in turn the points within 0.08 metres in y; the largest set, the first on a tie) holds 20 of the 40 points at y = 1.4785 (true 1.5). Their bounding box in x and z is the table, [-1.1845, 1.1877] by [3.9500, 6.3038]; inset by the ball radius 0.3 it is the clamp box [-0.8845, 0.8877] by [4.2500, 6.0038]. A ball centre rests 0.3 above the floor, y = 1.1785; the ray through a ball track's pixel meets that plane at the detected position (`ray_on_plane`), and the mean distance of the detections from the true positions over the 12 frames is 0.099, 0.092 and 0.111 metres for A, B and C.
<!-- /worked -->

Two numbers in that block are the reason for the care taken over it. The cameras are good: the largest rotation error is a quarter of a degree and the largest centre error 1.2 centimetres, and the reprojection error 0.336 pixels is within 6 percent of the 0.318 that the noise alone allows. The depth is not: the recovered points are 1.6 percent nearer than the true ones, and everything measured from them (the floor at 1.4785 metres where the truth is 1.5, the ball positions that are read off it) carries the bias. The one measured baseline fixes the scale of the cameras; it does not fix the depth of the points. The rest of the chapter works inside the reconstruction's own metres, and the 0.1 metre by which a ball detection is off (1.6 percent of a distance of 5 to 6 metres is 0.09, and the floor is 2 centimetres off as well) will be in every position error below.

The detection of a ball is one more use of the camera. Its track is a pixel in each frame; the ball rests on the floor, its centre is the ball's radius $r_{\mathrm{b}}$ above the plane $y = y_{\mathrm{f}}$, and the pixel's ray meets the plane $y = y_{\mathrm{f}} - r_{\mathrm{b}}$ in one point. The floor itself is found among the points, by consensus: the horizontal plane with the most points within 0.08 metres.

```text
function floor_plane(points: list<array<real>[3]>, tol: real) -> (real, list<int>):
    require len(points) >= 1, "at least one point"
    best = []
    for p in points:
        on = []
        for k in 0 .. len(points):
            if abs(points[k][1] - p[1]) < tol:
                append k to on
        if len(on) > len(best):
            best = on
    h = 0.0
    for k in best:
        h = h + points[k][1]
    return (h / len(best), best)

function ray_on_plane(R: array<real>[3, 3], t: array<real>[3], u: real, v: real,
                      y_plane: real, f: real, cx: real, cy: real) -> array<real>[3]:
    C = -matmul(transpose(R), t)
    d = matmul(transpose(R), [(u - cx) / f, (v - cy) / f, 1.0])
    s = (y_plane - C[1]) / d[1]
    require s > 0.0, "the plane is in front of the camera"
    return C + s * d
```

The plane is horizontal because camera 0 is level, so that the $y$ axis of the world is vertical; a tripod makes the same assumption, and a handheld camera needs the gravity direction that the cameras themselves give (see the last section). Ties between planes go to the first candidate. The bounding box of the floor's points in $x$ and $z$ is the table as the reconstruction knows it, and inset by the ball's radius it is the **clamp box**: the region in which a ball's centre can be. That box is the one fact about the world that the dynamics will be given and cannot learn from motion, and it comes from the reconstruction, not from the dynamics. The detection error that the ray adds from the pixels alone is small: the depth of the point where a ray meets a plane at height $h$ below the camera moves by about $z^2\,\delta v/(f h)$ for a vertical pixel error $\delta v$, which for a quarter of a pixel at $z = 5.5$ and $h = 1.1$ is 1.4 centimetres.

**Milestone 2 ends** when your pair has 9 iterations, 40 inliers and the tracks 40, 41 and 42 outside the model, your final reprojection error is 0.336 pixels to the digits shown, and your floor and clamp box agree with the block.

### Milestone 3: a first set of Gaussians from the structure

A fit by gradient descent depends on where it starts. The start here is what the structure gives and no more: the points off the floor are the wall, the points on it are the table, and each is cut into a handful of chunks, one Gaussian to a chunk. The recipe in the block is the one to follow digit by digit, since the loss of the starting scene is a number to match.

The reasons for the choices: the chunks are cut by $x$, a rule that needs nothing but a sort; the wall's scale along the optical axis is small and fixed (0.15) because the points say where the wall is but not how thick; the floor's scale across is 0.05, a thin sheet; and a colour is read from the pixel of frame 0 under the centre. The balls start at their first detection with a common radius of 0.25 and that pixel's colour. Nothing about this is the right initialization; it is a fixed, checkable one. The reconstruction of brain starts from every point of the cloud (see the last section), and the project's ten Gaussians stand for that cloud at the size of a page.

<!-- worked: project_dynamic_scene.initial_gaussians -->
From the structure of milestone 2. The points off the floor plane are the wall: sorted by x (ties by index) and cut into 6 chunks of consecutive points (chunk k holds the points from floor(k n / 6) up to floor((k + 1) n / 6), n = 20), one Gaussian at the mean of each. The 20 floor points are cut by x into 4 chunks the same way. Scales (before the exponential): wall (max(1.2 s_x, 0.3), max(1.2 s_y, 0.3), 0.15), floor (max(1.2 s_x, 0.3), 0.05, max(1.2 s_z, 0.4)), with s the standard deviation of the chunk (divisor n). Identity quaternion, opacity 0.8 (logit ln 4), and the colour of the pixel of observed frame 0 that the centre projects to (clipped to [0.05, 0.95]). The balls start at the detection of frame 0 with scale 0.25 on all three axes, opacity 0.8 and the colour of the pixel of frame 0 under it.

| Gaussian | part | mean | scales | colour |
|---:|---|---|---|---|
| 0 | wall | (-0.819, -1.02, 7.94) | (0.494, 0.35, 0.15) | (0.321, 0.528, 0.451) |
| 1 | wall | (-0.184, -0.463, 7.99) | (0.3, 0.358, 0.15) | (0.337, 0.444, 0.6) |
| 2 | wall | (0.177, -0.783, 7.98) | (0.3, 1.03, 0.15) | (0.335, 0.394, 0.665) |
| 3 | wall | (1.01, -0.143, 7.96) | (0.3, 0.997, 0.15) | (0.611, 0.336, 0.662) |
| 4 | wall | (1.45, -0.313, 7.78) | (0.302, 0.525, 0.15) | (0.667, 0.296, 0.663) |
| 5 | wall | (2.08, -0.729, 7.79) | (0.3, 0.832, 0.15) | (0.726, 0.249, 0.679) |
| 6 | floor | (-1.08, 1.48, 4.98) | (0.3, 0.05, 0.98) | (0.294, 0.734, 0.306) |
| 7 | floor | (-0.503, 1.48, 5.22) | (0.3, 0.05, 0.841) | (0.625, 0.519, 0.278) |
| 8 | floor | (0.157, 1.48, 5.42) | (0.363, 0.05, 0.906) | (0.543, 0.571, 0.472) |
| 9 | floor | (0.927, 1.48, 4.81) | (0.341, 0.05, 0.955) | (0.411, 0.634, 0.532) |

| ball | detection at frame 0 (x, z) | scale | colour |
|---|---|---|---|
| A (spring) | (-0.689, 4.526) | 0.25 | (0.421, 0.598, 0.288) |
| B (glider) | (-0.887, 4.428) | 0.25 | (0.238, 0.754, 0.302) |
| C (slider) | (-0.880, 5.857) | 0.25 | (0.494, 0.579, 0.284) |

The loss of the starting scene (the mean over the 12 observed frames of the mean squared error over pixels and channels) is 0.009656. One render of one frame visits every record at every pixel, 9984 pairs of the 768 pixels and 13 Gaussians, 119808 for the 12 frames.
<!-- /worked -->

A render of one frame visits every one of the 13 Gaussians at every one of the 768 pixels: 9,984 pairs, for as long as no Gaussian is culled. That is a fact about the size of the image: 32 by 24 pixels fit in four tiles of 16 by 16, and the tiling of Chapter 6, which exists to avoid pairs of far-apart pixels and Gaussians, has nothing to avoid at this size. It is also the reason the bill of milestone 8 can be counted exactly.

**Milestone 3 ends** when your starting loss is 0.009656 and your 13 Gaussians agree with the tables to the digits shown.

### Milestone 4: fitting the Gaussians to the frames

The fit changes two things: the 14 numbers of each of the ten static Gaussians (the mean, three log scales, the raw quaternion, the opacity logit and the colour, as in Chapter 6) and five numbers of each ball: the log scale (one number, shared by its three axes, whose gradient is the sum of the three), the opacity logit and the colour. The positions of the balls are not parameters. They stay at the detections, because the tracks locate a ball in a 640 pixel wide image and the frames show it in a 32 pixel wide one, and a fit of the position against the picture has twenty times less to go on (the second extension lets the fit move them). The cameras are the exported cameras of the reconstruction: for camera $i$ the pair $(\mathbf{R}_i, \mathbf{t}_i)$ of milestone 2, in metres, with the lens divided by 20 by the corner rule, $c' = s\,c$ for $s = \tfrac1{20}$, so that the principal point is $(16, 12)$ and the focal length 25.

The loss of an update is the mean over the 12 observed frames of the mean squared error of the frame over pixels and channels, whose gradient is `view_gradient` of Chapter 6 for each frame, summed, divided by 12. Adam is that of [Volume III, Chapter 7](../v03-neural-networks/07-optimizers.md): $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$ with bias correction, full batch, 120 updates, and rates twice those of Chapter 6's fit (the book's rates for the mean, log scale, quaternion, opacity logit and colour are 0.02, 0.03, 0.01, 0.1 and 0.03) because 120 updates stand where that fit used 300. The balls take 0.06 for the log scale, 0.2 for the logit and 0.06 for the colour. The 120th update is followed by one more evaluation of the loss, which is the number the table calls the loss after 120 updates.

```text
function fit_scene(cams: list<Camera>, images: list<array<real>[24, 32, 3]>,
                   static: list<array<real>[14]>, app: list<array<real>[5]>,
                   pos: list<list<array<real>[2]>>, y_ball: real, sky: array<real>[3],
                   steps: int) -> (list<array<real>[14]>, list<array<real>[5]>):
    ms, vs, ma, va = zeros of the shapes of static and app
    for step in 0 ..= steps:
        Gs, Ga = zeros of the shapes of static and app
        for t in 0 .. 12:
            params = static followed by, for each ball j,
                     [pos[j][t][0], y_ball, pos[j][t][1], app[j][0], app[j][0], app[j][0],
                      1.0, 0.0, 0.0, 0.0, app[j][1], app[j][2], app[j][3], app[j][4]]
            (l, g) = view_gradient(params, cams[t], mse_upstream(images[t]), sky)
            for i in 0 .. 10:
                Gs[i] = Gs[i] + g[i] / 12
            for j in 0 .. 3:
                Ga[j][0] = Ga[j][0] + (g[10 + j][3] + g[10 + j][4] + g[10 + j][5]) / 12
                Ga[j][1] = Ga[j][1] + g[10 + j][10] / 12
                Ga[j][2:5] = Ga[j][2:5] + g[10 + j][11:14] / 12
        if step == steps:
            break
        adam update of every number of static and app with the rates above, step + 1
    return (static, app)
```

`mse_upstream(image)` is the function of a pixel $(x, y)$ and its rendered colour $\mathbf{c}$ that returns the pixel's loss $\sum_k (c_k - I_{y,x,k})^2/(3P)$ and its gradient with respect to the colour, $2(c_k - I_{y,x,k})/(3P)$, with $P$ the number of pixels of the frame; `view_gradient(params, camera, upstream, sky)` of Chapter 6 renders the frame, sums the losses over its pixels and returns that sum with the gradient with respect to every Gaussian's numbers; the pixel $(x, y)$ is rendered at its centre $(x + 0.5, y + 0.5)$; and `sky` the background colour the renderer composites onto, (0.55, 0.65, 0.80). The sum of the three log-scale entries is the gradient of the one shared number. A ball's quaternion is the identity and is never changed: a sphere has no orientation.

<!-- worked: project_dynamic_scene.fit_curve -->
Full-batch Adam (beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-8, bias correction), 120 updates, on the 10 x 14 numbers of the static Gaussians (rates twice those of Chapter 6: mean 0.04, log scale 0.06, quaternion 0.02, opacity logit 0.2, colour 0.06) and the 3 x 5 appearance numbers of the balls (log scale, opacity logit, colour, at 0.06, 0.2, 0.06). The loss of an update is the mean over the 12 observed frames of the frame's mean squared error over pixels and channels; the gradient is `gaussian_splatting.view_gradient`, summed over the frames and divided by 12; the three numbers of a ball's log scale share one gradient (the sum of the three), and the balls' positions stay at the detections. Under the stable-digits rule the run is repeated with the first mean number of the first Gaussian changed by +1e-12, -1e-12, +3e-12 and -3e-12. A number that comes from a chaotic run is printed with only the digits that are the same in the run and in four copies of it whose first weight is changed by +1e-12, -1e-12, +3e-12 and -3e-12: in full if all agree on it, as 'about x' with the leading digits on which all agree, or as 'unstable' when not even one digit survives; the other digits cannot be reproduced in another implementation.

| updates | training loss |
|---:|---:|
| 0 | 9.656e-03 |
| 10 | 2.683e-03 |
| 30 | 1.048e-03 |
| 60 | 5.984e-04 |
| 120 | 4.888e-04 |

The balls' fitted appearance (scale is the exponential of the log scale; the true ball has scale 0.3 and opacity 0.95):

| ball | scale | opacity | colour (r, g, b) | true colour |
|---|---:|---:|---|---|
| A (spring) | 0.2738 | 0.4300 | 1.1662, -0.1362, -0.1190 | (0.9, 0.2, 0.2) |
| B (glider) | 0.3119 | 0.9966 | 0.3146, 0.7431, 0.3331 | (0.2, 0.8, 0.3) |
| C (slider) | 0.3392 | 0.7871 | 0.9091, 0.6995, 0.2108 | (0.9, 0.8, 0.2) |
<!-- /worked -->

The loss falls by a factor of 3.6 in the first ten updates and by 4.5 more over the next fifty; the last sixty buy a further 18 percent. What it reaches, $4.89 \times 10^{-4}$, is not the floor of the noise, which is $10^{-4}$ (the square of the 0.01 added to every value): ten Gaussians for a world of fourteen will not reach it, and that remainder is the model error that the held-out frames will inherit. The fitted balls are worth a look. Colours are not clamped in the fit (Chapter 6), and ball A shows it: an opacity of 0.43 and a colour of $(1.17, -0.14, -0.12)$ against a true $(0.9, 0.2, 0.2)$ at 0.95. A fitted parameter is not a measurement of the object that the picture contains. Ball B has a scale of 0.31 (true 0.3), an opacity of 0.9966 (true 0.95) and a colour near the true one.

**Milestone 4 ends** when your loss at updates 0, 10, 30, 60 and 120 matches the table to the digits shown.

### Milestone 5: a dynamics model for three balls

What the reconstruction says about the balls is their positions in the twelve frames. A model of their motion is a function from the recent past to the next position, and the recurrence of milestone 1 tells what family it should be in: with the velocity written as a difference of positions, $v = x - q$ for $q$ the position one frame earlier, the update is $x' = x + \eta(x - q) - \eta k_{\mathrm{s}}(x - x_{\mathrm{c}})$. That is

$$
p_{t+1} = c + a\,p_t + b\,p_{t-1}, \qquad a = 1 + \eta - \eta k_{\mathrm{s}}, \quad b = -\eta, \quad c = \eta k_{\mathrm{s}} x_{\mathrm{c}},
$$

for one coordinate $p$ of a ball. A spring, a glider ($a = 1 + \eta$, $b = -\eta$, $c = 0$) and a slider ($a = 2$, $b = -1$, $c = 0$) are three points of one three-parameter family, and the dynamics model of the project is that family: one $(a, b, c)$ for each coordinate of each ball, fitted by least squares to the detections of the twelve frames, ten transitions to a coordinate. A spring's coefficients can be read against the recurrence, which gives the check that the fit is doing what is claimed: ball A in $x$ should have $a$ near $1.8756$ and $b$ near $-0.995$.

Two of the three balls make that fit ill-posed, and the reader should expect it. The slider's $x$ goes up by the same step every frame, so that $p_t = p_{t-1} + 0.15$ and the three features $(p_t, p_{t-1}, 1)$ are linearly dependent up to the noise of the detections; the glider's steps have shrunk to a few hundredths of a metre over the window, so that $p_t$ and $p_{t-1}$ are nearly equal. The normal equations of such a fit are nearly singular, and the solution they give is whatever the noise in the data prefers. The fix of Volume II is the penalty: add $\lambda_{\mathrm{r}}$ to the diagonal entries of $a$ and $b$ (not of $c$) in the normal equations, and the smallest weights that fit are the ones chosen. $\lambda_{\mathrm{r}} = 10^{-4}$ is small enough to leave a well-posed fit alone, as the spring's is, and large enough that the solve does not depend on the last bits of a sum.

```text
function fit_ar2(series: array<real>[T], ridge: real) -> (real, real, real):
    require T >= 4, "at least two transitions"
    A = zeros(real, [3, 3])
    y = zeros(real, [3])
    for t in 1 .. T - 1:
        x = [series[t], series[t - 1], 1.0]
        for i in 0 .. 3:
            y[i] = y[i] + x[i] * series[t + 1]
            for j in 0 .. 3:
                A[i, j] = A[i, j] + x[i] * x[j]
    A[0, 0] = A[0, 0] + ridge
    A[1, 1] = A[1, 1] + ridge
    w = solve(A, y, true)
    return (w[0], w[1], w[2])
```

The sums run over the transitions in order; `solve(A, y, true)` is Gauss-Jordan elimination with partial pivoting, as in Chapter 7. The detections are the series (twelve values for each coordinate of each ball).

Running the model is the other half. Starting from the last two observed positions, each frame is fed the model's own output; and the one piece of knowledge the dynamics cannot learn is applied after every step: the position is clipped to the clamp box of milestone 2.

```text
function rollout(model: list<(real, real, real)>, p: array<real>[2], q: array<real>[2],
                 steps: int, lo: array<real>[2], hi: array<real>[2], clip: bool)
        -> list<array<real>[2]>:
    pc = p
    qc = q
    path = []
    for k in 0 .. steps:
        n = zeros(real, [2])
        for i in 0 .. 2:
            n[i] = model[i][2] + model[i][0] * pc[i] + model[i][1] * qc[i]
            if clip:
                n[i] = min(max(n[i], lo[i]), hi[i])
        append n to path
        qc = pc
        pc = n
    return path
```

<!-- worked: project_dynamic_scene.dynamics -->
The model of one coordinate of one ball is p' = c + a p + b q, with p the position at frame t and q the position at frame t - 1. It is fitted to the detections of the 12 observed frames (the 10 transitions of frames 1 to 10 into 2 to 11) by ridge regression on the features (p, q, 1): the normal equations with 0.0001 added to the diagonal entries of a and b, solved by Gauss-Jordan elimination with partial pivoting, each sum over the transitions in order. Fitted coefficients:

| ball | axis | a | b | c |
|---|---|---:|---:|---:|
| A (spring) | x | 1.8618 | -0.9831 | 0.0030 |
| A (spring) | z | 1.7499 | -0.8366 | 0.4636 |
| B (glider) | x | 0.6125 | 0.2629 | 0.0044 |
| B (glider) | z | 0.1541 | 0.6756 | 0.9172 |
| C (slider) | x | 0.5087 | 0.4939 | 0.2206 |
| C (slider) | z | 0.1579 | -0.2439 | 6.3962 |

The root mean square one-step residual over the 10 transitions, in metres: Ax 0.0025; Az 0.0225; Bx 0.0029; Bz 0.0100; Cx 0.0039; Cz 0.0154. A spring and a glider are linear: their coefficients can be read against the recurrences of milestone 1 (for the spring in x, a = 1 + d - dk = 1.8756 and b = -d = -0.995; for a glider a = 1.9 and b = -0.9), up to the noise of the detections.

Free run from the detections of frames 10 and 11, 6 frames, each prediction fed back (and, for the clamped model, clipped to the clamp box of milestone 2 after every step). Distance in metres of the predicted position from the true one at frames 12 .. 17:

| prediction | ball | frame 12 | frame 13 | frame 14 | frame 15 | frame 16 | frame 17 |
|---|---|---:|---:|---:|---:|---:|---:|
| learned, clamped | A | 0.067 | 0.039 | 0.099 | 0.195 | 0.301 | 0.407 |
|  | B | 0.108 | 0.106 | 0.109 | 0.108 | 0.109 | 0.109 |
|  | C | 0.111 | 0.107 | 0.110 | 0.111 | 0.111 | 0.110 |
| learned, unclamped | A | 0.067 | 0.039 | 0.099 | 0.195 | 0.301 | 0.407 |
|  | B | 0.108 | 0.106 | 0.109 | 0.108 | 0.109 | 0.109 |
|  | C | 0.111 | 0.176 | 0.310 | 0.454 | 0.600 | 0.749 |
| last position held | A | 0.287 | 0.575 | 0.845 | 1.068 | 1.224 | 1.305 |
|  | B | 0.134 | 0.165 | 0.195 | 0.224 | 0.249 | 0.273 |
|  | C | 0.203 | 0.203 | 0.203 | 0.203 | 0.203 | 0.203 |
<!-- /worked -->

![The table seen from above, x to the right and z away from the camera, with the table's rectangle and the dashed clamp box inside it. The detected positions of balls A, B and C over the twelve observed frames are dots; solid lines continue A and B along their true paths to frame 17; the slider C is detected moving right along the far side of the table, its true path stops at the right edge of the clamp box, and the dashed purple line of the unclamped model goes on past the table.](img/floor-plan.svg)

The figure draws the same thing from above: the detections of the twelve frames, the true paths to frame 17, and the slider that the clamped model stops at the edge of the box and the unclamped one carries out of the table.

Read the coefficients before the errors. Ball A in $x$ has $a = 1.8618$ and $b = -0.9831$, within 0.014 and 0.012 of the recurrence's; in $z$ the fit has 1.7499 and $-0.8366$, further off, because the depth of a detection is an order of magnitude noisier than its sideways position (the one-step residual of A in $z$ is 0.0225 metres, in $x$ 0.0025). That has a price, which is the one ball whose error grows in the table: A's error is 0.407 metres at frame 17. The glider and the slider have coefficients that bear no resemblance to their recurrences (B in $x$: 0.6125 and 0.2629, for a glider whose $a$ and $b$ are 1.9 and $-0.9$), and their errors are the flattest in the table, 0.11 metres at every frame, about the size of the reconstruction's own bias in milestone 2 (0.092 and 0.111 for B and C). A coefficient is not a measurement when the features are nearly dependent. What matters is the free run, and the free run of the slider is where the two rows of the table part: with the clamp, C stays at 0.110 metres from the truth; without it, C leaves the table, 0.176 at frame 13 and 0.749 at frame 17.

The model that holds each ball where it was last seen is the last block of the table: its error grows to 1.305 metres for the spring, which is what a model that knows nothing of motion costs.

**Milestone 5 ends** when your coefficients for ball A agree to three decimals and your clamped and unclamped errors for ball C at frame 17 are 0.110 and 0.749.

### Milestone 6: predicting six frames and scoring them

A prediction of a frame is a render. The Gaussians of milestone 4 are rendered at the camera of each held-out frame, with the three balls at the positions that a predictor gives them, and the result is compared with the true frame. Two decisions are worth saying out. The cameras of the held-out frames are the world's own, in the world's frame (the gauge makes the reconstruction's frame the world's to within its errors: centre errors of a centimetre), because the question the loop answers is what a viewer sitting where the camera will be would see, which is a render at a camera the viewer supplies, the way the action of Chapter 7's model is supplied. And the score is the mean squared error against the *clean* true frame, so that the number measures the prediction and not the noise of the photograph.

<!-- worked: project_dynamic_scene.held_out -->
Mean squared error over pixels and channels of each held-out frame (12 .. 17) against the clean true frame, rendered at the true camera of that frame (exported as in milestone 3), with the Gaussians of milestone 4 and the balls where each row puts them. Each number carries the digits that the run and its four nudged copies share. The last row is not a render: it scores the noisy observed frame 11 as if it were the prediction.

| prediction | frame 12 | frame 13 | frame 14 | frame 15 | frame 16 | frame 17 | mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| learned model, clamped | 2.71e-04 | 2.88e-04 | 3.26e-04 | 4.01e-04 | 5.27e-04 | 7.28e-04 | 4.23e-04 |
| learned model, unclamped | 2.71e-04 | 3.74e-04 | 6.36e-04 | 1.04e-03 | 1.63e-03 | 2.18e-03 | 1.02e-03 |
| balls held at their last detection | 4.31e-04 | 5.38e-04 | 8.18e-04 | 1.15e-03 | 1.43e-03 | 1.82e-03 | 1.03e-03 |
| balls at their true positions | 2.69e-04 | 2.82e-04 | 3.05e-04 | 3.44e-04 | 3.92e-04 | 4.81e-04 | 3.45e-04 |
| the last observed frame, copied | 3.15e-04 | 6.10e-04 | 1.01e-03 | 1.47e-03 | 1.90e-03 | 2.46e-03 | 1.29e-03 |
<!-- /worked -->

The learned model with its clamp is at $4.23 \times 10^{-4}$ over the six frames, between the reference of $3.45 \times 10^{-4}$ and everything else. The reference is the floor under every row: it is what the Gaussians cost with the balls where they truly are, since the scene they stand on was fitted to twelve noisy frames and not to the world. A model of dynamics is judged by how little it adds to that, and the clamped one adds 22 percent. The table shows the other things that were said about it. Copying the last observed frame has only the camera's motion to be wrong about on frame 12, $3.15 \times 10^{-4}$, and is 3.4 times worse than the model on frame 17 ($2.46 \times 10^{-3}$ against $7.28 \times 10^{-4}$), since the camera keeps moving. Holding the balls where they were is worse than copying at frame 12 and better from frame 13 on; both are worse than the clamped model on every frame. And the unclamped model starts exactly level with the clamped one, $2.71 \times 10^{-4}$ on frame 12, since its slider is only 0.004 metres past the box, and ends three times worse, $2.18 \times 10^{-3}$ against $7.28 \times 10^{-4}$: its slider goes 0.15 metres further out of the table with every frame, and the predicted ball and the true one separate in the image by about 0.6 pixels more with every frame. A test on frame 12 alone would not see the mistake.

**Milestone 6 ends** when your five rows agree with the table to the digits shown. The reference row is the one to look at first: if it is off, the fit or the cameras are, and the dynamics have not yet been tested.

### Milestone 7: three planted mistakes

A loop that works is not yet tested. Each of the mistakes below is the kind that a code review signs off, and each is planted alone in an otherwise correct loop; the acceptance of the project requires that every one of them be caught twice, once by a measurement that needs no fitted model and that a practitioner would take, and once by the loop's own score.

1. **A half pixel in the lens.** The renderer's lens at 32 by 24 pixels is made from the 640 by 480 one by the centre rule $c' = s(c + \tfrac12) - \tfrac12$ where the lens is in the corner convention, in which the first rule of Chapter 3, $c' = s\,c$, is exact: the principal point becomes $(15.525, 11.525)$ and not $(16, 12)$, a shift of $\tfrac12(1 - s) = 0.475$ pixels. The shift is invisible in a picture, 9.5 pixels at the lens the tracks are measured in and 0.0988 metres at the depth of the table, and it is the kind of constant that a review reads past because both numbers look like a principal point.
2. **A transposed rotation.** The cameras of the adjustment are $(\mathbf{R}, \mathbf{t})$, world to camera; the exporter hands the renderer $(\mathbf{R}^\top, \mathbf{t})$, so that the rotation of frame $i$ is turned the opposite way and the camera is placed at $-\mathbf{R}\mathbf{t}$ in place of $-\mathbf{R}^\top\mathbf{t}$ (the exporter handles the held-out cameras too, so they are wrong in the same way). Camera 0, with $\mathbf{R} = \mathbf{I}$, is untouched, and a wrongly turned camera of 18 degrees is wrong by 36.
3. **No clamp.** The rollout of milestone 5 with `clip` false.

<!-- worked: project_dynamic_scene.planted -->
Direct measurements, each a number that needs no fitted model:

| mistake | name | measurement | value |
|---|---|---|---:|
| 1 | half a pixel | shift of every projection in the 32 by 24 image (pixels) | 0.475 and 0.475 |
| 1 |  | the same shift at 640 by 480 (pixels); at depth 5.2 m (metres) | 9.5; 0.0988 |
| 1 |  | squared error between the two renders of the true scene, frames 11 and 17 | 5.146e-04; 5.630e-04 |
| 2 | transposed rotation | camera-centre error at frames 1, 4, 11 (metres): the cameras recovered in milestone 2, scaled, with R transposed, against the true centres | 0.007, 0.128, 1.018 |
| 2 |  | RMS reprojection error of the 40 points, 12 frames (pixels at 640 by 480); share in front | 196.2; 1.000 |
| 3 | no clamp | ball C's predicted x past the clamp box at frames 12 .. 17 (metres) | 0.004, 0.153, 0.302, 0.452, 0.602, 0.753 |

The pipeline run with each mistake planted alone. Runs 1 and 2 repeat milestones 3, 4 and 6 with the cameras exported the wrong way (in run 2 the rays of the ball tracks use the exported cameras too, so the detections move); run 3 changes only the rollout. Training loss after 120 updates; the held-out mean squared error of the prediction of that row and of the balls at their true positions (the reference, which no dynamics can influence); the largest distance of a predicted ball from the true one at frame 17:

| pipeline | training loss | held-out error | held-out error, true balls | worst ball error at frame 17 (m) |
|---|---:|---:|---:|---:|
| as specified | 4.89e-04 | 4.23e-04 | 3.45e-04 | 0.407 |
| 1: half a pixel | 4.70e-04 | 5.34e-04 | 4.88e-04 | 0.407 |
| 2: transposed rotation | 9.97e-04 | 2.42e-02 | 2.42e-02 | 1.750 |
| 3: no clamp | 4.89e-04 | 1.02e-03 | 3.45e-04 | 0.749 |
<!-- /worked -->

The direct measurements first, since they need no fit and are what a person who suspects a mistake would reach for. The half pixel is a shift of 0.475 pixels in both coordinates of every projection, 9.5 pixels at the full lens; two renders of the true scene through the two lenses differ by a squared error of $5.15 \times 10^{-4}$ at frame 11 and $5.63 \times 10^{-4}$ at frame 17, as large as the whole held-out error of the model, $4.23 \times 10^{-4}$. The transposed rotation puts the camera of frame 4 at 0.128 metres from where it is and that of frame 11 at 1.018 metres, and the recovered points reproject through the exported cameras with a root mean square error of 196.2 pixels, against 0.34 for the correct ones. And the missing clamp puts the slider 0.004 metres past the box at frame 12 and 0.753 at frame 17.

The loop's own score tells a different story for each. For the half pixel the training loss is not a witness: it is 4.70e-04, 4 percent *below* the correct run's 4.89e-04, and a gate that watches the loss of a fit would pass the mistake as a small improvement. (Commit `69a4363c6` records the same shape of failure at brain's scale: Gaussians assembled half a pixel off the pixels that made them scored higher than the correct scene at the renderer's default dilation, because a half pixel diagonal shift is a two by two blur and the score prefers blur.) The held-out error is the witness here, 5.34e-04 against 4.23e-04, and so is the reference row with the true balls, 4.88e-04 against 3.45e-04: 41 percent more, which cannot be the dynamics' fault. The ball positions, which come from the 640 by 480 lens and not from the mistaken one, are as good as before, 0.407 metres in the worst case. For the transposed rotation the loss is the half-witness that the opening saw, 9.97e-04, twice the correct, which a loose threshold on a fit's loss would accept, and the held-out error is 2.42e-02, 57 times the correct one, and the same for the true balls: the render is of another scene. The loss is no witness because the first frames are at small angles, where a transposed rotation differs little from the true one (the centre error of frame 1 is 0.007 metres), and the error grows with the angle (0.128 metres at frame 4, 1.018 at frame 11); the held-out frames are at 20 to 28 degrees, further than any frame the fit saw. And the missing clamp leaves the training loss exactly where it was, since the clamp is not in training, and shows only on the frames after the edge, 1.02e-03 against 4.23e-04.

**Milestone 7 ends** when each of the three planted runs produces the numbers of its rows, and the clean run does not.

### Milestone 8: what the loop costs

The last milestone counts the work, in operations (a multiply or an add is one) and in bytes of 32-bit values, from the shapes alone, using the counts of Chapter 6 for the renderer (260 operations a Gaussian for the projection, 21 for each (pixel, Gaussian) pair, whose backward the book does not count in operations and this table counts in pairs) and the Schur counts of Chapter 4 for the adjustment. The RANSAC rows count the scoring of the hypotheses only (45 operations for a Sampson distance, 40 for a reprojection, hand counts of the two routines), not the eight-point fit or the DLT inside, which are not counted.

<!-- worked: project_dynamic_scene.bill -->
Operations (a multiply or an add is one; the counts of `gaussian_splatting`: 260 per Gaussian for the projection, 21 per pair) and bytes (32-bit values) of every stage, counted from the loops. 'Bytes' is what one pass of the stage moves: the observations, the matrix that is factored, the images read and written, the gradients written. Counting rules. Adjustment: one damped step costs P (54 k + 108 k^2) + nc^3 / 6 multiply-adds, with P = 40 points, k = 12 views each and nc = 65 camera unknowns (6 per camera but two, plus 5), which is 693,770.8333 per step, and the table's count is 2 x that x 10 steps = 13,875,416.67, cut to an integer; its bytes are 4 (nc^2 + 9 P) = 18,340 (the reduced matrix and the 3 by 3 block of each point). Dynamics: each of the 6 ridge fits (3 balls x 2 coordinates) is 260 (a hand allowance: the sums are 10 x 12 x 2 = 240 and 20 cover the ridge and the elimination), and each of the 2 rollouts (clamped and unclamped) is 6 frames x 3 balls x 2 coordinates x 4 (p' = c + a p + b q is two multiplies and two adds) = 144, so 6 x 260 + 2 x 144 = 1,848. The tracks row is 12 x 43 x 2 values of 4 bytes.

| stage | count | operations | share | bytes |
|---|---|---:|---:|---:|
| tracks | 12 frames x 43 tracks x 2 values | - | - | 4,128 |
| pair: RANSAC scoring | 9 hypotheses x 43 correspondences x 45 | 17,415 | 0.01% | 688 |
| registration: RANSAC scoring | 343 hypotheses (10 frames) x 40 points x 40 | 548,800 | 0.17% | 800 |
| adjustment | 10 damped steps x 2 x 693,770.83 multiply-adds (Schur, 65 camera unknowns, 40 points, 12 views each) | 13,875,416 | 4.22% | 18,340 |
| fit: forward | 121 loss evaluations x 12 frames x (13 Gaussians x 260 + 9,984 pairs x 21) | 309,339,888 | 94.05% | 221,184 |
| fit: backward | 120 updates x 12 frames x 9,984 pairs replayed (not counted in operations) | - | - | 8,736 |
| dynamics | 6 ridge fits (10 transitions, 3 features) and 2 rollouts of 6 frames | 1,848 | 0.00% | - |
| held-out frames | 4 predictions x 6 frames x (13 x 260 + 9,984 x 21) | 5,113,056 | 1.55% | 221,184 |

The total is 328,896,423 operations, of which the forward passes of the fit are 94.1%. A render visits every record at every pixel, 9,984 pairs a frame for the starting scene (and 119,808 for the 12 frames of the fitted scenes: no Gaussian is culled, so the count does not move); an image of 32 by 24 pixels is four tiles of 16 by 16, so the tiled count of Chapter 6 is no smaller. One update reads the 12 target frames (110,592 bytes) and the parameters (8,736 bytes) for 2,556,528 forward operations: 21.4 operations per byte moved, and the backward pass replays the same 119,808 pairs.
<!-- /worked -->

The surprise of the table is how much of the work is the fit and how little the geometry costs: the forward passes of the 121 loss evaluations are 94 percent of the 3.3e8 operations of the loop, the adjustment 4.2 percent, and the two RANSAC scorings that decide whether the cameras are found at all together 0.2 percent. The backward pass replays the same 119,808 pairs, so that the fit's share is, if anything, higher than the table says. The work of the fit is set by the product of the pixels and the Gaussians (9,984 pairs a frame from 768 pixels and 13 Gaussians, none culled), so that doubling the width and the height of the images would multiply it by four and leave the geometry unchanged: cheap geometry, expensive pictures.

One update reads the 12 target frames (110,592 bytes) and the parameters (8,736 bytes) for 2,556,528 forward operations: 21.4 operations per byte of target and parameters, because each pixel is touched by 13 Gaussians. The dynamics, whose 1,848 operations are less than a thousandth of a percent of the loop, is the part of the loop the project is about, and the part that costs the least.

## What the scene pipeline must reproduce

Checks come from the values the worked module prints. Test in the order of the milestones: a disagreement at milestone 6 that follows agreement at 1 to 5 points at the prediction and not at the reconstruction. 'Exact' means equal integers; a tolerance is absolute unless the row says relative, and "the digits shown" means the digits on which the run and its four nudged copies agree.

<!-- worked: project_dynamic_scene.acceptance_tests -->
Expected values of the project, computed by this module. 'Exact' means equal; a number is compared with the tolerance in the last column, absolute unless it says relative; 'the digits shown' means the digits on which the run and its four nudged copies agree, and no more can be asked of another implementation.

| milestone | quantity | value | tolerance |
|---:|---|---|---|
| 1 | sum of the 2,304 values of clean frame 0; of observed frame 0 | 1389.157309; 1389.728264 | 1e-6 |
| 1 | pixel of landmark 0 in view 0 and 11; of ball A in view 0 | (305.6813, 197.2169); (364.2223, 203.9184); (243.9071, 370.1808) | 1e-4 |
| 2 | pair: iterations; inliers of 43; tracks outside the model | 9; 40; [40, 41, 42] | exact |
| 2 | adjustment: observations; points; final RMS reprojection (pixels) | 480; 40; 0.336 | exact; exact; 1e-2 |
| 2 | largest rotation error (degrees); largest centre error (metres) | 0.258; 0.0119 | 1e-2; 1e-3 |
| 2 | depth ratio; floor height; clamp box x range | 0.9842; 1.4785; [-0.8845, 0.8877] | 1e-3 |
| 3 | loss of the starting scene; pairs per frame | 0.009656; 9984 | 1e-6; exact |
| 4 | training loss after 10, 60 and 120 updates | 2.683e-03, 5.984e-04, 4.888e-04 | the digits shown |
| 5 | ball A, x: a, b; ball A, z: a, b | 1.8618, -0.9831; 1.7499, -0.8366 | 1e-3 |
| 5 | clamped model, error of A and C at frame 17 (metres) | 0.407, 0.110 | 1e-3 |
| 5 | unclamped model, error of C at frames 13 and 17 (metres) | 0.176, 0.749 | 1e-3 |
| 6 | held-out mean squared error: clamped, unclamped, held, true balls, copied | 4.23e-04, 1.02e-03, 1.03e-03, 3.45e-04, 1.29e-03 | the digits shown |
| 7 | half pixel: shift (pixels); squared error of the two renders, frames 11 and 17 | 0.475; 5.15e-04, 5.63e-04 | 1e-3; 1e-2 relative |
| 7 | half pixel: held-out error; training loss | 5.34e-04; 4.70e-04 | the digits shown |
| 7 | transposed (milestone-2 cameras, not the true ones): centre error at frame 11 (metres); RMS reprojection (pixels) | 1.018; 196.2 | 1e-3; 0.1 |
| 7 | transposed: held-out error; training loss | 2.42e-02; 9.97e-04 | the digits shown |
| 8 | operations of the whole loop; forward passes of the fit; pairs per frame | 328,896,423; 309,339,888; 9,984 | exact |
<!-- /worked -->

Beyond the table an implementation passes when it also meets these, each of which is in a block above: the three rejected tracks; the clamp box and the table of milestone 2; the 13 starting Gaussians; the five predictors on each of the six held-out frames; the ball errors of every frame for the clamped, the unclamped and the held model; and the three planted mistakes each producing the number of its row. A loop that passes the clean rows and cannot be made to fail the planted ones has tests that cannot fail.

## Where the loop's pieces live in brain

The loop crosses nine crates, and the correspondence is closer than the difference in size suggests, because the interfaces are the same ones, and one of the project's mistakes is a mistake that brain's own history records.

**The frames.** `crates/videocodec` is where real frames come from: `decode` in `crates/videocodec/src/lib.rs` calls back with a `Frame` that carries its index, its time in seconds and its RGB bytes, and the clip of this project is the sequence that a loop over those frames would hand to the next stage. The project's frames are floating-point renders and need no codec; what carries over is that a frame has a time as well as pixels, which is what makes a clip a sequence to predict and not a bag of photographs (Chapter 1).

**The lens.** `crates/camera` is the one lens model of the workspace, and its convention is the project's: pixel $(i, j)$ covers $[i, i + 1) \times [j, j + 1)$, and `Intrinsics::pinhole(f, width, height)` puts the principal point at the image centre, so the 640 by 480 camera with $f = 500$ has $(320, 240)$ and `Intrinsics::resized(32, 24)` gives $(16, 12)$ and $f = 25$ by multiplying every number by the factor, exactly the corner rule of milestone 4. The test that holds the rule is `resizing_scales_every_pixel_coordinate_exactly` in `crates/camera/tests/lens.rs`; the first planted mistake is what a resize written by hand would be if it used the other rule.

**Cameras and points from the frames.** `crates/sfm` is milestone 2: `twoview::ransac_essential` and `pnp::ransac_pnp` are the two RANSACs, `incremental::reconstruct` is the loop that picks a seed pair, registers the view that sees most of the triangulated points and adjusts, and `ba::bundle_adjust` takes the gauge exactly as the project does, an `anchor` camera whose pose is held and a `scale` camera whose distance from it is held. Its `georef.rs` is what the project replaces by assumptions: `gravity_from_cameras` finds the vertical from the cameras' own horizontal axes where the project took camera 0 to be level, and the scale comes from a satellite fix where the project was told a baseline. `crates/mvs` adds what the project does not have, a dense surface from the cameras and the tracks (`crates/mvs/src/init.rs` turns it into thin Gaussians, and its comment says why: a sparse start gives isotropic blobs with no geometry and answers held-out views with floaters), and `crates/recon` ties them together: `photogrammetry::training_set` takes photographs to cameras, a sparse cloud and the targets of a fit, which is milestones 2 and 3 as one call, and its `plan_chunks` is what a capture does when it is longer than one reconstruction can hold.

**The Gaussians and the fit.** `crates/splat` is milestones 3 and 4. `splat::init::from_points` starts one isotropic Gaussian at every point of the cloud, sized by the distance to the three nearest neighbours, where the project cut a handful of chunks; `crates/splat/src/opt.rs` is the fit, whose raw parameters are the log scales and opacity logits of the project and whose cameras may be refined with, in the words of its module comment, the gauge fixed (the first camera and the scale do not move). Commit `2e838f2a6` (25 September 2026) is the change that moved the fit there, onto the device, through each photograph's own lens and with that gauge, and that added `recon::eval`; the Adam step is a kernel (`splat_adam.wgsl`) and the camera is `splat::types::Camera`, which stores `c2w` and hands the rasterizer `viewmat`, the world-to-camera rows: the place where the project's second mistake lives, since an exporter that hands over a rotation in the other of the two conventions is wrong in exactly the way the table of milestone 7 measures.

**A reconstruction without the pipeline.** `crates/worldmirror2` replaces milestone 2 by one forward pass of a network that outputs the cameras and a point per pixel, and the half pixel is there too: commit `69a4363c6` (21 September 2026) found that the assembly of its Gaussians unprojected pixel $n$ as $n$, half a pixel up and left of where the rasterizer samples it, and records why a check that ran at the default dilation would have called the bug an improvement. `crates/worldmirror2/src/recon_impl.rs` is the seam through which `crates/recon` drives it.

**Judging on frames that were not shown.** The held-out frames of milestone 6 have a counterpart in `crates/recon/src/eval.rs`, whose module comment states the lesson of this chapter's second and third mistakes in its first paragraph: the loss on the training views says how well a fit memorized them, and nothing more, and the comment records a capture on which a fit reached 30.96 dB on its training views and 13.83 dB on the four it was not shown. `holdout` withholds every $k$-th registered photograph, spread over the capture; the project withholds the *future*, the six frames after the twelve, which is a harder test of the dynamics and a different one from a novel view, and `score_held_out` scores a render of each withheld view against its photograph.

**Predicting the next frame.** `crates/wm-core` has the interface of a playable world model: `reset` with context frames, then `step` with an action, returning the next frame, and `FakeWorldModel` is the deterministic model of Chapter 7's first section. `crates/diamond` is the one learned model behind it that plays: a ring of the last frames and actions and an Euler denoising loop, trained one transition at a time. That is a predictor of pixels from pixels, with no three-dimensional state, no floor and no clamp, and the project's loop is the other way of predicting: recover the state of the world, run a model of the state, render. The two meet at a frame: the manifests of `crates/diamond` and `crates/wm-core` name none of the reconstruction crates, and theirs name neither of them, so that the halves meet only in a caller that uses both. The loop of this chapter is the shape of the join, and a table such as the first of the chapter is how its pieces would be compared with a pixel-space model on the same held-out frames.

**Where the project's choices differ from brain's.** The project's world is closed: the frames are renders of 17 Gaussians and the fit has 13, so the model error is small and the answers are known to the metre, where brain reconstructs photographs of scenes it has no truth for. Its tracks are given, with the identity of every point known, where brain matches descriptors. Its pictures are 32 by 24 and its ten static Gaussians stand for a cloud of hundreds of thousands. Its ball positions are not fitted photometrically, which brain's fit would do for every Gaussian. And its dynamics is one linear family for three balls, where the dynamics of a real scene would have to be found as well as fitted.

## Check your understanding

1. A colleague looks at the half-pixel run and says: "its training loss is 4.70e-04 against 4.89e-04 for the correct one, so the centre rule fits better and we should keep it". Find the error in the argument, and say which numbers of milestone 7 show it.
2. The transposed run trains to a loss only twice that of the correct run, and its held-out error is 57 times larger. The same exporter was used for the observed and the held-out frames. Why can a fit hide the mistake in one set of frames and not the other?
3. On frame 12 the clamped and unclamped models have the same error, $2.71 \times 10^{-4}$, and on frame 17 they differ by a factor of three. A reviewer proposes to gate the dynamics on its one-step error. What does that test see, and what is the number of the table that it cannot?
4. The glider's fitted coefficients are $a = 0.6125$ and $b = 0.2629$ in $x$, where a glider that slows by a tenth has 1.9 and $-0.9$. Is the fit wrong? What would settle it?
5. Every pair of a render is visited at every pixel because the image is four tiles of 16 by 16. Predict how the 3.3e8 operations of the loop change if the images are 64 by 48 pixels, and which stage's share rises.

## Summary

If every acceptance test passes you have a complete loop on a world whose answers are known: a camera path and a sparse structure recovered from twelve frames to a quarter of a degree and a centimetre, in which the moving things stand out as the tracks the rigid model rejects; ten Gaussians for a wall and a floor and three for the balls, fitted to the frames by gradient descent through the recovered cameras; a linear model of each ball that is run six frames beyond its data, and clipped to the table that the reconstruction found; a render of that prediction at the cameras of six frames nobody saw, scored at $4.23 \times 10^{-4}$ against $3.45 \times 10^{-4}$ for the reference with the balls in place; three planted mistakes, each measured twice; and a bill in which the renderer is 94 percent of the work.

What the project established, in order. The tracks that a rigid model rejects are the moving objects: here all three, with no help from the gate that follows the adjustment. A reconstruction is as good as its gauge and its scale: the cameras were right to a centimetre and the depth 1.6 percent short, and every ball position, every floor height and every clamp edge carried the second and not the first. A fit of a dynamics model is a regression, and a regression on nearly dependent features returns coefficients that mean nothing and predictions that mean something; the held-out rollout is the test, not a coefficient. A model that has seen no wall will carry a ball through it, and the wall is a fact about the world that the reconstruction can supply (the extent of the floor) and the dynamics cannot learn from a ball that has not yet arrived. And a training loss is a witness to the fit and no one else: three mistakes, of which one improved it, one doubled it and one left it alone, were caught by the frames that were held out and by numbers that no model needed.

Where this stops. The world has a wall, a floor and three balls, and the answers are known; a real scene has neither. The tracks are given with their identity, the cameras' held-out poses are the world's own, the balls' heights are fixed by a plane, and the dynamics is a three-parameter family that the project knew to be the right one. The objects are spheres that cannot occlude one another in any way that matters, and nothing here predicts what a camera will do, what a ball does when it meets another, or what is behind the wall. The extensions are the reasons the limits exist.

## References

- B. Kerbl, G. Kopanas, T. Leimkuehler and G. Drettakis, *3D Gaussian splatting for real-time radiance field rendering*, 2023. The Gaussian, its renderer and its fit.
- J. L. Schönberger and J.-M. Frahm, *Structure-from-motion revisited*, 2016. The incremental pipeline of milestone 2.
- B. Triggs, P. McLauchlan, R. Hartley and A. Fitzgibbon, *Bundle adjustment: a modern synthesis*, 2000. The adjustment, its gauge and its Schur complement.
- M. A. Fischler and R. C. Bolles, *Random sample consensus*, 1981. The voting of both RANSACs.
- D. P. Kingma and J. Ba, *Adam: a method for stochastic optimization*, 2014. The optimizer of the fit.
- E. Alonso, A. Jelley, V. Micheli, A. Kanervisto, A. Storkey, T. Pearce and F. Fleuret, *Diffusion for world modeling: visual details matter in Atari*, 2024. The pixel-space world model that the last section sets beside this loop.

## Beyond a wall, a floor and three balls

These are optional. Each changes one thing of the project and asks you to predict the outcome before you run it.

1. **A floor that is not known to be level.** Tilt camera 0 by two degrees about its $x$ axis and keep everything else. Predict the error that the horizontal-floor assumption adds to the ball detections, and replace it by a plane fitted to the floor's points by three-point RANSAC.
2. **The positions fitted with the picture.** Make the ball positions parameters of the fit again, with rates of 0.002 and 0.02 metres. Predict the direction in which the slider's reconstructed position drifts and whether the held-out error moves.
3. **A wall for the balls to bounce off.** Give the slider a reflection at the table's edge, $v' = -v$, and keep the clamp. Predict the error of a clamped linear model, and what a model with the family $(a, b, c)$ of milestone 5 and a different clip for each axis could not learn.
4. **A second camera in the loop.** Add a second camera that films the table from the side, with the same tracks and the same noise. Predict the change of the detection error and the one of the depth bias.
5. **The pixel-space comparison.** Fit a small convolutional predictor of the next frame to the 12 observed frames and score it on the held-out ones in the table of milestone 6. Predict which of the five rows it lands between.
6. **A longer horizon.** Predict 12 held-out frames, not 6, and find the frame at which the clamped model's error equals that of holding the balls where they were. Say which ball it is, and why a spring's error keeps growing where a glider's does not.

## Answers

1. The centre is $\mathbf{C} = -\mathbf{R}^\top\mathbf{t}$. Using $\mathbf{R}^\top$ where $\mathbf{R}$ belongs turns the camera the opposite way (a rotation by the angle $\theta$ becomes one by $-\theta$, so the error is $2\theta$, 36 degrees for a camera turned by 18), places the centre at $-\mathbf{R}\mathbf{t}$ and projects the world origin correctly, since $\mathbf{R}\mathbf{0} = \mathbf{0}$ whatever the rotation is.
2. $(320, 240)/20 = (16, 12)$ by $c' = s\,c$. The rule $s(c + \tfrac12) - \tfrac12$ gives $320.5/20 - 0.5 = 15.525$ and $240.5/20 - 0.5 = 11.525$: wrong by $-\tfrac12(1 - s) = -0.475$ pixels in each coordinate.
3. One camera is held (it has no unknowns) and one is kept at a fixed distance (it has five, not six): $6(N_{\mathrm{c}} - 2) + 5$ camera unknowns, 65 for 12 cameras, and with $3P$ point coordinates 185 unknowns for $P = 40$.
4. Fourteen numbers: the mean (3), three log scales, a raw quaternion (4), an opacity logit and a colour (3). The scales are stored as logs and the opacity as a logit, so that any real number gives a positive scale and an opacity in $(0, 1)$. The gradient is exact where the render is smooth, but the cut-off at $\alpha < 1/255$, the cap at 0.99 and the stop at a transmittance of $10^{-4}$ make the render jump or flatten, so that a finite difference across them differs from the analytic gradient.
5. The error does not stay small: it accumulates, $e_{k+1} = \hat{\mathbf{A}}e_k + \boldsymbol\delta$, to the size of the world the model is confined to. The model does not know the track has a wall: it is linear, and the wall is not.
6. The normal equations are $\mathbf{X}^\top\mathbf{X}\,\mathbf{w} = \mathbf{X}^\top\mathbf{y}$. If two features are linear functions of each other, $\mathbf{X}^\top\mathbf{X}$ is singular and has many solutions; adding $\lambda$ to a diagonal entry makes the matrix invertible and, of all the solutions, picks the one with the smallest weights.

Check your understanding.

1. The training loss is the loss on the frames the fit saw and says nothing about the lens, as the fit moves the Gaussians to absorb a shift in the observed views. The held-out error is 5.34e-04 against 4.23e-04, 26 percent worse, and the reference with the true balls is 4.88e-04 against 3.45e-04, 41 percent worse; and the direct measurements (0.475 pixels of shift, 9.5 at the lens of the tracks, 0.0988 metres at the table, a squared error of $5 \times 10^{-4}$ between two renders of the same scene) show it with no fit at all. A lower training loss is a reason to look at what the fit absorbed, and it is not a reason to keep a convention.
2. A transposed rotation is wrong by twice the angle of the camera, and the angle grows along the path: the cameras of the first frames are near the identity and the exporter's error there is small (a centre error of 0.007 metres at frame 1), so the fit, which has the freedom of ten Gaussians, bends them to fit what it can see. The held-out frames are at 20 to 28 degrees, at 36 to 56 degrees of error, beyond any view the fit had, and nothing hides the error there.
3. A one-step error on frame 12 sees the slider 0.004 metres past the box, which is nothing, and the two models are the same to three digits. The mistake is a mistake of the free run, in which the error grows with every frame spent outside the table (0.153 metres at frame 13, 0.753 at frame 17), and a gate that looks at the next frame cannot see a failure that lives six frames out. The number it cannot see is the ratio of $2.18 \times 10^{-3}$ to $7.28 \times 10^{-4}$ at frame 17.
4. The fit is not wrong, but the coefficients are not evidence. The glider has almost stopped over the window, so its three features are nearly dependent and a wide set of triples fits the twelve points, the ridge picking the one of least weight. The test is the free run: its error is 0.11 metres at every frame, the reconstruction's own bias, and a longer window or a ball that has not yet stopped would settle what the coefficients are.
5. Every pixel still visits every Gaussian, so the pairs are $64 \times 48 \times 13 = 39{,}936$ a frame, four times as many, and the work of every render goes up by about four (the projection of 13 Gaussians is fixed): the loop is about $1.26 \times 10^{9}$ operations, 3.8 times as many, and the forward passes of the fit rise from 94 to about 97 percent of it. The geometry, which does not see the pixels, is unchanged, and the images are now 12 tiles, so that a tiled renderer would begin to skip pairs.
