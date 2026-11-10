<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Faces: Detection, Identity and Parsing

*How does a model find faces and recognize the same person across photos?*

Take two photographs of one person and one of another, and ask a program that knows nothing but numbers to say which two belong together. The first idea is the cheapest: find the eyes, the nose and the corners of the mouth in each picture, five points, and compare the five points. Here they are, for three photographs. A1 and A2 are person A, photographed twice, once from farther away and tilted a little one way, and once closer and tilted the other way. B is person B, photographed from the same distance and at the same angle as A1.

<!-- worked: faces.photos -->
Landmarks of three photographs, in pixels of each photograph (x to the right, y down), rounded to 0.1. A1 and A2 are two photographs of person A, B is person B photographed with the pose of A1. The generator is `faces.landmarks` (seed 0x5caf1237ad14cc80 (6678576802750057600)): each landmark is the canonical shape of the person moved by the pose of the photograph (scale 2.0, 3.0, 2.0; rotation 10, -12, 10 degrees; shift (60, 90), (150, 130), (60, 90)) plus a jitter of 0.4 times a standard normal in x and then in y, drawn in the order photo, point, coordinate.

| landmark | A1 | A2 | B |
|---|---:|---:|---:|
| left eye | (113.8, 206.5) | (289.3, 262.8) | (121.2, 203.5) |
| right eye | (190.1, 216.1) | (402.3, 231.9) | (182.8, 217.3) |
| nose | (146.4, 254.1) | (362.2, 308.8) | (145.1, 248.1) |
| left mouth | (109.0, 282.6) | (326.2, 371.4) | (112.6, 289.0) |
| right mouth | (170.7, 294.7) | (419.0, 352.3) | (163.0, 298.4) |

Root-mean-square distance between corresponding landmarks of two photographs (the square root of the mean over the five points of the squared distance), pixels:

| pair | distance |
|---|---:|
| A1 and A2 | 223.09 |
| A1 and B | 7.52 |
| A2 and B | 224.43 |
<!-- /worked -->

The answer the numbers give is wrong in the most instructive way. Person B's five points sit about 7.5 pixels from A1's, and A2's sit 223 pixels from them. By this measure the nearest neighbour of A1 is a stranger, and the second photograph of the same face is about thirty times farther away. Nothing is broken: a pixel coordinate is a measurement of the face together with a camera, and the camera contributed more than the face did. A different distance, a different tilt, a different framing, and every number moves. Whatever compares two faces has to compare them after the camera has been divided out.

Dividing it out is a geometric problem with a clean solution, and it is the first half of the chapter. The second half is what to do once the faces are in a common frame: how a network turns an aligned crop into a vector that depends on who the person is and on little else, why the loss that trains it has to be built from angles, and how a third network labels every pixel of the crop as skin, hair, eye or cloth. Three models, then, in one pipeline, and brain has a crate for each: `crates/scrfd` finds the faces and their five points, `crates/arcface` turns an aligned crop into an identity vector (and holds the training graph that makes the vectors good), and `crates/bisenet` parses the crop. The figure shows what passes between them.

![The photograph goes to a detector. Its five landmarks are cut into two crops, one aligned to a 112 pixel template and read by an embedding network that writes 512 numbers compared by cosine, and one aligned to a 512 pixel template and read by a parsing network that writes a class for every pixel.](img/faces-pipeline.svg)

Four things from earlier chapters are needed first. The answers are at the end.

1. ([Volume I, Chapter 3](../v01-mathematics/03-vectors.md), cosine similarity.) What is the cosine similarity of $(3, 4)$ and $(4, 3)$, and what does the value tell you about the angle between them?
2. ([Volume III, Chapter 3](../v03-neural-networks/03-softmax.md), softmax cross-entropy.) The logits are $(2, 0)$ and the correct class is the first. What is the gradient of the loss with respect to each logit?
3. ([Volume VII, Chapter 4](04-detection.md), anchor-free detection.) A point at pixel $(44, 28)$ of the stride-8 level reports the distances $(l, t, r, b) = (2.0, 1.5, 3.0, 2.5)$ in cells. Where is the box?
4. ([Volume VII, Chapter 5](05-dense-prediction.md), semantic segmentation.) A segmenter with 19 classes writes its scores at one eighth of a 512 by 512 input. How many scores is that, how many after the resize to the full size, and how does a pixel get its label?

## Five points and a template

Divide out the camera, and what is left should be the face. The camera's freedom, for a face that stays roughly flat to it, is four numbers: how far (a scale), how it is turned in the picture plane (an angle), and where the face lies in the frame (two shifts). A transformation of the plane made of exactly these is a **similarity**: it multiplies every length by one factor $\rho$, rotates by one angle $\varphi$ and shifts by $(t_x, t_y)$. Written with $a = \rho\cos\varphi$ and $b = \rho\sin\varphi$, it takes a point $(x, y)$ to

$$
\begin{pmatrix} a & -b \\ b & a \end{pmatrix}\begin{pmatrix} x \\ y \end{pmatrix} + \begin{pmatrix} t_x \\ t_y \end{pmatrix}. \tag{1}
$$

The idea is to pick one fixed arrangement of the five points, a **template** $\mathbf{d}_0, \dots, \mathbf{d}_4$ (a face as it should look in the crop the recognition network will see), and to move every photograph by the similarity that brings its landmarks $\mathbf{l}_0, \dots, \mathbf{l}_4$ as close to the template as a similarity can. Five points give ten equations for four unknowns, so no similarity fits exactly, and "as close as possible" means the least sum of squared distances,

$$
E(a, b, t_x, t_y) = \sum_{i=0}^{4} \bigl\lVert \mathbf{M}\mathbf{l}_i + \mathbf{t} - \mathbf{d}_i \bigr\rVert^2, \qquad \mathbf{M} = \begin{pmatrix} a & -b \\ b & a \end{pmatrix}.
$$

The shift comes out first. $E$ is a sum of squares that is quadratic in $\mathbf{t}$, its derivative with respect to $\mathbf{t}$ is $2\sum_i(\mathbf{M}\mathbf{l}_i + \mathbf{t} - \mathbf{d}_i)$, and setting it to zero says $\mathbf{t} = \bar{\mathbf{d}} - \mathbf{M}\bar{\mathbf{l}}$, with the bars the means of the five points: the centroid of the photograph lands on the centroid of the template. Put that back and write $(x'_i, y'_i)$ for the offset of landmark $i$ from its mean and $(u'_i, v'_i)$ for the offset of template point $i$ from its. What remains is

$$
E = \sum_i \bigl(a x'_i - b y'_i - u'_i\bigr)^2 + \bigl(b x'_i + a y'_i - v'_i\bigr)^2 .
$$

Its derivative with respect to $a$ is $2(aS - D)$ and with respect to $b$ is $2(bS - C)$, where

$$
S = \sum_i \bigl(x_i'^2 + y_i'^2\bigr), \quad D = \sum_i \bigl(x'_i u'_i + y'_i v'_i\bigr), \quad C = \sum_i \bigl(x'_i v'_i - y'_i u'_i\bigr). \tag{2}
$$

So $a = D/S$ and $b = C/S$, and the whole fit is five points' worth of sums: $S$ is how spread out the photograph's landmarks are, $D$ how much their offsets point the same way as the template's, and $C$ how much one is turned against the other. The scale is $\rho = \sqrt{a^2 + b^2}$ and the angle $\varphi = \operatorname{atan2}(b, a)$. In the pseudocode of this book:

```text
function fit_similarity(src: array<real>[n, 2], dst: array<real>[n, 2])
        -> (a: real, b: real, tx: real, ty: real):
    require n >= 2
    ms = mean(src, 0)
    md = mean(dst, 0)
    s2 = 0.0
    c00 = 0.0
    c01 = 0.0
    c10 = 0.0
    c11 = 0.0
    for i in 0 .. n:
        x = src[i, 0] - ms[0]
        y = src[i, 1] - ms[1]
        u = dst[i, 0] - md[0]
        v = dst[i, 1] - md[1]
        s2 = s2 + x * x + y * y
        c00 = c00 + u * x
        c01 = c01 + u * y
        c10 = c10 + v * x
        c11 = c11 + v * y
    require s2 > 0.0
    require c00 * c11 - c01 * c10 >= 0.0
    a = (c00 + c11) / s2
    b = (c10 - c01) / s2
    tx = md[0] - (a * ms[0] - b * ms[1])
    ty = md[1] - (b * ms[0] + a * ms[1])
    return (a, b, tx, ty)
```

The routine reads the sums of (2) as four products, `c00` to `c11`, because the one `require` that is not obvious is made from them, and it is explained below. The template that the recognition network of this chapter expects is a constant of that network, learned along with its weights, and the five points are in a fixed order, left eye, right eye, nose tip, left and right corner of the mouth, where "left" means the left of the picture. For its 112 by 112 crop they are $(38.2946, 51.6963)$, $(73.5318, 51.5014)$, $(56.0252, 71.7366)$, $(41.5493, 92.3655)$ and $(70.7299, 92.2041)$, in pixel-centre coordinates (the first pixel is centred on 0, as in the landmarks). Here is the fit for photograph A1, every sum spelled out, and then the three photographs side by side.

<!-- worked: faces.similarity_fit_table -->
The fit of photograph A1 onto the 112 template (`TEMPLATE_112`, printed in the text), with the sums over the five points taken left to right:

| quantity | value |
|---|---:|
| mean of the photograph's landmarks | (146.000, 250.800) |
| mean of the template | (56.026, 71.901) |
| sum of squared offsets from the mean | 11076.820 |
| sum of dot products (template offset, photograph offset) | 5386.250 |
| sum of cross products (photograph offset x template offset) | -867.986 |
| a = dot / squared | 0.486263 |
| b = cross / squared | -0.078361 |
| tx = template mean x - (a mean x - b mean y) | -34.6211 |
| ty = template mean y - (b mean x + a mean y) | -38.6134 |

The fit of the three photographs. Scale is the square root of a squared plus b squared and angle is atan2(b, a) in degrees; the last column is the root-mean-square distance between the five transformed landmarks and the template.

| photo | a | b | tx | ty | scale | angle | residual |
|---|---:|---:|---:|---:|---:|---:|---:|
| A1 | 0.4863 | -0.0784 | -34.62 | -38.61 | 0.4925 | -9.15 | 1.726 |
| A2 | 0.3180 | 0.0741 | -35.74 | -51.90 | 0.3265 | 13.13 | 1.751 |
| B | 0.4976 | -0.0944 | -39.81 | -39.46 | 0.5065 | -10.74 | 1.933 |
<!-- /worked -->

Photograph A1 was taken at a scale of about 2 and a turn of 10 degrees, and the fit finds $1/\rho$ near 2 and an angle near $-9$ degrees: it has undone the camera, to within what five jittered points can say. The residual column is the root-mean-square distance between the moved landmarks and the template, between 1.7 and 1.9 pixels for all three. It is not an error to be driven to zero. Five points and four degrees of freedom leave an irreducible remainder, here made of the jitter of the detector and of the real differences between each person's face and the template, and a test that demanded an exact match of the landmarks would be asserting something false. (The comment in `crates/arcface/src/align.rs` makes the same point about the residual of the reference's golden cases, about 1.4 pixels.) What the fit is for is the next table.

<!-- worked: faces.aligned_distances -->
Root-mean-square distance in template pixels, after each photograph is taken onto the template by its own fit:

| pair | distance |
|---|---:|
| A1 and A2 | 0.242 |
| A1 and B | 3.594 |
| A2 and B | 3.635 |
| A1 and the template | 1.726 |
| A2 and the template | 1.751 |
| B and the template | 1.933 |
<!-- /worked -->

![Two panels with the landmarks of the three photographs. In the pixels of the photographs the landmarks of person A in photo 2 are far from those of photo 1, and those of person B are close to photo 1. After the fit, in the template frame, the two photographs of A coincide at each of the five points and person B is displaced from them.](img/landmarks-aligned.svg)

Measured in the frame of the template, the two photographs of A are 0.24 pixels apart and A against B is 3.6, where the raw coordinates said 223 and 7.5. The ordering has reversed, which is the point, and the comparison has a unit again: both distances are in pixels of the same 112 by 112 crop. The recognition network of this chapter does not compare five points, it compares the appearance of the whole crop, but the logic is the same. A crop is a function of the face and of the camera; the fit takes the camera out, and what the network is asked to tell apart is what is left.

### Inputs the fit must not accept quietly

A fit that always returns an answer is dangerous, because the next stage will use the answer. The solver in brain, `similarity_transform_2d` in `crates/model/src/hostmath.rs`, refuses two kinds of input, and the table shows a third that it cannot refuse.

<!-- worked: faces.fit_refusals -->
What the fit does with inputs that are not a face in the expected order (photograph A1, 112 template). Scale is the square root of a squared plus b squared, angle is atan2(b, a) in degrees, and the residual is in template pixels.

| input | outcome | scale | angle | residual |
|---|---:|---:|---:|---:|
| A1 as detected | fitted | 0.4925 | -9.2 | 1.73 |
| the two mouth corners swapped | fitted | 0.3271 | -8.0 | 17.42 |
| the points in reverse order | fitted | 0.4888 | 170.7 | 3.33 |
| eyes and mouth corners swapped | refused: the best fit needs a reflection | - | - | - |
| picture mirrored, order kept | refused: the best fit needs a reflection | - | - | - |
| five copies of one point | refused: every point is the same | - | - | - |
<!-- /worked -->

If the order of the five points is wrong the fit can still return a rotation and a scale. The best similarity for the points in reverse order, which is the face turned upside down, has an angle of 170.7 degrees and a residual of 3.33 pixels, only about twice that of the true order (and the residual of person B, a different face in the correct order, is already 1.93). The crop would be an upside-down face, the network would embed it, and the vector would be a plausible vector of the wrong thing. The code's comment on `estimate_norm` says it exactly: a permuted landmark set "still solves, and produces a plausible, wrong crop". Swapping only the two mouth corners costs a residual of 17.42 pixels and a scale of 0.327, and would be caught by any residual threshold, though the fit does not apply one. What the solver does refuse is the case where the best orthogonal part of the map would be a reflection: when the determinant of the cross-covariance (the `require` in the pseudocode, `c00 * c11 - c01 * c10`) is negative, a mirror image fits better than any turn, and a proper similarity of the form (1) cannot be one. Swapping every left point for its right partner mirrors the face and is refused, as is a mirrored picture. Returning the nearest non-mirrored fit instead would have produced a plausible, wrong warp, which is why the code returns an error.

## Looking the pixels up: the warp

The fit gives the camera's undoing for points. To make the crop it must be applied to the pixels, and there is a choice of direction. Pushing each source pixel through (1) onto the crop leaves holes (a scale below 1 sends several source pixels to one destination and a scale above 1 leaves destinations that no source reaches). The other direction has no holes: for every pixel $(x, y)$ of the crop, ask where it came from, $\mathbf{p} = \mathbf{M}^{-1}\bigl((x, y)^\top - \mathbf{t}\bigr)$, read the photograph there, and write the value at $(x, y)$. A similarity has an inverse in closed form: with $q = a^2 + b^2$ the matrix $\mathbf{M}^{-1}$ is $\frac{1}{q}\begin{pmatrix} a & b \\ -b & a \end{pmatrix}$. The source position $\mathbf{p}$ is rarely a whole pixel, so the read is the bilinear interpolation of [Chapter 1](01-images.md) among the four pixels around it.

Two conventions are hiding in "read the photograph there". The first is where a pixel is. The landmarks and the template of this chapter put the centre of the first pixel at coordinate 0. A sampler that is handed a grid of positions in $[-1, 1]$ must say what $-1$ and $+1$ mean. If they are the outer edges of the first and last pixels, the position $x_s$ in pixel-centre coordinates is the grid value $g_x = (2x_s + 1)/W - 1$ for a picture $W$ wide; if they are the centres of the first and last pixels, it is $2x_s/(W - 1) - 1$. They agree at the middle of the picture and part by half a pixel at its edges, and the table puts numbers on the difference.

<!-- worked: faces.warp_conventions -->
The destination pixels of the 112 by 112 crop of photograph A1, which is 320 by 400 pixels, mapped back to the photograph by the inverse of the fit (a, b, tx, ty) of the previous table. gx = (2 xs + 1) / width - 1 and gy = (2 ys + 1) / height - 1 is the grid value of a sampler whose coordinate -1 is the left edge of the first pixel and +1 the right edge of the last. The last column is how far (in photograph pixels, x and y) a sampler that reads the same value with the corner convention (-1 at the centre of the first pixel, +1 at the centre of the last) would land from the right place.

| destination pixel | source (xs, ys) | grid (gx, gy) | error of the corner reading |
|---|---:|---:|---:|
| (0, 0) | (56.923, 88.582) | (-0.64110, -0.55459) | (0.321, 0.277) |
| (111, 0) | (279.417, 124.436) | (0.74948, -0.37532) | (-0.375, 0.188) |
| (56, 56) | (151.084, 218.919) | (-0.05260, 0.09710) | (0.026, -0.049) |
| (111, 111) | (243.563, 346.930) | (0.52539, 0.73715) | (-0.263, -0.369) |
<!-- /worked -->

Near the middle of the photograph the two readings are 0.03 pixel apart, and at the corners of the crop they are as much as 0.37 pixel apart. A third of a pixel in the photograph, here about a sixth of a pixel in the 112 crop, is small enough to leave every shape plausible and large enough to move the embedding, and no gradient check, no shape check and no "looks right" can tell the two apart, which is why `warp_grid` in `crates/arcface/src/align.rs` documents the first convention at the line that builds the grid and passes the mode to the kernel explicitly, as a number, rather than leaving a default to be inherited.

The second convention is what lies beyond the edge. A crop of a face near the border of a photograph reaches pixels that do not exist.

<!-- worked: faces.border_fill -->
One row of a picture whose first two pixels are 100 and 120 (pixel i centred at x = i), sampled between them and beyond the left edge by linear interpolation. The pixel at x = -1 does not exist; the three columns say what it is taken to be: 0 (zero padding), the edge pixel 100 (clamping) or a constant grey of 128.

| x | zero padding | clamp to the edge | constant grey 128 |
|---|---:|---:|---:|
| 0.50 | 110.00 | 110.00 | 110.00 |
| 0.00 | 100.00 | 100.00 | 100.00 |
| -0.25 | 75.00 | 100.00 | 107.00 |
| -0.50 | 50.00 | 100.00 | 114.00 |
<!-- /worked -->

Brain's sampler, the `grid_sample` kernel, uses zero padding, which drops the missing tap from the interpolation with the weight it would have had: at $x = -0.25$ the value is $0.75 \times 100 = 75$, where clamping to the edge would give 100. The reference that the crates are ported from fills with a constant grey, and the gap shows where it matters. The commit that added the parsing crate (commit `eac63b935`) reports that the network alone matches the reference to a cosine of 1.0000000000 and that the whole chain of alignment, parsing and mask reaches 0.9997813, and it attributes the gap to the border, carried through a hard decision at the boundary of a class. The module comment of `align.rs` explains a second gate. The warp is held tightly against the reference's own floating-point sampling routine, and only loosely, to half a grey level, against an image library's affine warp that the reference's alignment helper also offers, because that one interpolates with weights quantized to 5 bits. In the words of the comment, reporting a tight match to it "would mean the grid was wrong in a way that happened to cancel". A gate tighter than the precision of what it is compared with is not evidence of correctness. The division of labour in the crate is the plain one: the five-point solve and the grid, which are geometry over 20 numbers and $112 \times 112 \times 2$ positions, are host code, and the per-pixel work, a bilinear read for every channel of every destination pixel, is a kernel.

## A detector that also reports where the eyes are

The five points have to come from somewhere, and in the pipeline of the figure they come from the detector, which is the first of the three models. **Face detection** here is detection in the sense of [Chapter 4](04-detection.md), with one class, face, and one extra output: besides the box and the score, each candidate carries five landmarks, the points of the previous two sections. The detector of `crates/scrfd` is SCRFD-10GF, in the variant with keypoints. Its structure uses everything the volume has built. The backbone is a small residual network, a stem of three 3 by 3 convolutions of widths 28, 28 and 56 (the first with stride 2) and a 2 by 2 max pool, then four stages of 3, 4, 2 and 3 basic blocks of widths 56, 88, 88 and 224, whose strided blocks use the average-pool shortcut that [Chapter 3](03-cnns.md) showed to be a different network from the strided one. A neck of the feature-pyramid kind from the same chapter, top-down and then bottom-up, brings three maps of width 56 at strides 8, 16 and 32. On each of the three maps sits its own head: three 3 by 3 convolutions of width 80, and then three 3 by 3 convolutions that write, for each of two anchors, 1 score, 4 distances and 10 landmark offsets. Every convolution carries a bias, because the batch normalizations of the released network were folded into the convolutions before the weights were published (the folding of [Chapter 3](03-cnns.md) again), and the ReLUs sit where a residual network puts them.

<!-- worked: faces.scrfd_counts -->
The three levels of the face detector on its 640 by 640 canvas, with two anchors per location and, for each, 1 score, 4 distances and 10 landmark offsets (15 numbers).

| stride | grid | locations | candidate rows | numbers written |
|---|---:|---:|---:|---:|
| 8 | 80 x 80 | 6,400 | 12,800 | 192,000 |
| 16 | 40 x 40 | 1,600 | 3,200 | 48,000 |
| 32 | 20 x 20 | 400 | 800 | 12,000 |
| all |  | 8,400 | 16,800 | 252,000 |
<!-- /worked -->

The detector reads a 640 by 640 canvas, so the three levels have grids of 80, 40 and 20 and there are 16,800 candidate rows, twice the 8,400 cells because each location has two anchors. Each row is 15 numbers, 252,000 per image. The cost of those rows, and of everything that makes them, can be counted from the layer list above the same way that the chapter on dense prediction counted the parsing network's.

<!-- worked: faces.scrfd_cost_table -->
Parameters (weights and biases, the batch normalizations folded into the convolutions, and the three learned scales) and multiply-adds of the detector's convolutions on a 640 by 640 input, from the layer list of the text. Pooling, the shortcut average pools, the additions, the activations and the nearest-neighbour doublings are not counted.

| part | output | parameters | multiply-adds |
|---|---:|---:|---:|
| stem: three 3x3 convolutions | 56 x 160 x 160 | 22,036 | 2,245,017,600 |
| stage 1: 3 basic blocks | 56 x 160 x 160 | 169,680 | 4,335,206,400 |
| stage 2: 4 basic blocks | 88 x 80 x 80 | 537,944 | 3,437,772,800 |
| stage 3: 2 basic blocks | 88 x 40 x 40 | 286,968 | 458,444,800 |
| stage 4: 3 basic blocks | 224 x 20 x 20 | 2,456,608 | 982,016,000 |
| neck: lateral, smoothing, down and merge convolutions | 56 x 80, 40, 20 | 220,528 | 394,419,200 |
| three heads (3 convolutions, then 1, 4 and 10 values per anchor) | 30 x 80, 40, 20 | 532,173 | 1,487,808,000 |
| total |  | 4,225,937 | 13,340,684,800 |

Twice the multiply-adds, the FLOPs, is 26,681,369,600. The same layer list on a 640 by 480 input needs 10,005,513,600 multiply-adds (three quarters of the square canvas).
<!-- /worked -->

Half the work, 49 percent of the multiply-adds, is in the stem and the first stage, which run at the 320 by 320 and 160 by 160 resolutions, and the three heads together take 11 percent. It is the price of the smallest faces: the finest output has stride 8, so a face 16 pixels wide spans two cells, and keeping that much detail means running convolutions at 160 by 160 and above. The 10GF of the name is a budget of about ten billion operations at a 640 by 480 input, and there the layer list gives 10.0 billion multiply-adds, three quarters of the square canvas.

### What a candidate row means

[Chapter 4](04-detection.md) decoded a point with distances to the four sides. The detector of this chapter does the same, with three differences that are easy to miss and each of which moves every box. First, its anchor point is the corner of the cell: location $(g_x, g_y)$ of a level of stride $s$ is the point $(g_x s, g_y s)$, not the centre $(g_x + 0.5, g_y + 0.5)s$ of the earlier chapter. A network learns whichever convention it was trained with, and a half-cell shift is eight pixels at stride 16. Second, the two anchors of a location are the same point: the code repeats each location's centre twice, so a "second anchor" is a second prediction at one place, and the crate, which does not train the detector, says nothing about what the second one is for. Third, the head's regression output is multiplied by a learned scalar, one for each level, before the decoder multiplies it by the stride; the crate applies the scalar on the device and hands the decoder the scaled numbers, which is why the decoder's only factor is the stride.

How the numbers are laid out is the next thing. The head writes a map with $2 \times 4$ channels at every location of the grid; the candidate rows that the decoder reads are a transposition of that map, row $(g_y W + g_x)A + a$ for location $(g_x, g_y)$ of a grid $W$ wide and anchor $a$ of $A$, and the four distances of anchor $a$ are channels $4a$ to $4a + 3$. The decoder builds the anchor centres in the same order, which the code calls out as a trap: interleaving the two orders differently is "a silent, plausible box shift", and its tests (`anchor_rows_repeat_each_location_contiguously` and `head_output_permute_matches_transpose_then_reshape`) pin both on small cases. Here is one row decoded by hand.

<!-- worked: faces.scrfd_decode_row -->
One candidate of the stride-16 level (a 40 by 40 grid, two anchors per location), location column 17, row 9, second anchor. The head's numbers are the distances and offsets in strides, already multiplied by the level's learned scale.

| quantity | value |
|---|---|
| row of the table | (9 * 40 + 17) * 2 + 1 = 755 |
| anchor point | (17 * 16, 9 * 16) = (272, 144) |
| score | sigmoid(1.9) = 0.8699 |
| distances (l, t, r, b) | (1.4, 2.0, 1.3, 1.9) strides = (22.4, 32.0, 20.8, 30.4) pixels |
| box on the canvas | (249.6, 112.0, 292.8, 174.4) |
| box in the photograph (divide by 0.5) | (499.2, 224.0, 585.6, 348.8) |

The five landmark offsets (dx, dy) in strides, and the points they give on the canvas and in a photograph that was shrunk by 0.5 to fit it:

| landmark | offset | canvas | photograph |
|---|---|---:|---:|
| left eye | (-0.7, -0.9) | (260.8, 129.6) | (521.6, 259.2) |
| right eye | (0.6, -0.9) | (281.6, 129.6) | (563.2, 259.2) |
| nose | (0.0, 0.2) | (272.0, 147.2) | (544.0, 294.4) |
| left mouth | (-0.5, 0.9) | (264.0, 158.4) | (528.0, 316.8) |
| right mouth | (0.5, 0.9) | (280.0, 158.4) | (560.0, 316.8) |
<!-- /worked -->

The decode, with the score threshold and the suppression that follows it, is short enough to write in full. It uses `iou_incl`, the intersection over union of the next section, and the `nms` of [Chapter 4](04-detection.md) with that function in place of `iou`:

```text
function decode_level(score: array<real>[R], dist: array<real>[R, 4], kps: array<real>[R, 10],
                      side: int, stride: int, anchors: int, thr: real, det_scale: real)
        -> list<(array<real>[4], real, array<real>[10])>:
    out = []
    for r in 0 .. R:
        if score[r] >= thr:
            p = r // anchors
            ax = (p mod side) * stride
            ay = (p // side) * stride
            box = [(ax - dist[r, 0] * stride) / det_scale, (ay - dist[r, 1] * stride) / det_scale,
                   (ax + dist[r, 2] * stride) / det_scale, (ay + dist[r, 3] * stride) / det_scale]
            pts = [0.0 for k in 0 .. 10]
            for j in 0 .. 5:
                pts[2 * j] = (ax + kps[r, 2 * j] * stride) / det_scale
                pts[2 * j + 1] = (ay + kps[r, 2 * j + 1] * stride) / det_scale
            append (box, score[r], pts) to out
    return out
```

The three levels' lists are concatenated, in order, and suppressed once, and the thresholds of the reference are 0.5 for the score and 0.4 for the overlap. The division by `det_scale` happens before the suppression. That order is a decision, and its justification fills the next section.

## The two conventions that decide which box survives

Suppression keeps the most confident box of a pile and drops the others that overlap it by more than a threshold. [Chapter 4](04-detection.md) measured overlap by the intersection over union of half-open boxes, in which a box from $x_1$ to $x_2$ is $x_2 - x_1$ wide. The detector's reference computes it the older way, inherited from the first region-based detectors: a box covers the pixels $x_1, \dots, x_2$ inclusive, so it is $x_2 - x_1 + 1$ wide, and

```text
function iou_incl(a: array<real>[4], b: array<real>[4]) -> real:
    iw = max(min(a[2], b[2]) - max(a[0], b[0]) + 1.0, 0.0)
    ih = max(min(a[3], b[3]) - max(a[1], b[1]) + 1.0, 0.0)
    inter = iw * ih
    area_a = (a[2] - a[0] + 1.0) * (a[3] - a[1] + 1.0)
    area_b = (b[2] - b[0] + 1.0) * (b[3] - b[1] + 1.0)
    return inter / (area_a + area_b - inter)
```

It is a small change, and it makes the answer depend on the units. Without the $+1$, scaling both boxes by any factor changes neither the intersection nor the union in proportion, so the ratio is the same at every scale; with it, the $+1$ is a fixed one pixel, and its weight against the box grows as the box gets smaller.

<!-- worked: faces.nms_conventions -->
Two boxes a = (0, 0, 9, 9) and b = (5, 0, 14, 9) with scores 0.9 and 0.8. IoU in the half-open convention of the detection chapter (`detection.iou`) and in the inclusive convention (a side from x1 to x2 counts x2 - x1 + 1 pixels).

| coordinates | IoU, no +1 | IoU, with +1 |
|---|---:|---:|
| canvas pixels | 0.2857 | 0.3333 |
| photograph pixels (divided by 2) | 0.2857 | 0.3750 |

Boxes kept at the threshold 0.35 (inclusive convention):

| order of the two steps | boxes kept |
|---|---:|
| suppress on the canvas, then scale | 2 |
| scale, then suppress | 1 |

Two boxes (0, 0, 9, 9) and (1, 0, 10, 9) with the same score 0.5 and threshold 0.5 (inclusive IoU 0.8182); indices of the boxes kept, in order:

| tie rule | kept |
|---|---:|
| higher index first (the reference) | [1] |
| lower index first (a stable descending sort) | [0] |
<!-- /worked -->

The pair of boxes straddles a threshold of 0.35 only because of the scale: the same two faces overlap 0.3333 in canvas pixels and 0.375 in the photograph's, which is half the size. Suppressing on the canvas and then scaling keeps both; scaling and then suppressing keeps one. The reference divides by the scale first, and brain's `decode` does the same, because a detection set that disagrees with the reference's on exactly the small faces is the kind of error that nothing visible reveals. The test `nms_runs_in_source_pixels_not_detector_pixels` in `crates/scrfd/src/detect.rs` builds this very pair, runs the decoder at `det_scale` 1 and 2, and asserts two faces and one. The second rule in the table is the tie. Two boxes with the same score must be ordered, and the reference orders by a reversed stable ascending sort, which puts the higher index first among equals; a stable descending sort would keep the lower index first and the other member of the pair would survive. The two rules are indistinguishable until two boxes have a score that is exactly equal, and then they pick different boxes.

Commit `0337795a8`, which added the first version of the stack, lists both as defects that the goldens could not see, and gives the reason: all four of the photographs they were made from contain exactly one face. A fixture with one face never has two boxes to suppress, so neither the unit of the overlap nor the tie can matter. A parity gate is only as strong as its fixtures' variety, and a pair of faces closer together than the threshold is a case to build on purpose.

## Canvas and constants

The detector is built for one square size. A photograph has to be put on a 640 by 640 canvas, and brain follows the reference exactly. The picture is resized so its longer side is 640, the shorter one truncated to a whole number of pixels, and it is placed in the top left corner of the canvas, not centred; the rest is zero. [Chapter 4](04-detection.md) built the centred letterbox for the YOLO detectors, and `imaging::letterbox_rgb` is a centred, nearest-neighbour one tuned to that convention, so the detector builds its own from the bilinear resize and the padding kernel. The top-left placement has one consequence worth having: the inverse needs no padding, only a division, and the decode does it with one `det_scale`.

<!-- worked: faces.det_canvas -->
The detector's canvas for five pictures: the picture is resized to keep its aspect ratio so that its longer side is 640 (the other side is truncated to a whole number of pixels), placed in the top left corner of the 640 by 640 canvas, and the rest is zero. det_scale is new height / height. The last column is where the canvas column 640 lands in the picture when it is divided by det_scale, minus where the picture's own horizontal scale (new width / width) puts it.

| picture | resized to | padding right / bottom | det_scale | horizontal scale | error at the right edge, px |
|---|---:|---:|---:|---:|---:|
| 1280 x 720 | 640 x 360 | 0 / 280 | 0.500000 | 0.500000 | 0.00 |
| 720 x 1280 | 360 x 640 | 280 / 0 | 0.500000 | 0.500000 | 0.00 |
| 640 x 640 | 640 x 640 | 0 / 0 | 1.000000 | 1.000000 | 0.00 |
| 1000 x 333 | 640 x 213 | 0 / 427 | 0.639640 | 0.640000 | 0.56 |
| 1279 x 720 | 640 x 360 | 0 / 280 | 0.500000 | 0.500391 | 1.00 |
<!-- /worked -->

The scale is derived from the height, `new_height / height`, for every picture, and the table shows what that means for a wide picture whose truncated height is not exactly proportional: for 1000 by 333 the canvas column 640 goes back to 640 / 0.63964 = 1000.56 where the picture's own horizontal scale says 1000, and for 1279 by 720 the error at the right edge is a full pixel. Both are the reference's own arithmetic, followed to the digit because the goldens were made with it, and for ordinary pictures the error is a pixel or less at the far edge, against faces that are tens of pixels wide. It grows with the aspect ratio: a strip of 4000 by 13 is resized to 640 by 2, `det_scale` is $2/13 = 0.1538$ where the horizontal scale is $0.16$, and the far edge is off by 160 pixels.

The last constants are the normalizations, where the crates have a comment that is repeated in both of them, and for good reason: the detector subtracts 127.5 and divides by 128, and the embedding network subtracts 127.5 and divides by 127.5. Both are "map a byte to roughly $[-1, 1]$".

<!-- worked: faces.normalisations -->
Pixel value to network input, for the detector (subtract 127.5, divide by 128), for the embedder (subtract 127.5, divide by 127.5), and for the detector from a picture stored in [0, 1] (subtract 127.5/255, divide by 128/255). The detector constants shrink every deviation from the mean by the factor 127.5/128 = 0.99609.

| pixel | detector | embedder | detector from [0, 1] |
|---|---:|---:|---:|
| 0 | -0.99609 | -1.00000 | -0.99609 |
| 64 | -0.49609 | -0.49804 | -0.49609 |
| 128 | 0.00391 | 0.00392 | 0.00391 |
| 255 | 0.99609 | 1.00000 | 0.99609 |
<!-- /worked -->

The detector's constants shrink every deviation from the mean by 127.5/128, that is 0.39 percent. The wrong constant in either direction yields plausible boxes and plausible vectors; nothing is out of range, and nothing crashes. The two crates therefore each own their pair (`crates/scrfd/src/config.rs` and `crates/arcface/src/config.rs`), neither imports the other's, and each has a test that fixes it. When the picture arrives over brain's wire it is in $[0, 1]$ and not bytes, and the same constants are used in those units, $(x - 127.5/255)/(128/255)$ in the last column, which is the same function; the crate states the identity as a test (`unit_norm_is_the_same_affine_in_zero_to_one_units`) rather than keeping a second pair of constants.

When a picture holds several faces the question of which one the recognition path embeds has to be answered, and the answer in the reference, which brain follows, is the largest by box area, not the most confident.

<!-- worked: faces.primary_face -->
| face | score | box size | area |
|---|---:|---:|---:|
| f0 | 0.93 | 60 x 80 | 4,800 |
| f1 | 0.88 | 140 x 170 | 23,800 |
| f2 | 0.71 | 100 x 100 | 10,000 |

The highest score is f0; the largest area, the face the recognition path embeds by default, is f1.
<!-- /worked -->

The `select` parameter of the `embed` action in `crates/arcface/src/caps.rs` offers both, `largest` by default and `score`. The detector's own action, `detect`, returns all faces sorted by score and truncates to `max_faces` when asked. The difference matters in a group photograph, where the most confident face is a crisp one at the edge and the face the caller meant is larger.

## From a crop to a vector

A 112 by 112 crop, aligned and normalized, is the input of the second model, a residual network of 100 layers whose output is 512 numbers. That output is the **identity embedding**: a vector produced from a photograph of a person so that photographs of the same person give vectors that point the same way, and photographs of different people give vectors that do not. Everything about the vector is a convention of use. It is not "the identity", and no coordinate means anything; what is meaningful is the angle between two of them, and so a comparison of two faces is the cosine similarity of [Volume I](../v01-mathematics/03-vectors.md) of their embeddings, against a threshold chosen for the error rate the application can bear. The word returns in later volumes, for the vector of a voice and for what an image generator is conditioned on, and the use is always the same: a fixed-length vector that says who, and nothing else.

The network is IResNet-100, a ResNet of the pre-activation kind whose block [Chapter 3](03-cnns.md) built (the normalization first, and the shortcut reading the block's input). In `crates/arcface` its layers are a stem convolution, then four stages of 3, 13, 30 and 3 blocks of widths 64, 128, 256 and 512, in which the first block of each stage halves the map, a batch normalization, a flatten of the final $512 \times 7 \times 7$ map into 25,088 numbers, and one linear layer from 25,088 to 512 followed by a last batch normalization. The activation is a **PReLU**, a ReLU whose slope for negative inputs is a learned number, one for each channel, instead of 0. Count it.

<!-- worked: faces.iresnet_table -->
Parameters and multiply-adds of the embedding network on a 112 by 112 input, from the layer list of the text: 49 blocks (3, 13, 30 and 3 per stage) of an entry batch normalization (4 numbers per channel), two 3x3 convolutions with biases, a PReLU with one slope per channel, and in the first block of each stage a 1x1 convolution on the shortcut. Activations and additions are not counted.

| part | output | parameters | multiply-adds |
|---|---:|---:|---:|
| stem: one 3x3 convolution and PReLU | 64 x 112 x 112 | 1,856 | 21,676,032 |
| stage 1: 3 blocks | 64 x 56 x 56 | 226,688 | 1,053,294,592 |
| stage 2: 13 blocks | 128 x 28 x 28 | 3,779,840 | 3,127,771,136 |
| stage 3: 30 blocks | 256 x 14 x 14 | 35,180,800 | 7,058,358,272 |
| stage 4: 3 blocks | 512 x 7 x 7 | 13,117,440 | 815,661,056 |
| batch norm, flatten, one 25,088 -> 512 linear layer, batch norm | 512 | 12,849,664 | 12,845,056 |
| total |  | 65,156,288 | 12,089,606,144 |

Twice the multiply-adds, the FLOPs, is 24,179,212,288. The tensors of the file, counted as the text counts them (a block has 9, the four shortcut convolutions 2 each), number 462.
<!-- /worked -->

The count is the arithmetic the importer checks: the released file has 462 tensors, which are 3 for the stem (a convolution weight, its bias and the PReLU slopes), $49 \times 9$ for the blocks, $4 \times 2$ for the shortcut convolutions, and 4, 2 and 4 for the last batch normalization, the linear layer and the batch normalization after it. The three batch normalizations that survive the folding are the entry one of every block, the last one, and the one after the linear layer. The network is not small: 65 million parameters, 24 billion FLOPs for one face. More than half of the work, 58 percent, is in the 30 blocks of the third stage at a 14 by 14 resolution, and a fifth of the parameters, 12.8 million, sit in the one linear layer at the end that reads the whole 7 by 7 map and is the reason a crop of a fixed size is needed at all: its input width is $512 \times 7 \times 7$, and a different crop size would be a different layer.

The output has no length constraint. The released network ends in a batch normalization, so the 512 numbers are not normalized, and their norm is large, in the range 15 to 20 according to the code's own comment. Whether to normalize is a decision for the consumer. The `embed` action normalizes, because it promises a cosine-ready vector. An image generator that conditions on the identity wants the raw vector, as the reference does, and a consumer that took the normalized one would receive a first half about twenty times too small with nothing structural to say so: knowledge #10 records the reference's numbers, 20.11 for the length of the identity half of the conditioning vector against 1.0000 for the half that is normalized, and commit `6a9da997d` is where the shared core `embed_raw_chw` was extracted so that both consumers run one implementation and each applies its own convention. The cosine of a parity test has the opposite blindness, and it is the other face of the same fact: cosine ignores length, so `crates/arcface/tests/it/parity.rs` reports cosine and the largest absolute difference together ("cosine alone cannot see a scale error, and `max_abs` alone cannot see a rotation", the module says, and knowledge #2 generalizes it), and a stage passes only on both.

### A threshold, and what it trades

Given vectors, who is the same person? Take the cosine of two embeddings and say "the same" at or above a threshold $\tau_{\mathrm{id}}$. What the threshold can do depends on how far apart the cosines of genuine pairs (two photographs of one person) and impostor pairs (two people) lie. The next block simulates the situation. It is not a trained network: it draws 20 identities as random unit vectors of width 16, makes five "photographs" of each by adding noise of size $\rho$, and counts. Two spreads $\rho$ of the same noise stand for a tight embedding and a loose one.

<!-- worked: faces.verification_threshold -->
Simulated embeddings, not a trained network: 20 identities, 5 photographs each, vectors of width 16. The generator is `faces.verification` (seed 0xa17c8647f589891e (11636323180838750494)). Draw order: the 20 centres (identity 0 to 19, 16 standard normals each, each divided by its length), then the noise (identity, photograph, 16 standard normals). A photograph of identity i is its centre plus rho / sqrt(16) times the noise vector, divided by its length. The same noise is used for both spreads rho. Cosines are dot products of the unit vectors, summed over the 16 components in order. Genuine pairs are two photographs of one identity, impostor pairs of two identities. The closest two of the 20 centres have the cosine 0.625.

| rho | pairs | count | mean cosine | smallest | largest |
|---|---|---:|---:|---:|---:|
| 0.5 | genuine | 200 | 0.796 | 0.608 | 0.949 |
| 0.5 | impostor | 4,750 | 0.017 | -0.774 | 0.745 |
| 1.0 | genuine | 200 | 0.480 | 0.058 | 0.870 |
| 1.0 | impostor | 4,750 | 0.010 | -0.757 | 0.703 |

Wrong rejections (genuine pairs below the threshold) and wrong acceptances (impostor pairs at or above it):

| rho | threshold | genuine rejected | impostors accepted |
|---|---:|---:|---:|
| 0.5 | 0.2 | 0 of 200 | 1091 of 4750 |
| 0.5 | 0.3 | 0 of 200 | 555 of 4750 |
| 0.5 | 0.4 | 0 of 200 | 224 of 4750 |
| 0.5 | 0.5 | 0 of 200 | 88 of 4750 |
| 1.0 | 0.2 | 15 of 200 | 1053 of 4750 |
| 1.0 | 0.3 | 29 of 200 | 563 of 4750 |
| 1.0 | 0.4 | 58 of 200 | 259 of 4750 |
| 1.0 | 0.5 | 107 of 200 | 99 of 4750 |
<!-- /worked -->

With the tight spread no genuine pair is below 0.608, and a threshold of 0.5 rejects none of the 200 genuine pairs and accepts 88 of the 4,750 impostor pairs; some impostors are bound to survive, because the two closest of the 20 centres have a cosine of 0.625 already. With the loose spread the genuine cosines begin at 0.058 and a threshold of 0.5 rejects 107 of the 200 genuine pairs, while still accepting 99 impostors. Lowering the threshold to 0.2 makes the loose embedding reject 15 genuine pairs and accept 1,053 impostors. Among the four thresholds there is none at which the loose embedding does what the tight one does at 0.5: the two kinds of error cannot both be driven down by moving the threshold, because the overlap of the two distributions is in the embedding, and the only cure is to make the embedding tighter. That is the job of the loss of the next section. The same lesson is in knowledge #103, from the measurement that motivated the identity gate in `crates/yolov8`: strangers' faces scored at most 0.12 cosine against the reference and every target face large enough to identify scored 0.27 or better, a gap, the entry says, that a class logit never gave, with the remark that this is one held-out set and so a measurement and not a constant. The entry also records why a class head on a detector cannot do this job: its features are trained to separate categories and discard exactly the differences inside one.

## Training a space where angles are the metric

How would one train a network to produce such vectors? The obvious recipe is a classifier: take photographs of $N_{\mathrm{id}}$ people, put a linear layer after the embedding with one row per person, train it with the softmax cross-entropy of [Volume III](../v03-neural-networks/03-softmax.md), and throw the layer away afterwards, keeping the embedding. It works, up to a point. The trouble is that a classifier asks only that the right logit be the largest, and the logit is $\mathbf{w}_j\cdot\mathbf{e} = \lVert\mathbf{w}_j\rVert\lVert\mathbf{e}\rVert\cos\vartheta_j$ for the embedding $\mathbf{e}$ and the row $\mathbf{w}_j$. That has three terms, and the comparison at test time uses one of them. A sample can reduce the loss by growing $\lVert\mathbf{e}\rVert$ without changing its angle to anything, and a row can win by being longer than the others; neither helps the cosine of two photographs of one person, and both are free ways of lowering the loss.

The repair is to remove the freedom: normalize the embedding to $\hat{\mathbf{e}} = \mathbf{e}/\lVert\mathbf{e}\rVert$, the rows to $\hat{\mathbf{w}}_j$, and write the logit of identity $j$ as the cosine $c_j = \hat{\mathbf{e}}\cdot\hat{\mathbf{w}}_j = \cos\vartheta_j$, where $\vartheta_j \in [0, \pi]$ is the angle between the embedding and the centre of identity $j$. Now nothing but angles can lower the loss. The rows $\hat{\mathbf{w}}_j$ are the identity centres of the next paragraphs, learned along with the network and discarded after it. But cosines lie in $[-1, 1]$, and a softmax over numbers that small is nearly uniform whatever the sample does, so a fixed **scale** $s$ multiplies them, as the temperature of [Chapter 8](08-clip.md) did, and the logit is $s\cos\vartheta_j$. How large must $s$ be?

<!-- worked: faces.scale_floor -->
The smallest loss that a classifier on cosines can reach when the sample points exactly at its own centre (cosine 1) and every one of the other N - 1 centres is orthogonal to it (cosine 0): ln(1 + (N - 1) e^(-s)), for scale s.

| scale s | N = 10 | N = 1000 |
|---|---:|---:|
| 1 | 1.461150 | 5.909472 |
| 4 | 0.152584 | 2.959966 |
| 16 | 0.000001 | 0.000112 |
| 64 | 0.000000 | 0.000000 |
<!-- /worked -->

With $s = 1$ the loss of a classifier over a thousand identities cannot go below 5.91, however perfect the embedding: the best sample, exactly on its own centre and orthogonal to every other, has a target probability of $e/(e + 999) = 0.0027$. With $s = 4$ the floor is 2.96. Only from $s = 16$ does the loss floor become negligible, $0.000112$ for a thousand identities, and the reference recipe uses $s = 64$. Without a large scale, the loss would be dominated by a constant that no training can remove.

### Making the angle pay

The scaled cosine classifier still asks little. Here is what it demands of a sample, in angles: the loss falls below 0.01 as soon as the target logit beats the rest by a certain amount, and with orthogonal centres that is a condition on one angle.

<!-- worked: faces.required_angle -->
The largest angle between a sample and its own centre at which the loss is at most 0.01, when the other N - 1 centres are orthogonal to the sample: cos(theta) must reach c = ln((N - 1) / (e^0.01 - 1)) / s. With an additive angular margin of m = 0.5 radian (28.6 degrees) the condition is on theta + m instead, which lowers the angle by exactly m. Angles in degrees.

| s | N | c needed | largest angle, plain | largest angle, margin 0.5 |
|---|---:|---:|---:|---:|
| 4 | 10 | 1.6993 | unreachable | unreachable |
| 4 | 1000 | 2.8767 | unreachable | unreachable |
| 16 | 10 | 0.4248 | 64.9 | 36.2 |
| 16 | 1000 | 0.7192 | 44.0 | 15.4 |
| 64 | 10 | 0.1062 | 83.9 | 55.3 |
| 64 | 1000 | 0.1798 | 79.6 | 51.0 |
<!-- /worked -->

With the scale 64 and a thousand identities, a sample whose embedding makes an angle of 79.6 degrees with its own centre, nearly a right angle, already has a loss below 0.01, when all the other centres are at 90. That is the weakness: the classifier is satisfied by a cluster that is 80 degrees wide, and two clusters of such width cannot be separated by a threshold on the angle between photographs. The **angular margin loss** of ArcFace (Deng et al., 2019) closes it by demanding more of the correct class than of the others: it adds a margin $m$ to the angle of the target before the cosine is taken, so that the target logit is $s\cos(\vartheta_y + m)$, where $y$ is the true identity, and the others stay $s\cos\vartheta_j$. In symbols,

$$
\mathcal{L}_{\mathrm{arc}} = -\ln \frac{e^{s\cos(\vartheta_y + m)}}{e^{s\cos(\vartheta_y + m)} + \sum_{j \ne y} e^{s\cos\vartheta_j}}. \tag{3}
$$

This is the softmax cross-entropy of [Volume III](../v03-neural-networks/03-softmax.md) over the logits $s\cos\vartheta_j$ with the target's replaced. Since $\cos(\vartheta_y + m)$ is smaller than $\cos\vartheta_y$ whenever $\vartheta_y + m \le \pi$, the target logit is pushed down, the loss on every sample rises, and the network can lower it again only by bringing the sample closer to its centre. The condition for a loss of at most $\varepsilon$, in the example above, is $\cos(\vartheta_y + m) \ge c^\ast$, so it is $\vartheta_y \le \arccos c^\ast - m$: the margin moves the allowed angle down by exactly $m$ radians. With $m = 0.5$ (28.6 degrees) the thousand-identity, scale-64 case now needs an angle of 51.0 degrees, and the scale-16 case 15.4 degrees; clusters must be roughly 29 degrees tighter than the unmargined classifier would have been content with. The decision boundary of two identities is where $\cos(\vartheta_y + m) = \cos\vartheta_j$, that is $\vartheta_j = \vartheta_y + m$: a gap of $m$ radians between the two angles, the same everywhere on the sphere. That is why the margin is added to the angle, not to the cosine or multiplied into it: a margin on the cosine is a different gap at different angles. (The two other margins of the family, a multiple of the angle in SphereFace and a subtraction from the cosine in CosFace, have that fault in one form or the other; Liu et al., 2017, and Wang et al., 2018.)

One sample shows the mechanics. A sample with cosines 0.62, 0.35, 0.10, $-0.05$ and $-0.30$ against five identities, the first its own, scale 16:

<!-- worked: faces.margin_logits -->
One sample, five identities, the cosines of the sample's embedding with the five centres are 0.62, 0.35, 0.10, -0.05 and -0.30, the sample's identity is the first, and the scale is s = 16. The target logit is s cos(theta + m) written as s (c cos m - sqrt(1 - c^2) sin m); the others are s c. The last column is the derivative of the loss with respect to the target cosine c, (p - 1) times d(target logit)/dc, with d/dc = s (cos m + sin m c / sqrt(1 - c^2)).

| margin m | target logit | largest other logit | probability of the target | loss | dloss/dc |
|---|---:|---:|---:|---:|---:|
| 0.0 | 9.920 | 5.600 | 0.9866 | 0.0135 | -0.214 |
| 0.2 | 7.228 | 5.600 | 0.8332 | 0.1825 | -3.035 |
| 0.5 | 2.687 | 5.600 | 0.0506 | 2.9846 | -19.086 |

Check of the derivative at c = 0.62, m = 0.5: the formula gives 20.102869 and the central difference with h = 1e-6 in 64-bit arithmetic gives 20.102869; the relative difference is below 1e-7.
<!-- /worked -->

Without a margin the sample is already well classified: the target logit 9.920 beats the next one by 4.3, the probability is 0.9866 and the loss 0.0135, and the gradient on its cosine is $-0.214$. With $m = 0.5$ the target logit falls to 2.687, below the largest other logit of 5.600, the probability to 0.0506 and the loss rises to 2.98, with a gradient of $-19.1$ on the cosine: the sample that the plain loss considered finished is now strongly pushed. A test in the training crate says it in one line, `the_angular_margin_increases_the_loss`: on one batch, margin 0.5 gives a larger loss than margin 0. In pseudocode, the logits of the loss are

```text
function margin_logits(c: array<real>[N], y: int, s: real, m: real) -> array<real>[N]:
    require 0 <= y < N
    z = [s * c[j] for j in 0 .. N]
    sn = max(sqrt(max(1.0 - c[y] * c[y], 0.0)), 1.0e-6)
    z[y] = s * (c[y] * cos(m) - sn * sin(m))
    return z
```

and the loss is $\operatorname{lse}(z) - z_y$. The target logit is written as $\cos\vartheta\cos m - \sin\vartheta\sin m$ with $\sin\vartheta = \sqrt{1 - c^2}$, because the network produces the cosine $c$, never the angle, and the two guards, the `max(..., 0.0)` inside the root and the floor of $10^{-6}$ outside it, are discussed below.

### Where the margin form misbehaves

The cosine of a sum is monotone only while the sum stays below $\pi$. For larger angles, $\cos(\vartheta + m)$ rises again, and a sample that is far from its centre would be rewarded for being farther. The table gives the curve and the figure shows it.

<!-- worked: faces.margin_curve -->
The target logit divided by the scale, as a function of the angle theta between the sample and its own centre, for the margin m = 0.5 (theta + m reaches pi at theta = pi - m = 151.35 degrees): the plain cosine, the margin form c cos m - sqrt(1 - c^2) sin m, and the margin form with the clamp c - m sin m for theta beyond pi - m.

| theta, degrees | cos(theta) | margin form | margin form with clamp |
|---|---:|---:|---:|
| 0 | 1.0000 | 0.8776 | 0.8776 |
| 30 | 0.8660 | 0.5203 | 0.5203 |
| 60 | 0.5000 | 0.0236 | 0.0236 |
| 90 | 0.0000 | -0.4794 | -0.4794 |
| 120 | -0.5000 | -0.8540 | -0.8540 |
| 140 | -0.7660 | -0.9804 | -0.9804 |
| 150 | -0.8660 | -0.9997 | -0.9997 |
| 160 | -0.9397 | -0.9886 | -1.1794 |
| 170 | -0.9848 | -0.9475 | -1.2245 |
| 180 | -1.0000 | -0.8776 | -1.2397 |

At theta = pi - m the clamped branch starts at -cos m - m sin m = -1.1173, while the margin form has reached -1 there: the clamp steps the target logit down by 0.1173 at the switch.
<!-- /worked -->

![The target logit divided by the scale against the angle between the sample and its own centre, from 0 to 180 degrees. The plain cosine falls monotonically from 1 to minus 1. The margin form starts at 0.8776, lies below the cosine, reaches minus 1 at 151.35 degrees and then rises again. The clamped variant follows the margin form up to 151.35 degrees, steps down there and continues to fall.](img/margin-target-logit.svg)

Beyond $\pi - m$ (151.35 degrees for $m = 0.5$) the margin form turns upward. The reference's release clamps it there, to $\cos\vartheta - m\sin m$, which is monotone, but at the switch it steps down by 0.1173, the discontinuity in the table. `crates/kernels/wgsl/arcface_margin.wgsl` implements the plain form on purpose, with a header called "the plain form, deliberately": the clamp, like the "easy margin" test of another variant, is a piecewise switch that puts a kink in the objective, a central finite difference straddles a kink, and the paper's formula is the one that can be validated. A clamped variant "would be a separate kernel with its own gate, not a flag here". It is a trade made in the open: the training objective is the smooth one, and what the clamp protects against, a sample more than 151 degrees from its own centre, is a sample that the network has already failed on.

The other trouble is at the other end, at $\vartheta = 0$, where the slope of the target logit with respect to the cosine is $s(\cos m + \sin m\, c/\sqrt{1 - c^2})$ and the square root is zero.

<!-- worked: faces.margin_slope -->
The slope of the target logit against the cosine c, s (cos m + sin m c / sqrt(1 - c^2)), for s = 64 and m = 0.5, with the square root floored at 1e-6 as in the forward pass, at several angles theta between the sample and its centre. The plain logit s c has the slope 64 everywhere. The third column is the slope against the angle itself, s sin(theta + m); the fourth is the slope against c multiplied by sin(theta), the size of dc/dtheta.

| theta, degrees | slope against c | slope against theta | slope against c times sin(theta) |
|---|---:|---:|---:|
| 60 | 7.388e+01 | 6.398e+01 | 6.398e+01 |
| 10 | 2.302e+02 | 3.997e+01 | 3.997e+01 |
| 1 | 1.814e+03 | 3.166e+01 | 3.166e+01 |
| 0.1 | 1.764e+04 | 3.078e+01 | 3.078e+01 |
| 0.01 | 1.759e+05 | 3.069e+01 | 3.069e+01 |

At c = 1 exactly the floor is active. The formula with the floored sine in the denominator gives 3.068e+07 there; the forward is s (c cos m - 1e-6 sin m) on that side of the floor, whose slope is s cos m = 56.17.
<!-- /worked -->

The slope against the cosine grows without bound as the sample approaches its centre: 1,814 for $s = 64$ at one degree, 28 times the plain logit's 64, and it would reach $3.07 \times 10^7$ at $c = 1$ if the floored sine stayed in the denominator. The slope against the angle itself is harmless, $s\sin(\vartheta + m)$, which tends to $s\sin m = 30.7$. They are the same quantity written in two parametrizations, and the product in the last column, the slope against $c$ times $\sin\vartheta$ (how fast $c$ moves with the angle near zero), is the finite one. The divergence is therefore an artefact of asking the question in terms of the cosine, which is what the network computes, and it is cancelled by a factor that vanishes at the same place further down the chain; only in finite arithmetic does the product of an enormous number and a tiny one stop being a clean cancellation. That is the reason for the floor of $10^{-6}$ in the forward pass, applied to $\sin\vartheta$, and the reason the header of `arcface_margin_bwd.wgsl` insists that the backward use the same floor: "the forward and the backward must leave the $|\cos| = 1$ singularity at the same place, or the finite-difference check sees two different functions". The header goes on to say what the backward must do where the floor is active: there the forward no longer depends on $c$ through the sine, so the slope is $s\cos m$ and nothing more. Writing this chapter found that the kernel did not do it. It divided by the floored sine and returned the spike of the block above, the finite-difference check could not see it because no fp32 cosine lands between 1 and the floor, and the test the header named for that branch had never been written. Commit `d4140ed78` gives the branch its constant slope and adds the test, which checks the adjoint at $c = 1$ itself. The floor and the clamp of the sum under the root are a second, smaller point of the same kind. Rounding can push a cosine a hair beyond 1, and the root of a negative number is not a number; the `max(..., 0.0)` keeps it a number.

## What the training graph in brain does

The crate holds, besides the inference graph, a training graph of the same network (`crates/arcface/src/train.rs`). It trains the embedding backbone only: the detector and the alignment are preprocessing (the reference recipe trains on crops that are already aligned) and carry no recognition gradient. The head of (3) is not in the released weights, because the release ships the embedding alone, so `head.weight` of shape $[N_{\mathrm{id}}, 512]$ is initialized, as it would be when fine-tuning onto a new set of identities. The graph of the head is the one of the section above, and `ArcFaceTrainConfig::insightface` carries the reference recipe's constants, $s = 64$ and $m = 0.5$ radian. In order: the 512 numbers $\mathbf{e}$ that the network writes (after the linear layer and the last batch normalization) are normalized to unit length; so are the rows of `head.weight`, one for each identity; a matrix product of the batch with the transposed rows gives the table of cosines; the margin kernel turns it into the scaled logits of (3); and the mean cross-entropy over the batch is the loss.

Three details show how a test has to be built to see what the training code does. First, the finite-difference check of the whole graph (`gradcheck::check_arcface`) runs on a tiny IResNet, and its depth is not arbitrary: `ArcFaceTrainConfig::tiny` has the blocks $[2, 1, 1, 1]$, not $[1, 1, 1, 1]$, and the extra block is the point. The first block of a stage always strides and so always has a shortcut convolution; every later block takes the identity shortcut, a different branch of the code in both the forward and the backward, and in IResNet-100 45 of the 49 blocks take it. A tiny network with one block per stage would never execute the branch that most of the real network is made of. Second, the check uses a scale of $s = 8$, not 64: the margin kernels are linear in $s$ (it multiplies the logit and its derivative), so the code path is the same, while $s = 64$ over five classes saturates the softmax so hard that the finite difference of the loss is dominated by rounding. The margin of 0.5 is the real one, since that is the part with curvature. Third, the three batch normalizations that survive in the released graph run in training mode, on batch statistics, because only then does $\gamma$ carry a gradient, and the running-average update is left off so that the forward is deterministic, which a central difference needs. The module says that a real training loop must turn it back on.

The backward of the margin is the pair of kernels of the previous section, with the slope $s(\cos m + \sin m\, c/\sqrt{1 - c^2})$ on the target column and $s$ on the others. Gradient accumulation is tested separately, since a finite difference calls the backward once per step and cannot tell a stage that adds its gradient from one that assigns it: the test `every_parameter_gradient_accumulates_across_backward_calls` runs the backward twice and requires every gradient to double. That is the failure a batch size larger than memory allows would expose, found in advance.

## Labelling the pixels of a face

The third model answers a different question about the same crop: which pixel is what. **Face parsing** is semantic segmentation, in the sense of [Chapter 5](05-dense-prediction.md), with a fixed label set made of the parts of a head: the network writes 19 class scores for every pixel, and the class of a pixel is the largest. The network of `crates/bisenet` is the BiSeNet whose two-path design and arithmetic Chapter 5 already worked through (in brain the detail path is the context path's own stride-8 map, its parameters and multiply-adds are in that chapter's table), so what this chapter adds is the part that is about faces: what the network is shown and what is done with its answer.

What it is shown is a crop of a different kind from the recognition network's. Both crops are cut by the same kind of fit from the same five landmarks, to a template; but the template is the choice, and the parsing network's, a crop of 512 by 512 with its own five points, is not the recognition network's.

<!-- worked: faces.template_comparison -->
|  | recognition template | parsing template |
|---|---:|---:|
| crop side, pixels | 112 | 512 |
| distance between the eye centres, pixels | 35.24 | 125.92 |
| that distance as a share of the side | 0.3146 | 0.2459 |
| eye midpoint to mouth midpoint, as a share of the side | 0.3633 | 0.2563 |

The least-squares similarity that takes the parsing template onto the recognition template has scale 0.2921 (a plain shrink of 512 to 112 would be 0.2188), angle -0.0087 degrees and a residual of 1.922 pixels.
<!-- /worked -->

The distance between the eyes is 31.5 percent of the side in the recognition crop and 24.6 percent in the parsing crop; a plain shrink would take the one to the other at a factor of 0.2188, and the best similarity between them has scale 0.2921, a factor of 1.34 more. The parsing crop shows more of the head around the face, and it must: among its 19 classes are the neck, the clothing, the ears and the hats, and a crop cut tightly to the eyes and the mouth would not contain them. The geometry is a property of the model's training, not a free parameter, and brain's `crates/bisenet/src/align.rs` therefore holds its own template, `FFHQ_DST_512`, and its own copy of the warp: the module says why (a kernel index is meaningful only in the list that built it, knowledge #13 and #93) and what it costs (a few lines). Its normalization is the third in this chapter, the usual image-network constants, mean $(0.485, 0.456, 0.406)$ and standard deviation $(0.229, 0.224, 0.225)$ applied to a picture in $[0, 1]$, exported as `imagenet_normalize` so that no caller keeps a private copy of six constants.

The answer is a class map, and a map is not yet what a downstream model wants. The consumer in brain is an image generator that conditions on a face (`crates/pulid`): it wants the person, not the background, and its reference recipe takes the parsing network's output and whitens everything that is not face. The rule is in `crates/bisenet/src/mask.rs`: the pixels whose class is one of $0, 16, 18, 7, 8, 9, 14, 15$ (the background, and the cloth, hat, ear, earring, neck and necklace classes of the reference's list) become pure white, and every other pixel becomes its gray value $0.299R + 0.587G + 0.114B$ replicated on three channels, computed from the original aligned crop and not from its normalized copy. The comment in the module says that the list is the exact one, "not a semantic re-derivation of it".

<!-- worked: faces.parsing_mask -->
A picture of 12 pixels (3 by 4, row by row) after parsing: the class of each pixel, the pixel's colour in [0, 1], and the masked colour. The classes [0, 7, 8, 9, 14, 15, 16, 18] are whitened, the others become their gray value 0.299 R + 0.587 G + 0.114 B on all three channels.

| pixel | class | treatment | colour | masked colour |
|---:|---|---|---|---|
| 0 | 0 | whitened | (0.20, 0.40, 0.80) | (1.0000, 1.0000, 1.0000) |
| 1 | 17 | kept | (0.10, 0.08, 0.05) | (0.0826, 0.0826, 0.0826) |
| 2 | 1 | kept | (0.85, 0.62, 0.50) | (0.6751, 0.6751, 0.6751) |
| 3 | 10 | kept | (0.78, 0.55, 0.44) | (0.6062, 0.6062, 0.6062) |
| 4 | 16 | whitened | (0.30, 0.30, 0.60) | (1.0000, 1.0000, 1.0000) |
| 5 | 14 | whitened | (0.80, 0.60, 0.50) | (1.0000, 1.0000, 1.0000) |
| 6 | 1 | kept | (0.88, 0.65, 0.52) | (0.7039, 0.7039, 0.7039) |
| 7 | 11 | kept | (0.60, 0.20, 0.20) | (0.3196, 0.3196, 0.3196) |
| 8 | 18 | whitened | (0.15, 0.15, 0.15) | (1.0000, 1.0000, 1.0000) |
| 9 | 15 | whitened | (0.90, 0.80, 0.30) | (1.0000, 1.0000, 1.0000) |
| 10 | 12 | kept | (0.65, 0.25, 0.25) | (0.3696, 0.3696, 0.3696) |
| 11 | 1 | kept | (0.86, 0.63, 0.51) | (0.6851, 0.6851, 0.6851) |
<!-- /worked -->

Twelve pixels are enough to show the rule. Pixel 1, of class 17, one of the classes that are kept, becomes a gray of 0.0826, and pixel 5, of class 14 (the neck), is whitened. The class is what decides, not the colour: pixel 5 is as skin-coloured as pixel 2, of class 1, and is treated as background. The mask needs no softmax, because it reads only the argmax, and for the same reason it is not a visualization: the module remarks that it is never the colour-coded class map. The chain in `Bundle::face_embeds` of `crates/pulid/src/caps.rs` is the lower branch of the figure: the landmarks that the recognition path has already computed are aligned a second time to the 512 template, parsed, masked, and the masked crop is resized for an image tower.

## Three crates, one device, and the stories in their history

The three crates were once one. Commit `d3ec333c7` split a single crate that held both models (it was called facenet then) into `crates/scrfd` and `crates/arcface`, because the two are independent architectures with their own weights file, normalization and served model identifier, and it left one real coupling, which the type system now shows: the embedding action, with alignment on, has to find the face first, so `arcface` depends on `scrfd` and re-exports the detector's `Face` so that a consumer needs one dependency. The commit message also records why the two normalization constants are duplicated and not shared: "the whole point of the split is that these are never shared or defaulted".

The coupling has a runtime cost of its own, a consequence of running both models on one device, with different kernel sets, and a kernel is named by its position in the list that was registered, so a model built on the other's handle would bind the wrong pipelines, silently. `crates/arcface/src/caps.rs` therefore builds the detector's session on the same adapter with `Gpu::new_like`, which provides a second kernel set on one device, and a test says that the two lists differ and that they overlap, and that "that overlap is what a positional index would get wrong" (knowledge #13). The serving list of each crate is its inference list with the serving-only kernels appended at the end, never inserted, so that the positions already handed out remain valid; a test pins that too.

The history of the group has three more stories, and each has a general lesson. The first is that of the detect-then-verify gate in `crates/yolov8` (commit `74765b3bb`, knowledge #103). A COCO detector's class head cannot learn one person, for the reason that its features are made to discard differences within a category, so identity is decided by a second stage: the detector finds people, the face pair finds faces, and a cosine against a stored reference embedding labels them. The first plan was to cut each person box out and give the crop to the face detector, and it was measurably wrong. The crop is resized onto the detector's canvas, so a small crop is enlarged, and interpolated detail is not detail. The factors are plain arithmetic: a frame of 512 pixels meets the 640 canvas at 1.25, and crops of 400, 300 and 235 pixels meet it at 1.6, 2.1 and 2.7. The commit reports that a face of 17 by 24 source pixels was found at a score of 0.595 in the full frame and by nothing in the three crops. So the face search runs once over the whole frame at its native scale, and each face is attributed to the person box that contains its centre, which is both more accurate and cheaper by the number of boxes.

The second is the pair of defects of `0337795a8` already told: suppression in the wrong units and the tie that picks the wrong box, invisible to a fixture with a single face. The third is the one of the identity vector's length, from `6a9da997d`: a consumer that normalizes what the reference does not. Each has the same shape. The wrong pipeline is the right shape, the right type and the right range, it runs and returns numbers, and only a numerical comparison against the reference, with a fixture rich enough to contain the case, tells the difference. The first chapter of this volume said that preprocessing is where a model is quietly lost; the three models of this chapter add a second place, after the network, where its answer is cut into boxes and vectors and masks by rules with conventions of their own.

## Check your understanding

1. A fit of five landmarks to the template gives a residual of 1.7 pixels. A colleague proposes to reject any crop whose residual is above 1.0 as "badly aligned". What does the proposal confuse, and what would it do to the three photographs of this chapter?
2. The detector's decode divides the box by `det_scale` after suppression, "to save work: the division is the same for every box". Find the error in the argument, and give the pair of boxes that shows it.
3. A recognition model is evaluated with a threshold of 0.5 and a loose embedding. Reducing the false acceptances is proposed by raising the threshold to 0.6. What happens to the false rejections, and why is changing the loss the better answer?
4. With scale $s = 64$, a thousand identities and all other centres orthogonal, a plain classifier reaches a loss of 0.01 when the sample is 79.6 degrees from its centre. The margin form with $m = 0.5$ needs 51.0. Why is the difference exactly $m$ and not a function of $s$ or the number of identities?
5. The slope of the target logit against the cosine is 1,814 at one degree and the slope against the angle is 31.7. Which of them should a gradient check on the cosine table expect to be large, and why does the loss still train stably?
6. A detector reports the five landmarks in the order right eye, left eye, nose, right corner of the mouth, left corner of the mouth, that is, with the face's own left and right where the template has the picture's. What does `fit_similarity` do with them, and why is that better than what it does with the reverse order?

## Summary

A face comparison starts by taking the camera out. Five landmarks, found by a detector that is an anchor-free detector with an extra output, are fitted to a template by the least-squares similarity of four numbers, in closed form from three sums; the crop is built by looking each output pixel up in the photograph through the inverse of the fit, and the conventions of that lookup, the edge of a pixel and the fill beyond the border, are half a pixel and a grey level of difference that no shape check can see. The detector's output is made of conventions too: the anchor at the cell corner, the learned scale and the stride, the inclusive box that makes overlap depend on units, the division by the canvas scale before suppression, the tie rule, the canvas in the top-left corner, and two constants one letter apart. The embedding network turns the aligned crop into 512 numbers whose cosines are the comparison; the loss that trains it has to remove every freedom but the angle, give the logits a scale large enough to matter, and add a margin to the target's angle so that clusters are demanded to be tight by $m$ radians, and a threshold can then trade the two kinds of error at a meaningful place. A third network labels the pixels of a differently aligned crop, and a mask built from its argmax removes what is not face. The limits are as stated in the text: the pipeline assumes one flat face, the five points are the only thing that passes between the models, and a wrong order of the points, a wrong constant or a wrong template gives a plausible, wrong answer. The next chapter leaves the single image for video and builds a pipeline that detects, segments and tracks objects through frames, with the tools of the whole volume.

## References

- J. Deng, J. Guo, N. Xue and S. Zafeiriou, *ArcFace: Additive Angular Margin Loss for Deep Face Recognition*, 2019.
- J. Guo, J. Deng, A. Lattas and S. Zafeiriou, *Sample and Computation Redistribution for Efficient Face Detection*, 2022.
- C. Yu, J. Wang, C. Peng, C. Gao, G. Yu and N. Sang, *BiSeNet: Bilateral Segmentation Network for Real-time Semantic Segmentation*, 2018.
- S. Umeyama, *Least-Squares Estimation of Transformation Parameters Between Two Point Patterns*, 1991.
- W. Liu, Y. Wen, Z. Yu, M. Li, B. Raj and L. Song, *SphereFace: Deep Hypersphere Embedding for Face Recognition*, 2017.
- H. Wang, Y. Wang, Z. Zhou, X. Ji, D. Gong, J. Zhou, Z. Li and W. Liu, *CosFace: Large Margin Cosine Loss for Deep Face Recognition*, 2018.

## Exercises

1. **[Calculate]** For the landmarks $(0, 0), (2, 0), (1, 2)$ and the template $(10, 10), (14, 10), (12, 14)$, compute the similarity of (2) by hand: the means, $S$, $D$, $C$, $a$, $b$, the shift, and the residual. What are the scale and angle?
2. **[Calculate]** Two boxes $(0, 0, 19, 19)$ and $(10, 0, 29, 19)$ overlap in a band. Compute the IoU with and without the $+1$, then halve both boxes and compute both again. For which thresholds does the original pair give a different verdict under the two conventions, and for which does the scale alone change the verdict of the inclusive convention?
3. **[Calculate]** A picture is 1111 by 500 pixels. Give the resized size, the padding on the 640 canvas, `det_scale`, and the error at the right edge that the scale from the height produces (as in `faces.det_canvas`).
4. **[Calculate]** For $s = 32$, $N = 500$ identities and the other centres orthogonal, compute the loss floor, the largest angle at which the loss is at most 0.05, and the same angle with $m = 0.35$ radian.
5. **[Implement]** Reproduce the table of `faces.similarity_fit_table` from the printed landmarks of `faces.photos` and the template, in a language other than the one the book's module is written in, and check that the three residuals agree to the printed digits.
6. **[Implement]** Reproduce `faces.margin_logits` and `faces.margin_slope`: the five logits for $m = 0, 0.2, 0.5$ and the slope at the five angles, including the $10^{-6}$ floor, and verify the slope by a central difference with $h = 10^{-6}$ against a criterion of 1e-7 relative.
7. **[Implement]** Reproduce `faces.verification_threshold` from the generator named in the block (its seed is printed there), drawing in the stated order, and confirm the counts of wrong rejections and acceptances at the four thresholds for both spreads.
8. **[Investigate]** Using `fit_similarity`, move one landmark of photograph A1 by one pixel at a time, in each direction, and refit to the 112 template. How far do $a$ and $b$ move for each of the five landmarks, and what does the result say about which points a detector has to find most precisely?
9. **[Investigate]** Suppose the decode placed its anchor points at the cell centre $(g_x + 0.5)s$ while the weights were trained with the corner. By how many pixels does every box move, in each axis, at each of the three strides, and for a square box of side $w$ shifted that way in both axes, what is its IoU with the unshifted box? For which side $w$ does the IoU fall to 0.5 at each stride?
10. **[Investigate]** Extend `faces.margin_curve` to $m = 1.0$ and $m = 1.5$. Where does the margin form turn upward, and how large is the step of the clamped variant at the switch? For $m = 1.5$, is a sample with a cosine of 0 to its own centre (90 degrees) on the rising part of the curve?

## Key terms

- **face detection**: finding every face in a photograph as a box with a confidence and five landmark points, so that each face can be aligned to a template; a detector of one class that carries an extra output. <!-- concept: face-detection -->
- **angular margin loss**: a classification loss on cosines in which the embedding and the class centres are normalized, the logits are scaled by a constant, and a fixed margin is added to the angle between the sample and its own centre, so that the sample is asked to be closer to its centre by that angle than the plain classifier requires. <!-- concept: angular-margin-loss -->
- **identity embedding**: a vector computed from a photograph of a person so that photographs of the same person point the same way and photographs of different people do not; two of them are compared by the cosine of the angle between them. <!-- concept: identity-embedding -->
- **face parsing**: semantic segmentation of a face crop into the parts of a head and its surroundings, with a fixed set of classes, one for every pixel. <!-- concept: face-parsing -->

## Answers

**Recall questions.**

1. $\cos = (3 \cdot 4 + 4 \cdot 3)/(5 \cdot 5) = 24/25 = 0.96$. The angle is $\arccos 0.96 = 16.26$ degrees; a cosine of 1 is the same direction, 0 is a right angle and $-1$ opposite directions.
2. $p = e^2/(e^2 + 1) = 0.8808$. The gradient with respect to a logit is the probability minus the one-hot target: $-0.1192$ for the first logit and $+0.1192$ for the second.
3. $x_1 = 44 - 2.0 \cdot 8 = 28$, $y_1 = 28 - 1.5 \cdot 8 = 16$, $x_2 = 44 + 3.0 \cdot 8 = 68$, $y_2 = 28 + 2.5 \cdot 8 = 48$: the box is $(28, 16, 68, 48)$.
4. At one eighth of 512 the map is 64 by 64, so $19 \cdot 64 \cdot 64 = 77{,}824$ scores; after the resize to 512 by 512, $19 \cdot 512 \cdot 512 = 4{,}980{,}736$. A pixel takes the class of its largest score, the lowest class on a tie.

**Check your understanding.**

1. The residual measures how far the five points are from the template after the best similarity, which includes the real differences between a person's face and the template and the detector's jitter; it is not a measure of alignment error. The three photographs have residuals of 1.73, 1.75 and 1.93, so all three would be rejected, though every one of them is aligned as well as five points allow.
2. A suppression threshold on an overlap that carries the $+1$ is not scale-invariant: $0.3333$ on the canvas is $0.375$ at half the size. With the pair $(0, 0, 9, 9)$ and $(5, 0, 14, 9)$ and the threshold 0.35, suppressing first keeps two boxes and scaling first keeps one.
3. Raising the threshold to 0.6 removes some acceptances and adds rejections of genuine pairs, since the genuine cosines of a loose embedding are spread widely (in the simulated example, 107 of 200 genuine pairs are already rejected at 0.5). The overlap of the two distributions is in the embedding, so a threshold only moves along the trade; a loss that makes genuine pairs tighter moves the trade itself.
4. The loss of at most $\varepsilon$ depends on the target logit only through $s\cos(\vartheta_y + m)$, so the condition is $\cos(\vartheta_y + m) \ge c^\ast$ with $c^\ast$ a function of $s$, $N$ and $\varepsilon$; replacing $\vartheta_y$ by $\vartheta_y + m$ shifts the allowed angle by exactly $m$ whatever $c^\ast$ is.
5. The slope against the cosine is the large one: it grows as $1/\sin\vartheta$ near the centre. The cosine itself moves with the angle at the rate $\sin\vartheta$, so the slope against the angle, which is the product, stays near $s\sin m = 30.7$ and the gradient that reaches the weights stays bounded. Training is stable for that reason, and the floor of $10^{-6}$ keeps the arithmetic finite where $\sin\vartheta$ itself would round to zero.
6. The labelled points are the mirror image of the template in $x$, the best fit would need a reflection, the determinant of the cross-covariance is negative and the routine refuses, which is the "eyes and mouth corners swapped" row of the table. The reverse order, which is a half turn, is fitted with a residual of 3.33 pixels and an upside-down crop, and nothing in the routine objects: a refusal is better than a plausible wrong answer.

**Exercises.**

1. Means: landmarks $(1, \tfrac23)$, template $(12, 11\tfrac13)$. Offsets of the landmarks: $(-1, -\tfrac23)$, $(1, -\tfrac23)$, $(0, \tfrac43)$; of the template: $(-2, -\tfrac43)$, $(2, -\tfrac43)$, $(0, \tfrac83)$. $S = 2 + \tfrac89 + \tfrac{16}{9} = \tfrac{14}{3}$; $D = (2 + \tfrac89) + (2 + \tfrac89) + \tfrac{32}{9} = \tfrac{28}{3}$; $C = (\tfrac43 - \tfrac43) + (-\tfrac43 + \tfrac43) + 0 = 0$. So $a = 2$, $b = 0$, the scale is 2, the angle 0, the shift $\bar{\mathbf{d}} - 2\bar{\mathbf{l}} = (10, 10)$ and the residual 0: the template is the landmarks doubled and moved.
2. Half-open boxes: widths 19, overlap $9 \times 19 = 171$, union $361 + 361 - 171 = 551$, IoU $0.3103$. Inclusive: sides 20, overlap $10 \times 20 = 200$, union $400 + 400 - 200 = 600$, IoU $0.3333$. Halved, the half-open value is still $0.3103$ and the inclusive one is $57.75/162.75 = 0.3548$ (sides 10.5, overlap $5.5 \times 10.5$). The conventions give different verdicts for thresholds between $0.3103$ and $0.3333$; scaling alone changes the inclusive verdict for thresholds from $0.3333$ up to $0.3548$.
3. The ratio is $500/1111 = 0.45005$, so the picture becomes $640 \times 288$ ($\lfloor 640 \cdot 0.45005 \rfloor = 288$), the padding is 0 on the right and 352 at the bottom, `det_scale` $= 288/500 = 0.576$, the horizontal scale is $640/1111 = 0.576058$, and the canvas column 640 goes back to $640/0.576 = 1111.11$ instead of $1111$: an error of 0.11 pixel.
4. The floor is $\ln(1 + 499\,e^{-32}) = 6.3 \times 10^{-12}$. The condition is $c = \ln(499/(e^{0.05} - 1))/32 = 0.2870$, so the largest angle is 73.3 degrees; $m = 0.35$ radian is 20.05 degrees, so with the margin it is 53.3 degrees.
5. Implement: compare with the table of the block; the three residuals are 1.726, 1.751 and 1.933.
6. Implement: compare with the two blocks; the relative difference between the formula and the central difference should be below 1e-7.
7. Implement: the seed and the draw order are printed in the block, and the counts are in its second table (for example 107 of 200 genuine pairs rejected at 0.5 with the loose spread).
8. Moving landmark $i$ by $\boldsymbol{\delta}$ changes $S$, $D$ and $C$ of (2) by amounts proportional to the offset of that landmark from the centroid and to the offset of its template partner, and $a$ and $b$ by those divided by $S$. The effect of an error in one point therefore grows with its distance from the centroid: the two mouth corners and the eyes matter more than the nose, which sits near the middle, for the scale and the angle.
9. The shift is half a cell, $0.5\,s$ in each axis: 4, 8 and 16 pixels at strides 8, 16 and 32. A square of side $w$ shifted by $\delta$ in both axes keeps the IoU $(w - \delta)^2/(2w^2 - (w - \delta)^2)$ with itself. It falls to 0.5 when $(w - \delta)/w = \sqrt{2/3}$, that is $w = \delta/0.1835$: 21.8, 43.6 and 87.2 pixels at the three strides.
10. The margin form turns upward at $\vartheta = \pi - m$: 122.70 degrees for $m = 1.0$ and 94.06 degrees for $m = 1.5$. The clamped variant steps down by $1 - \cos m - m\sin m$, 0.3818 for $m = 1.0$ and 0.5670 for $m = 1.5$. With $m = 1.5$ a sample at 90 degrees is just before the turn, at $\vartheta + m = 3.07$ radians, still on the falling part.
