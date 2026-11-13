<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Project: A Vision Pipeline

*Can you build a pipeline that detects, segments and tracks objects through video frames?*

Frame 7 of a short synthetic clip shows two objects, and the right answer to "where is the red rectangle A?" is one that a careless program does not give. A is there, under the green rectangle B, which began to slide over it a frame ago and will begin to uncover it two frames from now. In frame 5 A showed 80 pixels of red; in frame 7 it shows none. A tracker that must answer anyway draws a mask on the green. A tracker that has learned to say "absent", and says it for frames 7 and 8 and not for any other frame, has done half of the job. The other half comes at frame 9, when a strip of red seven columns wide appears at the right edge of B. A second red rectangle, C, with the same colour, the same size and the same class label, sits twenty rows lower. The strip has to be handed to A and not to C, by a program that was told about A on frame 0 and was never told anything again.

Nothing in that paragraph is hard for a person, and every part of it is a place where a pipeline built from correct pieces fails. The detector of [Chapter 4](04-detection.md) and the segmenter and tracker of [Chapter 7](07-segment-anything.md) each read the same frame, and they do not read it the same way: one wants a grey-padded square of bytes divided by 255, the other a stretched square normalized by the ImageNet constants. A box leaves the first in one coordinate system and must arrive in the second in another. The tracker's memory has a fixed length, and an object that is hidden for longer than the memory is not a tracking problem any more. The pieces were each verified in their own chapter; what has not been verified is that they compose. That is what this project asks you to do, in a language of your choice, in five milestones:

1. **A clip with an answer key, and two views of every frame.** A deterministic video of three rectangles on a noisy background with the exact visible mask and box of every object in every frame, the letterboxed view the detector reads and the stretched, normalized view the segmenter reads (the shared preprocessing of [Chapter 1](01-images.md)), and the maps between the three coordinate systems.
2. **A detector tail.** A head that emits one box and score per cell, a confidence filter, class-aware non-maximum suppression (Chapter 4) and the inverse letterbox, so that boxes arrive in frame pixels.
3. **A segmenter that takes a box.** The box prompt of Chapter 7 turned into a mask, with the per-axis scale that separates it from the detector's map.
4. **A tracker with a bank.** The memory slots and memory attention of Chapter 7 over the cells of earlier frames, an object score that can say "absent", and a memory that records the absence.
5. **The whole run, its planted mistakes and its cost.** Every stage chained, six mistakes planted one at a time and each caught by a number, and a count of what the real models would cost for the same schedule.

A word about what is real in this list. The pipeline's interfaces are the real ones: the boxes in frame pixels, the two views, the prompt tokens, the bank with its slots, the object score and the rule that an absent object writes an empty memory. The three networks are not. A trained detector, a trained segmenter and a trained tracker cannot be specified on a page, and a reader who must bring their weights cannot reproduce a number. Each is therefore replaced by a **stand-in**: a small deterministic program with the network's inputs and outputs, written out completely, whose job is to make every seam of the pipeline testable. The stand-ins read the pixels, so a preprocessing mistake changes their output, which is why they are stand-ins and not recordings. Where a stand-in does what a network learns to do, that is said where it happens, and the real models in brain that would take each place are named at the end.

![Data flow of the pipeline: the frame feeds a detector row (letterbox, head, filter and suppression, inverse letterbox) and a segmenter row (stretch and normalize, segmenter with a box prompt, tracker with a bank of 7 slots), and an arrow carries the boxes of frame 0 from the first row to the segmenter.](img/project-vision-pipeline.svg)

Building along the arrows is a sound order. The two preprocessing boxes come first because every later stage reads their output, the detector tail ends in the boxes that the segmenter takes as prompts, and the tracker needs the segmenter's mask of frame 0 as its first memory. Build and check each milestone before the next: a mask that is wrong because the box arrived in the wrong frame is found in a minute at milestone 3 and in an afternoon at milestone 4. The **acceptance tests** at the end gather every expected value. You need the book's generator ([Volume I, Chapter 17](../v01-mathematics/17-randomness.md)), double-precision arithmetic with a correctly rounded `exp` and `sqrt`, and the bilinear resize of Chapter 1.

The milestones lean on five ideas from earlier in the volume, as far back as Chapter 1; bring each back before building on it.

1. ([Chapter 1](01-images.md), the letterbox and shared preprocessing.) A frame of 96 by 61 pixels is fitted to a square of side 64 with its aspect ratio kept. What are the scale, the size of the content and the padding on each axis?
2. ([Chapter 4](04-detection.md), suppression.) Two boxes of one class, $(0, 0, 10, 10)$ and $(0, 0, 10, 4.5)$, are candidates and the threshold is 0.45. Is the second dropped? Say why the comparison is strict.
3. ([Chapter 4](04-detection.md), distances from a point.) A head at the anchor point $(8.5, 28.5)$ predicts the distances $(3.5, 2.5, 3.5, 2.5)$ to the left, top, right and bottom edges. Which box is that?
4. ([Chapter 7](07-segment-anything.md), prompts.) How many prompt tokens does a box give to the mask decoder, and what labels do they carry?
5. ([Chapter 7](07-segment-anything.md), the bank.) A clip is prompted on frame 0. Which frames' slabs does frame 9 read, in which slots, with 7 slots?

## Building the pipeline in five stages

The pseudocode follows [Pseudocode Conventions](../pseudocode.md); it fixes interfaces and the order of operations, and it reuses by name the routines of earlier chapters (`resize_axis` of Chapter 1, `iou`, `nms` and `unletterbox` of Chapter 4, `grow` and `memory_slots` of Chapter 7). The only random input of the project is the clip, drawn from the book's SplitMix64 at a seed derived by the rule `seed_of(name)`. Nothing is trained, so every printed number is exact arithmetic on stated inputs. Sums run left to right with one accumulator, in the order the text states; a different order moves the last digits and nothing a table prints, and the one value that could flip under rounding, a read-back near the threshold 0.5, is at least 0.0109 away from it (milestone 4).

### Milestone 1: a clip, and two views of every frame

The clip is twelve frames of 96 by 61 pixels, 8-bit RGB. Three solid rectangles move over a flat background. A and C share the colour (200, 60, 50) and class 0; B is a larger green rectangle, colour (60, 180, 90) and class 1, painted last, so that where B covers A or C the green wins. A starts at the left and moves right 4 pixels a frame, C starts at the right and moves left 4 pixels a frame 20 rows lower, and B moves left 5 pixels a frame across A's row. The geometry was chosen so that B fully hides A on frames 7 and 8, partly hides it on frames 6 and 9, never touches C, and so that A and C pass each other in $x$ while staying in different rows. Every sample gets a small integer added to its base colour, drawn in a fixed order, so that a pixel of a rectangle is not exactly its colour and a segmenter has to tolerate noise. The truth for each object in each frame is its **visible mask**, the pixels where it is the topmost rectangle, and its visible box, the bounding box of those pixels (a box covers $x_1 \le x < x_2$, $y_1 \le y < y_2$, as in Chapter 4). A detector reports what it sees, so the visible box and not the full rectangle is the answer key.

<!-- worked: project_vision_pipeline.clip_table -->
seed 0xd41c20e97ed39c60 (15284127422615886944), generator `project_vision_pipeline.scene`. The clip has 12 frames of 96 by 61 pixels. Each object is a solid rectangle painted over a background of (40, 44, 48); objects are painted in the order A, C, B, so B covers A and C where they meet. Every one of the 3 * 61 * 96 samples of a frame, in the order frame, row, column, channel, gets one `integer(4)` added to the base colour of its pixel. Rectangles: A is 10 by 8 at y = 22 with x = 8 + 4t, colour (200, 60, 50), class 0; C is 10 by 8 at y = 42 with x = 76 - 4t, the same colour, class 0; B is 22 by 16 at y = 14 with x = 70 - 5t, colour (60, 180, 90), class 1. Each cell shows the number of visible pixels and the box (x1, y1, x2, y2) of the visible pixels. The last column is the first 12 hex digits of the SHA-256 of the frame's 17,568 bytes.

| t | A (class 0) | C (class 0) | B (class 1) | SHA-256 of the frame |
|---:|---|---|---|---:|
| 0 | 80 / (8, 22, 18, 30) | 80 / (76, 42, 86, 50) | 352 / (70, 14, 92, 30) | 5206fd6a9bf2 |
| 1 | 80 / (12, 22, 22, 30) | 80 / (72, 42, 82, 50) | 352 / (65, 14, 87, 30) | 0f8409eca546 |
| 2 | 80 / (16, 22, 26, 30) | 80 / (68, 42, 78, 50) | 352 / (60, 14, 82, 30) | 5616aefa7f44 |
| 3 | 80 / (20, 22, 30, 30) | 80 / (64, 42, 74, 50) | 352 / (55, 14, 77, 30) | d33e40e59968 |
| 4 | 80 / (24, 22, 34, 30) | 80 / (60, 42, 70, 50) | 352 / (50, 14, 72, 30) | 87a622a5a528 |
| 5 | 80 / (28, 22, 38, 30) | 80 / (56, 42, 66, 50) | 352 / (45, 14, 67, 30) | f0278d11fe83 |
| 6 | 64 / (32, 22, 40, 30) | 80 / (52, 42, 62, 50) | 352 / (40, 14, 62, 30) | 44d0f8774161 |
| 7 | 0 / hidden | 80 / (48, 42, 58, 50) | 352 / (35, 14, 57, 30) | 2f3971e198d7 |
| 8 | 0 / hidden | 80 / (44, 42, 54, 50) | 352 / (30, 14, 52, 30) | 4c9e708014b5 |
| 9 | 56 / (47, 22, 54, 30) | 80 / (40, 42, 50, 50) | 352 / (25, 14, 47, 30) | c8f42a9aedcf |
| 10 | 80 / (48, 22, 58, 30) | 80 / (36, 42, 46, 50) | 352 / (20, 14, 42, 30) | b4ec1852cb5e |
| 11 | 80 / (52, 22, 62, 30) | 80 / (32, 42, 42, 50) | 352 / (15, 14, 37, 30) | 1d0bc3156b7d |
<!-- /worked -->

The digest column is the first 12 hexadecimal digits of the SHA-256 of the frame's bytes, which is how a clip too large to print is compared: if your digest of frame 0 differs, your generator, your draw order or your painting order differs, and nothing after this point is worth debugging. Check it first.

The detector and the segmenter read a frame through different preprocessing, and it is worth looking at why, because brain keeps them different on purpose. A detector trained on letterboxed squares has learned that the grey border is not an object, and it expects values in $[0, 1]$ with no normalization. A promptable segmenter was trained on pictures resized to a square without keeping the aspect ratio and normalized by the ImageNet constants. Neither is more right than the other, and both are facts about the weights ([Chapter 1](01-images.md) calls it "one definition of each step": the operations are shared, the pipelines are not). Write two functions:

```text
function detector_view(frame: array<int>[H, W, 3], S: int, fill: real) -> array<real>[3, S, S]:
    s = min(S / W, S / H)
    new_w = floor(W * s + 0.5)
    new_h = floor(H * s + 0.5)
    pad_x = (S - new_w) // 2
    pad_y = (S - new_h) // 2
    out = zeros(real, [3, S, S])
    for c in 0 .. 3:
        for y in 0 .. S:
            for x in 0 .. S:
                out[c, y, x] = fill
    for yo in 0 .. new_h:
        ys = min((2 * yo + 1) * H // (2 * new_h), H - 1)
        for xo in 0 .. new_w:
            xs = min((2 * xo + 1) * W // (2 * new_w), W - 1)
            for c in 0 .. 3:
                out[c, pad_y + yo, pad_x + xo] = frame[ys, xs, c] / 255.0
    return out

function segmenter_view(frame: array<int>[H, W, 3], S: int, mean: array<real>[3], std: array<real>[3]) -> array<real>[3, S, S]:
    out = zeros(real, [3, S, S])
    for c in 0 .. 3:
        plane = zeros(real, [H, W])
        for y in 0 .. H:
            for x in 0 .. W:
                plane[y, x] = frame[y, x, c] / 255.0
        rows = zeros(real, [H, S])
        for y in 0 .. H:
            rows[y, :] = resize_axis(plane[y, :], S, "half")
        for x in 0 .. S:
            out[c, :, x] = (resize_axis(rows[:, x], S, "half") - mean[c]) / std[c]
    return out
```

The source index `(2 * o + 1) * n_in // (2 * n_out)` is the nearest rule of Chapter 1, round half up of $(o + 0.5)\, n_{\mathrm{in}}/n_{\mathrm{out}} - 0.5$ written in integers, with $n_{\mathrm{in}}$ the frame's extent on the axis and $n_{\mathrm{out}}$ the content's (the rows use 61 and 41, and not the scale $2/3$, because the content was rounded). The padding is $\lfloor (S - \text{content})/2 \rfloor$ on each axis, and it is used both to paste the content and, later, to map boxes. That is the convention of brain's own letterbox too (see the last section): a frame and its own coordinates should not disagree, and milestone 5 measures what the other choice costs. The fill is a parameter, 114/255, never a constant buried in the function.

<!-- worked: project_vision_pipeline.views_table -->
Frame 0 of the clip, read two ways. The detector's view: scale = min(64/96, 64/61); content size round half up of (96 * scale, 61 * scale); padding floor((64 - content) / 2) per axis; content pixel (yo, xo) copies the frame pixel (floor((2 yo + 1) 61 / (2 41)), floor((2 xo + 1) 96 / (2 64))) (the half-pixel nearest rule in integers: floor((2 o + 1) n_in / (2 n_out)) with the frame's extent on the axis and the content's); values byte / 255, padding 114 / 255, no normalization. The segmenter's view: each channel's byte / 255 plane resized to 64 by 64 by the half-pixel bilinear rule of Chapter 1, then (x - mean) / std with mean (0.485, 0.456, 0.406) and std (0.229, 0.224, 0.225). Sums run over channel, row, column with one accumulator.

| quantity | value | view |
|---|---|---:|
| scale min(64/96, 64/61) | 0.666667 | detector view |
| content size (round half up) | 64 x 41 | detector view |
| padding (pad_x, pad_y) = floor(slack / 2) | (0, 11) | detector view |
| padding rows | 0 to 10 and 52 to 63 (23 rows) | detector view |
| source rows of the first 6 content rows | [0, 2, 3, 5, 6, 8] | detector view |
| source columns of the first 8 content columns | [0, 2, 3, 5, 6, 8, 9, 11] | detector view |
| SHA-256 (first 12 hex digits) of the 12,288 bytes round half up (255 v), serialized in H W C order: for each row, each column, the bytes r, g, b | 257286e7bda2 | detector view |
| sum of the 12,288 values (tolerance 1e-9) | 3547.631373 | detector view |
| scale per axis (64/96, 64/61) | (0.666667, 1.049180) | segmenter view |
| sum of the 12,288 normalized values (tolerance 1e-9) | -13516.819217 | segmenter view |
| normalized value, channel 0, row 0, column 0 | -1.424351 | segmenter view |
| normalized value, channel 2, row 31, column 40 | -0.947929 | segmenter view |
<!-- /worked -->

The frame is 96 by 61, so the letterbox scale is exactly $2/3$, the content is 64 by 41 and the 23 padding rows split 11 above (the floor of half the odd slack) and 12 below, for the pixels and for the boxes alike. The segmenter's two scales are not equal, $64/96$ and $64/61$, and the next table shows what that means: a box in frame pixels lands in different places in the two inputs, and a program that carries a box from one to the other by the wrong map is wrong by an amount that depends on the box.

<!-- worked: project_vision_pipeline.box_maps_table -->
The visible box of each object on frame 0 (frame pixels, x1 <= x < x2, y1 <= y < y2) carried into the detector's input (x s + pad_x, y s + pad_y with the integer padding) and into the segmenter's input (x 64/96, y 64/61); the last column maps the detector's box back by the inverse letterbox of Chapter 4.

| object | box in the frame | in the detector's input | in the segmenter's input | detector input mapped back |
|---|---|---|---|---:|
| A | (8, 22, 18, 30) | (5.33, 25.67, 12.00, 31.00) | (5.33, 23.08, 12.00, 31.48) | (8.00, 22.00, 18.00, 30.00) |
| C | (76, 42, 86, 50) | (50.67, 39.00, 57.33, 44.33) | (50.67, 44.07, 57.33, 52.46) | (76.00, 42.00, 86.00, 50.00) |
| B | (70, 14, 92, 30) | (46.67, 20.33, 61.33, 31.00) | (46.67, 14.69, 61.33, 31.48) | (70.00, 14.00, 92.00, 30.00) |
<!-- /worked -->

The last column checks the round trip of the detector's map, and it is exact because the padding is the same on the way in and out. The two maps into the models differ in $y$ by a lot: A's top edge is at 25.67 in the detector's input and at 23.08 in the segmenter's, and C, lower in the frame, is off by more than five rows. Milestone 3 shows what that does to a mask.

The check for this milestone is exact: the digests of the frames, the visible pixel counts of the table, the view geometry and the two sums, as in the acceptance tests. Build the clip and the views before anything else, and compare.

### Milestone 2: a detector tail that returns boxes in frame pixels

The stand-in detector has the interface of an anchor-free head ([Chapter 4](04-detection.md)): a grid of cells over the input, and for each cell a box given as four distances from a point, and a score. The grid has stride 2, so a $64 \times 64$ input has $32 \times 32 = 1{,}024$ cells; the anchor of cell $(i, j)$ is the pixel $(a_x, a_y) = (2j, 2i)$, whose centre is the point $(a_x + 0.5, a_y + 0.5)$. That is a pixel centre and not the cell centre of Chapter 4, so that every walk below stays on whole pixels. The stand-in has to read pixels, so it works in four steps.

First, the **class map**: every pixel of the detector's input gets the class of the nearest of four prototype colours in squared Euclidean distance on values in $[0, 1]$, a tie going to the lower class. The prototypes are the base colours plus 1.5 (the mean of the added integer), divided by 255: class 0 is the background (40, 44, 48), class 1 the red of A and C, class 2 the green of B, and class 3 the padding grey, 114/255 in each channel. Classes 0 and 3 are not objects. The fourth prototype is not a detail: without it the padding grey is nearer to the green than to the background, and a detector fed a letterbox returns a box over the border. That is the padding being part of the input distribution, as Chapter 1 said, made visible.

Second, the distances. From the anchor pixel, walk left while the class stays the same, then right, up and down, and the four runs give the exact distances $l, t, r, b$ from the anchor point to the edges of the object along its row and column. A real head regresses these with error, and the error is part of what suppression exists for, so the stand-in adds a prescribed one: each distance is multiplied by $1 + 0.06\,\sigma_s$ with $\sigma_s = +1$ if $i + j + s$ is even and $-1$ otherwise, where $s = 0, 1, 2, 3$ indexes left, top, right, bottom. Neighbouring cells then disagree, in a pattern that any language reproduces exactly. The box is the anchor point minus $(l, t)$ and plus $(r, b)$.

Third, the score: the fraction of the $3 \times 3$ pixels around the anchor (clipped at the border, so as many as lie inside) that have the anchor's class, times a centerness of the predicted distances, $\sqrt{\min(l, r)/\max(l, r) \cdot \min(t, b)/\max(t, b)}$ (the measure of the FCOS (fully convolutional one-stage) detector, which keeps a cell far from the middle of an object from outscoring the cells in the middle). The class of the candidate is the class of the anchor pixel minus 1.

```text
function head_cells(view: array<real>[3, S, S], protos: array<real>[4, 3], stride: int, rel_err: real)
        -> list<(int, array<real>[4], real)>:
    cm = zeros(int, [S, S])
    for y in 0 .. S:
        for x in 0 .. S:
            best = 0
            best_d = sum((view[:, y, x] - protos[0, :])^2)
            for k in 1 .. 4:
                d = sum((view[:, y, x] - protos[k, :])^2)
                if d < best_d:
                    best = k
                    best_d = d
            cm[y, x] = best
    out = []
    for i in 0 .. S // stride:
        for j in 0 .. S // stride:
            ay = stride * i
            ax = stride * j
            k = cm[ay, ax]
            if k != 0 and k != 3:
                x1 = ax
                while x1 > 0 and cm[ay, x1 - 1] == k:
                    x1 = x1 - 1
                x2 = ax + 1
                while x2 < S and cm[ay, x2] == k:
                    x2 = x2 + 1
                y1 = ay
                while y1 > 0 and cm[y1 - 1, ax] == k:
                    y1 = y1 - 1
                y2 = ay + 1
                while y2 < S and cm[y2, ax] == k:
                    y2 = y2 + 1
                d = [ax + 0.5 - x1, ay + 0.5 - y1, x2 - (ax + 0.5), y2 - (ay + 0.5)]
                for s in 0 .. 4:
                    d[s] = d[s] * (1.0 + rel_err * (1 if (i + j + s) mod 2 == 0 else -1))
                box = [ax + 0.5 - d[0], ay + 0.5 - d[1], ax + 0.5 + d[2], ay + 0.5 + d[3]]
                votes = 0
                total = 0
                for yy in max(ay - 1, 0) .. min(ay + 2, S):
                    for xx in max(ax - 1, 0) .. min(ax + 2, S):
                        total = total + 1
                        if cm[yy, xx] == k:
                            votes = votes + 1
                cen = sqrt(min(d[0], d[2]) / max(d[0], d[2]) * min(d[1], d[3]) / max(d[1], d[3]))
                append (k - 1, box, votes / total * cen) to out
    return out
```

Here are the nine cells that touch the 10 by 8 object A on frame 0, with every intermediate value:

<!-- worked: project_vision_pipeline.head_example -->
The head's output for the nine cells of grid row 13 to 15 and column 3 to 5, whose anchor pixels (2j, 2i) lie on or at the edge of object A on frame 0. True l, t, r, b are the distances from the anchor point (2j + 0.5, 2i + 0.5) to the ends of the run of pixels of A's class along the anchor's row and column. Predicted distances multiply each by 1 + 0.06 sign, with sign +1 when i + j + s is even for the side index s = 0, 1, 2, 3 (l, t, r, b) and -1 otherwise. Box: the anchor point minus (l, t) and plus (r, b), in the detector's input. Votes: pixels of A's class among the 3 by 3 pixels around the anchor pixel. Centerness: sqrt(min(l, r) / max(l, r) * min(t, b) / max(t, b)) of the predicted distances. Score: votes / 9 * centerness.

| cell (i, j) | true l, t, r, b | predicted l, t, r, b | box (x1, y1, x2, y2) | votes | centerness | score | kept at 0.25 |
|---|---|---|---|---|---:|---:|---:|
| (13, 3) | (1.5, 0.5, 5.5, 4.5) | (1.590, 0.470, 5.830, 4.230) | (4.910, 26.030, 12.330, 30.730) | 6 of 9 | 0.174 | 0.116 | no |
| (13, 4) | (3.5, 0.5, 3.5, 4.5) | (3.290, 0.530, 3.290, 4.770) | (5.210, 25.970, 11.790, 31.270) | 6 of 9 | 0.333 | 0.222 | no |
| (13, 5) | (5.5, 0.5, 1.5, 4.5) | (5.830, 0.470, 1.590, 4.230) | (4.670, 26.030, 12.090, 30.730) | 6 of 9 | 0.174 | 0.116 | no |
| (14, 3) | (1.5, 2.5, 5.5, 2.5) | (1.410, 2.650, 5.170, 2.650) | (5.090, 25.850, 11.670, 31.150) | 9 of 9 | 0.522 | 0.522 | yes |
| (14, 4) | (3.5, 2.5, 3.5, 2.5) | (3.710, 2.350, 3.710, 2.350) | (4.790, 26.150, 12.210, 30.850) | 9 of 9 | 1.000 | 1.000 | yes |
| (14, 5) | (5.5, 2.5, 1.5, 2.5) | (5.170, 2.650, 1.410, 2.650) | (5.330, 25.850, 11.910, 31.150) | 9 of 9 | 0.522 | 0.522 | yes |
| (15, 3) | (1.5, 4.5, 5.5, 0.5) | (1.590, 4.230, 5.830, 0.470) | (4.910, 26.270, 12.330, 30.970) | 6 of 9 | 0.174 | 0.116 | no |
| (15, 4) | (3.5, 4.5, 3.5, 0.5) | (3.290, 4.770, 3.290, 0.530) | (5.210, 25.730, 11.790, 31.030) | 6 of 9 | 0.333 | 0.222 | no |
| (15, 5) | (5.5, 4.5, 1.5, 0.5) | (5.830, 4.230, 1.590, 0.470) | (4.670, 26.270, 12.090, 30.970) | 6 of 9 | 0.174 | 0.116 | no |
<!-- /worked -->

The cell in the middle of A scores 1.000 and its neighbours 0.522 and 0.222, and the corner cells 0.116, so that only three of the nine pass a confidence floor of 0.25. The rest of the tail is Chapter 4's, with the numbers brain's own detector pipeline ships with: scores below the confidence floor 0.25 are dropped, the survivors are suppressed class by class with a strict IoU threshold of 0.45 (`DetectOptions` in `crates/sdk/src/detect.rs` has exactly these two defaults) and at most 300 are kept, and the kept boxes go back through `unletterbox` with the padding of milestone 1, clamped to the frame once, last.

```text
function detect_frame(frame: array<int>[H, W, 3], protos: array<real>[4, 3], conf: real, thr: real)
        -> list<(array<real>[4], real, int)>:
    s = min(64.0 / W, 64.0 / H)
    pad_x = (64 - floor(W * s + 0.5)) // 2
    pad_y = (64 - floor(H * s + 0.5)) // 2
    view = detector_view(frame, 64, 114.0 / 255.0)
    cells = head_cells(view, protos, 2, 0.06)
    keep = []
    for n in 0 .. len(cells):
        if cells[n][2] >= conf:
            append n to keep
    boxes = zeros(real, [len(keep), 4])
    scores = zeros(real, [len(keep)])
    classes = zeros(int, [len(keep)])
    for m in 0 .. len(keep):
        boxes[m, :] = cells[keep[m]][1]
        scores[m] = cells[keep[m]][2]
        classes[m] = cells[keep[m]][0]
    kept = nms(boxes, scores, classes, thr, 300, true)
    out = []
    for m in kept:
        append (unletterbox(boxes[m, :], s, pad_x, pad_y, W, H), scores[m], classes[m]) to out
    return out
```

Here `protos`, a parameter, is the $4 \times 3$ array of the prototype colours of the class map. On frame 0, the 60 candidates, 29 of which pass the floor, collapse to three detections, one per object, after 30 IoU evaluations (18 operations each, as Chapter 4 counted, so 540 operations: the tail is not where this pipeline spends its work). Run on every frame, with the matching rule of Chapter 4's evaluation:

<!-- worked: project_vision_pipeline.detector_table -->
The detector of milestone 2 on every frame: confidence threshold 0.25, class-aware suppression at IoU > 0.45, at most 300 detections. A detection is matched to the unmatched visible object of its class with the largest IoU, at least 0.5; detections are visited by falling score. Raw candidates are the cells whose anchor is not background or padding.

| t | raw candidates | score >= 0.25 | kept by NMS | matched | false positives | missed | mean IoU of matches |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 60 | 29 | 3 | 3 | 0 | - | 0.823 |
| 1 | 66 | 30 | 3 | 3 | 0 | - | 0.860 |
| 2 | 66 | 28 | 3 | 3 | 0 | - | 0.827 |
| 3 | 60 | 29 | 3 | 3 | 0 | - | 0.862 |
| 4 | 66 | 30 | 3 | 3 | 0 | - | 0.861 |
| 5 | 66 | 28 | 3 | 3 | 0 | - | 0.828 |
| 6 | 60 | 28 | 3 | 3 | 0 | - | 0.807 |
| 7 | 54 | 28 | 2 | 2 | 0 | - | 0.835 |
| 8 | 57 | 26 | 2 | 2 | 0 | - | 0.842 |
| 9 | 57 | 28 | 3 | 3 | 0 | - | 0.858 |
| 10 | 66 | 30 | 3 | 3 | 0 | - | 0.861 |
| 11 | 66 | 28 | 3 | 3 | 0 | - | 0.828 |
<!-- /worked -->

All 34 visible objects of the clip are found, none twice and none wrongly, and on the two frames where A is hidden the detector returns two boxes, which is the right answer and also the thing the tracker will be asked to agree with. The mean IoU of 0.841 over all 34 matches (the acceptance table repeats it) is a property of the stand-in's prescribed error and of the nearest-neighbour resize, which loses up to a frame pixel at each edge; it is not a number to improve, it is a number to reproduce.

The threshold of the suppression is a choice with a range, and the stand-in lets you see the range:

<!-- worked: project_vision_pipeline.nms_sweep -->
Class-aware suppression of the candidates of frame 0 whose score is at least 0.25, at several thresholds (a candidate is dropped when its IoU with a kept box of its class is strictly greater than the threshold). IoU evaluations are the pairs compared, stopping at the first drop.

| threshold | candidates entering | kept | IoU evaluations |
|---|---:|---:|---:|
| 0.3 | 29 | 3 | 30 |
| 0.45 | 29 | 3 | 30 |
| 0.7 | 29 | 3 | 30 |
| 0.8 | 29 | 6 | 46 |
| 0.9 | 29 | 6 | 46 |
| 0.99 | 29 | 29 | 252 |
<!-- /worked -->

Any threshold up to 0.7 leaves one box per object. At 0.8 some candidates of a single object are far enough from the kept box (the error pattern makes some neighbours disagree) that their IoU with it is not above the threshold, so they survive, and at 0.99 nothing is suppressed at all. Three of the six boxes at 0.8 are false positives, and every false positive becomes a track in milestone 4: the cost of a loose threshold is paid downstream, in memory attention that runs three more times, as the ledger of milestone 5 prices.

Check: your counts of candidates, of survivors and of IoU evaluations on frame 0, the 34 of 34, and the kept boxes of the acceptance table (to 0.01 in position).

### Milestone 3: a segmenter that takes a box

A promptable segmenter takes an image and a prompt and returns a mask ([Chapter 7](07-segment-anything.md)); a box is two prompt points with the labels 2 and 3 for its corners. The stand-in is the toy of that chapter's opening, region growing from a click, with the box doing two jobs: it places the click and it bounds the growth.

The box, in frame pixels, is carried into the segmenter's input by the per-axis scale, $(x\,64/96,\ y\,64/61)$. The seed is the pixel containing the box's centre. The mask is the connected set of pixels, through 4-neighbours, whose centre lies inside the box grown by 2 input pixels and whose colour, as a vector of the three normalized values, is within 1.0 (Euclidean) of the colour at the seed. The set is the fixed point of the scan of Chapter 7's `grow`, so the order in which pixels are visited does not matter. The mask is returned to the frame by the nearest rule: frame pixel $(y, x)$ takes input pixel $(\lfloor (y + 0.5)\,64/61 \rfloor,\ \lfloor (x + 0.5)\,64/96 \rfloor)$, each clamped to 63.

```text
function admits(view: array<real>[3, S, S], b: array<real>[4], ref: array<real>[3], tol: real, margin: real, y: int, x: int) -> bool:
    inside = b[0] - margin <= x + 0.5 and x + 0.5 <= b[2] + margin and b[1] - margin <= y + 0.5 and y + 0.5 <= b[3] + margin
    return inside and sqrt(sum((view[:, y, x] - ref)^2)) <= tol

function segment_box(view: array<real>[3, S, S], box: array<real>[4], tol: real, margin: real, w0: int, h0: int)
        -> array<int>[S, S]:
    b = [box[0] * S / w0, box[1] * S / h0, box[2] * S / w0, box[3] * S / h0]
    sx = floor((b[0] + b[2]) / 2)
    sy = floor((b[1] + b[3]) / 2)
    require 0 <= sx and sx < S and 0 <= sy and sy < S
    ref = view[:, sy, sx]
    mask = zeros(int, [S, S])
    if admits(view, b, ref, tol, margin, sy, sx):
        mask[sy, sx] = 1
    changed = true
    while changed:
        changed = false
        for y in 0 .. S:
            for x in 0 .. S:
                if mask[y, x] == 0 and admits(view, b, ref, tol, margin, y, x):
                    next_to = (y > 0 and mask[y - 1, x] == 1) or (y < S - 1 and mask[y + 1, x] == 1) or (x > 0 and mask[y, x - 1] == 1) or (x < S - 1 and mask[y, x + 1] == 1)
                    if next_to:
                        mask[y, x] = 1
                        changed = true
    return mask
```

The mask in the frame is rarely the true one pixel for pixel: the nearest rule from a grid of $64 \times 64$ back to $96 \times 61$ repeats some input pixels and skips others, so a rectangle of 80 true pixels comes back as 72, 80 or 88 depending on where its edges fall. That is the stand-in's resolution, as a stride-16 embedding is the real model's.

<!-- worked: project_vision_pipeline.segmentation_table -->
The box prompts of frame 0 (the kept detections, in frame pixels) run through the segmenter of milestone 3: tolerance 1.0 on the colour distance, box grown by 2 input pixels. The last two columns carry the prompt to the segmenter by the detector's letterbox map instead of the per-axis scale, which is the mistake of confusing the two views.

| object | prompt box | pixels in the input | pixels in the frame | true pixels | IoU with the truth | pixels with the letterbox map | IoU with the letterbox map |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | (7.19, 22.72, 18.32, 29.78) | 56 | 88 | 80 | 0.909 | 77 | 0.805 |
| B | (71.18, 13.10, 90.92, 30.59) | 224 | 336 | 352 | 0.955 | 273 | 0.776 |
| C | (76.19, 42.32, 85.73, 50.78) | 48 | 72 | 80 | 0.900 | 123 | 0.015 |
<!-- /worked -->

The three masks have IoU 0.909, 0.955 and 0.900 with the truth, from boxes that were themselves detected (so their errors are included). The last two columns are the mistake this milestone exists to catch. Carrying the box to the segmenter by the detector's map, scale $2/3$ and a padding of 11 rows, leaves A and B with a seed that still lands on the right object, since both are tall and near the middle, and masks that are shifted and clipped (0.805 and 0.776); for C, which sits low in the frame, the same error puts the seed on the background, and the mask is a patch of background of 123 pixels (0.015). A single test on the largest object would not have seen it. The lesson of Chapter 4 about boxes carries over: every map between coordinate systems is a place for a one-line mistake that leaves the output in a plausible place, and it needs a test per object, including an object that is not in the middle.

Chapter 7's decoder returns several masks with a predicted overlap each, and a caller picks one; the stand-in returns one. That is where the stand-in is weakest, and the extensions return to it. Everything else of the interface is kept: the encoder runs once per image and each prompt is a cheap call, which is how milestone 5 prices three prompts on one frame.

Check: the pixel counts and IoUs of the table, and the two mistaken ones.

### Milestone 4: a tracker with a bank, and an object that can be absent

The tracker follows one object from a mask on frame 0 (the prompt frame, the only conditioning frame). It keeps a **bank**: for every frame it has processed, a slab holding the 256 cells of a $16 \times 16$ grid, each cell the mean normalized colour of its $4 \times 4$ block of the segmenter's input (summed row by row within the block) and a mask channel, the fraction of the block that the object's mask covers. Frame $t$ reads the slabs that the rule of Chapter 7 selects, here from a clip of 12 frames:

<!-- worked: project_vision_pipeline.bank_table -->
The slabs that the stand-in's tracker reads at some frames of the 12-frame clip, prompted on frame 0 (the rule of Chapter 7 with 7 slots; each slab holds the 256 cells of a 16 by 16 grid).

| frame t | source frames of the slabs | slots | keys of the stand-in's attention |
|---:|---|---|---|
| 1 | 0 | 0 | 256 |
| 2 | 0, 1 | 0, 6 | 512 |
| 6 | 0, 1, 2, 3, 4, 5 | 0, 2, 3, 4, 5, 6 | 1536 |
| 7 | 0, 1, 2, 3, 4, 5, 6 | 0, 1, 2, 3, 4, 5, 6 | 1792 |
| 8 | 0, 2, 3, 4, 5, 6, 7 | 0, 1, 2, 3, 4, 5, 6 | 1792 |
| 9 | 0, 3, 4, 5, 6, 7, 8 | 0, 1, 2, 3, 4, 5, 6 | 1792 |
| 10 | 0, 4, 5, 6, 7, 8, 9 | 0, 1, 2, 3, 4, 5, 6 | 1792 |
| 11 | 0, 5, 6, 7, 8, 9, 10 | 0, 1, 2, 3, 4, 5, 6 | 1792 |
<!-- /worked -->

Frame 1 reads only the prompt frame; frame 2 reads it and frame 1, at slot 6 because the previous frame is always the last slot; from frame 7 the bank is full. Frame 9, the frame where A comes out, reads frames 0 and 3 to 8, which reaches back over both hidden frames to four in which A was visible, three of them in full. Remember that: it is why frame 9 can work, and it is the thing a shorter bank loses.

The memory attention of the stand-in has no learned projections: queries and keys are the cell features themselves, and the score between cell $q$ of the current frame and cell $k$ of a remembered slab is

$$
\mathrm{score}(q, k) = -\frac{\lVert \mathbf{f}_q - \mathbf{f}_k \rVert^2}{\tau_{\mathrm{mem}}} - \frac{\Delta x^2 + \Delta y^2}{2\,\rho_{\mathrm{loc}}^2}, \qquad \tau_{\mathrm{mem}} = 0.25, \quad \rho_{\mathrm{loc}} = 1.5, \tag{1}
$$

where $\Delta x$ and $\Delta y$ are the offsets of the two cells in cells. The first term says that similar colours match; the second prefers cells that are near each other, which is the role that the rotary position of Chapter 7 plays in the real model and what lets a track tell its own object from a look-alike. The read-back is the weighted average of the mask channel under a softmax over all keys of all slabs, plus one **no-object** key that holds nothing, with the score $-4$ and the mask value 0:

$$
r_q = \frac{\sum_k e^{\mathrm{score}(q, k)}\, m_k}{e^{-4} + \sum_k e^{\mathrm{score}(q, k)}}. \tag{2}
$$

A cell reads back near 1 when the nearby remembered cells that look like it were the object, near 0 when they were not, and near 0 when nothing remembered looks like it (the no-object key then dominates the denominator). The constants $\tau_{\mathrm{mem}}$ and $\rho_{\mathrm{loc}}$ belong to this stand-in and carry subscripts so that they are not mistaken for the temperature or the correlation of other chapters.

```text
function read_memory(f: array<real>[256, 3], slabs: list<(array<real>[256, 3], array<real>[256])>,
                     tau_mem: real, rho_loc: real, z_nil: real) -> array<real>[256]:
    r = zeros(real, [256])
    for q in 0 .. 256:
        num = 0.0
        den = exp(z_nil)
        for s in 0 .. len(slabs):
            for k in 0 .. 256:
                d2 = sum((f[q, :] - slabs[s][0][k, :])^2)
                dc = ((q mod 16) - (k mod 16))^2 + ((q // 16) - (k // 16))^2
                w = exp(-d2 / tau_mem - dc / (2.0 * rho_loc^2))
                num = num + w * slabs[s][1][k]
                den = den + w
        r[q] = num / den
    return r
```

The frame's decision comes from the read-back. A cell is **on** when $r_q \ge 0.5$. The object is present when at least one cell is on; the object score is $\max_q r_q - 0.5$, positive exactly then. When present, the mask is decoded in two steps: the on-cells are dilated by one cell in each direction, and within the dilated region every pixel of the segmenter's input whose colour is within 1.0 of the track's reference colour (the mean normalized colour over the prompt mask, summed in row-major order) is in the mask. When absent, the mask is empty, and the slab that the frame writes into the bank has a mask channel of zeros: the bank records the absence. That last rule is the occlusion rule of Chapter 7 in the stand-in's terms (SAM 2 adds a learned vector to the memory of a frame whose object score is not positive; here the stand-in's mask channel carries the absence, since it has no learned vectors).

```text
function track_step(f_t: array<real>[256, 3], view_t: array<real>[3, S, S], bank: map<int, (array<real>[256, 3], array<real>[256])>,
                    t: int, n_recent: int, ref: array<real>[3]) -> (bool, array<int>[S, S], array<real>[256]):
    slabs = []
    for (tp, src) in memory_slots(t, [0], 7):
        if tp == 0 or tp > 6 - n_recent:
            append bank[src] to slabs
    r = read_memory(f_t, slabs, 0.25, 1.5, -4.0)
    on = zeros(int, [256])
    for q in 0 .. 256:
        if r[q] >= 0.5:
            on[q] = 1
    present = max(r) >= 0.5
    mask = zeros(int, [S, S])
    if present:
        dil = dilate3(on)
        for y in 0 .. S:
            for x in 0 .. S:
                if dil[(y // 4) * 16 + (x // 4)] == 1 and sqrt(sum((view_t[:, y, x] - ref)^2)) <= 1.0:
                    mask[y, x] = 1
    return (present, mask, r)
```

`dilate3` is the $3 \times 3$ maximum of the $16 \times 16$ map of on-cells, clipped at the border, and `memory_slots(t, [0], 7)` is Chapter 7's rule; `n_recent` is the number of recent slots kept (6 is the full bank). With the full bank, the three tracks over the clip:

<!-- worked: project_vision_pipeline.tracking_table -->
The three tracks over the clip with the full bank. Read-back is the largest cell read-back of the frame, tau = 0.25, rho = 1.5 cells, a no-object key of score -4; a cell is on at 0.5 or more; the mask is the pixels of the dilated on-cells within 1 of the track's reference colour. Pixels / true is the mask on the frame against the visible pixels of the object; IoU is of those two sets. Frame 0 is the prompt.

| t | track A: read-back, pixels / true, IoU | track B: read-back, pixels / true, IoU | track C: read-back, pixels / true, IoU |
|---:|---|---|---|
| 0 | prompt  88 / 80  0.909 | prompt  336 / 352  0.955 | prompt  72 / 80  0.900 |
| 1 | 0.904  80 / 80  1.000 | 0.988  368 / 352  0.957 | 0.987  80 / 80  1.000 |
| 2 | 0.907  72 / 80  0.900 | 0.989  352 / 352  1.000 | 0.990  88 / 80  0.909 |
| 3 | 0.909  88 / 80  0.909 | 0.988  336 / 352  0.955 | 0.988  72 / 80  0.900 |
| 4 | 0.910  80 / 80  1.000 | 0.975  368 / 352  0.957 | 0.988  80 / 80  1.000 |
| 5 | 0.907  72 / 80  0.900 | 0.976  352 / 352  1.000 | 0.989  88 / 80  0.909 |
| 6 | 0.699  72 / 64  0.889 | 0.975  336 / 352  0.955 | 0.987  72 / 80  0.900 |
| 7 | 0.012  absent  correct | 0.977  368 / 352  0.957 | 0.988  80 / 80  1.000 |
| 8 | 0.012  absent  correct | 0.984  352 / 352  1.000 | 0.988  88 / 80  0.909 |
| 9 | 0.814  64 / 56  0.875 | 0.986  336 / 352  0.955 | 0.986  72 / 80  0.900 |
| 10 | 0.898  80 / 80  1.000 | 0.976  368 / 352  0.957 | 0.987  80 / 80  1.000 |
| 11 | 0.901  72 / 80  0.900 | 0.976  352 / 352  1.000 | 0.988  88 / 80  0.909 |
<!-- /worked -->

Read the table against the puzzle of the opening. Tracks B and C never lose their object and their masks score 0.9 to 1.0 against the truth. Track A is on for frames 1 to 6, reports "absent" for 7 and 8 (its largest read-back falls from 0.699 on frame 6, where only an 8-column strip of A is left, to 0.0125 and 0.0118 on the hidden frames, so the score is negative by almost 0.5), and comes back at 0.814 on frame 9 with 64 pixels against 56 true. The prompt frame itself costs nothing here, since the mask of milestone 3 is what the tracker's bank starts from: segmenting frame 0 and conditioning the tracker are one operation, as they are in the real model.

Now remove things. The bank is the whole difference between a tracker and a re-detection, and its length is the first thing to vary:

<!-- worked: project_vision_pipeline.ablation_table -->
The same clip and tracks with different banks. Presence is one character per frame (1 present, 0 absent) from frame 0 to 11; the number after it is the mean mask IoU over the frames where the object is visible. The last row keeps the full bank and sets rho to 1000 cells, which removes the preference for nearby cells.

| memory | track A: presence, mean IoU | track B: presence, mean IoU | track C: presence, mean IoU |
|---|---:|---:|---:|
| prompt frame only | 111111100000 0.651 | 111111110000 0.614 | 111111100000 0.543 |
| prompt frame and 1 recent | 111111100000 0.651 | 111111111111 0.970 | 111111111111 0.830 |
| prompt frame and 2 recent | 111111100000 0.651 | 111111111111 0.970 | 111111111111 0.830 |
| prompt frame and 3 recent | 111111100000 0.651 | 111111111111 0.970 | 111111111111 0.936 |
| prompt frame and 6 recent (the full bank) | 111111100111 0.928 | 111111111111 0.970 | 111111111111 0.936 |
| full bank, no locality (rho = 1000) | 111111111111 0.505 | 111111111111 0.970 | 111111111111 0.601 |
<!-- /worked -->

![Mask IoU with the truth over the 12 frames for the trackers of A and C, with the full bank and with a bank of the prompt frame and 2 recent frames: after A's hidden frames 7 and 8 the full bank recovers A at about 0.9 while the short bank stays at 0, and the short bank's C falls to about 0.5.](img/project-vision-bank-length.svg)

A bank that holds only the prompt frame loses every object within a few frames, as soon as the object has moved farther than the locality of equation (1) reaches: the memory of where B was on frame 0 says nothing about where it is on frame 8. One recent frame is enough for B, which never leaves, and for C until frame 9; it is not enough for A, which is lost for good when it is hidden, because the bank then holds empty slabs in place of the last frames in which A was seen, and what is read back is zero. Two recent frames fail the same way. Three reach back over the two hidden frames to the frame-6 slab, which is not enough either, and the reason is instructive. Frame 6 shows only the 8-column strip of A that B has not yet covered, so the evidence for A that frame 9 can read is a few cells of one slab, against the red cells of C that are in every slab.

<!-- worked: project_vision_pipeline.lookalike_table -->
Frame 9, where A has come out from behind B. For the tracks of the two red objects, the cells whose read-back is at least 0.5, and how many of them have their centre (mapped to frame pixels) inside the full box of A and of C.

| bank | tracker | cells at least 0.5 | in the box of A | in the box of C | largest read-back |
|---|---|---:|---:|---:|---:|
| prompt frame and 6 recent (full) | track A | 2 | 2 | 0 | 0.814 |
| prompt frame and 6 recent (full) | track C | 2 | 0 | 2 | 0.986 |
| prompt frame and 3 recent | track A | 0 | 0 | 0 | 0.334 |
| prompt frame and 3 recent | track C | 2 | 0 | 2 | 0.987 |
| prompt frame and 2 recent | track A | 0 | 0 | 0 | 0.017 |
| prompt frame and 2 recent | track C | 3 | 1 | 2 | 0.984 |
<!-- /worked -->

With three recent frames, the largest read-back of A on frame 9 is 0.334, below the 0.5 that marks a cell on; with the full bank it is 0.814, because frames 3 to 5 add three full views of A close to where it is now. The look-alike appears in the table too. With the full bank, the track of C has two cells on, both in C's box, and A's cells are not claimed by it, because the remembered A cells have the same colour as the new ones and a mask value of 0 for the C track, and they sit close enough to outweigh C's far-away cells. With the short bank (the prompt frame and two recent frames, which on frame 9 are the two hidden ones), the memory it reads at frame 9 holds no A-coloured cell near where A is now, and the C track's cells spill onto A: one of its three cells lies in A's box, its masks on frames 9 to 11 fall to an IoU of about 0.5 (the plot), and its mean IoU drops to 0.830. The last row of the ablation is the other half: without the locality term ($\rho_{\mathrm{loc}} = 1000$) the two red objects cannot be told apart at any frame: A's mean IoU is 0.505 and C's is 0.601, because each track's mask covers a mixture of both.

The check for this milestone is the presence strings, the pixel counts of the tracking table, the mean IoUs of the full bank and the ablation, and the smallest distance of any read-back from 0.5, 0.0109, which is also the evidence that your implementation's rounding cannot change a decision.

### Milestone 5: the whole run, six mistakes and the cost

The run is a short function: detect on frame 0, give each detection to the segmenter as a box prompt in the order of falling score, start a track from each mask, and for every later frame view it once and step every track.

```text
function run_pipeline(frames: list<array<int>[H, W, 3]>, protos: array<real>[4, 3], mean: array<real>[3], std: array<real>[3])
        -> list<list<(bool, array<int>[S, S])>>:
    dets = detect_frame(frames[0], protos, 0.25, 0.45)
    scores = zeros(real, [len(dets)])
    for n in 0 .. len(dets):
        scores[n] = dets[n][1]
    order = argsort(scores, descending)
    view0 = segmenter_view(frames[0], 64, mean, std)
    refs = []
    banks = []
    results = []
    for n in order:
        m0 = segment_box(view0, dets[n][0], 1.0, 2.0, W, H)
        bank = map of int to (array<real>[256, 3], array<real>[256]), empty
        bank[0] = (cell_features(view0), mask_cells(m0))
        append mean_colour(view0, m0) to refs
        append bank to banks
        append [(true, m0)] to results
    for t in 1 .. len(frames):
        view = segmenter_view(frames[t], 64, mean, std)
        f = cell_features(view)
        for k in 0 .. len(refs):
            (present, mask, r) = track_step(f, view, banks[k], t, 6, refs[k])
            banks[k][t] = (f, mask_cells(mask) if present else zeros(real, [256]))
            append (present, mask) to results[k]
    return results
```

Here `cell_features(view)` is the $[256, 3]$ array of block means of milestone 4, `mask_cells(mask)` the $[256]$ array of the fraction of each block that a mask covers and `mean_colour(view, mask)` the mean of the normalized colours over the pixels of a mask, summed in row-major order. The features of a frame are computed once and shared by every track: that is the encode-once structure of Chapter 7, and the ledger below prices it. As a cross-check that needs nothing but the pipeline itself, run the detector on every frame too and compare each track's mask, taken to its bounding box, with the detections:

<!-- worked: project_vision_pipeline.monitor_table -->
Each frame is also run through the detector. For every track that reports its object present, the largest IoU between the bounding box of its mask (in frame pixels) and a detection of the track's class; absent where the track says so.

| t | track A | track B | track C |
|---|---:|---:|---:|
| 0 | 0.845 | 0.864 | 0.797 |
| 1 | 0.910 | 0.847 | 0.788 |
| 2 | 0.832 | 0.882 | 0.765 |
| 3 | 0.845 | 0.905 | 0.797 |
| 4 | 0.910 | 0.925 | 0.788 |
| 5 | 0.832 | 0.886 | 0.765 |
| 6 | 0.806 | 0.864 | 0.797 |
| 7 | absent | 0.847 | 0.788 |
| 8 | absent | 0.882 | 0.765 |
| 9 | 0.813 | 0.905 | 0.797 |
| 10 | 0.910 | 0.925 | 0.788 |
| 11 | 0.832 | 0.886 | 0.765 |
<!-- /worked -->

Every present track agrees with a detection of its class at an IoU of at least 0.75 (the smallest is 0.765), and the two frames where A is absent are the two where the detector returned two boxes. A pipeline that does this check on every frame has a tripwire: a track that is present with no agreeing detection is a drifted mask or a missed object, and one that is absent where a detection exists is an object the memory has lost.

Now the mistakes. The acceptance of a project is meaningful only if it fails when the pipeline is wrong, so each of six is planted into an otherwise correct pipeline and measured:

<!-- worked: project_vision_pipeline.planted_mistakes -->
Each planted mistake changes one thing in an otherwise correct pipeline and reports the measurement that sees it. Found is the number of visible objects matched at IoU 0.5 over the 12 frames (34 are visible).

| planted mistake | stage | what the pipeline reports | the check that sees it |
|---|---|---|---|
| none | all three stages | 34 of 34 found, mean IoU 0.841 | baseline |
| the detector's view has its channels in the order b, g, r | preprocessing | 12 of 34 found, mean IoU 0.885 | found count |
| the inverse letterbox does not subtract the padding | detector tail | 0 of 34 found | found count |
| the inverse uses half the slack, 11.5, where the pixels were pasted at 11 | detector tail | 34 of 34 found, mean IoU 0.823 | mean IoU, not the count |
| the box prompt reaches the segmenter through the letterbox map | segmenter | mask IoU on frame 0: A 0.805, B 0.776, C 0.015 (correct: 0.909, 0.955, 0.900) | mask IoU per object |
| attention without locality (rho = 1000) | tracker | mean mask IoU of A 0.505 and C 0.601 (correct: 0.928 and 0.936) | mean mask IoU of the two look-alikes |
| a bank of the prompt frame and 2 recent frames | tracker | A present 111111100000 (correct: 111111100111); C mean IoU 0.830 (correct: 0.936) | presence pattern of A |
<!-- /worked -->

Three of the six are caught by a count of the objects found or by a pixel-level IoU, and a count is the cheapest check there is. A channel-order error in the detector's view (here because the red of A and C looks like blue to a program that expects red) finds only B, the green object that survives a swap of red and blue almost unchanged; a test on that object alone would have passed. A missing padding subtraction finds nothing at all, because every box is 16.5 frame rows off (the padding of 11 input rows divided by the scale 2/3), more than the height of A.

The one the count does not catch is the padding. The inverse letterbox with 11.5 (half the slack, which no pixel row can sit at) where the pixels were pasted at 11 still finds all 34 objects, because a shift of half an input pixel is 0.75 frame pixel and the matching threshold is a coarse test; what it does is lower the mean IoU from 0.841 to 0.823, and the right acceptance test is the mean and not the count. This is the half-pixel disagreement that brain's letterbox once had between its pixels and its boxes (Chapter 1 tells the story of commit `4a7148ae9`, which gave them one whole-pixel pad), measured here on objects of 8 to 22 pixels: 0.018 of mean IoU, which is the price of one convention being used for pixels and another for boxes. The last two are the tracker's: locality removed, and a short bank, each leaving the two look-alikes entangled or a track lost.

The last number is the cost. A frame costs the segmenter side what the formulas of Chapter 7 say, and the point of the pipeline's structure is which parts are paid once. The trunk (Hiera, 1,812 billion FLOPs for the large model) depends on the frame only; the memory attention, the decoder and the memory encoder are paid per track.

<!-- worked: project_vision_pipeline.ledger -->
FLOPs (2 per multiply-add) of the segmenter side of the pipeline for the 12-frame clip priced as the large SAM 2.1 model at 1,024 by 1,024 pixels, with the formulas of Chapter 7, in billions of FLOPs to four decimals. The trunk costs 1,812.1469 once per frame. For every track: the memory encoder 11.6077; the decoder 3.6393 with the tracking prompt, and 3.6557 on frame 0, where the prompt is a box (2 corner tokens and the padding point); and, on frames 1 to 11, a memory attention of 115.9641 + 0.01703936 per key (in FLOPs, 115,964,116,992 + 17,039,360 per key), with keys = 4,096 per slab + 4 per pointer. The bank of a frame holds the slab of frame 0, the slabs of the (up to 6) most recent earlier frames other than frame 0, and one pointer from each earlier frame (at most 11 back); frame 0 has no memory attention. A track's cost on a frame is its memory attention, decoder and memory encoder; the shared column adds the trunk once to the three tracks, the other adds it to each. The detector is not counted. Every figure is the sum of exact counts, rounded once to four decimals; a reader who adds the printed terms gets the totals to within 0.01.

| frame | memory keys per track | trunk (GFLOPs) | per track: attention, decoder, encoder | 3 tracks, trunk shared | 3 tracks, trunk per track |
|---|---:|---:|---:|---:|---:|
| 0 | none | 1,812.1469 | 15.2634 | 1,857.9373 | 5,482.2312 |
| 1 | 4,100 | 1,812.1469 | 201.0725 | 2,415.3645 | 6,039.6584 |
| 2 | 8,200 | 1,812.1469 | 270.9339 | 2,624.9486 | 6,249.2425 |
| 3 | 12,300 | 1,812.1469 | 340.7953 | 2,834.5327 | 6,458.8266 |
| 4 | 16,400 | 1,812.1469 | 410.6566 | 3,044.1169 | 6,668.4108 |
| 5 | 20,500 | 1,812.1469 | 480.5180 | 3,253.7010 | 6,877.9949 |
| 6 | 24,600 | 1,812.1469 | 550.3794 | 3,463.2851 | 7,087.5790 |
| 7 | 28,700 | 1,812.1469 | 620.2408 | 3,672.8692 | 7,297.1631 |
| 8 | 28,704 | 1,812.1469 | 620.3089 | 3,673.0737 | 7,297.3676 |
| 9 | 28,708 | 1,812.1469 | 620.3771 | 3,673.2782 | 7,297.5721 |
| 10 | 28,712 | 1,812.1469 | 620.4452 | 3,673.4827 | 7,297.7766 |
| 11 | 28,716 | 1,812.1469 | 620.5134 | 3,673.6871 | 7,297.9810 |
| all 12 |  |  |  | 37,860.2770 | 81,351.8037 |

One track alone costs 27,117.27 billion FLOPs over the clip; three tracks that share the trunk cost 37,860.28 billion, 1.40 times as much, where three separate pipelines cost 81,351.80 billion, 3.00 times.
<!-- /worked -->

Three tracks that share the trunk cost 1.40 times what one track costs over the clip, where three separate pipelines would cost three times as much. Per track and frame, memory attention is nearly all of the 550 to 621 billion FLOPs of the per-track column once the bank fills, and it is the cost that a false positive at milestone 2 multiplies: the six detections of the loose threshold of 0.8 would be six tracks and six times the per-track column, with three of the six following an object that another track already follows.

## What a correct pipeline prints

The values below are the tests for the pipeline. Test in the order of the milestones: a disagreement at milestone 4 that follows agreement at 1 to 3 points at the tracker and not at the preprocessing. "Exact" means equal digests, integers and strings; a numeric tolerance is absolute.

<!-- worked: project_vision_pipeline.acceptance_fingerprint -->
Expected values of the project, computed by this module. A tolerance of 0.001 or 1e-6 is absolute; exact means equal.

| milestone | quantity | value | tolerance |
|---:|---|---|---|
| 1 | SHA-256 (first 16 hex) of frames 0, 7 and 11 | 5206fd6a9bf2e8c8 / 2f3971e198d70e95 / 1d0bc3156b7d8ead | exact |
| 1 | visible pixels of A at frames 0, 6, 7, 9 | 80 / 64 / 0 / 56 | exact |
| 1 | detector view: scale, content, padding | 0.666667; 64 x 41; (0, 11) | 6 digits; exact |
| 1 | sum of the 12,288 values: detector view, segmenter view of frame 0 | 3547.631373 / -13516.819217 | 1e-6 |
| 2 | frame 0: candidates, with score at least 0.25, kept | 60 / 29 / 3 | exact |
| 2 | frame 0: IoU evaluations of suppression at 0.45 and kept at 0.8 and 0.99 | 30; 6 and 29 | exact |
| 2 | boxes found, of all visible, over the 12 frames; mean IoU | 34 of 34; 0.841 | exact; 0.001 |
| 2 | frame 0 kept boxes in frame pixels, by falling score, with the object each matches | A (7.19, 22.72, 18.32, 29.78) 1.000; B (71.18, 13.10, 90.92, 30.59) 0.775; C (76.19, 42.32, 85.73, 50.78) 0.714 | 0.01; 0.001 |
| 3 | mask pixels in the input / in the frame (A, B, C) and IoU with the truth | A 56/88; B 224/336; C 48/72; IoU 0.909, 0.955, 0.900 | exact; 0.001 |
| 4 | presence string over frames 0 to 11 of A, B, C | A 111111100111, B 111111111111, C 111111111111 | exact |
| 4 | mean mask IoU over the visible frames of A, B, C | 0.928, 0.970, 0.936 | 0.001 |
| 4 | smallest distance of any read-back from 0.5 over frames 1 to 11, all tracks | 0.0109 | 0.001 |
| 4 | largest read-back of A at frames 7 and 8 | 0.0125 / 0.0118 | 0.001 |
| 4 | bank of the prompt frame and 2 recent: presence of A and mean IoU of C | 111111100000; 0.830 | exact; 0.001 |
| 5 | GFLOPs of the clip, 3 tracks sharing the trunk / not sharing it | 37,860.28 / 81,351.80 | 0.01 |
<!-- /worked -->

Beyond the table, an implementation passes when it also meets these, all of which are in the tables above: the per-frame rows of the detector table (the raw, filtered and kept counts, and 34 of 34 found); the pixel counts of the segmentation table and its two shifted-prompt IoUs (0.805, 0.776 and 0.015); the three presence strings of the full bank, with A absent on frames 7 and 8 and nowhere else; the ablation table's presence strings and mean IoUs; and every one of the six planted mistakes producing the number the table prints, to its stated resolution. The mistakes are not optional: a pipeline that passes the clean rows and does not fail the planted ones has a test that cannot fail.

## What stands behind each stand-in in brain

The seams of the pipeline are typed in brain's SDK. `DetectionPipeline::detect` in `crates/sdk/src/detect.rs` returns `Detection` values with `x1, y1, x2, y2`, a confidence and a class index in the original image's pixel coordinates, already mapped back through the model's internal letterbox, with the defaults 0.25 and 0.45 that this project uses. `Prompt::bbox` in `crates/sdk/src/segment.rs` takes a box in source-image pixel coordinates, the space that the image's own width and height report and not the model's internal square, and turns it into the two corner points with the labels 2 and 3 of Chapter 7 (`to_coords_labels` puts them before any click point). A detection's four numbers are therefore a box prompt's four numbers with no conversion, which is the property milestone 3 tests: the one map that has to be right is the one inside each model.

The detector tail is shared, as Chapter 4 told: IoU, class-aware suppression with a strict comparison and ties to the lower index, and the letterbox are in `crates/imaging/src/nms.rs` and `crates/imaging/src/letterbox.rs`, moved there from the YOLOv8 crate so that a second detector would not depend on its training code (commit `c3a161a5d`). The detector fills the padding with 114/255 and passes no normalization, in `crates/yolov8/src/infer.rs`. The segmenter does the opposite on every count, in `crates/sam2/src/caps.rs`: it resizes the source image to the square input with each axis on its own scale, by a bilinear resize, with the parity tests gated on the normalized input, then normalizes with the ImageNet constants held in `Sam2Config` (`pixel_mean`, `pixel_std`), and it maps a prompt's source-pixel coordinates with the same per-axis scales. Two models, two preprocessings, one coordinate convention at the seam: that is the structure of this project, and of brain's.

The tracker is `Tracker` in `crates/sam2/src/video.rs`, ported with its memory attention and encoder in commit `6bcc0e79f`: `prompt` places the conditioning frame and `track` conditions every later frame on the memories recorded so far, the slot rule is Chapter 7's, and the object score is carried in each memory entry with the meaning that a value at or below 0 is absent or fully occluded. A `Tracker` follows one object, so several objects take one tracker each, sharing the encoder's output for each frame, which is the arrangement of milestone 5 and the reason the shared-trunk column of its ledger is the right one. The mask sequence a tracker emits is defined in `crates/imaging/src/maskseq.rs`, one sequence that travels as a blob between served actions and as a directory of per-frame images and a manifest on disk, which declares the mask polarity and treats the mask of a negative object score as legitimately empty: a consumer that treats an empty mask as a bug chases a phantom, which is the stand-in's rule that an absent object is a valid output. The served action `track` (`crates/sam2/src/track.rs`, run as `brain sam2 track`) prompts with points on one frame; the `Prompt` of the library takes a box as well, so a detection can place the first frame's prompt.

The letterbox of the detector's view is one rule in one place. `Letterbox::compute` in `crates/imaging/src/letterbox.rs` takes the pad as the floor of half the slack, with the extra row of an odd slack below, and the pixels, `apply_box`, `invert_box` and `content_rect` all use it; the content's source taps come from `nearest_tap`, the integer form of the half-pixel nearest rule with the source and content extents, which is the rule of this milestone's table. The host loop `letterbox_rgb` and the device path `Ctx::letterbox` in `crates/imaging/src/device.rs` run that one rule, and a test (`host_letterbox_equals_the_device_letterbox_bitwise` in `crates/imaging/tests/it/device_ops.rs`) holds them bitwise equal. The half-pixel disagreement of the planted mistake of milestone 5 is the one the first version of the crate had: commit `4a7148ae9` found that pixels were pasted at the truncated pad while boxes were mapped with the float one, on an odd slack half a pixel apart, and replaced it with the single whole-pixel pad, a spec test (`pixels_and_boxes_share_one_integral_pad`) holding the two together. The depth model's INT8 calibration, the other caller Chapter 1 told about, does not letterbox at all: the served `calib` action (`crates/zipdepth/src/caps.rs`) prepares every image exactly as the predictor does, with `zipdepth::predict::preprocess_chw`, because scales fitted to a padded square would describe activations the model never produces at inference.

## Summary

If every acceptance test passes you have a pipeline that reads one frame two ways, finds objects with a head, a filter and a suppression and returns them in the frame's own pixels, turns each box into a mask, and follows each mask through a clip with a bank that reaches back over an occlusion, reports an object absent when it is hidden, finds it again when it comes out, and does not hand a look-alike's pixels to the wrong track. You also have six planted mistakes, each caught by a number.

What the project established, in order. A frame has as many coordinate systems as it has consumers: here the frame, the detector's input and the segmenter's, and the only safe interface between them is a box in frame pixels. A stand-in that reads pixels is worth more than a recording, because a preprocessing mistake changes what it reads. Suppression has a range of good thresholds, and a loose one is paid for downstream, in tracks. A prompt is carried by the map of the model that receives it, and a test per object, including one off-centre, is what sees a wrong map. A tracker's memory has two numbers that matter: its reach, which must exceed the occlusion, and what its slabs hold when the object is partly hidden, since a slab of a hidden object holds nothing; and locality is what keeps look-alikes apart. And the cost: the trunk once per frame and the rest per track, so the number of tracks, which the detector's threshold sets, is the number that scales the bill.

Where this stops. The stand-ins are colour programs on a clean synthetic scene; a real detector's errors are not a fixed pattern and a real tracker has no reference colour. The scene has one occlusion by a larger object and no appearance change, no object entering and no object leaving. The segmenter returns one mask where the real model returns several with a predicted overlap, and the tracker has no object pointers. The ledger prices the large model by formula and not by a run, and leaves the detector's cost uncounted. These limits are the reason the extensions exist. The volume closes here: its chapters taught what an image is as data, what a network finds in it, how it is boxed, labelled, segmented, described, read and recognized, and this project checks that those parts compose.

## References

- N. Ravi, V. Gabeur, Y.-T. Hu, R. Hu, C. Ryali, T. Ma, H. Khedr, R. Radle, C. Rolland, L. Gustafson and others, *SAM 2: segment anything in images and videos*, 2024. The memory bank, memory attention and the object score that the tracker's rules follow.
- A. Kirillov, E. Mintun, N. Ravi, H. Mao, C. Rolland, L. Gustafson, T. Xiao, S. Whitehead, A. C. Berg, W.-Y. Lo, P. Dollar and R. Girshick, *Segment anything*, ICCV 2023. The box prompt as two corner points.
- Z. Tian, C. Shen, H. Chen and T. He, *FCOS: fully convolutional one-stage object detection*, ICCV 2019. Boxes as distances from a point and the centerness of a cell.

## Making the stand-ins real

These are optional. Each is a larger test of the same ideas, and each changes one thing of the project and asks you to predict the outcome before you run it.

1. **An object that enters.** Add a fourth rectangle that enters the frame at frame 6. The detector run of milestone 5 sees it on every frame after, and no track follows it. Design a rule that starts a track from a detection that no present track's mask box overlaps at IoU 0.5, using frame 6 as a second conditioning frame (Chapter 7's slot rule puts every conditioning frame in slot 0), and decide in advance what the rule does to the two look-alikes when they cross.
2. **Longer occlusions.** Let B hold over A for $k$ frames by changing its speed. Predict the largest $k$ for which the full bank of 6 recent frames finds A again, and say whether the reach of the bank (frames, not pixels) or the evidence in the surviving slabs sets the limit; run $k$ from 1 to 7.
3. **Object pointers.** Add to each bank entry a summary of the object as the decoder saw it (the mask's centroid and mean colour, of the best frame so far), and let the tracker fall back on it when every slab it reads holds an empty mask. Predict which of the occlusion lengths of the previous extension this repairs.
4. **Several masks and a predicted overlap.** Return from the segmenter the three masks of milestone 3's opening toy at tolerances 0.5, 1.0 and 2.0 and choose among them by a predicted overlap of your own design (the mask's area against the box's). Decide what the right choice is for each of A, B and C and test it.
5. **A priced detector schedule.** Run the detector on every $N$-th frame only, using its boxes to restart tracks that have been absent. Count detector runs and the ledger's FLOPs for $N = 1, 2, 4$, and report the smallest $N$ at which A is still found on frame 9 by the pipeline and not by the tracker.

## Answers

**Recall questions.**

1. $s = \min(64/96, 64/61) = 2/3$; the content is $64 \times \mathrm{round}(40.67) = 64 \times 41$; the slack is $(0, 23)$ on $(x, y)$ and half of it $(0, 11.5)$. A pixel row cannot sit at 11.5, so the pad is whole pixels, 11 above and 12 below (the floor of half the slack above), and boxes are mapped with the same integer 11.
2. The IoU of the two boxes is $45/100 = 0.45$ exactly, which is not greater than 0.45, so the second is kept. The strict comparison makes the threshold an exclusive upper bound on the IoU of two same-class survivors, and the evaluation of Chapter 4 calls a detection at exactly the matching threshold correct: one convention, written down once.
3. $(8.5 - 3.5,\ 28.5 - 2.5,\ 8.5 + 3.5,\ 28.5 + 2.5) = (5, 26, 12, 31)$.
4. Three: the two corners with the labels 2 and 3, and the padding point with the label $-1$ that every prompt carries. With the six output tokens the decoder has 9.
5. Frame 0 in slot 0 (the prompted frame) and frames 3, 4, 5, 6, 7 and 8 in slots 1 to 6; the previous frame, 8, is slot 6.
