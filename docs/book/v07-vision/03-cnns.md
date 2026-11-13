<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Convolutional Networks and Residual Learning

*How do stacks of convolutions become image classifiers, and why did residual connections let them go deep?*

Here are two 8 by 8 pictures. Each is made of 32 ones and 32 zeros, so their mean pixel is 0.5 and their histograms are identical. One is a set of vertical stripes and the other is a checkerboard.

<!-- worked: cnns.same_pixels -->
Two 8 by 8 images of zeros and ones. Vertical stripes: pixel (i, j) is 1 when j is even. Checkerboard: pixel (i, j) is 1 when i + j is even ('#' is 1, '.' is 0).

```text
# . # . # . # .    # . # . # . # .
# . # . # . # .    . # . # . # . #
# . # . # . # .    # . # . # . # .
# . # . # . # .    . # . # . # . #
# . # . # . # .    # . # . # . # .
# . # . # . # .    . # . # . # . #
# . # . # . # .    # . # . # . # .
# . # . # . # .    . # . # . # . #
```

The horizontal difference is x[i][j+1] - x[i][j] over its 8 * 7 = 56 valid positions and the vertical one is x[i+1][j] - x[i][j] over 7 * 8 = 56; the last two columns are the mean of the absolute values.

| image | pixels equal to 1 | mean pixel | mean absolute horizontal difference | mean absolute vertical difference |
|---|---:|---:|---:|---:|
| vertical stripes | 32 | 0.500 | 1.000 | 0.000 |
| checkerboard | 32 | 0.500 | 1.000 | 1.000 |
<!-- /worked -->

Whatever tells them apart is not in the pixel values; it is in which pixel sits next to which. A dense layer of Volume III cannot be told this: it takes the 64 numbers as a vector, and a layer trained on pictures whose 64 positions were shuffled by one fixed permutation learns exactly as well as one trained on the originals. The convolution of the previous chapter is the layer that does know, and the table above shows how little of it is needed. A filter that takes the difference of two horizontal neighbours answers 1 on average for both pictures and so cannot separate them. A second filter that takes the difference of two vertical neighbours answers 0 on the stripes and 1 on the checkerboard. The pair of averages, (1, 0) against (1, 1), is the whole difference between the pictures.

That small calculation has the shape of every convolutional classifier. A first layer of filters reports, at every position, which local patterns are present. Something has to boil the maps of answers down to a few numbers that do not care where in the picture the patterns occurred, because the question "is this a checkerboard?" has the same answer whether the board is shifted by a pixel or not. And a second layer of filters can look for combinations of first-layer answers, a horizontal and a vertical change at once, which no single filter of the first layer sees. This chapter builds that machine on a toy small enough to print every number of, trains it, checks its backward pass, and asks what a shift of the picture does to it. It then asks what happens when the stack is made deep, which is where the history of the subject turned: a network that cannot be trained to be deeper cannot be made better by depth. The remedy, the residual block, is a few lines of the arithmetic of Volume III's chapter on training dynamics applied to images, and the chapter counts what it costs, what its bottleneck variant saves, how the normalization in each block is folded away at inference, and how the maps of a deep network at three different scales are combined into the feature pyramid that the next chapter's detectors read.

A convolutional network stacks six earlier ideas, so the questions come before the architecture.

1. (Volume VII, [Chapter 2](02-convolution.md), convolution.) A $3 \times 3$ convolution with stride 1 and padding 1 takes a $[4, 4, 4]$ map to $[6, 4, 4]$. What are the output side and the number of multiply-adds?
2. (Volume VII, [Chapter 2](02-convolution.md), receptive field.) A $3 \times 3$ stride-1 layer has receptive field $r = 3$ and jump $j = 1$. What are $r$ and $j$ after a following layer with window 2 and stride 2?
3. (Volume III, [Chapter 3](../v03-neural-networks/03-softmax.md), softmax and cross-entropy.) What is the gradient of the cross-entropy loss of one example with respect to its logits?
4. (Volume III, [Chapter 9](../v03-neural-networks/09-initialization.md), He initialization.) What standard deviation does a ReLU layer with $n_{\mathrm{in}}$ inputs per output use, and what is $n_{\mathrm{in}}$ for a $3 \times 3$ convolution with $C_{\mathrm{in}}$ input channels?
5. (Volume III, [Chapter 12](../v03-neural-networks/12-training-dynamics.md), residual connection.) For $\mathbf{y} = \mathbf{x} + f(\mathbf{x})$, what is $d\mathbf{x}$ in terms of $d\mathbf{y}$, and why does a deep stack of such layers pass a gradient that a plain stack loses?
6. (Volume III, [Chapter 10](../v03-neural-networks/10-normalization.md), batch normalization.) What does a batch-norm layer use for its mean and variance in eval mode, and how does it fold into the linear layer before it?

## What a pooling layer keeps

The maps after a convolution are as large as the picture, and the layers above them have to answer a question about the whole of it. **Pooling** is the layer that shrinks a map by summarizing each small window of it by one number. The two usual summaries are the largest value (**max pooling**) and the mean (**average pooling**). Both slide a window of side $k$ over the map every $s$ positions, separately for every channel, and have no weights at all. The window of 2 and the stride of 2 halve a map's sides; the original convolutional networks of LeCun and colleagues subsampled this way, and the 2012 network of Krizhevsky, Sutskever and Hinton used overlapping windows of 3 at stride 2.

Max pooling says "the pattern was somewhere in this window", and keeps how strongly. Average pooling says how much of the window was covered. Here they are on a 4 by 4 map, with their backward passes, which is what decides how a network can learn through them. For a mean the derivative is plain: each of the $k^2$ inputs moves the output by $1/k^2$ of its own change, so the gradient arriving at the output is shared equally. For a maximum, a small change of any input other than the winner changes nothing, and a change of the winner changes the output one for one; the gradient therefore goes to the winner alone, and the other inputs of the window receive zero. A window with equal maxima has no unique winner, and the convention is the first in row-major order (`argmax` of the book's pseudocode does the same). When windows overlap, one input can win several of them, and its gradient is the sum of all they send it, which is the rule of Volume III's chapter 5 for a value used more than once.

<!-- worked: cnns.pooling -->
The input X is 4 by 4, the gradient arriving at the 2 by 2 output is dy = [[1, 2], [3, 4]]. A window with equal maxima sends its gradient to the first of them in row-major order (the top-left window has 2 twice).

```text
X =
 2  2  0  1
 1  0  3  5
 0  6  2  2
 3  1  7  4

max pool 2x2, stride 2: y =
 2  5
 6  7
winning positions (row:column):
0:0  1:3
2:1  3:2 
dx =
 1  0  0  0
 0  0  0  2
 0  3  0  0
 0  0  4  0

average pool 2x2, stride 2: y =
  1.25   2.25
  2.50   3.75
dx = dy / 4 over each window =
  0.25   0.25   0.50   0.50
  0.25   0.25   0.50   0.50
  0.75   0.75   1.00   1.00
  0.75   0.75   1.00   1.00
```

Overlapping windows: X' is 5 by 5, window 3 by 3, stride 2, so the four windows share the centre pixel; it is the maximum of all four.

```text
X' =
 1  2  0  1  2
 0  3  1  0  1
 2  1  9  1  0
 1  0  2  3  1
 0  1  1  2  4

max pool 3x3, stride 2: y =
 9  9
 9  9
dx for dy = [[1, 2], [3, 4]] =
 0  0  0  0  0
 0  0  0  0  0
 0  0 10  0  0
 0  0  0  0  0
 0  0  0  0  0
```

The centre receives 10 = 1 + 2 + 3 + 4: a pixel that wins several windows gets the sum of their gradients.

Global average pooling of the map with 2 channels of 2 by 3 values [[[1, 2, 3], [4, 5, 6]], [[0, 0, 3], [3, 0, 0]]]: y = [3.5, 1.0]. With the gradient [6.0, -3.0] arriving, every position of channel c receives dy[c] / 6: [[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], [[-0.5, -0.5, -0.5], [-0.5, -0.5, -0.5]]].
<!-- /worked -->

The routine below is the max pool with its record of winners, and its backward pass, in the book's pseudocode (`out_size` is the function of Chapter 2). The winner is stored as a position, not recomputed, because the backward pass runs after the forward one and the forward pass already knows.

```text
function maxpool2d(x: array<real>[C, H, W], k: int, s: int)
        -> (array<real>[C, H_o, W_o], array<int>[C, H_o, W_o], array<int>[C, H_o, W_o]):
    H_o = out_size(H, k, s, 0, 1)
    W_o = out_size(W, k, s, 0, 1)
    y = zeros(real, [C, H_o, W_o])
    win_row = zeros(int, [C, H_o, W_o])
    win_col = zeros(int, [C, H_o, W_o])
    for c in 0 .. C:
        for i in 0 .. H_o:
            for j in 0 .. W_o:
                best = -infinity
                for a in 0 .. k:
                    for b in 0 .. k:
                        if x[c, i * s + a, j * s + b] > best:      # strict: the first maximum keeps a tie
                            best = x[c, i * s + a, j * s + b]
                            win_row[c, i, j] = i * s + a
                            win_col[c, i, j] = j * s + b
                y[c, i, j] = best
    return (y, win_row, win_col)

function maxpool2d_backward(dy: array<real>[C, H_o, W_o], win_row: array<int>[C, H_o, W_o],
                            win_col: array<int>[C, H_o, W_o], H: int, W: int) -> array<real>[C, H, W]:
    dx = zeros(real, [C, H, W])
    for c in 0 .. C:
        for i in 0 .. H_o:
            for j in 0 .. W_o:
                r = win_row[c, i, j]
                q = win_col[c, i, j]
                dx[c, r, q] = dx[c, r, q] + dy[c, i, j]            # += : a winner of several windows
    return dx
```

One more pooling is used more than any other, at the very end of a classifier. **Global average pooling** takes the window to be the whole map: a $[C, H, W]$ tensor becomes $C$ numbers, the mean of each channel, and its backward pass hands every position of channel $c$ the same share $dy_c/(HW)$ (the last lines of the table above). It replaces the flatten-and-dense step that older networks put there, and the count of weights shows why: a dense layer on the flattened map has $C H W$ inputs per class, global pooling has $C$, and the layer after it does not depend on the picture size at all.

The kernel in `crates/kernels/wgsl/maxpool2d.wgsl` does what the routine does, with one thread per output and the winner's flat input index stored beside the output as `argmax`, so that `maxpool2d_dx.wgsl` can be a gather: one thread per input element, which sums the gradient of every output whose window covers it and whose `argmax` is that element. The rule for which outputs cover an input inverts the window: input row $i$ is covered by the output rows $i_{\mathrm{o}}$ with $\lceil (i + p - k + 1)/s \rceil \le i_{\mathrm{o}} \le \lfloor (i + p)/s \rfloor$ (the same for columns), and commit `3a565d609` is explicit about the one trap in it: the lower bound is negative near the top-left border, and in unsigned arithmetic it wraps and pulls in the whole first row of outputs, which is why the kernel computes it in signed integers. The kernel's tests (`crates/gradcheck/tests/it/maxpool2d_kernels.rs`) are checks of the kind this chapter applies to the toy: a single hot pixel must land at the right outputs, the backward pass must be the adjoint of the forward pass with the winners frozen, a sweep over awkward sizes, kernels, strides and paddings must agree with a plain host loop, and a window that sees only padding must send nothing. The block of `crates/vision` built on it, `MaxPool` with its `PoolSpec`, has its own test (`maxpool_backward_routes_gradient_to_the_winning_tap` in `crates/vision/tests/imaging_blocks.rs`) that checks the gradient against a host argmax and not against the block's own `argmax` buffer. The same block with a window of 5, stride 1 and padding 2 does not shrink the map at all: three of them chained (the SPPF block of the YOLO family) see windows of 5, 9 and 13 pixels, because each adds $k - 1 = 4$ to the field, and pooling is then a way to widen the receptive field without any weights.

The average has its own block, `AvgPool`, whose kernel is adaptive: it is given the output size and covers output row $i_{\mathrm{o}}$ with the input rows from $\lfloor i_{\mathrm{o}}H/H_o \rfloor$ up to but not including $\lceil (i_{\mathrm{o}}+1)H/H_o \rceil$, which for $H_o$ dividing $H$ is the plain box pool and for $H_o = 1$ is global average pooling. `AvgPool::half` asserts an even map so that the adaptive rule cannot silently overlap its windows (`crates/vision/src/blocks.rs`).

## A network small enough to print

Pooling, the convolution of the previous chapter and the layers of Volume III are enough to build the toy classifier. The data come from a stated rule, so that anyone can regenerate them.

<!-- worked: cnns.toy_set -->
Image n has class n mod 3 (0 stripes, 1 checkerboard, 2 blobs); n = 0 .. 59 is the training set (20 of each class) and n = 60 .. 149 the test set (30 of each). One generator for all of them: the book's SplitMix64, seed 0xb48e91712679b174 (13010496288682783092), the name cnns.toy_data. Draws for one image, in order:

- stripes: `integer(2)` (0 for stripes that vary along the columns, 1 along the rows), `integer(2) + 1` for the stripe width, `integer(2 * width)` for the phase; pixel = 1 when floor((t + phase) / width) is even, t being the column or the row;
- checkerboard: `integer(2) + 1` for the cell size, `integer(2 * size)` for the row offset oy, `integer(2 * size)` for the column offset ox; pixel = 1 when floor((i + oy) / size) + floor((j + ox) / size) is even;
- blobs: `integer(2) + 2` bumps, then for each bump the row centre `uniform() * 7` and the column centre `uniform() * 7`; pixel = the largest over bumps of exp(-((i - ci)^2 + (j - cj)^2) / (2 * 1.2^2));
- then, for every class, the noise: 64 draws of `normal()` in row-major order, each times 0.4 and added to the pixel.

The images n = 0, 1, 2 (the noisy images themselves, each pixel drawn as # when it exceeds 0.5 and . otherwise; pixel values range over [-1.75, 2.66]):

```text
stripes    checker    blobs   
#.#.#.#.   #.#.#.#.   ....#...
#.#...#.   .#.###.#   .#...##.
#.#.#.#.   #.#...#.   .....#.#
#.#.##..   ...#.#.#   .......#
#.#.#.#.   #.#.###.   #..##...
#.#.#.#.   .#.#.#.#   #####..#
#.#.#.#.   #.#.#.#.   ###..#..
#.#...#.   .#.#.#.#   .#......
```
<!-- /worked -->

The network is two convolutions, a pooling after the first, global average pooling after the second and a linear layer to the three classes, with a ReLU after each convolution:

![The toy network: the 1 by 8 by 8 image, a 3 by 3 convolution with 4 filters giving 4 by 8 by 8, ReLU and 2 by 2 max pool giving 4 by 4 by 4, a 3 by 3 convolution with 6 filters giving 6 by 4 by 4, ReLU and global average pooling giving 6 values, a linear layer giving 3 logits and the softmax, with the parameter count of each layer and the receptive field after each convolution and the pool.](img/toy-cnn.svg)

Everything about its size can be counted before it runs, with the formulas of Chapter 2: the parameters of a convolution are $C_{\mathrm{out}}C_{\mathrm{in}}k^2 + C_{\mathrm{out}}$, and its work is $H_{\mathrm{o}}W_{\mathrm{o}}C_{\mathrm{out}}C_{\mathrm{in}}k^2$ multiply-adds.

<!-- worked: cnns.toy_layers -->
Parameters, multiply-adds (MACs) and FLOPs (2 per multiply-add) of one image through each layer of the toy network; ReLU, pooling and the softmax add comparisons and a few additions that are not counted as multiply-adds.

| layer | input to output | parameters | multiply-adds | FLOPs |
|---|---|---:|---:|---:|
| conv 3x3, pad 1 | [1, 8, 8] to [4, 8, 8] | 40 | 2,304 | 4,608 |
| ReLU | [4, 8, 8] to [4, 8, 8] | 0 | 0 | 0 |
| max pool 2x2, stride 2 | [4, 8, 8] to [4, 4, 4] | 0 | 0 | 0 |
| conv 3x3, pad 1 | [4, 4, 4] to [6, 4, 4] | 222 | 3,456 | 6,912 |
| ReLU | [6, 4, 4] to [6, 4, 4] | 0 | 0 | 0 |
| global average pool | [6, 4, 4] to [6] | 0 | 0 | 0 |
| linear | [6] to [3] | 21 | 18 | 36 |

Total: 283 parameters and 5,778 multiply-adds (11,556 FLOPs) per image. Receptive field along one axis by the recursion r = r + d (k - 1) j, j = j s, from r = j = 1: conv 3x3: r = 3, j = 1; pool 2x2 stride 2: r = 4, j = 2; conv 3x3: r = 8, j = 2. Flatten and a linear layer on the 6 * 4 * 4 = 96 values of the second convolution's output would need 291 parameters where global average pooling needs 21.
<!-- /worked -->

The receptive field in the last line is the recursion (4) of Chapter 2 and it is the design: after the pooling the second convolution has a $3 \times 3$ window on a map whose cells are 2 pixels apart, so one of its outputs reads $r = 8$ input pixels along each axis, the whole picture. The classifier therefore never needs a large kernel to see the whole board; it gets there with two small ones and a pooling. And the count of parameters shows the same point as the opening of Chapter 2 from the other side. The two convolutions have 40 and 222 weights and biases, and nothing about them depends on the picture being 8 by 8.

## Training it, and checking the backward pass

The initialization is the He rule of Volume III, Chapter 9, for the convolution layers: a normal draw times $\sqrt{2/(9C_{\mathrm{in}})}$, since each output of a $3 \times 3$ convolution sums $9C_{\mathrm{in}}$ products. All draws come from one generator, the book's SplitMix64 with the name `cnns.toy_init`, in this order: the 36 weights of the first convolution (filter by filter, taps in row-major order), the 216 weights of the second (filter, input channel, kernel row, kernel column), then the 18 weights of the linear layer (a row per class) with the factor $\sqrt{1/6}$. Every bias starts at 0. The loss is the mean over the images of the cross-entropy of Volume III, Chapter 3.

The backward pass is that chapter's too: $d\mathbf{z} = \mathbf{p} - \mathbf{y}$ at the logits, the linear layer's rule, global average pooling spreading $dy_c/16$ over the 16 positions, the ReLU mask, Chapter 2's two matrix products for the second convolution (the weight gradient and the input gradient), the routing of the max pool, another ReLU mask, and the weight gradient of the first convolution (whose input gradient is not needed). Before a long run is trusted it is checked, by the method of Volume III's chapter on gradient checking: every parameter in turn is moved by $\pm h$ and the central difference compared with the analytic value.

<!-- worked: cnns.gradient_check -->
The initialized network (cnns.toy_init, seed 0xe68ba4d2e700a615 (16612552876211873301)) on the training images n = 0 .. 5 (two of each class): mean cross-entropy, analytic gradient from the backward pass of the chapter against the central difference (L(w + h) - L(w - h)) / (2h) with h = 1e-06 in 64-bit arithmetic, one weight at a time. The gate of Volume III, Chapter 6 is |analytic - numeric| <= 1e-7 + 1e-5 max(|analytic|, |numeric|).

| parameters | count | largest absolute difference |
|---|---:|---:|
| W1 | 36 | below 1e-7 |
| b1 | 4 | below 1e-7 |
| W2 | 216 | below 1e-7 |
| b2 | 6 | below 1e-7 |
| Wl | 18 | below 1e-7 |
| bl | 3 | below 1e-7 |

283 of 283 parameters pass the gate (0 fail).
<!-- /worked -->

The routing of the pool is what makes the check delicate in principle: a perturbation that changes which input wins a window, or flips a ReLU, makes the loss non-differentiable at that point. At this initialization and with $h = 10^{-6}$ no parameter is near such a kink, and the check passes everywhere. The gradient check of a trained network could not promise that, which is one reason the check is made at the start.

Now the training. The optimizer is Adam of Volume III, Chapter 7, with the settings named in the table, on all 60 images at once. A trained number is chaotic in principle (each update depends on the last), so the table prints only the digits that survive the nudges of the book's stable-digits rule.

<!-- worked: cnns.training -->
The toy network trained on the 60 training images by full-batch Adam (beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-8, learning rate 0.02, bias correction as in Volume III, Chapter 7), 100 updates, mean cross-entropy, from the initialization of cnns.toy_init. Training loss and accuracy on all 60 images after t updates. Sums over taps run in the order (input channel, kernel row, kernel column), over positions row-major and over images in index order. A number that comes from a chaotic run is printed with only the digits that are the same in the run and in four copies of it whose first weight is changed by +1e-12, -1e-12, +3e-12 and -3e-12: in full if all agree on it, as 'about x' with the leading digits on which all agree, or as 'unstable' when not even one digit survives; the other digits cannot be reproduced in another implementation.

| updates t | training loss | training accuracy |
|---|---:|---:|
| 0 | 1.1668 | 0.3333 |
| 10 | 0.9769 | 0.5500 |
| 25 | 0.5266 | 0.8000 |
| 50 | 0.0329 | 1.0000 |
| 100 | 0.0022 | 1.0000 |

Test set (90 images) after 100 updates: cross-entropy 0.0997, accuracy 0.9556.
<!-- /worked -->

The loss falls from the $\ln 3 = 1.0986$ of a network that knows nothing to a fraction of a percent, and the network ends with all 60 training images right. On the 90 test images it is right on 86, an accuracy of 0.9556, with a cross-entropy of 0.0997, about forty-five times the training loss. That gap is the ordinary one of a 283-parameter model fitted to 60 noisy images, and it is worth keeping in view: the later chapters count their training in millions of images because a network this small already memorizes what it is given.

## Shifting the picture

The opening claimed that a layer built from convolutions respects shifts and pooling turns respect into indifference. That can be measured. Take the circular shift of an image (the columns wrap around), apply the network, and compare. The first statement is about the convolution alone. Write $\mathrm{roll}_s$ for the shift to the right by $s$ columns and $\mathrm{conv}(\mathbf{x}; \mathbf{w})$ for the sliding sum of Chapter 2 with the weights $\mathbf{w}$. A convolution with circular padding satisfies $\mathrm{conv}(\mathrm{roll}_s\,\mathbf{x}; \mathbf{w}) = \mathrm{roll}_s\,\mathrm{conv}(\mathbf{x}; \mathbf{w})$ exactly, because it applies the same weights at every position and the shifted picture presents the same neighbourhoods at shifted places; this is **equivariance**, a statement about maps and not about answers. ReLU acts on each value alone and keeps it. Max pooling with stride 2 does not: its windows sit at even positions, so a shift by an even number of pixels carries the windows onto windows and the pooled map is shifted by half as much, while a shift by an odd number puts the windows on different pairs of pixels and the pooled map is no shift of the old one at all. Global average pooling then sums over all positions, and a sum does not care where its terms sit, so for the shifts that survive the pool the logits do not change.

<!-- worked: cnns.shift_equivariance -->
The initialized network (cnns.toy_init) with circular padding in both convolutions (indices wrap around, so no border), on the 90 test images x, each shifted to the right by s columns with wrap-around (roll). Largest absolute difference over all images and values between: (a) the ReLU map of the shifted image and the ReLU map of x shifted by s; (b) the pooled map of the shifted image and the pooled map of x shifted by s / 2 columns (for odd s, for each image separately, the smaller of the two differences against x shifted by (s - 1) / 2 and by (s + 1) / 2 columns; the error shown is the largest over the images); (c) the logits of the shifted image and of x.

| shift s | (a) after conv and ReLU | (b) after pooling | (c) logits |
|---|---:|---:|---:|
| 0 | below 1e-15 | below 1e-15 | below 1e-15 |
| 1 | below 1e-15 | 2.4096 | 0.6083 |
| 2 | below 1e-15 | below 1e-15 | below 1e-15 |
| 3 | below 1e-15 | 2.4096 | 0.6083 |
| 4 | below 1e-15 | below 1e-15 | below 1e-15 |
<!-- /worked -->

The first column is zero for every shift (equivariance, to rounding), the second and the third are zero for even shifts and not for odd ones. The pool halves the resolution of the map, and a stack with stride $2^m$ in total is invariant, in this exact sense, to shifts by multiples of $2^m$ and only approximately to the rest. That is the aliasing that Chapter 1 met in the resampling of pictures, now inside the network.

The trained network, with zero padding as it was trained, is only approximately equivariant at the border, and the measurement shows how much the approximation costs on this data.

<!-- worked: cnns.shift_stability -->
The trained network (zero padding, as trained) on the 90 test images, each shifted to the right by s columns with wrap-around. Share of images whose predicted class is the same as for the unshifted image, and the mean over images of the largest change of a class probability. A number that comes from a chaotic run is printed with only the digits that are the same in the run and in four copies of it whose first weight is changed by +1e-12, -1e-12, +3e-12 and -3e-12: in full if all agree on it, as 'about x' with the leading digits on which all agree, or as 'unstable' when not even one digit survives; the other digits cannot be reproduced in another implementation.

| shift s | same predicted class | mean largest probability change |
|---|---:|---:|
| 0 | 1.0000 | 0.0000 |
| 1 | 0.9556 | 0.0488 |
| 2 | 0.9556 | 0.0485 |
| 3 | 0.9444 | 0.0652 |
| 4 | 0.9444 | 0.0747 |
<!-- /worked -->

The predicted class of a test image is unchanged under a shift in at least 94 images of every 100, and the largest class probability moves by about five to seven hundredths on average. The network was never shown a shifted picture; its stability is the convolution and the pooling, not the training, and the gap from perfect is the border and the odd shifts. Neither is a flaw to explain away. A pattern a pixel off a border is a different picture to a network with zero padding, and a pool of stride 2 has no way to treat both parities of position alike.

## When a stack gets deep

Two convolutions are enough for the toy. A photograph of a thousand classes is not, and when networks were first made much deeper a puzzle appeared that was not overfitting. He and colleagues reported that a plain stack of 56 layers had a higher error on the training set than a plain stack of 20 on the same data, and that a plain 34-layer network did worse than an 18-layer one at the same task. A deeper model contains the shallower one (let the extra layers compute the identity), so the optimizer was failing to find a solution that existed. The reason sits in the arithmetic of Volume III's chapter on training dynamics: through a stack of layers every gradient is a product of Jacobians, and a product of $d$ factors each a little under 1 is exponentially small.

The toy can show the mechanism, with two caveats stated first. The stack here is narrow (4 channels), it has no normalization and no biases, and its trained numbers come from one optimizer, 50 updates and 24 images, so what it shows about training is a small experiment and not a result about networks. What it can show cleanly is what is measured at initialization, before any optimizer is involved. The stem and the data are those of the toy; the body is a stack of $d$ convolutions $4 \to 4$ with a ReLU after each (plain), or the same $d$ convolutions arranged in $d/2$ residual blocks $\mathrm{relu}(\mathbf{x} + f(\mathbf{x}))$ with $f$ two convolutions and a ReLU between them, the block of the next section. The He rule of Volume III, Chapter 9 sets the weights; in the residual blocks the second convolution is started at 0.3 times that, the small-branch rule of Chapter 12.

<!-- worked: cnns.depth_signal -->
Stem (3x3 conv, 4 filters, ReLU, 2x2 max pool, from the 8 by 8 image to 4 by 4 maps), then the body, then global average pooling and a linear layer (4 to 3, no bias). The body is d convolutions 4 to 4 of 3 by 3 with padding 1 and ReLU after each (plain), or d / 2 residual blocks relu(x + conv(relu(conv(x)))) with the same d convolutions. No biases, no normalization. Initialization: one generator per seed, the book's SplitMix64 with the names cnns.depth_init.0, cnns.depth_init.1, cnns.depth_init.2, cnns.depth_init.3 (seed 0xff8a53c40bbd4cf5 (18413622127933345013); seed 0xb4f99d548fe87838 (13040627182545172536); seed 0x1c619a2c829373fc (2045085221762855932); seed 0x84f054f0558b375b (9579249798621116251)); draws in order the stem (4 filters of 9 taps), then the d body convolutions (4 filters of 36 taps each, in layer order), then the read-out (3 rows of 4); the same draws for the plain and the residual stack of one depth and seed; standard deviation sqrt(2 / 9) for the stem and sqrt(2 / 36) for the body, times 0.3 for the second convolution of each residual block, 0.5 for the read-out. Sums over taps run in the order (input channel, kernel row, kernel column), over positions row-major and over images in index order. Measured at initialization, for the seeds 0 to 3, on the first 9 training images (n = 0 .. 8); each entry is the geometric mean over the four seeds of a ratio formed within one seed. Columns 2 and 3: the root mean square of the stream (the post-ReLU activation; its root mean square is taken over all elements of all those images pooled together) after the last convolution (plain) or block (residual) divided by the same quantity after the stem's ReLU and pooling. Column 4: the factor 0.8333^d, where 0.6944 is the share of the 9 taps of a 3 by 3 window with padding 1 that land on a real pixel of a 4 by 4 map, averaged over positions (on 8 by 8 it is 0.8403, on 28 by 28 it is 0.9529). Columns 5 and 6: the Frobenius norm of the weight gradient of the first body convolution (the one farthest from the loss) of the mean cross-entropy of the images.

| d (convolutions) | plain: stream | residual: stream | plain: border alone | plain: first-layer gradient | residual: first-layer gradient |
|---|---:|---:|---:|---:|---:|
| 2 | 6.66e-01 | 1.07e+00 | 6.94e-01 | 5.13e-01 | 1.46e-01 |
| 4 | 3.70e-01 | 1.11e+00 | 4.82e-01 | 1.53e-01 | 1.58e-01 |
| 8 | 1.54e-01 | 1.30e+00 | 2.33e-01 | 6.66e-02 | 2.01e-01 |
| 12 | 6.19e-02 | 1.22e+00 | 1.12e-01 | 5.31e-02 | 2.05e-01 |
| 16 | 1.06e-02 | 1.23e+00 | 5.41e-02 | 1.17e-02 | 1.35e-01 |
| 24 | 1.71e-03 | 1.33e+00 | 1.26e-02 | 2.88e-03 | 2.18e-01 |
| 32 | 2.94e-04 | 1.59e+00 | 2.93e-03 | 3.79e-04 | 1.80e-01 |
| 48 | 1.91e-06 | 2.07e+00 | 1.58e-04 | 9.32e-06 | 2.85e-01 |
<!-- /worked -->

![The norm of the weight gradient reaching the first body convolution at initialization against the number d of convolutions, on a logarithmic axis: the plain stack falls from 0.5 at d = 2 to about 1e-5 at d = 48, the residual stack stays between 0.1 and 0.3.](img/depth-signal.svg)

The stream shrinks with depth in the plain stack, by a factor of five hundred thousand over 48 convolutions, and the gradient reaching the first body convolution with it; in the residual stack the stream stays of order one and the gradient between 0.1 and 0.3. Part of the plain stack's shrinkage is a fact of the zero padding on small maps that the He rule, which assumes every window is full, does not know about. On a 4 by 4 map only 0.6944 of the taps of a $3 \times 3$ window with padding 1 land on a real pixel, so the variance of each output falls by that factor before any ReLU acts, and the column "border alone" gives the stream that factor alone would leave. It explains a part of the decay (the measured stream is below it at every depth of the table) and the rest comes from the narrow width and the ReLU. It is also a reason that the effect is stronger here than on the large maps of a photograph, where the share is 0.9529 on a map of 28 by 28.

Does the fall of the signal show in training? The table trains the 4-convolution and 24-convolution versions of both stacks from four starting points each, 50 updates of Adam at a rate of 0.01.

<!-- worked: cnns.depth_table -->
The stacks, seeds and initialization of the previous table, trained on the first 24 training images (8 of each class) by Adam (0.9, 0.999, 1e-8), learning rate 0.01, 50 full-batch updates of the mean cross-entropy. Each cell is the training loss after 50 updates and, in brackets, the training accuracy. A number that comes from a chaotic run is printed with only the digits that are the same in the run and in four copies of it whose first weight is changed by +1e-12, -1e-12, +3e-12 and -3e-12: in full if all agree on it, as 'about x' with the leading digits on which all agree, or as 'unstable' when not even one digit survives; the other digits cannot be reproduced in another implementation.

| stack | seed 0 | seed 1 | seed 2 | seed 3 |
|---|---:|---:|---:|---:|
| 4 convolutions, plain | 0.682 (0.67) | 0.003 (1.00) | 0.148 (0.96) | 0.062 (1.00) |
| 4 convolutions, residual, 2 blocks | 1.099 (0.33) | 0.004 (1.00) | 0.001 (1.00) | 0.001 (1.00) |
| 24 convolutions, plain | 0.733 (0.33) | 0.732 (0.33) | 0.732 (0.33) | 0.729 (0.42) |
| 24 convolutions, residual, 12 blocks | 0.678 (0.67) | 0.697 (0.67) | 0.526 (0.67) | 0.496 (0.67) |
<!-- /worked -->

Read the table for what it says and no more. At 4 convolutions neither stack is the better one: seed 0 trains with the plain stack and not with the residual one (it has not left $\ln 3$), and the other three seeds reach accuracy 1.00 or 0.96 with both. At 24 convolutions the four plain stacks have an accuracy of 0.33 or 0.42, which is the accuracy of a coin among three classes, although their loss has come down to 0.73 (below $\ln 3$, so the network has learned something about the probabilities but not how to say which class); the four residual stacks all reach 0.67. The residual stacks are not solved after 50 updates, and the difference between them and the plain stacks at this depth is the evidence of this chapter's toy, nothing more. It agrees with the larger evidence of Volume III's chapter on training dynamics, where the same two stacks of 24 matrices were trained on a synthetic task and the plain ones stayed at the entropy of the labels while their residual twins fell, and with the reports of He and colleagues for photographs.

## The residual block, and the shortcut when shapes change

A **residual network** is built from blocks that compute $\mathbf{y} = \mathrm{relu}(f(\mathbf{x}) + \mathrm{sh}(\mathbf{x}))$, where $f$ is a short stack of convolutions (with normalization) and the shortcut $\mathrm{sh}$ is the identity. The gradient of Volume III's chapter 12 explains the choice. With $\mathbf{u} = f(\mathbf{x}) + \mathbf{x}$ and $d\mathbf{u}$ the gradient $d\mathbf{y}$ masked by the slope of the last ReLU, the identity shortcut gives $d\mathbf{x} = d\mathbf{u} + J_f^\top d\mathbf{u}$, which always contains a path that crosses the block through the addition without meeting a weight or a slope, so the gradient arriving at the first block is the sum over a family of paths one of which is the identity, and the signal of the forward pass has the same path. The block is the basic block of ResNet-18 and ResNet-34: two $3 \times 3$ convolutions, a ReLU between them and a ReLU after the sum. The name **ResNet** (residual network) is given to the whole family of networks of such blocks, stage after stage, each stage halving the map and doubling the channels.

The two sides of the addition must have the same shape, and at the first block of a stage they do not: the stride halves $H$ and $W$ and the channels double. The identity cannot be added. The shortcut then becomes a $1 \times 1$ convolution with the same stride and the new channel count, the **projection shortcut**, which is a weight matrix on each pixel and costs little (131,072 weights for the shortcut from 256 to 512 channels, against 3,538,944 in the two $3 \times 3$ convolutions of that block, 256 to 512 and 512 to 512). In pseudocode, with the normalization left out for the moment:

```text
function residual_block(x: array<real>[C_in, H, W], w1: array<real>[C_out, C_in, 3, 3],
                        w2: array<real>[C_out, C_out, 3, 3], ws: optional array<real>[C_out, C_in, 1, 1],
                        s: int) -> array<real>[C_out, H_o, W_o]:
    h = map(relu, conv2d(x, w1, s, 1, 1, 1))            # 3x3, stride s, padding 1
    f = conv2d(h, w2, 1, 1, 1, 1)                       # 3x3, stride 1, padding 1
    if ws is present:
        shortcut = conv2d(x, ws, s, 0, 1, 1)            # 1x1, the same stride, no padding
    else:
        require s == 1 and C_in == C_out, "an identity shortcut needs the same shape"
        shortcut = x
    return map(relu, f + shortcut)
```

The projection is not the only way to match the shapes, and the choice is not a detail. A $1 \times 1$ convolution of stride 2 reads every second pixel and ignores the others, along each axis. The $3 \times 3$ convolution of stride 2 and padding 1 in the branch is centred on those same pixels, so the two sides of the sum look at the same places. A variant of the shortcut, from He and colleagues' study of training tricks (called ResNet-D), takes a $2 \times 2$ average over each pair of pixels first and applies the $1 \times 1$ convolution to the result.

<!-- worked: cnns.shortcut_alignment -->
Which input columns feed output i of a stride-2 shortcut, along one axis. The 3x3 convolution of the block (stride 2, padding 1) reads columns 2i - 1 .. 2i + 1, centred on column 2i.

| output i | 3x3 stride 2, pad 1 reads | 1x1 stride 2 reads | 2x2 average pool reads | centre of the average |
|---|---:|---:|---:|---:|
| 0 | -1 .. 1 | 0 | 0 .. 1 | 0.5 |
| 1 | 1 .. 3 | 2 | 2 .. 3 | 2.5 |
| 2 | 3 .. 5 | 4 | 4 .. 5 | 4.5 |
| 3 | 5 .. 7 | 6 | 6 .. 7 | 6.5 |

On the ramp x[j] = j (j = 0 .. 7) the strided 1x1 shortcut (weight 1) gives [0, 2, 4, 6] and the average pool gives [0.5, 2.5, 4.5, 6.5]: the same values shifted by half a pixel.
<!-- /worked -->

Both have the right shape. They compute different numbers: the average is centred half a pixel to the right of the pixel the strided $1 \times 1$ reads, and weights trained with one are wrong, quietly, for the other. This is no hypothetical. The SCRFD face detector of `crates/scrfd` is a ResNet whose released weights use the average-pool shortcut; its block (`BasicBlock` in `crates/scrfd/src/model.rs`) says that "both produce the right shape; only one produces the right numbers", commit `1bf23f188` added `AvgPool` to `crates/vision` for exactly this, and the importer asserts at load time that the pool nodes of the released graph are $2 \times 2$ with stride 2 and no padding (`check_structural` in `crates/scrfd/src/import.rs`), since a pool has no weights for the usual weight-by-name matching to notice. Shapes bind; positions do not; only a reference output shows the difference.

A second subtlety of the block is one line of a comment. `crates/arcface/src/model.rs` builds the residual block of the face-recognition network IResNet, in which the normalization comes first, $\mathbf{y} = \mathrm{conv}_2(\mathrm{prelu}(\mathrm{conv}_1(\mathrm{bn}_1(\mathbf{x})))) + \mathbf{x}$, and the block's doc notes that the shortcut reads the block's input, not the output of $\mathrm{bn}_1$: "feeding it `bn1(x)` instead still runs, still produces plausible features, and is wrong". It is a block-level error that no unit test of any single convolution can see, and it is the reason the project compares whole blocks and networks against reference outputs, not just kernels. The ResNet-18 context path of the face-parsing model in `crates/bisenet` (`RESNET18_STAGES`: widths 64, 128, 256, 512 with 2 blocks each, a projection on the first block of stages 2 to 4) is gated by a parity test, `crates/bisenet/tests/parity.rs`, against a reference dump, at a cosine floor of 0.999 on the logits and a per-pixel class agreement above 98 percent (commit `eac63b935` ports the model; the test skips when the reference files are absent).

### Bottlenecks

The two $3 \times 3$ convolutions of a basic block are expensive in wide stages: each has $9C^2$ weights. For deeper networks He and colleagues replaced them by three convolutions, $1 \times 1$ down to a quarter of the width, $3 \times 3$ at that narrow width, $1 \times 1$ back up, the **bottleneck block**. The $3 \times 3$ convolution, the one that looks at neighbours, now works on a quarter of the channels, and the $1 \times 1$ convolutions are channel mixing only. The counts:

<!-- worked: cnns.block_costs -->
Convolution weights (no bias, no normalization parameters) and multiply-adds of blocks on 56 by 56 maps (stride 1; the projection shortcut has a 28 by 28 output).

| block | parameters | multiply-adds | FLOPs |
|---|---:|---:|---:|
| basic block, 256 to 256 (3x3, 3x3) | 1,179,648 | 3,699,376,128 | 7,398,752,256 |
| bottleneck, 256 to 256 (1x1 to 64, 3x3, 1x1 to 256) | 69,632 | 218,365,952 | 436,731,904 |
| basic block, 64 to 64 (3x3, 3x3) | 73,728 | 231,211,008 | 462,422,016 |
| projection shortcut 1x1, stride 2, 256 to 512 (28x28 output) | 131,072 | 102,760,448 | 205,520,896 |
| identity shortcut | 0 | 0 | 0 |

The bottleneck holds 16.94 times fewer weights than the basic block of the same width (1,179,648 against 69,632), and the same ratio of multiply-adds (16.94).
<!-- /worked -->

At 256 channels the bottleneck block holds 16.94 times fewer weights and does 16.94 times fewer multiply-adds than the basic block of the same width, and it costs about as much as a basic block at a quarter of the width (69,632 weights against 73,728). The saving is what lets a network be 50 layers deep for a little more than the arithmetic of a plain one of 34. Whole networks are counted the same way, and the totals of a stage-by-stage count are a check on understanding:

<!-- worked: cnns.resnet_counts -->
Parameters and multiply-adds of ResNet-18 (basic blocks) and ResNet-50 (bottleneck blocks) on a 3 by 224 by 224 image, 1000 classes. The stem is followed by a 3x3 max pool of stride 2 (112 to 56). The first block of stages 2 to 4 has stride 2 in its 3x3 convolution and a 1x1 projection shortcut, as does the first block of stage 1 of ResNet-50 (64 to 256 channels); parameters count every convolution weight, two numbers (gain, shift) per channel of every batch normalization, and the classifier weights and bias; multiply-adds count convolutions and the classifier.

| network | part | parameters | multiply-adds |
|---|---|---:|---:|
| ResNet-18 | stem 7x7, stride 2 (112 x 112) | 9,536 | 118,013,952 |
|  | stage 1: 2 blocks, 64 x 56 x 56 | 147,968 | 462,422,016 |
|  | stage 2: 2 blocks, 128 x 28 x 28 | 525,568 | 411,041,792 |
|  | stage 3: 2 blocks, 256 x 14 x 14 | 2,099,712 | 411,041,792 |
|  | stage 4: 2 blocks, 512 x 7 x 7 | 8,393,728 | 411,041,792 |
|  | classifier (global average pool, linear) | 513,000 | 512,000 |
|  | total | 11,689,512 | 1,814,073,344 |
| ResNet-50 | stem 7x7, stride 2 (112 x 112) | 9,536 | 118,013,952 |
|  | stage 1: 3 blocks, 256 x 56 x 56 | 215,808 | 667,942,912 |
|  | stage 2: 4 blocks, 512 x 28 x 28 | 1,219,584 | 1,027,604,480 |
|  | stage 3: 6 blocks, 1024 x 14 x 14 | 7,098,368 | 1,464,336,384 |
|  | stage 4: 3 blocks, 2048 x 7 x 7 | 14,964,736 | 809,238,528 |
|  | classifier (global average pool, linear) | 2,049,000 | 2,048,000 |
|  | total | 25,557,032 | 4,089,184,256 |
<!-- /worked -->

Two observations from the table. The classifier of ResNet-50 holds 2,049,000 of its 25,557,032 weights, about one weight in twelve, on the last 2,048 numbers produced by global average pooling; the 7 by 7 maps of its last stage hold the largest block of the convolutional weights (14,964,736) but run at the smallest resolution, so the multiply-adds are spread much more evenly than the weights (stage 3 has the most, 1,464,336,384). And ResNet-18's first stage does 462,422,016 multiply-adds for 147,968 weights, three thousand uses for each weight, against the classifier's one: the early, large maps are where a convolutional network spends its arithmetic, and the late, small ones are where it keeps its parameters.

Brain's crate for these blocks, `crates/vision`, has the residual add in `Bottleneck` and the stage machinery elsewhere. Its `Bottleneck` is the one of the YOLO family's C2f block: two $3 \times 3$ convolutions, each with its normalization and SiLU, and an optional shortcut added after the second (taken only when the channels match, `c_in == c_out && shortcut`) with nothing after the sum: a basic block of its own kind and not the three-convolution bottleneck above; the name is shared, the block is not. The ResNet-18 basic block and its projection shortcut live in the models that need them (`crates/bisenet`, `crates/scrfd`), composed from `Conv` units: a convolution carries its normalization and activation, with the activation placed before the add and never after, so the post-add ReLU is a unit of its own (the comment on `relu` in `crates/bisenet/src/model.rs`). The gradient of the add is the one of Volume III's chapter 5: `Bottleneck::backward` passes `d_out` to the branch and, when the shortcut is on, adds it into `d_in`.

A last property links this section to the next. At inference an identity shortcut is a convolution, one whose kernel is 1 at the centre tap of each channel's own filter and 0 elsewhere, so a residual sum $\mathrm{conv}_3(x) + \mathrm{conv}_1(x) + x$ with no ReLU between its parts is one $3 \times 3$ convolution with the three kernels added. That is the re-parameterization of the ZipDepth blocks in `crates/zipdepth/src/fuse.rs`, tested against an independent convolution for five configurations (residual, plain, with a change of channels, grouped, depthwise), and it carries a trap that its test pins: the identity has no normalization of its own in that network, so it adds a kernel and no bias, and adding one is silent (`identity_branch_adds_no_bias`).

## Batch normalization inside a convolutional block

Every convolution of a ResNet is followed by a normalization, and in a convolutional network the group of Volume III's chapter 10 is a channel: the mean and variance are taken over the batch and over every position, $N\cdot H\cdot W$ values per channel, with one gain $\gamma_j$ and one shift $\beta_j$ per channel. For an image batch that is thousands of values per statistic (the toy's 60 images give 3,840 per channel), so the estimate is steady and the bias of the divisor $B$ is negligible. In eval mode the layer uses the running statistics of that chapter and is a fixed affine map of each channel.

A fixed affine map after a convolution is one more convolution. For channel $j$ with weights $\mathbf{w}_j$, bias $b_j$, running mean $\mu_j$, running variance $\sigma_j^2$, gain $\gamma_j$ and shift $\beta_j$:

$$
\begin{aligned}
\mathrm{sc}_j &= \frac{\gamma_j}{\sqrt{\sigma_j^2 + \epsilon_{\mathrm{norm}}}}, \\
y_j &= \mathrm{sc}_j\,\bigl(\mathrm{conv}(\mathbf{x}; \mathbf{w}_j) + b_j - \mu_j\bigr) + \beta_j \\
&= \mathrm{conv}(\mathbf{x}; \mathrm{sc}_j\,\mathbf{w}_j) + \bigl(\mathrm{sc}_j(b_j - \mu_j) + \beta_j\bigr),
\end{aligned} \tag{1}
$$

where the second form moves the scalar $\mathrm{sc}_j$ into the kernel because the sliding sum is linear in its weights: $\mathbf{w}'_j = \mathrm{sc}_j\,\mathbf{w}_j$ and $b'_j = \mathrm{sc}_j(b_j - \mu_j) + \beta_j$. This is the folding of Volume III, Chapter 10 with the scale named $\mathrm{sc}_j$ (the stride of Chapter 2 is $s$), one per output channel. The routine is two lines per channel:

```text
function fold_bn_conv(w: array<real>[C_out, C_in, k, k], b: array<real>[C_out],
                      gamma: array<real>[C_out], beta: array<real>[C_out],
                      mu: array<real>[C_out], var: array<real>[C_out], eps: real)
        -> (array<real>[C_out, C_in, k, k], array<real>[C_out]):
    require eps > 0.0, "the floor under the square root must be positive"
    w_f = zeros(real, [C_out, C_in, k, k])
    b_f = zeros(real, [C_out])
    for o in 0 .. C_out:
        sc = gamma[o] / sqrt(var[o] + eps)
        w_f[o, :, :, :] = sc * w[o, :, :, :]
        b_f[o] = sc * (b[o] - mu[o]) + beta[o]
    return (w_f, b_f)
```

To test it, take the first convolution of the toy at its initialization with a bias, attach a batch normalization with chosen gains and shifts whose running statistics are those of the 60 training images, and compare the two ways to compute the layer on the 90 test images:

<!-- worked: cnns.bn_fold -->
The first convolution of the initialized toy network (cnns.toy_init: 4 filters of 3 by 3 on the 8 by 8 image, padding 1) with the biases b = [0.05, -0.05, 0.1, 0.0], followed by batch normalization with gain gamma = [1.5, 0.5, 1.0, 2.0], shift beta = [0.1, -0.3, 0.0, 0.2], epsilon = 1e-05. The running mean mu and variance sigma2 (divisor N H W) of each channel are those of the convolution output over the 60 training images (3,840 values per channel, summed in image order, then row-major). Folded: s = gamma / sqrt(sigma2 + epsilon), w' = s w, b' = s (b - mu) + beta.

| channel | mu | sigma2 | s | folded bias b' |
|---|---:|---:|---:|---:|
| 0 | 0.1481 | 0.4848 | 2.1543 | -0.1114 |
| 1 | -0.0347 | 0.3675 | 0.8248 | -0.3126 |
| 2 | 0.0418 | 0.6592 | 1.2316 | 0.0716 |
| 3 | 0.3231 | 0.2355 | 4.1215 | -1.1316 |

Over the 90 test images and all 4 * 64 outputs of each, the largest difference between the convolution followed by batch normalization and the single folded convolution is below 1e-13 in 64-bit arithmetic (criterion 1e-13) and below 1e-5 when the weights, the image, every product and every running sum are rounded to 32 bits (criterion 1e-5; the printed bound does not depend on the order of the sums).
<!-- /worked -->

The two agree to rounding, in 64 bits and, with weights, image, products and sums rounded to 32 bits, to within $10^{-5}$ (the bound the table states). The criterion, not the last digit, is what a test should assert: a fold that puts the variance where the standard deviation belongs is wrong in the first digit, and one whose $\epsilon$ is $10^{-3}$ where the layer used $10^{-5}$ is wrong by about 0.2 percent of the scale of a channel of variance 0.25, orders of magnitude above these tolerances, so both are found; a fold that is right is right to the rounding of the format. Note also what the second form of (1) does not need: any change of the network's structure. The kernel is the same size, so every shape and every count of this chapter holds, and the normalization's cost, three FLOPs and one pass over the map per output, goes to zero.

The fold applies to a normalization that comes after the convolution. When the order is the other way, normalization first (the pre-activation arrangement the IResNet block above uses), the same algebra works for the interior and fails at the border, because the padding is zeros in the normalized domain and not in the input's:

<!-- worked: cnns.bn_before_conv -->
Batch normalization of a single channel (mu = 0.4, sigma2 = 0.2, gamma = 1.2, beta = 0.3, epsilon = 1e-05, so s = 2.6832 and the shift of the affine map is t = beta - mu s = -0.7733) applied to training image 0 and then followed by the four 3 by 3 filters of the initialized network with padding 1; against one convolution with the filters scaled by s and the bias t * (sum of the 9 taps). Largest difference at each kind of output position.

| output positions | largest difference |
|---|---:|
| interior (36 positions) | below 1e-13 |
| edge, not corner (24) | 1.2533 |
| corner (4) | 1.3772 |
<!-- /worked -->

At the 36 interior positions the folded convolution reproduces the normalized one to rounding; at the 28 positions along the border it is wrong by more than 1, because a padded tap contributes 0 in the original and the shift $t$ in the folded form. Brain's two face networks come from released graphs whose batch normalizations are folded into the convolutions before them (the module notes of `crates/arcface/src/model.rs` and `crates/scrfd/src/model.rs` are headed "BatchNorm-FOLDED"). SCRFD has no normalization in front of a convolution, and no batch-norm kernel is dispatched. In ArcFace three kinds of normalization survive as nodes of their own: the block-entry `bn1` of every block, which sits in front of a convolution, the final `bn2` and `features`. The code does not say why `bn1` is not folded, but the arithmetic above is a sufficient reason, and a fold that is exact only in the interior is exactly the kind that no per-pixel test on a large image would find.

Brain has two forms of folding, and the difference matters. `Conv`'s eval path in `crates/vision/src/blocks.rs` computes the per-channel scale and shift (`pack_sb`, with `sc = gamma / sqrt(run_var + eps)` and `shift = beta - run_mean * sc`) and hands them to the fused kernel, which applies them to the sliding sum in the epilogue, before the activation, without a trip through memory (Volume IV, Chapter 8); the weights are not touched, and re-entering train mode invalidates the cache. `fold_bn` in `crates/vision/src/fold.rs` is the weight form (1): a pure function over flat slices, used by the exporter to a hardware runtime in `crates/npu` and by the ZipDepth re-parameterization. Commit `c93484b91` moved it into `vision` so that these consumers share one formula and the constant $\epsilon_{\mathrm{norm}} = 10^{-5}$, `BN_EPS`; `pack_sb` reads the same constant, and its comment explains what a second literal would risk: the folded and the fused paths drifting apart "by an epsilon nothing compares". The test in `fold.rs`, `folded_conv_reproduces_conv_then_bn`, is the check of the table above on a case small enough to compute by hand.

## Features at three scales

A classifier needs one answer per image, and the last map of a ResNet, 2048 channels at one thirty-second of the resolution, is where that answer is read. A detector needs an answer per object, and objects come in sizes. Here are the three last stage outputs of a ResNet-50 on a $640 \times 640$ input (the size of the YOLO models of the next chapter), at strides 8, 16 and 32, called $\mathrm{F}_3$, $\mathrm{F}_4$ and $\mathrm{F}_5$ for the exponent of the stride.

<!-- worked: cnns.pyramid_levels -->
The three levels of a ResNet-50 on a 3 by 640 by 640 input: stride 2^level, output shape, the jump and the receptive field along one axis by the recursion of Volume VII, Chapter 2 along the main path of the blocks (their 3x3 convolutions; the identity shortcut adds nothing), and how many cells an object 32 and 256 pixels across spans along one axis.

| level | stride | output shape | jump j | receptive field r | cells across a 32 px object | cells across a 256 px object |
|---|---:|---:|---:|---:|---:|---:|
| 3 | 8 | 512 x 80 x 80 | 8 | 91 | 4.00 | 32.00 |
| 4 | 16 | 1024 x 40 x 40 | 16 | 267 | 2.00 | 16.00 |
| 5 | 32 | 2048 x 20 x 20 | 32 | 427 | 1.00 | 8.00 |
<!-- /worked -->

The levels differ in two ways that pull in opposite directions. $\mathrm{F}_3$ has a cell every 8 pixels, so an object 32 pixels across spans four cells along each axis, enough to place its edges; $\mathrm{F}_5$ has one cell for the whole object, and the cell's position is good to a stride of 32. But the deep level has seen far more: a value of $\mathrm{F}_3$ depends on 91 pixels along each axis and one of $\mathrm{F}_5$ on 427, and it has passed through many more nonlinear layers, so it carries a more abstract description of what is there. The fine level knows where and little about what; the coarse level knows what and little about where. A detector that read only the last map would find large objects and blur small ones, and one that read only the early map would localize well and recognize badly.

The **feature pyramid** is the structure that gives every level both. The top-down path of Lin and colleagues starts at the deepest level, projects it with a $1 \times 1$ convolution to a common width (256 here), doubles its sides by nearest-neighbour upsampling and adds it to the $1 \times 1$ projection of the next finer level, so that the sum has the fine level's resolution and the coarse level's context; a $3 \times 3$ convolution then smooths each sum. Writing $\mathrm{lat}_i$ for the lateral projection, $\mathrm{up}$ for the doubling, $\mathrm{M}_i$ for the merged maps and $\mathrm{P}_i$ for the pyramid maps:

$$
\begin{aligned}
\mathrm{M}_5 &= \mathrm{lat}_5(\mathrm{F}_5), & \mathrm{M}_4 &= \mathrm{lat}_4(\mathrm{F}_4) + \mathrm{up}(\mathrm{M}_5), \\
\mathrm{M}_3 &= \mathrm{lat}_3(\mathrm{F}_3) + \mathrm{up}(\mathrm{M}_4), & \mathrm{P}_i &= \mathrm{smooth}_i(\mathrm{M}_i).
\end{aligned} \tag{2}
$$

The $3 \times 3$ convolution is there because nearest-neighbour doubling repeats each value over a $2 \times 2$ block, an artefact the next filter spreads out. The path of Liu and colleagues adds a bottom-up path: $\mathrm{N}_3 = \mathrm{P}_3$, then $\mathrm{N}_4 = \mathrm{P}_4 + \mathrm{down}(\mathrm{N}_3)$ and $\mathrm{N}_5 = \mathrm{P}_5 + \mathrm{down}(\mathrm{N}_4)$ with $\mathrm{down}$ a $3 \times 3$ convolution of stride 2, so that the fine level's precise localization reaches the coarse maps by a short path and does not have to climb the whole backbone. In pseudocode (the lateral and smoothing weights are arguments; `upsample2` repeats every value of a map into a $2 \times 2$ block, `y[c, i, j] = x[c, i // 2, j // 2]`):

```text
function fpn_top_down(F3: array<real>[C3, H3, W3], F4: array<real>[C4, H4, W4], F5: array<real>[C5, H5, W5],
                      lat3: array<real>[C, C3, 1, 1], lat4: array<real>[C, C4, 1, 1], lat5: array<real>[C, C5, 1, 1],
                      sm3: array<real>[C, C, 3, 3], sm4: array<real>[C, C, 3, 3], sm5: array<real>[C, C, 3, 3])
        -> (array<real>[C, H3, W3], array<real>[C, H4, W4], array<real>[C, H5, W5]):
    require H4 == 2 * H5 and W4 == 2 * W5 and H3 == 2 * H4 and W3 == 2 * W4
    M5 = conv2d(F5, lat5, 1, 0, 1, 1)
    M4 = conv2d(F4, lat4, 1, 0, 1, 1) + upsample2(M5)
    M3 = conv2d(F3, lat3, 1, 0, 1, 1) + upsample2(M4)
    return (conv2d(M3, sm3, 1, 1, 1, 1), conv2d(M4, sm4, 1, 1, 1, 1), conv2d(M5, sm5, 1, 1, 1, 1))
```

![The data flow of a feature pyramid on a 640 by 640 input: the backbone levels F3, F4, F5 of 512, 1024 and 2048 channels at 80 by 80, 40 by 40 and 20 by 20 are each projected by a 1 by 1 convolution to a merged map M of 256 channels, M5 is doubled and added into M4 and M4 into M3, each M gives a pyramid map P by a 3 by 3 convolution, and the bottom-up path copies P3 to N3 and then halves each N by a 3 by 3 convolution of stride 2 and adds the P of the next level.](img/fpn-pan.svg)

Counted for this input:

<!-- worked: cnns.fpn_table -->
Feature pyramid on the ResNet-50 levels F3, F4, F5 of a 3 by 640 by 640 input, 256 channels at every level, parameters with biases. Top-down path: the lateral projection of F5 is M5; each lower level adds the 2x nearest-neighbour upsampling of the merged map M of the level above to its own lateral projection, and a 3x3 convolution of every merged map M gives P. Bottom-up path (N3 = P3, then N4 and N5) halves the map with a strided 3x3 convolution and adds the pyramid level of the same size.

| step | output shape | parameters | multiply-adds |
|---|---|---:|---:|
| lateral 1x1: F5 (2048) to 256  (gives M5) | 256 x 20 x 20 | 524,544 | 209,715,200 |
| smooth 3x3: M5, 256 to 256  (gives P5) | 256 x 20 x 20 | 590,080 | 235,929,600 |
| lateral 1x1: F4 (1024) to 256 | 256 x 40 x 40 | 262,400 | 419,430,400 |
| upsample x2 of M5, add  (gives M4) | 256 x 40 x 40 | 0 | 0 |
| smooth 3x3: M4, 256 to 256  (gives P4) | 256 x 40 x 40 | 590,080 | 943,718,400 |
| lateral 1x1: F3 (512) to 256 | 256 x 80 x 80 | 131,328 | 838,860,800 |
| upsample x2 of M4, add  (gives M3) | 256 x 80 x 80 | 0 | 0 |
| smooth 3x3: M3, 256 to 256  (gives P3) | 256 x 80 x 80 | 590,080 | 3,774,873,600 |
| PAN: downsample 3x3 stride 2 of N3, add P4 (gives N4) | 256 x 40 x 40 | 590,080 | 943,718,400 |
| PAN: downsample 3x3 stride 2 of N4, add P5 (gives N5) | 256 x 20 x 20 | 590,080 | 235,929,600 |

Top-down path: 2,688,512 parameters; with the bottom-up path: 3,868,672; multiply-adds in all: 7,602,176,000.
<!-- /worked -->

The neck adds 3,868,672 weights, small against the backbone's 25 million, and 7.6 billion multiply-adds, which is not small: the backbone on this input needs about 33 billion (the 4.09 billion of the table above for 224 pixels times $(640/224)^2 \approx 8.16$, ignoring the classifier), so the neck adds a little under a quarter. The table says where. The 3x3 smoothing at 80 by 80 does 3,774,873,600 multiply-adds, just under half of the total, for the same 590,080 weights that the smoothing at 20 by 20 spends 236 million on: a weight is used once per cell of its map, and the finest map has sixteen times the cells of the coarsest. The designers of a real-time detector choose where to cut for this reason, and the choices show in the code.

The YOLOv8 family has no separate lateral and smoothing convolutions and merges by concatenation. In brain's nano configuration (`YoloConfig::yolov8n` in `crates/yolov8/src/config.rs`) the backbone gives 64, 128 and 256 channels at strides 8, 16 and 32, and the neck of `crates/yolov8/src/model.rs` runs:

<!-- worked: cnns.yolo_neck -->
The PAN-FPN neck of brain's nano YOLOv8 configuration on a 640 by 640 input (backbone outputs of 64, 128 and 256 channels at strides 8, 16 and 32, called F3, F4 and F5 here and P3, P4 and P5 in the code; stage widths 128, 64, 64, 128, 128, 256; T4 is the top-down stage at stride 16 and N3, N4, N5 are the outputs that go to the head): channel concatenation instead of addition, nearest-neighbour doubling for the way up and a 3x3 stride-2 convolution for the way down. The last column is the weight count of each downsampling convolution (3 * 3 * C_in * C_out, before its batch normalization); a C2f stage is counted by its output only.

| step | output shape | downsampling weights |
|---|---|---:|
| backbone F5 | 256 x 20 x 20 |  |
| upsample F5, concat F4 (256 + 128) | 384 x 40 x 40 |  |
| C2f to T4 | 128 x 40 x 40 |  |
| upsample T4, concat F3 (128 + 64) | 192 x 80 x 80 |  |
| C2f to N3 (head input 0) | 64 x 80 x 80 |  |
| conv 3x3 stride 2 on N3, concat T4 (64 + 128) | 192 x 40 x 40 | 36,864 |
| C2f to N4 (head input 1) | 128 x 40 x 40 |  |
| conv 3x3 stride 2 on N4, concat F5 (128 + 256) | 384 x 20 x 20 | 147,456 |
| C2f to N5 (head input 2) | 256 x 20 x 20 |  |
<!-- /worked -->

Each fusion is a channel concatenation followed by a C2f block (the blocks of `crates/vision`, whose first layer is a $1 \times 1$ convolution that plays the part of the lateral projection, and whose last one mixes the concatenated channels), the way up is `Up`, the same nearest doubling, and the way down is a $3 \times 3$ convolution of stride 2. SCRFD in `crates/scrfd` keeps the textbook form of (2): lateral $1 \times 1$ convolutions, nearest upsampling (with the ONNX `asymmetric` and `floor` rule, $y[i] = x[\lfloor i/2 \rfloor]$), an add, smoothing convolutions and then downsampling convolutions and an add for the second path. The two designs show what is fixed and what is a choice: the levels, the doubling and the strided convolution are the idea; add against concatenation, and where the smoothing sits, are decisions that change the counts and not the principle.

One piece of the wiring is the old rule of this chapter again. A backbone map that feeds two consumers, as the stride-8 map feeds a stride-2 convolution of the backbone and a concatenation of the neck, receives the sum of the two gradients; the model's own comment says so, and its `Acc` unit is the out-of-place addition of them (`crates/yolov8/src/model.rs`), the gradient of a fan-out, the same sum of paths as the residual connection's identity. Commit `89d2eefb8` moved the three stages of this plumbing (`Up`, `Cat`, `Acc`) from the detector into `crates/vision` so that a depth-estimation decoder could use them, and it deliberately left the topology behind: "an `Fpn` abstraction would fit neither", because the wiring of the pyramid differs between models in exactly the choices above, and the stages are the right granularity to share. The proof it records is that the detector's logits stayed bitwise identical across the move.

The chapter that follows reads these three maps. A detector places a box in a cell of one of them, and what that cell can say about the object (where it begins, how big it is, what it is) is what the pyramid has put there.

## Check your understanding

1. Find the error: "Max pooling splits the incoming gradient among the elements of the window in proportion to their values."
2. In the table of the shift experiment, the circular-padded toy network has the same logits for a shift of 2 columns and different ones for a shift of 3. A colleague says the 3-column shift is broken by the padding. Is that right? Which shifts would leave the logits unchanged if the pool had stride 4 on a 16-pixel-wide picture?
3. A classifier replaces global average pooling by flatten-and-dense on the $6 \times 4 \times 4$ map. How many weights does the last layer need, and what happens to the network when it is run on a $12 \times 12$ picture?
4. A 40-layer plain stack trains no better than a 20-layer one. Someone proposes widening the layers to give the optimizer more freedom. Using the measurements of this chapter, say why that does not address what was measured, and what does.
5. A residual block has the same two convolutions as its plain twin. Where do the extra parameters come from, and what does the block cost beyond them?
6. A team imports a ResNet-D network but builds the stride-2 shortcut as a strided $1 \times 1$ convolution. Every shape check passes. Describe what is wrong with the output and what test would show it.
7. Batch normalization is folded into the convolution that follows it, and the exported model is checked on a $224 \times 224$ image and matches to $10^{-6}$ at every position but the outermost ring, where it is off by about 1. What is the cause, and which position of the normalization makes the fold exact?
8. In a feature pyramid with 256 channels at every level, which level's $3 \times 3$ smoothing costs the most multiply-adds on a $640 \times 640$ input, and by what factor against the coarsest? What would change if the neck concatenated instead of added?

## Summary

A pooling layer summarizes each window of a map by its maximum or its mean, without weights: the maximum sends its gradient to the winner of the window (the first one on a tie, the sum over windows when they overlap) and the mean shares it, and global average pooling takes the whole map and turns a classifier's last stage into one number per channel with no dependence on the picture size. A convolutional network alternates convolutions with nonlinearities and pooling until one output sees the whole picture and a linear layer reads the result; the toy of this chapter, with 283 parameters and 5,778 multiply-adds per image, separates stripes, checkerboards and blobs to 0.9556 on held-out images, its backward pass checked against central differences. Convolution is exactly equivariant to shifts and the pooling makes the logits invariant to shifts by multiples of its stride and only approximately to others. Plain stacks lose signal and gradient with depth, about 2e-6 of the stream and 9e-6 of the gradient at 48 convolutions in the toy, and the residual block $\mathrm{relu}(f(\mathbf{x}) + \mathrm{sh}(\mathbf{x}))$, whose identity shortcut is a path without weights, keeps the stream at order one and the gradient between 0.1 and 0.3; a projection shortcut matches shapes when a stage changes them and its exact form (strided or averaged) is part of the model, and the bottleneck block cuts the weights and multiply-adds by 16.94 at 256 channels. Batch normalization in a block is per channel and folds into the preceding convolution by (1), exactly in the interior and not at the border if it precedes a padded one. A feature pyramid merges the coarse maps' context into the fine maps' resolution by lateral $1 \times 1$ convolutions, upsampling and addition (or concatenation), and a bottom-up path returns the fine localization to the coarse maps; its cost is dominated by its finest level.

What the pyramid delivers, three maps of 256 channels at strides 8, 16 and 32, is the input of the next chapter, where each cell of each map has to say whether an object begins there, how large it is and what it is, and where the boxes that it proposes have to be compared, scored and pruned.

## References

- Y. LeCun, L. Bottou, Y. Bengio and P. Haffner, *Gradient-based learning applied to document recognition*, Proceedings of the IEEE, 1998. The convolutional network with subsampling layers.
- A. Krizhevsky, I. Sutskever and G. E. Hinton, *ImageNet classification with deep convolutional neural networks*, 2012. A deep convolutional network trained on ImageNet, with ReLU activations and overlapping max pooling.
- K. He, X. Zhang, S. Ren and J. Sun, *Deep residual learning for image recognition*, CVPR 2016. The residual block, the projection shortcut, the bottleneck block and the training-error comparison of plain and residual networks of 18 to 152 layers.
- T. He, Z. Zhang, H. Zhang, Z. Zhang, J. Xie and M. Li, *Bag of tricks for image classification with convolutional neural networks*, CVPR 2019. The average-pool shortcut of the ResNet-D variant.
- S. Ioffe and C. Szegedy, *Batch normalization: accelerating deep network training by reducing internal covariate shift*, 2015. Batch normalization, and its per-channel form for convolutions.
- T.-Y. Lin, P. Dollar, R. Girshick, K. He, B. Hariharan and S. Belongie, *Feature pyramid networks for object detection*, CVPR 2017. The top-down path with lateral connections.
- S. Liu, L. Qi, H. Qin, J. Shi and J. Jia, *Path aggregation network for instance segmentation*, CVPR 2018. The bottom-up path added to the pyramid.

## Exercises

1. **[Calculate]** For $X$ of the pooling example, compute the average pool with window 2 and stride 1 (output $3 \times 3$) and its backward pass for $dy$ equal to all ones. Which input receives the most gradient, and how much?
2. **[Calculate]** The first convolution of the toy gets 8 filters instead of 4 and the second keeps 6 filters on the new input width. Give the parameters and multiply-adds of each layer and the totals.
3. **[Calculate]** Compare a basic block and a bottleneck block (mid-width a quarter of the output width) at 512 channels on a $28 \times 28$ map: weights and multiply-adds of each, and the ratio. Compare with the 256-channel case on $56 \times 56$.
4. **[Calculate]** A convolution with a single channel in and out has the taps $(1, -2, 3)$ along a row, bias 0.5, and is followed by batch normalization with $\gamma = 2$, $\beta = 1$, running mean 0.5, running variance 0.24999 and $\epsilon = 10^{-5}$. Give the folded taps and bias.
5. **[Calculate]** On a 6 by 6 map, what share of the taps of a $3 \times 3$ window with padding 1 land on a real pixel? What factor does a stack of 10 such layers lose in rms by this effect alone?
6. **[Calculate]** For a $512 \times 512$ input, give the shapes of the three levels of a ResNet-50, and the multiply-adds of the $3 \times 3$ smoothing of the finest level of the feature pyramid with 256 channels.
7. **[Implement]** Reproduce `cnns.pooling` and `cnns.toy_set`: the max, average and global average pools with their backward passes (the tie rule, the sum for overlapping windows) and the toy images from the generator, seed and draw order printed with the example.
8. **[Implement]** Reproduce `cnns.toy_layers`, `cnns.gradient_check` and `cnns.training`: the network of the figure with the printed initialization, its backward pass against central differences with the gate of Volume III, and the Adam run with the digits that agree under the nudges of the book's rule.
9. **[Implement]** Reproduce `cnns.bn_fold` and `cnns.bn_before_conv`: the folded convolution against the convolution followed by batch normalization (in 64 bits and with 32-bit rounding), and the border error of the fold into a following padded convolution.
10. **[Investigate]** Reproduce `cnns.depth_signal` and then change the toy stack: widen the layers to 16 channels, or add batch normalization after every convolution (training-mode statistics, per channel), and report what happens to the first-layer gradient at $d = 24$ and to the border effect. Say which of the two cures the shrinking of the stream and by how much.
11. **[Investigate]** Reproduce `cnns.resnet_counts` and `cnns.fpn_table`, then compute the same pyramid with concatenation in place of addition (a $1 \times 1$ convolution to 256 after each concatenation of two 256-channel maps) and report the change in weights and multiply-adds.

## Key terms

- **pooling**: a layer that summarizes each window of a map by one number, the largest (max pooling, whose gradient goes to the winner of the window) or the mean (average pooling, whose gradient is shared equally), with no weights; global average pooling takes the whole map. <!-- concept: pooling -->
- **convolutional network**: a network built from convolutions, nonlinearities and pooling (or strided convolutions) that shrinks a picture's maps while widening their channels until one output sees the whole picture and a linear layer reads it. <!-- concept: convolutional-network -->
- **residual network**: a network of blocks that compute $\mathrm{relu}(f(\mathbf{x}) + \mathrm{sh}(\mathbf{x}))$, the sum of a short stack of convolutions and a shortcut (the identity, or a $1 \times 1$ projection when the shape changes), which lets gradients and signals pass a deep stack along a path without weights; the basic block has two $3 \times 3$ convolutions and the bottleneck block a $1 \times 1$ reduction, a $3 \times 3$ and a $1 \times 1$ expansion. <!-- concept: resnet -->
- **feature pyramid**: a set of maps of the same width at several strides built from a backbone's levels, each coarse level's context upsampled and added to the finer level's projection (top-down) and, in the path aggregation form, the fine level's detail carried back to the coarse ones by strided convolutions (bottom-up), so that every scale has both resolution and meaning. <!-- concept: feature-pyramid -->

## Answers

**Recall questions.**

1. The side stays 4 (with $2p = k - 1$ and stride 1 the output size is the input size) and the work is $4 \cdot 4 \cdot 6 \cdot 4 \cdot 9 = 3{,}456$ multiply-adds (the second convolution of the toy).
2. By the recursion $r \leftarrow r + d(k - 1)j$, $j \leftarrow js$: $r = 3 + 1 \cdot 1 \cdot 1 = 4$ and $j = 1 \cdot 2 = 2$.
3. $\mathbf{p} - \mathbf{y}$, the softmax probabilities minus the one-hot target (divided by the number of examples for a mean loss).
4. $\sqrt{2/n_{\mathrm{in}}}$, and $n_{\mathrm{in}} = 9C_{\mathrm{in}}$ (each output sums $C_{\mathrm{in}}\cdot 3 \cdot 3$ products).
5. $d\mathbf{x} = d\mathbf{y} + J_f^\top d\mathbf{y}$: the gradient arriving from above is copied unchanged along the addition and the branch adds its own term. Over $L$ layers the product of Jacobians contains one term, the identity, that crosses all of them with coefficient exactly 1, while in a plain stack every term has all $L$ factors, which are below 1 and give an exponentially small gradient.
6. It uses the running mean and variance accumulated in training, fixed numbers, so the layer is a per-channel affine map $\mathrm{sc}\,(x - \mu) + \beta$ with $\mathrm{sc} = \gamma/\sqrt{\sigma^2 + \epsilon}$; folded into the linear layer before it, $\mathbf{W}' = \mathrm{diag}(\mathrm{sc})\,\mathbf{W}$ and $\mathbf{b}' = \mathrm{sc}\odot(\mathbf{b} - \boldsymbol{\mu}) + \boldsymbol{\beta}$.

**Check your understanding.**

1. The maximum has a derivative of 1 with respect to the winner and 0 with respect to every other element of the window, so the whole gradient goes to the winner (the first of equal maxima); the proportional split is the rule of a softmax-weighted pooling, a different layer. It is the average pool that shares the gradient, equally.
2. No. The pooling has stride 2, so a shift by 3 carries even window positions to odd ones, which no shift of the pooled map reproduces; the convolutions are exactly equivariant (first column of the table). With a pool of stride 4 the exact invariance would be for the shifts that are multiples of 4 (0, 4, 8, 12) and the others would change the logits.
3. $96 \cdot 3 + 3 = 291$ weights against 21. On a $12 \times 12$ picture the second map is $6 \times 6 \times 6$ with 216 values, and the dense layer, built for 96, cannot be applied: the picture size is baked into the head, where global average pooling gives 6 numbers for any size.
4. The measurements are about signal and gradient: at 48 convolutions the stream of the plain toy stack had fallen by a factor of about $5 \times 10^5$ and the first layer's gradient to about $10^{-5}$, because each layer multiplies the signal by a factor below 1 (the border share and the ReLUs among them) and a product of 48 such factors is tiny. A wider layer changes the constant of each factor, not the product over depth. What does is a path along which the signal and gradient do not pass through the layers' factors: the identity of the residual block, or a normalization that restores the scale at each layer (Volume III, Chapter 10), each addressing the product itself.
5. They do not come from the block's own computation: the identity shortcut has no parameters and costs one addition per output value. Extra parameters appear only for a projection shortcut (131,072 for 256 to 512 channels, a $1 \times 1$ convolution), when the shape changes, and at inference an identity can be folded into a convolution kernel at no parameter cost at all.
6. The strided $1 \times 1$ convolution reads the even pixels and the average-pool-then-$1 \times 1$ reads their pairs, so the shortcut contributes the features of positions half a pixel to the left, added to a branch that is centred on the right ones; every feature of every later layer is shifted, shapes and parameter counts are unchanged, and the accuracy loss is gradual and looks like a weak model. Only a comparison of outputs against the reference (a golden of a real picture) shows it, as in the importer check in `check_structural` that insists on the $2 \times 2$ pool.
7. The padding: the fold of (1) puts the shift of the normalization into the bias on the assumption that every tap reads a normalized value, and the taps of the outermost ring that fall on the padding read 0 in the original but the shift in the folded form (the border table has errors above 1 at all 28 border positions of its 8 by 8 example; on 224 by 224 the ring has 892 positions). The fold is exact when the normalization follows the convolution; for a normalization before the convolution it is exact only in the interior.
8. The finest level, $\mathrm{P}_3$ at 80 by 80: 3,774,873,600 multiply-adds against 235,929,600 at 20 by 20, a factor of 16 (the number of cells). With concatenation each merge adds a $1 \times 1$ or the first convolution of a C2f block on twice the channels, which changes the constant per level but not the ratio between levels, since the work of every convolution at a level is proportional to its number of cells.

**Exercises** (calculation answers).

1. The $3 \times 3$ output is $[[1.25, 1.25, 2.25], [1.75, 2.75, 3.0], [2.5, 4.0, 3.75]]$. Each of the nine windows sends $1/4$ to each of its four inputs, so an input receives $1/4$ times the number of windows that cover it: the four interior inputs are covered by four windows and receive the most, 1.0 each (corners 0.25, other border inputs 0.5).
2. Conv 1: $8 \cdot 9 + 8 = 80$ parameters and $64 \cdot 8 \cdot 9 = 4{,}608$ multiply-adds. Conv 2 on 8 channels: $6 \cdot 8 \cdot 9 + 6 = 438$ parameters and $16 \cdot 6 \cdot 8 \cdot 9 = 6{,}912$ multiply-adds. Linear: 21 and 18. Totals: 539 parameters, 11,538 multiply-adds.
3. Basic block: $2 \cdot 512 \cdot 512 \cdot 9 = 4{,}718{,}592$ weights and, times $28^2 = 784$, 3,699,376,128 multiply-adds. Bottleneck (128 in the middle): $512 \cdot 128 + 128 \cdot 128 \cdot 9 + 128 \cdot 512 = 278{,}528$ weights and 218,365,952 multiply-adds. The ratio is 16.94 for both. The multiply-adds equal those of the 256-channel case at $56 \times 56$ exactly: the weights are four times as many and the cells four times fewer.
4. $\mathrm{sc} = 2/\sqrt{0.24999 + 0.00001} = 2/0.5 = 4$, so the taps are $(4, -8, 12)$ and the bias is $4(0.5 - 0.5) + 1 = 1$.
5. Per axis, the 6 positions have $2, 3, 3, 3, 3, 2$ real taps out of 3 each: $16/18$; the share is $(16/18)^2 = 0.7901$. Ten layers lose $\sqrt{0.7901}^{10} = 0.7901^5 = 0.308$ of the rms, a factor of about 3.2.
6. $\mathrm{F}_3$: $[512, 64, 64]$, $\mathrm{F}_4$: $[1024, 32, 32]$, $\mathrm{F}_5$: $[2048, 16, 16]$. The smoothing of the finest level is $64^2 \cdot 256 \cdot 256 \cdot 9 = 2{,}415{,}919{,}104$ multiply-adds.
