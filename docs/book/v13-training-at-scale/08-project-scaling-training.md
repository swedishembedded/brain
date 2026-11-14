<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Project: Scaling a Training Run

*Can you move a training run from one device to several without changing what it learns?*

Train one small network for 120 updates on one device, and then train it again from the same starting numbers on the same examples in the same order, with the work of each update dealt to four ranks (devices that run the same program on different rows). The first run ends with a loss of 0.00281. The second ends with a loss of 0.00140, half of it. Four devices have not only matched the single device but beaten it, and the reflex is to keep the faster build and move on. The third row of the table below is what that build is: its ranks average their four shard gradients as plain means, and a shard of two rows counts as much as a shard of three. Its trained numbers are 0.389 away from the numbers of the one-rank run, and the run is a different run that happens to score better on the examples it trains on.

<!-- worked: project_scaling_training.opening -->
The base run of milestone 2 (the network of milestone 1, 120 updates of 10 draws, Adam with rate 0.01) trained three ways from the same initial vector and the same draws. 'Loss' is the mean over the 40 examples of half the squared error against T0 of the network after the 120 updates; 'deviation' is the largest absolute difference, over the 337 numbers, from the one-rank run. A number that comes from a chaotic run is printed with only the digits that are the same in the run and in four copies of it whose first weight is changed by +1e-12, -1e-12, +3e-12 and -3e-12: in full if all agree on it, as 'about x' with the leading digits on which all agree, or as 'unstable' when not even one digit survives; the other digits cannot be reproduced in another implementation.

| build | loss against T0 after 120 updates | deviation from the one-rank run |
|---|---:|---:|
| one rank | 0.00280691 | 0 |
| four ranks, the shard gradients weighted by their row counts | 0.00280691 | below 1e-9 |
| four ranks, the mean of the four shard means | 0.00140075 | 0.389 |
<!-- /worked -->

The loss cannot tell the second row from the third, since both are small, and a loss is the one number every build is allowed to look at. What separates them is a comparison that no one pays for in a run: the numbers of the network after the last update, one build against the other, entry by entry. The project is that comparison made into the first tool of the build. You assemble, in steps, a system that trains a small network in two stages (all its numbers, then 128 numbers of low-rank adapters on the frozen result), with the work of each update dealt to four ranks, the activations replayed in segments instead of stored, and the run killed and brought back from a checkpoint; and every step ends in numbers from a one-device run that your system must match to a stated tolerance, with the planted mistakes that a loss does not see and the comparison does.

The pieces are those of the volume, in the order a run meets them: the order of examples that a restart can recompute (Chapter 1), the replay of layers in the backward pass (Chapter 3), the adapters on a frozen base (Chapter 4), the ranks, the all-reduce and the weighted average (Chapter 5) and the generation that a crash cannot destroy (Chapter 7). The arithmetic is in 64-bit floating point, so that the comparison measures what the order of additions does to a result and not what a narrow format does; the narrow formats of Chapter 2 and the distillation of Chapter 6 have no part in this run.

![A left-to-right diagram. The ten draws of an update are dealt to four ranks, rank j taking the draws b with b mod 4 = j, so that ranks 0 and 1 hold three rows and ranks 2 and 3 hold two. Each rank replays the network in segments of two layers and forms the adapter gradient, reading a frozen base of 337 numbers. A ring all-reduce sums the four vectors and divides by 10; Adam updates the 128 adapter numbers; every 10 updates a generation of values, state and meta is written; and the next update draws the next ten rows.](img/scaling-system.svg)

`scripts/book/worked/project_scaling_training.py`, the module that produced every number below, imports the epoch order of Chapter 1 (`training_data.epoch_order`), the 64-bit FNV-1a digest of Chapter 7 (`checkpoints_and_recovery.fnv1a64`) and the ring all-reduce of Chapter 5 (`parallelism._ring`), and implements the network, the replay, the adapters, the ranks and the generations itself. You build the system in any language. You need 64-bit floating point with a correctly rounded square root and no fused multiply-add (a product and a sum that are rounded together would change the digits below), the book's generator ([Volume I, Chapter 17](../v01-mathematics/17-randomness.md)) with its `uniform`, and the 64-bit FNV-1a hash. Six questions bring back what the steps lean on.

1. (Chapter 5, data parallelism.) Ten rows are cut into shards of 4, 3 and 3 rows, and each rank computes the mean gradient of its own shard. With what weights are the three shard gradients combined to give the mean gradient of the ten rows, and what weights does the plain average of the three use instead?
2. (Chapter 5, all-reduce.) The coordinator adds the ranks' vectors in rank order from zero. In what order does a ring of four ranks add the four numbers of chunk $c$? Both are correct sums: what can differ between them, and what must be the same at the end of every all-reduce within one run?
3. (Chapter 3, activation checkpointing.) A backward pass that replays a segment from its kept input reads something besides that input. What, and under what condition is the replayed gradient the very gradient that keeping everything would give? How much more arithmetic does a step cost when every layer's forward is replayed once?
4. (Chapter 4, low-rank adapters.) An adapter of rank $r$ adds $s\,\mathbf{B}\mathbf{A}$ to a frozen weight. Which factor starts at zero, what is the gradient of $\mathbf{A}$ at that start, and why may not both factors start at zero?
5. (Chapter 7, transactional checkpoints, and Volume III, Chapter 7, Adam.) After $t$ updates of Adam, what does the next update read besides the weights and the gradient, and what must a reader check before it accepts a generation that is on the disk?
6. (Chapter 1, the order of feeding.) A run has drawn $d$ examples from 40. Which entry of which epoch's order is draw $d$, and why is $d$ all that a restarted run needs to know about the data?

## What it means for two runs to be the same run

Every random input of the project comes from the book's SplitMix64 at the seed `seed_of(name)` printed beside its table. Three generators are used, each started once, at its first value, and never restarted: `project_scaling_training.data` (the 40 examples), `project_scaling_training.init` (the network's numbers) and `project_scaling_training.adapters` (the adapter factors). The order in which the examples are fed is not a stream: draw $d$ of a run is entry $d \bmod 40$ of `training_data.epoch_order(seed, 40, d // 40)`, a function of the seed and the draw index alone, with the seed `seed_of('project_scaling_training.order')` for the base run and `seed_of('project_scaling_training.order_ft')` for the adapter run. Every number is a 64-bit float. The operations are addition, subtraction, multiplication, division, square root and comparison, each rounded on its own, and every sum runs in the order the text states, from `0.0`. Indices count from 0 (layers, rows, ranks, draws), and updates count from 1.

Two runs are the same run, here, when two things hold: every trained number of one is within $10^{-9}$ of the same number of the other, and their losses agree in the digits printed. The tolerance is chosen from what reordered sums do. Dealing the rows of an update to four ranks changes the order in which the ten per-row gradients are added, and that changes the last bits of every gradient and, through the updates, the last bits of every number; Chapter 5 counted how many of 840 elements differ in the last bits at 2, 3, 4 and 8 ranks. The distance that results over the 60 updates of the adapter run is below $10^{-15}$ (the table of the whole run prints it), six orders below the tolerance; and the smallest deviation of any planted mistake of the project is $3.18 \times 10^{-4}$ (the undivided sum of milestone 4), five orders above it. A resume on the same ranks is held to a harder standard than the tolerance: it must give the same bits as the run that was never stopped, because every addition of the continuation happens in the same order.

The decimals that tables print are 6 significant digits of the 64-bit result, correctly rounded, or 3 for a deviation. Every number that comes out of training goes through the stable-digits rule that is printed under each table: the run and four copies of it, whose first initial weight is changed by $\pm 10^{-12}$ and $\pm 3 \times 10^{-12}$, must agree on a digit before it is printed. Nothing in this project is chaotic, and the printed digits of its tables are the digits on which all five runs agree; another implementation that follows the stated orders reproduces them.

## The task, the network and the numbers it starts from

A run is made of examples, a network, a loss and a starting point, and all four are fixed by a rule short enough to print. The examples are 40 points $\mathbf{x}_i$ of four coordinates each, $x = 2u - 1$ for the next `uniform()` $u$ of the generator, drawn example by example and coordinate by coordinate. Each has two targets: $y^{\mathrm{base}} = (x_0 x_1 + 0.5\,x_2) - x_3 x_3$, what the base run learns, and $y^{\mathrm{ft}} = (y^{\mathrm{base}} + 0.5\,x_0) - (0.25\,x_1)\,x_2$, the changed task that the adapters of milestone 5 are asked for. In the tables these are `T0` and `T1`, and the products are evaluated before the sums, left to right.

The network has $L = 6$ layers. A layer computes $\mathbf{y} = \mathbf{W}\mathbf{x} + \mathbf{b}$ and, in all but the last, replaces each entry by $\max(\cdot, 0)$ (the ReLU, which here is `z if z > 0.0 else 0.0`). The weight matrix of a layer has shape $[\text{outputs}, \text{inputs}]$, row-major; a neuron's sum starts at its bias and adds $W[o][i]\,x[i]$ in increasing $i$. The loss of an example is $\tfrac12(o - y)^2$ for the network's output $o$, and the loss of a set is the mean of these over the examples, summed in example order from `0.0`.

<!-- worked: project_scaling_training.data_and_network -->
The generator `project_scaling_training.data` (seed 17480193994437005951) draws, in this order and each by uniform(), the 4 coordinates of example 0, then of example 1, and so on to example 39: x = 2u - 1. The targets are evaluated in this order of operations: T0 = (x0 x1 + 0.5 x2) - x3 x3 and T1 = (T0 + 0.5 x0) - (0.25 x1) x2 (products first, then the sums left to right). Each row prints an example, its four coordinates and its two targets to 6 decimals. The 160 coordinates, written as 64-bit floats (IEEE 754, little-endian) in this order, have the FNV-1a 64 digest 7ffcc7784550bd18; the 40 targets T0 and the 40 targets T1 have the digests 96391b2ff4047629 and 81c9fc4273021526.

| example | x0 | x1 | x2 | x3 | T0 | T1 |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.793706 | -0.417420 | 0.007787 | -0.322759 | -0.431589 | -0.033923 |
| 1 | -0.802464 | -0.651557 | 0.901883 | -0.517433 | 0.706056 | 0.451731 |
| 2 | 0.052096 | 0.556686 | 0.862447 | 0.795256 | -0.172208 | -0.266188 |
| 39 | 0.122950 | -0.076129 | 0.602387 | -0.868065 | -0.461703 | -0.388764 |

The network has 6 layers, each y = W x + b followed by max(., 0) except the last, which is linear. A neuron's sum starts at its bias and adds the products W[o][i] x[i] in increasing i. The flat vector of the 337 numbers holds the layers in order, each as W (row-major, [outputs, inputs]) and then b. The generator `project_scaling_training.init` (seed 3934880003332642749) draws, layer by layer and row-major, one uniform() for every entry of W, W = (2u - 1) limit with limit = sqrt(6 / (inputs + outputs)) (the square root correctly rounded); every b starts at 0.0 and takes no draw, so 296 numbers are drawn. The initial vector has the FNV-1a 64 digest f3a94ce165fe1ac6, and its first three numbers are -0.67967990, 0.59778289, -0.35061673.

| layer | W shape | weights | biases | numbers | offset | activation |
|---:|---|---:|---:|---:|---:|---|
| 0 | 8 x 4 | 32 | 8 | 40 | 0 | ReLU |
| 1 | 8 x 8 | 64 | 8 | 72 | 40 | ReLU |
| 2 | 8 x 8 | 64 | 8 | 72 | 112 | ReLU |
| 3 | 8 x 8 | 64 | 8 | 72 | 184 | ReLU |
| 4 | 8 x 8 | 64 | 8 | 72 | 256 | ReLU |
| 5 | 1 x 8 | 8 | 1 | 9 | 328 | none |
| all |  | 296 | 41 | 337 |  |  |
<!-- /worked -->

The flat vector of the 337 numbers is the book's order for a model's parameters: the layers in sequence, each as $\mathbf{W}$ and then $\mathbf{b}$ (Volume III, Chapter 14). The forward and backward passes of one layer are two small routines. Written for one example at a time, the layer keeps its input $\mathbf{x}$ and its output $\mathbf{y}$ for the backward pass; the branch about the adapter $\mathbf{A}, \mathbf{B}$ is the subject of milestone 5 and is not taken until then (every `A[l]` is absent).

```text
function layer_forward(l: int, x: array<real>[n_in], net: Net) -> array<real>[n_out]:
    t = zeros(real, [r])
    if net.A[l] is present:
        for q in 0 .. r:
            c = 0.0
            for i in 0 .. n_in:
                c = c + net.A[l][q, i] * x[i]
            t[q] = c
    y = zeros(real, [n_out])
    for o in 0 .. n_out:
        z = net.b[l][o]
        for i in 0 .. n_in:
            z = z + net.W[l][o, i] * x[i]
        if net.A[l] is present:
            c = 0.0
            for q in 0 .. r:
                c = c + net.B[l][o, q] * t[q]
            z = z + s * c
        y[o] = z if (l == L - 1 or z > 0.0) else 0.0
    return y

function layer_backward(l: int, x: array<real>[n_in], y: array<real>[n_out],
                        dy: array<real>[n_out], net: Net, g: Net,
                        train_base: bool) -> array<real>[n_in]:
    dz = [dy[o] if (l == L - 1 or y[o] > 0.0) else 0.0 for o in 0 .. n_out]
    if train_base:
        for o in 0 .. n_out:
            for i in 0 .. n_in:
                g.W[l][o, i] = g.W[l][o, i] + dz[o] * x[i]
            g.b[l][o] = g.b[l][o] + dz[o]
    dx = zeros(real, [n_in])
    for i in 0 .. n_in:
        c = 0.0
        for o in 0 .. n_out:
            c = c + dz[o] * net.W[l][o, i]
        dx[i] = c
    if net.A[l] is present:
        t = zeros(real, [r])
        for q in 0 .. r:
            c = 0.0
            for i in 0 .. n_in:
                c = c + net.A[l][q, i] * x[i]
            t[q] = c
        for o in 0 .. n_out:
            for q in 0 .. r:
                g.B[l][o, q] = g.B[l][o, q] + (s * dz[o]) * t[q]
        dt = zeros(real, [r])
        for q in 0 .. r:
            c = 0.0
            for o in 0 .. n_out:
                c = c + dz[o] * net.B[l][o, q]
            dt[q] = s * c
        for q in 0 .. r:
            for i in 0 .. n_in:
                g.A[l][q, i] = g.A[l][q, i] + dt[q] * x[i]
        for i in 0 .. n_in:
            c = 0.0
            for q in 0 .. r:
                c = c + dt[q] * net.A[l][q, i]
            dx[i] = dx[i] + c
    return dx
```

`Net` holds, for every layer $l$, `W[l]`, `b[l]` and, for the layers that carry an adapter, `A[l]` and `B[l]`; `g` is a `Net` of the same shapes that collects gradients, and its flat vector is the one of the tables. The gradient of a row accumulates into `g` entry by entry: each entry of $\mathbf{W}$, $\mathbf{b}$ and the adapters receives one product per row, so the sum over a batch runs in the order of the rows.

**Milestone 1 ends** when your data reproduces the three digests and the printed rows, your initial vector reproduces its digest and its first three numbers, and a check of your gradient against central differences passes. The check is the one of Volume III, Chapter 6, here on the initial network and ten examples:

<!-- worked: project_scaling_training.gradient_check -->
The initial network and the 10 examples 0, 5, 9, 17, 23, 31, 38, 12, 25, 3 (in this order). The analytic gradient is the sum of the per-example gradients in this order divided by 10; the numeric one is the central difference of the mean loss with step 1e-6 at every one of the 337 numbers. The deviation is |numeric - analytic| / max(1, |analytic|), and the criterion is that its largest value is below 1e-6.

| quantity | value |
|---|---:|
| largest deviation over all 337 numbers | passes (below 1e-6) |
<!-- /worked -->

## The run that every other build is held to

The reference is the simplest run that can be written: one rank, every activation kept, the base network trained in all its 337 numbers. An update draws 10 examples, forms the gradient of their summed loss in draw order and divides by 10, and applies Adam of Chapter 4 (rate 0.01, $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$, bias-corrected, no decay) to every number. The bias corrections are the powers $0.9^t$ and $0.999^t$ formed by repeated multiplication from `1.0`, one factor per update, so that a library `pow` cannot change a digit. With 40 examples and 10 draws an update, an epoch is 4 updates and the 120 updates are 30 epochs.

<!-- worked: project_scaling_training.reference_run -->
One device, one rank, every activation kept. Step t draws the 10 examples of draws 10 (t - 1) to 10 t - 1 from the order of `training_data.epoch_order` with the seed 13353539597274381162 (seed_of 'project_scaling_training.order'), computed afresh from the epoch of each draw; the gradient is the sum over the 10 rows in draw order, from 0.0, divided by 10, and the update is `adam_update` of Chapter 4 with rate 0.01 and step count t (the bias corrections by repeated multiplication from 1.0). The losses are the mean over all 40 examples, summed in example order from 0.0, of half the squared error of the network after t updates, against T0 (what this run trains on) and T1 (what the adapters of milestone 5 will be asked for). A number that comes from a chaotic run is printed with only the digits that are the same in the run and in four copies of it whose first weight is changed by +1e-12, -1e-12, +3e-12 and -3e-12: in full if all agree on it, as 'about x' with the leading digits on which all agree, or as 'unstable' when not even one digit survives; the other digits cannot be reproduced in another implementation.

| updates t | loss against T0 | loss against T1 |
|---:|---:|---:|
| 0 | 0.194272 | 0.258913 |
| 20 | 0.104152 | 0.153958 |
| 40 | 0.0233749 | 0.105893 |
| 60 | 0.0108251 | 0.084512 |
| 80 | 0.00566095 | 0.0530399 |
| 100 | 0.00420787 | 0.0785154 |
| 120 | 0.00280691 | 0.058736 |
<!-- /worked -->

The loss against `T0` falls from 0.194 to 0.0028; the loss against `T1`, a task the run never saw, starts at 0.259 and ends at 0.0587, which is the room that the adapters are given to work in. The loss against `T1` is not monotone (0.0530 at update 80, 0.0785 at update 100 and 0.0587 at update 120), as is usual for a task the run does not train on, and the table is the point of comparison for the rest of the project: the 337 numbers after update 120 are the base that milestones 5 to 7 freeze, and every build below is compared with them, number by number.

**Milestone 2 ends** when your one-rank run reproduces the seven rows. A disagreement in the first row means the initial vector or the loss; in the later rows, the order of the draws, the bias corrections or the order of the sum over a batch.

## Rebuilding layers in the backward pass

A layer's backward pass needs its input and the signs of its output. Keeping every one of them makes the peak memory of an example the sum of all of them; the alternative of Chapter 3 keeps the input of every $k$-th layer, the boundary of a segment of $k$ layers, and replays a segment's forward from its boundary just before differentiating it.

```text
function example_gradient(net: Net, x: array<real>[4], target: real,
                          k: optional int, g: Net, train_base: bool) -> none:
    if k is absent:
        v = [x]
        for l in 0 .. L:
            append layer_forward(l, v[l], net) to v
        dy = [v[L][0] - target]
        for l in L - 1 down to 0:
            dy = layer_backward(l, v[l], v[l + 1], dy, net, g, train_base)
        return
    kept = map of int to tensor, empty
    kept[0] = x
    cur = x
    for l in 0 .. L:
        cur = layer_forward(l, cur, net)
        if (l + 1) mod k == 0 and l + 1 < L:
            kept[l + 1] = cur
    dy = [cur[0] - target]
    for gi in (L - 1) // k down to 0:
        s0 = gi * k
        e = min(s0 + k, L)
        w = [kept[s0]]
        for l in s0 .. e:
            append layer_forward(l, w[l - s0], net) to w
        for l in e - 1 down to s0:
            dy = layer_backward(l, w[l - s0], w[l - s0 + 1], dy, net, g, train_base)
```

The routine is `recompute_backward` of Chapter 3 for layers that are not blocks. The vectors between layers are $\mathbf{v}_0 = \mathbf{x}$ (4 numbers), $\mathbf{v}_1, \dots, \mathbf{v}_5$ (8 numbers each) and $\mathbf{v}_6 = o$ (1 number), 45 numbers in all. Memory is counted by one rule: a vector is alive from the operation that makes it until the last operation that reads it, the output of the last layer is read once, to form the first gradient, and the gradient vectors that flow backward are not counted, being the same in every policy. A forward costs 2 FLOPs for every multiply-add of a layer's product, which is $2 \times 296 = 592$ FLOPs for the 296 weights of the table, and the backward costs twice that, so keeping everything is $3 \times 592 = 1{,}776$ FLOPs for an example and replaying every layer's forward once, in the backward, is 592 more.

<!-- worked: project_scaling_training.recomputation -->
The initial network, the 10 rows of step 1 (draws 0 to 9 of the order of `project_scaling_training.order`) and the gradient of their summed loss, accumulated in draw order. For each policy one example is run alone to count what is alive at once: the input and every layer's output is a vector of 4, 8, 8, 8, 8, 8 and 1 numbers; a vector is alive from the operation that makes it until the last operation that reads it (the output of the last layer is read once, to form the first gradient, and the gradient vectors that flow backward are not counted, being the same in every policy). A policy keeps a vector as a boundary when its index is a multiple of the segment length (the input has index 0) and the backward pass replays a segment from its boundary before differentiating it. FLOPs count 2 for every multiply-add of a layer's product and nothing else: 2 x inputs x outputs per layer for a forward, twice that for the backward (the products for the weights and for the input), whatever the layer. The last column counts the entries of the 337-number gradient of the batch whose 64-bit pattern differs from the keep run.

| policy | numbers alive at the peak (one example) | FLOPs, one example | gradient entries that differ |
|---|---:|---:|---:|
| keep every activation | 45 | 1776 | 0 of 337 |
| segments of 1 layer | 45 | 2368 | 0 of 337 |
| segments of 2 layers | 29 | 2368 | 0 of 337 |
| segments of 3 layers | 29 | 2368 | 0 of 337 |
| one segment of 6 layers | 45 | 2368 | 0 of 337 |

The same batch when a pending update adds 0.3 to one weight after the forward pass of every example and before its backward pass: W[0][1] of layer 0 (flat index 1) or W[0][1] of layer 1 (flat index 41). The distance is the Euclidean norm of the difference from the keep gradient of the unchanged weights, divided by the norm of that gradient.

| backward after the late update | weight of layer 0 | weight of layer 1 |
|---|---:|---:|
| keep every activation | 0 | 0.008829 |
| segments of 2 layers | 0.06247 | 0.008829 |
<!-- /worked -->

Two policies hold 45 numbers at their peak and two hold 29, and the rule explains which. With $k = 1$ every boundary is kept, and each layer's output is the next layer's input, so keeping the boundaries is keeping everything (and the replay costs a third more arithmetic for no memory). With one segment of six layers the boundary is the input alone, and replaying the segment rebuilds all six vectors at once, which is 45 again. With $k = 2$ the boundaries $\mathbf{v}_0, \mathbf{v}_2, \mathbf{v}_4$ are 20 numbers and the last segment's replay rebuilds $\mathbf{v}_5$ and $\mathbf{v}_6$, 9 more: 29. With $k = 3$ the boundaries $\mathbf{v}_0, \mathbf{v}_3$ are 12 and the last segment rebuilds $\mathbf{v}_4, \mathbf{v}_5, \mathbf{v}_6$, 17: 29 again. The cost of the saving is 2,368 against 1,776 FLOPs for an example, a third more, for every segment length, as in Chapter 3. The last column is the strong claim of that chapter and brain's contract for recomputation (see below): no entry of the 337-number gradient differs by even one bit, because the replay computes the forward again from the same input with the same weights in the same order.

The second table is the premise of the claim broken. A trainer that applies an update while a backward pass is still running changes a weight between the forward and the backward, and a replay reads that weight; keeping everything does not read every weight in the backward pass. The weight of layer 0 is read by nothing the keep build keeps: the backward of layer 0 forms an input gradient that no layer uses, so a changed weight there leaves every gradient entry as it was (distance 0), while the replay recomputes layer 0's output from the changed weight and its gradient moves (0.06247). The weight of layer 1 is read by both builds, in the backward of layer 1, and both are off by the same 0.008829. A late update is wrong for the replay in the first case and for both builds in the second, and only the comparison with the gradient of the unchanged weights shows by how much.

**Milestone 3 ends** when your routine reproduces the five rows of peak memory and FLOPs, when its gradients agree with the keep gradient in all 337 entries to the bit for every $k$, and when the late update reproduces the two distances.

## Ten rows over four ranks

An update draws 10 rows, and draw $b$ of the update ($b = 0, \dots, 9$) belongs to rank $b \bmod R$, as in Chapter 1. With $R = 4$ the ranks hold 3, 3, 2 and 2 rows, and the distinction between the weighted and the plain average of Chapter 5 is the whole of this milestone. A rank adds the gradients of its rows in draw order into a vector of zeros, and sends that sum. The coordinator adds the vectors of ranks 0, 1, 2, 3 in this order from zeros; the ring (`parallelism._ring`, with the vector padded by zeros to a multiple of $R$) adds each chunk in the order of Chapter 5; both divide the total by 10, the number of rows in the batch. Dividing the sum by the number of rows is the weighted average $\sum_j n_j \mathbf{g}^{(j)} / \sum_j n_j$, since a rank's sum is already $n_j$ times its mean.

```text
function batch_gradient(net: Net, rows: list<int>, R: int, k: optional int,
                        train_base: bool, ring: bool) -> array<real>[P]:
    vecs = []
    for j in 0 .. R:
        g = zeros-like(net)
        for b in 0 .. len(rows):
            if b mod R == j:
                example_gradient(net, X[rows[b]], target[rows[b]], k, g, train_base)
        append flat(g) to vecs
    total = zeros(real, [P])
    if ring:
        total = ring_all_reduce(padded(vecs))[0][0 : P]
    else:
        for j in 0 .. R:
            total = total + vecs[j]
    return total / len(rows)
```

`X` and `target` are the arrays of the examples and of the targets of the stage (`T0` for the base run, `T1` for the adapter run), `flat(g)` is the flat vector of the trainable numbers, `padded` appends zeros to each vector until its length is a multiple of $R$ and `[0][0 : P]` reads the first $P$ entries of the first rank's copy. At the first update, on the initial network, the gradient of every layout must be the one-rank gradient to rounding:

<!-- worked: project_scaling_training.shard_layouts -->
The initial network and the gradient of step 1 (the 10 draws 0 to 9). Draw b of the step belongs to rank b mod R; a rank adds the gradients of its rows in draw order from zeros; the coordinator adds the ranks in rank order from zeros and the ring adds each chunk in the order of the ring (the vector is padded with zeros to a multiple of R); the sum is divided by 10. The deviation is the largest absolute difference of an entry from the gradient of one rank, and the criterion is that it is below 1e-12.

| ranks R | rows of each rank | coordinator | ring |
|---:|---|---|---|
| 1 | 10 | passes (below 1e-12) | passes (below 1e-12) |
| 2 | 5 + 5 | passes (below 1e-12) | passes (below 1e-12) |
| 3 | 4 + 3 + 3 | passes (below 1e-12) | passes (below 1e-12) |
| 4 | 3 + 3 + 2 + 2 | passes (below 1e-12) | passes (below 1e-12) |

The same step on 4 ranks (shards of 3, 3, 2 and 2 rows) with three mistakes; the distance is the Euclidean norm of the difference from the one-rank gradient divided by the norm of that gradient.

| build | relative distance |
|---|---:|
| mean of the four shard means | 0.04198 |
| sum of the shards, not divided by the batch | 9 |
| every rank reads the shard of rank 0 | 0.599 |
<!-- /worked -->

All four layouts pass with either reduction, to below $10^{-12}$, and the second table is the same step with three mistakes, each a plausible slip. The mean of the four shard means weighs a row of a two-row shard $1.5$ times as much as a row of a three-row shard and is 0.04198 away from the gradient in relative terms. The sum that is not divided by the batch is ten times the gradient, so its distance from it, over the norm of the gradient, is 9. Reading the shard of rank 0 on every rank computes the same three rows on all four, so the step sees three of the ten rows, and it is 0.599 from the gradient of the ten.

The distances of a first step are small because one step is short. Over a whole run, the mistakes compound or they do not, and the comparison at the end is where they are told apart:

<!-- worked: project_scaling_training.parallel_runs -->
The base run of milestone 2, all 120 updates, on R ranks. The deviation is the largest absolute difference, over the 337 numbers after 120 updates, from the one-rank run started from the same (nudged) initial vector, and the criterion is that it is below 1e-9 for the run and its four nudged copies. 'Ranks hold the same bits' is whether, after the last reduction, every rank's copy of the summed gradient is the same list of 64-bit patterns (checked at the final weights, on the rows of the last step). A number that comes from a chaotic run is printed with only the digits that are the same in the run and in four copies of it whose first weight is changed by +1e-12, -1e-12, +3e-12 and -3e-12: in full if all agree on it, as 'about x' with the leading digits on which all agree, or as 'unstable' when not even one digit survives; the other digits cannot be reproduced in another implementation.

| build | deviation from the 1-rank run | loss against T0 after 120 updates | ranks hold the same bits |
|---|---|---|---:|
| 1 rank | passes (below 1e-9) | 0.00280691 | yes |
| 2 ranks, coordinator | passes (below 1e-9) | 0.00280691 | yes |
| 3 ranks, coordinator | passes (below 1e-9) | 0.00280691 | yes |
| 4 ranks, coordinator | passes (below 1e-9) | 0.00280691 | yes |
| 4 ranks, ring | passes (below 1e-9) | 0.00280691 | yes |
| 4 ranks, ring, segments of 2 layers | passes (below 1e-9) | 0.00280691 | yes |

The same run on 4 ranks with three planted mistakes. The deviation is the largest absolute difference from the one-rank run.

| build | deviation from the 1-rank run | loss against T0 after 120 updates |
|---|---:|---:|
| mean of the shard means | 0.389 | 0.00140075 |
| sum, not divided by the batch | 0.000318 | 0.00280622 |
| every rank reads the shard of rank 0 | 0.437 | 0.0435331 |
<!-- /worked -->

Every correct build, from 1 rank to 4 ranks on either topology and with replayed segments, ends within $10^{-9}$ of the one-rank run, with the loss of the one-rank run to six digits, and every rank ends each all-reduce with the same bits as every other. The three mistakes end 0.389, 0.000318 and 0.437 away. The first is the opening of the chapter: the loss of the mean of the shard means is 0.00140, half the right loss, a result that would have been reported as an improvement; the weights of the rows have been changed, so the loss measures another objective. The third, in which every rank trains on the same three of the ten rows, is the only one whose loss is plainly wrong (0.0435, more than fifteen times the right one). The second is the instructive one: the loss is 0.00280622 against 0.00280691, indistinguishable at the third digit, and the weights are 0.000318 apart. A gradient ten times too large is almost invisible to Adam, since Adam divides the first moment by the square root of the second, which scales with it. What is left of the difference is attributed to the $\epsilon$ in the denominator, and a run with a tiny $\epsilon$ tests it:

<!-- worked: project_scaling_training.blind_mistake -->
The base run of milestone 2 on 4 ranks with the gradient left as the sum of the ten rows (not divided by 10), against the one-rank run with the same epsilon in the denominator of Adam. The deviation is the largest absolute difference over the 337 numbers after 120 updates. A number that comes from a chaotic run is printed with only the digits that are the same in the run and in four copies of it whose first weight is changed by +1e-12, -1e-12, +3e-12 and -3e-12: in full if all agree on it, as 'about x' with the leading digits on which all agree, or as 'unstable' when not even one digit survives; the other digits cannot be reproduced in another implementation.

| epsilon of Adam | deviation of the undivided sum | its loss against T0 | loss against T0 of the one-rank run |
|---|---:|---:|---:|
| 1e-08 | 0.000318 | 0.00280622 | 0.00280691 |
| 1e-300 | passes (below 1e-9) | 0.00280615 | 0.00280615 |
<!-- /worked -->

With $\epsilon = 10^{-300}$ in both runs the undivided sum agrees with the right run to below $10^{-9}$, and with $\epsilon = 10^{-8}$ it does not: the factor of ten changes the size of $\sqrt{\hat{v}}$ next to $\epsilon$ for the entries whose second moment is small. The consequence is the part to keep. A learning-rate schedule, a gradient clip or a reduction that is off by a constant factor is hidden by Adam in the loss curve and is found only by the comparison, so the comparison of trained numbers is the test, and the curve is not.

**Milestone 4 ends** when your layouts pass the criterion of the first table, your three-rank, four-rank and ring runs end within $10^{-9}$ of your one-rank run, and the three mistakes end where the second table puts them. A build whose loss looks right and whose numbers are off has the first or the second of the mistakes.

## Arithmetic against bytes on the wire

The point of dividing the batch is the arithmetic that leaves each rank, and the price is the bytes that cross between ranks. Both are counted, never timed, by the rules of the tables: a rank's FLOPs are those of its rows (2 FLOPs per multiply-add of a product, the replay included), and the payload of one all-reduce is 8 bytes for each of the 64-bit numbers of the gradient, without framing.

<!-- worked: project_scaling_training.wire_per_step -->
Work of one step of the base run (10 rows) on R ranks, counted by the rules of milestones 3 and 4. A rank's FLOPs are those of its rows (counted while the rank runs them); the busiest rank, rank 0, has the most rows. Bytes are the payload of one gradient all-reduce, 8 bytes to a number (64-bit floats), without framing: the ring pads the vector with zeros to a multiple of R and a rank sends 2 (R - 1) chunks of one R-th of it; in the coordinator star rank 0 receives (R - 1) whole vectors (and sends as many back). 'Full' is the 337 numbers of the base run, 'adapters' the 128 numbers of the adapter run.

| R | rows of the busiest rank | its FLOPs, keep | its FLOPs, segments of 2 | full: ring bytes sent per rank | full: star bytes received by rank 0 | adapters: ring bytes per rank | adapters: star bytes received by rank 0 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 10 | 17760 | 23680 | 0 | 0 | 0 | 0 |
| 2 | 5 | 8880 | 11840 | 2704 | 2696 | 1024 | 1024 |
| 3 | 4 | 7104 | 9472 | 3616 | 5392 | 1376 | 2048 |
| 4 | 3 | 5328 | 7104 | 4080 | 8088 | 1536 | 3072 |
<!-- /worked -->

The busiest rank is rank 0, with the most rows: at $R = 4$ it holds 3 of the 10 rows, and it does 7,104 FLOPs with segments of 2 layers where one rank did 23,680. It also sends 4,080 bytes in the ring, more than half a byte for every FLOP (under 2 FLOPs for every byte), so a step of this network is mostly bytes moved, and the project measures equivalence and not speed. What the table counts is the structure. The ring's bytes per rank grow slowly with $R$ (2,704, 3,616 and 4,080 bytes at 2, 3 and 4 ranks) while the star's rank 0 receives bytes proportionally to $R - 1$ (2,696, 5,392 and 8,088), and at the adapter run's 128 numbers both drop to under half of that of the full run (1,536 bytes per rank in the ring at $R = 4$ against 4,080): the adapters are all that the ranks need to agree on, a second reason to freeze the base besides the memory it saves.

## A frozen base and 128 trainable numbers

The second stage trains a change to the base and not the base. The adapter of Chapter 4 adds $s\,\mathbf{B}\mathbf{A}$ to a layer's weight, with $\mathbf{A}$ of shape $[r, \text{inputs}]$, $\mathbf{B}$ of shape $[\text{outputs}, r]$, rank $r = 2$ and scale $s = \alpha / r = 1$ (with $\alpha = r$). It sits on layers 1 to 4, the four 8-by-8 layers, so that the layers that read and write the data are untouched, and it is not formed as a matrix: the layer computes $\mathbf{t} = \mathbf{A}\mathbf{x}$ first, adds $s\,\mathbf{B}\mathbf{t}$ to the base sum, and its backward pass does the products of `layer_backward` above, which never materialise a $[\text{outputs}, \text{inputs}]$ gradient.

<!-- worked: project_scaling_training.adapter_layout -->
Rank r = 2, scale s = alpha / r = 1.0 (alpha = r), on layers 1 to 4 (the 8 by 8 layers). A layer's adapter is A of [2, in] and then B of [out, 2] in the flat vector of the 128 adapter numbers. In the forward pass t = A x (sums from 0.0 in increasing index), and the neuron's sum is the base sum (as in milestone 1) plus s times the sum over r in increasing order of B[o][r] t[r] (from 0.0), and then max(., 0). The generator `project_scaling_training.adapters` (seed 11641845542066936341) draws, for layers 1 to 4 in order, one uniform() for each entry of A row-major, A = (2u - 1) x 0.5, so 64 numbers are drawn; B starts at zero (the planted mistake below draws B in the same way right after the A of the same layer).

| layer | A shape | B shape | numbers | offset |
|---:|---|---|---:|---:|
| 1 | 2 x 8 | 8 x 2 | 32 | 0 |
| 2 | 2 x 8 | 8 x 2 | 32 | 32 |
| 3 | 2 x 8 | 8 x 2 | 32 | 64 |
| 4 | 2 x 8 | 8 x 2 | 32 | 96 |
| all |  |  | 128 |  |

With the trained base of milestone 2 the loss against T1 is 0.05873598296 for the base alone and 0.05873598296 with the adapters at their start (a difference of 0); with a nonzero B (every entry drawn as (2u - 1) x 0.5) it is 0.0530558. A gradient check on the 10 rows of milestone 1 with B drawn that way compares all 128 analytic entries with central differences of step 1e-6: it passes (below 1e-6). At B = 0 the gradient of the same rows has 0 nonzero entries among the 64 of A and 56 among the 64 of B.
<!-- /worked -->

The start is the property that makes a frozen-base run a continuation: with $\mathbf{B} = \mathbf{0}$ the network is the base to the bit (a difference of 0 in the loss), and a nonzero $\mathbf{B}$ is not (0.0530558 against 0.0587360). At $\mathbf{B} = \mathbf{0}$ the gradient of $\mathbf{A}$ is exactly zero (0 nonzero entries of 64) and that of $\mathbf{B}$ is not (56 of 64 on those ten rows), which is why one factor starts at zero and the other does not, and the gradient check passes with a nonzero $\mathbf{B}$, which tests all 128 entries.

The adapter run takes the 337 numbers of milestone 2 as a base that no update touches, trains the 128 adapter numbers for 60 updates against `T1` with the same Adam, and draws its rows from the order of `project_scaling_training.order_ft`. The data-parallel step is milestone 4's, with `train_base` false, so that `layer_backward` forms the gradients of the adapters and the input gradients that reach them and nothing else.

<!-- worked: project_scaling_training.adapter_runs -->
The adapter run: the base is the trained vector of milestone 2 (after 120 updates, from the same nudged start), frozen; the 128 adapter numbers are trained for 60 updates against T1 with the Adam of milestone 2 (rate 0.01), the draws from the order of `project_scaling_training.order_ft` (seed 12933484095127756386), batches of 10 as before. The deviation is the largest absolute difference of the 128 adapter numbers from those of the one-rank run, and the criterion is that it is below 1e-9 for the run and its four nudged copies. After the 60 updates the digest of the base, written as 64-bit floats, is unchanged. A number that comes from a chaotic run is printed with only the digits that are the same in the run and in four copies of it whose first weight is changed by +1e-12, -1e-12, +3e-12 and -3e-12: in full if all agree on it, as 'about x' with the leading digits on which all agree, or as 'unstable' when not even one digit survives; the other digits cannot be reproduced in another implementation.

| run | deviation of the adapters | loss against T1 | loss against T0 |
|---|---|---:|---:|
| the base alone (no adapter) | - | 0.058736 | 0.00280691 |
| 1 rank, every activation kept | passes (below 1e-9) | 0.00449672 | 0.056375 |
| 4 ranks, ring, segments of 2 layers | passes (below 1e-9) | 0.00449672 | 0.056375 |

Three planted mistakes on one rank. 'After 60 updates' is the largest absolute adapter number for the first row, the loss against T1 before the first update for the second and the largest absolute change of a base number for the third.

| build | check | loss against T1 after 60 updates |
|---|---:|---:|
| A and B both start at zero | 0 | 0.058736 |
| B starts nonzero | 0.0530558 | 0.00549287 |
| the base is trained too (465 numbers) | 0.236 | 0.00138464 |
<!-- /worked -->

The adapters take the loss against `T1` from 0.0587 to 0.00450, and the 4-rank build with replayed segments of two layers ends within $10^{-9}$ of the one-rank build in all 128 numbers and at the same loss in six digits. The base after the 60 updates has the digest it had before them. The loss against `T0` rises from 0.00281 to 0.0564, which is what a change to the function costs the function it replaces; an adapter run is judged on the task it was trained for.

The three mistakes are the three ways the frozen-base start fails. If both factors start at zero, nothing ever moves (largest absolute adapter number 0, the loss stays at 0.0587). If $\mathbf{B}$ starts nonzero, the network is not the base at the first update (the loss against `T1` before any update is 0.0530558, not 0.0587), the run is no longer a continuation, and what it ends at (0.00549) is a worse loss than the right build's 0.00450. And if the base is trained as well, all 465 numbers move: the base differs from the frozen one by 0.236 at its largest entry, and the run reaches a lower loss (0.00138) than any adapter build, again a better number from a different run. The frozen base is checked by its digest and not by the loss.

**Milestone 5 ends** when the start of your adapters reproduces the layout, the loss before any update and the gradient counts; your 1-rank and 4-rank adapter runs agree with each other to $10^{-9}$ and reproduce the two losses; and the three mistakes end where the second table puts them.

## Killing the run and bringing it back

The adapter run is the one that is killed. The state it needs to go on is what Chapter 7 listed for Adam: the 128 numbers (its weights), the 128 first moments, the 128 second moments, the update count $t$ and the number of draws $d$ (the base is an input and the seed is a constant of the run, and neither changes). The checkpoint stores them in three files, written into a temporary directory, in this order:

1. `values.bin`: the 128 adapter numbers as 64-bit floats, little-endian.
2. `state.bin`: the 128 first moments, the 128 second moments, then $t$ and $d$ as 64-bit unsigned integers, little-endian.
3. `meta.json`: the JSON text, keys sorted and no spaces, `{"base":"<h>","state":{"digest":"<h>","length":<n>},"step":<t>,"values":{"digest":"<h>","length":<n>}}`, where each `<h>` is a 64-bit FNV-1a digest as sixteen lower-case hexadecimal digits (in quotes), `<n>` is a length in bytes, `<t>` is the update count and `base` is the digest of the base's 337 numbers written as 64-bit floats. With the lengths of the next table it is 142 bytes.

The directory is then renamed to `gen-<t>` (the update count in 4 digits). After a commit the two newest generations that verify are kept. A reader opens the generations from the newest and accepts the first whose meta parses, whose base digest is the base of this run, and whose two files have the recorded lengths and digests; if none does there is nothing to resume from.

```text
function commit(st: AdapterRun, base_digest: text, disk: map<text, bytes>) -> none:
    v = bytes of st.p, each as a 64-bit float
    s = concat(bytes of st.m, bytes of st.v, bytes of st.t and st.d as u64)
    tmp = "tmp-" + four digits of st.t
    disk[tmp + "/values.bin"] = v
    disk[tmp + "/state.bin"] = s
    disk[tmp + "/meta.json"] = meta_text(st.t, base_digest, v, s)
    rename every name under tmp + "/" to the same name under "gen-" + four digits of st.t
    keep the two newest generations that verify, and delete every other name

function restore(disk: map<text, bytes>, base_digest: text, check: bool)
        -> optional AdapterRun:
    for gen in the generation names of disk, newest first:
        if the meta of gen parses and its base digest equals base_digest:
            if not check or both files of gen have the recorded length and fnv1a64:
                return the AdapterRun decoded from the two files of gen
    return nothing
```

`AdapterRun` is the record `(p, m, v, t, d)` of the adapter run (the 128 numbers, their two moments, the update count and the draw count), `meta_text` is the text of the third file, and a name sorts newest first because the update count is written in 4 digits. The routine is Chapter 7's reader with one more check, the digest of the base: a generation of adapters is the continuation of one base only, and a generation written against another base is not a generation of this run. A disk is a map from path to bytes in the worked module, so a kill is a prefix of the operations of a commit:

<!-- worked: project_scaling_training.commit_protocol -->
The adapter run on one rank with a checkpoint every 10 updates. A commit writes three files into a temporary directory `tmp-<step>`, in this order, and renames it to `gen-<step>` (4-digit step): `values.bin`, `state.bin` and, last, `meta.json`. A disk is a map from path to bytes. After a commit the two newest generations that verify are kept. The base is the trained vector of milestone 2; its FNV-1a digest, e34c25615456cb12, is a field of every meta file and a generation whose base differs is not accepted. A reader opens generations from the newest, accepts the first whose meta parses, whose base digest matches and whose two files have the recorded lengths and FNV-1a digests, and reads nothing else. After the updates 10, 20 and 30 are committed the disk holds gen-0030, gen-0020 (step 10 was pruned). The files of the commit of step 40:

| file | bytes | content |
|---|---:|---|
| values.bin | 1024 | the 128 adapter numbers as 64-bit floats (128 x 8) |
| state.bin | 2064 | m (128 x 8), v (128 x 8), then t and d as 64-bit unsigned integers (2 x 8) |
| meta.json | 142 | keys sorted, no spaces: base digest, step, length and FNV-1a digest of the two files |

The process is killed after each prefix of that commit's operations (everything written before the kill survives, nothing after; power loss is the subject of Chapter 7):

| killed after | generations on the disk | generation the reader accepts |
|---|---|---|
| nothing written | gen-0030, gen-0020 | gen-0030 (step 30) |
| values.bin written | gen-0030, gen-0020 | gen-0030 (step 30) |
| state.bin written | gen-0030, gen-0020 | gen-0030 (step 30) |
| meta.json written | gen-0030, gen-0020 | gen-0030 (step 30) |
| directory renamed to gen-0040 | gen-0040, gen-0030, gen-0020 | gen-0040 (step 40) |
<!-- /worked -->

The commit has four operations and a kill after any of the first three leaves the disk as it was, to a reader: the newest generation it accepts is the update 30. After the fourth, the rename, the update 40 is there. That is the property that a reader of Chapter 7's crash table counted for a larger set of failures: the meta file last and the rename after it, so that nothing in the directory of the commit is visible until all of it is true.

The resume experiment kills the 4-rank adapter run after 37 updates, which leaves the generations 20 and 30 on the disk, drops everything in memory, restores from the disk, changes what the row says, and carries on to update 60 with the same step function.

<!-- worked: project_scaling_training.resume_runs -->
The adapter run of milestone 5 on 4 ranks (ring, segments of 2 layers), 60 updates, a checkpoint every 10 updates as in the previous table. The run is killed after the stated number of updates (37, or 47 for the last two rows), everything in memory is lost, and a second process restores from the disk, applies the change named in the first column, and carries on to update 60 with the same step function. 'Same bits' compares the 128 adapter numbers, as 64-bit patterns, with the uninterrupted 4-rank run; the deviation is the largest absolute difference from it. The last column compares with the one-rank run of milestone 5 (criterion below 1e-9, for the run and its four nudged copies). The damage is that a power loss left the last 16 bytes of the newest `values.bin` as zeros. A number that comes from a chaotic run is printed with only the digits that are the same in the run and in four copies of it whose first weight is changed by +1e-12, -1e-12, +3e-12 and -3e-12: in full if all agree on it, as 'about x' with the leading digits on which all agree, or as 'unstable' when not even one digit survives; the other digits cannot be reproduced in another implementation.

| what the second process does | generation restored | same bits as the uninterrupted run | deviation from it | against the 1-rank run |
|---|---|---:|---:|---|
| every part restored | gen-0030 | yes | 0 | passes (below 1e-9) |
| first and second moments set to zero | gen-0030 | no | 0.18 | FAILS |
| examples drawn set to zero | gen-0030 | no | 0.022 | FAILS |
| resumed on 2 ranks (ring) instead of 4 | gen-0030 | no | below 1e-9 | passes (below 1e-9) |
| newest generation damaged, digests checked | gen-0030 | yes | 0 | passes (below 1e-9) |
| newest generation damaged, accepted by name | gen-0040 | no | 0.0116 | FAILS |
<!-- /worked -->

The first row is the transaction: the run resumes from the update 30, repeats the updates 31 to 37, and ends with the same 128 numbers, to the bit, as the run that was never stopped, and within $10^{-9}$ of the one-rank run. The second and third are Chapter 7's omissions: without the moments the continuation takes the first steps of a new run from the middle of an old one (0.18 away at the end); without the draws it reads the examples again from the start of the epoch order (0.022). The fourth is the one to read twice. A resume on two ranks instead of four goes on correctly, within $10^{-9}$ of the one-rank run, and is not the same bits as the run on four: the ring adds in another order, so the run is another run of the same family. The bits are a function of the number of ranks, which is why brain names the world size in the identity of a distributed run (Chapter 5), and the tolerance of this project is looser than the bit comparison on purpose.

The last two rows damage the newest generation, update 40, with what a power loss can leave, the last 16 bytes of `values.bin` as zeros, and kill the run after 47 updates. The reader that recomputes lengths and digests rejects the generation (the digest of `values.bin` is not the one in the meta file), falls back to the update 30, and the run ends with the bits of the run that was not killed. The reader that trusts the name takes the generation of update 40, two of the adapter numbers are zero, and the run ends 0.0116 from the right one. The loss it ends at, 0.0044949 against 0.00449672, would not have shown it.

**Milestone 6 ends** when your generation has the three files at the sizes of the first table (1,024, 2,064 and 142 bytes) and your reader accepts the generation the second table names after each kill; your resume on the same ranks gives the same bits as your run that was never stopped; and the omissions, the other rank count and the damaged generation end where the third table puts them.

## The whole run against the reference

The last milestone assembles everything. The system trains the base on 4 ranks (ring, segments of 2 layers) for 120 updates, freezes it, and trains the 128 adapter numbers on the same 4 ranks for 60 updates with a generation every 10 updates, killed after 37 and resumed. The reference is the one-rank run of milestones 2 and 5, never stopped, every activation kept.

<!-- worked: project_scaling_training.whole_run -->
Both runs of the project end to end. The reference trains the base for 120 updates and then the 128 adapter numbers for 60 on one rank with every activation kept. The system trains the base on 4 ranks (ring, segments of 2 layers), then the adapters on the same 4 ranks with a checkpoint every 10 updates, killed after 37 (or 47) updates and resumed as in milestone 6; each planted mistake is made in the adapter stage, after a correct 4-rank base. The deviation is the largest absolute difference, over the 337 base numbers and the 128 adapter numbers, from the reference run started from the same (nudged) initial vector, and the criterion is that it is below 1e-9 for the run and its four nudged copies. The loss is the mean over the 40 examples against T1 of the final network. A number that comes from a chaotic run is printed with only the digits that are the same in the run and in four copies of it whose first weight is changed by +1e-12, -1e-12, +3e-12 and -3e-12: in full if all agree on it, as 'about x' with the leading digits on which all agree, or as 'unstable' when not even one digit survives; the other digits cannot be reproduced in another implementation.

| build | largest deviation from the reference | loss against T1 |
|---|---:|---:|
| the reference: one rank, every activation kept, never killed | 0 | 0.00449672 |
| the specified system, killed after 37 updates and resumed | passes (below 1e-9) | 0.00449672 |
| a gradient averaged without the row counts (mistake of milestone 4) | 0.0634 | 0.00451237 |
| the base trained beside the adapters (mistake of milestone 5) | 0.35 | 0.00138464 |
| resumed without the moments (mistake of milestone 6) | 0.18 | 0.00419444 |
| resumed from a damaged generation accepted by name (mistake of milestone 6) | 0.0116 | 0.0044949 |
<!-- /worked -->

The first row is the reference and the second is the system: all 465 trained numbers end within $10^{-9}$ of the reference, at the same loss of 0.00449672 against `T1`. The other four are the system with one planted mistake each, with deviations of 0.0634, 0.35, 0.18 and 0.0116 and losses of 0.00451237, 0.00138464, 0.00419444 and 0.0044949 against the reference's 0.00449672. The first is above the reference's loss by 0.35 percent and the last below it by 0.04 percent, and the other two are far below it (by 69 and 6.7 percent), so no direction of the loss is the sign of a mistake.

![The largest deviation of the 128 adapter numbers from the one-rank run after each of 60 updates, on a logarithmic axis from 1e-18 to 1. The specified system stays between 1e-17 and 1e-15, below the line of the criterion at 1e-9. The mean of the shard means is at 0.02 from the first update and rises to 0.07. The run that lost its moments at update 30 is at the rounding floor until update 30 and then jumps to 0.01 and grows to 0.18. The run that accepted a damaged generation of update 40 jumps at update 40 to 0.014.](img/deviation-traces.svg)

The figure shows why the comparison is better than the loss, and when it is better. It is the distance from the one-rank run after every update, for the system and for three of its mistakes. The system sits at the rounding floor of 64-bit arithmetic, below $10^{-15}$ at every update (the line is hidden under the run that accepts the damaged generation, which leaves the floor only at update 40); the unweighted mean is off from the first update; the lost moments leave the floor at the update that loses them and not before; the damaged generation does the same at update 40. A comparison made once, at the end, finds all three; a comparison made after every update finds the last two at the update that caused them.

<!-- worked: project_scaling_training.deviation_table -->
The adapter run of milestone 5 on 4 ranks (ring, segments of 2 layers) from the reference base, and three planted mistakes (the mean of the shard means from the first update; the moments set to zero after update 30; the last two numbers of the adapter vector set to zero after update 40, which is what accepting a generation whose `values.bin` ends in zeros does). Each cell is the largest absolute difference of the 128 adapter numbers after the update named in the column head from the one-rank run after the same update. A number that comes from a chaotic run is printed with only the digits that are the same in the run and in four copies of it whose first weight is changed by +1e-12, -1e-12, +3e-12 and -3e-12: in full if all agree on it, as 'about x' with the leading digits on which all agree, or as 'unstable' when not even one digit survives; the other digits cannot be reproduced in another implementation.

| build | update 1 | update 10 | update 30 | update 31 | update 40 | update 41 | update 60 |
|---|---:|---:|---:|---:|---:|---:|---:|
| the specified system | below 1e-15 | below 1e-15 | below 1e-15 | below 1e-15 | below 1e-15 | below 1e-15 | below 1e-15 |
| mean of the shard means | 0.02 | 0.0519 | 0.0673 | 0.0681 | 0.0738 | 0.0733 | 0.0634 |
| moments lost at update 30 | below 1e-15 | below 1e-15 | below 1e-15 | 0.0122 | 0.0931 | 0.101 | 0.18 |
| damaged generation of update 40 accepted | below 1e-15 | below 1e-15 | below 1e-15 | below 1e-15 | 0.014 | 0.0138 | 0.0116 |
<!-- /worked -->

**Milestone 7 ends** when your complete system reproduces the second row's criterion and the loss, each of your four planted mistakes ends at the row's deviation to the digits printed, and your deviation at every update of the table passes the floor of the first row. Where it does not, the milestone at which the failing number first appears is the one to debug: a deviation above $10^{-9}$ in the first update is the shards or the order, in the update that follows a restart is the state, and in no update at all with a different loss is the loss and not the run.

## What brain does at each stage of the run

Each part of the project has a counterpart in brain, and the evidence for each is the code, not a comment on it.

The order of the examples is a function of the draw index. `Objective::shard` (commit `1f0e66dc5`, 23 October 2026) gives rank $r$ of a world of $R$ the records $n$ of the deterministic stream with $n \bmod R = r$, a record being one example of a dataset that has example boundaries and one drawn batch of a plain token stream (the commit message speaks of batches; the code in `crates/model/src/train.rs` keeps a record at a time). Every rank still steps through every record, and drops its peers', so that all stay on one stream. The unit tests `the_ranks_draw_disjoint_shards_that_union_to_the_unsharded_stream` and `a_shard_of_one_trains_exactly_like_an_unsharded_run` in the same file check that the shards are disjoint, that their interleaving is the stream, and that one rank draws the whole stream. This is milestone 4's dealing of draw $b$ to rank $b \bmod R$, with records as the unit.

The reduction is `reduce_grads_weighted` in `crates/model/src/distributed.rs`, and `fit_controlled_sharded` in `crates/model/src/train.rs` calls it after the micro-steps of a step and before the optimizer step. What a rank's gradient is comes from `micro_steps` (commit `c2a048cbf`, 6 October 2026): every micro-batch's loss is divided by one denominator, the mean count of supervised positions of the rank's own micro-batches in the step, so that a long record weighs what it holds within a rank; after the optimizer's division by the number of accumulated micro-batches (`scale`), the rank's gradient is the sum over its supervised positions divided by their count. The reduction weighs each rank by that count $n_j$, taken from the objective's step work (weight 1 when an objective reports none): it all-reduces the weight and each gradient times the weight, and multiplies the summed gradients by one over the summed weights, so every rank receives $\sum_j n_j \mathbf{g}^{(j)} / \sum_j n_j$, the weighted form of milestone 4 with supervised positions as the rows. Until commit `e399f536b` (14 November 2026) the loop called `reduce_grads`, which divides the sum by the world size and weighs the ranks equally; with ranks of 1 and 3 positions and gradients 1 and 3 it updated along 2 where the per-position mean is 2.5, the shard-mean mistake of milestone 4. The test `reduce_grads_weighted_gives_the_per_token_mean_of_unequal_ranks` in `crates/model/tests/it/distributed_model.rs` checks exactly that case, and `reduce_grads` remains as the equal-weight form.

Recomputation is `SavedPolicy::Recompute` in `crates/model/src/saved.rs`, and the contract of milestone 3 is its shared check: `check_recompute` runs a step under `Keep` and a step under `Recompute` and returns an error unless the loss and every parameter's gradient are equal bit for bit (and, where the backend accounts for its allocations, the live bytes of the peak are fewer). The project's third column is that check on a network that fits on a page.

The adapter is `crates/model/src/lora.rs` (`Pair::project` forms the gradients of the two factors from the gradient of the whole weight, the formulas of Chapter 4) and the device runtime in `crates/model/src/adapter/exec.rs`, and the frozen base is `Role::Frozen` of `crates/paramstore/src/lib.rs`, which allocates only a weight. The test that a pipeline run resumes exactly, `a_pipeline_run_resumes_exactly` in `crates/qwen3/tests/it/lora_pipeline.rs` (added in commit `6c849ced1`, 1 October 2026), is milestone 6 for adapters on a pipeline. It builds a two-layer model with rank-3 adapters as a pipeline whose stages sit on the devices of `stage_gpus()`, which are devices 0 and 1 (two cards, two stages) unless the environment variable `SHARD_TEST_GPUS` names others; it runs 8 updates straight, then a run that stops after update 4 and writes its state file, then a run that resumes from that file. It asserts that the interrupted run reports `interrupted` with the state file on disk, that the resumed run reports `resumed_at == Some(4)`, and that every adapter tensor of the resumed run is within an absolute $10^{-5}$ of the same tensor of the straight run (32-bit numbers, where the project's resume is held to the bit). The test returns at once, printing a SKIP line and asserting nothing, when `MOE_SKIP_GPU_TESTS` is set or when fewer discrete GPUs are present than `stage_gpus()` needs.

The generation is `CheckpointDir` in `crates/optim/src/checkpoint.rs`: a commit writes `values.bin` and `state.bin` and then `meta.json`, renames the directory to `gen-<step>`, and the reader tries the newest generation first, recomputes the length and 64-bit FNV-1a digest of each file and falls back to the next older one (`restore`), with `KEEP = 2` generations that verify retained by `prune`. What the project adds to it is one field, the digest of the base: brain's own generation records the optimizer rule and the table of tensors and refuses a generation that differs from the running run in either.

## Changing one stage and reading the effect

The system is small enough to change one part at a time and read the effect against the reference. The ring can give way to the tensor cut of one matrix product or to a pipeline of stages (Chapter 5), and the criterion still applies, with a tolerance that follows the new order of the sums. The moments can be divided between the ranks as in Chapter 5's sharded optimizer, and the result must stay within the tolerance of the replicated run, where its bits agree with it. The adapters' rank can be moved from 2 to 4 and the planted mistake of a nonzero $\mathbf{B}$ repeated. And the 64-bit arithmetic can be replaced by the narrow formats of Chapter 2, with the tolerance of $10^{-9}$ replaced by one that the format can meet.

## Check your understanding

1. The one-rank run ends with a loss of 0.00281 and the four-rank run that averages the shard means ends with 0.00140. A colleague says the second is better and should be kept. Find the error, and say which numbers show it.
2. The undivided sum of milestone 4 trains a network whose loss is 0.00280622 against 0.00280691 and whose numbers are 0.000318 away from the right ones. Why can the loss not see this mistake, why does the comparison see it, and what changes when Adam's $\epsilon$ is $10^{-300}$?
3. In milestone 3 the late update of a weight of layer 0 leaves the keep gradient unchanged (distance 0) and moves the replayed gradient by 0.06247. A reviewer concludes that keeping everything is the safer policy. What is wrong with that conclusion?
4. The resume on two ranks of milestone 6 ends within $10^{-9}$ of the one-rank run and not at the same bits as the four-rank run. A colleague proposes to loosen the check of the resume on the same ranks to $10^{-9}$ as well. What would that check stop seeing, and which row of the third table is the evidence?
5. The generation of update 40 has its last 16 bytes of `values.bin` zeroed and the reader that trusts the name accepts it. The final loss is 0.0044949 and the right one 0.00449672. What does this show about a loss as a test of a resume, and what check on the reader would have caught the damage before the run continued?
6. The adapters of milestone 5 are trained with the base unfrozen by mistake and reach a loss of 0.00138, lower than any correct build. How is the mistake found without the loss, and which part of the loss's meaning has changed?

## Summary

If every step passes, you have a system that trains a six-layer network in two stages on four ranks, replays its layers in segments of two instead of storing them, trains 128 adapter numbers on a frozen base of 337, and survives a kill after update 37 by restoring the newest generation that verifies, and that ends within $10^{-9}$ of a one-rank run in every one of its 465 trained numbers at a loss of 0.00449672. The planted mistakes of the project are the ways a move to several devices ends up as another run: a mean that weighs shards equally where they hold different numbers of rows (0.389 from the reference), a sum that Adam hides (0.000318), ranks that read the same rows (0.437), adapters that start away from the base or never move, a base that is trained when it should be frozen (its base numbers move by 0.236), moments or draws left behind by a resume (0.18 and 0.022), and a damaged generation read by its name (0.0116). The loss of the final network moves in either direction under these mistakes, or hardly at all, and the comparison of trained numbers shows every one.

Three things limit what the project establishes. The arithmetic is 64-bit, so the comparison measures the order of additions and leaves open what a 16-bit run with stochastic rounding does to the criterion, which Chapter 2 gives the ingredients for. The network has 337 numbers, and the work per rank (under 8,000 FLOPs for 4,080 bytes) is far too small for any measure of speed, so the project establishes that the move does not change what is learned and not that it pays. And the failures are planted ones: a kill after 37 updates, a damage of 16 bytes. Real runs fail in more ways than these, and the generation protocol of Chapter 7 is the part of the argument that covers the rest.

## Answers

1. Weights $n_j / 10$: $0.4$, $0.3$ and $0.3$, since the mean gradient of the ten rows is $\sum_j n_j \mathbf{g}^{(j)} / 10$ with $n = (4, 3, 3)$. The plain average gives each shard $1/3$, which counts a row of a 4-row shard as $0.0833$ of the total and a row of a 3-row shard as $0.111$ (the weights of rows are $1/(3 n_j)$ against the right $1/10$ for every row).
2. In the ring the four numbers of chunk $c$ are added as $((v_c + v_{c+1}) + v_{c+2}) + v_{c+3}$ with the indices modulo 4, and in the coordinator as $(((0 + v_0) + v_1) + v_2) + v_3$. The last bits of a floating-point sum depend on the order, so the two can differ in the last bits and are not bit-comparable across numbers of ranks or topologies. Within one run, every rank must end with the same bits: all of them take one sum (the coordinator) or the finished chunks of one ring (the all-gather phase copies them).
3. The weights: the replay recomputes the layers of a segment from the kept boundary with the weights as they are at that moment. The gradient is the one of keeping everything when the forward is a pure function of its input and of weights that have not changed since the forward (and, in a layer with a random mask, when the mask is redrawn from the same key). Replaying every layer's forward once adds one forward to a step of one forward and two forwards' worth of backward: $4/3$ of the arithmetic, a third more.
4. $\mathbf{B}$ starts at zero and $\mathbf{A}$ starts random. The gradient of $\mathbf{A}$ is $s\,\mathbf{B}^\top d\mathbf{W}$, which is zero when $\mathbf{B}$ is zero; the gradient of $\mathbf{B}$, $s\,d\mathbf{W}\,\mathbf{A}^\top$, is not zero when $\mathbf{A}$ is not. If both started at zero both gradients would be zero and nothing would ever move, and the adapted network starts as the base exactly because $\mathbf{B}\mathbf{A} = \mathbf{0}$.
5. The next update reads the first and second moments $\mathbf{m}$ and $\mathbf{v}$ and the update count $t$ (in the bias corrections, the schedule and any keyed rounding). A reader must check that the generation is whole and is this run's: that its meta parses, that the recorded length and the digest of every file match the bytes, and, in this project, that the digest of the base is the one in use. It must not trust the name of the directory.
6. Draw $d$ is entry $d \bmod 40$ of the shuffled order of epoch $\lfloor d / 40 \rfloor$, the order being a function of the run seed and the epoch alone. A restarted run recomputes the order and reads from entry $d \bmod 40$ of epoch $\lfloor d / 40 \rfloor$, so the count $d$, with the seed, is all of the data's state.

Check your understanding.

1. A loss is a number the run is allowed to look at, and it measures only the objective that the run trained on. The four-rank build weighed a row of a 2-row shard 1.5 times a row of a 3-row shard, so it trained on another objective, whose minimum has a different loss on the 40 examples; the trained numbers are 0.389 from the one-rank run's. The comparison of the numbers after the update 120, against the one-rank run, shows it; the loss does not, and the build that is right ends with 0.00281 and within $10^{-9}$ of the one-rank numbers.
2. Adam's update is the first moment over the square root of the second, so a gradient ten times as large gives nearly the same update, and the loss curve is nearly the same (0.00280622 against 0.00280691). The numbers do not return to the same values because the denominator is $\sqrt{\hat{v}} + \epsilon$ and for gradients of the order of $10^{-8}$ the $\epsilon$ is not small next to them, so a factor of ten changes the step. With $\epsilon = 10^{-300}$ the two runs end within $10^{-9}$ of each other, which is the table's evidence.
3. The keep build did not read that weight in its backward pass, and the replay did. That the keep gradient is unchanged says that this weight is not used by the backward of keep, not that keep is immune to a late update: the late change of the layer-1 weight, which both read, moves both by the same 0.008829. A trainer must not change a weight between a forward and the end of its backward whichever policy it uses, and the replay makes that condition visible for more weights.
4. It would stop telling a resume that restored everything from a continuation that is merely close. The same-rank resume must be bit-identical because every addition it does is in an order that is fixed, so any difference at all is a lost part of the state, and a lost part that moves the numbers by less than $10^{-9}$ would pass the loosened check and not the bit comparison. The evidence is the row of the resume on two ranks: it is within $10^{-9}$ of the one-rank run and not the same bits as the four-rank run, so it passes the tolerance and fails the bit comparison, as a different run should.
5. That a loss is a weak test of a resume: the damage of 16 bytes moved the 128 numbers by 0.0116 and the final loss by 0.00002 (0.0044949 against 0.00449672). The reader must recompute the length and the FNV-1a digest of both files and compare them with the meta file before accepting a generation, and fall back to the next older one when they differ, which is what the reader with the digests did in the fifth row.
6. The base's digest changes, and the largest change of a base number is 0.236, so a comparison of the base against its saved digest fails before the run's loss is read. The loss now measures a network of 465 trained numbers fitted to the 40 examples of `T1`, not 128 numbers added to a base that was fitted to `T0`: it is lower because more is fitted, and it no longer says what the adapters learned.
