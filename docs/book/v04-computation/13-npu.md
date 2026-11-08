<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Neural Processing Units and Graph Compilers

*How does a whole network become one compiled program for a fixed-function accelerator?*

A small camera board has to find objects in every frame. Its general-purpose processor is one small core, with a few tens of megabytes of memory (the roadmap `.agents/roadmap/rknpu.md` records 33 MB visible to the operating system on the reference board), and a detector asks for far more arithmetic per image than that core is built for. Next to the processor sits a block of silicon built for one job: a **neural processing unit** (NPU), which multiplies and adds 8-bit integers in wide arrays, with a small on-chip buffer for its input, and does very little else. It has no floating-point unit and no operating system, and it does not run Python. What it runs is a list of command words, written into memory beforehand, that say which numbers to read from where, which constants to use, and where to put the result.

Everything the earlier chapters did one kernel at a time (Chapters 5 to 9) must now be settled before the first image arrives. This chapter takes the smallest network that shows every step and carries it from a float model to a list of command words. The network has two layers: a linear layer of 6 inputs and 4 outputs, a rectifier, and a linear layer of 4 inputs and 3 outputs. Its weights come from the book's generator:

<!-- worked: npu.network_weights -->
The weights of the network, seed 15132813171768601218 (seed_of 'npu.network'). Draw order: normal() for W1[0][0], W1[0][1], ... W1[3][5] (row-major, 24 draws), then b1[0..3], then W2[0][0] ... W2[2][3] (12 draws), then b2[0..2]; weights are round(0.6 z, 2), biases round(0.2 z, 2). The printed values are the inputs of everything below.

Layer 1, W1 of shape [4, 6] and b1 of shape [4]:

| row | i=0 | i=1 | i=2 | i=3 | i=4 | i=5 | bias |
|---|---:|---:|---:|---:|---:|---:|---:|
| W1[0] | +1.02 | -0.87 | -0.57 | +0.78 | +0.54 | +0.28 | +0.02 |
| W1[1] | +0.06 | +0.70 | +0.69 | -0.03 | -0.68 | -0.51 | -0.14 |
| W1[2] | +0.24 | -0.24 | -0.21 | +0.62 | +0.24 | +0.49 | +0.36 |
| W1[3] | +0.29 | +0.02 | +0.61 | +0.27 | -0.32 | +0.24 | +0.26 |

Layer 2, W2 of shape [3, 4] and b2 of shape [3]:

| row | i=0 | i=1 | i=2 | i=3 | bias |
|---|---:|---:|---:|---:|---:|
| W2[0] | -0.03 | -1.00 | -0.32 | +0.01 | +0.17 |
| W2[1] | +0.23 | -0.37 | +0.06 | -1.34 | -0.22 |
| W2[2] | +1.37 | -0.21 | -0.14 | +0.42 | +0.00 |
<!-- /worked -->

Run one input through it in floating point, as Volume III taught. These are the numbers the integer program at the end of the chapter must come close to:

<!-- worked: npu.float_forward -->
One input, x_i = round(normal(), 3) for i = 0 .. 5 from the generator, seed 6064223036259098784 (seed_of 'npu.example'). The float forward pass of the network above (binary64, dot products summed from index 0, bias added last).

| step | value |
|---|---|
| x | [-0.065, -2.366, -1.095, -0.016, +1.128, +1.349] |
| h1 = W1 x + b1 | [+3.6106, -4.0102, +2.0640, -0.5156] |
| a1 = relu(h1) | [+3.6106, +0.0000, +2.0640, +0.0000] |
| y = W2 a1 + b2 | [-0.5988, +0.7343, +4.6576] |
<!-- /worked -->

Three questions stand between this table and a device that knows only integers and command words.

1. *What exactly is the program?* The model is Python code; the device cannot run it. The computation must become data that lists every operation, every tensor shape and every constant (graph export).
2. *What are the numbers?* The device multiplies integers. Every float weight and every float activation needs a scale and a zero point (Chapter 12), and the activations, unlike the weights, are not known until data has passed through (post-training quantization).
3. *Where does everything go?* The device has a small input buffer, a fixed layout for feature maps and no allocator. Which operations run together, how a layer is cut to fit the buffer, which tensors share which bytes and what the command words contain must all be decided ahead of time (the graph compiler).

![Five boxes in a row joined by arrows: the model with float weights and code; the exported graph, 172 bytes of f32; the quantized graph, 85 bytes plus ranges; the command stream, 2 tasks of 1,088 bytes, 2,176 bytes; and the device that runs the stream with no host code. The arrows are labelled export, quantize, compile and submit. Under each box a line names what is decided at that stage: nothing yet; operators, order and shapes; ranges, scales, zero points and codes; fusion, tiles, addresses and words; only the values of the input.](img/npu-pipeline.svg)

The figure is the map of the chapter. Each arrow is a step that runs once, on a host, before deployment; what remains for the device is the last box.

Six older ideas carry the road from here to the command words: affine quantization and its error, the narrow sum, fusion, ahead-of-time compilation and the computational graph. Bring them back from memory; the answers are at the end of the chapter.

1. (Chapter 12, affine quantization.) A tensor takes values in $[-0.5, 1.5]$ and is stored in unsigned 8-bit codes $0 \ldots 255$. What are the scale $\Delta$ and the zero point $q_0$?
2. (Chapter 12, quantization error.) A value is rounded to the nearest code of a grid with step $\Delta$. What is the largest error, and what happens to a value outside the range the codes cover?
3. (Chapter 1, narrow sums.) Why are the products of 8-bit integers added in a 32-bit accumulator and not an 8-bit or 16-bit one?
4. (Chapter 8, fusion.) A kernel writes a 1 MiB intermediate tensor and the next kernel reads it back. How many bytes cross memory for that tensor, and how many after the two kernels are fused?
5. (Chapter 9, ahead-of-time compilation.) What is done before a program ships, and what does the program do at run time, when kernels are compiled ahead of time?
6. (Volume III, Chapter 4, computational graph.) What are the nodes and the edges of a computational graph, and in what order does the forward pass visit the nodes?

## Graph export: the whole network as data

A model written as a program decides things while it runs: it loops over layers, branches on a flag, reads a tensor's shape from the data. A fixed-function device cannot follow any of that. It needs the whole network at once, as data, before the first input arrives. **Graph export** is the one-time conversion of a model's forward computation into such a description: a static dataflow graph in which every operation is a node of a known type, every tensor has a fixed shape, and every constant (weights, biases) is stored in the graph itself. The exported graph is the computational graph of Volume III, Chapter 4, with the values of the parameters inside it and nothing left to the program that built it.

The network of the opening, exported, can be written by hand:

<!-- worked: npu.graph_listing -->
```text
graph tiny_mlp
  input   x : f32[1, 6]
  output  y : f32[1, 3]
  initializer W1 : f32[4, 6]    b1 : f32[4]
  initializer W2 : f32[3, 4]    b2 : f32[3]
  node 0  Gemm(x, W1, b1)   attr transB = 1   -> h1 : f32[1, 4]
  node 1  Relu(h1)                             -> a1 : f32[1, 4]
  node 2  Gemm(a1, W2, b2)  attr transB = 1   -> y  : f32[1, 3]
  opset 13, ir_version 8, every dimension static
```

Counted as in Volume IV, Chapter 8: a Gemm of a vector with an [out, in] weight does 2 x out x in FLOPs for the products and sums plus out for the bias; every distinct operand is read once and the result written once, in float32 (4 bytes). The exported file carries 43 initializer values, 172 bytes of float32.

| node | op | inputs | output | initializer values | FLOPs | bytes read | bytes written |
|---|---|---|---|---|---:|---:|---:|
| 0 | Gemm | x, W1, b1 | h1 [1, 4] | 28 | 52 | 136 | 16 |
| 1 | Relu | h1 | a1 [1, 4] | 0 | 4 | 16 | 16 |
| 2 | Gemm | a1, W2, b2 | y [1, 3] | 15 | 27 | 76 | 12 |
| total |  |  |  | 43 | 83 | 228 | 44 |
<!-- /worked -->

Read the listing as a complete program. It names the input and the output with their types and shapes; it lists the constants; it lists three nodes in an order in which every input is defined before it is used, each naming its operator, the tensors it reads and the tensor it writes. Nothing else is needed to run it, and nothing in it depends on a value that is only known at run time. That is what static means here: every shape is a number in the file.

### The ONNX graph

The exporter and the compiler should not need to know each other, so the description needs a form that tools and accelerators agree on. The **ONNX graph** is the widely used file form of such a description. A file holds one model, and the model holds one graph made of four kinds of entry, each of which appears in the listing above:

```text
record ValueInfo:
    name: text
    elem: text                 # "f32", "i8", "i32", ...
    dims: list<int>            # the shape; in an exported graph every entry is >= 1

record Initializer:           # a constant tensor stored in the graph
    name: text
    elem: text
    dims: list<int>
    data: bytes                # row-major, little-endian, numel(dims) * width(elem) bytes

record Node:
    op: text                   # an operator of the declared opset: "Gemm", "Relu", "Conv", ...
    inputs: list<text>         # names of tensors: graph inputs, initializers or earlier outputs
    output: text
    attrs: list<(text, int)>

record ExportedGraph:
    inputs: list<ValueInfo>
    outputs: list<ValueInfo>
    initializers: list<Initializer>
    nodes: list<Node>          # in an order that defines every input before use
    opset: int                 # the version that fixes the meaning of every operator
```

A tensor is a name; the edges of the graph are the names that appear as an output of one node and an input of another. An **initializer** is a constant tensor (a weight or a bias) stored under its name; the listing shows the shape and the file holds the bytes. The **opset** number matters more than it looks: it fixes what each operator means. A graph that says `Gemm` at opset 13 means the same thing to every reader of opset 13, and an operator or an attribute introduced in a later opset does not exist for it. We will meet a case where that limit shaped a design.

A reader of an exported file cannot trust it. Before anything is compiled, the graph is checked:

```text
function check_graph(g: ExportedGraph) -> none:
    defined = [v.name for v in g.inputs]
    for t in g.initializers:
        append t.name to defined
    for node in g.nodes:
        for name in node.inputs:
            require any([d == name for d in defined]), "a node reads a tensor nobody has defined"
        require not any([d == node.output for d in defined]), "a tensor is written twice"
        append node.output to defined
    for v in g.inputs:
        require all([d >= 1 for d in v.dims]), "a dimension is not a fixed positive number"
    for v in g.outputs:
        require all([d >= 1 for d in v.dims]), "a dimension is not a fixed positive number"
    for t in g.initializers:
        require len(t.data) == numel(t.dims) * width(t.elem), "an initializer has the wrong number of bytes"
    for v in g.outputs:
        require any([d == v.name for d in defined]), "an output is never produced"
```

Here `width(elem)` is the bytes of one element (4 for `f32`, 1 for `i8`). The checks are the three promises of the export: every tensor has exactly one producer, every shape is static, and every byte of every constant is present. Later stages rely on them without looking again.

### Why the device needs the whole graph

Three consequences of a static graph carry the rest of the chapter.

- **No host code in the loop.** Once the graph is exported, the numbers that the device will compute are determined by the graph and the input alone. There is no step at which the device must ask a processor to decide something.
- **Every size is known.** The table above counts, before any input exists, the FLOPs and the bytes of each node (the accounting of Chapter 8). A compiler can therefore plan memory, cut layers into tiles and count commands by arithmetic.
- **The compiler sees producers and consumers together.** A graph names which node reads which node's output. That is what allows two nodes to be fused, a tensor to be given a lifetime and a buffer to be shared. A model run op by op from Python offers none of this: each op is a separate request whose consumer is not yet known.

What export gives up is everything that depends on data: a loop whose length is read from the input, a branch on a value. Such constructs are removed by fixing them (a sequence length becomes a fixed capacity; an input is padded or resized to the fixed shape) before the graph is exported. The exported graph is exactly as general as the shapes it was built with.

## Post-training quantization

Chapter 12 quantized the weights and left the activations in floating point, because a weight tensor is known when the model is saved. The device needs more: its multipliers take integers on both sides. **Post-training quantization** converts a trained float network into an integer network without further training, by choosing the scale and zero point of each weight tensor from the weights and of each activation tensor from ranges measured on data. The weights need one pass over numbers already in hand. An activation depends on the input, so its range must be found by running inputs.

### The calibration set

The inputs used to measure the ranges are the **calibration set**: a small sample of inputs, representative of what the device will see, run through the float graph with every tensor recorded. Take 32 inputs of the kind used throughout this chapter and record the four tensors of the exported graph for each:

<!-- worked: npu.calibration_ranges -->
A calibration set of 32 inputs, each x_i = round(normal(), 3) for i = 0 .. 5, drawn in order from the generator, seed 12490842275477104722 (seed_of 'npu.calibration'). Each float tensor is recorded for every input; 'values' counts all numbers recorded for the tensor.

| tensor | shape | values | min | max |
|---|---|---:|---:|---:|
| x | [1, 6] | 192 | -2.7480 | +2.3410 |
| h1 | [1, 4] | 128 | -2.8369 | +3.8403 |
| a1 | [1, 4] | 128 | +0.0000 | +3.8403 |
| y | [1, 3] | 96 | -4.2642 | +5.2031 |
<!-- /worked -->

The routine is the forward pass of Volume III with an observer attached to every tensor (`run_float(g, x)` is that forward pass on the exported graph, returning the values of every tensor as one vector each, and `n_tensors(g)` is the number of distinct tensor names):

```text
function calibrate(g: ExportedGraph, cal: list<array<real>>, tail: real) -> list<(real, real)>:
    require len(cal) >= 1, "a calibration set needs at least one input"
    require 0.0 <= tail < 0.5, "the tails clipped from the two ends must leave something"
    seen = [[] for t in 0 .. n_tensors(g)]               # all values recorded for tensor t
    for x in cal:
        values = run_float(g, x)                          # the value of every tensor, in node order
        for t in 0 .. n_tensors(g):
            for v in values[t]:
                append v to seen[t]
    return [(percentile(seen[t], tail), percentile(seen[t], 1.0 - tail)) for t in 0 .. n_tensors(g)]
```

With `tail` equal to 0 the pair is the smallest and largest value ever seen, because `percentile(a, 0)` and `percentile(a, 1)` are the smallest and the largest element. A positive `tail` ignores that fraction of the recorded values at each end; a later section asks when that helps.

### Scales and zero points

The ranges are turned into codes by the affine rule of Chapter 12, now with the signed 8-bit codes $-128 \ldots 127$ that a device multiplies. Chapter 12 used unsigned codes $0 \ldots 2^{n_b} - 1$; a signed code is the unsigned one less 128, so the rule shifts by 128:

$$\Delta = \frac{\mathrm{hi} - \mathrm{lo}}{255}, \qquad q_0 = \mathrm{clamp}\bigl(\mathrm{round\_even}(-128 - \mathrm{lo}/\Delta),\; -128,\; 127\bigr), \qquad q = \mathrm{clamp}\bigl(\mathrm{round\_even}(r/\Delta) + q_0,\; -128,\; 127\bigr),$$

with the range first widened to contain 0 ($\mathrm{lo} \leftarrow \min(\mathrm{lo}, 0)$, $\mathrm{hi} \leftarrow \max(\mathrm{hi}, 0)$) so that 0 is exactly a code. The real number of code $q$ is $\Delta\,(q - q_0)$. The lowest code, $-128$, stands for $\mathrm{lo}$ (to within half a step) and 127 for $\mathrm{hi}$. The three tensors that the device stores, the input $x$, the hidden activation $a$ and the output $y$, get their own $(\Delta, q_0)$ pair; the pre-rectifier tensor $h_1$ is not stored, because the rectifier will be applied to the accumulator (see below), so it needs none.

<!-- worked: npu.activation_params -->
The calibration ranges of the previous block (min and max over the 32 inputs), widened to contain 0, turned into signed int8 codes: Delta = (hi - lo) / 255, q0 = round_even(-128 - lo / Delta), clamped to -128 .. 127. The last two columns are the real numbers of the lowest and highest code, Delta (-128 - q0) and Delta (127 - q0). h1 is not stored (the rectifier is fused into layer 1), so it has no row.

| tensor | lo | hi | Delta | q0 | code -128 means | code 127 means |
|---|---:|---:|---:|---:|---:|---:|
| x | -2.7480 | +2.3410 | 0.019957 | 10 | -2.7540 | +2.3350 |
| a1 | +0.0000 | +3.8403 | 0.015060 | -128 | +0.0000 | +3.8403 |
| y | -4.2642 | +5.2031 | 0.037127 | -13 | -4.2696 | +5.1977 |
<!-- /worked -->

The hidden activation $a$ is the output of a rectifier, so its range starts at 0, and the rule gives $q_0 = -128$: the lowest code is exactly 0, and all 256 codes are spent on the non-negative half. The input and the output are two-sided and get zero points 10 and $-13$.

The weights are quantized per output channel, one scale for each row of $W$, with symmetric codes $-127 \ldots 127$ (Chapter 12's symmetric rule with a group equal to a row), so that they need no zero point. The reason is what an offset costs in a product, which the next section shows: a weight zero point must be subtracted from every weight before the multiplication, or corrected afterwards by a term that depends on the input. The biases are not quantized to 8 bits; they are added to a 32-bit accumulator, so each is stored as an integer in the units of the accumulator, which are the units $\Delta_{\mathrm{in}}\Delta_o$ of one product of an input code and a weight code:

<!-- worked: npu.weight_codes -->
Per-output-channel symmetric int8 weights, Delta_o = max(abs(row)) / 127, code = round_even(w / Delta_o). Biases are stored as int32 codes at the scale of the accumulator, Delta_in Delta_o: bq = round_even(b / (Delta_in Delta_o)), with Delta_in = Delta_x for layer 1 and Delta_a for layer 2 (values of the activation table above).

| row | max abs | Delta_o | codes |
|---|---:|---:|---|
| W1[0] | 1.02 | 0.008031 | +127 -108 -71 +97 +67 +35 |
| W1[1] | 0.70 | 0.005512 | +11 +127 +125 -5 -123 -93 |
| W1[2] | 0.62 | 0.004882 | +49 -49 -43 +127 +49 +100 |
| W1[3] | 0.61 | 0.004803 | +60 +4 +127 +56 -67 +50 |
| W2[0] | 1.00 | 0.007874 | -4 -127 -41 +1 |
| W2[1] | 1.34 | 0.010551 | +22 -35 +6 -127 |
| W2[2] | 1.37 | 0.010787 | +127 -19 -13 +39 |

| bias | value | accumulator scale | bq |
|---|---:|---:|---:|
| b1[0] | +0.02 | 1.603e-04 | 125 |
| b1[1] | -0.14 | 1.100e-04 | -1273 |
| b1[2] | +0.36 | 9.743e-05 | 3695 |
| b1[3] | +0.26 | 9.586e-05 | 2712 |
| b2[0] | +0.17 | 1.186e-04 | 1434 |
| b2[1] | -0.22 | 1.589e-04 | -1384 |
| b2[2] | +0.00 | 1.625e-04 | 0 |
<!-- /worked -->

```text
function quantize_weights(W: array<real>[O, K]) -> (codes: array<int>[O, K], delta: array<real>[O]):
    codes = zeros(int, [O, K])
    delta = zeros(real, [O])
    for o in 0 .. O:
        delta[o] = max([abs(W[o, i]) for i in 0 .. K]) / 127
        for i in 0 .. K:
            codes[o, i] = clamp(round_even(W[o, i] / delta[o]), -127, 127)
    return (codes, delta)
```

A row of zeros would have scale 0; the exporter gives such a row any positive scale, since all its codes are 0.

### One integer layer

A quantized layer should be integer arithmetic from the input codes to the output code, with a single real number left over to be found. Take output channel $o$ of a linear layer with $K$ inputs. In real numbers $y_o = \sum_i w_{o,i}\,x_i + b_o$. Substitute the decoded values, $x_i = \Delta_{\mathrm{in}}(q_i - q_{0,\mathrm{in}})$ and $w_{o,i} = \Delta_o\,q^w_{o,i}$, and the bias in the units of the product, $b_o = \Delta_{\mathrm{in}}\Delta_o\,b^{\mathrm{q}}_o$ with $b^{\mathrm{q}}_o = \mathrm{round\_even}\bigl(b_o/(\Delta_{\mathrm{in}}\Delta_o)\bigr)$:

$$y_o \;=\; \Delta_{\mathrm{in}}\Delta_o\Bigl(\underbrace{\sum_i (q_i - q_{0,\mathrm{in}})\,q^w_{o,i} + b^{\mathrm{q}}_o}_{A_o}\Bigr).$$

The bracket $A_o$ is the accumulator of Chapter 1: a sum of products of integers plus an integer, exact in 32 bits as long as $K$ is not huge (Exercise 3 finds the limit). The real output is $A_o$ times the one real number $\Delta_{\mathrm{in}}\Delta_o$, so the output code is

$$q_{y,o} = \mathrm{clamp}\bigl(\mathrm{round\_even}(M_o\,A_o) + q_{0,\mathrm{out}},\; -128,\; 127\bigr), \qquad M_o = \frac{\Delta_{\mathrm{in}}\Delta_o}{\Delta_{\mathrm{out}}}.$$

The step from an accumulator to an output code is **requantization**, and $M_o$ is its factor. It is a real number, typically far smaller than 1 (the next block lists them), and the device has no floating point. The remaining work is to write $M_o$ in integers.

The zero point folds into the bias. The sum contains $(q_i - q_{0,\mathrm{in}})$, a subtraction for every input of every output. But $\sum_i (q_i - q_{0,\mathrm{in}})\,q^w_{o,i} = \sum_i q_i\,q^w_{o,i} - q_{0,\mathrm{in}}\sum_i q^w_{o,i}$, and the last term depends only on the weights. The compiler computes it once and adds it to the bias:

$$A_o = \sum_i q_i\,q^w_{o,i} + \tilde b_o, \qquad \tilde b_o = b^{\mathrm{q}}_o - q_{0,\mathrm{in}}\sum_i q^w_{o,i}.$$

The device then multiplies the raw stored codes, with no per-element offset. This is also why the weights here are symmetric: a weight zero point $q_{0,w}$ would add a second term, $-q_{0,w}\sum_i q_i$, that depends on the input and must be computed at run time, or else the device must subtract $q_{0,w}$ from every weight inside its multiplier. (The device of the case study takes the second route: its accumulator is $\sum x\,(w - w_{\mathrm{zp}}) + \text{bias}$, with a zero point per output channel, as its source states.)

To write $M$ with integers, pick an integer multiplier $\hat M$ and a shift $n_{\mathrm{sh}}$ so that $M \approx \hat M\,2^{-n_{\mathrm{sh}}}$. Multiplying $A$ by $\hat M$ is an integer product, and dividing by $2^{n_{\mathrm{sh}}}$ is a rounded shift (the `round_shift` of the pseudocode page: the nearest integer, ties to even, computed exactly). The multiplier has 15 bits (the width of the multiplier field of the case-study device, which the source gives as 1 to 32767); make it as large as 15 bits allow, since a larger multiplier is a finer approximation. The shift is the largest $n_{\mathrm{sh}}$ with $M\,2^{n_{\mathrm{sh}}} \le 32767$, which is $n_{\mathrm{sh}} = \mathrm{exponent\_of}(32767/M)$, and then $\hat M = \mathrm{round\_even}(M\,2^{n_{\mathrm{sh}}})$:

```text
function fixed_point(M: real) -> (mult: int, shift: int):
    require 0.0 < M < 32767.0, "the factor does not fit a 15-bit multiplier"
    shift = exponent_of(32767 / M)
    return (round_even(M * 2^shift), shift)
```

Since $2^{n_{\mathrm{sh}}+1} > 32767/M$, we have $M\,2^{n_{\mathrm{sh}}} > 16383.5$, so $\hat M \in [16384, 32767]$ always has all 15 bits in use, and rounding to an integer moves it by at most $\tfrac12$: the relative error of the factor is at most $0.5/16383.5 \approx 3.05\times 10^{-5}$, about $2^{-15}$. The product $\hat M A$ is below $2^{15}\cdot 2^{31} = 2^{46}$ in magnitude, which fits a 64-bit intermediate.

<!-- worked: npu.fixed_point_table -->
The requantization factor of every output channel, M = Delta_in Delta_o / Delta_out, as a 15-bit multiplier and a shift: shift = exponent_of(32767 / M), multiplier = round_even(M 2^shift), so that 16384 <= multiplier <= 32767. 'relative error' is the absolute difference of multiplier 2^-shift and M, divided by M.

| channel | M | shift | multiplier | multiplier / 2^shift | relative error |
|---|---:|---:|---:|---:|---:|
| layer 1, channel 0 | 0.01064288 | 21 | 22320 | 0.01064301 | 1.16e-05 |
| layer 1, channel 1 | 0.00730394 | 22 | 30635 | 0.00730395 | 2.04e-06 |
| layer 1, channel 2 | 0.00646920 | 22 | 27134 | 0.00646925 | 7.30e-06 |
| layer 1, channel 3 | 0.00636486 | 22 | 26696 | 0.00636482 | 5.99e-06 |
| layer 2, channel 0 | 0.00319405 | 23 | 26794 | 0.00319409 | 1.43e-05 |
| layer 2, channel 1 | 0.00428002 | 22 | 17952 | 0.00428009 | 1.54e-05 |
| layer 2, channel 2 | 0.00437585 | 22 | 18354 | 0.00437593 | 2.03e-05 |
<!-- /worked -->

The factors range from 0.0032 to 0.0106 and the shifts from 21 to 23. The relative errors are all below the bound of $3.05\times 10^{-5}$ (the largest is $2.03\times 10^{-5}$), so the fixed-point factor differs from the real one by a few parts in $10^5$; an output code of size 100 moves by at most about $0.003$ of a step because of it. The requantizer is therefore:

```text
function int_layer(q: array<int>[K], codes: array<int>[O, K], bias: array<int>[O],
                   mult: array<int>[O], shift: array<int>[O], q0_out: int, relu: bool) -> array<int>[O]:
    out = zeros(int, [O])
    for o in 0 .. O:
        acc = bias[o]                                # the folded bias of the formula above
        for i in 0 .. K:
            acc = acc + q[i] * codes[o, i]
        if relu:
            acc = max(acc, 0)
        out[o] = clamp(round_shift(mult[o] * acc, shift[o]) + q0_out, -128, 127)
    return out
```

The `relu` line needs a word. A rectifier applied to the real output, $\max(y_o, 0)$, equals the requantization of $\max(A_o, 0)$, because $y_o = \Delta_{\mathrm{in}}\Delta_o A_o$ with a positive factor: the sign of $y_o$ is the sign of $A_o$. The rectifier is therefore a clamp of the accumulator at zero, and it costs nothing beyond what the requantizer already does. This is the fusion of Chapter 8 in integer form: the rectifier never exists as a tensor.

### The integer network on one input

Now the example of the opening goes through the integer program. The quantized input codes are $q_i = \mathrm{clamp}(\mathrm{round\_even}(x_i/\Delta_x) + q_{0,x}, -128, 127)$; layer 1 accumulates, clamps at zero and requantizes to the codes of $a$; layer 2 does the same without the rectifier and ends in codes of $y$, which decode as $\Delta_y(q - q_{0,y})$.

<!-- worked: npu.one_input -->
The input of the first example, x = [-0.065, -2.366, -1.095, -0.016, +1.128, +1.349], through the network quantized from the 32-input calibration set (Delta_x = 0.019957, q0_x = 10; Delta_a = 0.015060, q0_a = -128; Delta_y = 0.037127, q0_y = -13), one multiplier and shift per output channel. Folded bias of layer 1: [-1345, -1693, 1365, 412]; of layer 2: [-20454, -18536, 17152].

| step | value |
|---|---|
| input codes q_x | [7, -109, -45, 9, 67, 78] |
| accumulators of layer 1, raw codes + folded bias | [22603, -36624, 21210, -5404] |
| the same, (q_x - q0_x) codes + bias code | [22603, -36624, 21210, -5404] |
| after relu, requantized: a1 codes | [113, -128, 9, -128] |
| a1 decoded, Delta_a (q - q0_a) | [+3.6295, +0.0000, +2.0632, +0.0000] |
| float a1 | [+3.6106, +0.0000, +2.0640, +0.0000] |
| accumulators of layer 2, raw codes + folded bias | [-5147, 4740, 28826] |
| the same, (q - q0_a) codes + bias code | [-5147, 4740, 28826] |
| y codes | [-29, 7, 113] |
| y decoded, Delta_y (q - q0_y) | [-0.5940, +0.7425, +4.6779] |
| float y | [-0.5988, +0.7343, +4.6576] |
| error, decoded - float | [+0.0048, +0.0082, +0.0203] |
<!-- /worked -->

The two accumulator rows agree: the raw-code form with the folded bias and the form that subtracts the zero point before the products give the same integers, as the algebra says they must. The output decodes to $[-0.5940, +0.7425, +4.6779]$ against the float $[-0.5988, +0.7343, +4.6576]$: errors of 0.005, 0.008 and 0.020, against half a step of $y$ of 0.0186 (a step is $\Delta_y = 0.037127$). The hidden layer shows where the error starts: the decoded activation $3.6295$ differs from the float $3.6106$ by 0.0189, a little more than one step $\Delta_a = 0.015060$, because each of the six input codes carries up to half a step $\Delta_x/2 \approx 0.01$ of rounding error, weighted by the weights; the second, $2.0632$ against $2.0640$, is off by 0.0008.

One input proves little. The integer program must be right on every input, and "right" has two meanings that are best kept apart. The first is that the compiled integer program computes what the *specification* says, bit for bit. The specification is the direct form: subtract the zero point before the products, and round the requantization as an exact rational. The compiled form folds the bias and uses a shift. They must agree on every output code, with no tolerance, because both are exact integer arithmetic. The second is that the integer network is close to the *float* network, which is a statement about error and has a tolerance. We test both on 256 test inputs:

<!-- worked: npu.exactness -->
The network quantized from the 32-input calibration set run on 256 test inputs, x_i = round(normal(), 3) drawn in order, seed 4800775136902161489 (seed_of 'npu.test'). 'Compiled form' is the integer program of the chapter (raw codes, folded bias, fixed-point requantizer); 'direct reference' subtracts the zero point before each product and rounds the requantization as an exact rational.

| quantity | value |
|---|---:|
| test inputs | 256 |
| outputs compared (3 per input) | 768 |
| inputs where compiled form == direct reference, all 3 codes | 256 of 256 |
| output codes that differ from an ideal real-valued requantizer | 0 of 768 |
| largest absolute accumulator, layer 1 / layer 2 | 32053 / 33776 |
| int32 limit 2^31 - 1 | 2147483647 |
| largest absolute error of the decoded output | 1.8661 |
| largest absolute value of the float output | 6.8411 |
| inputs with no clamped value, and their largest absolute error | 235 of 256, 0.0457 |
| half a step of y, Delta_y / 2 | 0.0186 |
| SQNR of the decoded output against the float output, dB | 20.65 |
| values clamped when quantized (inputs, a1, y) | 0.75% |
<!-- /worked -->

The compiled form equals the direct reference on all 256 inputs, which verifies the folding of the bias and the rounded shift. None of the 768 output codes differs from an ideal requantizer that multiplies by the real $M$; with other numbers a code can differ by one unit, and the requirement is only that this be rare and never larger. The accumulators stay below 34,000 in magnitude, more than 60,000 times below the 32-bit limit.

The SQNR of 20.65 dB is the answer to "how close to float", and it is the measure of Chapter 12 applied to the whole network's output. Look at where the error comes from. The 235 inputs for which no value was clamped have a largest error of 0.0457, about 1.2 steps of $y$ (a step is $\Delta_y = 0.0371$; half a step, the rounding bound of one output, is 0.0186; the hidden activations' own rounding adds to it). The other 21 inputs, in which at least one of the 0.75% of quantized values was clamped, produce the largest error of all, 1.87. A test input larger than anything in the calibration set sent a hidden activation beyond the calibrated range of $a$, the code 127 could not represent it, and the error passed through the second layer. This is the error that a calibration set causes, and it is the subject of the next section.

### How much calibration is enough, and of what

A calibration range is a prediction about the future: the device will see values in this interval. It fails in two ways, and the two are opposite. A range that is too narrow clamps real values, and the error of a clamped value is as large as the amount by which it overshoots. A range that is too wide spends codes on values that never occur, so every step is larger than it needs to be and every rounding error grows with it.

Two tests show the effect. First, the size of the calibration set: the first $n$ inputs of one stream, for $n$ from 1 to 128. Second, whether it is representative: a stream whose inputs are a quarter of the size of the deployed ones (a camera at night used to calibrate a daylight deployment).

<!-- worked: npu.calibration_size -->
Output SQNR (dB, against the float network) on the 256 test inputs of the previous block, as a function of the size of the calibration set. 'representative' sets are the first n inputs of the calibration stream (128 inputs, seed 12490842275477104722 (seed_of 'npu.calibration')); 'narrow' sets are the first n inputs of a stream of x_i = round(0.25 normal(), 3), seed 11496424872099761511 (seed_of 'npu.calibration_narrow') (an input a quarter of the size of the deployed ones). 'clamped' is the share of the quantized values (inputs, hidden activations, outputs) that fell outside the calibrated range and were clamped.

| calibration inputs | SQNR representative | clamped | SQNR narrow | clamped |
|---|---:|---:|---:|---:|
| 1 | 5.49 | 22.7% | 1.33 | 39.8% |
| 2 | 6.17 | 14.6% | 1.54 | 38.5% |
| 4 | 7.70 | 5.2% | 1.73 | 34.9% |
| 8 | 17.17 | 1.4% | 2.61 | 33.3% |
| 16 | 17.17 | 1.4% | 3.01 | 28.8% |
| 32 | 20.65 | 0.8% | 4.42 | 25.4% |
| 64 | 22.37 | 0.6% | 4.56 | 25.0% |
| 128 | 26.53 | 0.2% | 4.56 | 25.0% |
<!-- /worked -->

![SQNR of the quantized network's output in decibels against log2 of the number of calibration inputs, for 1 to 128 inputs. The curve for inputs of the deployed size rises from 5.5 dB with one input to 26.5 dB with 128, with a plateau between 8 and 16 inputs. The curve for inputs a quarter of that size stays between 1.3 and 4.6 dB.](img/calibration-set-size.svg)

Read the first column. The SQNR rises from 5.49 dB with one calibration input to 26.53 dB with 128, and the share of clamped values falls from 22.7% to 0.2%. It does not rise smoothly. Eight and 16 inputs give the same 17.17 dB, because a min and max range changes only when a new extreme value enters the set; between 8 and 16 inputs no tensor had a new extreme, so the ranges, and therefore the integer network, were identical. A calibration set is judged by whether it contains the extremes that the deployment will produce, and the count of inputs is only a proxy for that.

The second pair of columns is the failure to be representative. The calibration inputs are a quarter of the deployed size, so every range is about a quarter of what it should be and a quarter of all test values (25.0%, at the end) are clamped. Adding inputs from the same narrow stream does not help: from 64 to 128 the SQNR stays at 4.56 dB, since the extreme values of a narrow stream are narrow. No amount of data repairs a calibration set that does not resemble the deployment.

### Clipping the range

If a few extreme values set the range, they cost every ordinary value resolution (the argument of Chapter 12 about outliers, now on activations). A calibration that ignores the extreme tails sets a narrower range, with finer steps for the bulk, and clamps the rare large value. Whether that pays depends on whom the error is measured for. The test below uses inputs in which 2 in 100 are six times larger than the rest (a bright flash in a dark scene), and it measures the error per input, as a fraction of the size of that input's output, so that the typical input and the rare large input are seen separately.

<!-- worked: npu.clipping -->
A calibration range that ignores the tails of the recorded values. Inputs are 'spiky': for each, six normal() draws then one uniform(), and when the uniform is below 0.02 all six values are multiplied by 6 (rounded to 3 decimals). The calibration set is 512 such inputs, seed 11795650988708937134 (seed_of 'npu.spiky_calibration'); the test set 1000, seed 6408658945436704731 (seed_of 'npu.spiky_test'). Each activation range is [percentile(v, tail), percentile(v, 1 - tail)] of the values recorded for the tensor (the percentile of the book's pseudocode page; tail 0 is the minimum and maximum). 'relative error' of an input is the norm of (decoded - float) over the norm of the float output; 'median' and '99th percentile' are over the 1000 test inputs; SQNR is over all their outputs.

| tail clipped per end | values clamped | median relative error | 99th percentile of it | SQNR dB |
|---|---:|---:|---:|---:|
| 0.00% | 0.02% | 0.0601 | 0.4695 | 27.24 |
| 0.05% | 0.12% | 0.0415 | 0.3805 | 15.71 |
| 0.10% | 0.23% | 0.0335 | 0.3945 | 11.86 |
| 0.20% | 0.44% | 0.0269 | 0.4256 | 10.73 |
| 0.50% | 0.86% | 0.0187 | 0.5930 | 9.32 |
| 1.00% | 1.45% | 0.0158 | 0.6609 | 8.62 |
| 2.00% | 2.65% | 0.0164 | 0.7275 | 7.85 |
| 5.00% | 7.15% | 0.0371 | 0.9727 | 6.06 |
<!-- /worked -->

![Relative output error against the tail clipped from each end of every calibration range, in percent from 0 to 5, each curve divided by its value for the min and max range. The median input falls from 1 to 0.26 at 1% clipped and then rises to 0.62 at 5%. The 99th percentile input falls slightly to 0.81 at 0.05%, then rises steadily to 2.07 at 5%.](img/clipping-percentile.svg)

Clipping 1% from each end cuts the error of the median input from 0.0601 to 0.0158, 3.8 times, and raises that of the 99th percentile input from 0.4695 to 0.6609 (the figure shows both divided by their values at no clipping). By the whole-test SQNR, which is dominated by the rare large inputs because they carry the most output energy, min and max is best: 27.24 dB against 8.62 dB at 1%. No rule in the table is right for all uses. A range is a choice of whom to serve: a detector that must find the large, rare event wants the full range, and a system whose typical frames carry the value wants the clipped one. The tail is a parameter to be chosen on held-out data (Volume II), as any other hyperparameter, and the calibration set that determines it must be held apart from the test set that judges it.

### The requantizer the device has

The device of the case study, the NPU of the Rockchip RV1103 that `crates/rknpu` compiles for, has a narrower requantizer than the per-channel multiplier and shift above. In the source it is one multiplier and one shift for the whole layer, with a 14-bit fraction per output channel in front of it: the channel's accumulator is multiplied by $c_o = \mathrm{round\_even}(\Delta_o/\max\Delta \cdot 2^{14})$, rounded to an integer by a 14-bit shift, and then the layer's multiplier and shift (for $M = \Delta_{\mathrm{in}}\max\Delta/\Delta_{\mathrm{out}}$) are applied. The same network, calibration set and test inputs, quantized this way:

<!-- worked: npu.two_stage -->
The requantizer with one multiplier and shift per layer. Channel o first scales its accumulator by a 14-bit fraction c_o = round_even(Delta_o / max(Delta) 2^14) and rounds (ties to even) to an integer, then the layer multiplies by the multiplier of M = Delta_in max(Delta) / Delta_out and rounds again. Same network, calibration set and 256 test inputs as before.

| quantity | value |
|---|---|
| layer 1 Q14 factors (16384 = 1.0) | [16384, 11244, 9959, 9798] |
| layer 1 multiplier, shift | 22320, 21 |
| layer 2 Q14 factors | [11959, 16025, 16384] |
| layer 2 multiplier, shift | 18354, 22 |
| output codes that differ from the per-channel form | 1 of 768 |
| SQNR per-channel multiplier and shift, dB | 20.65 |
| SQNR two-stage, dB | 20.65 |
<!-- /worked -->

One code of 768 differs from the per-channel form, and the SQNR is the same to two decimals. Two roundings in a row can differ by one unit from a single rounding, and here they did once. What the narrower form costs is range, not these units: every channel shares one shift, so a channel whose weight scale is below $2^{-15}$ of the largest gets $c_o = 0$ and outputs a constant. The lesson recurs through the compiler section: the quantization scheme cannot be the best one on paper; it is the best one that the registers of the device can hold.

## The graph compiler

With the numbers settled, the quantized graph still has to become a list of commands, and each pass of the program that does it takes a decision. A **graph compiler** is a program that takes a whole graph and produces the program a device runs. It differs from the kernel compilers of Chapter 9 in what it sees. A kernel compiler sees one function; the graph compiler sees every op, every tensor, and which op reads which tensor. It is ahead-of-time compilation (Chapter 9) for the whole network at once, and its passes are the decisions that no single kernel can take: which ops to merge (fusion), how each tensor is laid out (layout), how a layer that does not fit is cut (tiling), where each tensor lives (memory planning), and what words to write (emission). The graph-level passes of the compiler of Chen et al. 2018 include operator fusion, data-layout transformation and static memory planning, the same passes; what that tool searches over, the schedule of each operator, is largely fixed by the pipeline of a fixed-function device.

### Fusion

Chapter 8 counted what fusing costs and saves in bytes. For an integer layer the intermediate tensors are 32-bit, which makes them four times as expensive as the 8-bit tensors on either side. Take a real layer shape (the one of a detector at 320 x 320 input that the board check of `crates/rknpu` runs): input $40 \times 40 \times 128$, a $1 \times 1$ convolution, 64 output channels. Run as four ops (accumulate, add the bias, apply the rectifier, requantize) with every result in memory, then as the two stages the integer layer above suggests, then fused entirely:

<!-- worked: npu.fusion_bytes -->
One layer of the 320 x 320 detector of the board check: input 40 x 40 x 128 int8, kernel 1 x 1, 64 output channels, stride 1. It has 102,400 outputs, 8,192 int8 weights, 64 int32 biases, and does 13,107,200 multiply-adds = 26,214,400 operations in every design. An int32 accumulator is 4 bytes, an int8 activation 1. Each op reads its distinct operands once and writes its result once (the accounting of Chapter 8; the requantization constants are not counted).

| design | ops | bytes read | bytes written | total bytes | total fewer by |
|---|---:|---:|---:|---:|---:|
| conv, bias, relu, requantize as four ops | 4 | 1,442,048 | 1,331,200 | 2,773,248 | 1.000 |
| conv + bias + relu, then requantize | 2 | 622,848 | 512,000 | 1,134,848 | 2.444 |
| one op: accumulator stays on chip | 1 | 213,248 | 102,400 | 315,648 | 8.786 |
<!-- /worked -->

The fully fused op moves 8.8 times fewer bytes than the four-op chain, and does exactly the same 26,214,400 operations. The 32-bit accumulator of each output is never stored: it is produced, clamped at zero, scaled and rounded, and only the 8-bit code leaves. The rectifier fuses for the reason of the last section, the bias for the reason of the folding, and the requantization because it is the same arithmetic on the same value. The constants (the bias, the multiplier and the shift) are numbers the compiler puts into the command words.

### Layout

The device reads feature maps in one fixed layout, set by its arrays. In the case study a tensor is stored in **16-lane planes**: channels in groups of 16, each group a plane that holds the image row by row with the 16 channel bytes of a pixel together (a 16-byte atom), and each plane starting at a multiple of 64 bytes. A feature map of $c$ channels occupies $\lceil c/16\rceil$ planes. Three consequences are visible in the bytes:

<!-- worked: npu.layout_bytes -->
Bytes of an int8 feature map stored densely (h w c) and in 16-lane planes: ceil(c / 16) planes, each h w 16 bytes rounded up to a multiple of 64. The first two rows hold the same pixels; the difference between their 16-lane sizes is 1,228,800 bytes.

| tensor | h x w x c | dense bytes | 16-lane bytes | ratio |
|---|---:|---:|---:|---:|
| detector input | 320 x 320 x 3 | 307,200 | 1,638,400 | 5.333 |
| the same image after 2 x 2 space-to-depth | 160 x 160 x 12 | 307,200 | 409,600 | 1.333 |
| a feature map | 160 x 160 x 32 | 819,200 | 819,200 | 1.000 |
| a head output | 40 x 40 x 255 | 408,000 | 409,600 | 1.004 |
| a deep feature map | 20 x 20 x 256 | 102,400 | 102,400 | 1.000 |
<!-- /worked -->

A tensor whose channel count is a multiple of 16 has no overhead. A three-channel input does: each pixel fills one 16-byte atom for 3 useful bytes, and the image takes 5.33 times its dense size. After a $2 \times 2$ space-to-depth (four pixels become the 12 channels of one pixel), the same pixels fit 1.33 times their dense size, and 1,228,800 bytes are saved. The compiler of the case study has this as an open item and not an implemented pass; the point of the table is what a layout choice is worth. The second consequence is a free operation. Concatenating tensors along the channel axis, a data copy on a processor, is a placement here: if every input but the last has a multiple of 16 channels, the inputs are written directly into adjacent planes of the output, and the concatenation vanishes from the command stream. The compiler plans for that when it places the tensors (below).

### Tiling to the on-chip buffer

The device reads a convolution's input from a buffer on the chip. In the source of the case study that input buffer holds at most 6144 atoms of 16 bytes, 98,304 bytes (a capacity the source states, measured by the project it was ported from, not derived here). When a layer's input exceeds it, the layer is cut along the height into bands of output rows, and each band is a separate task that reads just the input rows it needs. This is tiling as in Chapter 6, with the buffer capacity as the constraint.

Write $C_{\mathrm{buf}} = 6144$ for the capacity. A layer reads $h \cdot w \cdot \lceil c/16 \rceil$ atoms; it fits when that is at most $C_{\mathrm{buf}}$. Otherwise a band of input rows may hold at most $\lfloor C_{\mathrm{buf}} / (w \lceil c/16 \rceil)\rfloor$ rows, and a band of output rows $[o_0, o_1)$ needs the input rows $o_0 s - p_t$ to $(o_1 - 1)s - p_t + k - 1$ for stride $s$, kernel $k$ and top padding $p_t$, clipped to the image. The compiler takes, from each starting row, the largest band whose input rows fit, with one further constraint of the hardware: every band must start on a 64-byte boundary (4 atoms) in the input and in the output plane.

```text
function bands(ih: int, iw: int, c: int, k: int, stride: int, pad: array<int>[4], capacity: int)
        -> list<(int, int, int, int)>:
    # returns (out_lo, out_hi, in_lo, in_hi) per band; pad = (top, left, bottom, right)
    tiles = ceil(c / 16)
    out_h = (ih + pad[0] + pad[2] - k) // stride + 1
    out_w = (iw + pad[1] + pad[3] - k) // stride + 1
    if ih * iw * tiles <= capacity:
        return [(0, out_h, 0, ih)]
    max_rows = capacity // (iw * tiles)
    require max_rows >= k, "one window of kernel rows does not fit the buffer"
    result = []
    o0 = 0
    while o0 < out_h:
        found = false
        for o1 in out_h down to o0 + 1:
            first = o0 * stride - pad[0]
            last = (o1 - 1) * stride - pad[0] + k - 1
            in_lo = max(first, 0)
            in_hi = min(last + 1, ih)
            if in_hi - in_lo > max_rows or (in_lo * iw) mod 4 != 0 or (o0 * out_w) mod 4 != 0:
                continue
            if o1 < out_h:
                next_lo = max(o1 * stride - pad[0], 0)
                if (next_lo * iw) mod 4 != 0 or (o1 * out_w) mod 4 != 0:
                    continue
            append (o0, o1, in_lo, in_hi) to result
            o0 = o1
            found = true
            break
        require found, "no aligned band starts at this row"
    return result
```

Applied to eight layer shapes (the first is a small smoke test; the others are shapes of a 320 x 320 detector):

<!-- worked: npu.tiling -->
Height tiling of eight convolutions against an input buffer of 6144 atoms of 16 bytes (the capacity stated by the source of `crates/rknpu`, 98,304 bytes). A layer is written h x w x c -> oc, kernel k, stride s; padding is 1 for 3 x 3 kernels except the stem, whose padding is [0, 0, 1, 1] (top, left, bottom, right). 'atoms' is h w ceil(c / 16); 'rows read' is the input rows summed over the bands divided by h (1.000 = no row is read twice); 'weight bytes' is oc ceil(c / 16) 16 k^2.

| layer | input atoms | fits | bands | rows read | weight bytes |
|---|---:|---:|---:|---:|---:|
| 8 x 8 x 16 -> 16, k3 s1 | 64 | yes | 1 | 1.000 | 2,304 |
| 320 x 320 x 3 -> 12, k3 s2 | 102,400 | no | 18 | 1.053 | 1,728 |
| 160 x 160 x 12 -> 32, k3 s1 | 25,600 | no | 5 | 1.050 | 4,608 |
| 160 x 160 x 32 -> 64, k3 s2 | 51,200 | no | 9 | 1.050 | 18,432 |
| 40 x 40 x 128 -> 64, k1 s1 | 12,800 | no | 3 | 1.000 | 8,192 |
| 20 x 20 x 256 -> 256, k3 s1 | 6,400 | no | 2 | 1.100 | 589,824 |
| 10 x 10 x 1024 -> 512, k1 s1 | 6,400 | no | 2 | 1.000 | 524,288 |
| 40 x 40 x 128 -> 255, k1 s1 | 12,800 | no | 3 | 1.000 | 32,640 |
<!-- /worked -->

The 8 x 8 x 16 layer fits and is one task. The stem, whose input has 102,400 atoms, is cut into 18 bands (its input rows are read 1.053 times: neighbouring bands share rows, since a $3 \times 3$ kernel of stride 2 needs one row beyond the band's last output row). The source's own count for that layer, 18 tasks, is the number this model gives. The alignment rule can force a smaller band than the capacity allows, and the layer $10 \times 10 \times 1024 \to 512$ shows it:

<!-- worked: npu.tiling_detail -->
The layer 10 x 10 x 1024 -> 512 with kernel 1 has 10 x 10 x 64 = 6400 atoms, more than 6144. A row of the input is 10 x 64 = 640 atoms, so at most 9 rows fit. Nine output rows would need the band starting at row 9 to begin at atom 9 x 10 = 90, which is not a multiple of 4, so the first band is cut at 8 rows.

| band | output rows | input rows | start atom of input band mod 4 | start atom of output band mod 4 |
|---|---:|---:|---:|---:|
| 0 | 0..7 | 0..7 | 0 | 0 |
| 1 | 8..9 | 8..9 | 0 | 0 |
<!-- /worked -->

Nine rows would fit the buffer, but the second band would then start at atom 90, which is not a multiple of 4. The compiler takes the largest aligned band instead, 8 rows, and a second band of 2. A tiling is correct when its bands cover every output row exactly once and each band reads only rows inside its limits; it is *good* when it reads each input row about once, and the "rows read" column is that measure, 1.000 (no row twice) to 1.100.

The model counts only the input buffer, which is the one capacity the source of the case study enforces. The weights and the bias block are fetched from memory by the same task, and the "weight bytes" column counts them as in the source ($oc \cdot \lceil c/16 \rceil\cdot 16\cdot k^2$ int8 weights). A model of a device with other limits (a weight buffer, an output buffer) adds those constraints to the same loop.

### Static memory planning

Every tensor of the graph has a size in bytes, and the graph says when each is written and last read. The **tensor lifetime** of tensor $t$ is the interval of operations $[f_t, l_t]$ from the op that writes it (or the start, for the input) to the last op that reads it (or the end, for an output). Two tensors can occupy the same bytes exactly when their lifetimes do not intersect; a tensor and the output of an op that reads it intersect at that op, so they never share. A **static memory plan** assigns every tensor an offset in one region so that tensors with intersecting lifetimes do not overlap. It is computed once, and the command words contain the addresses it chose.

A plan has a floor: at each op the tensors alive there must all fit, so the region cannot be smaller than $\max_i \sum_{t:\, f_t \le i \le l_t} B_t$ for tensor sizes $B_t$. Finding the best plan is a packing problem; the compiler of the case study uses a greedy rule in the spirit of the bit search of Chapter 12, applied to rectangles: place tensors largest first, each at the lowest offset where it clashes with no already placed tensor that is alive at the same time.

```text
function plan_memory(size: array<int>[T], first: array<int>[T], last: array<int>[T]) -> array<int>[T]:
    order = argsort(size, descending)                      # ties keep the original order
    offset = zeros(int, [T])
    placed = []                                            # tensors already given an offset
    for t in order:
        busy = [(offset[u], offset[u] + size[u]) for u in placed
                if first[u] <= last[t] and first[t] <= last[u]]
        cands = [0]                                        # a tensor starts at 0 or where a busy range ends
        for (lo, hi) in busy:
            append hi to cands
        at = infinity
        for c in cands:
            if all([c + size[t] <= lo or hi <= c for (lo, hi) in busy]):
                at = min(at, c)
        offset[t] = at
        append t to placed
    return offset
```

The `busy` list holds the byte ranges of the tensors whose lifetimes intersect $t$'s; the lowest offset that clears all of them is either 0 or the end of one of them, so the loop tries those and keeps the smallest that fits. Take a five-op graph on 16-lane tensors (a convolution, a stride-2 convolution, a $1 \times 1$ convolution, an add that joins a skip connection, and a last $1 \times 1$ convolution):

<!-- worked: npu.memory_plan -->
A graph of five ops on 16-lane tensors (sizes by the 16-lane rule of the layout block, in bytes). Lifetimes are inclusive op intervals: an op touches the tensors it reads and the one it writes, the input is touched by op 0 and the output after the last op (op 5).

| op | computation | reads | writes |
|---|---|---|---|
| 0 | conv 3 x 3, 16 -> 32 | t0 | t1 |
| 1 | conv 3 x 3, 32 -> 32, stride 2 | t1 | t2 |
| 2 | conv 1 x 1, 32 -> 32 | t2 | t3 |
| 3 | add | t2,t3 | t4 |
| 4 | conv 1 x 1, 32 -> 16 | t4 | t5 |

| tensor | h x w x c | bytes | lifetime (ops) | offset | end |
|---|---:|---:|---:|---:|---:|
| t0 | 16x16x16 | 4,096 | 0..0 | 8,192 | 12,288 |
| t1 | 16x16x32 | 8,192 | 0..1 | 0 | 8,192 |
| t2 | 8x8x32 | 2,048 | 1..3 | 8,192 | 10,240 |
| t3 | 8x8x32 | 2,048 | 2..3 | 0 | 2,048 |
| t4 | 8x8x32 | 2,048 | 3..4 | 2,048 | 4,096 |
| t5 | 8x8x16 | 1,024 | 4..5 | 0 | 1,024 |

| op | tensors alive | bytes alive |
|---|---|---:|
| 0 | t0, t1 | 12,288 |
| 1 | t1, t2 | 10,240 |
| 2 | t2, t3 | 4,096 |
| 3 | t2, t3, t4 | 6,144 |
| 4 | t4, t5 | 3,072 |
| 5 | t5 | 1,024 |

Without reuse the plan needs 19,456 bytes (every tensor its own bytes). The largest total alive at one op is 12,288 bytes, a lower bound for any plan. First fit, largest first, ends at byte 12,288.
<!-- /worked -->

![A timeline with the op index on the horizontal axis (op 0 to op 4 and "after") and the byte offset on the vertical axis (0 to 12,288). Six rectangles are the tensors t0 to t5, each as wide as its lifetime and as tall as its size in bytes: t1 of 8,192 bytes at the bottom for ops 0 and 1, t0 of 4,096 bytes above it for op 0 only, t2 of 2,048 bytes above t1 for ops 1 to 3, t3 and t4 of 2,048 bytes at low offsets for ops 2 to 4, and t5 of 1,024 bytes at the bottom for op 4 and after. A dashed line at 12,288 bytes marks the largest total alive at any op.](img/buffer-reuse-timeline.svg)

Without reuse the six tensors need 19,456 bytes. The plan ends at byte 12,288, which is exactly the floor (the 12,288 bytes alive at op 0): the plan is optimal on this graph, and 1.58 times smaller than one buffer per tensor. The picture shows what sharing means. The 4,096 bytes of $t_0$ (the input, needed only by op 0) are reused for $t_2$ at op 1; the 8,192 bytes of $t_1$ are reused for $t_3$ and $t_4$; $t_5$ sits in the bytes that $t_3$ vacated. No two rectangles overlap, and rectangles that share bytes never share an op.

A greedy rule is not optimal in general (Exercise 6 gives a graph where it exceeds the floor), and the floor is itself only a lower bound. What a static plan does guarantee is the other property that matters for a device without an allocator: its size is a number the compiler prints. Either the plan fits the memory the board has or it does not, and the compiler knows which before deployment. One consequence for the runtime is easy to miss. The input tensor is dead after its last read, so its bytes are reused; the input of the *next* run must therefore be written before that run starts, and cannot stay where it was. `Model::run_with` in `crates/rknpu` takes the input as an argument for exactly this reason.

### Emitting the command stream

The last pass writes the program. In the case study a **command stream** is a sequence of tasks. A task is a block of register writes, each a 64-bit word that holds the target engine in bits 63 to 48, a 32-bit value in bits 47 to 16 and the register address in bits 15 to 0, followed by words that link it to the next task. A task writes the same set of 126 registers (the source's count), so that no state leaks from one task to the next; the compiler's job is to compute the value of each one. Take the first layer of the tiny network as a $1 \times 1$ convolution of $1 \times 1 \times 6$ input, with the requantizer of the previous section:

<!-- worked: npu.command_stream -->
Layer 1 of the tiny network as a convolution of kernel 1 on a 1 x 1 image of 6 channels, 4 output channels, with the two-stage requantizer (multiplier 22320, shift 21). Seven of the registers one task writes, with the register addresses, the targets (CNA 0x0201, CORE 0x0801, DPU 0x1001) and the field formulas of `crates/rknpu/src/conv.rs`, encoded as target << 48 | value << 16 | register.

| register | address | value | command word | meaning |
|---|---|---|---|---|
| CNA_DATA_SIZE0 | 0x1020 | 0x00010001 | 0x0201000100011020 | input width 1 and height 1: width shifted left by 16, or the height |
| CNA_DATA_SIZE1 | 0x1024 | 0x00050010 | 0x0201000500101024 | channels - 1 shifted left by 16, or the lanes: 6 channels in 16 lanes |
| CNA_WEIGHT_SIZE0 | 0x1030 | 0x00000040 | 0x0201000000401030 | weight table bytes: 4 x 16 x 1 |
| CORE_DATAOUT_SIZE_1 | 0x3018 | 0x0000000f | 0x08010000000f3018 | output lanes - 1: 16 - 1 |
| DPU_OUT_CVT_OFFSET | 0x4080 | 0xffffff80 | 0x1001ffffff804080 | output zero point q0_a = -128, as a 32-bit two's complement value |
| DPU_OUT_CVT_SCALE | 0x4084 | 0x00005730 | 0x1001000057304084 | layer multiplier of the requantizer |
| DPU_OUT_CVT_SHIFT | 0x4088 | 0x00000015 | 0x1001000000154088 | layer shift of the requantizer |

A task writes the 126 registers of the profile (the source's count) and then 4 tail words that link to the next task: 130 words of 8 bytes = 1040 bytes, rounded up to a multiple of 64: 1088 bytes per task. A network of 2 such tasks is 2,176 bytes of commands; a network of 18 tasks (the stem above) 19,584; the 172 tasks the commit cited below counts, 187,136.
<!-- /worked -->

Every field is a number the earlier passes produced: the shapes from the export, the zero point and the multiplier and shift from the quantization, the lane counts from the layout. The seven words shown are a handful of the 126; a layer larger than the buffer gets one task per band, with that band's input and output addresses. The tasks are linked, so the whole chain, or any run of it, is one submission to the driver: the host starts it and waits, and no host code runs between layers.

That does not mean the device does everything. Operations the device cannot do (in the case study a max pool and a $2\times$ upsample) run on the processor between segments of tasks, and the compiler splits the chain into runs of device tasks with those operations in between, with explicit cache maintenance at each hand-over. An add of two tensors is rewritten, when its operands can sit side by side in memory, into a concatenation and a $1\times 1$ convolution with identity weights, which is exact in integers and keeps the add on the device.

One check closes the loop. The compiled program is correct when the device writes the same bytes as an independent model of the same arithmetic, not when its output looks plausible. The case study keeps a host function that computes, with the exact integers above, every layer's output; the board is checked byte for byte against it, with no tolerance. The tolerance appears once, between the integer network and the float one, and it is stated as the SQNR above.

## The chain in brain: exported graphs, a vendor compiler and a compiler of its own

**`crates/onnx`: the exported graph.** `crates/onnx/src/builder.rs` is a `GraphBuilder` that assembles an ONNX graph and serializes it: typed inputs and outputs with static shapes (`input_f32`, `output_f32`), initializers in `f32`, `i8`, `i64` and packed 4-bit form, and nodes with attributes. The default opset is 13, which its comment gives as the floor that provides per-axis quantize and dequantize operators (the `axis` attribute used for per-output-channel weights). The owned graph model in `crates/onnx/src/graph.rs` is decoupled from the generated protocol-buffer types; the doc comment of its `ValueInfo` says that brain always exports static shapes, and a tensor larger than a threshold goes to a side file ("external data") so that the model file stays under the format's 2 GB limit. Commit f47f50ee9 (26 June 2026) added the crate for export only; commit c8861e4bd (31 July 2026) added a reader (`crates/onnx/src/read.rs`, and the positional walk in `crates/onnx/src/walk.rs`) because a face model's reference release ships only ONNX, with the reasoning that a second private decoder of the wire format would be a second implementation that nothing compares against the first.

**`crates/npu`: one graph per model, a vendor compiler behind it.** The crate builds an ONNX graph for each model family with a shared emission base (`crates/npu/src/topo.rs`, `TopoBase`) and hands the bytes to a runtime that compiles them for the Intel NPU through OpenVINO. The crate's header is the chapter's pipeline in one sentence: export, brain-native 8-bit post-training quantization (calibrate, then a graph with quantize and dequantize pairs), then compile and run. The graph compiler on this path is the vendor's; brain's part is the graph and the numbers in it. The calibration is `crates/npu/src/calib.rs`: a `RangeCollector` records each quantized convolution's input range over images, `symmetric_act_scale` turns the range into $\max(|\mathrm{lo}|, |\mathrm{hi}|)/127$ with zero point 0, and weights are quantized per output channel (`quantize_weight_per_channel` in `crates/npu/src/fold.rs`). That is the symmetric variant of this chapter's scheme; the affine variant is in the next crate. `crates/npu/src/sim.rs` reproduces the exported graph's arithmetic in float on brain's own engine (a quantize and dequantize of each activation, and of each weight), so that the accuracy of the int8 graph can be measured with no NPU present. A model implements one trait, `NpuModel` in `crates/npu/src/lib.rs`: `build` its graph and give a `cache_key` for the compiled blob, and the generic runner compiles, caches and runs.

**`crates/rknpu`: a graph compiler of brain's own.** For the Rockchip RV1103 board of the case study, brain does not use a vendor SDK: the crate lowers an int8 network to the register command programs the NPU executes and submits them through the kernel driver. Its modules follow this chapter's passes. `crates/rknpu/src/graph.rs` is the float network the compiler receives (convolution with a fused-rectifier flag, add, concatenation, max pool, upsample), a float reference executor, and `Calibration::measure`, which runs the reference over the inputs and records each tensor's minimum and maximum. `crates/rknpu/src/quant.rs` is the quantization: `QParams::from_range` is the affine rule of this chapter (it widens the range to include 0, uses scale $(\mathrm{hi}-\mathrm{lo})/255$ and zero point $\mathrm{round}(-128 - \mathrm{lo}/\Delta)$), weights are per-output-channel with a zero point, the input zero point is folded into the bias, and the output conversion is the two-stage one of the section above (a Q14 factor per channel, then one multiplier of at most 32767 and one shift for the layer), documented in the module header as rounding half to even. `crates/rknpu/src/plan.rs` is the compiler proper: `quant_classes` forces the operations that only move values (concatenation, max pool, upsample) onto one shared set of parameters so that they need no requantization; `layout` computes the tensor lifetimes and places roots first fit, largest first, with a concatenation's inputs placed inside its output's planes; `Plan::simulate` runs the whole plan on a host byte array with the exact integers. `crates/rknpu/src/conv.rs` holds the tiling (`INPUT_ATOMS = 6144`, `bands`), `crates/rknpu/src/regs.rs` the 126-register profile and the 64-bit command word, and `crates/rknpu/src/job.rs` the linked chain of tasks. The tests are `crates/rknpu/tests/plan.rs` (a YOLO-shaped graph, quantized, laid out and simulated on the host must track the float graph) and, on the board, `crates/rknpu/examples/board_check.rs`, which compares every output byte of every layer shape of the detector and of the whole network against the host reference.

**A scale for the whole channel.** Commit 1f4c3b6e7 (11 August 2026) is about the NPU path's weight quantization. The linear layers of the exported language-model graphs used one scale per output channel, and on a decode graph the result measured a cosine of 0.994 against float where the same graph in float gave 1.000000 (the figures are those of the commit message and of the comment above `QUANT_GROUP` in `crates/npu/src/topo.rs`; this chapter did not re-run them). The cause is Chapter 12's: one outlier weight sets the step for every other weight of the channel, which can have 3072 or more inputs. The fix, `linear_quant` with `QUANT_GROUP = 32`, gives each group of 32 weights along the reduction axis its own scale, 32 being the block size of the GGUF formats of Chapter 12. The part that belongs to this chapter is how the group scale is expressed in the exported graph. ONNX's `DequantizeLinear` at opset 13 applies one scale per whole channel, and the comment records that block-wise quantization came with a later opset (21) with no established coverage on the NPU plugin the project uses; so the graph dequantizes explicitly: a `Cast` of the integer weight to float, a `Reshape` to `[groups, 32, out]`, a `Mul` by the scale reshaped to `[groups, 1, out]` (a broadcast over the group), and a `Reshape` back. The representation limited the quantization scheme, and the scheme had to be written in the vocabulary the opset has. The commit also made three model-specific copies of the emitter share this one, and knowledge #65 records the same defect and the same group size on the serving side. Chapter 12 tells the other half of that story.

**A compiler in a day, checked against its own arithmetic.** The commits that built `crates/rknpu` are all dated 27 October 2026. Commit 9d438e1a3 added the driver and the dense 8-bit convolution with its host reference, and a board check that compares every layer shape of the detector at 320 x 320 byte for byte; commit 8a80c8488 added the graph, the plan and the host simulation, and a board check that compiles a YOLO-shaped graph and compares its outputs with the plan's simulation; commit 75f25dae2 added tensor memory reuse across lifetimes (first fit, largest first, with a concatenation's inputs extending its output's interval) and, as the lesson recorded above, made the input a per-run argument because its bytes are now reused; commit 787279f8a made the weights stream from the memory-mapped checkpoint one layer at a time and split NPU memory into buffers of about 1 MiB, because (as the comment in `crates/rknpu/src/plan.rs` says) the board's contiguous memory is fragmented and several 1 MiB blocks can be available when one 7 MiB block is not; commit 50b86d4f2 lowered an add to a concatenation and an identity convolution so that it runs on the device, and put the remaining operations on the processor's vector unit. The pattern worth copying is the oracle. The board checks of four of these commits (9d438e1a3, 8a80c8488, 787279f8a and 50b86d4f2) are stated as byte-exact against a host model of the device's own integer arithmetic, with zero differing bytes as the criterion, so that "the compiler is wrong" and "the quantization is lossy" are never confused: the first is a bit-exact question and the second a tolerance. The roadmap in `.agents/roadmap/rknpu.md` lists what remained open when this chapter was written.

**Calibration is an artifact.** Commit 00d202281 (27 October 2026) connected a detector to the compiler and made a decision that follows from this chapter: the calibration is measured once, on a host, over representative images at the board's input size, and stored beside the checkpoint as `<model>.npu-calibration.json` (`crates/yolov5/src/npu.rs`, `calibration_path`); the board only loads it, because it has neither the memory nor the time to measure it (the module header says so). The file records one range per graph value and the number of images, and loading refuses it when the number of ranges differs from the graph's. Its test, `crates/yolov5/tests/it/npu_int8.rs`, asserts what matters at the end of a quantization chain: the integer network, simulated with the device's exact integers, must find the objects the float network finds on a real photograph, each confident float detection (above 0.5) having an integer detection of its class with an overlap above 0.8 and a confidence within 0.15. It skips with a named reason when the fixtures are absent.

## Check your understanding

Answers are at the end of the chapter.

1. **Find the error.** "A quantized layer needs the zero point of its input subtracted from every input code before the products, which is $K$ extra subtractions per output." Show the algebra that removes them, and say what the term that remains depends on.
2. **Predict.** A tensor is calibrated on 10 inputs whose largest value is 3.0, and in deployment a value of 4.5 occurs. With signed codes and a range $[0, 3.0]$, what code does 4.5 get, and what number does it decode to? Is the error bounded by half a step?
3. **Explain.** In the calibration-size table the SQNR at 8 and at 16 inputs is the same to the last digit. Give the reason, and say what a calibration set must contain for more inputs to help.
4. **Find the error.** "A requantization factor $M = 0.0064$ is stored as the 15-bit multiplier $\hat M = 32767$ with a shift of 0, because 32767 is the largest 15-bit number." What are the correct shift and multiplier, and what does the wrong choice compute?
5. **Predict.** Two tensors are alive over the op intervals $[0, 2]$ and $[2, 4]$. Can they share bytes? Change the second to $[3, 4]$ and answer again.
6. **Explain.** In the fusion table the fully fused op moves 2,457,600 fewer bytes than the four-op chain. Where do exactly those bytes come from?
7. **Find the error.** "The tiling of a layer is correct if the bands cover the output rows once; any band that fits the buffer is therefore allowed." Use the $10 \times 10 \times 1024$ example to show what else the compiler checks.

## Summary

A fixed-function accelerator runs a program fixed before the first input arrives, so a network must become data. **Graph export** produces a static graph of typed nodes with fixed shapes and constants stored in it, in the ONNX form: names, initializers, nodes in dependency order, and an opset that gives every operator its meaning; a check at the boundary confirms that every tensor has one producer and every shape is a number. **Post-training quantization** runs a calibration set through the float graph to record each tensor's range, turns ranges into scales and zero points (signed codes $-128 \ldots 127$ for activations, symmetric per-channel codes for weights), and rewrites each layer as an integer sum into a 32-bit accumulator, with the input zero point folded into the bias and the rectifier a clamp at zero; the output code is the accumulator times the factor $M = \Delta_{\mathrm{in}}\Delta_o/\Delta_{\mathrm{out}}$, held as a 15-bit multiplier and a shift with a relative error under $3.05 \times 10^{-5}$. On the two-layer network the compiled form equals the direct reference on all 256 inputs, bit for bit, and is 20.65 dB from the float output; the dominant error is a clamped value that the calibration set did not contain. More calibration inputs help only when they add extremes, a narrow calibration stream cannot be repaired by more of itself, and clipping the range buys the typical input at the price of the rare one. The **graph compiler** then makes the decisions no single kernel can: fusion (the 32-bit accumulator never reaches memory, 8.8 times fewer bytes for the layer of the example), layout (16-lane planes, concatenation as placement), tiling to a 6144-atom buffer under a 64-byte alignment rule (18 bands for the stem), a static memory plan from tensor lifetimes (19,456 bytes to 12,288 on the example, the floor) and the command stream, 1,088 bytes per task. brain implements the chain twice: `crates/onnx` and `crates/npu` export graphs and calibrate them for a vendor compiler, and `crates/rknpu` is its own compiler for one device, checked against exact integer arithmetic.

The chapter has taken one device from a graph to a program. The next chapter, the project that ends the volume, asks whether the same matrix multiply can be made to run correctly and fast on more than one kind of hardware, using the formats, kernels, backends and quantization of this volume.

## References

- Benoit Jacob, Skirmantas Kligys, Bo Chen, Menglong Zhu, Matthew Tang, Andrew Howard, Hartwig Adam and Dmitry Kalenichenko, "Quantization and training of neural networks for efficient integer-arithmetic-only inference", 2018.
- Markus Nagel, Marios Fournarakis, Rana Ali Amjad, Yelysei Bondarenko, Mart van Baalen and Tijmen Blankevoort, "A white paper on neural network quantization", 2021.
- Tianqi Chen, Thierry Moreau, Ziheng Jiang, Lianmin Zheng, Eddie Yan, Meghan Cowan, Haichen Shen, Leyuan Wang, Yuwei Hu, Luis Ceze, Carlos Guestrin and Arvind Krishnamurthy, "TVM: An automated end-to-end optimizing compiler for deep learning", 2018.

## Exercises

1. **[Calculate]** A tensor is calibrated on the range $[-1.2, 4.1]$ and stored in signed 8-bit codes. Give $\Delta$ and $q_0$ (ties to even), the code of $0.85$, the number that code decodes to and the error. What range of real numbers do the 256 codes represent?
2. **[Calculate]** Find the multiplier and shift for $M = 0.0123$ as a 15-bit multiplier, and the relative error of $\hat M 2^{-n_{\mathrm{sh}}}$ against $M$. Then do it for $M = 0.8$. What happens at $M \ge 32767$?
3. **[Calculate]** Each product $(q - q_0)\,q^w$ has magnitude at most $255 \times 127$. How many products can an int32 accumulator add before an overflow is possible in the worst case? How does the answer change if the inputs are raw codes of magnitude at most 128 and the weights have magnitude at most 255 (the form with a folded zero point and a weight zero point)?
4. **[Calculate]** For a layer with input $64 \times 64 \times 48$, kernel 3, stride 1, padding 1 on all sides, compute the atoms read, whether it fits 6144 atoms, the maximum rows per band and the bands that the `bands` routine returns (the first three, if there are many).
5. **[Calculate]** Count, for the four-op chain of the fusion table, the bytes written to memory if only the rectifier is fused into the bias add (a three-op chain) and compare with the table's rows.
6. **[Calculate]** Four tensors $t_0 \ldots t_3$ have sizes 4, 2, 3 and 2 (in units of 64 bytes) and lifetimes $[1,1]$, $[1,3]$, $[3,4]$ and $[2,4]$. Find the floor (the largest total alive at one op), run `plan_memory` by hand and give the size of the plan. Then find an arrangement that reaches the floor, and say which choice of the greedy rule prevented it.
7. **[Implement]** Implement `affine_params`, `quantize_weights`, `fixed_point`, `round_shift` and the integer layer, and reproduce `npu.network_weights`, `npu.float_forward`, `npu.calibration_ranges`, `npu.activation_params`, `npu.weight_codes`, `npu.fixed_point_table` and `npu.one_input`. Then implement the direct reference and reproduce `npu.exactness`: 256 of 256 inputs equal, with no tolerance.
8. **[Implement]** Implement `calibrate` with a `tail`, and reproduce `npu.calibration_size` and `npu.clipping`. Report the tail at which the median relative error is smallest and whether it moves with the seed of the calibration stream (derive three other seeds by the book's `derive_seed`).
9. **[Implement]** Implement the two-stage requantizer, and reproduce `npu.two_stage`. Then reduce the Q14 fraction to 8 bits and report how many of the 768 output codes change.
10. **[Implement]** Implement `bands` and `plan_memory` and reproduce `npu.tiling`, `npu.tiling_detail` and `npu.memory_plan`. Add the rule that a band is limited to 8 input rows and report the new band counts of the first four layers.
11. **[Investigate]** The weights of the chapter's network are symmetric per output channel. Replace them by one scale for each whole layer, and measure the SQNR of `npu.exactness` again. Then multiply one weight of $W_1[2]$ by 10 and compare per-channel and per-layer scales. Which result does the argument of Chapter 12 predict?
12. **[Investigate]** Read `Plan::new`, `layout` and `quant_classes` in `crates/rknpu/src/plan.rs`. State, for a concatenation, a max pool and an upsample, why each is given one shared set of quantization parameters across its inputs and output, and what would be needed if it were not.

## Key terms

- **graph export**: the one-time conversion of a model's forward computation into a static dataflow graph, with every operation a node of a known type, every tensor of a fixed shape and every constant stored in the graph, so that a compiler or device can run it without any of the model's own code. <!-- concept: graph-export -->
- **ONNX graph**: the interchange form of an exported graph: typed and shaped inputs and outputs, constant tensors (initializers), nodes in an order that defines every input before use, each naming an operator, its inputs, its output and its attributes, and an opset number that fixes the meaning of every operator. <!-- concept: onnx-graph -->
- **post-training quantization**: converting a trained float network to integer arithmetic without retraining, by choosing the scale and zero point of each weight tensor from the weights and of each activation tensor from ranges measured on data, and rewriting each layer as an integer accumulation followed by a requantization. <!-- concept: post-training-quantization -->
- **calibration set**: the sample of inputs that is run through the float graph to record the range of every tensor; it must be representative of the deployment, because a range that is too narrow clamps real values and one that is too wide wastes codes. <!-- concept: calibration-set -->
- **graph compiler**: a program that takes a whole graph and produces the program an accelerator runs, by passes that no single kernel can make: fusing operations, choosing layouts, tiling layers to the on-chip buffer, planning memory from tensor lifetimes and emitting the command words. <!-- concept: graph-compiler -->

## Answers

**Recall questions.**

1. $\Delta = 2/255 = 0.007843$; $-\mathrm{lo}/\Delta = 0.5/0.007843 = 63.75$, so $q_0 = 64$.
2. Half a step, $\Delta/2$, for a value inside the range; a value outside is clamped to the end code and its error is the amount by which it lies outside the range, which is not bounded by the step.
3. A product of two 8-bit integers needs up to 16 bits, and a sum of $K$ of them up to $16 + \log_2 K$ bits; a narrow accumulator overflows or, in a float format, stops growing (the stall of Chapter 1). Thirty-two bits hold $K$ into the tens of thousands.
4. Two MiB cross memory (one write, one read) unfused; none after fusion, as the intermediate stays in registers or on chip.
5. The kernels are compiled for known device architectures before shipping and stored with a key for their inputs; at run time the program looks up and loads the one that matches the device, and compiles only when none matches.
6. The nodes are the values (inputs, parameters, results of operations) and the edges say which values were used to compute which; the forward pass visits the nodes in an order in which every node's inputs have been computed before it (a topological order).

**Check your understanding**

1. The sum $\sum_i (q_i - q_0)\,q^w_i = \sum_i q_i q^w_i - q_0\sum_i q^w_i$. The second term depends only on the weights, so it is computed once at compile time and added to the integer bias; the device then multiplies raw codes. The folded bias $\tilde b_o = b^{\mathrm{q}}_o - q_0\sum_i q^w_{o,i}$ is a constant per output channel.
2. The range $[0, 3.0]$ has $\Delta = 3.0/255 = 0.011765$ and $q_0 = -128$. $4.5/\Delta = 382.5$ rounds (ties to even) to 382, plus $q_0$ gives 254, and the clamp gives 127, which decodes to $\Delta\,(127 + 128) = 3.0$. The error is $1.5$, which is not bounded by half a step: it is the amount by which the value lies outside the calibrated range.
3. A min and max range changes only when a new extreme enters the set; between 8 and 16 inputs none of the tensors saw a new minimum or maximum, so the ranges, the scales, the codes and the output errors are identical. More inputs help only if they add the extreme values that the deployment will produce; a set of any size that lacks them leaves the range too narrow.
4. $32767/0.0064 \approx 5.12 \times 10^6$ lies between $2^{22} = 4{,}194{,}304$ and $2^{23}$, so $n_{\mathrm{sh}} = 22$, and $\hat M = \mathrm{round\_even}(0.0064 \cdot 2^{22}) = \mathrm{round\_even}(26843.5456) = 26844$, which lies in $[16384, 32767]$. The wrong choice computes $32767\,A$ instead of $0.0064\,A$, a factor of about $5.1 \times 10^6$ too large, so every nonzero output saturates.
5. $[0,2]$ and $[2,4]$ intersect at op 2, so they cannot share bytes: an op that reads the first can write the second, and the first must still be intact when the second is written. With $[3,4]$ the lifetimes are disjoint and they can share.
6. The three int32 intermediates (the accumulator after the convolution, after the bias add and after the rectifier), each of 102,400 elements of 4 bytes = 409,600 bytes, are each written once and read once: $3 \times 2 \times 409{,}600 = 2{,}457{,}600$. What remains, $2{,}773{,}248 - 2{,}457{,}600 = 315{,}648$ bytes, is the floor of the layer: the input, the weights, the bias and the output code.
7. Each band must also start on a 64-byte boundary (a multiple of 4 atoms) in the input and in the output plane. A first band of 9 rows would fit the buffer (at most 9 rows of 640 atoms) but would leave the next band starting at atom $9 \times 10 = 90$, which is not a multiple of 4, so the compiler uses 8 rows; a tiling is checked for coverage, fit and alignment.
