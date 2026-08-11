# Tensor parallelism

The third parallelism dimension: instead of splitting *between* layers
(pipeline parallelism) or replicating the whole model (data parallelism),
tensor parallelism splits **one matrix multiply inside a single layer**
across GPUs. Where pipeline parallelism only needs to communicate at stage
boundaries, tensor parallelism needs a round of communication in the middle
of every layer's forward and backward pass — it's the most
communication-hungry of the three dimensions, so it's meant for the narrow
case where even one layer's weights are too large for one GPU, not as a
general speedup technique.

## What a column-parallel / row-parallel split means

Take a layer that computes `Y = X · W` for some weight matrix `W`. Split `W`
one of two ways:

- **Column-parallel**: give each GPU a slice of `W`'s output columns. Each
  GPU computes its own slice of `Y` directly from the full input `X` — no
  communication needed for this step, because each GPU's output columns
  don't depend on any other GPU's slice of `W`.
- **Row-parallel**: give each GPU a slice of `W`'s input rows (paired with a
  matching slice of the incoming activations). Each GPU computes a partial,
  incomplete result, and the GPUs' partial results have to be **summed**
  together to get the true `Y` — a communication step.

A transformer layer's attention and feed-forward blocks are built from pairs
of these: one column-parallel projection whose output feeds directly into a
row-parallel projection, so the communication step only has to happen once
per pair rather than after every individual matrix multiply. In the backward
pass, the same pattern repeats in reverse for the input gradient, while each
GPU's own weight-slice gradients stay local and need no communication at all.

## Current scope

The communication primitives (the combine operations described above) and
the specific column-parallel/row-parallel arithmetic for a transformer's
attention and feed-forward blocks are implemented and tested, including
end-to-end forward and backward numerical parity against the equivalent
single-GPU computation. What doesn't exist yet is a shipped model that wires
this primitive into its own full forward and backward pass — today it's a
proven, ready-to-use building block, not a capability any model exposes
end-to-end. Composing it into a model is a per-architecture integration, the
same shape of work as it took to add [pipeline sharding](pipeline.md) to a
model, just not done yet for tensor parallelism.

## Composing all three dimensions

Data, pipeline, and tensor parallelism are designed to be orthogonal and
combine: a group of GPUs can be tensor-parallel with each other for one
stage's layers, that stage can be one of several pipeline stages, and the
whole pipeline can be replicated data-parallel. Data-parallel and pipeline
parallelism have each been validated independently; running all three
together end-to-end isn't yet a shipped capability, since it depends on
tensor parallelism first being woven into a real model as described above.

## Choosing a degree, if you're extending a model

Because a tensor-parallel split needs a communication round on every layer
(rather than once per stage boundary, or once per training step), raising the
degree isn't free the way adding more pipeline stages or more data-parallel
replicas is: past a point, the communication cost per layer outweighs the
compute saved by splitting the layer further, and splitting a matrix multiply
too thin also lowers how efficiently each GPU's slice runs. The practical
rule of thumb, consistent with how large-model training frameworks generally
use tensor parallelism: use the smallest degree that makes a layer fit,
reach for data or pipeline parallelism for throughput and capacity instead,
and keep the GPUs sharing a tensor-parallel group on the fastest interconnect
you have, since they pay for it every layer.
