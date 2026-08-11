# Scaling brain across GPUs

How to train and run brain's models on more than one device. There are three
independent ways to spread work across GPUs — **data**, **pipeline**, and
**tensor** parallelism — plus a transport layer that works the same whether
the GPUs are in one machine or spread across a network. This page is the
umbrella; each dimension has its own page:

- [Data parallelism](data-parallel.md) — replicate the model, split the batch
- [Pipeline parallelism](pipeline.md) — split a model's layers across GPUs
- [Tensor parallelism](tensor-parallel.md) — split one layer's math across GPUs

## The three dimensions, in plain terms

**Data parallel.** Put a full copy of the model on each GPU. Each copy trains
on a different slice of the batch at the same time, then the copies combine
their gradients so every copy ends the step with the identical update. This
is a *speed* technique: it doesn't change how much model fits in memory, it
changes how fast you get through training data.

**Pipeline parallel.** Cut the model's layers into contiguous groups ("stages")
and put each stage on its own GPU. Data flows stage 0 → stage 1 → stage 2 …
during the forward pass and back during the backward pass. This is a
*capacity* technique: it lets you train or run a model whose weights don't
fit on a single card, because no single GPU ever holds more than its own
stage.

**Tensor parallel.** Instead of splitting *between* layers, split *inside* one
— e.g. give each GPU a slice of a layer's weight matrix, have it compute its
slice of the output, then combine the partial results. This is also a
*capacity* technique, for the case where even one layer is too large for one
GPU, but it needs a round of communication in the middle of every layer
instead of just at stage boundaries, so it's the most communication-hungry of
the three.

## Which to reach for

1. **Model fits on one GPU and you want training to go faster** → data
   parallel. It's the simplest of the three, and the only one that raises
   throughput on its own.
2. **Model's weights don't fit on one GPU** → pipeline parallel. Split the
   layers across your cards; the only thing that crosses a stage boundary is
   the residual stream, so it's cheap to communicate.
3. **A single layer itself is too wide for one GPU** → the tensor-parallel
   communication primitives exist and are tested (splitting a weight matrix
   column-wise or row-wise across GPUs and combining the partial results), but
   as of today no shipped model wires that primitive into a full forward pass.
   Treat tensor parallelism as a building block that's ready to be composed
   into a model, not yet as a one-step capability for an existing one.

## Single machine or a network of machines

All three dimensions move data through the same abstraction: whichever GPUs
are cooperating exchange sums, gathers, and broadcasts through one collective
interface. That abstraction already has two implementations — one for GPUs
that are threads inside a single process on one machine, and one that runs
the identical operations over TCP sockets between separate processes, which
may be on separate machines. Callers don't change: the same data-parallel (or
pipeline) code that runs across the GPUs in one box also runs across GPUs on
a network, because it never talks to the transport directly. So data-parallel
training across a single machine or across a network of machines is a working
capability today, not something planned for later.

## What "correct" and "fast" mean here

**Correctness is bit-exact.** Both data-parallel and pipeline-parallel runs
are validated against the equivalent single-GPU run: the parallel run has to
reproduce the serial run's loss and gradients (small floating-point summation
differences from combining values in a different order aside).

**Speed** depends entirely on your interconnect and GPU count — how fast your
GPUs can exchange gradients or activations relative to how fast they compute.
A slow link between GPUs caps the achievable speedup regardless of how many
cards you add; a fast one lets the speedup scale with card count. Any
concrete number is a property of one measured configuration, not a general
guarantee — actual speedup on your hardware depends on your interconnect and
GPU count.
