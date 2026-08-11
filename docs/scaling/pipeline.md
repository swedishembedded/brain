# Pipeline parallelism (multi-GPU model sharding)

Split a model's **layers** across several GPUs so a model whose weights
exceed a single card's memory fits across the pool. Each GPU holds and
computes only its own contiguous range of layers ("stage"); data flows
through the stages in order on the forward pass and back in reverse order on
the backward pass.

## Why this works cheaply

In brain's transformer models, the only thing that crosses a layer boundary
is the residual stream — everything else (attention scores, intermediate
activations, per-layer weights) stays local to the layer that produced it.
That means cutting the model into contiguous layer ranges has two nice
properties:

- each GPU only needs to allocate weights and activations for **its own
  layers** (plus the input embedding on the first stage, and the output head
  on the last stage);
- the only cross-GPU traffic is one residual tensor per stage boundary per
  pass — a small, fixed-size transfer, regardless of how big the model's
  hidden layers are.

Because the residual has the same size at every layer, the transfer cost is
identical no matter where you make the cut. So deciding where to place the
stage boundaries is purely a **load-balancing** problem — put roughly equal
work on each GPU — not a problem of minimizing what crosses each boundary.

## Automatic placement

Stage boundaries are chosen automatically to balance per-stage cost (parameter
count) across the available GPUs, exactly minimizing the maximum load on any
one stage. Because the first and last stages also carry the embedding and
output-head weights, they're given fewer transformer layers than the middle
stages, which balances memory across cards and lets a bigger model fit.

## Micro-batching (overlapping the stages)

Run naively, a pipeline has only one stage doing real work at a time — the
rest are idle waiting for data to arrive. Splitting a batch into
micro-batches and pipelining them (a GPipe-style schedule) fixes that: while
one stage is working on micro-batch *k*, the next stage works on
micro-batch *k-1*, so all stages overlap. This is bit-exact with running the
micro-batches sequentially and accumulating their gradients — pipelining
changes only the schedule, not the result.

## Which models support it

Pipeline sharding is a small per-architecture integration, not something
generic across every model (splitting a model's execution graph at
arbitrary points is inherently architecture-specific). Implemented and
tested today for:

- GPT
- Qwen3
- Qwen3.5-35B-A3B

Both bit-exact forward/backward parity against the equivalent single-GPU
model and the automatic load-balanced placement are validated for each of
these. Not yet supported: pipeline sharding for brain's other trainable
architectures (GLM-5.2, the Sparse MoE Transformer, and the rest) — extending
it to a new model is a small, well-scoped integration, not a redesign.

## Composes with data parallelism

Pipeline parallelism and data parallelism combine: shard a large model across
a group of GPUs, then replicate that whole group data-parallel across more
GPUs. See [Data parallelism](data-parallel.md).

## Current scope

Pipeline sharding is a real, tested mechanism today. What's not yet in place
is a single command-line flag that launches a multi-GPU pipeline-sharded run
directly — as with data parallelism, driving it today means writing to
brain's training entry points yourself.

## What to expect

**Correctness is bit-exact**: a sharded run reproduces the single-GPU
model's forward loss and gradients exactly.

**Speed**: sharding alone doesn't speed anything up — it only adds capacity,
since only one stage is doing useful work at a time in the naive schedule.
Micro-batching recovers overlap between stages, and the achievable speedup
from that grows with how many stages you have — in one measured configuration
(two stages, four micro-batches, a 0.6B model), overlapping the stages
brought step time down to about 0.8x of the naive sequential pipeline, and
the win grows with pipeline depth. Treat that as one data point, not a
guarantee — see [Hardware notes](../performance/hardware-notes.md) for why
numbers from one machine don't transfer to yours: actual speedup depends on
your interconnect and GPU count.
