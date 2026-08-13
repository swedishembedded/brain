# Data-parallel training across GPUs

A full replica of the model on each GPU, with each replica training on a
different slice of the step's batch concurrently, then a gradient combine so
every replica ends the step with the identical weights. This is the
*throughput* path — a training speedup — for a model that already fits on
one GPU. It composes with [pipeline parallelism](pipeline.md): shard a large
model across a group of GPUs, then replicate that whole group data-parallel.

## How it works

Each GPU holds one full copy of the model and runs its own forward and
backward pass on its own micro-batch, in parallel with the other replicas.
Once every replica has computed its gradients, they're combined into one
gradient (summed across replicas), a single optimizer step is applied, and
the updated weights are sent back out to every replica — so every copy stays
identical without a separate broadcast-then-update round trip. Reading every
replica's gradients once and updating once, instead of doing a full
all-reduce followed by an independent optimizer step per replica, is what
makes the combine step cheap enough for the speedup to show up in practice.

Optimizer state (the running moments AdamW keeps per parameter) can be kept
off the GPUs and in host RAM instead, controlled by the `BRAIN_OFFLOAD_ADAM`
environment variable — useful when a model's weights and gradients already
use most of a card's VRAM.

## Single machine or a network of machines

The gradient combine step moves data through the same transport-generic
interface used by every parallel dimension in brain: one implementation
handles GPUs that are threads inside a single process on one machine, and a
second implementation runs the identical operations over TCP sockets between
separate processes — which may be on different machines. Both are real and
tested. So data-parallel training already works whether your GPUs are all in
one box or spread across a network of machines; it's the same mechanism
either way, not a separate feature.

## Which models support it

Data-parallel training is generic — it's built once against the shared
per-model training interface every trainable architecture implements, so it
applies uniformly rather than being reimplemented per model. Today's
trainable models are:

- GPT
- Qwen3
- Qwen3.5-35B-A3B
- GLM-5.2
- Sparse MoE Transformer
- PID event/effect Transformer
- Seq2seq (encoder-decoder Transformer)
- LFM2.5-Encoder
- Bottleneck autoencoder
- YOLOv8-style detector
- Z-Image
- Nemotron (streaming ASR — transducer and acoustic-model heads)

Bit-exact gradient parity between a data-parallel run and the equivalent
single-GPU run has been validated on architecturally distinct models covering
both the language-model batch shape and a plain tensor-regression batch
shape, so the guarantee isn't specific to one architecture family.

## GPU selection

Device selection for a training process is per-process: `BRAIN_GPU_INDEX`
pins one process to one GPU card. See
[Configuration](../using/configuration.md) for the full device-selection
reference (`BRAIN_DEVICE`, `BRAIN_GPU_INDEX`, and related variables).

## Current scope

Data-parallel training is a real, tested mechanism in brain's training
engine today, validated for both single-machine and networked GPU groups.
What's not yet in place is a single command-line flag that launches a
multi-GPU data-parallel run directly — driving it today means writing to
brain's training entry points yourself rather than passing one flag to an
existing `brain <model> train` command.

## What to expect

**Correctness is bit-exact**: a data-parallel run reproduces the single-GPU
run's accumulated gradient, differing only by floating-point summation order
in the combine step.

**Speed** depends on your interconnect and GPU count. The gradient combine is
a fixed per-step cost, so the achievable speedup grows as you increase how
much compute happens between combines (i.e. more micro-batches per optimizer
step) and shrinks the tighter your interconnect is; the forward-and-backward
compute itself parallelizes close to linearly across cards, and the gap
between that and your end-to-end speedup is the fixed combine cost being
amortized over more compute. There is no fixed number to cite here - it
depends on your interconnect, GPU count, and micro-batch count. See
[Performance](../performance/overview.md) for how to profile and measure your
own configuration rather than trust a number measured on different
hardware.
