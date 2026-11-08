<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Volume IV: How Computers Actually Execute Machine Learning

Volume III built networks whose forward and backward passes are a few lines of
arithmetic on arrays. Written that way they are correct and, for anything
beyond a toy, far too slow: a modern model asks for billions of multiplications
for every word it reads or writes, and the lines of arithmetic do not say where
the numbers live, how many bytes they occupy or which of the machine's many
workers computes what. This volume is about the machine. It follows one
question from the bits of a single number to a whole program running on
several kinds of hardware: where does the time go, and what can we do about it?

## Why it comes here

Every later volume trains or runs models whose cost is set by the ideas of this
one. Language models are limited by the bytes they read per token, image
generators by the matrix products inside them, large training runs by memory.
The vocabulary of the volume (number formats, caches, bandwidth, the roofline,
kernels, fusion, quantization) is the vocabulary in which those limits are
stated, and a reader who has it can tell a computation that is slow because it
does too much arithmetic from one that is slow because it moves too much data.
The volume uses the arrays and matrix multiplication of Volume I, the
floating-point numbers of its Chapter 15 and the training loop of Volume III.

Speed is stated here as work, never as a stopwatch reading: the arithmetic a
computation needs (floating-point operations) and the data it must move
(bytes) are the same on every machine, and an improvement is fewer of either,
or a larger share of what the device can do. Times appear only as dated
observations of a named setup.

## What you can do after it

You can say how many bits a number needs and what each format gives up, lay
out data so a core and its caches use it well, decide from a computation's
arithmetic intensity whether arithmetic or memory limits it, write the
data-parallel programs (kernels) a GPU runs and the tiled matrix multiplication
at their heart, fuse steps to keep intermediate results out of memory, run one
set of kernels on several devices and check that they agree, pick the fastest
correct implementation on hardware nobody tuned for, measure performance
without fooling yourself, and store a model's weights in four bits. The project
that ends the volume builds a small tensor engine that does these things on
more than one kind of device.

## Chapters

1. [Bits and Number Formats](01-number-formats.md): how many bits a number
   needs, and what we lose with each bit we remove.
2. [The CPU: Cores, Caches and Vectors](02-cpu.md): why the same arithmetic runs
   ten times faster when the data is arranged differently.
3. [Memory Bandwidth, Compute and the Roofline](03-roofline.md): whether a computation is limited
   by arithmetic or by moving data, and how to tell.
4. [The GPU: Thousands of Small Workers](04-gpu.md): how a GPU organizes its thousands of
   threads, and what that demands of our algorithms.
5. [Writing Compute Kernels](05-compute-kernels.md): how to express one parallel computation so a GPU
   can run it.
6. [Making Matrix Multiplication Fast](06-matmul.md): how a matrix multiply goes
   from a three-line loop to running near the hardware's peak.
7. [Parallel Reductions, Scans and Sorting](07-reductions.md): how thousands of
   threads add up one array, and why the obvious way is slow.
8. [Kernel Fusion and Memory Traffic](08-fusion.md): why the cheapest computation is
   the one whose intermediate result never touches memory.
9. [One Program, Many Devices](09-backends.md): how one set of kernels can run on
   CPUs, Vulkan GPUs and CUDA GPUs with the same results.
10. [Choosing the Fastest Correct Kernel](10-kernel-selection.md): how an engine runs near
    the roofline on hardware nobody tuned it for.
11. [Measuring Performance Honestly](11-profiling.md): why most performance numbers
    lie, and how to produce ones that do not.
12. [Quantization: Fewer Bits per Weight](12-quantization.md): how a model's weights can be stored in
    four bits each and still work.
13. [Neural Processing Units and Graph Compilers](13-npu.md): how a whole network becomes one
    compiled program for a fixed-function accelerator.
14. Project: A Device-Independent Tensor Engine: make the same matrix multiply
    run correctly and fast on more than one kind of hardware.
