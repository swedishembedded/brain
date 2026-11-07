<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 427. A CUDA binding limit sized for one-byte elements capped the splat gradient records at a quarter of what u32 word indexing reaches

**Lesson:** A limit named for one element size is not the limit of a buffer of another: size word
buffers against the word figure and byte buffers against the byte figure. A too-dense error on a
card with ample memory may be a limit that is not real.

## What happened

Question (the brief of 2026-11-02): `brain splat train` on the 251-photograph Tanks and Temples
truck died at step ~900 with "one frame has 69548237 (tile, gaussian) instances, 4.4 GiB of gradient
records, but one storage binding on this device holds 63161283 (4095 MiB)", on a 96 GB card.

## Why

- `max_storage_binding_bytes` is the figure that holds for an array of ANY element type, so the CUDA
  backend reports `u32::MAX` bytes: a `u32` element index over one-byte elements. The renderer sized
  its word buffers (gradient records `array<f32>`, sort keys and values, the pair log) against it.
- Every one of those is an array of 4-byte elements, indexed by `u32` ELEMENT in the kernels
  (`splat_grad_reduce`: `r = j * stride`), and `wgsl-cuda` computes the address in `size_t` and
  passes lengths as `u64`. A word buffer reaches 2^32 elements = 16 GiB: four times the byte figure,
  252M records of 17 words.

## What to do

- Fix: `Backend::max_word_binding_bytes` (default: the byte figure, so wgpu and Vulkan are
  unchanged; CUDA: 2^32 words bounded by the card) and the renderer's three sizing sites read it.
  The error stays loud past it.

### Rules

- A limit named for one element size is not the limit of a buffer of another: size a word buffer
  against the word figure, a byte buffer against the byte figure.
- A "too dense, thin it" error on a card with ten times the memory is a limit that is not real:
  check which index arithmetic the limit stands for before asking the user to shrink the scene.

## Evidence

- Test at scale (`crates/splat/tests/it/s29_large_frame.rs`): `splat_grad_reduce` over 69.5M records
  whose word index passes 2^30 (byte offset past 4 GiB) sums exactly, and a 1920x1088 ray frame of
  73M instances renders and differentiates with finite gradients. Red before the fix with the
  production message, green after.
