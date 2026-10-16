<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 231. A recomputed qwen35 layer reproduces the dropped activations bit for bit

The resident qwen35 trainer used to keep every layer's activations from the
forward to the backward (about 0.95 MB per token per Gated-DeltaNet layer), so
training memory grew with depth times sequence length. With `TrainOpts::recompute`
the forward keeps only the layer-boundary residuals `res[l]` and the backward
runs `layer_forward(l, res[l], scratch)` again to rebuild layer `l`'s
activations right before differentiating it, then drops them.

Nothing in the layer forward is order-dependent across calls (no atomics on the
saved paths, no state carried between layers), so the rebuilt activations equal
the dropped ones exactly: `tests/lora_recompute.rs` asserts equal loss and equal
gradients with `assert_eq!` on fp32 and bf16 bases, with a GDN sequence of
three chunks. No tolerance was needed or chosen.

Two details that are easy to get wrong:

- The rebuilt layer must write its output residual somewhere that is not
  `res[l+1]` once residuals can be stored narrower than fp32; one scratch
  buffer is reused for every layer.
- Buffers are allocated fresh per call and freed when the last handle drops, so
  a backend with deferred reclaim holds a dropped layer's buffers until it is
  polled. The backward drains (`poll_wait`) after each rebuilt layer so the
  next layer's allocation sees the memory back; without it the deferred bytes
  sum over the whole depth, which is the very growth recompute removes.
