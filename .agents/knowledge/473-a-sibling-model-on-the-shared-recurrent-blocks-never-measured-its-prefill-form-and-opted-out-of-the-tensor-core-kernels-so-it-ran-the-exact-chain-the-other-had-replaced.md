<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 473. A sibling model on the shared recurrent blocks never measured its prefill form and opted out of the tensor-core kernels, so it ran the exact chain the other had replaced

**Lesson:** Sharing a block's kernels is not sharing its fast path. When a shared op takes an opt-in (a flag in the
caller's kernel ids, a measurement at placement, a round table), every model that calls it has to make the same
calls, and a test has to show the fast kernels actually ran: an answer that matches proves nothing about speed.

## What happened

The 35B-A3B (`qwen35moe`) and the dense 27B (`qwen35`) call the same Gated-DeltaNet code in `model`. A 16K prefill of
the MoE spent 1.1 s of 2.7 s in the GDN layers (`gdn_chunk_loop_f32`, `l2norm_scale`, `bmm_tiled`, the layout
permutes) where the dense model's serving rounds take ~0.12 us a token a layer on the segment kernels. Every MoE parity
gate passed: the slow chain is the reference.

## Why

Three opt-ins were the dense model's alone:

- `Gpu::prepare_gdn_prefill` at placement. Unmeasured, `gdn_bf16_prefill_selected` is false and every caller gets the
  exact TF32 form.
- `GdnFastIds::tensor_core: true` in the model's `gdn_ids`. The MoE had `false`, which turns off the tensor-core prefix,
  the token-major convolution and the segment kernels' `tokens_prefix_offered`.
- `model::gdn_segments::gdn_mixer_segments_fwd` with a `GdnSegmentRound` built once per round. The MoE called only the
  per-chunk `gdn_mixer_stream_fwd`.

## What to do

- The MoE now makes all three calls; its private prefill state is a pool of one row (`SegmentSpan { slot: 0 }`).
  16K prefill 6.26K -> 11.06K tok/s (gpu3, GH200, int8, bf16 KV).
- A spec test asserts the kernel that ran (kernel timing names `gdn_chunk_loop_seg*`) as well as the answer, and was
  checked to fail on a deliberately broken round (half the rows left to no kernel: 1.2e-3 against a 1e-4 bound).
- When a model is the second caller of a shared fast path, diff its kernel-id tables and placement calls against the
  first caller's before measuring anything else.
