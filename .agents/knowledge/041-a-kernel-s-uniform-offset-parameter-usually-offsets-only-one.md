<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 41. A kernel's uniform offset parameter usually offsets only ONE named side - check which before reusing it in the other direction

`qwen3::Qwen::decode_steps`'s per-token DeepStack add needed to read
`deepstack_bufs[level]` (a compact `[n_rows,d]` block) starting at
`local_row*d`, while writing into `res[l+1]` (already a single `[d]`-sized
row) at offset 0 - the opposite offset direction from `splice_add.wgsl`'s
OTHER caller (prefill's whole-range add, which reads the compact block from
its own index 0 and writes it into the big sequence buffer at `row0*d`). The
existing code used `step_sliced` to put `local_row*d` on the SOURCE as an
actual bind-group buffer offset - which produced a real, deterministic wgpu
validation failure ("Buffer offset N does not respect
`min_storage_buffer_offset_alignment`") on hardware enforcing the full 256B
limit, since `local_row*d*4` has no reason to be a multiple of 256.

The fix that looked obvious - switch to a plain `g.step` and pass
`local_row*d` as `splice_add.wgsl`'s own `base` uniform, since uniform
offsets have no alignment constraint - was *wrong* and initially replaced a
crash with a silent wrong-value bug that a numeric parity test caught
(`deepstack_step_matches_full_recompute` went from a wgpu panic to a
`maxabs=0.499` mismatch). Reading the kernel's own `dst[p.base+idx] +=
src[idx]` body shows why: `base` was written to parameterize the
DESTINATION offset only, matching prefill's need exactly - it has no way to
express a SOURCE offset. Decode needed the other direction. The real fix was
a new sibling kernel (`splice_add_offset_src.wgsl`, `dst[dst_base+idx] +=
src[src_base+idx]`) rather than reusing the existing one's single offset
knob backwards.

**The general shape**: before reusing an existing kernel's uniform offset
parameter for a new call site, read the kernel body (not just its
signature) to confirm which of its several buffer operands that parameter
actually addresses - a `base`/`offset` name gives no guarantee it's
symmetric across every operand, and assuming so produces a bug the
compiler, the alignment validator, and a coarse test can all miss (only a
value-level parity check caught this one).
