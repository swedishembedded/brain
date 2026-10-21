<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 344. The chunked MLP backward added its gradient into uncleared memory, and the 27B LoRA diverged on stale data

Symptom: the 27B (bf16 base, rank 16, alpha 32, one 15K-token record per micro-step, two per step) diverged on the real
CE8 records at every learning rate that moved: held-out loss 0.117 (step 10) then 0.92 / 2.7 / 4.9 / 6.5 at steps 20 / 25 /
30 / 38 at lr 5e-5, 0.16 to 12.9 by step 25 at 2e-4, and an NF4 sweep (runs/b27/lrsweep) where 1e-4 rose and 2e-4 blew up
while 5e-5 and 2e-5 tracked the lr 1e-9 control to three digits (no learning at all). The adapter the run kept had every
`lora_b` element at the same sign it started moving in (absmax 1.67e-3, the sum of the learning rates over the 38 steps,
rms 7e-4): Adam kept pushing along a gradient that did not see the loss rise. A Qwen3-0.6B through the same ChatFineTune path
on the same records improved steadily at 5e-5 and 2e-4, so the defect was in the qwen35 path, not in the recipe.

Cause: `mlp_bwd_saved` (the MLP run in row chunks on a recomputing build, `TrainOpts::mlp_rows`; on by default for rows of
2048 tokens or more) assembles `d_xn2` from the chunks with `put_rows`, which is `SPLICE_ADD` (`dst += src`), into a buffer
that f3c83d307 (2026-10-19) had changed from `storage` to `storage_uninit`, on the reasoning "each chunk is written whole".
An add into an uncleared buffer adds the gradient to whatever the allocator last freed there. The CPU backend hands out
zeroed memory, so every CPU gate (all at 12 tokens, never chunked) and the CUDA parity test at 12 tokens passed; the
commit's own evidence, "three 27B probe steps (identical loss bits)", cannot see a gradient. On CUDA the stale values are
real earlier activations: on the first four layers of the real 27B, at 10142 tokens, the global gradient norm of step 1 was
279760 (and 1.3e5 to 2.8e5 on every step) where the true norm is 0.98. The clip (1.0) then scaled the true gradient by
about 4e-6 and Adam stepped along the stale data: a persistent direction, unrelated to the loss.

Evidence that found it: a CUDA-versus-CPU comparison at the published model's own mixer geometry (16 key and 48 value
heads of 128, 256-wide attention heads, 64-token GDN chunks, 320 rows). Whole sequence, recomputed, segmented: cosine
1.000000 on every adapter gradient. With `mlp_rows` set: 0.9886 on the last layer's `o_proj.lora_a`, 0.9911 on
`v_proj.lora_b` (the layer whose MLP backward feeds them first). `put_rows_copy` instead of `put_rows` takes all four
settings, and the bf16 residual stream, to cosine 1.000000 (0.999 bound for the bf16 stream). The same 4-layer real-weight
step then reports gradient norm 0.83 to 1.0, with the identical step-1 loss bits (8.8402).

Rule: an accumulating kernel (`SPLICE_ADD`, `acc = 1`, `+=`) needs a cleared destination; a copy kernel needs none. When a
clear is dropped, every writer of that buffer is read for its kind, not for "covers all rows". Parity tests that exist to
protect a device path must run the shape that selects it (real head widths, more than one chunk) on the device whose memory
is not zero-filled; a tiny shape on the CPU backend certifies the algorithm, not the allocator contract.

Also added: the global gradient norm of every clipped step (`Model::last_grad_norm`, `StepReport::grad_norm`, one
`step N  loss  grad_norm  lr` line per step), so a run whose clip is eating the signal is visible in its log.
