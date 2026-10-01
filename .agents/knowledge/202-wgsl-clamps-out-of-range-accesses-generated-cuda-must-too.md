<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 202. WGSL clamps out-of-range accesses; generated CUDA must too

`qwen35`'s `chunked_prefill_at_int8_matches_token_by_token_with_persistent_gdn_state`
passed alone on the CUDA backend and failed often under a parallel `cargo test`,
with differences from one ulp to 3e-3. The CPU backend was clean. (A second,
independent cause of the same symptom is #203.)

`compute-sanitizer --tool memcheck` on the single test named the cause: the
generated `brain_matmul_i8_dyn` made out-of-bounds global **reads**, for
example 113 bytes past a 16-byte allocation. WGSL guarantees an out-of-range
array access never touches memory outside the array (the implementation clamps
the index, returns zero or drops the store). naga's backends clamp to the last
element ("restrict"), which is what wgpu runs the project's kernels under, and
the kernels lean on it: a register-tiled matmul reads a few elements past the
end at a ragged tile edge and masks the result. `wgsl-cuda` emitted the raw
index, so the read landed in whatever allocation lay next. That is
nondeterministic by construction and only visible when other threads are
allocating, which is why a test run alone never showed it.

The fix is in the generator, not in the kernels: every storage, workgroup,
local-array and dynamic vector-component index is clamped
(`__brain_clamp(index, count)`). A pointer cannot say where its range ends, so
a generated kernel now takes the length in 32-bit words of each bound range as
extra `unsigned long long` arguments after its pointers (`wgsl_cuda::Kernel`
documents the ABI); the backend computes them per binding, from the slice for
a sliced step, and they are part of the graph node signature so two
submissions differing only in a slice length never share a captured graph.
`arrayLength` reads the same value, which made `flash_attn_bidir_spans`
translatable. Native (`kernels-cuda`) kernels keep their own signature.

Evidence: memcheck on that test, on `chunked_prefill_..._past_one_gemm_tile`
and on the model oracle tests reported out-of-bounds reads before and none
after; `wgsl_cuda_bounds.rs` pins the semantics on a device with marks around
each binding (red without the clamp).

Rule: a flaky GPU test that passes alone is an out-of-bounds access until
memcheck says otherwise, and a translator must reproduce the source language's
bounds guarantee rather than inherit the target's lack of one.
