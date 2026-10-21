<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 352. A kernel name derived from the source's address gave three flash kernels one name in release builds

Symptom: the bidirectional flash attention passed every debug-build test, and the same trainer in a `--release` step-time run died in
wgpu with "Number of bindings in bind group descriptor (9) does not match the number of bindings defined in the bind group layout
(6)" on `flash_attn_causal_gqa_bwd_dkv#CAUSAL=0,UNIT_SCALE=1`.

Cause: `block::flash_bidir_kernels` specialised the three flash kernels and chose each one's base name by comparing the source
`&'static str` against `kernels::FLASH_ATTN_CAUSAL_GQA` with `std::ptr::eq`. Each use of a `const` string may be a distinct
allocation (the debug profile compared them equal; the optimised one did not, which is inferred from the symptom and from the fix removing it), so every source fell through to the last arm and all three
were registered as `...bwd_dkv#...`; the forward then dispatched through the dkv pipeline's six-binding layout.

Fix: the base name is stated beside its source (`specialise("flash_attn_causal_gqa_bwd_dq", kernels::FLASH_ATTN_CAUSAL_GQA_BWD_DQ)`),
never recovered from a pointer. `the_specialised_kernels_have_distinct_names_in_trio_order` pins the names and their order. A test in
the debug profile cannot see this class of defect; any kernel-set change gets one release-profile run of a consumer.
