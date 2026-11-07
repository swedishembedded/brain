<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 352. A kernel name derived from the source's address gave three flash kernels one name in release builds

**Lesson:** A name recovered from a source string's address is not stable across build profiles,
and debug-only tests cannot see the breakage. State each name beside its source, and run a
consumer once in the release profile after any kernel-set change.

## What happened
The bidirectional flash attention passed every debug-build test. The same trainer in a `--release`
step-time run died in wgpu with "Number of bindings in bind group descriptor (9) does not match
the number of bindings defined in the bind group layout (6)" on
`flash_attn_causal_gqa_bwd_dkv#CAUSAL=0,UNIT_SCALE=1`.

## Why
`block::flash_bidir_kernels` specialised the three flash kernels and chose each one's base name by
comparing the source `&'static str` against `kernels::FLASH_ATTN_CAUSAL_GQA` with `std::ptr::eq`.

- Each use of a `const` string may be a distinct allocation. The debug profile compared them
  equal; the optimised one did not (inferred from the symptom and from the fix removing it).
- Every source therefore fell through to the last arm and all three were registered as
  `...bwd_dkv#...`.
- The forward then dispatched through the dkv pipeline's six-binding layout.

## What to do
State the base name beside its source
(`specialise("flash_attn_causal_gqa_bwd_dq", kernels::FLASH_ATTN_CAUSAL_GQA_BWD_DQ)`), never
recover it from a pointer. `the_specialised_kernels_have_distinct_names_in_trio_order` pins the
names and their order.

A test in the debug profile cannot see this class of defect; any kernel-set change gets one
release-profile run of a consumer.
