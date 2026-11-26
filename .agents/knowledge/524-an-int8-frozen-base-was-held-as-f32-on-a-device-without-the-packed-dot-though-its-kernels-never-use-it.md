<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 524. An int8 frozen base was held as f32 on a device without the packed dot, though its kernels never use it

**Lesson:** `Dtype::promote` answers "can this device run the tier's packed kernel", which for int8 is the
`numeric.int8_dot` capability. A caller that reads the codes by float arithmetic (the weight-only `#w=w8` kernels, or
`Ops::matmul`'s tile decode) needs no such capability, and widening its weights to f32 on a device without one gives
it a base four times larger than it asked for while the test that asks for int8 fails on the very device that is
slowest to hold f32. Decide the tier by what the caller's kernels need, in one place, not by the packed kernel's
capability.

## What happened

Five tests of the `rl` integration target failed on the CPU backend, all qwen3 over int8: the frozen base of
`objective_fits` (DPO, GRPO and distillation), `lora_acceptance` ("the base is held as F32") and the mixed-tier base of
`chat_trainer_bases`. They are only built with `--features qwen3`, so a run of `cargo test -p brain-rl` without it ran
the 38 other tests and passed; the failing set is not a regression of an earlier fix but a set that plain runs never
built. The CPU backend reports `int8_dot: false` on purpose (its packed WGSL kernels do not JIT), and
`Weight::upload`, `upload_from`, `upload_source::encoding_of` and qwen3's `lora_base_tier` each called
`want.promote(numeric)`, so every int8 request landed on f32 there.

## Fix

`Ops::lands(want)` is the one answer to "which tier does this device hold `want` at"; it is `promote` except that
int8 stays int8 on an `Ops` that was told its caller reads codes by float arithmetic
(`Ops::hold_int8_by_float_read`, since renamed `Ops::hold_packed_by_float_read` when it came to cover the K-quants too, #601). qwen3 sets it for a LoRA training build, whose int8 products are the `#w=w8`
kernels and the tile decode. A caller that dispatches the packed kernel itself keeps the widening. The weight-only
kernels themselves stay on devices with the native int8 dot (`int8_base`): the CPU JIT cannot compile their
register-tiled form, so there the int8 weights are read through `Ops`' tile decode like NF4 and FP8, and
`lora_int8_base` checks that route against the fp32 model's loss and gradients for real instead of returning early. Pinned by
`model` `ops_int8_without_packed_dot::an_int8_request_is_held_as_int8_only_where_the_caller_reads_the_codes_by_float_arithmetic`
and the five `rl` tests, which now assert int8 for real on the CPU backend.

## What to do

* A tier decision made before an upload asks `Ops::lands`, never `Dtype::promote` on the caps.
* Run the feature-gated test targets of a crate (`required-features`) when a change touches what they cover.
