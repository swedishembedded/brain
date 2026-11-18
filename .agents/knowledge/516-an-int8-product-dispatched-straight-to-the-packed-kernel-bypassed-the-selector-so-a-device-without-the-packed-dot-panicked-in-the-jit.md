<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 516. An int8 product dispatched straight to the packed kernel bypassed the selector, so a device without the packed dot panicked in the JIT

**Lesson:** A model that names a kernel itself (`gpu.dispatch(K_MATMUL_I8_DYN, ...)`) has made the selection the
selector exists to make, and made it for every device. The selector answers `Reference` alone for an int8 weight on a
device without `numeric.int8_dot`, and no int8 tier has a reference kernel, so the only caller that can run there is
one that asks the selector and reads the codes through the float-read route it leaves. Models call the op; the op
chooses.

## What happened

ltxv's quantized blocks (`qgemm`) dispatched `matmul_i8_dyn` (and `matmul_q4_dyn`) by slot, after their own
`max_abs_row` / `quant_pack` / two-level residual steps. On the CPU backend (`int8_dot` false, so the packed
kernel is not JIT-compilable) 22 ltxv tests aborted with `wgsl-cpu: kernel "matmul_i8_dyn" was not JIT-compiled`:
`device_residency`, `av_dit_parity`, `block_weight_cache`, `lora_runtime` and `int8_compute`. Production would have
hit the same on any device lacking the packed dot. `wan` and `s3dit` dispatched the same kernel the same way and
failed the same way (`wan` `lora_runtime_dev` x2, `s3dit` `dit_i8_rebuilt_from_cache`); `gemma4`, `flux1` and `flux2`
did too, behind a model-side "int8 needs a GPU" refusal that hid it from their CPU tests.

## Why

ltxv predates `Ops::act_two_level` (which was extracted from its two-level activation) and kept a private copy of the
quantize, product and fold chain. A private copy of the dispatch cannot follow the selector: the capability check,
the tuned native candidates and the decode-regime GEMV all live behind `Ops::matmul`.

## What to do

* ltxv's `qquant_linear` builds a `Weight::I8` / `Weight::Q4` over its uploaded codes and calls
  `Ops::act_two_level` + `Ops::matmul`. Its kernel list is its indexed table followed by the facade sets
  (`two_level_kernel_list`, `frozen_base_kernels`).
* `Ops::matmul` on an int8/int4 weight, where `select::candidates` offers `Reference` alone, reads the weights through
  the tile decode every non-dense tier has (`matmul_rows`), against the f32 rows; `Ops::act` skips the quantisation
  nothing then reads. The tile scratch is bounded by `DEFAULT_TILE_BUDGET_BYTES` (256 MiB), inside ltxv's
  `activation_reserve_bytes` base, so residency planning is unchanged: the resident bytes stay the packed codes.
* `crates/model/tests/it/ops_int8_without_packed_dot.rs` pins the route against the decoded weights.
* wan, s3dit, gemma4, flux1 and flux2 (DiT and the device trainer's int8 frozen base) take the same route. A model
  whose linears read row ranges of one activation (a joint text/image slab, a batch) quantizes the slab once and
  reads each range through `Ops::act_rows`; one whose stack is recorded once uses `Ops::act_shared_scratch`, one
  packed copy per width and row count, reused across blocks.
* `make check/packed-dispatch` (`scripts/gates/check-packed-dispatch.sh`) fails a model crate that registers, looks
  up or records a packed kernel by name; `scripts/gates/packed-dispatch.allow` lists, with reasons, the int8 MoE
  expert paths (no `Ops` int8 expert product yet), `qwen3::serve`'s own tuned selector and two benches, and only
  shrinks. Migrate a listed path to `Ops` rather than add a capability check to the model.
