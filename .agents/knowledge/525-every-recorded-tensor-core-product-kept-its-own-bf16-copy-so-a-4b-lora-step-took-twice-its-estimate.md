<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 525. Every recorded tensor-core product kept its own bf16 copy, so a 4B LoRA step took twice its estimate

**Lesson:** A recorded tape keeps every buffer a step binds alive for as long as the tape lives. A helper that
allocates a fresh scratch per call is cheap when the call runs once. On a layer stack recorded once and replayed,
it costs one copy per layer and pass. Reuse per shape belongs to whatever already promises in-order execution,
and the on-card footprint test is the gate that catches what the CPU gate cannot see.

## What happened

The ignored card test `on_a_card_a_qwen3_4b_lora_step_takes_no_more_than_its_reservation` (36-layer Qwen3-4B, bf16
base, every projection at rank 16) ran on a GH200 for the first time. At 4,096 tokens it took 33.06 GiB of the card
against a 22.89 GiB reservation, over a 15.36 GiB itemised estimate.

The device's own accounting, probed between build stages:

| Stage | Live GiB |
|---|---|
| struct built (weights, layer scratch, logits) | 15.36 |
| after recording the forward tape | 20.58 (+5.22) |
| after recording the backward tape | 32.73 (+12.15) |

- **Forward, about 155 MB a layer.** That is the bf16 rounding of xn1, ctx, xn2 and h at 4,096 rows. A bf16 base
  reads fp32 activations, so `qwen3::Qwen::ops_act` takes `Ops::act_f32`, whose bf16 cast (`Ops::bf16_rows`) was
  never shared, although the other branch already used `Ops::act_shared_scratch`.
- **Backward, about 340 MB a layer.** That is the bf16 rounding of every `dy` and recomputed `x` that
  `LibraryGemmProvider::product` casts for itself (`library::cast_bf16`, a new `storage_uninit` per call).

The CPU footprint test could not see either: the CPU backend runs no library products.

## Fix

- `LibraryGemmProvider::share_product_casts`: a product that rounds its own fp32 activation rounds it into one bf16
  scratch per size. The cast is the step right before the product that reads it, so in-order execution on one
  stream makes reuse safe.
- `Ops::share_product_scratch` turns that on and makes every `Act` of the `Ops` (`act_f32` included) cast into the
  per-shape scratch of `Ops::act_shared_scratch`, on that function's terms. `set_gemm_precision` keeps the setting.
- `qwen3::Qwen::new_impl` turns it on for every build. The model already promised that ordering for its packed
  activations.

Measured on gpu0 (GH200 96 GB), same test:

| Tokens | Taken before | Taken after | Reserved |
|---|---|---|---|
| 4,096 | 33.06 GiB | 17.13 GiB | 22.89 GiB |
| 16,384 | (not reached) | 28.55 GiB | 34.07 GiB |

Tests:

- `gpu-core` `cuda_library_gemm::shared_casts_hold_one_scratch_per_size_and_each_product_reads_its_own_activation`:
  four products of one size grow live bytes once, and each product still matches the f64 oracle for its own input
  after one in-order submit. Without the switch, each product allocates 64 KiB.
- The ignored card test above now passes at both lengths.

## Not covered

- Live bytes are still about 1.3 GiB (4K) to 2.2 GiB (16K) above the itemised estimate after the backward tape. The
  reservation's margin absorbs this, but no term itemises it.
- On CUDA, `a_lora_step_holds_no_more_than_its_footprint_and_the_footprint_is_tight` is short by 4.5 KB at 2 layers
  and 128 tokens, before and after this change. It is a CPU specification, and the fixed 64 KiB small-buffer
  allowance does not grow with the per-dispatch buffers a card adds.
- Other trainers that record through `Ops` (qwen35, Ripple) have not been checked for per-call casts on a card.
