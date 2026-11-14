<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 456. Two int8 evaluations of one model diverge because the W8A8 function is discontinuous, so a fixed floor between them is not a bound

**Lesson:** A W8A8 forward rounds every activation to an int8 code, and a code jumps a whole quantization step
when its input crosses a rounding midpoint. Two evaluations of the same quantized model that differ only in
binary32 rounding (another accumulation order, a fused kernel, another device) therefore agree to the last bits
until one code lands on the other side of a midpoint, and from there they diverge layer by layer. Neither their
distance from each other nor the quantized model's distance from fp32 is bounded by anything the int8 kernels
control. A parity gate for an int8 tier compares it with the reference evaluated under the same rounding, and
states the quantization loss as a property of the weights, not of the code.

## What happened

The Qwen3.5-35B-A3B reference goldens (`tools/goldens/qwen35moe_dump_reference.py`, tiny random weights, 8
layers) put the int8 tier at 0.195 / 0.394 / 0.461 relative L2 from the fp32 reference's logits on three prompts,
against the bound the crate documented for it (`model_i8_smoke`: cosine above 0.99, relative L2 below 0.1). The
reference itself, with the same rounding applied at the same sites, is 0.195 / 0.418 / 0.461 from fp32, and the
tier matches that W8A8 reference to 4e-7 on the prompt where no code flips. On the other two, the tier and the
W8A8 reference agree to 5e-7 for the first layer or three and then part: 2.6e-3 after the first flip, 7.8e-2 at
the last layer. The reference's own binary32 and binary64 evaluations of its W8A8 function part the same way
(3.8e-2 and 2.4e-2 at the logits) while the unquantized fp32 and fp64 forwards stay within 1.6e-5.

## Why

- The documented bound was measured on `init_weights`, whose branches are small against the residual stream:
  the int8 tier measured 6.6e-6 from fp32 there, so the test certified the wiring and never saw quantization.
- The Gated DeltaNet projections carry most of the loss (`in_proj_qkv` alone: 2.6e-2 after one layer, against
  6.6e-3 for a layer's experts), and later layers amplify it.
- `fast_path_parity_real` holds the fast int8 path to the portable int8 path with a fixed relative-L2 floor
  (`PREFILL_REL_L2 = 5e-2`) and fails on one prompt at 5.06e-2. Both sides there are int8 evaluations that differ
  in rounding, which is exactly the comparison above: at forty layers a code flip early in a prompt can carry the
  last hidden state past any fixed floor without either path being wrong.

## Fix

`reference_parity::the_int8_tier_computes_the_reference_w8a8_function` compares the int8 tier with the
reference's W8A8 function (the tier must sit nearer it than it sits from fp32) and prints the loss and the
binary32 against binary64 spread from the manifest. The fp32 tier is held to the binary32 round-off bound
element by element in the same file. `model_i8_smoke`'s and `q8.rs`'s documentation say what their numbers are
(the untied head is int8 under an int8 policy, the smoke tolerance is that init's).
