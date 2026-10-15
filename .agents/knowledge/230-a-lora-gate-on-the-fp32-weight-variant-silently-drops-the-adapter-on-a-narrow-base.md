<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 230. A LoRA gate on the fp32 weight variant silently drops the adapter on a narrow base

`Qwen35::ops_linear` used to report `matches!(w, Weight::F32 { .. })` and its
callers applied `lora_fwd` only when it did. That reads as "LoRA only targets
unquantized weights", but it is a gate on the storage variant, not on what the
adapter needs. Put the base in bf16 and every adapter is skipped in the
forward: the model computes the plain base, B never receives gradient through
the output, and nothing fails - the adapters stay at their init and the loss
does not move.

What the adapter needs is a base whose matmul reads fp32 activations (the
tier decodes the WEIGHT on the fly), because the correction `x A^T B^T` is
computed from that same fp32 `x`. `train_base::reads_float_acts` states it as
the negation of the packed-activation tiers (int8, int4, K-quant), so a new
float-read tier (NF4) is covered without touching the gate. The same predicate
decides `ops_act`: those tiers read `act_f32`, never an int8 packing of the
activation, which would be two dispatches and a scratch allocation per linear
that nothing reads.

The backward needs the weight through the same tier: `dX = dY W` goes through
`Ops::matmul_dx(Weight)` for the 12 per-layer linears (the packed buffer is
not in the fp32 store at all, `ps.w(name)` panics for it); only `lm_head` and
MTP's `fc_e`/`fc_h`, which are fp32 store tensors, keep the plain kernel.

`tests/lora_bf16_base.rs` is the gate: with the predicate narrowed back to
`Weight::F32` the bf16 build's in_proj_qkv.lora_a gradient has cosine 0.07
against the fp32 build's, the save/reload test fails, and the overfit test
does not learn.
