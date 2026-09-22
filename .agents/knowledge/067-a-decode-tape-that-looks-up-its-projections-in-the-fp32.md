<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 67. A decode tape that looks up its projections in the fp32 `ParamStore` cannot serve a quantized build, and a prefill-only int8 test cannot see it

`qwen35::model::Qwen35::new_i8(..).step(tok)` panicked with `no weight
blocks.0.linear_attn.in_proj_qkv.weight` - not a subtle numeric wrong answer, a
hard panic on the first token of the first layer. An int8 build deliberately
excludes the 12 `is_i8_linear` leaves from the fp32 `ParamStore` (they live in
`self.weights` as `model::ops::Weight`s instead), and PREFILL respected that -
`layer_gdn_fwd`/`layer_gqa_fwd`/`mlp_fwd` all dispatch through `ops_linear`.
The DECODE siblings, written later as "the same math at `n = 1`", reached for
`self.w(&p("q_proj.weight"))` directly, which is the store that no longer holds
it. So the crate's whole int8 tier was prefill-only, and `crates/qwen35moe`'s
decode step has the identical defect at the identical call sites.

**Why five int8 tests and eight decode tests all passed.** They partition
cleanly: every int8 test (`model_i8_smoke.rs`, `int8_real_weight_sanity.rs`)
calls `logits_all` - prefill; every decode test (`decode_step.rs`, `serve.rs`,
`sample_generate.rs`, `caps.rs`) builds with `new_on` - fp32. Nothing in the
suite ever crossed the two axes, and a defect that lives ONLY in the crossing
is invisible to any number of tests on either axis alone. Two-axis features
(tier x path, tier x shard, path x shard) need a test at the crossing, or the
crossing is untested however green the matrix margins look.

**The generalisable rule.** When a build makes a weight reachable through ONE
accessor only, every consumer must go through that accessor - a second lookup
path that happens to work for the default tier is a tier-specific landmine.
`ops_linear` is that accessor here; `self.w` is the fp32-only shortcut, and it
is spelled almost identically. Worth noting the shape of the correct fix too:
the activation side is NOT free to copy from prefill, because `Ops::act`'s
int8 packing is two dispatches plus an allocation per activation and a decode
tape pays it per layer per token where prefill amortizes it over `t` rows -
`qwen3::model::Qwen::ops_act` already had the answer (fall back to `act_f32`
when every resident weight is `F32`), and the qwen35 decode path now mirrors it.
