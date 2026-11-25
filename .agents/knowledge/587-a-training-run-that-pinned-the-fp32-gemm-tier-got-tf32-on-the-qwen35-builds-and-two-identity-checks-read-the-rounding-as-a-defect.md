<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 587. A training run that pinned the fp32 GEMM tier got TF32 on the qwen35 builds, and two identity checks read the rounding as a defect

**Lesson:** `FitOpts::gemm_precision` reaches a model only through `Model::new_with_gemm`, and the trait's default
ignores it. A model whose products run through `Ops` must implement it, or a run that pins `Fp32Reference` silently
trains on the device's default (TF32 on CUDA). And a check that holds two paths of a model to an identity exact in
real arithmetic (one backward against the sum of backwards it is linear in, a gradient against a finite difference)
builds its model at the pinned tier, as #367 says. A TF32 product rounds its operands to a 10-bit mantissa, so
`round(a) + round(b) != round(a + b)` and the identity holds only to about `2^-11` of every gradient.

## What happened

2026-11-25, the async-GRPO suites on a GH200 (CUDA, `Auto` tier); on the CPU every one of them passed.

- `rl` `external_policy_gradient_scale::a_steps_gradient_is_the_documented_aggregation_over_unequal_records_of_both_advantage_signs`:
  `SeqMean` directional derivative, brain `3.493e-3`, FP64 reference `1.152e-2` (3.3x).
  - brain's own gradient was right. The CPU gives `3.516e-3`, and TF32 moved it by 0.7 %.
  - The reference was wrong. It is a central difference (`h = 5e-3`) of the loss through the model's TF32 forward,
    and that forward's rounding divided by `2h` was larger than the derivative.
  - With the test's model at `Fp32Reference` on the same card: brain `3.5165e-3`, reference `3.5365e-3`.
- `rl` `--test conformance`, `qwen35_conforms`, `qwen35moe_conforms` (C3b, extra targets equal their label
  equivalents) and `qwen35_trains_online` (score centring, the same identity): `blocks.3.self_attn.q_proj.weight.lora_a`
  was `1.18e-8` off at a largest gradient of `5.44e-5`, against a bound of `1e-4` of the largest.
- Measured per tensor, the off was not a missing term:
  - it was 2e-4 to 1e-3 of each tensor's own magnitude, every tensor of every layer alike;
  - the run-to-run difference was exactly 0, and so was the difference with zeroed extra-target coefficients;
  - at `Fp32Reference` it fell to about 5e-7 of each tensor, and every check passed.
  - The full-attention layer's tensors failed first only because they hold the largest gradients.
- The thread switch `model::ops::fp32_kernels` did not close it (off `1.12e-7` against `1.71e-7`). It pins only the
  library's fp32-weight GEMMs, not every product the build's tier decides.
- The conformance builds through `Model::new`, and `qwen35`/`qwen35moe` did not implement `Model::new_with_gemm`. So
  no generic caller could pin their arithmetic, and a training run's `gemm_precision = Fp32Reference` was dropped on
  both.

## What to do

- A model whose products go through `Ops` implements `Model::new_with_gemm`. `qwen3`, `qwen35` and `qwen35moe` do.
  `ricursive`, `ouro` and `inkling2` build at one fixed tier (`Fp32Reference`) and
  ignore a run's `Auto`; that is open in the async policy-gradient ledger.
- An identity or derivative check in the conformance suite builds with `model::conformance::build_exact`
  (`Model::new_with_gemm(.., Fp32Reference)`). A test of the same kind on one model builds with
  `new_with_gemm`/`TrainOpts::gemm_precision`, as `rl`'s `external_policy_gradient::lora_model_rows` does.
- Do not widen the bound for the tensor-core tier. At binary32 the identity holds to 5e-7, and that margin is what lets
  `1e-4` of the largest gradient see a dropped term. How close the tensor-core tier is to the fp32 tier is a separate
  claim, gated by `qwen35` `lora_gemm_precision`.

## Evidence

- `qwen35` `lora_gemm_precision::a_runs_gemm_precision_reaches_the_training_build`, `qwen35moe`
  `lora_freezes_base::a_runs_gemm_precision_reaches_the_training_build`: the build's `gemm_path` names the requested
  reference tier, on every device. Before the fix the `qwen35` one failed on the CPU: its path named the device
  instead (`qwen35moe` had no `gemm_path` to read).
- `rl` `--test conformance` (15/15) and `--test it` (111/111) on CUDA (GH200, `BRAIN_BACKEND=cuda`).
