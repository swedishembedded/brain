<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# modelopt - model optimization

Scope: Measured, budget-constrained model compression (mixed precision, KV precision, QAT/QAD, pruning, distillation, draft training) in `crates/modelopt`.

Goal: given a model, a representative workload, a target device and a quality
budget, brain finds and produces the best validated deployment variant, with
no per-architecture optimization code. Every choice is measured, every result
is checked end to end against the original, and every artifact can be
reproduced from its manifest.

## Where it lives

`crates/modelopt` is a layer-5 crate (beside `rl` and `modelcost`). It composes
contracts the lower layers already own and adds only the search:

| responsibility | owner |
|---|---|
| a tier's codec, its host round trip and its physical bytes | `model::ops::Weight` (`round_trip`, `bytes_for`) |
| the quantizable linears of a model, their shapes and roles | `Model::adapter_sites` (`model::adapter::LinearSite`) |
| loss, gradients, weight read/write, held-out scoring | `model::Model`, `model::score` |
| the serving precision a result is expressed in | `model::ops::TierPolicy` (`--weight-format`) |
| KV clip ceilings | `model::kvcalib` |
| measured GEMM cost per tier on a device | `model::probe` |
| packed GGUF export | `checkpoint::quantize` (`brain quantize`) |
| distillation objective | `rl::objective::distill` |
| sensitivity, cost tables, constrained search, validation, reports | `modelopt` |

A transform separates analysis from execution: `analyze` measures,
`plan` decides under a budget and touches no weight, and only `apply`
changes the model. A plan is inspectable before anything is written.

## Invariants

1. **The search estimate never certifies a result.** Additive sensitivities
   pick candidates; only a whole-model measurement on held-out data accepts one.
2. **Costs are physical.** Memory is `Weight::bytes_for` (packed words and
   scales), not average bit width; time comes from a measurement on the target
   device, never from a spec table.
3. **A format is a candidate only where it executes.** A tier the device or the
   model cannot run is excluded from the search, not costed as if it could.
4. **Calibration data and acceptance data are disjoint.** The set a search
   ranks on is never the set that accepts its result.
5. **Simulated quantization is the deployed codec.** Fake quantization is
   `Weight::round_trip`, the same host codec `Weight::upload` packs with.
6. **Every result names its baseline** (the original, and every uniform tier
   that fits the same budget) and reports the measured delta against each.

## Weight precision search (AutoQuantize)

Per linear site `i` and candidate tier `q`: sensitivity
`E[i,q] = 1/2 * sum_j F_j * (w_j - Q_q(w_j))^2`, with `F` the diagonal empirical
Fisher (mean squared gradient over calibration batches), the second-order
Taylor estimate of the loss increase. The solve is a multiple-choice knapsack:
minimize `sum E[i, q_i]` subject to `sum bytes[i, q_i] <= budget`.

- [x] `Weight::round_trip` and `Weight::packs` in `model::ops`: the host
      codec every uploadable tier packs with, decoded back to f32, held equal
      to the device's own decode (`model/tests/weight_round_trip.rs`).
- [x] `modelopt::sites`: the stored tensors behind `Model::adapter_sites`,
      with `[n, k]`, layer and roles, joined into the groups a deployed engine
      fuses (`Model::coupled_tensors`; qwen3's Q/K/V and gate/up), which take
      one tier.
- [x] `modelopt::sensitivity`: diagonal-Fisher sensitivity per site and tier.
- [x] `modelopt::assign`: greedy multiple-choice knapsack under a byte
      budget (`Plan::policy` is the `TierPolicy`, f32 outside the plan).
- [x] `modelopt::measure_policy`: a plan applied by fake quantization, scored
      on held-out batches, restored.
- [x] Acceptance (`modelopt/tests/autoquant.rs`): a trained tiny qwen3 at a
      budget halfway between uniform int8 and q4 (43008 of 196608 f32 bytes)
      meets the budget, has lower held-out loss than every uniform tier that
      fits it, and beats the median of the same tiers dealt at random among
      same-shape sites. Held over 13 random seeds.
- [ ] Bounded host memory: `sensitivity` holds one f32 per site element; a
      model whose linears do not fit host RAM in fp32 needs it per layer.
- [x] Runtime cost: every distinct site product timed per tier on the
      device (`Timing`, `model::probe::gemm` in the SDK) at decode rows; the
      solver limits bytes and, optionally, summed product seconds
      (`Limits`); reported for the plan, f32 and each baseline.
- [ ] Time a product on the serving engine's own kernel selection (the
      engine's tuned int8 GEMV, fused groups) rather than the ops façade's.
- [x] `modelopt::search_weights` and its `Report`: estimate, solve, then
      measure the plan, the original and every uniform tier that fits the
      budget; calibration and held-out batches must not overlap.
- [x] Measurement as deployed: through the family's serving engine where it
      has a scoring path (`ChatTrainer::served_loss`; qwen3
      `eval::served_loss` over `serve::Engine` at the policy, int8 activation
      quantization included), else on the trainer's frozen-base build
      (`LoraRequest::base` is a `TierPolicy`; a zero-delta adapter leaves the
      base's loss). The trainer build decodes packed weights to fp32 products,
      so it does not see activation quantization; the outcome says which ran.
- [x] Front end: `brain::ModelOptimizer` (`crates/sdk/src/optimize.rs`) over
      `ChatEntry::{search, measure}`, and `brain optimize` over it; accepted
      only on the deployed measurements; `optimize.json` holds the outcome.
      Candidates are trainable bases (measurable), `ChatTrainer::SERVED_WEIGHTS`
      (servable per linear) and held by the device as themselves.
- [x] Serving a plan: qwen3's engine takes a per-leaf `TierPolicy`
      (`Engine::check_tiers` refuses a split fused group);
      `serve::tests::mixed_tiers_serve_each_linear_at_its_own_tier`.
- [ ] A real checkpoint: Qwen3-0.6B on ARC-Easy chat records. At the uniform
      int8 budget the only feasible plan was all-int8 (f32/bf16/int8
      candidates), which is why the default budget is midway; the midway run
      and serving its policy through `brain serve` are next.
- [ ] qwen35: no engine scoring path yet (`served_loss` is `None`), so its
      plans are accepted on the trainer build.
- [ ] Faster host codecs: fp8 and nf4 round trips are serial and dominate a
      search that offers them on a 0.6B model.

## KV precision search

`model::kvcalib` sizes a chosen tier's scale; this chooses the tier per layer
(and per K/V side where the kernels allow).

- [ ] Per-layer KV tier in `ServePolicy`, refused per model where unsupported.
- [ ] Sensitivity per layer and side, the same knapsack over KV bytes per token.
- [ ] Acceptance: a per-layer map beats uniform int8 KV at equal bytes on a
      long-context held-out set.

## Quantization-aware training and distillation (QAT/QAD)

- [x] Recovery by distillation over the quantized base: `rl::fit::TeacherFit`
      collects the original's top-K distributions on the calibration split,
      `DistillFit` trains a LoRA adapter over the base held at the plan
      (`ChatEntry::distill`), and the plan is measured without and with it on
      the same trainer build (`ModelOptimizer::recover`, `--recover-steps`).
- [ ] Measure the recovered plan on the serving engine with the adapter as an
      overlay (the engine scoring path folds an adapter into the weights
      before quantizing, which is not how a quantized base serves one).
- [ ] Full QAT: fake-quantized forward with a straight-through backward
      through the generic trainer, updating the base weights themselves.
- [ ] Acceptance on a real checkpoint: recovery lowers the plan's held-out
      loss, and the adapter serves beside the policy.

## Structured pruning and distillation

- [ ] Importance of heads, MLP channels and whole layers from the same Fisher
      statistics; the pruned config, tensors and site list rebuilt consistently.
- [ ] Recovery by distillation from the original.
- [ ] Acceptance: the smaller model runs smaller dense shapes (no masked
      compute) and its measured objective improves at equal quality.

## Speculative draft training

- [ ] Train a draft model or head against `model::serve::speculate`'s drafter
      seam; score by measured acceptance rate and end-to-end speedup.

## Variant selection

- [ ] Pareto set over weight policy, KV policy and draft across devices, each
      variant content-addressed with its source digest, manifest and
      validation results.
