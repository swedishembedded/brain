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
6. **Every result names its baseline** (the original, and the uniform tier of
   at least the same size) and reports the measured delta against each.

## Weight precision search (AutoQuantize)

Per linear site `i` and candidate tier `q`: sensitivity
`E[i,q] = 1/2 * sum_j F_j * (w_j - Q_q(w_j))^2`, with `F` the diagonal empirical
Fisher (mean squared gradient over calibration batches), the second-order
Taylor estimate of the loss increase. The solve is a multiple-choice knapsack:
minimize `sum E[i, q_i]` subject to `sum bytes[i, q_i] <= budget`.

- [ ] `Weight::round_trip(raw, n, k, tier)` in `model::ops`: the host codec
      every uploadable tier packs with, decoded back to f32.
- [ ] `modelopt::manifest`: the quantizable sites of any `Model` with their
      `[n, k]`, layer and role, from `Model::adapter_sites`.
- [ ] `modelopt::sensitivity`: diagonal-Fisher sensitivity per site and tier.
- [ ] `modelopt::solve`: exact-enough constrained assignment under a byte
      budget, returning a `TierPolicy`.
- [ ] `modelopt::validate`: apply a plan by fake quantization, score held-out
      loss, restore; compare against the original and uniform baselines.
- [ ] Acceptance: on a trained tiny qwen3, a searched policy meets its byte
      budget and has lower measured held-out loss than every uniform tier that
      is no smaller.
- [ ] Runtime cost: per-tier GEMM time from `model::probe` at each site's
      shape as a second constraint (`min_memory` with a latency cap, and
      `min_latency` with a memory cap).
- [ ] Front end: one verb through the SDK that the CLI calls, writing the
      policy and a JSON report (sites, tiers, estimated and measured deltas,
      baselines, calibration fingerprint, device).
- [ ] A real checkpoint: Qwen3-0.6B, searched against uniform int8 and bf16 on
      held-out text, with the served policy re-measured through `brain serve`.

## KV precision search

`model::kvcalib` sizes a chosen tier's scale; this chooses the tier per layer
(and per K/V side where the kernels allow).

- [ ] Per-layer KV tier in `ServePolicy`, refused per model where unsupported.
- [ ] Sensitivity per layer and side, the same knapsack over KV bytes per token.
- [ ] Acceptance: a per-layer map beats uniform int8 KV at equal bytes on a
      long-context held-out set.

## Quantization-aware training and distillation (QAT/QAD)

- [ ] Fake-quantized forward with a straight-through backward, through the
      generic trainer, using `Weight::round_trip` (no model-crate code).
- [ ] QAD: the student is the quantized model, the teacher the original, with
      `rl::objective::distill`'s top-K KL plus the task loss.
- [ ] Acceptance: QAT/QAD recovers measured held-out loss over PTQ at the same
      policy, the packed export reloads, and served logits match the simulated
      ones within the tier's tolerance.

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
