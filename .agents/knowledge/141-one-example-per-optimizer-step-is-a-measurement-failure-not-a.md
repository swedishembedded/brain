<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 141. One example per optimizer step is a measurement failure, not a speed one

**Lesson:** A training loop's batch size is part of its contract, not a tuning detail. When an
accuracy looks like an architecture ceiling, rule out the optimizer first.

## What happened
`DecisionPipeline::train_choices` took one example per step. Every caller of the decision SDK
(four samples and every published accuracy for that arm) trained at batch one.

The trap is that batch one TRAINS. The loss moves, accuracy beats chance, and a reader concludes
the model has learned as much as this architecture can and goes looking for a better architecture.
This repo did: the sample's own README reported 3.3% and named the model as the limit.

## Why
`crates/decide` had `zero_grads`/`accumulate`/`adamw_scaled` from the beginning, gated by
`tests/minibatch.rs`. The SDK simply never passed a batch size down.

What that costs is not throughput. It is the only thing the run is for. At batch one the logged
loss alternates between ~0 and ~6 from step to step. AdamW's moment estimates are tracking a single
decision's gradient, which is a very noisy estimate of the objective's, and the run spends its
budget chasing it.

## What to do
A training loop's batch size is part of its contract, not a tuning detail left at whatever the
first implementation happened to do. A batch of one needs the same justification any other unusual
choice does.

When an accuracy looks like an architecture ceiling, rule out the optimizer first. It is cheaper to
test than a new model, and a noisy-gradient run and a capacity-limited one look identical from the
outside.

## Evidence
Measured on `samples/decision/rubiks`, whose labels come from an exact planner so held-out accuracy
is checkable rather than estimated, at a FIXED example budget of 32k:

| batch | held-out first pick gets closer |
|---|---|
| 1 | 30% |
| 32 | 85% |

Same examples, same learning rate, same everything else.
