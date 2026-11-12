<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 436. A policy gradient that revisits a fixed prompt set lets its negative advantages memorise the prompts

**Lesson:** When a policy-gradient run does not improve the held-out score, check the data before the
loss. The objective was sound (step-0 ratio exactly 1, gradchecks hold); what decided the outcome
was whether the prompts were fresh each round. Negative advantages alone destroy the model, and on
a reused prompt set they memorise it.

## What happened

Question: the default two-sided, standardised advantage gave held-out exact match of 0.38-0.44 or a
collapse on every learning-rate, KL and shaping variant tried, while a positive-only advantage
reached 0.60. Was the objective wrong?

## Why

Toy `qwen3` LoRA model (`brain::toy`, a three-letter cipher, held-out exact match, CPU):

- The objective is correct: the first step's importance ratio is exactly 1.000 under the
  start-of-run behaviour, and the finite-difference gradchecks hold.
- Negative advantages alone drive the held-out loss from 2.93 to 5.9; positive advantages alone
  improve the score.
- Revisiting the same 90 prompts every round lets the negatives memorise those prompts: held-out
  0.39 -> 0.44 -> 0.34 over 8 rounds.
- A fresh slice of prompts each round, same objective and settings, improves every round: 0.39 to
  0.53 over 6.
- Larger groups on the same prompts help little; a higher learning rate or a wider clip band makes
  it worse.

## What to do

- A policy-gradient loop draws fresh prompts every round. Do not tune the loss, the learning rate
  or the clip band to rescue a run that reuses its prompts.
- A positive-only advantage is available where a reused-prompt regime must be supported; with the
  default two-sided advantage a reused set is a defect of the data plan.
- Open: whether the same holds on a real model was not run.

## Evidence

Roadmap `grpo-corrections.md`, section "External policy gradient on a toy" (brain 520f34e44), on
the toy checkpoint of ab90a4d10. Related: #290 (the objective's own fixes), #292.
