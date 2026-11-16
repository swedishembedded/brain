<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 497. Without the exit-entropy term, a from-scratch looped decoder left its second pass unused and its chat fine-tune collapsed

**Lesson:** An adaptive-depth objective that weighs each depth's loss by its exit probability rewards whichever depth
learns first, and the exit mass runs there and stays. The published recipe (arXiv:2510.25741, Stage I) subtracts
`beta` times the entropy of the exit distribution (a KL to a uniform prior over exit steps, up to a constant) so
every depth keeps training signal. Measure exit mass and per-depth quality before trusting that the extra depth is
in use.

## What happened

brain's Ouro objective was the exit-weighted loss alone. After
[496](496-a-chat-fine-tune-moved-a-looped-decoders-exit-mass-to-its-first-pass-and-the-server-read-the-last-one.md)
the from-scratch models were seen to put most of the exit mass on the first pass. Measured on gpu3 with a 2.1M
two-pass model from one initialisation (pretraining: 1,500 steps of 32 rows of 512 on the application's world
corpus; held-out: 60 world documents with a "the code of NAME is ..." retrieval each; then a chat fine-tune of 400
steps of 32 demonstration records at lr 7e-4, scored on 150 held-out records of other episodes; values first pass /
second pass):

| | exit mass | held-out loss | token accuracy | retrieval | held-out calls right |
|---|---|---|---|---|---|
| pretrain, beta 0 | 0.86 / 0.14 | 1.582 / 1.590 | 0.569 / 0.572 | 0.782 / 0.767 | 0.61 / 0.59 |
| pretrain, beta 0.1 | 0.49 / 0.51 | 1.527 / 1.525 | 0.580 / 0.584 | 0.789 / 0.802 | 0.71 / 0.70 |
| fine-tune, beta 0 | 0.96 / 0.04 (chat positions) | | | | 0.14 / 0.09 |
| fine-tune, beta 0.05 | 0.50 / 0.50 (chat positions) | | | | 0.79 / 0.60 |

With the term the second pass carries the better pretraining predictions (lower loss, better retrieval), the whole
model is better at both depths, and the chat fine-tune improves instead of collapsing. After the fine-tune the first
pass is the stronger one; a 48-token repeated-sequence copy probe was at chance on every model (no in-context
copying learned at this size and budget), so it does not separate them.

## What changed

- `model::Model::set_exit_entropy` / `FitOpts::exit_entropy`: `loss - beta * H(p)` over the supervised positions,
  with the same masking and averaging as the task term; the derivative enters the gate gradient as `beta * ln p_j`
  per pass (`gradcheck::check_ouro_exit_entropy`, finite differences at beta 0.05, 0.1 and 1). A model without exit
  gates refuses a non-zero weight by name.
- `Pretrain` and `ChatFullFineTune` (and the served `pretrain` and `full_train`, param `exit_entropy`) default to
  the recipe's weights for a family with exit gates: 0.1 for pretraining (its first stage), 0.05 for a fine-tune
  (its later stages); 0 turns the term off. The weight is part of a resumable run's identity and of the report.
