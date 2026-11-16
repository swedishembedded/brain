<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 496. A chat fine-tune moved a looped decoder's exit mass to its first pass, and the server read the last one

**Lesson:** When the training objective is a mixture over outputs (Ouro weighs each pass's loss by its exit
probability) and serving picks one output by a separate rule, check that the rule picks the output the objective
trains. Evaluate every output the model has, not only the served one: the loss can fall while the served output
gets worse.

## What happened

An application training a small agent model reported that `full_train` (role-masked chat SFT) of a from-scratch Ouro
dropped its development probes from 3/12 to 0/12, while the same conversations trained as text (`pretrain`) kept them. The chat path was checked
and is consistent: rendering and masks match the template, the training graph's loss on a padded row equals the
inference model's on CPU and on gpu3, gradient accumulation adds every micro-batch (`backwards_accumulate`, now a
conformance check), and the exported checkpoint is the trained one.

Measured per pass on gpu3 (the application's 2.1M pretrained model, 4,400 demonstration records of 1,200 episodes,
150 held-out records of other episodes, 400 steps of 32 records at lr 7e-4): before the fine-tune the first pass
held 96% of the exit mass and the first and last passes got 0.36 and 0.39 of the held-out calls wholly right. The
fine-tune moved 98% of the mass to the first pass, which rose to 0.56; the last pass, 2% of the objective, fell to
0.03. The checkpoint's `early_exit_threshold` was 1.0 (always the last pass, the released checkpoints' setting), so
the server read the pass the fine-tune had stopped training. The text objective kept both passes good (0.62, 0.61).

## What changed

- A from-scratch Ouro whose configuration names no `early_exit_threshold` is initialised with 0.5
  (`ouro::finetune::FROM_SCRATCH_EXIT_THRESHOLD`): the server reads, per token, the pass at which half the exit mass
  is reached, the one the objective trains. A threshold the configuration names is kept; checkpoints initialised
  before this keep their 1.0 and need re-initialising (or a configuration naming the threshold).
- `crates/sdk/tests/it/from_scratch_chat.rs`: over every from-scratch architecture (ouro, qwen3), a chat fine-tune
  of a pretrained model makes the served answers to unseen conversations more likely, scored teacher-forced through
  the serving engine.
