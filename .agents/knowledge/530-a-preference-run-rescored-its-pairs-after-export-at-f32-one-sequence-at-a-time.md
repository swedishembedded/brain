<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 530. A preference run rescored its pairs after export at f32, one sequence at a time, for longer than it trained

**Lesson:** A step that only reports on a run (a score of the exported adapter) must cost a small fraction of the run,
and is sized by the run's own arithmetic, not by a wider reference tier. Count the forwards before choosing a default:
a scorer that is cheap on a toy base is hours on a 4B base with 8k-token rows.

## What happened

A served DPO job on Qwen3-4B (bf16 base, 175 training pairs of about 8k tokens a side) trained its 67 steps in about 20
minutes, then sat in post-training scoring for more than 85 minutes with the card at 100%. Qwen3's trainer scores the
exported adapter (`SCORES_EXPORTED_PAIRS`), so after the run `qwen3::eval::preference_margins` loaded a fresh model for
the reference and one for the policy, both at f32 whatever tier the run trained at, and scored every candidate of every
pair, training and held-out, one sequence at a time: four forwards of one 8k row per pair, about 1,400 for this job.

## What changed

- The scoring models hold the base at the run's own tier when it is bf16 (a quantized base still scores at f32, as before),
  carried as an argument of `ChatTrainer::preference_margins`. The public `score_preference` stays at f32.
- The training pairs of a run are scored on an even sample of at most 32 (`MAX_TRAIN_SCORE_PAIRS`); the held-out pairs,
  which are the measurement, are scored whole.
- Gated by `dpo_trains_over_the_base_format_it_is_given` (the bf16 run's training margin against the public f32 scorer's,
  within a tenth) and `the_training_score_sample_is_even_ordered_and_bounded`.

## Not done, and why

Batching the sequences: a 8k-token row already fills the card (the cost is the forward, not a launch gap), and a batch
of rows that long multiplies the activation memory; the lever is the tier and the number of forwards. Reusing the
resident trained model for the score would remove the reloads and the reference forwards (the trainer holds the
reference margins it scored before its first step) but needs the Qwen3 trainer to keep its model across the export,
which it does not today. Feedback (KTO) and distill scoring have no post-run pair scoring and are not affected. The new
cost was not measured on a card.
