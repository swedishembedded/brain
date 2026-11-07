<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 421. The fit loop's initial loss was measured without the pictures of the records it reported, because its probe drew windows that named no example

**Lesson:** A trainer whose batches carry more than tokens must take every batch, including its loss
probe and its held-out draws, from the one path that names the examples. A second way of drawing a
batch is a second way of not feeding the model its inputs.

## What happened

Question (2026-11-01, LoRA SFT of Qwen3.8-27B on records with images): the first image-training test
passed on every training step and still had one number wrong. A record trained on picture A and the
same record on picture B reported the same `initial_loss` (3.3666084 to the last digit, at the f32
and the bf16 tier) although the pictures reached the loss of every training step. Why?

## Why

`CausalLm` draws a training batch from an example index (`draw_examples(picks)`) and names the
picks, but two of its draws did not: the loss probe that measures the loss before the first step
(`loss_probe`) and the random held-out draw (`eval` with a batch count) took
`TokenDataset::get_batch`, which samples a random example per row and returns only the tokens.

A model that reads inputs of its own per example (the images of a record, through
`Batch::LmExamples`) got a batch with no example named, so it read the placeholder tokens as
ordinary tokens. The report of the run therefore said what a model that had never been shown the
picture scores. Every number the fit loop printed came from the same loop, so nothing disagreed with
it.

## What to do

A trainer whose batches carry something beyond tokens takes every batch from the one path that names
its examples (`TokenDataset::get_batch_examples` returns the picks of the windows it samples). A
second way of drawing a batch is a second way of not feeding the model its inputs.

## Evidence

A test that trains the same record with two different pictures and compares the loss before any
step: the pictures decide it, or the batch did not carry them.
`train::tests::a_run_that_asks_names_the_example_in_every_row_of_every_batch` asserts that the probe
and the random held-out draw name the example in the row, and `vision_train_streamed.rs` asserts the
initial losses differ.
