<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 453. An overlapped pipeline step divided each micro-batch by its own token count, so a short record outweighed a long one

**Lesson:** a normalisation decided in one accumulation path is not decided for a sibling path that
bypasses it. The shared loop gave every micro-batch of a step one denominator, the step's mean
supervised count, but only on the path that runs the micro-batches in turn; the path a pipeline takes
to overlap them across its stages (`Model::accumulate_overlapped`) received the bare batches and
divided each by its own count. A test whose records all supervise the same number of positions
cannot tell the two apart.

## What happened

`CausalLm::micro_steps` drew the step's batches, then either handed them to
`accumulate_overlapped` (a `PipelineModel` with two or more stages and more than one micro-batch) or
ran them one by one with `set_loss_denominator(mean_supervised)` after each upload. The overlapped
branch returned before the denominator was computed, and `Pipeline::pipelined_fwd_bwd` set each
micro-batch on the head stage with its own count. A pipelined run with gradient accumulation
therefore descended the mean of the micro-batches' means: with records supervising 1 and 5
positions, the short record's token weighed three times each long one's, and the gradient was off
the per-token mean of the step by more than its own size. The existing parity test
(`lora_pipeline_overlap.rs`) compared the two paths on a corpus whose windows all held the same
number of supervised positions.

## Fix

`Model::accumulate_overlapped` takes the step's loss denominator, and the pipeline schedule sets it
on the head stage after each `set_batch`, the recomputing one in the backward phase included
(`Pipeline::pipelined_fwd_bwd_over`). The loop computes the denominator once and gives it to both
paths.

## Gate

`crates/qwen3/tests/it/accumulation_normalisation.rs` holds both paths, on the CPU backend, to one
forward of both records as the rows of one batch (whose cross-entropy is by definition the mean over
all its supervised positions), with records of 1 and 5 supervised positions.
