<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 443. The trainer's own GRPO sampler costs 254 s a group where the serving path costs 1.7 s, and sampled decode is launch and top-k bound

**Lesson:** Sampling a GRPO group with the model being trained (one whole-sequence `logits_all` per token) is two orders
of magnitude slower than submitting the same group to the serving scheduler, and the serving path itself is far from the
card's ceiling for sampled decode: the top-k extraction is 64 rounds of three launches, and the card is busy under a third
of the time.

## What happened

`qwen_bench rollout --weights Qwen/Qwen3-4B` (2026-11-13, one GH200, bf16 base, the card to itself), one group of 8 samples
of a 256-token prompt, 128 new tokens each, temperature 1.0, no stop token, same prompts and sampler on both paths:

| path | wall per group | generated tok/s | prompt tokens from the prefix cache |
|---|---:|---:|---:|
| `ModelRollout` over the rank-16 LoRA training model | 254.4 s | 4.0 | 0 of 2048 |
| `SchedulerRollout` over the paged `Engine` | 1.66 s | 615 | 1680 of 2048 (82 %) |
| the same, greedy | 1.02 s | 1005 | 1680 of 2048 |

The group shares its prompt through the scheduler: the first member prefills and each of the other seven adopts 240 of
its 256 prompt tokens (the last block is always recomputed).

## Why

- The model's own rollout pays a forward over the whole sequence per token, reads every position's logits back to the
  host (about 180 MB at T = 300 for the 152k vocabulary) and sorts the vocabulary on the host for the sampler.
- Under `nsys` (`--trace=cuda`, an 8.3 s window of the serving path at temperature 1.0) the card was busy 30.8 % of the
  window across 276,356 kernel launches. The largest share, 31 % of kernel time, is `argmax_part` + `argmax_final`
  (17,600 launches each, about 68 per decode step): `submit_topk_head` extracts the top 64 candidates by 64 rounds of
  (argmax, mask), 192 launches a step, whatever the sampler needs. Greedy decoding skips it and is 63 % faster.
- A single-pass device top-k kernel (one launch per row, selected like every other op by capability and shape) would
  take most of that back; it is not built.

## What to do

Sample groups through `SchedulerRollout` (the SDK's improve cycle does by default, `ImproveOptions::model_rollouts`
selects the old sampler as an oracle); a LoRA refit publishes the adapters to the engine's bank and does not rebuild it.
Open: the fused top-k kernel; sampled decode is the dominant cost of a served rollout.

Update 2026-11-13 (lane rollout): the single-launch top-k kernel this entry called unbuilt existed
(`model::readout::topk_rows_steps`, the native `topk_rows_f32` on CUDA devices) and the 27B's engine used it; the Qwen3
engine kept its own copy of the 64-round chain. It now records the shared steps, and on Qwen3-4B (same group, bf16, one
GH200) sampled decode goes from 615 to 729 tok/s (greedy 1005 -> 1075, run to run); the sampled path is still 32 %
below greedy, which is the candidates' read-back and the host sampler, not the extraction.

