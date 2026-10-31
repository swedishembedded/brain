<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 395. The FP8 release drafts with its own MTP head, but a draft round costs more than it earns below 70 % acceptance, so the gain sits in the round's price and not in the head

Question: `Qwen/Qwen3.8-27B-FP8` ships `mtp.*` tensors. Can the paged engine speculate with them (exactly), and does that beat the plain scoreboard
(c = 1 / 4 / 16 / 64 80.6 / 251 / 561 / 838 tok/s) and vLLM + MTP(2) (124 / 310 / 639 at 1 / 4 / 16)?

## What landed (all exact)

- The drafter's source is the resident checkpoint: an HF directory's `mtp.safetensors` opens as a one-layer stage (FP8 layer held as E4M3 with its
  128 x 128 scales, bf16 `mtp.fc` split by the shared `Weight` upload, the three norms with the same `(1+w)` fold). Greedy output under speculation equals
  plain decoding (`hf_fp8_speculative`: 40 of 40 / 20 of 20 tokens, rows within 5e-2 of the separate verify pass); sampled requests use the existing
  rejection test against the drafter's point mass (`rejection_sampling_preserves_the_target_distribution`).
- A mixed iteration's decode rows verify drafts in the same forward (`forward_mixed_verify`: the Gated DeltaNet verify kernel on the leading rows, the
  chunked kernels on the segments), and a verify round of equally wide sequences is recorded and replayed as one graph (padded to an even row count).
- The drafter read each prompt token's embedding from the mapped checkpoint: one cold page of a network file per token, 50+ ms a prompt, which made the
  prefill gate stop preparing the drafter. A host copy of the bf16 table (2.5 GiB, load 4.7 -> 5.4 s) took acceptance on random prompts 0.33 -> 0.96 in the sweep.
- The cost gate ruled speculation out on its first (cold-launch) step and then for 384 steps; windows were `mean accepted + 2`. Both replaced (warm-up, outlier
  clipping, windows from the measured price of a proposal against the worth of a token).

## What it measures (same client, base gpu0 / new gpu1 at once, 1 / 4 / 16 / 64 output tok/s)

| | c=1 | c=4 | c=16 | c=64 | 31K TTFT |
|---|---|---|---|---|---|
| `--speculative off` | 80.6 | 250.6 | 561.1 | 838.1 | 3.50 s |
| `--speculative auto` (this change) | 75.3 | 235.0 | 475.4 | 684.1 | 3.61 s |

Greedy (temperature 0, as the client sends) acceptance on its random prompts is 0.43-0.54 per proposal, 1.9-2.1 tokens a round. Per round at c = 1 (nsys):
verify 15.4 ms replayed (the plain step is 11.6; each extra row adds ~0.65 ms of serial Gated DeltaNet recurrence), draft 3-7 ms (host prep 0.6-1.2 ms, a
projection, the layer, the shared head, a readback each, per chained proposal). 15.4 + 4.5 ms for 2.0 tokens is 9.9 ms a token against 12.9: at best +20 %, and
the bookkeeping a speculating engine does on the steps it does not speculate (hidden rows to the host, catch-up feeds every 16 steps, no recorded plain step)
costs that back. At c = 16 / 64 the plain step is already in the GEMM regime, so 3 rows a sequence cost 30 ms against 17.6 and the gate rightly declines; the
mixed iterations carry ~40 % of the wall time and their decode rows are 16 of ~800.

## Decision

`--speculative mtp|auto` serves the FP8 release's head and is exact; the default for an HF directory stays off until a round is cheaper. The levers, by weight:
a device-resident drafter (embedding gather, norms, projection, layer and head on the card, one readback per round; today four host round trips per chained
proposal), a recorded plain step that also leaves its hidden row on the card, and the prompt feed from the forward's own output rows. Targets not met: c=1 125,
c=4 320, c=16 760, c=64 836.
