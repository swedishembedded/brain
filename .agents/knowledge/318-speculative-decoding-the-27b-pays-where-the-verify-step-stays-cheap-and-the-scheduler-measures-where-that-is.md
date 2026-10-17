<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 318. Speculative decoding on the 27B pays at small batches, and only as far as the verify step stays cheap; the scheduler measures where that is

Question: add speculative decoding to the ONE continuous-batching scheduler
(`model::serve::Scheduler`), drive it from the released Q8_0 GGUF served INT8 on one GH200 (CUDA,
bf16 KV, ctx 1024), with the checkpoint's own MTP head as drafter, and measure what it buys. Every
figure is **VERIFIED-BY-RUN** with `qwen35_spec_sweep`, on a card shared with other lanes (GPU lock
held, `other_contexts_start=0` each run) and a host at load average 18-37, which stretches the
host-bound parts; treat differences under about 10 percent as noise. Greedy, 128 new tokens.

## Aggregate decode, tokens/s (first run of the final scheduler, scratch 96 rows, host load 18-37)

| batch | code: plain | n-gram | MTP | agent turns: plain | n-gram | MTP |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 76.3 | 63.9 | 92.7 (112.9 on a quiet host) | 67.3 | 69.4 | 116.3 |
| 8 | 309.6 | 224.0 | 260.8 | 195.8 | 172.2 | 290.2 |
| 16 | 571.9 | 402.7 | 436.3 | 317.3 | 268.1 | 485.3 |
| 32 | 842.5 | 625.0 | 884.7 | 472.8 | 439.1 | 716.6 |
| 64 | 1094.2 | 853.6 | 1058.5 | 602.3 | 595.5 | 875.5 |
| 128 | 1634.4 | - | - | 987.2 | - | - |

Acceptance (accepted proposals per proposal, MTP): code 0.72 at batch 1, 0.69 at 8 and 16, 0.88-0.93 at
32-64 (the window shrinks, only confident proposals are made); agent turns 0.88-0.94. Tokens per
sequence-round: 4.6 (code, batch 1), 6.3 (agent, batch 1). Batch 128 is plain: the verify scratch is
sized for 96 rows. These rows were taken before the cost gate (below): speculation was on at every
batch size, which is why code at batches 8 and 16 is below plain. The gate keeps such buckets on the
plain step; it was verified on a virtual clock (`tests/serve_speculative.rs`) and not re-swept on the
card because the GPU lock was held by a long-running server.

## Why the crossover is where it is: what a verify step costs

`qwen35_spec_sweep --profile-verify` (batch 1): a plain step is 12.9 ms; verifying 1, 2, 3, 4, 5, 6, 8
rows costs 14.0, 17.8, 21.6, 22.7, 27.5, 28.1, 29.7 ms. The extra rows are not free here: the int8 GEMV
(`matmul_i8_gemv`, dp4a, up to 8 rows per weight pass) is compute-bound past two rows and grows by
1.3-1.9 ms a row, the serial per-row recurrence in `gdn_verify_pool` adds 0.55 ms a row, and the
snapshot restore (`gdn_commit_pool`) 1 ms before it was vectorised. The tensor-core GEMMs that take over
at nine rows stream the weights at 1.6 TB/s (17.8 ms at 16 rows against 9.7 ms for the single-row GEMV
at 3 TB/s). The MTP head costs about 2.3 ms a proposal at batch 1 and 8 ms at batch 8 (the shared head
GEMV is compute-bound in the rows). So a round of `k` proposals costs about 14 + 5.8 k ms at batch 1 and
buys `1 + accepted` tokens: ahead of plain from acceptance 0.5, well ahead at 0.9 (agent turns, tool
call arguments copy the schema) and level at 0.7 with wide windows. A small-M tensor-core int8 kernel
that streams at bandwidth for 2-16 rows would flatten that curve and is the lever that matters most;
it is not done.

## Correctness

Greedy output with speculation equals plain decode token for token when the batch composition does not
change the arithmetic: 32 real prompts (16 code, 16 tool-call turns, 96 tokens) each decoded alone,
speculative against plain: 0 differ, MTP and n-gram. The engine's own plain decode is not invariant to
batch composition (the fused split-key attention picks its split count from the batch, the GEMV and
tensor-core tiers round differently): the same requests decoded as a batch of 16 against decoded alone
differ in 19 of 32. Speculative against plain of the same batch differs in 5 of 32 at batch 16, 20-22 of
58 over batches 1-16, the same order as that floor. Rejection sampling for temperature > 0 leaves the
distribution unchanged (closed form and end to end on the toy decoder). DFlash2 is not wired: its draft
checkpoint is not on disk, and the drafter runs only on the single-stream path.

## Decisions

Speculation is a scheduler feature (`PagedDecoder` hooks with "no speculation" defaults), not a model
one. The verify launch keeps the Gated DeltaNet state after every row but the last, so a rejection costs
one copy, not a second forward. `--speculative auto` loads the MTP head when the checkpoint carries it.
The checkpoint's head was trained on the normed target hidden state (0.72 against 0.62 acceptance on
code with the raw residual). The scheduler times plain and speculative steps per batch-size bucket and
speculates where that produces more tokens a second (decoders opt in with `spec_cost_aware`).
