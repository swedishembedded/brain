<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 538. One Qwen3-4B instance with a wide batch beats replicas on one card, and the agent run was bound by sandbox setup

**Lesson:** on one GH200, Qwen3-4B serves agent turns fastest as ONE instance with `--max-batch 256`. Two instances
on the same card (each half the memory) were 19% slower in aggregate and 22% slower per request: they time-slice the
same SMs, and a single instance at batch 64 already keeps the card about 90% busy. The default `--max-batch` (64, the
top of `serve_defaults::MAX_BATCH_DEFAULTS`) leaves 40% of the 4B's throughput on the table under 256 clients. Above
256 requests in flight the HTTP edge sheds with 503 (`residency::admission::EDGE_CONCURRENCY` = 256, a constant), so a
`--max-batch` above 256 cannot be filled over HTTP.

## Measured (2026-11-21, gpu0, brain-5e25d5b82, int8 KV, bf16 weights)

Closed-loop clients, each a 6-turn conversation: a shared 1.8k-token system prompt, a unique task, a 90-word "tool
result" appended per turn; mean prompt 2.26k tokens, completion `max_tokens` 1024 (mean 0.86k, Qwen3 thinking),
T 1.0 / top-p 0.95 / top-k 20, non-streamed. Tokens/s is generated tokens over wall time; GPU busy is the mean of
`nvidia-smi` utilisation samples.

| serve | clients | gen tok/s | p50 s | p95 s | GPU busy | errors |
|---|---:|---:|---:|---:|---:|---:|
| 1 x b64 (512 max tokens) | 64 | 5413 | 6.0 | 6.3 | 89% | 0 |
| 1 x b64 (512 max tokens) | 128 | 4906 | 12.3 | 14.9 | 90% | 0 |
| 1 x b128 | 64 | 4813 | 11.5 | 12.9 | 92% | 0 |
| 1 x b128 | 128 | 6278 | 17.8 | 21.8 | 93% | 0 |
| **1 x b256** | **256** | **7585** | **30.0** | **35.0** | 92% | 0 |
| 1 x b512 | 512 | 7568 | 28.7 | 33.9 | 93% | 768 x 503 |
| 2 x b128 (`--reserve-gb` split) | 256 | 6166 | 36.5 | 41.2 | 99% | 0 |

The planner sized the pool alike at every batch (74.3-74.8 GiB int8 KV, about 1M tokens, capped by
`MAX_DEFAULT_POOL_TOKENS`), so KV never limited the 4B on this card; activations grew 0.47 -> 1.70 GiB from b64 to
b512. Two processes on one card are the only replica form for Qwen3 today (`ResidentModel::replicas` is 1 for
`QwenResident`; the Qwen3.8 GGUF resident places replicas one per card group, never two per card).

## The agent run (a Vim-editing benchmark through an agent runtime, 168 tasks x 5 + 14 x 5, 160-224 trials at once)

With b256, the 4B served 4278 requests (the 16-task smoke and the three runs), 2.3k generated tok/s over the 25-minute 4-attempt run, about a third of
what the server sustains: a trial's median environment setup was 174-251 s (each sandbox unpacks a 300 MB image
archive into memory, 224 at once) against a median agent phase of 67-71 s. The card was the bottleneck in no phase of
the run; the sandbox start was. Chat-scan anomalies: 22 `discarded_tool_call_tail`, 17 `ended_inside_think`, 9
`truncated_tool_call`, 1 `tool_call_rescued_from_think` (0.02% of requests, against 1.3% in the accounting runs of
536).

## What to do

- Serve a small model to many agents as one instance with `--max-batch 256` and keep clients at or below 256.
- KV spill to host (`--kv-offload-gb`, over C2C on a GH200) would not help this load: the int8 pool already holds
  about 1M tokens, several times what 256 agent conversations need. It is for preempted long contexts.
- Fixed after this measurement: an unset `--max-batch` now starts from the cards' decode operating point
  (`serve_defaults::decode_operating_point`: `bytes_per_weight * tensor_rate / (2 * memory_bandwidth)`, nearest power of
  two; 256 on a GH200 at bf16, int8 and fp8 weights) and halves until the plan fits; a card without published rates
  keeps the fixed 64..1 ranking. Wired for the Qwen3 resident; the Qwen3.8 and MoE residents still rank 64..1 (an MoE
  reads every expert but multiplies only the active ones, so its point needs its own ratio). The edge ceiling is now
  `admission::edge_concurrency`: twice the served models' decode slots, at least 256, reported at startup, so requests
  past the batch queue for a slot instead of being shed. Re-measure on GPU pending.
