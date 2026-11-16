<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 495. A hybrid model's KV pool reserved a whole context per sequence because its recurrent state was keyed by its KV block

**Lesson:** a hybrid decoder keeps two kinds of per-sequence state, KV rows that grow with the context and a recurrent
row that does not. Keying the recurrent row by the sequence's first KV block forces one block per sequence, so the pool
must be `max_concurrent * max_seq_len` rows and the batch is bought with context. Give the recurrent row an identity of
its own (`BlockTable::state_row`, a free list) and the KV can be paged through the shared block tables with no change
to any number. Then audit what else was sized by the context: a step sized for the engine's whole context holds split
records for that context in every attention layer, and a plan that lets one sequence take a 256K context must charge
them, or the card runs out on the first wide batch.

## What happened

The 35B-A3B engine (`qwen35moe::serve`) used `table.blocks()[0]` as the row of its Gated-DeltaNet state pool, which is
stable only when `block_size == max_seq_len`. Its planner (`catalog::resident_qwen35moe::plan`) therefore charged every
one of `max_batch` sequences a whole context: at fp32 KV on a GH200 it chose 32 sequences of 40704 tokens, because 64
left a 19200-token context under the 32K floor.

The engine now takes blocks of `model::paged::KV_BLOCK_ROWS` from one pool sized by the plan
(`EngineOptions::kv_pool_rows`), gives each sequence a recurrent row from a free list, and reaches its history through
`[batch, max_bt]` decode tables and `[n, max_bt]` prefill tables (`model::block::gqa_chunk_step_kv_paged`, the dense
27B's path). Tokens are bit-identical to one whole-context block on the real checkpoint
(`real_decode_parity::a_sequence_paged_over_blocks_decodes_to_the_bit_what_one_block_gives`).

The first plan of the paged engine let one sequence take the checkpoint's whole 262144-token window and gave the rest
of the card to the pool. The first fp32-KV serve run at concurrency 64 ran out of device memory over two hundred times and
refused 87 of 96 requests: a recorded decode step sized its attention's split count by `max_seq_len`, so at 64 rows every
attention layer held 288 MiB of split records the activation charge, calibrated at a 74K context, did not cover.

## What to do

- Per-sequence state that does not grow with the context gets its own row identity (`BlockTable::state_row`), never a
  KV block's.
- A charge for a decode step includes the split records at the context the step is sized for
  (`model::kv_tier::flash_part_words_bound`, any device and kernel); `qwen35moe::serve::decode_step_bytes` charges it.
- Size a recorded step's split count by its batch's longest live sequence, rounded to a power of two
  (`Engine::decode_cap`, part of the tape key), not by the engine's context: the work and the records follow what the
  batch holds.
- Still open: the recurrent rows are reserved for every one of `max_batch` sequences (64 rows of 62.8 MiB each on the
  35B-A3B) whether they run or not; the dense engine lends them from the KV pool (`qwen35::int8_gguf_resident`'s
  unified pool).

## Evidence

gpu3 (GH200), Qwen3.6-35B-A3B Q8_0, `brain perf run serve --concurrency 64 --input 2048,8192,32768 --output 128
--requests 96 --warmup 4 --seed 4242`, every other setting the server's default, base/new interleaved:

| KV | engine | sequences at once | completed | req/s | TTFT p50 / p95 ms | inter-token p50 ms |
|---|---|---|---|---|---|---|
| fp32 | whole context each (32 x 40704) | 32 | 96 / 96 | 0.712 / 0.713 | 44872 / 76122, 45035 / 75974 | 30.6 |
| fp32 | paged, 1164288-token pool, 64 rows | 64 | 96 / 96 | 0.723 / 0.724 | 23416 / 73564, 23477 / 73427 | 43.5 / 43.9 |
| int8 | whole context each (64 x 76032) | 64 | 96 | 0.748 | 23206 / 72492 | 26.6 |
| int8 | paged, 4292096-token pool, 64 rows | 64 | 96 | 0.749 | 23163 / 72394 | 26.5 |

The run is prefill-bound (about 10K prompt tokens a second either way), so twice the sequences halve the median TTFT
and leave the rate; the wider decode step costs each token more. `perf run longctx --context 32768 --ladder 1,512
--steps 2 --prefill 4096,16384` (fp32 KV) is unchanged: 40 sequences admitted at 32K, batch-1 decode 5.35 / 5.22 ms
base against 5.18 / 5.21 ms new, prefill 12.1-12.8K tok/s both.
