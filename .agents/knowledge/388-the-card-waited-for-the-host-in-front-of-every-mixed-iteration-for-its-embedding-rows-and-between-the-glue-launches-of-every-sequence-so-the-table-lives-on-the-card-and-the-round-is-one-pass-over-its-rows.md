<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 388. The card waited for the host in front of every mixed iteration for its embedding rows and between the glue launches of every sequence, so the table lives on the card and the round is one pass over its rows

Question: knowledge 385-386 left a step at 87-92 % card busy (c = 16 / 64), with the host in front of the first kernel of every step and a
second graph behind a read-back of the last layer. Where does the host time of a decode step and a mixed iteration go now, and which
parts can be taken off the critical path without changing a value?

Measured (gpu1, FP8 Qwen3.8-27B, `--max-batch 64 --kv-cache int8 --speculative off`, `vllm bench serve` random 1K in / 128 out, nsys
`-t cuda,nvtx --cuda-graph-trace=node`, 192 requests at c = 64 and 48 at c = 16, NVTX ranges `mixed.embed`, `decode.embed`, `serve.sample`,
`serve.accept`; card busy = union of kernel intervals over the steady middle of the run, spinning PDL kernels counted as busy):

1. **The embedding rows were the largest host cost of a mixed iteration, and it grows with the prompt rows.** `ResidentCheckpoint::embed_rows`
   decoded the ~900 rows of a c = 64 iteration on the host (median 15.6 ms, 10-30 ms across iterations with random tokens; 2.7 ms when warm in
   knowledge 385) and uploaded them as 18 MB of fp32 before the first kernel. A decode step paid 1.1 ms at 64 rows. The HF release stores the
   table as plain bf16, which widens exactly, so the first stage now holds it (`Shard::embed`, tier from `ResidentCheckpoint::embed_tier`, 2.5 GB
   charged by `serving_cost` and refused when it is more than 1/16 of the smallest budgeted card) and the step uploads token ids. A GGUF (quantized
   table) or any scaled encoding keeps the host path. `decode.embed` 1114 us -> 1 us, `mixed.embed` 15.6 ms -> 1 us; the gap between the previous
   step's top-k and a decode step's first kernel at c = 64 3056 -> 722 us; card busy over the steady window 88.1 -> 92.2 %, busy inside mixed
   iterations 90 -> 95 %.
2. **The GQA glue of a mixed round was per sequence.** Per full-attention layer, the decode rows and each prompt chunk ran three `concat_split`
   row copies, the 6-launch split/norm/rope chain, the two K/V appends, the sigmoid, the product and a splice back: ~175 launches a layer, 6.6 ms of
   the 109 ms iteration at c = 64. Every one of them works a row at a time, so a row's result does not depend on which sequence it belongs to. The
   round now builds its per-row coordinates once (append block and offset, M-RoPE row; decode rows first) and runs the chain and the append over all
   rows (`qkv_prepare`, `kv_append_rows`); only the attention stays per sequence (`gqa_chunk_attend_steps_kv_paged`, the decode rows' attention
   reads the head of the round's buffers and writes the head of the context in place, a lone sequence needs no copy). The sigmoid gate rides on the
   `o_proj` quantiser (`quant_epilogue` mode 2) as in a decode step. Launches of the median mixed iteration 2625 -> 1972; `concat_split` 2.2 -> 1.1 ms,
   `sigmoid` + `mul` + `splice` 2.8 -> 0.8 ms (the 18 gated quantiser launches cost 0.7 ms more), `rmsnorm_rows` + `rope` + `kv_append` 1.7 -> 0.9 ms; the
   median iteration at c = 64 (nsys) 109.5 -> 92.8 ms of span with the row count the latency-target fit chose. Bit-identical: six
   greedy chats are byte-equal to the base.
3. **The head was a second graph behind a synchronising read.** The last layer's residual was read to the host, uploaded and normed in a pass
   of its own: 103-120 us of idle card in every token (knowledge 386 item 4). The step keeps the block as a device buffer and holds layers and head in
   one pass (`Qwen35GgufInstance::step_scope`); only the top-k candidates are read. Decode step median at c = 16 16.14 -> 15.85 ms, at c = 1
   11.80 -> 11.62 ms; idle gaps above 20 us inside a c = 1 step 733 -> 412 us. The drafter (speculation on) still reads the rows, so it keeps the old order.

Served (same client, same session, base run first, output tok/s at c = 1 / 4 / 16 / 64): base 78.5 / 297.5 / 553.4 / 804.4, + device embedding
78.0 / 298.6 / 562.8 / 837.0, + round-wide GQA 78.4 / 299.5 / 569.5 / 856.7, + device head 78.5 / 299.4 / 568.8 / 850.9 (the head's gain is below the
+-1 % the shared node adds); 31K TTFT 4.46 -> 4.08 s; median ITL at c = 1 11.54 -> 11.48 ms. Training: the NF4 W8K probe (`--layers 64 --tokens 8192
--rank 16 --seed 7 --steps 4 --residuals bf16 --base nf4 --gdn-segment 2048 --mlp-rows 2048`, gpu1, warm steps 2-4, 2 runs each) 1051 / 1037 base,
1042 / 1048 now, losses equal to the fourth digit: none of these paths is a training path.

What is left (nsys, c = 64, steady window): card busy 92.2 %, and what the 7.8 % is.
- **Decode step**: 473 us from the top-k read to the first kernel (sampling + accept 100 us, ten parameter uploads, 64 steps resolved before the
  first graph, a 80-90 us `cuGraphLaunch`), 40-60 us of PDL ramp between layers. The host rebuilds the step's ~2500 dispatches every token (hidden behind the
  first graph); the dispatch list is the same per (rows, context bucket), so the recording seam of `decode_tape` (today a sliding-window config only) is
  the way to take that out, and a sampler on the device (greedy first) is what lets the next step be launched before the host has seen this one's token.
- **Mixed iteration**: 2 ms in `begin_slices` per iteration that admits a prompt (96 `cuMemsetD8Async` + 48 `pool_rows_scatter2` to write a zero state to the
  new sequence's pool row; the GDN kernels could read zero for a `fresh` segment instead), 0.75 ms tail (the per-span reads of the decode rows and each
  prompt's last row, then the head's upload) and ~30 `put_rows_copy`/`take_rows` pairs a layer for the prompt segments, which the prefill attention kernels
  could read and write in place with a row offset. Kernel time is 96 ms of the 93 ms span: the iteration is now kernel bound (FP8 GEMM 55-64 ms, GDN 20 ms,
  quantisers 12 ms, attention 5 ms).
- Training takes no recorded replay (`graph_replays=0` in the probe; 6.6 s of the 7.8 s step is host time, `wait_s` 5.2 s).
