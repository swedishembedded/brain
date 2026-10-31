<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 396. A bf16 recurrent state halves the GDN decode bytes, and a streaming kernel reaches the roof only when the state arrives by bulk copy, not by per-thread loads

Question: the 27B's Gated DeltaNet decode costs 154 us a layer at 64 sequences (conv 14 + recurrence 140) against 79.5 us for the reference server's.
What does the arithmetic say?

Arithmetic. The state is 48 value heads x 128 x 128 per sequence: 3.1 MB in f32. At 64 sequences read plus write is 403 MB, 122 us at 3.3 TB/s; the
f32 recurrence kernel (140 us) is already at 86 % of that roof. A 79.5 us layer cannot move f32 state both ways: it needs the state stored in two
bytes (201 MB, 61 us). So the 2x is a storage tier (`--gdn-state bf16`, default f32), and the kernel work is reaching 61 us of traffic.

What the kernel (`gdn_decode_pool_stream_bf16`, one launch with the conv and gates folded in) taught, GH200, us a layer at 1 / 4 / 16 / 64 sequences
(bf16 pool, previous conv + recurrence pair: 15.9 / 18.7 / 33.9 / 112.9):
- 128 state rows in registers, 4-byte loads a thread: 14.5 / 15 / 44 / 162. A multiprocessor's load queues bound the bytes in flight, not its registers.
- the head's 32 KiB per value head requested by ONE bulk-copy instruction each into shared memory, a thread owning two columns over all rows, fully
  unrolled shared-memory walk: spilled (168-register cap, 8 M local requests), 364 us; unroll 2: 89.7 us at 64.
- the same with a thread owning two columns over HALF the rows (two row halves meeting in shared memory after each pass; 12 warps a block instead of
  6): 10.7 / 11.0 / 17.9 / 78.0. Issue slots were 78 % idle at 6 warps: the kernel is latency bound between phases, not instruction bound.

Not bit-identical to the exact bodies (f32 reassociation plus bf16 storage rounding); `the_wide_candidate_agrees_with_the_exact_body_within_a_stated_tolerance`
covers it, three value heads a key head included.

Hopper recurrence (`gdn_chunk_loop_wg`, prefill): holding the state TRANSPOSED as the fp32 accumulator of the update wgmma makes the registers that receive
an update the register A operand of the next chunk's products (and v_new^T that of the other two), so the state never visits shared memory: 0.604 ->
0.446 ms a layer at 8K tokens at one block a head (48 of 132 multiprocessors). With the operand staging switched off it runs 0.282 ms: the remainder is
the ~1 GB of scratch the separate prep writes and the recurrence re-reads (88 KiB a chunk-head), which only fusing the prep into the same kernel removes.
A third (producer) warpgroup is not free: register allocation in groups of four warps caps a 288- or 384-thread block at 168 registers, below the 188
this kernel needs; staging by the compute threads themselves keeps 256 threads and 255 registers.

Serving A/B (`scoreboard/run.sh`, gpu0 base f32 vs gpu1 `--gdn-state bf16`, same build): output tok/s c = 1 / 4 / 16 / 64 80.7 / 255.3 / 568.4 / 836.4 -> 80.7 / 260.1 /
575.3 / 862.2 (+3 % at 64: the decode layer saved ~35 us x 48 layers of a ~70 ms step); mean TTFT c64 1412 -> 1410 ms, median ITL 76.6 -> 75.8 ms, P99 ITL
114.6 -> 118.4 ms; 31K prompt TTFT 3.47 -> 3.61 s (unchanged path, run-to-run).

Quality gate: NOT passed, so the default stays f32. The teacher-forced `qwen35_fp8_parity` scores through the working copy and never reads the pool (identical
numbers for both tiers), so it cannot gate this tier. Greedy chats through the server (six prompts, 300 tokens, temperature 0): the f32 pool is bit-reproducible
across three runs (base spread zero), bf16 pools diverge in two of five non-empty chats (after 889 of 1067 and 78 of 669 characters). A gate that exercises
the pool (greedy chats and a pooled teacher-forced decode) and a bf16 state with stochastic or compensated rounding are the open work.
