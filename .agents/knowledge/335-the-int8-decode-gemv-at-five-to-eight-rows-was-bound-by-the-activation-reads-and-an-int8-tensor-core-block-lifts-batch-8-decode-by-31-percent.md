<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 335. The int8 decode GEMV at five to eight rows was bound by the activation reads; an int8 tensor-core block lifts batch-8 decode by 31 percent

Question: why did the native int8 decode GEMV (`matmul_i8_gemv`, 73 percent of a batch-8 27B decode step) stream the
weights at 1.1-1.9 TB/s at eight rows of x when it reaches 2.7-3.0 TB/s at one row and the card's HBM streams 3.8 TB/s
(lm head, one row)? Knowledge 331 ruled out the load/sync structure (pipelining the stages: +1-9 percent) and the
per-word f32 arithmetic (summing each 32-element group as an int32 first: +8-15 percent at four rows, nothing at eight).

Cause: from five rows of x the dp4a block reads every row's activation vector back from shared memory once per weight
vector, at as many bytes as the weights themselves (eight activation bytes per weight byte at eight rows); the shared
memory pipe, not HBM, sets the rate. A tensor core takes 16 weight rows against eight rows of x in one `mma.sync
m16n8k32` with the activation in registers. Block structure and the fold are in the header of `cu/matmul_i8_gemv.cu`:
256 threads cover 16 weight rows and split K over eight warps, each group's scale is folded per MMA (the construction of
the prefill GEMM), the warps' totals are added in a fixed order.

What it bought (device-timed, card 1, weights cycled past L2, best of rounds, GB/s of weights and scales,
`runs/b27/gemv-q7.log`, `gemv-q6.log`; dp4a before / tensor-core after): at eight rows ffn down 1868 / 1912, ffn gate+up
1543 / 2129, attn q 1448 / 2039, gdn qkv 1459 / 1857, attn out 1345 / 1585, ffn narrow 1113 / 1676; at five rows the same
pattern; at one row 3.0 TB/s on both (the tensor-core block is 20-25 percent slower there, which is why it starts at
five rows - `kernels_cuda::I8_GEMV_MMA_MIN_ROWS` - and at two rows the dp4a kernel stays at its old rate). The decode
profile (`qwen35_decode_profile`, 27B Q8_0, synthetic context 512, int8 KV, alternating old and new binaries on the same
card, `runs/b27/decode-mma2.log`, `decode-mma3.log`), wall ms per step:

| rows | before | after | aggregate tok/s before / after |
|---:|---:|---:|---:|
| 1 | 13.7 | 13.4 | 73 / 74 |
| 2 | 15.4 | 14.7 | 130 / 136 |
| 4 | 19.1 | 18.0 | 210 / 222 |
| 5 | 24.3 | 19.7 | 206 / 253 |
| 6 | 24.8 | 18.9 | 242 / 318 |
| 8 | 26.0 | 19.9 | 308 / 402 |

Batch 16 (and 9-16 rows generally) still takes the tiled int8 GEMM (`matmul_i8_mma_m16`), whose sweep tops out at
1.5-2.1 TB/s of weights on the same shapes (`runs/b27/sweep-decode1.log`): no tile configuration of that kernel reaches
what the new block reaches at eight rows. Extending the new block past eight rows (it would re-read the weights per tile
of eight rows) was not measured.

Numerics. The tensor-core rows are not bit-identical to the WGSL tier or to the dp4a block: the scale is applied once
per 32 products with a fused multiply-add instead of per word. At the kernel (`i8_gemv_native`, random data, every
row count to 32) the difference is within 2e-5 of the output rms. Through the real 27B, eight streams decoded together
against the same streams decoded alone (`qwen35_gguf_logit_check --batched 8`, teacher-forced greedy streams of 24 steps
over eight 4000-token prompts, `runs/b27/logit-batched3.log`): with the dp4a block (this build with the tensor-core block
withheld) mean rel-L2 8.47e-2 = 0.96x the numerics floor 8.85e-2, mean KL 1.4e-2 to 3.4e-2, argmax flips at near-ties
and at margins up to 0.9; with the tensor-core block (`runs/b27/window11.log`, same prompts and streams) mean rel-L2 8.79e-2 = 0.99x the floor, mean KL
1.1e-2 to 3.2e-2, argmax flips at near-ties and at margins up to 0.98 (stream 5, step 7) - the same picture as the dp4a
baseline, a few percent apart per stream and inside the run-to-run spread of the settings in knowledge 328 (bound 3x).

Consequence: batched decode (separate sequences in one step) was never bit-identical to single-sequence decode (the baseline above is already at
the floor), and from five rows it now uses the tensor-core block. A SINGLE sequence's exact round (a speculative verify pass,
`Qwen35::run_exact_round`) must still equal its decode steps to the last bit (`tests/chunk_tape_decode_parity.rs`), so it runs in
groups of at most `I8_GEMV_MMA_MIN_ROWS - 1` rows (four) on a device with the tensor-core block: a 5-8 row round costs two weight passes
instead of one, and stays exact. Replies change under any numerics change of this size (knowledge 328); gate a change like this
against the floor, not against identity, except where a documented contract (lossless speculation) needs identity.

Trap: the plain gate binaries pick GPU 0 unless `BRAIN_DEVICE=gpuN` is set, and the reservation helper does not pin a
device (knowledge 331). A bench on a card another job is using reads a fraction of the real rate.
