<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 394. Halving the activation bytes between the FP8 products took 5 % off a mixed iteration, because the kernels that weigh most are bound by arithmetic and state, and a bf16 store from the accumulator must go through the TMA

Question (knowledge 393): the card is power limited, the serving activation stream is f32 end to end, and every memory-bound kernel moves
twice the bytes a bf16 stream would. What does a declared bf16 activation stream buy, kernel by kernel, and where does it stop?

## The stream, per row per layer (Qwen3.8-27B FP8, d 5120, ff 17408, conv 10240, value 6144; bytes, f32 -> after)

| tensor | written / read by | f32 | now |
|---|---|---|---|
| gate, up (MLP, every layer) | FP8 GEMM epilogue / SwiGLU pass | 2 x 69,632 each way | 2 x 34,816 (bf16 twin of the 128x256 GEMM, `--act-precision`) |
| SwiGLU product | SwiGLU pass / nobody without an adapter or backward | 69,632 | not stored |
| norm front end (2 a layer) | `add_rms_quant`: a, b, sum, xn, bf16 rows | 20,480 each + 10,240 | a, b, sum f32; `xn` only for the dense gates of a GDN layer; E4M3 rows (5,120 + 160) from the same pass |
| `quant_e4m3_128` launch after the front | reads xn, writes E4M3 | 20,480 + 5,120 | gone (fused) |
| GDN qkv, z (48 layers) | GEMM / conv, gated norm | 40,960 + 24,576 each way | 20,480 + 12,288 (bf16) |
| conv q,k,v; recurrence out | conv / prep; loop / gated norm | 40,960; 24,576 each way | 20,480; 12,288 |
| gated rows (GDN) | gated norm / output projection | 24,576 | not stored (the E4M3 rows are what is read) |
| residual `a`, `sum`, `b` (down, out products) | front end | 20,480 each | **still f32** |

## Built

- `gpu_core::ActPrecision` / `ActChoice` (`--act-precision auto|f32|bf16`, `model::policy`, resolved per card: bf16 from compute capability
  8.0, f32 elsewhere; a model declares support in `PolicySupport`). The qwen35 forward asks the ops for a bf16 product per layer and keeps f32
  for that layer when the device or shape has no twin, when an adapter or a backward reads the f32 rows, or when the SwiGLU/gated-norm
  consumer is not offered.
- `matmul_fp8_wgmma_tma_n256_bf16o`: the 128x256 GEMM with the same f32 sum rounded to bf16 (nearest even) in the epilogue, stored by the TMA
  (the byte-typed output map with 32 columns to a 64-byte box). Bit-identical to the f32 kernel's rows rounded to bf16 (ragged M and N tested).
- `silu_mul_quant_e4m3_128` reads bf16 gate/up and skips the product; `gdn_conv_silu_seg`, `gdn_chunk_prep_seg_bf16`,
  `gdn_chunk_loop_seg_bf16`, `gdn_gated_norm` read and write bf16 where the chain is bf16 (the leading decode rows of a mixed round are widened
  into a small f32 buffer by `cast_bf16_f32`); `add_rms_quant_e4m3`, a wide-row front end that writes the E4M3 rows and scales itself.

## Measured (one GH200 pair, `scoreboard/run.sh`, base and new on both cards, averages)

Output tok/s at concurrency 1 / 4 / 16 / 64 (1K prompt, 128 tokens), median TTFT at 1K / 31K, median TPOT and ITL at 64, `scoreboard/run.sh`, each
binary on both cards at the same time (a card runs 0-2 % apart), mean of the two:

| | c=1 | c=4 | c=16 | c=64 | TTFT c=64 | TPOT c=64 | ITL c=64 | 31K TTFT |
|---|---|---|---|---|---|---|---|---|
| base (`5acd90774`) | 80.1 | 251.7 | 553.2 | 794.5 | 459 ms | 74.5 ms | 82.0 ms | 3.71 s |
| `--act-precision bf16` + fused front end | 80.8 | 252.7 | 562.5 | 835.9 | 438 ms | 70.9 ms | 77.6 ms | 3.53 s |
| per card (gpu0 / gpu1) c=64 | 793.4 / 795.6 | | | 848.8 / 823.1 | | | | |

nsys of the c=64 sweep (192 prompts, two runs each, kernel time): 25.12 -> 24.13 s (-3.9 %); a mixed iteration 84.8 -> 80.2 ms (-5.4 %); a
decode-only step 25.3 -> 23.5 ms. Per kernel (us a launch, base -> now; launches in brackets):

| kernel | base | now |
|---|---|---|
| `matmul_fp8_wgmma_tma_n256` (all FP8 prefill GEMMs, 12.9 s) | 121.3 (106.7 k) | 106.3 (41.7 k) + bf16-output 123.4 (65.3 k), 12.5 s |
| `silu_mul_quant_e4m3_128` | 40.4 | 27.7 |
| `add_rms_quant` + `quant_e4m3_128` -> `add_rms_quant_e4m3` | 21.3 (47.8 k) + 5.8 (67.7 k) = 1,412 ms | 20.8 (45.3 k) + 3.9 (22.6 k) = 1,030 ms |
| `gdn_chunk_loop_seg` / `prep_seg` / `conv_silu_seg` / `gated_norm` | 65.7 / 70.6 / 42.1 / 17.3 | 62.1 / 68.7 / 41.6 / 14.9 |
| `gdn_recur_pool` (decode rows' state) | 109.8 | 108.9 |

Targets: c=64 >= 950 and c=16 >= 650 are not reached (836 and 563).

## What the bytes did not buy

1. **A bf16 store straight from the accumulator layout is slower than the f32 TMA store.** Four-byte stores from the m64n256 fragment write
   half sectors: +16 to +29 % on the GEMM at 2048 and 8192 rows (and +8 to +25 % on the 128x128 kernel's direct stores, which got no twin). The
   twin that stages the bf16 rows in shared memory and stores 64-byte boxes through the same TMA map is 1.5-5 % *faster* than the f32 kernel at
   640-2048 rows (gate/up 117 -> 112 us at 640 rows, 151 -> 145 at 768, 334 -> 318 at 2048), the half of the epilogue the card no longer
   writes. The 8192-row down GEMM measures 9 % slower and is within the spread of that card's clock.
2. **The Gated DeltaNet kernels are not bound by their activations.** Their bf16 forms moved `gdn_chunk_loop_seg` 65.7 -> 62.1 us,
   `gdn_chunk_prep_seg` 70.6 -> 68.7, `gdn_conv_silu_seg` 42.1 -> 41.6, `gdn_gated_norm` 17.3 -> 14.9 although they move half the bytes: they are
   bound by the tensor-core chain per chunk, the serial recurrence and the state, not by the stream. `gdn_recur_pool` (the decode rows'
   recurrent step, 7.7 % of the kernel time) moves 400 MB of f32 state per layer: the state, not the activations, is its byte count.
3. **The fused front end is not at the memory roof.** `add_rms_quant_e4m3` takes 20.8 us on 768 rows for 51 MB (2.5 TB/s of 3.4): one block a
   row reads, reduces and writes in sequence and the wave is 0.73 of the card, so the phases do not overlap. Against the pair it replaces
   (`add_rms_quant` 21.3 us + `quant_e4m3_128` 5.8 us) a row-count sweep gives 36.1 -> 27.2 us at 768 rows (30.0 with `xn` stored).
4. **A wide kernel for the decode token's front end cost the decode token.** The first form put the E4M3 path into `add_rms_quant`; its extra
   registers and shared memory made the one-row decode launch 5 us slower (80.7 -> 76.7 tok/s at one sequence). It is a kernel of its own now.
5. **The row count that fills whole waves is not a lever.** Cost model: output tiles of 128 x 256 against 132 SMs, a product costs its waves
   times its K (`model::serve::WaveCost`): down 17408x5120 is one wave at 768 rows and two at 896, gate/up four waves at 768 and 896.
   Moving the mixed iteration's row count to the best of its neighbours (two quanta each way) lowered c=64 throughput (813 -> 790 tok/s, TPOT 72.9 ->
   75.9 ms) and cut median TTFT (441 -> 385 ms at c=64, 434 -> 359 at c=16): longer iterations raise every running sequence's token interval, and
   the measured GEMM stack at 768 rows is 16.5 % slower than at 640 for 20 % more rows, a 3 % gain on half of the iteration. Not landed.
6. **A bf16 residual is not measured.** The residual traffic of the front end (a read, sum write: 31 MB of its 51) would, with a bf16 `b` from the
   down/out twins too, take it to about 9 us, the figure the byte count says; the residual is also what stage hand-off, taps and the head read
   as f32, so it is the next step only if the quality gate is run on it. The f32 residual stayed.

## Quality

`qwen35_fp8_parity` (8 x 1500 tokens, 32 steps), `--act-precision f32` against `bf16` in one build (the f32 run is the base's arithmetic with the
E4M3 front end), two source texts: run 1 A f32 98.05 % / KL 1.93e-3, bf16 96.48 % / 1.70e-3, B f32 98.83 % / 1.65e-3, bf16 99.61 % / 1.84e-3; run 2 (final build) A f32 98.83 % / 1.76e-3,
bf16 97.66 % / 1.53e-3, B f32 98.83 % / 1.94e-3, bf16 99.61 % / 1.81e-3 (no flip above margin 0.5 in any run). The base's own spread
(knowledge 393): 98.05-99.22 % and KL 1.5-3.2e-3; the bf16 stream's KL is the same or lower and its agreement moves by 1-3 positions of 256 in
either direction. Six greedy chats (128 tokens, thinking on) are identical between the two settings.

Training: nothing on the NF4 training path uses the FP8 GEMM or the segment kernels; the shared GDN prep/loop sources gained a flag that is off
there. W8K probe (`--layers 64 --tokens 8192 --rank 16 --seed 7 --steps 4 --residuals bf16 --base nf4 --gdn-segment 2048 --mlp-rows 2048`,
gpu1): 1608 tok/s warm (1597 before), losses 13.4643 / 13.4072 / 13.3341 / 13.2478 against 13.4643 / 13.4072 / 13.3342 / 13.2478.
