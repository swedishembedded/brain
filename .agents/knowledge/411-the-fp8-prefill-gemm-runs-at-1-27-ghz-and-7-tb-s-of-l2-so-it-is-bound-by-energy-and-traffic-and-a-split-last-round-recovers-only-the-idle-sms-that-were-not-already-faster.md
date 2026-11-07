<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 411. The FP8 prefill GEMM runs at 1.27 GHz and 7 TB/s of L2, so it is bound by energy and traffic, and a split last round recovers only the idle SMs that were not already faster

**Lesson:** Under a power-capped clock the prefill GEMM is bound by energy and cache traffic, not by
stalls, so only fewer bytes and fewer instructions per FLOP help. A split last round recovers only
the idle SMs that were not already faster.

## What happened
Question (the brief of 2026-11-01, round 2, after knowledge 407): `matmul_fp8_wgmma_tma_n256`
against the reference (DeepGEMM `sm90_fp8_gemm_1d2d` through vLLM, bf16 output) at 2048 / 4096 /
8192 rows on the six production products, on the same card. What is left, and what does it cost?

Where the kernel stands (gpu1, interleaved, two rounds each, TFLOPS; brain is the f32-output kernel,
the reference writes bf16):

| shape | rows | reference | brain before | brain after |
|---|---|---|---|---|
| gate/up 5120x34816 | 2048 / 4096 / 8192 | 1271-1273 / 1106-1159 / 1083-1097 | 1175 / 1180 / 1175 | 1145-1170 / 1155-1167 / 1157-1161 |
| down 17408x5120 | 2048 / 4096 / 8192 | 833-1077 / 1075-1086 / 1098-1128 | 1121 / 1227 / 1217 | 1204-1205 / 1194-1216 / 1223-1243 |
| qkv 5120x10240 | 2048 / 4096 / 8192 | 1026-1198 / 1029-1040 / 1028-1046 | 1184 / 1157 / 1150 | 1186 / 1156-1158 / 1130-1158 |
| qkv 5120x12288 | 2048 / 4096 / 8192 | 1033-1186 / 1047-1070 / 1047-1051 | 1180 / 1162 / 1153 | 1184 / 1154 / 1154 |
| out 6144x5120 | 2048 / 4096 / 8192 | 802-895 / 1036-1041 / 1040-1047 | 1028 / 1176 / 1157 | 1175-1181 / 1185-1186 / 1160-1162 |
| gdn-in 5120x16384 | 2048 / 4096 / 8192 | 1113-1199 / 1069-1088 / 1051-1073 | 1178 / 1162 / 1158 | 1166-1170 / 1151-1153 / 1152-1156 |

The kernel is at or above the reference everywhere except gate/up at 2048 rows, where the
reference's 1272 TFLOPS (574 us) is 8 % ahead of ours (625 us); at 4096 gate/up the two are level.
The reference's own figures move 5-20 % between runs of one shape (the card's clock), ours move 1-2
%.

The 8K x 1 prefill gap in the served trace (383.7 against 355.3 ms in the GEMM) is therefore not
this isolated rate. The served engine runs this GEMM on 2048-row chunks (the trace shows gate/up at
680 us, 2048 rows), and there it takes 8-20 % longer than the probe's figure for the same shape (677
against 597 us bf16-out gate/up) while the reference runs its larger chunks.

## Why
The bound is energy and L2 traffic. `ncu --clock-control none` on gate/up 8192 rows reports the SM
clock at 1.27 GHz for the whole kernel (84 % tensor pipe active), against 1.98 GHz when idle: the
card holds 1.27-1.3 GHz under this load, which is also what the probe's 1.18 PFLOPS at 84 % activity
implies (132 SMs x 8192 FLOP/cycle x 0.84 x 1.28 GHz).

- L2-to-SM bytes at 4096 rows are 9.1 GB in 1.24 ms, 7.4 TB/s (the reference 7.1 GB at 6.5 TB/s).
- `nvidia-smi` reports 1980 MHz and 136 W through the whole run and is not evidence of either.
- A stall that only idles a warp costs nothing in this regime (knowledge 393); the levers are bytes
  moved per FLOP and instructions per FLOP.
- Of the bytes, the weight tile re-read once per 128-row tile of x is 5.7 of the 9.1 GB at 4096
  rows.

Removing the whole epilogue (stores and the scale multiplies) made the kernel 8 % slower (ncu base
clock 1.37 -> 1.48 ms, tensor pipe 85.1 -> 75.9 %): the epilogue is what staggers the two consumer
warpgroups, which otherwise wake together on every TMA arrival and multiply at the same time.

## What to do
What landed:

1. **Tail split (commit "the last part-filled round of the grid runs as column halves").**
   - With `rem = tiles % grid` tiles left over and `2 * rem <= grid`, each leftover tile is run as
     two 128-column halves by two blocks (both warpgroups on `wgmma m64n128k32`, the producer
     loading one 128-row weight box), not as `rem` whole tiles with the rest of the card idle.
   - A first form that split the tile by rows (one warpgroup a block, the other only releasing
     stages) gained 2.8 % on down at 2048 rows; the column halves gain 7.7 %.
   - ncu, base clock, 2048 x 17408 x 5120: 387.5 -> 344.8 us, elapsed cycles 474.8 k -> 410.5 k,
     tensor pipe 72.3 -> 83.8 %, instructions 82.8 -> 84.8 M (unchanged for the shapes without a
     tail).
   - Probe: down 2048 rows 1121 -> 1204, out 2048 rows 1028 -> 1175 TFLOPS; gate/up 2048 (rem 64)
     627 -> 625 us, qkv and gdn-in at 8192 rows (rem 52, 36, 4) within noise.
   - The idealised gain of 1/rounds x 0.5 is not reached where the full rounds are bandwidth-bound:
     the part-filled round with half the SMs idle runs faster per tile than a full one, so the idle
     SMs were not all lost time.
2. **The tuner's output buffer is poisoned.**
   - `time_class` checked each candidate against an output buffer shared by all candidates, so an
     element a candidate never stored kept the value an earlier candidate left there: with the tail
     split's second column offset broken, the admission test still passed.
   - The buffer is NaN before each checked launch now; the broken split fails (`non-finite output`)
     and every existing candidate still passes.
   - The admission shapes gain three with a split round (ragged rows, a half past `n`).
3. **`prefill_tma_back_to_back_stress`** (ignored test, knowledge 410's rule applied to the prefill
   kernels): the three TMA prefill candidates over rows 129 / 768 / 2048 / 2300 / 4096 / 8192 and
   the six products, 20 rounds back to back.
   - 13 minutes, `dmesg | grep -i xid` unchanged (16 old lines before and after).
   - compute-sanitizer memcheck, racecheck, synccheck on the split shape: 0 errors, 0 hazards.
   - `qwen35_fp8_parity` (8 x 1500, 32 steps, the repository's own sources as text): 98.05 % / KL
     3.09e-3 inside the recorded 98.05-99.22 % / 1.74-3.16e-3; the arithmetic per output element is
     unchanged.

Open:

- Gate/up at 2048 rows, 8 % behind the reference, and the served 2048-row chunk's own overhead (8-20
  % over the probe). The tail split does not reach it: gate/up 2048 has a 64-tile tail that the
  split did not speed up.
- A tail of more than half the grid (rem 84-128 of 132 for down / qkv at 4096 rows) cannot be split
  in two; stream-K or an N split in thirds would be needed.
- Less traffic per FLOP (the 2-block multicast, knowledge 407 item 3, measured at the base clock
  where the L2 is not the limit; it was never measured at the 1.27 GHz the card holds) and fewer
  instructions per FLOP are the remaining levers; both were rejected at the base clock only.

Training: the FP8 GEMM is not on the NF4 training path; nothing to gradcheck.

## Evidence
Served A/B (gpu1, `matrix.sh`, SPEC unset, base = origin/main before, interleaved twice): 8K x 1
mean TTFT base 571.7 / 574.3 ms, now 572.4 / 560.0; 16K x 1 base 1168.0 / 1169.5, now 1182.9 /
1162.5; 8K x 16 314.7 / 314.1 tok/s base, 310.1 / 315.0 now. No served change outside the run-to-run
spread, as the 2048-row down and out products are a few percent of the prefill.
