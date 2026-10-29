<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 372. The group-scale fold bounds the int8 prefill GEMM at about three issue slots per output, not at the tensor cores

Question: why does the Q8_0 prefill GEMM stop at 200-290 TOPS on a part whose int8 tensor cores measure 1191 TOPS
(`mma.sync` probe) and are rated 1979?

Finding: GGUF Q8_0 carries one weight scale per 32 elements and the activation quantiser one scale per 32, both along K,
so no library GEMM applies and every output element needs its own int32 sum of every k32 group scaled and added to an f32
total. That fold costs issue slots, not tensor-core time. Measured on a GH200 with the Hopper warpgroup kernel
(`matmul_i8_wgmma`) at 4096x5120x5120: with the fold replaced by a trivial accumulate the kernel runs at ~950 TOPS (loads,
descriptors and `wgmma` included); with the fold it runs at 278. The fold alone is about 0.65 of 0.95 ms. Per element per
group the instruction count was 4.8 (seed `MOV`, bias `FADD`, scale-product `FMUL`, `FFMA`, plus scale loads); the SM's 128
FP32 lanes then bound the kernel near 1024 MAC/clk/SM, a quarter of the tensor-core rate.

What moved it: the bias subtraction and the scale product merge into one fused multiply-add when `bias * xa` is exact,
which holds when the activation scale has at most 22 significant bits. The kernels round it to 21 (relative change at most
4.8e-7, under the f32 noise of the running sum): `fma(d, xa, -bias*xa)` is the group sum times the activation scale, one more
`fma` applies the weight scale. That is 3 slots per element with the seed (2 in the `mma.sync` kernel, which seeds through
the MMA's C operand): +10% on `mma.sync`, +22% on `wgmma`. What did not move it: a 128x128 `mma.sync` tile (no better than
64x64 at any width), 256-wide tiles (slower), and hoisting the scale loads (+8%).

Not tried: overlapping the fold of group g with the `wgmma` of g+1 (needs two accumulator sets, so BN 64 or a producer
warpgroup with `setmaxnreg`), and a cuBLAS bf16 route, which rounds weights to bf16/f16 and cannot meet the 2e-5 oracle bar.

Rule: grade a Q8_0 prefill GEMM against the fold bound (instruction slots per output per k32 group), not only the MMA roof;
and read a TOPS figure from a sweep only beside a checked output (knowledge 319), and a timing only from interleaved
repeats - one-off runs of the same kernel differed by 15%.
