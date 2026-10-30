<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 386. The FP8 weight stream stopped at 2 TB/s on instructions and barriers, not on the memory, so each lane copies its own fragment and the hardware decodes the bytes

Question: knowledge 382 left `matmul_fp8_dq_stream_r4` at ~2.1 TB/s (12.3 ms of an 18.8 ms decode graph) with the m <= 8 variants "not limited by
shared memory or ring depth". The card reads 3.68 TB/s (the tuner's `roof_read` probe, 1 GiB, 16-byte `ld.global.nc` grid-stride). Why does the
stream stop, and what reaches the roof?

Measured (gpu1, `make test/one P=brain-gpu-core T=fp8_linear_native ARGS="--ignored --nocapture fp8_decode_bandwidth"`: weights cycled past the L2,
device events; ncu on the same binary, `-k regex:fp8_dq_stream_r4 -c 1`):

1. **The limiter was the issue slots and the barrier, not bytes in flight.** r4 at the 5120 x 34816 shape: 25.3 M warp instructions for 178 MB of
   weights (4.5 thread instructions per weight byte), issue slots 42 % busy at 2.09 TB/s, 3 blocks (12 warps) an SM because the stage ring
   is sized for 64 rows of x (74 KB) whatever the batch, one `__syncthreads` per 8 KB stage. Little's law was met (48 KB in flight an SM against the
   ~21-30 KB the memory needs); the 8-row-x variants of 382 freed shared memory but kept the 2.5-instruction bf16 decode (`prmt`, `shr`, two `lop3`,
   `fma.bf16x2` a pair) and the barrier, which is why they did not move.
2. **What was built** (`cu/matmul_fp8_gemv_f16.cu`, four candidates `matmul_fp8_gemv_f16_a{tiles}w{warps}d{slots}b{blocks}` through the tuner):
   - the lane copies its own A fragment (16-byte `cp.async` chunks of the (g, g+8) rows, the k permutation of 377) into a per-warp ring and reads the
     same bytes back: no barrier, the bytes in flight cost no registers (a register-prefetch version spilled at 255 registers and ran at 2.4 TB/s);
   - `cvt.rn.f16x2.e4m3x2` decodes two weights in one instruction and is exact (f16 holds every E4M3 code), so the block scale is not in the operand:
     the tensor core adds the 128 k of a block unscaled and one f32 fma per accumulator folds `scale * partial`. The bf16 activations are widened to f16
     once per fragment (`satfinite`; exact for 6e-5 <= |x| <= 65504). The result is closer to the f64 reference of the stored weights than the bf16
     kernels (they round `code * scale` to bf16);
   - the copies of the first line steps are issued before `griddepcontrol.wait` (the weights do not depend on the previous kernel) and
     `launch_dependents` is signalled first, so under programmatic dependent launch the next GEMV streams while this one finishes;
   - a block is `warps` warps taking the 128-wide line steps of K in turn, 16 x `tiles` weight rows, partial sums added in warp order through shared memory
     (deterministic).
3. **Result**, isolated kernel (us, TB/s of weight bytes), best candidate against r4, rows 1:

   | shape (K x N) | r4 | f16 stream |
   |---|---|---|
   | 5120 x 34816 | 73.5 us 2.43 | 56.5-59 us 3.03-3.16 |
   | 5120 x 17408 | 43.8 us 2.04 | 31.3 us 2.85 |
   | 17408 x 5120 | 41.3 us 2.16 | 32.8 us 2.72 |
   | 5120 x 10240 | 29.0 us 1.81 | 21.5 us 2.44 |
   | 6144 x 5120 | 19.7 us 1.60 | 15.0 us 2.09 |

   The isolated figure includes a ~6 us launch floor of the event timing; those are 89-97 % of "bytes at the 3.68 TB/s roof plus the floor".
4. **Decode graph** (nsys, FP8 27B, c = 1, critical-path attribution = time between consecutive kernel completions): `matmul_fp8_*` 11.82 -> 9.13 ms of
   15.81 -> 13.15 ms a token. At 24.3 GB of weights a token that is 2.66 TB/s in the graph: a launch's ramp and tail cost what the isolated floor does.

Not solved here: rows 9-16 (the activation fragment is converted per MMA, two x tiles, and the L1 serves the B fragments at 8 lines a load, so the f16 stream
loses to r4 above 8 rows); a GEMV whose x is the A operand would amortise both.
