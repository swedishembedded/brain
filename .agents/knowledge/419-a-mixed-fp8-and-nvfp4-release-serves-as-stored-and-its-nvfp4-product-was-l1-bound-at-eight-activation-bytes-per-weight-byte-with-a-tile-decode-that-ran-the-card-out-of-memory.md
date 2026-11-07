<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 419. A mixed FP8 and NVFP4 release serves as stored, and its NVFP4 product was L1 bound at eight activation bytes per weight byte, with a tile decode that ran the card out of memory

**Lesson:** A mixed-precision release can be served as stored by the generic tiers, without a
conversion. The slow part was kernel shape: a GEMV that rereads wide activations for every weight
byte is bound by the first-level cache, and a product that decodes whole weight tiles to a wider
type on each call can run the card out of memory.

## What happened

Question (the brief of 2026-11-01): the user asked for FP8 and NVFP4 weight comparisons at matched
settings. `nvidia/Qwen3.8-27B-NVFP4` (ModelOpt MIXED_PRECISION: FP8 E4M3 with ONE static f32
`weight_scale` per tensor for every attention and Gated DeltaNet projection, NVFP4 for the MLPs and
the head, a bf16 MTP head) was refused at load ("no resident tier"). Which engines serve it on a
GH200 (sm_90, no FP4 tensor cores), and can brain serve it without a conversion?

### Which engines serve it (gpu1, defaults plus `--kv-cache-dtype bf16` and f32 state)

- vLLM 0.32: serves. FP8 projections `CutlassFP8ScaledMMLinearKernel`, NVFP4
  `HummingNvFp4LinearKernel` (a weight-only W4A16 JIT kernel: sm_90 has no FP4 tensor cores). 8K x
  1: TTFT 771 ms, TPOT 10.4 ms.
- SGLang 0.5.22: **does not serve it by default**. The scheduler dies capturing CUDA graphs: the
  dense MLP's `down_proj` is given the `(fp4, scale)` tuple of the fused SiLU+mul+FP4-quant
  (`_maybe_enable_silu_fp4_quant_fusion`), and on sm_90 the NVFP4 linear is the FP4 Marlin
  weight-only op, whose signature takes a Tensor: `RuntimeError: sglang::apply_fp4_marlin_linear()
  Expected a value of type 'Tensor' for argument 'input' but instead found type 'tuple'`. With the
  engine's own kill switch `SGLANG_DISABLE_SILU_FP4_QUANT_FUSION=1` it serves (8K x 1: TTFT 1380 ms,
  TPOT 9.6 ms).

## Why

### Where the NVFP4 time went (gpu1, device events, 17408 x 5120 gate/up; FP8 is 36 us at 1 to 16 rows)

| rows | before | after | what |
|---|---|---|---|
| 1 | 102 us | 38 | GEMV, 4 weight rows a warp |
| 2 | 239 | 55 | `matmul_nvfp4_mma8` |
| 8 | 239 | 56 | " |
| 16 | 229 | 68 | `matmul_nvfp4_mma16` |
| 32 | ~230 | 142 | `matmul_nvfp4_mma32` |
| 64 | 228 | 253 | `matmul_nvfp4_mma64` (4 warps along K, 255 registers) |
| 256 | 241 | 236 | tile decode + library (unchanged) |
| 8192 | 2480 | 2284 | " (FP8 1213) |

down (5120 x 17408): 1 row 114 -> 46 us, 8 rows 225 -> 57, 16 rows 226 -> 64; LM head (248320 x
5120): 1 row 1253 -> 396 us (FP8 350), 8 rows 3201 -> 539, 16 rows 3233 -> 648.

- **The GEMV was L1 bound.** ncu on the single-row kernel: L1/TEX 97 % busy, DRAM 10 %, 17,408
  local-memory spill requests. A 32-weight chunk is 16 bytes against 128 bytes of f32 activations,
  so a half warp read eight activation bytes per weight byte. A warp now owns four weight rows and
  reads the activation chunk once for the four (two bytes per weight byte); 32 lanes a group,
  because 16 left the 5120-row down projection at 80 blocks on 132 SMs.
- **The staged multi-row build had 32-way shared-memory bank conflicts**: lanes of a warp read
  chunk-major float4 128 bytes apart. Stored transposed (`xs[t][chunk]`) three rows fell from 396 to
  78 us. (An eight-row build of that kernel still lost to the library on the down projection, 253
  against 215 us; the tensor-core product replaced the range.)
- **The tile decode wrote the whole weight as bf16 (178 MB for gate/up) per product and kept it
  until the step was submitted.** A decode step of 8 sequences is 192 such products: the first
  concurrent batch died in `cuMemAllocFromPoolAsync` at any `--ctx` (the planner reserved nothing
  for it, and a smaller tile budget costs 1.6 to 2.4 times the time: 64 MB 359 us, 32 MB 344, 16 MB
  418 at 16 rows). `matmul_nvfp4_mma8/16/32/64` (up to 8, 16, 32, 64 bf16 activation rows) decode
  the codes in registers into the `mma.m16n8k16` fragments, weight rows as the 16-row dimension.

  The contraction order is permuted so a lane owns a whole 32-weight chunk of its row (one 16-byte
  load, two block scales) and a k16 step pairs four consecutive weights of each chunk with the same
  four activation positions per lane (one 8-byte load). The scale multiplies in f32 before the bf16
  rounding, which is exact (an E2M1 value times an E4M3 scale has at most five significant bits). A
  512-thread block with 128 registers spilled at 64 rows (accumulators 64 registers): one entry per
  activation-tile count, and the 64-row build runs 256 threads at 255 registers. The planner still
  keeps back two tile budgets for the rows above 64 (the library path).

## What to do

### What brain needed (all of it is the checkpoint and the generic tiers; no per-model flag)

- Per-tensor FP8: the checkpoint layer already read `weight_scale` as a `Tensor`-granular scale and
  the device sink held it on every tile of the grid, but `stored_tier` named only the tiled
  encodings and a fused group (q|k|v, gate|up) had no way to join parts with different scales. A new
  lossless `Repack` (`RepackKind::TensorScaleToTiles`) copies the payload and writes the tensor's
  f32 scale on every grid entry (a bf16 grid would round it); an f32 tiled release (not just
  per-tensor) is now held as stored, where it was converted through bf16.
- NVFP4: `Dtype::F4E2M1` names the stored NVFP4 tier ([`Weight::Micro`]); `upload_from_source`
  builds it only from a tensor stored that way (the other meaning of the dtype, the W4A8 layout, is
  never built behind the caller's back). `ResidentCheckpoint::head_tier` and `held_tiers` hold an
  NVFP4 head as stored (the resident held every HF head as bf16).
- The passes that write only the E4M3 rows of a producer (`silu_mul_e4m3`, `gate_mul_e4m3`, the E4M3
  fused front end, the gated norm) leave the f32 rows unwritten; a float-read NVFP4
  `gate`/`up`/`down` would read garbage. `E4m3Readers` (per family, from the tier policy) turns each
  off where a reader is not FP8. `Fusion` (q|k|v, qkv|z) is unchanged: it holds a group fused only
  when every part is FP8, which here is the attention side; the NVFP4 MLP keeps separate weights.
- The static `input_scale` is not used: brain quantises activations dynamically per 128 elements.

### Open

- Prefill: a product of 256 or more rows still decodes the tile and calls the library (8192 rows:
  2284 us against FP8's 1213, the bf16 tensor rate against the FP8 rate plus a 145 us decode): 8K x
  1 TTFT 909 ms against vLLM's 783 and the FP8 release's 568. A large-M NVFP4 GEMM (wgmma, the codes
  decoded into shared memory) would close part of it.
- The 64-row build (253 us) is no faster than the tile decode, and the LM head at 64 rows is 2.47 ms
  (FP8 386 us).
- `qops_gemv_small_m` fails on the base at the FP8 check (the tuned `matmul_fp8_gemv_f16_*`
  candidate multiplies in half precision against a 1e-5 relative tolerance); the NVFP4 half of it
  passes.

## Evidence

### Parity (`qwen35_fp8_parity`, now taking `--reference <dir>`; model.rs as text, 8 prompts of ~1300 tokens, 32 greedy steps, int8 KV)

- NVFP4 against the FP8 release (two quantisations of the same model; FP8 is the reference): 256
  positions, 93.36 % greedy agreement, mean rel-L2 9.43e-2, worst 2.84e-1, mean KL 1.76e-2 nats, 2
  flips at a reference margin above 0.5 (a first pass, 4 prompts x 24 steps with the unblocked
  kernels, gave 95.83 % and 2.0e-2 nats).
- NVFP4 against the same weights decoded to bf16 (kernel noise alone): 256 positions, 98.83 %
  agreement, mean rel-L2 1.85e-2, worst 4.98e-2, mean KL 1.17e-3 nats, no flip above margin 0.5 (the
  FP8 release against its bf16 decode, same prompts: 97.92 %, rel-L2 2.75e-2, KL 1.76e-3).
- An f32 oracle: the mixed-release fixture (`write_mixed_release_checkpoint`: per-tensor FP8, NVFP4
  MLPs and head, a ModelOpt layer declaration) built at the stored tiers equals the same weights
  decoded to f32 (CPU backend, relative L2 below 1e-12); the CUDA kernels are held to f64 oracles
  (`qgemv_native`: GEMV at ragged counts and windows, the tensor-core product over bf16-rounded
  activations up to K = 17408).
