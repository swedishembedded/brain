// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for memory-bandwidth-bound LLM
// decode on GPUs for its clients. If your team needs expertise in hand-written
// CUDA kernels that stream FP8 and NVFP4 weights at the speed of the memory
// system then you can procure our services by sending an email to
// info@swedishembedded.com.
//
// Float-read GEMVs over weights held as the checkpoint stores them, with the
// identical buffer layouts and uniforms of their WGSL siblings
// (`matmul_fp8_gemv.wgsl`, `matmul_nvfp4_gemv.wgsl`, `moe_nvfp4_gemv_gather.wgsl`):
//
//   brain_matmul_fp8_gemv          y[r, c] = sum_k x[xr0 + r, k] * e4m3(w[first + c, k]) * scale[tile]
//   brain_matmul_nvfp4_gemv        y[r, c] = global * sum_k x[xr0 + r, k] * e2m1(w[first + c, k]) * e4m3(scale[k / 16])
//   brain_moe_nvfp4_gemv_gather    out[s, c] = global[e] * (the same product over expert e = ids[s]'s row c, x row s / xdiv)
//
//   fp8    params : u32 [m, k, count, first, block, cb, xr0, rpw]   m <= 8, k % 16 == 0, block % 16 == 0
//          x f32 [.., k]; w u32 [n, k/4] (four E4M3 bytes per word); s f32 [ceil(n/block), cb]; y f32 [m, count]
//   nvfp4  params : u32 [m, k, count, first, xr0, global_bits, rpw, pad]   m <= 8, k % 64 == 0
//          x f32; w u32 [n, k/8] (eight E2M1 codes per word); s u32 [n, k/64] (four E4M3 bytes per word); y f32 [m, count]
//   gather params : u32 [slots, k, n, xdiv]   k % 64 == 0
//          x f32 [rows, k]; ids u32 [slots]; w, s as above for the bank [blocks * n]; gl f32 [blocks]; out f32 [slots, n]
//
// Why these kernels exist
// -----------------------
// Decode streams every weight once per token, so the figure of merit is the fraction of HBM bandwidth a weight row is
// read at. The generated kernel (`wgsl-cuda`) is built without fused multiply-add, has no vector loads and
// zero-initialises its shared tables at entry; this one reads 16 bytes per lane (`ld.global.nc.L1::no_allocate.v4`,
// the weights are used once and must not evict the activations every row re-reads), issues every load of a stage before
// it consumes any, decodes the codes with integer bit moves instead of table lookups (an FP8 or E2M1 code placed in
// the top bits of an f16 is the value times 2^-8 or 2^-14: the exponent biases differ by that much, and the subnormal
// range lines up exactly, so the code decodes by two masks and a shift and the factor folds into one multiply at the
// end), and reduces the 16 lanes of a row with warp shuffles.
//
// Numerics: each lane accumulates its chunks in ascending order into f32 with fused multiply-adds and the 16 lanes are
// folded by a butterfly, so the sum is not the WGSL tier's order; the result agrees with it to f32 rounding of the
// running sum (the gate holds both to an f64 oracle). The decode itself is exact.
//
// Alignment: x, w and the bank are used through 16-byte vector loads, so their bindings must start 16-byte aligned (a
// whole device buffer does). Output stores are scalar. No `__restrict__`: brain's device buffers alias by design.

#define BRAIN_QG_COLS 16
#define BRAIN_QG_LANES 16
#define BRAIN_QG_THREADS (BRAIN_QG_COLS * BRAIN_QG_LANES)
#define BRAIN_QG_MAXM 8

// 16-byte weight load that does not allocate in L1 (the weights stream through once).
__device__ __forceinline__ uint4 brain_qg_ld_stream(const uint4* p) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    uint4 v;
    asm("ld.global.nc.L1::no_allocate.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
    return v;
#else
    return __ldg(p);
#endif
}

// Two f16 values packed in `bits` (low half first) as floats (no header needed: the conversion is the PTX instruction).
__device__ __forceinline__ float2 brain_qg_h2f(unsigned int bits) {
    float lo, hi;
    asm("{ .reg .b16 a, b; mov.b32 {a, b}, %2; cvt.f32.f16 %0, a; cvt.f32.f16 %1, b; }" : "=f"(lo), "=f"(hi) : "r"(bits));
    return make_float2(lo, hi);
}

// Two E4M3 bytes at bit 0 and bit 16 of `u` (each byte alone in its half) as the pair (value * 2^-8).
__device__ __forceinline__ float2 brain_qg_e4m3_pair(unsigned int u) {
    return brain_qg_h2f(((u & 0x007F007Fu) << 7) | ((u & 0x00800080u) << 8));
}

// Two E2M1 codes at bit 0 and bit 16 of `u` (each nibble alone in its half) as the pair (value * 2^-14).
__device__ __forceinline__ float2 brain_qg_e2m1_pair(unsigned int u) {
    return brain_qg_h2f(((u & 0x00070007u) << 9) | ((u & 0x00080008u) << 12));
}

// One E4M3 scale byte as value * 2^-8.
__device__ __forceinline__ float brain_qg_e4m3_scale(unsigned int b) {
    return brain_qg_e4m3_pair(b & 0xFFu).x;
}

// The 16 lanes of a row (a half warp) summed; every lane of the row holds the total.
__device__ __forceinline__ float brain_qg_row_sum(float v) {
#pragma unroll
    for (int off = BRAIN_QG_LANES / 2; off > 0; off >>= 1) { v += __shfl_xor_sync(0xffffffffu, v, off, BRAIN_QG_LANES); }
    return v;
}

// ---- FP8 -----------------------------------------------------------------------------------------------------------

// The dot product of one 16-weight chunk (four words of E4M3 bytes) with the 16 activations at `xc`, times 2^-8.
__device__ __forceinline__ float brain_f8_chunk_dot(const uint4 w, const float4* xc) {
    const unsigned int words[4] = {w.x, w.y, w.z, w.w};
    float sum = 0.0f;
#pragma unroll
    for (int q = 0; q < 4; ++q) {
        const float4 xv = __ldg(xc + q);
        const float2 even = brain_qg_e4m3_pair(words[q] & 0x00FF00FFu);          // bytes 0 and 2
        const float2 odd = brain_qg_e4m3_pair((words[q] >> 8) & 0x00FF00FFu);    // bytes 1 and 3
        sum = fmaf(even.x, xv.x, sum);
        sum = fmaf(odd.x, xv.y, sum);
        sum = fmaf(even.y, xv.z, sum);
        sum = fmaf(odd.y, xv.w, sum);
    }
    return sum;
}

template <int MR>
__device__ __forceinline__ void brain_matmul_fp8_gemv_body(const unsigned int* params, const float* x, const unsigned int* w, const float* s, float* y) {
    const unsigned int m = params[0], k = params[1], count = params[2], first = params[3];
    const unsigned int block = params[4], cb = params[5], xr0 = params[6];
    const unsigned int blk = blockIdx.y * gridDim.x + blockIdx.x;
    const unsigned int col = blk * BRAIN_QG_COLS + threadIdx.x / BRAIN_QG_LANES;
    const unsigned int lane = threadIdx.x % BRAIN_QG_LANES;
    const bool live = col < count;
    const unsigned int chunks = k >> 4;                                    // 16-weight chunks per row
    const unsigned int row = first + (live ? col : 0u);
    const uint4* wrow = reinterpret_cast<const uint4*>(w) + static_cast<unsigned long long>(row) * chunks;
    const float* srow = s + static_cast<unsigned long long>(row / block) * cb;

    float acc[MR];
#pragma unroll
    for (int r = 0; r < MR; ++r) { acc[r] = 0.0f; }

    // Every load of a stage (weights and their scales) is issued before the first is consumed.
    // A single activation row leaves registers free for a deeper stage: the more of a row's loads are in flight at
    // once, the fewer round trips to memory a short row costs.
    constexpr int UNROLL = (MR == 1) ? 8 : 4;
    for (unsigned int j0 = lane; j0 < chunks; j0 += UNROLL * BRAIN_QG_LANES) {
        uint4 wv[UNROLL];
        float sc[UNROLL];
#pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
            const unsigned int j = j0 + u * BRAIN_QG_LANES;
            if (live && j < chunks) {
                wv[u] = brain_qg_ld_stream(wrow + j);
                sc[u] = __ldg(srow + (j * 16u) / block);
            } else {
                wv[u] = make_uint4(0u, 0u, 0u, 0u);
                sc[u] = 0.0f;
            }
        }
#pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
            const unsigned int j = j0 + u * BRAIN_QG_LANES;
            if (j < chunks) {
#pragma unroll
                for (int r = 0; r < MR; ++r) {
                    if (static_cast<unsigned int>(r) < m) {
                        const float4* xc = reinterpret_cast<const float4*>(x + static_cast<unsigned long long>(xr0 + r) * k) + 4ull * j;
                        acc[r] = fmaf(brain_f8_chunk_dot(wv[u], xc), sc[u], acc[r]);
                    }
                }
            }
        }
    }
#pragma unroll
    for (int r = 0; r < MR; ++r) {
        const float total = brain_qg_row_sum(acc[r]) * 256.0f;              // undo the 2^-8 of the f16 placement
        if (static_cast<unsigned int>(r) < m && lane == 0u && live) { y[static_cast<unsigned long long>(r) * count + col] = total; }
    }
}

extern "C" __global__ void __launch_bounds__(BRAIN_QG_THREADS, 4)
brain_matmul_fp8_gemv_m1(const unsigned int* params, const float* x, const unsigned int* w, const float* s, float* y) {
    brain_matmul_fp8_gemv_body<1>(params, x, w, s, y);
}

extern "C" __global__ void __launch_bounds__(BRAIN_QG_THREADS, 4)
brain_matmul_fp8_gemv(const unsigned int* params, const float* x, const unsigned int* w, const float* s, float* y) {
    brain_matmul_fp8_gemv_body<BRAIN_QG_MAXM>(params, x, w, s, y);
}

// ---- NVFP4 ---------------------------------------------------------------------------------------------------------

// One 16-weight block (two words of E2M1 codes) against the activations at `xb` (16 floats), times 2^-14.
__device__ __forceinline__ float brain_fp4_block_dot(const unsigned int w0, const unsigned int w1, const float4* xb) {
    const unsigned int words[2] = {w0, w1};
    float sum = 0.0f;
#pragma unroll
    for (int q = 0; q < 2; ++q) {
        const float4 xa = __ldg(xb + 2 * q);
        const float4 xc = __ldg(xb + 2 * q + 1);
        const float xe[4] = {xa.x, xa.y, xa.z, xa.w};
        const float xo[4] = {xc.x, xc.y, xc.z, xc.w};
#pragma unroll
        for (int g = 0; g < 4; ++g) {
            // Nibbles g and g + 4 of the word: elements g and g + 4 of its eight.
            const float2 v = brain_qg_e2m1_pair((words[q] >> (4 * g)) & 0x000F000Fu);
            sum = fmaf(v.x, xe[g], sum);
            sum = fmaf(v.y, xo[g], sum);
        }
    }
    return sum;
}

// A 32-weight chunk (one uint4 of codes, two block scales packed in the low and high byte of `sw16`) against `xc`
// (32 floats): sum over the two blocks of dot * scale, times 2^-22.
__device__ __forceinline__ float brain_fp4_chunk(const uint4 wv, const unsigned int sw16, const float4* xc) {
    const float s0 = brain_qg_e4m3_scale(sw16 & 0xFFu);
    const float s1 = brain_qg_e4m3_scale((sw16 >> 8) & 0xFFu);
    return fmaf(brain_fp4_block_dot(wv.z, wv.w, xc + 4), s1, brain_fp4_block_dot(wv.x, wv.y, xc) * s0);
}

template <int MR>
__device__ __forceinline__ void brain_matmul_nvfp4_gemv_body(const unsigned int* params, const float* x, const unsigned int* w, const unsigned int* s, float* y) {
    const unsigned int m = params[0], k = params[1], count = params[2], first = params[3], xr0 = params[4];
    const float global = __uint_as_float(params[5]);
    const unsigned int blk = blockIdx.y * gridDim.x + blockIdx.x;
    const unsigned int col = blk * BRAIN_QG_COLS + threadIdx.x / BRAIN_QG_LANES;
    const unsigned int lane = threadIdx.x % BRAIN_QG_LANES;
    const bool live = col < count;
    const unsigned int chunks = k >> 5;                                    // 32-weight chunks per row
    const unsigned int row = first + (live ? col : 0u);
    const uint4* wrow = reinterpret_cast<const uint4*>(w) + static_cast<unsigned long long>(row) * chunks;
    const unsigned int* srow = s + static_cast<unsigned long long>(row) * (k >> 6);

    float acc[MR];
#pragma unroll
    for (int r = 0; r < MR; ++r) { acc[r] = 0.0f; }

    constexpr int UNROLL = (MR == 1) ? 4 : 2;
    for (unsigned int j0 = lane; j0 < chunks; j0 += UNROLL * BRAIN_QG_LANES) {
        uint4 wv[UNROLL];
        unsigned int sw[UNROLL];
#pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
            const unsigned int j = j0 + u * BRAIN_QG_LANES;
            if (live && j < chunks) {
                wv[u] = brain_qg_ld_stream(wrow + j);
                sw[u] = __ldg(srow + (j >> 1)) >> ((j & 1u) * 16u);
            } else {
                wv[u] = make_uint4(0u, 0u, 0u, 0u);
                sw[u] = 0u;
            }
        }
#pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
            const unsigned int j = j0 + u * BRAIN_QG_LANES;
            if (j < chunks) {
#pragma unroll
                for (int r = 0; r < MR; ++r) {
                    if (static_cast<unsigned int>(r) < m) {
                        const float4* xc = reinterpret_cast<const float4*>(x + static_cast<unsigned long long>(xr0 + r) * k) + 8ull * j;
                        acc[r] += brain_fp4_chunk(wv[u], sw[u], xc);
                    }
                }
            }
        }
    }
    // 2^-14 (E2M1 placed in an f16) times 2^-8 (the E4M3 block scale) is undone once, with the tensor scale.
    const float scale = 4194304.0f * global;
#pragma unroll
    for (int r = 0; r < MR; ++r) {
        const float total = brain_qg_row_sum(acc[r]) * scale;
        if (static_cast<unsigned int>(r) < m && lane == 0u && live) { y[static_cast<unsigned long long>(r) * count + col] = total; }
    }
}

extern "C" __global__ void __launch_bounds__(BRAIN_QG_THREADS, 4)
brain_matmul_nvfp4_gemv_m1(const unsigned int* params, const float* x, const unsigned int* w, const unsigned int* s, float* y) {
    brain_matmul_nvfp4_gemv_body<1>(params, x, w, s, y);
}

extern "C" __global__ void __launch_bounds__(BRAIN_QG_THREADS, 4)
brain_matmul_nvfp4_gemv(const unsigned int* params, const float* x, const unsigned int* w, const unsigned int* s, float* y) {
    brain_matmul_nvfp4_gemv_body<BRAIN_QG_MAXM>(params, x, w, s, y);
}

// ---- NVFP4 expert gather ---------------------------------------------------------------------------------------------

// One block covers 16 weight rows of ONE slot's expert; the grid is slot-major, `ceil(n / 16)` blocks per slot.
extern "C" __global__ void __launch_bounds__(BRAIN_QG_THREADS, 4)
brain_moe_nvfp4_gemv_gather(const unsigned int* params, const float* x, const unsigned int* ids, const unsigned int* w,
                            const unsigned int* s, const float* gl, float* out) {
    const unsigned int slots = params[0], k = params[1], n = params[2], xdiv = params[3];
    const unsigned int blk = blockIdx.y * gridDim.x + blockIdx.x;
    const unsigned int tiles = (n + BRAIN_QG_COLS - 1u) / BRAIN_QG_COLS;
    const unsigned int slot = blk / tiles;
    const unsigned int col = (blk - slot * tiles) * BRAIN_QG_COLS + threadIdx.x / BRAIN_QG_LANES;
    const unsigned int lane = threadIdx.x % BRAIN_QG_LANES;
    const bool live = slot < slots && col < n;
    const unsigned int expert = live ? ids[slot] : 0u;
    const unsigned int chunks = k >> 5;
    const unsigned long long rowi = static_cast<unsigned long long>(expert) * n + (live ? col : 0u);
    const uint4* wrow = reinterpret_cast<const uint4*>(w) + rowi * chunks;
    const unsigned int* srow = s + rowi * (k >> 6);
    const float4* xrow = reinterpret_cast<const float4*>(x + static_cast<unsigned long long>(live ? slot / xdiv : 0u) * k);

    float acc = 0.0f;
    constexpr int UNROLL = 2;
    for (unsigned int j0 = lane; j0 < chunks; j0 += UNROLL * BRAIN_QG_LANES) {
        uint4 wv[UNROLL];
        unsigned int sw[UNROLL];
#pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
            const unsigned int j = j0 + u * BRAIN_QG_LANES;
            if (live && j < chunks) {
                wv[u] = brain_qg_ld_stream(wrow + j);
                sw[u] = __ldg(srow + (j >> 1)) >> ((j & 1u) * 16u);
            } else {
                wv[u] = make_uint4(0u, 0u, 0u, 0u);
                sw[u] = 0u;
            }
        }
#pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
            const unsigned int j = j0 + u * BRAIN_QG_LANES;
            if (j < chunks) { acc += brain_fp4_chunk(wv[u], sw[u], xrow + 8ull * j); }
        }
    }
    const float total = brain_qg_row_sum(acc) * (4194304.0f * (live ? __ldg(gl + expert) : 0.0f));
    if (lane == 0u && live) { out[static_cast<unsigned long long>(slot) * n + col] = total; }
}
