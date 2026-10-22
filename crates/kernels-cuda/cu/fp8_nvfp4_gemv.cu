// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for memory-bandwidth-bound LLM
// decode on GPUs for its clients. If your team needs expertise in native
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

// The 16 weights of one chunk (four words of E4M3 bytes) as floats times 2^-8, in element order (decoded once per chunk,
// then dotted against every activation row).
__device__ __forceinline__ void brain_f8_decode16(const uint4 w, float* wf) {
    const unsigned int words[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
    for (int q = 0; q < 4; ++q) {
        const float2 even = brain_qg_e4m3_pair(words[q] & 0x00FF00FFu);          // bytes 0 and 2
        const float2 odd = brain_qg_e4m3_pair((words[q] >> 8) & 0x00FF00FFu);    // bytes 1 and 3
        wf[4 * q + 0] = even.x;
        wf[4 * q + 1] = odd.x;
        wf[4 * q + 2] = even.y;
        wf[4 * q + 3] = odd.y;
    }
}

// The decoded chunk `wf` against the 16 activations at `xc`.
__device__ __forceinline__ float brain_f8_dot16(const float* wf, const float4* xc) {
    float sum = 0.0f;
#pragma unroll
    for (int q = 0; q < 4; ++q) {
        const float4 xv = __ldg(xc + q);
        sum = fmaf(wf[4 * q + 0], xv.x, sum);
        sum = fmaf(wf[4 * q + 1], xv.y, sum);
        sum = fmaf(wf[4 * q + 2], xv.z, sum);
        sum = fmaf(wf[4 * q + 3], xv.w, sum);
    }
    return sum;
}

// `brain_f8_dot16` against activations held in shared memory.
__device__ __forceinline__ float brain_f8_dot16_smem(const float* wf, const float4* xc) {
    float sum = 0.0f;
#pragma unroll
    for (int q = 0; q < 4; ++q) {
        const float4 xv = xc[q];
        sum = fmaf(wf[4 * q + 0], xv.x, sum);
        sum = fmaf(wf[4 * q + 1], xv.y, sum);
        sum = fmaf(wf[4 * q + 2], xv.z, sum);
        sum = fmaf(wf[4 * q + 3], xv.w, sum);
    }
    return sum;
}

// Activation columns staged in shared memory per round by the multi-row builds.
#define BRAIN_QG_KB 1024

// The multi-row (2 to 8) FP8 build: the activation rows of a round are staged in shared memory once per block, so the 16
// weight rows of the block read them from there instead of every lane re-reading global memory per chunk.
template <int MR>
__device__ __forceinline__ void brain_matmul_fp8_gemv_staged(const unsigned int* params, const float* x, const unsigned int* w, const float* s, float* y) {
    const unsigned int m = params[0], k = params[1], count = params[2], first = params[3];
    const unsigned int block = params[4], cb = params[5], xr0 = params[6];
    const unsigned int blk = blockIdx.y * gridDim.x + blockIdx.x;
    const unsigned int col = blk * BRAIN_QG_COLS + threadIdx.x / BRAIN_QG_LANES;
    const unsigned int lane = threadIdx.x % BRAIN_QG_LANES;
    const bool live = col < count;
    const unsigned int chunks = k >> 4;
    const unsigned int row = first + (live ? col : 0u);
    const uint4* wrow = reinterpret_cast<const uint4*>(w) + static_cast<unsigned long long>(row) * chunks;
    const float* srow = s + static_cast<unsigned long long>(row / block) * cb;
    const float4* x4 = reinterpret_cast<const float4*>(x);
    __shared__ float4 xs[MR][BRAIN_QG_KB / 4];

    float acc[MR];
#pragma unroll
    for (int r = 0; r < MR; ++r) { acc[r] = 0.0f; }
    constexpr int UNROLL = 2;
    for (unsigned int kb = 0; kb < k; kb += BRAIN_QG_KB) {
        const unsigned int kn = (k - kb) < BRAIN_QG_KB ? (k - kb) : BRAIN_QG_KB;
        const unsigned int v4 = kn >> 2;
        for (unsigned int i = threadIdx.x; i < m * v4; i += BRAIN_QG_THREADS) {
            const unsigned int r = i / v4, c = i - r * v4;
            xs[r][c] = __ldg(x4 + static_cast<unsigned long long>(xr0 + r) * (k >> 2) + (kb >> 2) + c);
        }
        __syncthreads();
        const unsigned int nch = kn >> 4, cbase = kb >> 4;
        for (unsigned int c0 = lane; c0 < nch; c0 += UNROLL * BRAIN_QG_LANES) {
            uint4 wv[UNROLL];
            float sc[UNROLL];
#pragma unroll
            for (int u = 0; u < UNROLL; ++u) {
                const unsigned int c = c0 + u * BRAIN_QG_LANES;
                if (live && c < nch) {
                    wv[u] = brain_qg_ld_stream(wrow + cbase + c);
                    sc[u] = __ldg(srow + ((cbase + c) * 16u) / block);
                } else {
                    wv[u] = make_uint4(0u, 0u, 0u, 0u);
                    sc[u] = 0.0f;
                }
            }
#pragma unroll
            for (int u = 0; u < UNROLL; ++u) {
                const unsigned int c = c0 + u * BRAIN_QG_LANES;
                if (c < nch) {
                    float wf[16];
                    brain_f8_decode16(wv[u], wf);
#pragma unroll
                    for (int r = 0; r < MR; ++r) {
                        if (static_cast<unsigned int>(r) < m) { acc[r] = fmaf(brain_f8_dot16_smem(wf, &xs[r][4u * c]), sc[u], acc[r]); }
                    }
                }
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int r = 0; r < MR; ++r) {
        const float total = brain_qg_row_sum(acc[r]) * 256.0f;
        if (static_cast<unsigned int>(r) < m && lane == 0u && live) { y[static_cast<unsigned long long>(r) * count + col] = total; }
    }
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
                float wf[16];
                brain_f8_decode16(wv[u], wf);
#pragma unroll
                for (int r = 0; r < MR; ++r) {
                    if (static_cast<unsigned int>(r) < m) {
                        const float4* xc = reinterpret_cast<const float4*>(x + static_cast<unsigned long long>(xr0 + r) * k) + 4ull * j;
                        acc[r] = fmaf(brain_f8_dot16(wf, xc), sc[u], acc[r]);
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

extern "C" __global__ void __launch_bounds__(BRAIN_QG_THREADS, 2)
brain_matmul_fp8_gemv(const unsigned int* params, const float* x, const unsigned int* w, const float* s, float* y) {
    brain_matmul_fp8_gemv_staged<BRAIN_QG_MAXM>(params, x, w, s, y);
}

// ---- NVFP4 ---------------------------------------------------------------------------------------------------------

// The 32 weights of one chunk (a uint4 of E2M1 codes) as floats times 2^-14, in element order.
__device__ __forceinline__ void brain_fp4_decode32(const uint4 wv, float* wf) {
    const unsigned int words[4] = {wv.x, wv.y, wv.z, wv.w};
#pragma unroll
    for (int q = 0; q < 4; ++q) {
#pragma unroll
        for (int g = 0; g < 4; ++g) {
            // Nibbles g and g + 4 of the word: elements g and g + 4 of its eight.
            const float2 v = brain_qg_e2m1_pair((words[q] >> (4 * g)) & 0x000F000Fu);
            wf[8 * q + g] = v.x;
            wf[8 * q + g + 4] = v.y;
        }
    }
}

// A decoded 32-weight chunk (two block scales packed in the low and high byte of `sw16`) against `xc` (32 floats): the
// sum over the two blocks of dot * scale, times 2^-22.
__device__ __forceinline__ float brain_fp4_dot32(const float* wf, const unsigned int sw16, const float4* xc) {
    const float s0 = brain_qg_e4m3_scale(sw16 & 0xFFu);
    const float s1 = brain_qg_e4m3_scale((sw16 >> 8) & 0xFFu);
    float b0 = 0.0f, b1 = 0.0f;
#pragma unroll
    for (int t = 0; t < 4; ++t) {
        const float4 xv = __ldg(xc + t);
        b0 = fmaf(wf[4 * t + 0], xv.x, b0);
        b0 = fmaf(wf[4 * t + 1], xv.y, b0);
        b0 = fmaf(wf[4 * t + 2], xv.z, b0);
        b0 = fmaf(wf[4 * t + 3], xv.w, b0);
    }
#pragma unroll
    for (int t = 0; t < 4; ++t) {
        const float4 xv = __ldg(xc + 4 + t);
        b1 = fmaf(wf[16 + 4 * t + 0], xv.x, b1);
        b1 = fmaf(wf[16 + 4 * t + 1], xv.y, b1);
        b1 = fmaf(wf[16 + 4 * t + 2], xv.z, b1);
        b1 = fmaf(wf[16 + 4 * t + 3], xv.w, b1);
    }
    return fmaf(b1, s1, b0 * s0);
}

// `brain_fp4_dot32` against activations held in shared memory.
__device__ __forceinline__ float brain_fp4_dot32_smem(const float* wf, const unsigned int sw16, const float4* xc) {
    const float s0 = brain_qg_e4m3_scale(sw16 & 0xFFu);
    const float s1 = brain_qg_e4m3_scale((sw16 >> 8) & 0xFFu);
    float b0 = 0.0f, b1 = 0.0f;
#pragma unroll
    for (int t = 0; t < 4; ++t) {
        const float4 xv = xc[t];
        b0 = fmaf(wf[4 * t + 0], xv.x, b0);
        b0 = fmaf(wf[4 * t + 1], xv.y, b0);
        b0 = fmaf(wf[4 * t + 2], xv.z, b0);
        b0 = fmaf(wf[4 * t + 3], xv.w, b0);
    }
#pragma unroll
    for (int t = 0; t < 4; ++t) {
        const float4 xv = xc[4 + t];
        b1 = fmaf(wf[16 + 4 * t + 0], xv.x, b1);
        b1 = fmaf(wf[16 + 4 * t + 1], xv.y, b1);
        b1 = fmaf(wf[16 + 4 * t + 2], xv.z, b1);
        b1 = fmaf(wf[16 + 4 * t + 3], xv.w, b1);
    }
    return fmaf(b1, s1, b0 * s0);
}

// The multi-row (2 to 8) NVFP4 build: as `brain_matmul_fp8_gemv_staged`, the activations of a round staged in shared memory.
template <int MR>
__device__ __forceinline__ void brain_matmul_nvfp4_gemv_staged(const unsigned int* params, const float* x, const unsigned int* w, const unsigned int* s, float* y) {
    const unsigned int m = params[0], k = params[1], count = params[2], first = params[3], xr0 = params[4];
    const float global = __uint_as_float(params[5]);
    const unsigned int blk = blockIdx.y * gridDim.x + blockIdx.x;
    const unsigned int col = blk * BRAIN_QG_COLS + threadIdx.x / BRAIN_QG_LANES;
    const unsigned int lane = threadIdx.x % BRAIN_QG_LANES;
    const bool live = col < count;
    const unsigned int chunks = k >> 5;
    const unsigned int row = first + (live ? col : 0u);
    const uint4* wrow = reinterpret_cast<const uint4*>(w) + static_cast<unsigned long long>(row) * chunks;
    const unsigned int* srow = s + static_cast<unsigned long long>(row) * (k >> 6);
    const float4* x4 = reinterpret_cast<const float4*>(x);
    __shared__ float4 xs[MR][BRAIN_QG_KB / 4];

    float acc[MR];
#pragma unroll
    for (int r = 0; r < MR; ++r) { acc[r] = 0.0f; }
    constexpr int UNROLL = 2;
    for (unsigned int kb = 0; kb < k; kb += BRAIN_QG_KB) {
        const unsigned int kn = (k - kb) < BRAIN_QG_KB ? (k - kb) : BRAIN_QG_KB;
        const unsigned int v4 = kn >> 2;
        for (unsigned int i = threadIdx.x; i < m * v4; i += BRAIN_QG_THREADS) {
            const unsigned int r = i / v4, c = i - r * v4;
            xs[r][c] = __ldg(x4 + static_cast<unsigned long long>(xr0 + r) * (k >> 2) + (kb >> 2) + c);
        }
        __syncthreads();
        const unsigned int nch = kn >> 5, cbase = kb >> 5;
        for (unsigned int c0 = lane; c0 < nch; c0 += UNROLL * BRAIN_QG_LANES) {
            uint4 wv[UNROLL];
            unsigned int sw[UNROLL];
#pragma unroll
            for (int u = 0; u < UNROLL; ++u) {
                const unsigned int c = c0 + u * BRAIN_QG_LANES;
                if (live && c < nch) {
                    const unsigned int j = cbase + c;
                    wv[u] = brain_qg_ld_stream(wrow + j);
                    sw[u] = __ldg(srow + (j >> 1)) >> ((j & 1u) * 16u);
                } else {
                    wv[u] = make_uint4(0u, 0u, 0u, 0u);
                    sw[u] = 0u;
                }
            }
#pragma unroll
            for (int u = 0; u < UNROLL; ++u) {
                const unsigned int c = c0 + u * BRAIN_QG_LANES;
                if (c < nch) {
                    float wf[32];
                    brain_fp4_decode32(wv[u], wf);
#pragma unroll
                    for (int r = 0; r < MR; ++r) {
                        if (static_cast<unsigned int>(r) < m) { acc[r] += brain_fp4_dot32_smem(wf, sw[u], &xs[r][8u * c]); }
                    }
                }
            }
        }
        __syncthreads();
    }
    const float scale = 4194304.0f * global;
#pragma unroll
    for (int r = 0; r < MR; ++r) {
        const float total = brain_qg_row_sum(acc[r]) * scale;
        if (static_cast<unsigned int>(r) < m && lane == 0u && live) { y[static_cast<unsigned long long>(r) * count + col] = total; }
    }
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
                float wf[32];
                brain_fp4_decode32(wv[u], wf);
#pragma unroll
                for (int r = 0; r < MR; ++r) {
                    if (static_cast<unsigned int>(r) < m) {
                        const float4* xc = reinterpret_cast<const float4*>(x + static_cast<unsigned long long>(xr0 + r) * k) + 8ull * j;
                        acc[r] += brain_fp4_dot32(wf, sw[u], xc);
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

extern "C" __global__ void __launch_bounds__(BRAIN_QG_THREADS, 2)
brain_matmul_nvfp4_gemv(const unsigned int* params, const float* x, const unsigned int* w, const unsigned int* s, float* y) {
    brain_matmul_nvfp4_gemv_staged<BRAIN_QG_MAXM>(params, x, w, s, y);
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
            if (j < chunks) {
                float wf[32];
                brain_fp4_decode32(wv[u], wf);
                acc += brain_fp4_dot32(wf, sw[u], xrow + 8ull * j);
            }
        }
    }
    const float total = brain_qg_row_sum(acc) * (4194304.0f * (live ? __ldg(gl + expert) : 0.0f));
    if (lane == 0u && live) { out[static_cast<unsigned long long>(slot) * n + col] = total; }
}

// ---- NVFP4 gate + up + SwiGLU ------------------------------------------------------------------------------------------

// The gate and up projections of one slot's expert (two banks of the same shape) and `silu(gate) * up`:
// out[s, c] = silu(G[s, c]) * U[s, c], G and U each the gather kernel's product. One block covers 16 weight rows of one
// slot; the activation row is read once for both.
extern "C" __global__ void __launch_bounds__(BRAIN_QG_THREADS, 4)
brain_moe_nvfp4_gateup_silu(const unsigned int* params, const float* x, const unsigned int* ids, const unsigned int* wg,
                            const unsigned int* sg, const float* gg, const unsigned int* wu, const unsigned int* su,
                            const float* gu, float* out) {
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
    const uint4* grow = reinterpret_cast<const uint4*>(wg) + rowi * chunks;
    const uint4* urow = reinterpret_cast<const uint4*>(wu) + rowi * chunks;
    const unsigned int* gsrow = sg + rowi * (k >> 6);
    const unsigned int* usrow = su + rowi * (k >> 6);
    const float4* xrow = reinterpret_cast<const float4*>(x + static_cast<unsigned long long>(live ? slot / xdiv : 0u) * k);

    float acc_g = 0.0f, acc_u = 0.0f;
    constexpr int UNROLL = 2;
    for (unsigned int j0 = lane; j0 < chunks; j0 += UNROLL * BRAIN_QG_LANES) {
        uint4 wgv[UNROLL], wuv[UNROLL];
        unsigned int sgv[UNROLL], suv[UNROLL];
#pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
            const unsigned int j = j0 + u * BRAIN_QG_LANES;
            if (live && j < chunks) {
                wgv[u] = brain_qg_ld_stream(grow + j);
                wuv[u] = brain_qg_ld_stream(urow + j);
                sgv[u] = __ldg(gsrow + (j >> 1)) >> ((j & 1u) * 16u);
                suv[u] = __ldg(usrow + (j >> 1)) >> ((j & 1u) * 16u);
            } else {
                wgv[u] = make_uint4(0u, 0u, 0u, 0u);
                wuv[u] = make_uint4(0u, 0u, 0u, 0u);
                sgv[u] = 0u;
                suv[u] = 0u;
            }
        }
#pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
            const unsigned int j = j0 + u * BRAIN_QG_LANES;
            if (j < chunks) {
                float wf[32];
                brain_fp4_decode32(wgv[u], wf);
                acc_g += brain_fp4_dot32(wf, sgv[u], xrow + 8ull * j);
                brain_fp4_decode32(wuv[u], wf);
                acc_u += brain_fp4_dot32(wf, suv[u], xrow + 8ull * j);
            }
        }
    }
    const float g = brain_qg_row_sum(acc_g) * (4194304.0f * (live ? __ldg(gg + expert) : 0.0f));
    const float u = brain_qg_row_sum(acc_u) * (4194304.0f * (live ? __ldg(gu + expert) : 0.0f));
    if (lane == 0u && live) { out[static_cast<unsigned long long>(slot) * n + col] = (g / (1.0f + expf(-g))) * u; }
}

// ---- NVFP4 grouped expert GEMM (prefill) ---------------------------------------------------------------------------

#define BRAIN_GR_THREADS 128
#define BRAIN_GR_ROWS 64          // weight rows per block (two threads each)
#define BRAIN_GR_SLOTS 8          // slots per tile of one expert (the router tables' MR)
#define BRAIN_GR_KB 512           // activation columns staged per round

// Slots of one expert, up to eight, against 64 of its weight rows. A tile and a column tile come from the flat block index
// exactly as in `moe_nvfp4_grouped.wgsl` (column tile fastest); the routing tables are `moe_route_scan`'s. Each weight chunk is
// loaded and decoded ONCE and applied to every slot of the tile, with the slots' activations staged in shared memory.
extern "C" __global__ void __launch_bounds__(BRAIN_GR_THREADS, 4)
brain_moe_nvfp4_grouped(const unsigned int* params, const float* x, const unsigned int* tab, const unsigned int* perm,
                        const unsigned int* w, const unsigned int* s, const float* gl, float* out) {
    const unsigned int k = params[1], n = params[2], xdiv = params[3], ne = params[4];
    const unsigned int blk = blockIdx.y * gridDim.x + blockIdx.x;
    const unsigned int col_tiles = (n + BRAIN_GR_ROWS - 1u) / BRAIN_GR_ROWS;
    const unsigned int tile = blk / col_tiles;
    const unsigned int coltile = blk - tile * col_tiles;
    if (tile >= tab[2u * ne + 1u]) { return; }
    unsigned int lo = 0u, hi = ne;
    while (hi - lo > 1u) {
        const unsigned int mid = (lo + hi) / 2u;
        if (tab[ne + 1u + mid] <= tile) { lo = mid; } else { hi = mid; }
    }
    const unsigned int e = lo;
    const unsigned int first = tab[e] + (tile - tab[ne + 1u + e]) * BRAIN_GR_SLOTS;
    const unsigned int rem = tab[e + 1u] - first;
    const unsigned int cnt = rem < BRAIN_GR_SLOTS ? rem : BRAIN_GR_SLOTS;

    __shared__ float4 xs[BRAIN_GR_SLOTS][BRAIN_GR_KB / 4];
    __shared__ unsigned int xrow[BRAIN_GR_SLOTS];
    const unsigned int t = threadIdx.x;
    if (t < BRAIN_GR_SLOTS) { xrow[t] = (t < cnt) ? perm[first + t] / xdiv : 0u; }
    __syncthreads();

    const unsigned int half = t & 1u;
    const unsigned int row = coltile * BRAIN_GR_ROWS + (t >> 1);
    const bool live = row < n;
    const unsigned long long rowi = static_cast<unsigned long long>(e) * n + (live ? row : 0u);
    const unsigned int chunks = k >> 5;
    const uint4* wrow = reinterpret_cast<const uint4*>(w) + rowi * chunks;
    const unsigned int* srow = s + rowi * (k >> 6);
    const float4* x4 = reinterpret_cast<const float4*>(x);

    float acc[BRAIN_GR_SLOTS];
#pragma unroll
    for (int r = 0; r < BRAIN_GR_SLOTS; ++r) { acc[r] = 0.0f; }

    for (unsigned int kb = 0; kb < k; kb += BRAIN_GR_KB) {
        const unsigned int kn = (k - kb) < BRAIN_GR_KB ? (k - kb) : BRAIN_GR_KB;
        const unsigned int v4 = kn >> 2;
        for (unsigned int i = t; i < cnt * v4; i += BRAIN_GR_THREADS) {
            const unsigned int r = i / v4, c = i - r * v4;
            xs[r][c] = __ldg(x4 + static_cast<unsigned long long>(xrow[r]) * (k >> 2) + (kb >> 2) + c);
        }
        __syncthreads();
        const unsigned int nch = kn >> 5;                 // 32-weight chunks in this round
        const unsigned int cbase = kb >> 5;
        constexpr int UNROLL = 4;
        for (unsigned int c0 = half; c0 < nch; c0 += 2u * UNROLL) {
            uint4 wv[UNROLL];
            unsigned int sw[UNROLL];
#pragma unroll
            for (int u = 0; u < UNROLL; ++u) {
                const unsigned int c = c0 + 2u * u;
                if (live && c < nch) {
                    const unsigned int j = cbase + c;
                    wv[u] = brain_qg_ld_stream(wrow + j);
                    sw[u] = __ldg(srow + (j >> 1)) >> ((j & 1u) * 16u);
                } else {
                    wv[u] = make_uint4(0u, 0u, 0u, 0u);
                    sw[u] = 0u;
                }
            }
#pragma unroll
            for (int u = 0; u < UNROLL; ++u) {
                const unsigned int c = c0 + 2u * u;
                if (c < nch) {
                    float wf[32];
                    brain_fp4_decode32(wv[u], wf);
#pragma unroll
                    for (int r = 0; r < BRAIN_GR_SLOTS; ++r) {
                        if (static_cast<unsigned int>(r) < cnt) { acc[r] += brain_fp4_dot32_smem(wf, sw[u], &xs[r][8u * c]); }
                    }
                }
            }
        }
        __syncthreads();
    }
    const float scale = 4194304.0f * __ldg(gl + e);
#pragma unroll
    for (int r = 0; r < BRAIN_GR_SLOTS; ++r) {
        const float total = (acc[r] + __shfl_xor_sync(0xffffffffu, acc[r], 1)) * scale;
        if (half == 0u && live && static_cast<unsigned int>(r) < cnt) { out[static_cast<unsigned long long>(perm[first + r]) * n + row] = total; }
    }
}
