// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for memory-bandwidth-bound LLM
// decode on GPUs for its clients. If your team needs expertise in native
// CUDA kernels that stream bf16 weights at the speed of the memory system
// then you can procure our services by sending an email to
// info@swedishembedded.com.
//
// Float-read GEMV of a dense bf16 weight against one f32 activation row, with the buffer layouts and uniform of
// its WGSL sibling (`matmul_bf16_gemv.wgsl`):
//
//   brain_matmul_bf16_gemv_m1    y[0, c] = sum_k x[xr0, k] * bf16(w[first + c, k])
//
//   params : u32 [m = 1, k, count, first, xr0, rpw, pad, pad]   k % 8 == 0
//   x f32 [.., k]; w u32 [n, k/2] (two bf16 to a word, the even element in the low half); y f32 [m, count]
//
// Why it exists: the library's skinny bf16 product needs the activations cast to bf16 first (a launch of its own) and
// reaches about 2 TB/s on mid-sized decode projections; this kernel keeps the cast out and streams the weight with
// 16-byte loads that do not allocate in L1, eight (one activation row) or four chunk loads in flight per lane, then
// reduces a row's 16 lanes with warp shuffles. Numerics: each lane accumulates its chunks in ascending order with fused
// multiply-adds into f32 and the lanes are folded by a butterfly, so the sum agrees with the WGSL tier to f32 rounding of
// the running sum (the gate holds both to an f64 oracle). The bf16 decode is exact (a shift or a mask).
//
// Alignment: x and w are read through 16-byte vector loads, so their bindings must start 16-byte aligned (a whole
// device buffer does). No `__restrict__`: brain's device buffers alias by design.

#define BRAIN_BG_COLS 8
#define BRAIN_BG_LANES 32
#define BRAIN_BG_THREADS (BRAIN_BG_COLS * BRAIN_BG_LANES)

__device__ __forceinline__ uint4 brain_bg_ld_stream(const uint4* p) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    uint4 v;
    asm("ld.global.nc.L1::no_allocate.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
    return v;
#else
    return __ldg(p);
#endif
}

__device__ __forceinline__ float brain_bg_row_sum(float v) {
#pragma unroll
    for (int off = BRAIN_BG_LANES / 2; off > 0; off >>= 1) { v += __shfl_xor_sync(0xffffffffu, v, off, BRAIN_BG_LANES); }
    return v;
}

template <int MR, int UNROLL>
__device__ __forceinline__ void brain_matmul_bf16_gemv_body(const unsigned int* params, const float* x, const unsigned int* w, float* y) {
    const unsigned int m = params[0], k = params[1], count = params[2], first = params[3], xr0 = params[4];
    const unsigned int blk = blockIdx.y * gridDim.x + blockIdx.x;
    const unsigned int col = blk * BRAIN_BG_COLS + threadIdx.x / BRAIN_BG_LANES;
    const unsigned int lane = threadIdx.x % BRAIN_BG_LANES;
    const bool live = col < count;
    const unsigned int chunks = k >> 3;                                    // 8-weight chunks per row
    const unsigned int row = first + (live ? col : 0u);
    const uint4* wrow = reinterpret_cast<const uint4*>(w) + static_cast<unsigned long long>(row) * chunks;

    float acc[MR];
#pragma unroll
    for (int r = 0; r < MR; ++r) { acc[r] = 0.0f; }

    for (unsigned int j0 = lane; j0 < chunks; j0 += UNROLL * BRAIN_BG_LANES) {
        uint4 wv[UNROLL];
#pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
            const unsigned int j = j0 + u * BRAIN_BG_LANES;
            wv[u] = (live && j < chunks) ? brain_bg_ld_stream(wrow + j) : make_uint4(0u, 0u, 0u, 0u);
        }
#pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
            const unsigned int j = j0 + u * BRAIN_BG_LANES;
            if (j < chunks) {
                const unsigned int words[4] = {wv[u].x, wv[u].y, wv[u].z, wv[u].w};
                float wf[8];
#pragma unroll
                for (int q = 0; q < 4; ++q) {
                    wf[2 * q] = __uint_as_float(words[q] << 16);
                    wf[2 * q + 1] = __uint_as_float(words[q] & 0xFFFF0000u);
                }
#pragma unroll
                for (int r = 0; r < MR; ++r) {
                    if (static_cast<unsigned int>(r) < m) {
                        const float4* xc = reinterpret_cast<const float4*>(x + static_cast<unsigned long long>(xr0 + r) * k) + 2ull * j;
                        const float4 xa = __ldg(xc), xb = __ldg(xc + 1);
                        float a = acc[r];
                        a = fmaf(wf[0], xa.x, a);
                        a = fmaf(wf[1], xa.y, a);
                        a = fmaf(wf[2], xa.z, a);
                        a = fmaf(wf[3], xa.w, a);
                        a = fmaf(wf[4], xb.x, a);
                        a = fmaf(wf[5], xb.y, a);
                        a = fmaf(wf[6], xb.z, a);
                        a = fmaf(wf[7], xb.w, a);
                        acc[r] = a;
                    }
                }
            }
        }
    }
#pragma unroll
    for (int r = 0; r < MR; ++r) {
        const float total = brain_bg_row_sum(acc[r]);
        if (static_cast<unsigned int>(r) < m && lane == 0u && live) { y[static_cast<unsigned long long>(r) * count + col] = total; }
    }
}

extern "C" __global__ void __launch_bounds__(BRAIN_BG_THREADS, 4)
brain_matmul_bf16_gemv_m1(const unsigned int* params, const float* x, const unsigned int* w, float* y) {
    brain_matmul_bf16_gemv_body<1, 10>(params, x, w, y);
}
