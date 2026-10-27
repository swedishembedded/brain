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
__device__ __forceinline__ void brain_matmul_bf16_gemv_body(unsigned int m, unsigned int k, unsigned int count, unsigned int first, unsigned int xr0, unsigned int blk, const float* x, const unsigned int* w, float* y) {
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
    brain_matmul_bf16_gemv_body<1, 10>(params[0], params[1], params[2], params[3], params[4], blockIdx.y * gridDim.x + blockIdx.x, x, w, y);
}

// ---- several activation rows --------------------------------------------------------------------------------------
//
// Two to eight rows: the activation rows of a round of `BRAIN_BG_KB` columns are staged in shared memory once per block,
// and a warp walks FOUR weight rows over the same chunk so that every activation value it reads from shared memory is
// used against four weights (the staging traffic stays a fraction of the weight stream). Each lane issues all of its
// chunk loads for a round (four chunks of four rows) before consuming any. A warp reduces its 4 x MR accumulators with
// shuffles at the end.

#define BRAIN_BG_MR_ROWS 4                                   // weight rows per warp
#define BRAIN_BG_MR_BLOCK_ROWS (8 * BRAIN_BG_MR_ROWS)        // weight rows per block (eight warps)
#define BRAIN_BG_KB 1024                                     // activation columns staged per round
#define BRAIN_BG_MR_CHUNKS (BRAIN_BG_KB / 8 / 32)            // chunks per lane per round

template <int MR>
__device__ __forceinline__ void brain_matmul_bf16_gemv_staged(unsigned int m, unsigned int k, unsigned int count, unsigned int first, unsigned int xr0, unsigned int blk, float4 (*xs)[BRAIN_BG_KB / 4], const float* x, const unsigned int* w, float* y) {
    const unsigned int warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    const unsigned int col0 = blk * BRAIN_BG_MR_BLOCK_ROWS + warp * BRAIN_BG_MR_ROWS;
    const unsigned int chunks = k >> 3;
    const uint4* wrow[BRAIN_BG_MR_ROWS];
    bool live[BRAIN_BG_MR_ROWS];
#pragma unroll
    for (int q = 0; q < BRAIN_BG_MR_ROWS; ++q) {
        live[q] = col0 + q < count;
        wrow[q] = reinterpret_cast<const uint4*>(w) + static_cast<unsigned long long>(first + (live[q] ? col0 + q : 0u)) * chunks;
    }
    const float4* x4 = reinterpret_cast<const float4*>(x);

    float acc[BRAIN_BG_MR_ROWS][MR];
#pragma unroll
    for (int q = 0; q < BRAIN_BG_MR_ROWS; ++q) {
#pragma unroll
        for (int r = 0; r < MR; ++r) { acc[q][r] = 0.0f; }
    }
    for (unsigned int kb = 0; kb < k; kb += BRAIN_BG_KB) {
        const unsigned int kn = (k - kb) < BRAIN_BG_KB ? (k - kb) : BRAIN_BG_KB;
        const unsigned int v4 = kn >> 2;
        __syncthreads();
        for (unsigned int i = threadIdx.x; i < m * v4; i += 256u) {
            const unsigned int r = i / v4, c = i - r * v4;
            xs[r][c] = __ldg(x4 + static_cast<unsigned long long>(xr0 + r) * (k >> 2) + (kb >> 2) + c);
        }
        __syncthreads();
        const unsigned int nch = kn >> 3, cbase = kb >> 3;
        uint4 wv[BRAIN_BG_MR_CHUNKS][BRAIN_BG_MR_ROWS];
#pragma unroll
        for (int c = 0; c < BRAIN_BG_MR_CHUNKS; ++c) {
            const unsigned int j = lane + 32u * c;
#pragma unroll
            for (int q = 0; q < BRAIN_BG_MR_ROWS; ++q) {
                wv[c][q] = (live[q] && j < nch) ? brain_bg_ld_stream(wrow[q] + cbase + j) : make_uint4(0u, 0u, 0u, 0u);
            }
        }
#pragma unroll
        for (int c = 0; c < BRAIN_BG_MR_CHUNKS; ++c) {
            const unsigned int j = lane + 32u * c;
            if (j < nch) {
                float wf[BRAIN_BG_MR_ROWS][8];
#pragma unroll
                for (int q = 0; q < BRAIN_BG_MR_ROWS; ++q) {
                    const unsigned int words[4] = {wv[c][q].x, wv[c][q].y, wv[c][q].z, wv[c][q].w};
#pragma unroll
                    for (int h = 0; h < 4; ++h) {
                        wf[q][2 * h] = __uint_as_float(words[h] << 16);
                        wf[q][2 * h + 1] = __uint_as_float(words[h] & 0xFFFF0000u);
                    }
                }
#pragma unroll
                for (int r = 0; r < MR; ++r) {
                    if (static_cast<unsigned int>(r) < m) {
                        const float4 xa = xs[r][2u * j], xb = xs[r][2u * j + 1u];
#pragma unroll
                        for (int q = 0; q < BRAIN_BG_MR_ROWS; ++q) {
                            float a = acc[q][r];
                            a = fmaf(wf[q][0], xa.x, a);
                            a = fmaf(wf[q][1], xa.y, a);
                            a = fmaf(wf[q][2], xa.z, a);
                            a = fmaf(wf[q][3], xa.w, a);
                            a = fmaf(wf[q][4], xb.x, a);
                            a = fmaf(wf[q][5], xb.y, a);
                            a = fmaf(wf[q][6], xb.z, a);
                            a = fmaf(wf[q][7], xb.w, a);
                            acc[q][r] = a;
                        }
                    }
                }
            }
        }
    }
#pragma unroll
    for (int q = 0; q < BRAIN_BG_MR_ROWS; ++q) {
#pragma unroll
        for (int r = 0; r < MR; ++r) {
            float v = acc[q][r];
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) { v += __shfl_xor_sync(0xffffffffu, v, off); }
            if (static_cast<unsigned int>(r) < m && lane == 0u && live[q]) { y[static_cast<unsigned long long>(r) * count + col0 + q] = v; }
        }
    }
}

extern "C" __global__ void __launch_bounds__(256, 2)
brain_matmul_bf16_gemv_m2(const unsigned int* params, const float* x, const unsigned int* w, float* y) {
    __shared__ float4 xs[2][BRAIN_BG_KB / 4];
    brain_matmul_bf16_gemv_staged<2>(params[0], params[1], params[2], params[3], params[4], blockIdx.y * gridDim.x + blockIdx.x, xs, x, w, y);
}

extern "C" __global__ void __launch_bounds__(256, 2)
brain_matmul_bf16_gemv_m4(const unsigned int* params, const float* x, const unsigned int* w, float* y) {
    __shared__ float4 xs[4][BRAIN_BG_KB / 4];
    brain_matmul_bf16_gemv_staged<4>(params[0], params[1], params[2], params[3], params[4], blockIdx.y * gridDim.x + blockIdx.x, xs, x, w, y);
}

extern "C" __global__ void __launch_bounds__(256, 2)
brain_matmul_bf16_gemv_m8(const unsigned int* params, const float* x, const unsigned int* w, float* y) {
    __shared__ float4 xs[8][BRAIN_BG_KB / 4];
    brain_matmul_bf16_gemv_staged<8>(params[0], params[1], params[2], params[3], params[4], blockIdx.y * gridDim.x + blockIdx.x, xs, x, w, y);
}

// ---- up to four matrices that read one activation, in one launch ---------------------------------------------------
//
//   brain_matmul_bf16_gemv_multi    params : u32 [m, k, n0, n1, n2, n3]   n_i = 0 leaves set i unused; m <= 8
//   x, then for each set its w_i and y_i: exactly the buffers a single bf16 GEMV takes (row window 0, activation row 0).
//
// A decode layer multiplies the same activation by several projections (a GDN layer's qkv, b, a and z, a GQA layer's q, k
// and v, the SwiGLU's gate and up), and each as its own launch has a last wave that is mostly empty: the 32-row b and a
// projections run a handful of blocks for a few microseconds of latency each. Merged, the blocks of all the sets fill the
// card together. A block runs the single-matrix body on its own set, so every output is bit-identical to the separate
// launches. The blocks of set 0 come first, then set 1, and so on; a block covers BRAIN_BG_COLS weight rows at one
// activation row and BRAIN_BG_MR_BLOCK_ROWS at several.

extern "C" __global__ void __launch_bounds__(256, 2)
brain_matmul_bf16_gemv_multi(const unsigned int* params, const float* x,
                             const unsigned int* w0, float* y0, const unsigned int* w1, float* y1,
                             const unsigned int* w2, float* y2, const unsigned int* w3, float* y3) {
    const unsigned int m = params[0], k = params[1];
    const unsigned int cols = (m == 1u) ? BRAIN_BG_COLS : BRAIN_BG_MR_BLOCK_ROWS;
    unsigned int local = blockIdx.y * gridDim.x + blockIdx.x;
    const unsigned int* w = w0;
    float* y = y0;
    unsigned int n = 0u;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const unsigned int ni = params[2 + i];
        const unsigned int blocks = (ni + cols - 1u) / cols;
        if (n == 0u && local < blocks) {
            n = ni;
            w = (i == 0) ? w0 : (i == 1) ? w1 : (i == 2) ? w2 : w3;
            y = (i == 0) ? y0 : (i == 1) ? y1 : (i == 2) ? y2 : y3;
        } else if (n == 0u) {
            local -= blocks;
        }
    }
    if (n == 0u) { return; }  // past the last set's blocks; block-uniform
    __shared__ float4 xs[8][BRAIN_BG_KB / 4];   // the largest build's staging, shared by the builds below
    if (m == 1u) {
        brain_matmul_bf16_gemv_body<1, 10>(1u, k, n, 0u, 0u, local, x, w, y);
    } else if (m <= 2u) {
        brain_matmul_bf16_gemv_staged<2>(m, k, n, 0u, 0u, local, xs, x, w, y);
    } else if (m <= 4u) {
        brain_matmul_bf16_gemv_staged<4>(m, k, n, 0u, 0u, local, xs, x, w, y);
    } else {
        brain_matmul_bf16_gemv_staged<8>(m, k, n, 0u, 0u, local, xs, x, w, y);
    }
}
