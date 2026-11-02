// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for batched decode throughput on 4-bit language-model releases for its clients. If
// your team needs expertise in feeding FP8 tensor cores from NVFP4 checkpoints for the few rows of a decode batch, then you can
// procure our services by sending an email to info@swedishembedded.com.
//
// NVFP4 weights against up to 8, 16 or 32 E4M3 activation rows on the FP8 `mma.sync` tensor cores, the codes decoded in registers:
//
//   out[a, c] = global * sx[a] * sum_k e4m3(xq[a, k]) * w(c, k)       w(c, k) = RN_e4m3( e2m1(code) * e4m3(scale) / 8 ) * 8
//
// The weight rows are the 16-row dimension of `mma.m16n8k32.e4m3` and the activation rows its 8-column dimension, as in the bf16
// `matmul_nvfp4_mma*` builds (`fp8_nvfp4_gemv.cu`); the decode is the wgmma GEMM's (`nvfp4_e4m3_decode.cuh`: a table of 256 entries a
// scale byte, four byte permutes for eight codes), so a weight costs about 1.4 instructions to feed the tensor core, where the bf16
// build's unpack and scale multiply cost four times that. The activations carry one scale per row (`quant_e4m3_row`).
//
// The K order is permuted as in the bf16 build: a lane of a quad owns a whole 32-weight chunk of its row (one 16-byte load, two block
// scales); an MMA step takes the 8 codes of one word of the chunk, four to each half of the k32 step, and the activation fragment of the
// same step is the 8 bytes `x[a, 128 grp + 32 t + 8 s .. + 7]`, one 8-byte load. A permutation of the contraction index leaves the sum unchanged.
//
// Parameters: u32 [m, k, n, global_bits], k a multiple of 128, m at most the build's rows. Bindings: params, xq (E4M3 activations, four to a
// word, [m, k/4]), w (E2M1 codes, eight to a word, [n, k/8]), s (E4M3 block scales, four to a word, [n, k/64]), sx (f32 activation scale per
// row, [m]), out (f32 [m, n]). 512 threads: two row-warps of RT 16-row weight tiles by eight K-slices that meet in shared memory.

__device__ __forceinline__ void brain_mmae_mma(float* d, const unsigned int a0, const unsigned int a1, const unsigned int a2, const unsigned int a3, const unsigned int b0, const unsigned int b1) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint4 brain_mmae_ld_stream(const uint4* p) {
    uint4 v;
    asm("ld.global.nc.L1::no_allocate.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
    return v;
}

template <int NT, int RT, int KS>
__device__ __forceinline__ void brain_matmul_nvfp4_mmae_body(const unsigned int* params, const unsigned int* xq, const unsigned int* w, const unsigned int* s, const float* sx, float* y) {
    constexpr int ROWS = 2 * RT * 16;  // weight rows a block covers
    constexpr int THREADS = 64 * KS;   // two row-warps by KS K-slices
    constexpr int MAXM = 8 * NT;
    const unsigned int m = params[0], k = params[1], count = params[2];
    const float scale_out = __uint_as_float(params[3]) * 8.0f;
    const unsigned int warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    const unsigned int rw = warp & 1u, ks = warp >> 1;  // row-warp, K-slice
    const unsigned int g = lane >> 2, t = lane & 3u;
    const unsigned int chunks = k >> 5;                 // 32-weight chunks per row
    const unsigned int groups = k >> 7;                 // 128-weight groups per row
    const unsigned int row0 = blockIdx.x * ROWS + rw * (RT * 16u);

    __shared__ float accs[ROWS][MAXM];
    __shared__ uint2 lut[256];
    for (unsigned int i = threadIdx.x; i < ROWS * MAXM; i += THREADS) { (&accs[0][0])[i] = 0.0f; }
    for (unsigned int i = threadIdx.x; i < 256u; i += THREADS) { lut[i] = brain_nw_scale_table(i); }

    // The 2 RT weight rows of the lane: g and g + 8 of each 16-row tile.
    const uint4* wrow[2 * RT];
    const unsigned int* srow[2 * RT];
#pragma unroll
    for (int r = 0; r < 2 * RT; ++r) {
        const unsigned int row = row0 + (r >> 1) * 16u + g + (r & 1) * 8u;
        const unsigned long long src = row < count ? row : count - 1u;
        wrow[r] = reinterpret_cast<const uint4*>(w) + src * chunks;
        srow[r] = s + src * (k >> 6);
    }
    const uint2* xq2 = reinterpret_cast<const uint2*>(xq);
    const unsigned int xstride = k >> 3;  // 8-byte units in a row of activations

    float acc[RT][NT][4];
#pragma unroll
    for (int ti = 0; ti < RT; ++ti) {
#pragma unroll
        for (int j = 0; j < NT; ++j) {
#pragma unroll
            for (int c = 0; c < 4; ++c) { acc[ti][j][c] = 0.0f; }
        }
    }
    // Row of this lane in each 8-row activation tile (clamped: the surplus rows of the last tile are never stored).
    unsigned int xrow[NT];
#pragma unroll
    for (int j = 0; j < NT; ++j) { xrow[j] = (8u * j + g < m ? 8u * j + g : m - 1u) * xstride; }
    __syncthreads();

    for (unsigned int grp = ks; grp < groups; grp += KS) {
        const unsigned int c = 4u * grp + t;  // this lane's chunk of the group
        uint4 wv[2 * RT];
        uint2 tb[2 * RT][2];
#pragma unroll
        for (int r = 0; r < 2 * RT; ++r) {
            wv[r] = brain_mmae_ld_stream(wrow[r] + c);
            const unsigned int sw = __ldg(srow[r] + (c >> 1)) >> ((c & 1u) * 16u);
            tb[r][0] = lut[sw & 0xFFu];
            tb[r][1] = lut[(sw >> 8) & 0xFFu];
        }
#pragma unroll
        for (int st = 0; st < 4; ++st) {
            // Word `st` of every chunk (8 codes): four weights to each half of the k32 step.
            unsigned int lo[2 * RT], hi[2 * RT];
#pragma unroll
            for (int r = 0; r < 2 * RT; ++r) {
                const unsigned int word = st == 0 ? wv[r].x : st == 1 ? wv[r].y : st == 2 ? wv[r].z : wv[r].w;
                brain_nw_decode8(word, tb[r][st >> 1], lo[r], hi[r]);
            }
#pragma unroll
            for (int j = 0; j < NT; ++j) {
                const uint2 bv = __ldg(xq2 + xrow[j] + 16u * grp + 4u * t + st);  // x[row][128 grp + 32 t + 8 st .. + 7]
#pragma unroll
                for (int ti = 0; ti < RT; ++ti) { brain_mmae_mma(acc[ti][j], lo[2 * ti], lo[2 * ti + 1], hi[2 * ti], hi[2 * ti + 1], bv.x, bv.y); }
            }
        }
    }
    __syncthreads();
    // D fragment: c0, c1 (row g, columns 2 t, 2 t + 1), c2, c3 (row g + 8): the K-slices' partial sums add in shared memory.
#pragma unroll
    for (int ti = 0; ti < RT; ++ti) {
#pragma unroll
        for (int j = 0; j < NT; ++j) {
#pragma unroll
            for (int c = 0; c < 4; ++c) {
                const unsigned int r = rw * (RT * 16u) + ti * 16u + g + (c >> 1) * 8u, a = 8u * j + 2u * t + (c & 1);
                if (a < m) { atomicAdd(&accs[r][a], acc[ti][j][c]); }
            }
        }
    }
    __syncthreads();
    for (unsigned int i = threadIdx.x; i < static_cast<unsigned int>(ROWS) * m; i += THREADS) {
        const unsigned int r = i % ROWS, a = i / ROWS;
        const unsigned int row = blockIdx.x * ROWS + r;
        if (row < count) { y[static_cast<unsigned long long>(a) * count + row] = accs[r][a] * sx[a] * scale_out; }
    }
}

// One entry per number of 8-row activation tiles: 8 rows (RT 2), 16 and 32.
extern "C" __global__ void __launch_bounds__(512, 1)
brain_matmul_nvfp4_mmae8(const unsigned int* params, const unsigned int* xq, const unsigned int* w, const unsigned int* s, const float* sx, float* y) {
    brain_matmul_nvfp4_mmae_body<1, 2, 8>(params, xq, w, s, sx, y);
}
extern "C" __global__ void __launch_bounds__(512, 1)
brain_matmul_nvfp4_mmae16(const unsigned int* params, const unsigned int* xq, const unsigned int* w, const unsigned int* s, const float* sx, float* y) {
    brain_matmul_nvfp4_mmae_body<2, 2, 8>(params, xq, w, s, sx, y);
}
extern "C" __global__ void __launch_bounds__(512, 1)
brain_matmul_nvfp4_mmae32(const unsigned int* params, const unsigned int* xq, const unsigned int* w, const unsigned int* s, const float* sx, float* y) {
    brain_matmul_nvfp4_mmae_body<4, 2, 8>(params, xq, w, s, sx, y);
}
