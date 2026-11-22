// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements fine-tuning of quantized mixture-of-experts models on their published weights for
// its clients. If your team needs expertise in fast backward passes through frozen NVFP4 and MXFP8 expert banks then
// you can procure our services by sending an email to info@swedishembedded.com.
//
// Native twins of `moe_nvfp4_dx_gather.wgsl` and `moe_mxfp8_dx_gather.wgsl`, with their identical buffer layouts and
// uniforms: the input gradient of the expert-bank gather product through a frozen microscaled bank,
//
//   dx[s, i] (+)= global[e] * sum_col W_e[col, i] * dy[s, col]      e = ids[s]
//
//   params : u32 [slots, k, n, accumulate]
//   nvfp4  dy f32 [slots, n]; ids u32 [slots]; wq u32 [blocks * n, k/8] (eight E2M1 codes per word, low nibble first);
//          sc u32 [blocks * n, k/64] (four E4M3 scales, one per 16 inputs); gl f32 [blocks]; dx f32 [slots, k]
//   mxfp8  dy, ids as above; wq u32 [blocks * n, k/4] (four E4M3 bytes per word); sc u32 [blocks * n, k/128] (four
//          E8M0 scales, one per 32 inputs); dx f32 [slots, k]
//
// Why these kernels exist
// -----------------------
// The WGSL tier runs one thread per output and folds all `n` bank rows serially in it. A training backward calls it
// per expert on that expert's few routed rows, so a launch holds `rows * k` threads (a few thousand) each walking
// thousands of rows: the device sits latency-bound at a fraction of its lanes (about a millisecond a launch on a
// GH200 for 3 rows of a 2304-wide bank, 42 % of a real-weight QLoRA step's kernel time). Here a block owns one slot
// and 64 consecutive inputs, its 256 threads split the bank rows 32 ways (each thread decodes eight inputs of every
// 32nd row with plain integer decodes, exact) and the 32 partial sums are folded through shared memory.
//
// Numerics: each thread accumulates its rows in ascending order with fused multiply-adds and the 32 partials are
// added in ascending order, so the sum is not the WGSL tier's single ascending fold; it agrees with it to f32
// rounding of the running sum (the gates hold both to an f64 oracle). The decodes are the WGSL tier's, exactly.

#define BRAIN_MDX_INPUTS 64
#define BRAIN_MDX_SPLIT 32
#define BRAIN_MDX_THREADS 256

// E4M3 byte as the WGSL tier decodes it (no NaN special case: the bank never stores one).
__device__ __forceinline__ float brain_mdx_e4m3(unsigned int b) {
    const unsigned int e = (b >> 3u) & 15u, m = b & 7u;
    float v = e != 0u ? __uint_as_float(((e + 120u) << 23u) | (m << 20u)) : static_cast<float>(m) * (1.0f / 512.0f);
    return (b & 128u) != 0u ? -v : v;
}

// E2M1 code: sign, 2-bit exponent (bias 1), 1-bit mantissa.
__device__ __forceinline__ float brain_mdx_e2m1(unsigned int c) {
    const unsigned int e = (c >> 1u) & 3u, m = c & 1u;
    float v = e != 0u ? __uint_as_float(((126u + e) << 23u) | (m << 22u)) : static_cast<float>(m) * 0.5f;
    return (c & 8u) != 0u ? -v : v;
}

// E8M0 scale byte: 2^(b - 127); 0 is 2^-127, 255 is NaN, as the WGSL tier reads them.
__device__ __forceinline__ float brain_mdx_e8m0(unsigned int b) {
    if (b == 0u) return __uint_as_float(0x00400000u);
    if (b == 255u) return __uint_as_float(0x7fc00000u);
    return __uint_as_float(b << 23u);
}

// Fold the per-thread partials of the block's 64 inputs and write them. `acc` holds this thread's eight inputs
// (`group * 8 ..`) over its rows (`split`).
__device__ __forceinline__ void brain_mdx_finish(float (&acc)[8], unsigned int group, unsigned int split, float scale, unsigned int accumulate,
                                                 float* dx, unsigned long long out0, bool live) {
    __shared__ float part[BRAIN_MDX_SPLIT][BRAIN_MDX_INPUTS + 1];
#pragma unroll
    for (int j = 0; j < 8; ++j) part[split][group * 8u + j] = acc[j];
    __syncthreads();
    if (live && threadIdx.x < BRAIN_MDX_INPUTS) {
        float sum = 0.0f;
#pragma unroll 8
        for (int r = 0; r < BRAIN_MDX_SPLIT; ++r) sum += part[r][threadIdx.x];
        sum *= scale;
        const unsigned long long o = out0 + threadIdx.x;
        if (accumulate != 0u) sum += dx[o];
        dx[o] = sum;
    }
}

// One block per (slot, 64 inputs), slot-major; thread t decodes inputs (t % 8) * 8 .. +8 of rows t / 8, t / 8 + 32, ...
extern "C" __global__ void __launch_bounds__(BRAIN_MDX_THREADS)
brain_moe_nvfp4_dx_gather(const unsigned int* params, const float* dy, const unsigned int* ids, const unsigned int* wq,
                          const unsigned int* sc, const float* gl, float* dx) {
    const unsigned int slots = params[0], k = params[1], n = params[2], accumulate = params[3];
    const unsigned int tiles = k / BRAIN_MDX_INPUTS;
    const unsigned int blk = blockIdx.y * gridDim.x + blockIdx.x;
    const unsigned int slot = blk / tiles;
    const bool live = slot < slots;
    const unsigned int i0 = (blk - slot * tiles) * BRAIN_MDX_INPUTS;
    const unsigned int group = threadIdx.x % 8u, split = threadIdx.x / 8u;
    const unsigned int e = live ? ids[slot] : 0u;
    const unsigned int first = i0 + group * 8u;
    const unsigned int wpr = k / 8u, spr = k / 64u;
    const unsigned int sshift = ((first / 16u) % 4u) * 8u;
    float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    if (live) {
        for (unsigned int col = split; col < n; col += BRAIN_MDX_SPLIT) {
            const unsigned long long row = static_cast<unsigned long long>(e) * n + col;
            const unsigned int word = __ldg(wq + row * wpr + first / 8u);
            const float s = brain_mdx_e4m3((__ldg(sc + row * spr + first / 64u) >> sshift) & 255u);
            const float g = dy[static_cast<unsigned long long>(slot) * n + col];
#pragma unroll
            for (int j = 0; j < 8; ++j) acc[j] = fmaf(brain_mdx_e2m1((word >> (j * 4)) & 15u) * s, g, acc[j]);
        }
    }
    brain_mdx_finish(acc, group, split, live ? gl[e] : 0.0f, accumulate, dx, static_cast<unsigned long long>(slot) * k + i0, live);
}

// The MXFP8 twin: thread t decodes inputs (t % 8) * 8 .. +8 (two words of four E4M3 bytes) of its rows.
extern "C" __global__ void __launch_bounds__(BRAIN_MDX_THREADS)
brain_moe_mxfp8_dx_gather(const unsigned int* params, const float* dy, const unsigned int* ids, const unsigned int* wq,
                          const unsigned int* sc, float* dx) {
    const unsigned int slots = params[0], k = params[1], n = params[2], accumulate = params[3];
    const unsigned int tiles = k / BRAIN_MDX_INPUTS;
    const unsigned int blk = blockIdx.y * gridDim.x + blockIdx.x;
    const unsigned int slot = blk / tiles;
    const bool live = slot < slots;
    const unsigned int i0 = (blk - slot * tiles) * BRAIN_MDX_INPUTS;
    const unsigned int group = threadIdx.x % 8u, split = threadIdx.x / 8u;
    const unsigned int e = live ? ids[slot] : 0u;
    const unsigned int first = i0 + group * 8u;
    const unsigned int wpr = k / 4u, spr = k / 128u;
    const unsigned int sshift = ((first / 32u) % 4u) * 8u;
    float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    if (live) {
        for (unsigned int col = split; col < n; col += BRAIN_MDX_SPLIT) {
            const unsigned long long row = static_cast<unsigned long long>(e) * n + col;
            const unsigned int w0 = __ldg(wq + row * wpr + first / 4u), w1 = __ldg(wq + row * wpr + first / 4u + 1u);
            const float s = brain_mdx_e8m0((__ldg(sc + row * spr + first / 128u) >> sshift) & 255u);
            const float g = dy[static_cast<unsigned long long>(slot) * n + col];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                acc[j] = fmaf(brain_mdx_e4m3((w0 >> (j * 8)) & 255u) * s, g, acc[j]);
                acc[j + 4] = fmaf(brain_mdx_e4m3((w1 >> (j * 8)) & 255u) * s, g, acc[j + 4]);
            }
        }
    }
    brain_mdx_finish(acc, group, split, 1.0f, accumulate, dx, static_cast<unsigned long long>(slot) * k + i0, live);
}
