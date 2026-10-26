// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for latency-bound LLM decode on GPUs for its clients. If your team needs
// expertise in collapsing the chain of tiny kernels between two weight streams into one launch, then you can procure
// our services by sending an email to info@swedishembedded.com.
//
// The MoE router head, the native form of `moe_router_topk.wgsl` with the identical buffers and uniform: softmax over the
// experts' logits, the `top_k` largest probabilities (ties to the lower expert), renormalised by their own sum, and the
// shared expert's sigmoid weight as the last slot.
//
//   params : u32 [rows, n_experts, top_k, has_shared, norm, scale_bits]   n_experts <= 1024, top_k <= 16
//            norm = 1 renormalises the kept probabilities, 0 keeps them raw; scale is the routed scaling factor (f32 bits)
//   logits : [rows, n_experts + has_shared] f32
//   ids    : [rows, top_k + has_shared] u32     weight : [rows, top_k + has_shared] f32
//
// One 256-thread block per row; thread t owns experts t, t + 256, ... The reductions are warp shuffles plus one shared
// word per warp, and the top-k rounds select with the same (value descending, index ascending) order as the reference, so
// the chosen experts are the reference's; the probabilities agree to f32 rounding of the softmax sum (a different
// summation order). The reference spends most of its time in 8 + 16 * 8 barrier-separated tree steps.
//
// No `__restrict__` anywhere, deliberately: brain's device buffers alias by design.

#define BRAIN_RT_THREADS 256
#define BRAIN_RT_WARPS 8
#define BRAIN_RT_PER 4
#define BRAIN_RT_MAXK 16

__device__ __forceinline__ float brain_rt_block_max(float v, float* red) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) { v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, off)); }
    if ((threadIdx.x & 31u) == 0u) { red[threadIdx.x >> 5] = v; }
    __syncthreads();
    float m = red[0];
#pragma unroll
    for (int w = 1; w < BRAIN_RT_WARPS; ++w) { m = fmaxf(m, red[w]); }
    __syncthreads();
    return m;
}

__device__ __forceinline__ float brain_rt_block_sum(float v, float* red) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) { v += __shfl_xor_sync(0xffffffffu, v, off); }
    if ((threadIdx.x & 31u) == 0u) { red[threadIdx.x >> 5] = v; }
    __syncthreads();
    float s = 0.0f;
#pragma unroll
    for (int w = 0; w < BRAIN_RT_WARPS; ++w) { s += red[w]; }
    __syncthreads();
    return s;
}

extern "C" __global__ void __launch_bounds__(BRAIN_RT_THREADS)
brain_moe_router_topk(const unsigned int* params, const float* logits, unsigned int* ids, float* weight) {
    const unsigned int rows = params[0], e_n = params[1], top_k = params[2], has_shared = params[3], norm = params[4];
    const float scale = __uint_as_float(params[5]);
    const unsigned int row = blockIdx.y * gridDim.x + blockIdx.x;
    if (row >= rows) { return; }
    const unsigned int t = threadIdx.x;
    const unsigned long long base = static_cast<unsigned long long>(row) * (e_n + has_shared);
    __shared__ float red_v[BRAIN_RT_WARPS];
    __shared__ unsigned int red_i[BRAIN_RT_WARPS];

    float lv[BRAIN_RT_PER];
    float mx = -3.4e38f;
#pragma unroll
    for (int j = 0; j < BRAIN_RT_PER; ++j) {
        const unsigned int e = t + j * BRAIN_RT_THREADS;
        lv[j] = (e < e_n) ? logits[base + e] : -3.4e38f;
        mx = fmaxf(mx, lv[j]);
    }
    mx = brain_rt_block_max(mx, red_v);

    float pe[BRAIN_RT_PER];
    float part = 0.0f;
#pragma unroll
    for (int j = 0; j < BRAIN_RT_PER; ++j) {
        const unsigned int e = t + j * BRAIN_RT_THREADS;
        pe[j] = (e < e_n) ? expf(lv[j] - mx) : 0.0f;
        part += pe[j];
    }
    const float sm = brain_rt_block_sum(part, red_v);

    float prob[BRAIN_RT_PER];
    bool taken[BRAIN_RT_PER];
#pragma unroll
    for (int j = 0; j < BRAIN_RT_PER; ++j) { prob[j] = pe[j] / sm; taken[j] = false; }

    const unsigned int k = top_k < BRAIN_RT_MAXK ? top_k : BRAIN_RT_MAXK;
    unsigned int sel_i[BRAIN_RT_MAXK];
    float sel_p[BRAIN_RT_MAXK];
    for (unsigned int kk = 0; kk < k; ++kk) {
        float best_v = -1.0f;
        unsigned int best_i = 0xffffffffu;
#pragma unroll
        for (int j = 0; j < BRAIN_RT_PER; ++j) {
            const unsigned int e = t + j * BRAIN_RT_THREADS;
            if (e < e_n && !taken[j] && prob[j] > best_v) { best_v = prob[j]; best_i = e; }
        }
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, best_v, off);
            const unsigned int oi = __shfl_xor_sync(0xffffffffu, best_i, off);
            if (ov > best_v || (ov == best_v && oi < best_i)) { best_v = ov; best_i = oi; }
        }
        if ((t & 31u) == 0u) { red_v[t >> 5] = best_v; red_i[t >> 5] = best_i; }
        __syncthreads();
        float wv = red_v[0];
        unsigned int wi = red_i[0];
#pragma unroll
        for (int w = 1; w < BRAIN_RT_WARPS; ++w) {
            if (red_v[w] > wv || (red_v[w] == wv && red_i[w] < wi)) { wv = red_v[w]; wi = red_i[w]; }
        }
        __syncthreads();
        sel_i[kk] = wi;
        sel_p[kk] = wv;
        if (wi != 0xffffffffu && (wi & (BRAIN_RT_THREADS - 1u)) == t) { taken[wi / BRAIN_RT_THREADS] = true; }
    }

    if (t == 0u) {
        float sel_sum = 0.0f;
        for (unsigned int kk = 0; kk < k; ++kk) { sel_sum += sel_p[kk]; }
        const float inv = (norm != 0u ? 1.0f / fmaxf(sel_sum, 1e-9f) : 1.0f) * scale;
        const unsigned long long out_w = k + has_shared;
        for (unsigned int kk = 0; kk < k; ++kk) {
            ids[row * out_w + kk] = sel_i[kk];
            weight[row * out_w + kk] = sel_p[kk] * inv;
        }
        if (has_shared != 0u) {
            ids[row * out_w + k] = e_n;
            weight[row * out_w + k] = 1.0f / (1.0f + expf(-logits[base + e_n]));
        }
    }
}
