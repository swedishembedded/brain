// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements long-sequence attention training for its
// clients. If your team needs expertise in training transformers at sequence
// lengths where the attention matrix dominates the step, you can procure our
// services by sending an email to info@swedishembedded.com.
//
// The element-wise parts of causal grouped-query attention computed in query
// blocks around tensor-core GEMMs (Attention is computed blockwise: QK^T and
// PV are library GEMMs; these kernels do the softmax, its row statistics and
// the score gradient between them). Scores arrive as fp32 [batch, rows, keys]
// from the library, one batch entry per query head of a KV group; probabilities
// and score gradients leave as packed bf16 for the next GEMM.
//
// Row `i` of a block starting at absolute query position `r0` attends keys
// `0..=r0+i`; the rest of the row is written as zero so the following GEMM can
// read the whole width.
//
// No `__restrict__` anywhere: brain's device buffers alias by design.

#define BRAIN_ATTN_THREADS 256

__device__ __forceinline__ unsigned brain_attn_bf16_bits(float v) {
    const unsigned b = __float_as_uint(v);
    return (b + 0x7fffu + ((b >> 16) & 1u)) >> 16;
}

__device__ __forceinline__ unsigned long long brain_attn_block() { return (unsigned long long)blockIdx.y * gridDim.x + blockIdx.x; }

__device__ __forceinline__ float brain_attn_block_reduce_max(float v, float* red) {
    for (int o = 16; o > 0; o >>= 1) { v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o)); }
    if ((threadIdx.x & 31) == 0) { red[threadIdx.x >> 5] = v; }
    __syncthreads();
    float r = red[0];
    for (int w = 1; w < BRAIN_ATTN_THREADS / 32; ++w) { r = fmaxf(r, red[w]); }
    __syncthreads();
    return r;
}

__device__ __forceinline__ float brain_attn_block_reduce_sum(float v, float* red) {
    for (int o = 16; o > 0; o >>= 1) { v += __shfl_xor_sync(0xffffffffu, v, o); }
    if ((threadIdx.x & 31) == 0) { red[threadIdx.x >> 5] = v; }
    __syncthreads();
    float r = 0.0f;
    for (int w = 0; w < BRAIN_ATTN_THREADS / 32; ++w) { r += red[w]; }
    __syncthreads();
    return r;
}

// Forward: softmax of the scaled, causally masked scores; writes the bf16
// probabilities and the row log-sum-exp (in scaled units).
// params: [g, rows, keys, r0, h0, t, scale (f32 bits)]; one block per (head, row).
// lse is [n_heads, t]: lse[(h0 + head) * t + r0 + row].
extern "C" __global__ void __launch_bounds__(BRAIN_ATTN_THREADS) brain_attn_softmax_fwd(const unsigned int* params, const float* s, unsigned int* p, float* lse) {
    const unsigned rows = params[1], keys = params[2], r0 = params[3], h0 = params[4], t = params[5];
    const float scale = __uint_as_float(params[6]);
    const unsigned long long blk = brain_attn_block();
    if (blk >= (unsigned long long)params[0] * rows) { return; }
    const unsigned head = (unsigned)(blk / rows), row = (unsigned)(blk % rows);
    const float* sr = s + blk * keys;
    const unsigned valid = min(keys, r0 + row + 1u);
    __shared__ float red[BRAIN_ATTN_THREADS / 32];

    float m = __int_as_float((int)0xff800000u);
    for (unsigned j = threadIdx.x; j < valid; j += BRAIN_ATTN_THREADS) { m = fmaxf(m, scale * sr[j]); }
    m = brain_attn_block_reduce_max(m, red);
    float sum = 0.0f;
    for (unsigned j = threadIdx.x; j < valid; j += BRAIN_ATTN_THREADS) { sum += expf(scale * sr[j] - m); }
    sum = brain_attn_block_reduce_sum(sum, red);
    const float l = m + logf(sum);
    if (threadIdx.x == 0) { lse[(unsigned long long)(h0 + head) * t + r0 + row] = l; }

    // Two probabilities to a word; keys is a multiple of 8.
    unsigned short* pr = reinterpret_cast<unsigned short*>(p) + blk * keys;
    for (unsigned j = threadIdx.x; j < keys; j += BRAIN_ATTN_THREADS) {
        pr[j] = (unsigned short)(j < valid ? brain_attn_bf16_bits(expf(scale * sr[j] - l)) : 0u);
    }
}

// Backward: rebuild the probabilities from the scores and the saved log-sum-exp,
// and form the score gradient ds = p * (dp - D) * scale, both as bf16.
// params: [g, rows, keys, r0, h0, t, scale (f32 bits), n_heads];
// dsum is [t, n_heads]: D = dsum[(r0 + row) * n_heads + h0 + head].
extern "C" __global__ void __launch_bounds__(BRAIN_ATTN_THREADS) brain_attn_dscore(const unsigned int* params, const float* s, const float* dp, const float* lse, const float* dsum, unsigned int* p, unsigned int* ds) {
    const unsigned rows = params[1], keys = params[2], r0 = params[3], h0 = params[4], t = params[5], nh = params[7];
    const float scale = __uint_as_float(params[6]);
    const unsigned long long blk = brain_attn_block();
    if (blk >= (unsigned long long)params[0] * rows) { return; }
    const unsigned head = (unsigned)(blk / rows), row = (unsigned)(blk % rows);
    const float l = lse[(unsigned long long)(h0 + head) * t + r0 + row];
    const float d = dsum[(unsigned long long)(r0 + row) * nh + h0 + head];
    const unsigned valid = min(keys, r0 + row + 1u);
    const float* sr = s + blk * keys;
    const float* dr = dp + blk * keys;
    unsigned short* pr = reinterpret_cast<unsigned short*>(p) + blk * keys;
    unsigned short* gr = reinterpret_cast<unsigned short*>(ds) + blk * keys;
    for (unsigned j = threadIdx.x; j < keys; j += BRAIN_ATTN_THREADS) {
        float pv = 0.0f, gv = 0.0f;
        if (j < valid) {
            pv = expf(scale * sr[j] - l);
            gv = pv * (dr[j] - d) * scale;
        }
        pr[j] = (unsigned short)brain_attn_bf16_bits(pv);
        gr[j] = (unsigned short)brain_attn_bf16_bits(gv);
    }
}

// D = rowsum(a * b) over rows of `width` contiguous floats.
// params: [rows, width]; one block per row.
extern "C" __global__ void __launch_bounds__(BRAIN_ATTN_THREADS) brain_attn_rowdot(const unsigned int* params, const float* a, const float* b, float* out) {
    const unsigned long long blk = brain_attn_block();
    const unsigned rows = params[0], width = params[1];
    if (blk >= rows) { return; }
    __shared__ float red[BRAIN_ATTN_THREADS / 32];
    float acc = 0.0f;
    for (unsigned j = threadIdx.x; j < width; j += BRAIN_ATTN_THREADS) { acc += a[blk * width + j] * b[blk * width + j]; }
    acc = brain_attn_block_reduce_sum(acc, red);
    if (threadIdx.x == 0) { out[blk] = acc; }
}
