// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for long-context prompt processing on GPUs for its clients. If your team needs
// expertise in cutting the memory passes between an attention layer's projections and its attention then you can procure our
// services by sending an email to info@swedishembedded.com.
//
// The QK norm and the rotary embedding of a dense attention layer's query or key heads, four heads of one token a block:
//
//   y   = rmsnorm(x) * w                    (each head of `head_dim` channels, w the head's gain)
//   out = rotate(y)                         (half-split: (y[m], y[m + half]) -> (y0 c - y1 s, y1 c + y0 s), c, s of
//                                            pos * base^(-2m / head_dim), pos the token's position)
//
// The portable `qknorm_rope_fused` runs a workgroup of 64 lanes per (token, head) row, each computing its own angles: an
// 8K-row prefill of 32 query heads is 262144 tiny workgroups and a power, a sine and a cosine per channel pair of every head.
// Here a block carries four heads of one token: the angles depend only on the token's position, so the block computes them
// once for its four heads.
//
//   params    : u32 [rows (tokens * heads), heads, head_dim, eps (f32 bits), base (f32 bits)]   the portable kernel's
//   x         : [rows, head_dim] f32
//   w         : [head_dim] f32
//   positions : [rows / heads] u32
//   out       : [rows, head_dim] f32
//
// `brain_qknorm_rope_qk` runs a layer's query and key heads in one launch (params gain heads_k; xk, wk, outk follow).
//
// BIT-IDENTICAL to `qknorm_rope_fused`: 64 lanes a head, lane t summing the squares of channels t, t + 64, ... in that order,
// the 64 partials folded left in index order, `1 / sqrt(ss / head_dim + eps)`, a channel `(x * inv) * w`, the angle
// `f32(pos) * pow(base, -f32(2m) / f32(head_dim))` and the rotation's two products and one sum each rounded on its own (the
// file is compiled without contraction). head_dim is even and at most 256.
//
// No `__restrict__` anywhere: brain's device buffers alias by design.

#define BRAIN_QKR_THREADS 256
#define BRAIN_QKR_HEADS (BRAIN_QKR_THREADS / 64)  // heads a block runs at once, 64 lanes each
#define BRAIN_QKR_MAX_HALF 128

// Four heads `h0..h0 + 4` of `token` (of `heads` per token), by the whole block.
__device__ __forceinline__ void norm_rope_heads(unsigned long long token, unsigned int h0, unsigned int heads, unsigned int d, float eps, float base,
                                                const float* x, const float* w, const unsigned int* positions, float* out) {
    __shared__ float cos_t[BRAIN_QKR_MAX_HALF], sin_t[BRAIN_QKR_MAX_HALF];
    __shared__ float partial[BRAIN_QKR_HEADS][64];
    __shared__ float inv_s[BRAIN_QKR_HEADS];
    const unsigned int half = d / 2u;

    const unsigned int pos = positions[token];
    for (unsigned int m = threadIdx.x; m < half; m += BRAIN_QKR_THREADS) {
        const float angle = (float)pos * powf(base, -(float)(2u * m) / (float)d);
        cos_t[m] = cosf(angle);
        sin_t[m] = sinf(angle);
    }

    const unsigned int slot = threadIdx.x / 64u, t = threadIdx.x % 64u;
    const unsigned int h = h0 + slot;
    const bool live = h < heads;
    const unsigned long long row = (token * heads + h) * (unsigned long long)d;
    float acc = 0.0f;
    if (live) {
        for (unsigned int c = t; c < d; c += 64u) {
            const float v = x[row + c];
            acc = acc + v * v;
        }
    }
    partial[slot][t] = acc;
    // Publishes the partials and the angle tables.
    __syncthreads();
    if (live && t == 0u) {
        float ss = 0.0f;
        for (int i = 0; i < 64; ++i) { ss = ss + partial[slot][i]; }
        inv_s[slot] = 1.0f / sqrtf(ss / (float)d + eps);
    }
    __syncthreads();
    if (live) {
        const float inv = inv_s[slot];
        for (unsigned int m = t; m < half; m += 64u) {
            const float x0 = x[row + m] * inv * w[m];
            const float x1 = x[row + m + half] * inv * w[m + half];
            const float cs = cos_t[m], sn = sin_t[m];
            out[row + m] = x0 * cs - x1 * sn;
            out[row + m + half] = x1 * cs + x0 * sn;
        }
    }
}

extern "C" __global__ void __launch_bounds__(BRAIN_QKR_THREADS)
brain_qknorm_rope_rows(const unsigned int* params, const float* x, const float* w, const unsigned int* positions, float* out) {
    const unsigned int rows = params[0], heads = params[1], d = params[2];
    const float eps = __uint_as_float(params[3]), base = __uint_as_float(params[4]);
    // A block is one token's group of four heads: a decode step's few tokens still fill the card.
    const unsigned int groups = (heads + BRAIN_QKR_HEADS - 1u) / BRAIN_QKR_HEADS;
    const unsigned long long block = (unsigned long long)blockIdx.y * gridDim.x + blockIdx.x;
    const unsigned long long token = block / groups;
    // The whole block leaves together, before its first barrier.
    if (token * heads >= rows) { return; }
    norm_rope_heads(token, (unsigned int)(block % groups) * BRAIN_QKR_HEADS, heads, d, eps, base, x, w, positions, out);
}

// The query and the key heads of the same tokens in one launch: per token, the query's groups of four heads, then the
// key's. params [tokens * heads_q, heads_q, head_dim, eps, base, heads_k]; each tensor its own x, w and out.
extern "C" __global__ void __launch_bounds__(BRAIN_QKR_THREADS)
brain_qknorm_rope_qk(const unsigned int* params, const float* xq, const float* wq, const unsigned int* positions, float* outq, const float* xk, const float* wk, float* outk) {
    const unsigned int rows = params[0], heads_q = params[1], d = params[2], heads_k = params[5];
    const float eps = __uint_as_float(params[3]), base = __uint_as_float(params[4]);
    const unsigned int gq = (heads_q + BRAIN_QKR_HEADS - 1u) / BRAIN_QKR_HEADS, gk = (heads_k + BRAIN_QKR_HEADS - 1u) / BRAIN_QKR_HEADS;
    const unsigned long long block = (unsigned long long)blockIdx.y * gridDim.x + blockIdx.x;
    const unsigned long long token = block / (gq + gk);
    const unsigned int g = (unsigned int)(block % (gq + gk));
    if (token * heads_q >= rows) { return; }
    if (g < gq) {
        norm_rope_heads(token, g * BRAIN_QKR_HEADS, heads_q, d, eps, base, xq, wq, positions, outq);
    } else {
        norm_rope_heads(token, (g - gq) * BRAIN_QKR_HEADS, heads_k, d, eps, base, xk, wk, positions, outk);
    }
}
