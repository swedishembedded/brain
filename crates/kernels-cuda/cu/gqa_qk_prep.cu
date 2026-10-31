// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for long-context prompt processing on GPUs for its clients. If your team needs
// expertise in cutting the memory passes between an attention layer's projections and its attention then you can procure our
// services by sending an email to info@swedishembedded.com.
//
// The front half of a gated grouped-query attention layer over a block of prompt rows, in one launch: for every (row, head)
//
//   query head:  x = the query half of the head's [query | gate] block of the projection output
//   key head:    x = the head's `hd` values of the key projection
//   y   = rmsnorm(x) * w                    (the QK norm, w the query's or the key's gain)
//   out = rotate(y)                         (the first 2 * half channels, partial rotary: (y1, y2) -> (y1 c - y2 s, y2 c + y1 s))
//
// It replaces `concat_split` (the query half, one copy), `rmsnorm_rows` and `rope2d_partial` for the query and for the key:
// five launches that move the query rows five times, here read once and written once. The gate half is left where the
// projection wrote it (`gate_mul_quant_e4m3_128` reads it in place).
//
//   params : u32 [rows, n_heads, n_kv_heads, head_dim (256), half, eps (f32 bits), tmod]
//   q_full : [rows, n_heads * 2 * hd] f32   the query projection: per head hd query values then hd gate values
//   k      : [rows, n_kv_heads * hd] f32    the key projection
//   qw, kw : [hd] f32                       the norm gains
//   cos, sin : [tmod, half] f32             the rotation tables; row r uses table row r % tmod
//   q_out  : [rows, n_heads * hd] f32       normalised and rotated query
//   k_out  : [rows, n_kv_heads * hd] f32    normalised and rotated key
//
// BIT-IDENTICAL to the chain it replaces. A head is 64 threads, as `rmsnorm_rows` has it: thread t sums the squares of channels
// t, t + 64, t + 128, t + 192 in that order, the 64 partials are folded in order by every thread, and a channel is
// `(x * inv) * w`; the rotation reads those rounded values. Every operation is explicit round-to-nearest (the chain is
// compiled without contraction). A block is four heads (256 threads) so each SM has bytes in flight.
//
// No `__restrict__` anywhere: brain's device buffers alias by design.

#define BRAIN_QKP_HD 256
#define BRAIN_QKP_HEADS 4  // heads a block carries

extern "C" __global__ void __launch_bounds__(BRAIN_QKP_HEADS * 64)
brain_gqa_qk_prep(const unsigned int* params, const float* q_full, const float* k, const float* qw, const float* kw, const float* cos_t, const float* sin_t, float* q_out, float* k_out) {
    const unsigned long long rows = params[0];
    const unsigned int n_heads = params[1], n_kv = params[2], half = params[4];
    const float eps = __uint_as_float(params[5]);
    const unsigned int tmod = params[6];
    const unsigned int per_row = n_heads + n_kv;
    const unsigned int t = threadIdx.x & 63u, h = threadIdx.x >> 6;
    const unsigned long long inst = (static_cast<unsigned long long>(blockIdx.y) * gridDim.x + blockIdx.x) * BRAIN_QKP_HEADS + h;
    // A head past the end still takes part in the barriers.
    const bool live = inst < rows * per_row;
    const unsigned long long row = live ? inst / per_row : 0;
    const unsigned int slot = live ? static_cast<unsigned int>(inst % per_row) : 0u;
    const bool is_q = slot < n_heads;
    const float* x = is_q ? q_full + row * (2ull * n_heads * BRAIN_QKP_HD) + static_cast<unsigned long long>(slot) * 2ull * BRAIN_QKP_HD
                          : k + row * (static_cast<unsigned long long>(n_kv) * BRAIN_QKP_HD) + static_cast<unsigned long long>(slot - n_heads) * BRAIN_QKP_HD;
    const float* w = is_q ? qw : kw;
    float* out = is_q ? q_out + row * (static_cast<unsigned long long>(n_heads) * BRAIN_QKP_HD) + static_cast<unsigned long long>(slot) * BRAIN_QKP_HD
                      : k_out + row * (static_cast<unsigned long long>(n_kv) * BRAIN_QKP_HD) + static_cast<unsigned long long>(slot - n_heads) * BRAIN_QKP_HD;

    __shared__ float partial[BRAIN_QKP_HEADS][64];
    __shared__ float normed[BRAIN_QKP_HEADS][BRAIN_QKP_HD];

    float v[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) { v[j] = live ? x[t + 64u * j] : 0.0f; }
    float acc = 0.0f;
#pragma unroll
    for (int j = 0; j < 4; ++j) { acc = __fadd_rn(acc, __fmul_rn(v[j], v[j])); }
    partial[h][t] = acc;
    __syncthreads();
    float ss = 0.0f;
#pragma unroll
    for (int i = 0; i < 64; ++i) { ss = __fadd_rn(ss, partial[h][i]); }
    const float inv = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(__fdiv_rn(ss, static_cast<float>(BRAIN_QKP_HD)), eps)));
#pragma unroll
    for (int j = 0; j < 4; ++j) { normed[h][t + 64u * j] = __fmul_rn(__fmul_rn(v[j], inv), live ? w[t + 64u * j] : 0.0f); }
    __syncthreads();
    if (!live) { return; }
    const unsigned long long table = (row % tmod) * static_cast<unsigned long long>(half);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const unsigned int c = t + 64u * j;
        float r = normed[h][c];
        if (c < half) {
            r = __fsub_rn(__fmul_rn(r, cos_t[table + c]), __fmul_rn(normed[h][c + half], sin_t[table + c]));
        } else if (c < 2u * half) {
            const unsigned int d = c - half;
            r = __fadd_rn(__fmul_rn(r, cos_t[table + d]), __fmul_rn(normed[h][d], sin_t[table + d]));
        }
        out[c] = r;
    }
}
