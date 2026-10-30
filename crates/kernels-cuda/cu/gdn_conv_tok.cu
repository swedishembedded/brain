// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements memory-bound training kernels for linear-
// attention language models for its clients. If your team needs expertise in
// collapsing the layout and activation glue around a depthwise convolution
// into one pass over the activations, then you can procure our services by
// sending an email to info@swedishembedded.com.
//
// The causal depthwise convolution of Gated DeltaNet's mixer with its SiLU and
// the query / key / value split, directly on the token-major rows the
// projections produce. The portable path moves the activations through a
// channel-major copy: a token-major to channel-major transpose, the generic
// convolution, SiLU, a transpose back and three row splits (and the mirror
// image, plus two row concatenations, in the backward) - eight passes over a
// [tokens, channels] tensor in the forward and seven in the backward. Here one
// launch does the forward and one the backward; a thread owns a channel and
// walks a run of rows with the convolution window in registers.
//
//   pre[t]  = sum_k w[k] * x[t + k - (K-1)]       x[q < 0] = history row (ctx) or 0
//   act[t]  = silu(pre[t])                        split into query | key | value by channel
//
//   d_pre[t] = silu'(pre[t]) * d_act[t]           d_act = d_query | d_key | d_value by channel
//   d_x[s]   = sum_k w[k] * d_pre[s + (K-1) - k]  s in [-ctx, T): rows s < 0 are the history's gradient
//
// The weight gradient is NOT formed here: a build that trains the convolution weight keeps
// the portable path.
//
// Contract (u32 params, then pointers; every tensor is token-major, `rows` = batch * seq_len, C = conv_dim =
// 2 * key_dim + value_dim, w is [C, K] as the generic convolution's [C, 1, K] weight):
//   forward  `gdn_conv_silu_tok`
//     params : [rows, C, seq_len, key_dim, value_dim, has_hist, save_pre]
//     read   : x [rows, C]; w [C, K]; hist [C, K-1] channel-major, oldest first (read when has_hist != 0;
//              then batch must be 1 and the history precedes row 0)
//     write  : query, key [rows, key_dim]; value [rows, value_dim]; pre [rows, C] (when save_pre != 0)
//   history  `gdn_conv_hist_tok` (launch AFTER the forward: it overwrites the history it read there)
//     params : [rows, C]
//     read   : x [rows, C]    read/write: hist [C, K-1]  (becomes the last K-1 rows of [hist | x])
//   backward `gdn_conv_silu_bwd_tok`
//     params : [rows, C, seq_len, key_dim, value_dim, ctx]  (ctx = 0 or K-1)
//     read   : d_query, d_key [rows, key_dim]; d_value [rows, value_dim]; pre [rows, C]; w [C, K]
//     write  : d_x [rows, C]; d_hist [ctx, C] token-major (when ctx != 0)
// The kernel size is fixed at K = 4 (the model's); the provider declines any other. Every output element is overwritten.
//
// No `__restrict__`: brain's device buffers alias by design.

#define THREADS 128
#define ROWS_PER_THREAD 32
#define K 4                   // taps of the convolution (linear_conv_kernel_dim); the provider declines any other

__device__ __forceinline__ float silu_fwd(float v) { return v / (1.0f + expf(-v)); }

__device__ __forceinline__ float silu_bwd(float v, float dy) {
    const float s = 1.0f / (1.0f + expf(-v));
    return dy * (s * (1.0f + v * (1.0f - s)));
}

extern "C" __global__ void __launch_bounds__(THREADS)
brain_gdn_conv_silu_tok(const unsigned int* params, const float* x, const float* w, const float* hist, float* query, float* key, float* value, float* pre_out) {
    const unsigned int rows = params[0], C = params[1], T = params[2], key_dim = params[3], value_dim = params[4];
    const bool has_hist = params[5] != 0, save_pre = params[6] != 0;
    // The grid is one linear block index (folded into two dimensions when long): channel tiles fastest.
    const unsigned int blk = blockIdx.y * gridDim.x + blockIdx.x, tiles = (C + THREADS - 1) / THREADS;
    const unsigned int c = (blk % tiles) * THREADS + threadIdx.x;
    const unsigned int r0 = (blk / tiles) * ROWS_PER_THREAD;
    if (c >= C || r0 >= rows) { return; }
    const unsigned int r1 = min(r0 + ROWS_PER_THREAD, rows);

    float wk[K];
#pragma unroll
    for (unsigned int k = 0; k < K; ++k) { wk[k] = w[(unsigned long long)c * K + k]; }

    // The K inputs feeding output row r: x[r + k - (K-1)] within r's sequence, the history or zero before it.
    float win[K];
    auto gather = [&](unsigned int r) {
        const unsigned int seq0 = r - r % T, p = r % T;
        for (unsigned int k = 0; k < K; ++k) {
            const int q = (int)p + (int)k - (int)(K - 1);
            win[k] = q >= 0 ? x[(unsigned long long)(seq0 + q) * C + c] : (has_hist ? hist[(unsigned long long)c * (K - 1) + (q + (int)(K - 1))] : 0.0f);
        }
    };
    for (unsigned int r = r0; r < r1; ++r) {
        if (r == r0 || r % T == 0) {
            gather(r);
        } else {
            for (unsigned int k = 0; k + 1 < K; ++k) { win[k] = win[k + 1]; }
            win[K - 1] = x[(unsigned long long)r * C + c];
        }
        float acc = 0.0f;
        for (unsigned int k = 0; k < K; ++k) { acc = acc + win[k] * wk[k]; }
        if (save_pre) { pre_out[(unsigned long long)r * C + c] = acc; }
        const float act = silu_fwd(acc);
        if (c < key_dim) {
            query[(unsigned long long)r * key_dim + c] = act;
        } else if (c < 2 * key_dim) {
            key[(unsigned long long)r * key_dim + (c - key_dim)] = act;
        } else {
            value[(unsigned long long)r * value_dim + (c - 2 * key_dim)] = act;
        }
    }
}

extern "C" __global__ void __launch_bounds__(THREADS) brain_gdn_conv_hist_tok(const unsigned int* params, const float* x, float* hist) {
    const unsigned int rows = params[0], C = params[1];
    const unsigned int c = (blockIdx.y * gridDim.x + blockIdx.x) * THREADS + threadIdx.x;
    if (c >= C) { return; }
    // The last K-1 rows of [hist | x], read whole before any is written (one thread owns the channel).
    float next[K - 1];
    for (unsigned int j = 0; j + 1 < K; ++j) {
        const unsigned int e = rows + j;  // index into [hist | x]
        next[j] = e < K - 1 ? hist[(unsigned long long)c * (K - 1) + e] : x[(unsigned long long)(e - (K - 1)) * C + c];
    }
    for (unsigned int j = 0; j + 1 < K; ++j) { hist[(unsigned long long)c * (K - 1) + j] = next[j]; }
}

// `v` rounded to bf16, round to nearest even, NaN to the canonical quiet NaN: `cast_f32_bf16`'s own rule.
__device__ __forceinline__ unsigned int brain_gdn_bf16_bits(float v) {
    const unsigned int b = __float_as_uint(v);
    if ((b & 0x7fffffffu) > 0x7f800000u) { return 0x7fc0u; }
    return (b + 0x7fffu + ((b >> 16) & 1u)) >> 16;
}

// The forward for every prompt segment of a round in one launch (`table`: [row0, rows, slot, first chunk] per segment, see
// `brain_gdn_chunk_prep_seg_bf16`), reading each segment's history from its row of the history pool and leaving the
// next history there: the block that owns a segment's first rows reads the old history into its window and then
// overwrites it (no other block of the segment reads it), so the update needs no launch of its own. Rows are global
// rows of the token-major tensors.
// Contract: params [blocks, C, key_dim, value_dim, nseg, history words per pool row]; read x [rows, C], w [C, K], table; hist
// pool [slots, C, K-1] (read and written); write query, key [rows, key_dim], value [rows, value_dim] at the segments' rows.
// Params word 6 (absent is 0): bit 0, x is bf16 rows; bit 1, query / key / value are written as bf16 rows. The history stays f32 (it
// holds the bf16 values widened), the window and the sum are f32 either way, so the result is the f32 form's for bf16-valued input
// rounded once to bf16 on the way out.
extern "C" __global__ void __launch_bounds__(THREADS) brain_gdn_conv_silu_seg(const unsigned int* params, const float* x, const float* w, const unsigned int* table,
                                                                             float* hist_pool, float* query, float* key, float* value) {
    const unsigned int blocks = params[0], C = params[1], key_dim = params[2], value_dim = params[3], nseg = params[4];
    const bool x_bf16 = (params[6] & 1u) != 0u, out_bf16 = (params[6] & 2u) != 0u;
    const unsigned long long hist_words = params[5];
    const unsigned int blk = blockIdx.y * gridDim.x + blockIdx.x, tiles = (C + THREADS - 1) / THREADS;
    if (blk >= blocks) { return; }
    // The segment this block belongs to: each owns channel tiles x runs of ROWS_PER_THREAD rows.
    unsigned int b = blk, seg = 0;
    for (; seg < nseg; ++seg) {
        const unsigned int nb = tiles * ((table[4 * seg + 1] + ROWS_PER_THREAD - 1) / ROWS_PER_THREAD);
        if (b < nb) { break; }
        b -= nb;
    }
    if (seg == nseg) { return; }
    const unsigned int row0 = table[4 * seg], rows = table[4 * seg + 1], slot = table[4 * seg + 2];
    const unsigned int c = (b % tiles) * THREADS + threadIdx.x;
    const unsigned int r0 = (b / tiles) * ROWS_PER_THREAD;
    if (c >= C) { return; }
    const unsigned int r1 = min(r0 + ROWS_PER_THREAD, rows);
    float* hist = hist_pool + slot * hist_words + (unsigned long long)c * (K - 1);
    const unsigned long long x0 = (unsigned long long)row0 * C + c;
    const unsigned short* const xh = reinterpret_cast<const unsigned short*>(x);
    auto ldx = [&](unsigned long long off) -> float { return x_bf16 ? __uint_as_float(static_cast<unsigned int>(xh[x0 + off]) << 16) : x[x0 + off]; };

    float wk[K];
#pragma unroll
    for (unsigned int k = 0; k < K; ++k) { wk[k] = w[(unsigned long long)c * K + k]; }
    // The window feeding the segment's row r0: its inputs before the segment come from the history.
    float win[K];
    for (unsigned int k = 0; k < K; ++k) {
        const int q = (int)r0 + (int)k - (int)(K - 1);
        win[k] = q >= 0 ? ldx((unsigned long long)q * C) : hist[q + (int)(K - 1)];
    }
    if (r0 == 0) {
        // The last K-1 rows of [hist | x] become the next history; all of them are read before any is written.
        float next[K - 1];
        for (unsigned int j = 0; j + 1 < K; ++j) {
            const unsigned int e = rows + j;
            next[j] = e < K - 1 ? hist[e] : ldx((unsigned long long)(e - (K - 1)) * C);
        }
        for (unsigned int j = 0; j + 1 < K; ++j) { hist[j] = next[j]; }
    }
    for (unsigned int r = r0; r < r1; ++r) {
        if (r != r0) {
            for (unsigned int k = 0; k + 1 < K; ++k) { win[k] = win[k + 1]; }
            win[K - 1] = ldx((unsigned long long)r * C);
        }
        float acc = 0.0f;
        for (unsigned int k = 0; k < K; ++k) { acc = acc + win[k] * wk[k]; }
        const float act = silu_fwd(acc);
        const unsigned long long row = row0 + r;
        if (out_bf16) {
            const unsigned short h = static_cast<unsigned short>(brain_gdn_bf16_bits(act));
            if (c < key_dim) {
                reinterpret_cast<unsigned short*>(query)[row * key_dim + c] = h;
            } else if (c < 2 * key_dim) {
                reinterpret_cast<unsigned short*>(key)[row * key_dim + (c - key_dim)] = h;
            } else {
                reinterpret_cast<unsigned short*>(value)[row * value_dim + (c - 2 * key_dim)] = h;
            }
        } else if (c < key_dim) {
            query[row * key_dim + c] = act;
        } else if (c < 2 * key_dim) {
            key[row * key_dim + (c - key_dim)] = act;
        } else {
            value[row * value_dim + (c - 2 * key_dim)] = act;
        }
    }
}

extern "C" __global__ void __launch_bounds__(THREADS)
brain_gdn_conv_silu_bwd_tok(const unsigned int* params, const float* d_query, const float* d_key, const float* d_value, const float* pre, const float* w,
                            float* d_x, float* d_hist) {
    const unsigned int rows = params[0], C = params[1], T = params[2], key_dim = params[3], value_dim = params[4], ctx = params[5];
    const unsigned int blk = blockIdx.y * gridDim.x + blockIdx.x, tiles = (C + THREADS - 1) / THREADS;
    const unsigned int c = (blk % tiles) * THREADS + threadIdx.x;
    // Output rows are s in [-ctx, T) of each sequence; a run of ROWS_PER_THREAD of them per thread, numbered
    // from -ctx for the first sequence (a continued round has a single sequence).
    const int s0 = (int)((blk / tiles) * ROWS_PER_THREAD) - (int)ctx;
    const int s_end = min(s0 + (int)ROWS_PER_THREAD, (int)rows);
    if (c >= C || s0 >= (int)rows) { return; }

    float wk[K];
#pragma unroll
    for (unsigned int k = 0; k < K; ++k) { wk[k] = w[(unsigned long long)c * K + k]; }

    // d_pre of row `row` (global, 0 <= row < rows) at this channel.
    auto dpre = [&](int row) -> float {
        float d_act;
        if (c < key_dim) {
            d_act = d_query[(unsigned long long)row * key_dim + c];
        } else if (c < 2 * key_dim) {
            d_act = d_key[(unsigned long long)row * key_dim + (c - key_dim)];
        } else {
            d_act = d_value[(unsigned long long)row * value_dim + (c - 2 * key_dim)];
        }
        return silu_bwd(pre[(unsigned long long)row * C + c], d_act);
    };
    // dw[k] = d_pre of the row s + (K-1) - k in s's sequence (zero outside it).
    float dw[K];
    auto gather = [&](int s) {
        const int seq0 = s >= 0 ? s - s % (int)T : 0, p = s - seq0;
        for (unsigned int k = 0; k < K; ++k) {
            const int j = p + (int)(K - 1) - (int)k;
            dw[k] = (j >= 0 && j < (int)T) ? dpre(seq0 + j) : 0.0f;
        }
    };
    for (int s = s0; s < s_end; ++s) {
        if (s == s0 || (s >= 0 && s % (int)T == 0)) {
            gather(s);
        } else {
            for (int k = (int)K - 1; k > 0; --k) { dw[k] = dw[k - 1]; }
            const int seq0 = s - s % (int)T, p = s - seq0;
            const int j = p + (int)K - 1;
            dw[0] = j < (int)T ? dpre(seq0 + j) : 0.0f;
        }
        float acc = 0.0f;
        for (unsigned int k = 0; k < K; ++k) { acc = acc + wk[k] * dw[k]; }
        if (s >= 0) {
            d_x[(unsigned long long)s * C + c] = acc;
        } else {
            d_hist[(unsigned long long)(s + (int)ctx) * C + c] = acc;
        }
    }
}
