// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements low-latency chunked linear-attention training
// kernels for its clients. If your team needs expertise in collapsing a
// sequential GPU dependency chain of small launches into one persistent kernel,
// then you can procure our services by sending an email to
// info@swedishembedded.com.
//
// The reverse sweep of Gated DeltaNet's across-chunk recurrence (`model::gdn`'s
// `gdn_chunk_bwd`, "the sequential part") in ONE launch. The portable path walks
// the chunks newest first on the host and issues six dispatches per chunk, each
// waiting on the last through the running state gradient `d_state`:
//
//   d_decayed_k  = v_new @ d_state^T                      (overwrite)
//   d_v_new     += decayed_k @ d_state
//   d_g_cs[last]+= decay_last * sum(d_state * state_in)
//   d_state     *= decay_last
//   d_state     += q_scaled^T @ d_out
//   d_state     -= w^T @ d_v_new                          (the d_v_new just completed)
//
// Both products with `d_state` contract over a different one of its two axes, so
// the state cannot be split across blocks the way the forward splits its value
// columns: one block owns one head and keeps the whole 128 x 128 state in
// shared memory (padded to 129 columns so a column read is conflict-free) for
// every chunk. That is more than the 48 KiB static limit, so the buffer is
// dynamic shared memory the registration opts in to.
//
// NUMERICS: each product is the ascending-k fp32 sum starting from 0, added to
// its destination the way the portable kernels do (`acc = acc + a * b`, compiled
// without contraction). The state-update products are folded into the state
// element in one accumulator, so their rounding differs from the dispatch
// sequence by the fp32 summation bound; so does the decay-scale reduction.
//
// Contract (u32 params, then pointers; every buffer is the chunk-major layout
// `model::gdn` documents, `bhc = chunk * (B*H) + b*H + h`):
//   params : [bh, c_len, dk, dv, n_chunks]
//   read   : v_new_all, decayed_k_all, q_scaled_all, w, d_out
//            [n_chunks, bh, c_len, dk or dv];  state_hist [n_chunks, bh, dk, dv]
//            (the state ENTERING each chunk);  g_cs [n_chunks, bh, c_len];
//            d_final_state [bh, dk, dv]
//   write  : d_decayed_k [n_chunks, bh, c_len, dk]   (overwritten)
//            d_v_new     [n_chunks, bh, c_len, dv]   (accumulated into)
//            d_g_cs      [n_chunks, bh, c_len]       (accumulated into)
//            d_initial_state [bh, dk, dv]
// dk = dv = 128 and c_len <= 64 (the provider checks; the entry does not).
// Dynamic shared memory: BWD_SWEEP_SMEM_BYTES.
//
// No `__restrict__`: brain's device buffers alias by design.

#define DK 128
#define DV 128
#define MAXC 64
#define THREADS 256
#define SD (DK + 1)          // padded row stride of the state and of the staged tiles
#define KC 16                // k-depth staged per step

// out[r][col] (RM*16 rows x 128 cols, thread (ty, tx) owns rows ty + 16 i and
// columns tx + 16 j) += sum over k < K of fa(k, r) * fb(k, col). AK / BK say
// whether the A / B operand is contiguous along k in memory, which picks the
// coalesced staging order.
template <int RM, bool AK, bool BK, class FA, class FB>
__device__ __forceinline__ void tile_gemm(float (&acc)[RM][8], float* ta, float* tb, unsigned int K, FA fa, FB fb) {
    constexpr unsigned int ROWS = RM * 16;
    const unsigned int tid = threadIdx.x, ty = tid >> 4, tx = tid & 15u;
    for (unsigned int k0 = 0; k0 < K; k0 += KC) {
        for (unsigned int e = tid; e < KC * ROWS; e += THREADS) {
            const unsigned int kk = AK ? e % KC : e / ROWS, r = AK ? e / KC : e % ROWS;
            ta[kk * SD + r] = (k0 + kk < K) ? fa(k0 + kk, r) : 0.0f;
        }
        for (unsigned int e = tid; e < KC * DV; e += THREADS) {
            const unsigned int kk = BK ? e % KC : e / DV, col = BK ? e / KC : e % DV;
            tb[kk * SD + col] = (k0 + kk < K) ? fb(k0 + kk, col) : 0.0f;
        }
        __syncthreads();
#pragma unroll
        for (unsigned int kk = 0; kk < KC; ++kk) {
            float a[RM], b[8];
#pragma unroll
            for (int i = 0; i < RM; ++i) { a[i] = ta[kk * SD + ty + 16 * i]; }
#pragma unroll
            for (int j = 0; j < 8; ++j) { b[j] = tb[kk * SD + tx + 16 * j]; }
#pragma unroll
            for (int i = 0; i < RM; ++i) {
#pragma unroll
                for (int j = 0; j < 8; ++j) { acc[i][j] = acc[i][j] + a[i] * b[j]; }
            }
        }
        __syncthreads();
    }
}

extern "C" __global__ void __launch_bounds__(THREADS)
brain_gdn_bwd_sweep_f32(const unsigned int* params, const float* v_new, const float* decayed_k, const float* q_scaled, const float* w, const float* d_out,
                        const float* state_hist, const float* g_cs, const float* d_final_state, float* d_decayed_k, float* d_v_new, float* d_g_cs,
                        float* d_initial_state) {
    extern __shared__ float smem[];
    float* ds = smem;                    // running d_state [DK][SD]
    float* ta = ds + DK * SD;            // staged A tile [KC][SD]
    float* tb = ta + KC * SD;            // staged B tile [KC][SD]
    float* red = tb + KC * SD;           // block-reduction scratch, one float per warp

    const unsigned int bh = params[0];
    const unsigned int C = params[1];
    const unsigned int n_chunks = params[4];
    const unsigned int bhi = blockIdx.y * gridDim.x + blockIdx.x;
    if (bhi >= bh) { return; }  // block-uniform

    const unsigned int tid = threadIdx.x, ty = tid >> 4, tx = tid & 15u;
    const unsigned long long head_state = (unsigned long long)bhi * DK * DV;

    for (unsigned int i = tid; i < DK * DV; i += THREADS) { ds[(i / DV) * SD + i % DV] = d_final_state[head_state + i]; }
    __syncthreads();

    for (unsigned int step = 0; step < n_chunks; ++step) {
        const unsigned int ci = n_chunks - 1 - step;
        const unsigned long long row_base = ((unsigned long long)ci * bh + bhi) * C;
        const unsigned long long off_dk = row_base * DK, off_dv = row_base * DV;
        const unsigned long long state_in_off = ((unsigned long long)ci * bh + bhi) * DK * DV;

        // d_decayed_k[c, p] = sum_q v_new[c, q] * d_state[p, q]
        {
            float acc[4][8];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
#pragma unroll
                for (int j = 0; j < 8; ++j) { acc[i][j] = 0.0f; }
            }
            tile_gemm<4, true, true>(
                acc, ta, tb, DV, [&](unsigned int q, unsigned int r) { return r < C ? v_new[off_dv + (unsigned long long)r * DV + q] : 0.0f; },
                [&](unsigned int q, unsigned int p) { return ds[p * SD + q]; });
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const unsigned int r = ty + 16 * i;
                if (r < C) {
#pragma unroll
                    for (int j = 0; j < 8; ++j) { d_decayed_k[off_dk + (unsigned long long)r * DK + tx + 16 * j] = acc[i][j]; }
                }
            }
        }
        // d_v_new[c, q] += sum_p decayed_k[c, p] * d_state[p, q]
        {
            float acc[4][8];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
#pragma unroll
                for (int j = 0; j < 8; ++j) { acc[i][j] = 0.0f; }
            }
            tile_gemm<4, true, false>(
                acc, ta, tb, DK, [&](unsigned int p, unsigned int r) { return r < C ? decayed_k[off_dk + (unsigned long long)r * DK + p] : 0.0f; },
                [&](unsigned int p, unsigned int q) { return ds[p * SD + q]; });
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const unsigned int r = ty + 16 * i;
                if (r < C) {
#pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        const unsigned long long at = off_dv + (unsigned long long)r * DV + tx + 16 * j;
                        d_v_new[at] = d_v_new[at] + acc[i][j];
                    }
                }
            }
        }

        // d_g_cs[last] += decay_last * sum(d_state * state_in); then d_state *= decay_last.
        const float decay_last = expf(g_cs[row_base + C - 1]);
        float part = 0.0f;
        for (unsigned int i = tid; i < DK * DV; i += THREADS) { part = part + ds[(i / DV) * SD + i % DV] * state_hist[state_in_off + i]; }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) { part += __shfl_xor_sync(0xffffffffu, part, o); }
        if ((tid & 31u) == 0) { red[tid >> 5] = part; }
        __syncthreads();
        if (tid == 0) {
            float total = 0.0f;
            for (unsigned int i = 0; i < THREADS / 32; ++i) { total = total + red[i]; }
            d_g_cs[row_base + C - 1] = d_g_cs[row_base + C - 1] + total * decay_last;
        }

        // d_state = d_state * decay_last + q_scaled^T @ d_out - w^T @ d_v_new (d_v_new completed above)
        float acc[8][8];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
#pragma unroll
            for (int j = 0; j < 8; ++j) { acc[i][j] = ds[(ty + 16 * i) * SD + tx + 16 * j] * decay_last; }
        }
        tile_gemm<8, false, false>(
            acc, ta, tb, C, [&](unsigned int i, unsigned int p) { return q_scaled[off_dk + (unsigned long long)i * DK + p]; },
            [&](unsigned int i, unsigned int q) { return d_out[off_dv + (unsigned long long)i * DV + q]; });
        tile_gemm<8, false, false>(
            acc, ta, tb, C, [&](unsigned int c, unsigned int p) { return -w[off_dk + (unsigned long long)c * DK + p]; },
            [&](unsigned int c, unsigned int q) { return d_v_new[off_dv + (unsigned long long)c * DV + q]; });
#pragma unroll
        for (int i = 0; i < 8; ++i) {
#pragma unroll
            for (int j = 0; j < 8; ++j) { ds[(ty + 16 * i) * SD + tx + 16 * j] = acc[i][j]; }
        }
        __syncthreads();
    }

    for (unsigned int i = tid; i < DK * DV; i += THREADS) { d_initial_state[head_state + i] = ds[(i / DV) * SD + i % DV]; }
}
