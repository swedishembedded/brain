// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements tensor-core training kernels for linear-
// attention language models for its clients. If your team needs expertise in
// fusing a long chain of small gradient launches over a recurrent layer into
// one tensor-core kernel without changing the model's gradients beyond a
// stated bound, then you can procure our services by sending an email to
// info@swedishembedded.com.
//
// Everything of Gated DeltaNet's chunk backward (`model::gdn::gdn_chunk_bwd`)
// that is independent across chunks, in ONE launch with one block per
// (head, chunk). The portable backward issues about sixty whole-tensor
// dispatches around the reverse sweep - eleven batched products, a dozen row
// scales, row dots and splices, the closed-form unit-triangular-solve backward,
// two mask kernels, the decay and cumulative-sum gradients - each streaming its
// operands through HBM. Here a chunk's tiles are read once, every intermediate
// that is not an output stays in shared memory or registers, and the thirteen
// matrix products run as three-pass TF32 `mma.sync.m16n8k8` (about 20 mantissa
// bits, fp32 accumulation).
//
// Inputs besides the forward's saved tensors are the two products of the
// tensor-core sweep (`gdn_bwd_sweep_tc`): d_v_new (final) and the history of the
// state gradient D entering each chunk's reverse step. With S the state a chunk
// starts from, O = d_out, and decay_last = exp(g_last):
//
//   v_new        = u - w @ S                        d_intra   = O @ v_new^T
//   d_q_scaled   = O @ S^T                          d_query   = scale * exp_g_cs * d_q_scaled
//   d_decayed_k  = v_new @ D^T                      d_key     = decay_scale * d_decayed_k
//   d_w          = -d_v_new @ S^T                   d_u       = d_v_new
//   d_t_mat      = d_w @ k_beta_decay^T + d_u @ v_beta^T
//   d_k_beta_dec = T^T @ d_w                        d_v_beta  = T^T @ d_u
//   d_attn0      = T^T @ d_t_mat @ T^T              (closed-form inverse adjoint)
//   d_raw_intra  = d_intra * decay_mask             d_decay_mask = d_intra * raw_intra + d_attn0 * raw_attn0
//   d_query     += scale * d_raw_intra @ key        d_key    += scale * d_raw_intra^T @ query
//   d_k_beta     = exp_g_cs * d_k_beta_dec - strict_lower(d_attn0 * decay_mask) @ key
//   d_key       += -strict_lower(d_attn0 * decay_mask)^T @ k_beta + beta * d_k_beta
//   d_value      = beta * d_v_beta
//   d_beta       = rowdot(d_v_beta, value) + rowdot(d_k_beta, key)
//   d_g_cs       = decay terms, d_raw_g = its suffix sum
//
// The d_g_cs sources are the same four as the portable path (the state decay of the
// sweep, decay_scale, exp_g_cs and the decay mask).
//
// Contract (u32 params, then pointers; chunk-major, `bhc = chunk * bh + head`):
//   params : [bhc, c_len, dk, dv, bh, scale (f32 bits), heads_tok, n_chunks]
//   read   : query, key, value, w, u, d_out, d_v_new [bhc, c_len, 128]; beta, g_cs, exp_g_cs,
//            decay_scale_hist [bhc, c_len]; state_hist (the state ENTERING each chunk) and
//            d_state_hist [bhc, 128, 128]; t_mat, raw_intra, raw_attn0 [bhc, c_len, c_len]
//   write  : d_query, d_key, d_value [bhc, c_len, 128]; d_raw_g, d_beta [bhc, c_len]
//            (every element is overwritten; no clearing is needed)
//   TOKEN-MAJOR form (heads_tok != 0, the mixer's layout; n_chunks chunks per sequence): d_out is read as
//            [tokens, heads_tok, 128] and d_value, d_raw_g, d_beta are written as [tokens, heads_tok, 128]
//            and [tokens, heads_tok]; d_query and d_key stay chunk-major (the per-key-head gather and the
//            normalisation adjoint, `gdn_qk_bwd_tok`, follow). The chunk-major permutes of the portable
//            path are folded into these loads and stores.
// dk = dv = 128 and 1 <= c_len <= 64 (the provider checks; the entry does not).
// Dynamic shared memory: POST_SMEM_FLOATS floats.
//
// No `__restrict__`: brain's device buffers alias by design.

#define D 128
#define MAXC 64
#define THREADS 256          // 8 warps: 4 row tiles x 2 column halves
#define SK (D + 4)           // row stride of the 128-wide tiles
#define SC (MAXC + 4)        // row stride of the 64 x 64 matrices
#define SLAB_K 16            // contraction depth of one staged slab of a global 128 x 128 matrix
#define SKL (D + 8)          // row stride of a k-major 16 x 128 slab (conflict-free as a k-major B operand)
#define SSL (SLAB_K + 4)     // row stride of an n-major 128 x 16 slab (conflict-free as an n-major B operand)
#define SLAB_FLOATS (D * SSL)  // the larger of the two slab shapes: 2560 floats

// Four 64 x 128 tiles, four 64 x 64 matrices, eight 64-float row vectors and 16 floats of reduction scratch.
// The two 64 x 64 matrices sDM and sDT are idle until the middle of the kernel and hold the double-buffered slabs.
#define POST_SMEM_FLOATS (4 * MAXC * SK + 4 * MAXC * SC + 8 * MAXC + 16)

// x = hi + lo with both parts TF32-representable. Truncation (a mask) instead of `cvt.rna.tf32`: the convert runs on
// a slow pipe and the splits are most of the instructions around the MMAs. lo = x - hi is exact in fp32 and is then
// truncated to TF32 itself, so the pair carries about 20 mantissa bits.
__device__ __forceinline__ void brain_split_tf32(float x, unsigned int& hi, unsigned int& lo) {
    hi = __float_as_uint(x) & 0xFFFFE000u;
    lo = __float_as_uint(x - __uint_as_float(hi)) & 0xFFFFE000u;
}

__device__ __forceinline__ void brain_mma_tf32(float (&d)[4], const unsigned int (&a)[4], unsigned int b0, unsigned int b1) {
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void cp_async_16(void* dst_shared, const void* src_global) {
    const unsigned int dst = static_cast<unsigned int>(__cvta_generic_to_shared(dst_shared));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(dst), "l"(src_global));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

template <int NT>
__device__ __forceinline__ void zero_acc(float (&acc)[NT][4]) {
#pragma unroll
    for (int t = 0; t < NT; ++t) {
#pragma unroll
        for (int i = 0; i < 4; ++i) { acc[t][i] = 0.0f; }
    }
}

// acc(16 x 8*NT) += sum_{k < K} fa(r, k) * fb(k, n) for this warp's rows m0 .. and columns n0 .., as
// three TF32 products per tile (head*head + head*remainder + remainder*head). The caller zeroes acc.
template <int NT, class FA, class FB>
__device__ __forceinline__ void warp_mma_acc(float (&acc)[NT][4], unsigned int K, unsigned int m0, unsigned int n0, FA fa, FB fb) {
    const unsigned int lane = threadIdx.x & 31u, g = lane >> 2, t4 = lane & 3u;
    for (unsigned int k0 = 0; k0 < K; k0 += 8) {
        unsigned int ah[4], al[4];
        brain_split_tf32(fa(m0 + g, k0 + t4), ah[0], al[0]);
        brain_split_tf32(fa(m0 + g + 8, k0 + t4), ah[1], al[1]);
        brain_split_tf32(fa(m0 + g, k0 + t4 + 4), ah[2], al[2]);
        brain_split_tf32(fa(m0 + g + 8, k0 + t4 + 4), ah[3], al[3]);
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            const unsigned int n = n0 + t * 8 + g;
            unsigned int bh0, bl0, bh1, bl1;
            brain_split_tf32(fb(k0 + t4, n), bh0, bl0);
            brain_split_tf32(fb(k0 + t4 + 4, n), bh1, bl1);
            brain_mma_tf32(acc[t], al, bh0, bh1);
            brain_mma_tf32(acc[t], ah, bl0, bl1);
            brain_mma_tf32(acc[t], ah, bh0, bh1);
        }
    }
}

// Visit a 128 x 128 row-major matrix M in global memory as eight slabs of SLAB_K contraction steps, double-buffered
// through shared memory with `cp.async` so the next slab's coalesced loads overlap the MMAs on the current one (the
// MMA fragments would otherwise read M straight from global memory with strided 4-byte loads). With KMAJOR the
// operand is M(k, n) = M[k * 128 + n] and a slab holds SLAB_K rows of M, `slab[k * SKL + n]`; otherwise it is the
// transpose, M(k, n) = M[n * 128 + k], and a slab holds SLAB_K columns, `slab[n * SSL + k]`. `compute(slab, s)` runs
// for slab s on every thread of the block (this function synchronises) and covers contraction steps
// [s * SLAB_K, (s + 1) * SLAB_K). `bufs` is two slabs of SLAB_FLOATS floats.
template <bool KMAJOR, class Compute>
__device__ __forceinline__ void for_each_slab128(const float* M, float* bufs, Compute compute) {
    const unsigned int tid = threadIdx.x;
    auto stage = [&](unsigned int s, float* dst) {
        for (unsigned int e = tid; e < SLAB_K * (D / 4); e += THREADS) {
            if (KMAJOR) {
                const unsigned int row = e / (D / 4), c4 = e % (D / 4);
                cp_async_16(dst + row * SKL + c4 * 4, M + (s * SLAB_K + row) * D + c4 * 4);
            } else {
                const unsigned int row = e / (SLAB_K / 4), c4 = e % (SLAB_K / 4);
                cp_async_16(dst + row * SSL + c4 * 4, M + row * D + s * SLAB_K + c4 * 4);
            }
        }
        cp_async_commit();
    };
    stage(0, bufs);
    for (unsigned int s = 0; s < D / SLAB_K; ++s) {
        if (s + 1 < D / SLAB_K) {
            stage(s + 1, bufs + ((s + 1) & 1u) * SLAB_FLOATS);
            cp_async_wait<1>();
        } else {
            cp_async_wait<0>();
        }
        __syncthreads();
        compute(bufs + (s & 1u) * SLAB_FLOATS, s);
        __syncthreads();
    }
}

// Rows of a [rows, 128] source (consecutive rows `row_stride` floats apart) into a zero-padded [64, stride] shared tile,
// synchronously.
__device__ __forceinline__ void load_rows(float* dst, unsigned int stride, const float* src, unsigned long long row_stride, unsigned int C) {
    for (unsigned int e = threadIdx.x; e < MAXC * (D / 4); e += THREADS) {
        const unsigned int r = e / (D / 4), c4 = e % (D / 4);
        float4 v = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (r < C) { v = *reinterpret_cast<const float4*>(src + (unsigned long long)r * row_stride + c4 * 4); }
        *reinterpret_cast<float4*>(dst + r * stride + c4 * 4) = v;
    }
}

// The same, asynchronously: the rows are in the tile after `cp_async_wait` and a barrier. Padding rows are written now.
__device__ __forceinline__ void load_rows_async(float* dst, unsigned int stride, const float* src, unsigned long long row_stride, unsigned int C) {
    for (unsigned int e = threadIdx.x; e < MAXC * (D / 4); e += THREADS) {
        const unsigned int r = e / (D / 4), c4 = e % (D / 4);
        if (r < C) {
            cp_async_16(dst + r * stride + c4 * 4, src + (unsigned long long)r * row_stride + c4 * 4);
        } else {
            *reinterpret_cast<float4*>(dst + r * stride + c4 * 4) = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
    }
    cp_async_commit();
}

// Sum a per-row partial over the four lanes that share a row of an MMA accumulator fragment.
__device__ __forceinline__ float quad_sum(float x) {
    x += __shfl_xor_sync(0xffffffffu, x, 1);
    x += __shfl_xor_sync(0xffffffffu, x, 2);
    return x;
}

extern "C" __global__ void __launch_bounds__(THREADS, 1)
brain_gdn_chunk_bwd_post_tc(const unsigned int* params, const float* query, const float* key, const float* value, const float* beta, const float* w,
                            const float* u, const float* g_cs, const float* exp_g_cs, const float* ds_hist, const float* state_hist, const float* t_mat,
                            const float* raw_intra, const float* raw_attn0, const float* d_out, const float* d_state_hist, const float* d_v_new,
                            float* d_query, float* d_key, float* d_value, float* d_raw_g, float* d_beta) {
    const unsigned int bhc = params[0];
    const unsigned int C = params[1];
    const float scale = __uint_as_float(params[5]);

    const unsigned int blk = blockIdx.y * gridDim.x + blockIdx.x;
    if (blk >= bhc) { return; }  // block-uniform

    extern __shared__ float smem[];
    float* sK = smem;                    // key rows, [64][SK]
    float* sO = sK + MAXC * SK;          // d_out rows, then the value rows, then d_k_beta
    float* sV = sO + MAXC * SK;          // v_new, then d_w, then query
    float* sKB = sV + MAXC * SK;         // w, then d_v_new (= d_u)
    float* sT = sKB + MAXC * SK;         // T = (I - attn0)^-1, [64][SC]
    float* sDI = sT + MAXC * SC;         // d_intra_scores, then d_raw_intra, then X, [64][SC]
    float* sDM = sDI + MAXC * SC;        // the slab buffers, then d_decay_mask (sDT follows: two slabs of 2560 floats fit both)
    float* sDT = sDM + MAXC * SC;        // d_t_mat, then strict_lower(d_attn0 * decay_mask)
    float* sg = sDT + MAXC * SC;         // cumulative decay
    float* sexp = sg + MAXC;             // exp of it
    float* sds = sexp + MAXC;            // decay_scale
    float* sbeta = sds + MAXC;
    float* sdexp = sbeta + MAXC;         // d_exp_g_cs
    float* sdds = sdexp + MAXC;          // d_decay_scale
    float* sdg = sdds + MAXC;            // d_g_cs
    float* sdb = sdg + MAXC;             // d_beta
    float* sred = sdb + MAXC;            // block-reduction scratch

    const unsigned int tid = threadIdx.x, lane = tid & 31u, warp = tid >> 5;
    const unsigned int g = lane >> 2, t4 = lane & 3u;
    const unsigned int mt = warp & 3u, nh = warp >> 2;
    // d_query and d_key are built up over many products that all use this thread's fragment positions (rows m0 + g and
    // m0 + g + 8, 16 columns of the 128): they stay in registers and are stored once instead of read-modify-written.
    float dq[8][4], dk[8][4];
    const unsigned int m0 = mt * 16u;                       // this warp's 16-row tile
    const unsigned int c8 = (C + 7u) & ~7u;                 // contraction length over chunk rows
    const unsigned long long row_base = (unsigned long long)blk * C;
    const unsigned long long cc_base = row_base * C;
    const float* S = state_hist + (unsigned long long)blk * D * D;
    const float* Dh = d_state_hist + (unsigned long long)blk * D * D;

    // Where d_out is read and d_value, d_raw_g, d_beta are written: chunk-major by default, the mixer's token-major
    // layout when `heads_tok` is given (this block's first token row, head, and the row strides of each tensor).
    const unsigned int heads_tok = params[6];
    unsigned long long vec_base = row_base * D, vec_stride = D, scal_base = row_base, scal_stride = 1;
    if (heads_tok != 0) {
        const unsigned int n_chunks = params[7];
        const unsigned int bh = bhc / n_chunks, ci = blk / bh, rem = blk % bh;
        const unsigned long long tok0 = ((unsigned long long)(rem / heads_tok) * n_chunks + ci) * C;
        vec_base = (tok0 * heads_tok + rem % heads_tok) * D;
        vec_stride = (unsigned long long)heads_tok * D;
        scal_base = tok0 * heads_tok + rem % heads_tok;
        scal_stride = heads_tok;
    }

    for (unsigned int i = tid; i < MAXC * SC; i += THREADS) {
        sT[i] = 0.0f;
        sDI[i] = 0.0f;
    }
    if (tid < MAXC) {
        const bool live = tid < C;
        sg[tid] = live ? g_cs[row_base + tid] : 0.0f;
        sexp[tid] = live ? exp_g_cs[row_base + tid] : 0.0f;
        sds[tid] = live ? ds_hist[row_base + tid] : 0.0f;
        sbeta[tid] = live ? beta[row_base + tid] : 0.0f;
        sdexp[tid] = 0.0f;
        sdds[tid] = 0.0f;
        sdg[tid] = 0.0f;
        sdb[tid] = 0.0f;
    }
    __syncthreads();
    load_rows(sK, SK, key + row_base * D, D, C);
    load_rows(sO, SK, d_out + vec_base, vec_stride, C);
    load_rows(sKB, SK, w + row_base * D, D, C);
    for (unsigned int e = tid; e < C * C; e += THREADS) { sT[(e / C) * SC + e % C] = t_mat[cc_base + e]; }
    __syncthreads();

    // ---- v_new = u - w @ S ----
    {
        float acc[8][4];
        zero_acc<8>(acc);
        for_each_slab128<true>(S, sDM, [&](const float* slab, unsigned int s) {
            if (m0 < C) {
                warp_mma_acc<8>(acc, SLAB_K, m0, nh * 64u, [&](unsigned int r, unsigned int k) { return sKB[r * SK + s * SLAB_K + k]; },
                                [&](unsigned int k, unsigned int n) { return slab[k * SKL + n]; });
            }
        });
#pragma unroll
        for (int t = 0; t < 8; ++t) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const unsigned int r = m0 + g + ((i & 2) ? 8u : 0u);
                const unsigned int col = nh * 64u + t * 8 + 2 * t4 + (i & 1);
                sV[r * SK + col] = r < C ? u[(row_base + r) * D + col] - acc[t][i] : 0.0f;
            }
        }
    }
    __syncthreads();
    load_rows_async(sKB, SK, d_v_new + row_base * D, D, C);   // w is dead: the final d_v_new (= d_u) takes its place while the next products run

    // ---- d_intra = O @ v_new^T (c x c) ----
    {
        float acc[4][4];
        zero_acc<4>(acc);
        if (m0 < C && nh * 32u < C) {
            warp_mma_acc<4>(acc, D, m0, nh * 32u, [&](unsigned int r, unsigned int k) { return sO[r * SK + k]; }, [&](unsigned int k, unsigned int n) { return sV[n * SK + k]; });
        }
#pragma unroll
        for (int t = 0; t < 4; ++t) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const unsigned int r = m0 + g + ((i & 2) ? 8u : 0u);
                const unsigned int col = nh * 32u + t * 8 + 2 * t4 + (i & 1);
                sDI[r * SC + col] = (r < C && col < C) ? acc[t][i] : 0.0f;
            }
        }
    }

    // ---- d_decayed_k = v_new @ D^T: d_key = decay_scale * it, and d_decay_scale = rowdot(it, key) ----
    {
        float acc[8][4];
        zero_acc<8>(acc);
        for_each_slab128<false>(Dh, sDM, [&](const float* slab, unsigned int s) {
            if (m0 < C) {
                warp_mma_acc<8>(acc, SLAB_K, m0, nh * 64u, [&](unsigned int r, unsigned int k) { return sV[r * SK + s * SLAB_K + k]; },
                                [&](unsigned int k, unsigned int n) { return slab[n * SSL + k]; });
            }
        });
        float dots[2] = {0.0f, 0.0f};
#pragma unroll
        for (int t = 0; t < 8; ++t) {
#pragma unroll
            for (int h2 = 0; h2 < 2; ++h2) {
                const unsigned int r = m0 + g + h2 * 8u;
                const unsigned int col = nh * 64u + t * 8 + 2 * t4;
                if (r < C) {
                    const float a0 = acc[t][2 * h2], a1 = acc[t][2 * h2 + 1];
                    dk[t][2 * h2] = a0 * sds[r];
                    dk[t][2 * h2 + 1] = a1 * sds[r];
                    dots[h2] += a0 * sK[r * SK + col] + a1 * sK[r * SK + col + 1];
                }
            }
        }
#pragma unroll
        for (int h2 = 0; h2 < 2; ++h2) {
            const float d = quad_sum(dots[h2]);
            const unsigned int r = m0 + g + h2 * 8u;
            if (t4 == 0 && r < C) { atomicAdd(&sdds[r], d); }
        }
    }
    cp_async_wait<0>();
    __syncthreads();   // d_v_new is in sKB; v_new (sV) is dead

    // ---- d_q_scaled = O @ S^T and d_w = -d_v_new @ S^T, both against the same slabs of S ----
    {
        float acc[8][4], accw[8][4];
        zero_acc<8>(acc);
        zero_acc<8>(accw);
        for_each_slab128<false>(S, sDM, [&](const float* slab, unsigned int s) {
            if (m0 < C) {
                auto fb = [&](unsigned int k, unsigned int n) { return slab[n * SSL + k]; };
                warp_mma_acc<8>(acc, SLAB_K, m0, nh * 64u, [&](unsigned int r, unsigned int k) { return sO[r * SK + s * SLAB_K + k]; }, fb);
                warp_mma_acc<8>(accw, SLAB_K, m0, nh * 64u, [&](unsigned int r, unsigned int k) { return sKB[r * SK + s * SLAB_K + k]; }, fb);
            }
        });
        // d_query = scale * exp_g_cs * it, and d_exp_g_cs = scale * rowdot(it, query).
        float dots[2] = {0.0f, 0.0f};
#pragma unroll
        for (int t = 0; t < 8; ++t) {
#pragma unroll
            for (int h2 = 0; h2 < 2; ++h2) {
                const unsigned int r = m0 + g + h2 * 8u;
                const unsigned int col = nh * 64u + t * 8 + 2 * t4;
                if (r < C) {
                    const float a0 = acc[t][2 * h2], a1 = acc[t][2 * h2 + 1];
                    const float2 q = *reinterpret_cast<const float2*>(query + (row_base + r) * D + col);
                    dq[t][2 * h2] = (a0 * sexp[r]) * scale;
                    dq[t][2 * h2 + 1] = (a1 * sexp[r]) * scale;
                    dots[h2] += a0 * q.x + a1 * q.y;
                }
            }
        }
#pragma unroll
        for (int h2 = 0; h2 < 2; ++h2) {
            const float d = quad_sum(dots[h2]);
            const unsigned int r = m0 + g + h2 * 8u;
            if (t4 == 0 && r < C) { atomicAdd(&sdexp[r], scale * d); }
        }
#pragma unroll
        for (int t = 0; t < 8; ++t) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const unsigned int r = m0 + g + ((i & 2) ? 8u : 0u);
                sV[r * SK + nh * 64u + t * 8 + 2 * t4 + (i & 1)] = r < C ? -accw[t][i] : 0.0f;   // d_w
            }
        }
    }
    __syncthreads();   // d_out (sO) is dead: the value rows take its place
    load_rows_async(sO, SK, value + row_base * D, D, C);

    // ---- the state-decay term of d_g_cs[last]: decay_last * sum(D * S) ----
    {
        float part = 0.0f;
        for (unsigned int e = tid; e < D * D / 4; e += THREADS) {
            const float4 a = *reinterpret_cast<const float4*>(Dh + e * 4), b = *reinterpret_cast<const float4*>(S + e * 4);
            part += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) { part += __shfl_xor_sync(0xffffffffu, part, o); }
        if (lane == 0) { sred[warp] = part; }
    }
    cp_async_wait<0>();
    __syncthreads();

    // ---- d_t_mat = d_w @ k_beta_decay^T + d_u @ v_beta^T ----
    {
        float acc[4][4];
        zero_acc<4>(acc);
        if (m0 < C && nh * 32u < C) {
            warp_mma_acc<4>(acc, D, m0, nh * 32u, [&](unsigned int r, unsigned int k) { return sV[r * SK + k]; },
                            [&](unsigned int k, unsigned int n) { return (sK[n * SK + k] * sbeta[n]) * sexp[n]; });
            warp_mma_acc<4>(acc, D, m0, nh * 32u, [&](unsigned int r, unsigned int k) { return sKB[r * SK + k]; },
                            [&](unsigned int k, unsigned int n) { return sO[n * SK + k] * sbeta[n]; });
        }
#pragma unroll
        for (int t = 0; t < 4; ++t) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const unsigned int r = m0 + g + ((i & 2) ? 8u : 0u);
                const unsigned int col = nh * 32u + t * 8 + 2 * t4 + (i & 1);
                sDT[r * SC + col] = (r < C && col < C) ? acc[t][i] : 0.0f;
            }
        }
    }

    // ---- d_v_beta = T^T @ d_u: d_value = beta * it, d_beta = rowdot(it, value) ----
    {
        float acc[8][4];
        zero_acc<8>(acc);
        if (m0 < C) {
            warp_mma_acc<8>(acc, c8, m0, nh * 64u, [&](unsigned int i, unsigned int j) { return sT[j * SC + i]; }, [&](unsigned int j, unsigned int q) { return sKB[j * SK + q]; });
        }
        float dots[2] = {0.0f, 0.0f};
#pragma unroll
        for (int t = 0; t < 8; ++t) {
#pragma unroll
            for (int h2 = 0; h2 < 2; ++h2) {
                const unsigned int r = m0 + g + h2 * 8u;
                const unsigned int col = nh * 64u + t * 8 + 2 * t4;
                if (r < C) {
                    const float a0 = acc[t][2 * h2], a1 = acc[t][2 * h2 + 1];
                    const float2 v = *reinterpret_cast<const float2*>(sO + r * SK + col);
                    *reinterpret_cast<float2*>(d_value + vec_base + r * vec_stride + col) = make_float2(a0 * sbeta[r], a1 * sbeta[r]);
                    dots[h2] += a0 * v.x + a1 * v.y;
                }
            }
        }
#pragma unroll
        for (int h2 = 0; h2 < 2; ++h2) {
            const float d = quad_sum(dots[h2]);
            const unsigned int r = m0 + g + h2 * 8u;
            if (t4 == 0 && r < C) { atomicAdd(&sdb[r], d); }
        }
    }
    __syncthreads();  // sO held the value rows until here; d_k_beta overwrites them below

    // ---- d_k_beta_decay = T^T @ d_w: d_k_beta = exp_g_cs * it, d_exp_g_cs += rowdot(it, k_beta) ----
    {
        float acc[8][4];
        zero_acc<8>(acc);
        if (m0 < C) {
            warp_mma_acc<8>(acc, c8, m0, nh * 64u, [&](unsigned int i, unsigned int j) { return sT[j * SC + i]; }, [&](unsigned int j, unsigned int p) { return sV[j * SK + p]; });
        }
        float dots[2] = {0.0f, 0.0f};
#pragma unroll
        for (int t = 0; t < 8; ++t) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const unsigned int r = m0 + g + ((i & 2) ? 8u : 0u);
                const unsigned int col = nh * 64u + t * 8 + 2 * t4 + (i & 1);
                sO[r * SK + col] = r < C ? acc[t][i] * sexp[r] : 0.0f;
                if (r < C) { dots[i >> 1] += acc[t][i] * (sK[r * SK + col] * sbeta[r]); }
            }
        }
#pragma unroll
        for (int h2 = 0; h2 < 2; ++h2) {
            const float d = quad_sum(dots[h2]);
            const unsigned int r = m0 + g + h2 * 8u;
            if (t4 == 0 && r < C) { atomicAdd(&sdexp[r], d); }
        }
    }

    __syncthreads();
    load_rows(sV, SK, query + row_base * D, D, C);   // d_w is dead: the query takes its place

    // ---- d_raw_intra = d_intra * decay_mask, d_decay_mask = d_intra * raw_intra ----
#pragma unroll
    for (unsigned int it = 0; it < MAXC * MAXC / THREADS; ++it) {  // unrolled so the global loads of the iterations overlap
        const unsigned int e = tid + it * THREADS;
        if (e < C * C) {
            const unsigned int i = e / C, j = e % C;
            if (j <= i) {
                const float di = sDI[i * SC + j];
                sDI[i * SC + j] = di * expf(sg[i] - sg[j]);
                sDM[i * SC + j] = di * raw_intra[cc_base + e];
            } else {
                sDI[i * SC + j] = 0.0f;
            }
        }
    }
    __syncthreads();

    // ---- d_query += scale * d_raw_intra @ key and d_key += scale * d_raw_intra^T @ query ----
    {
        float acc[8][4];
        zero_acc<8>(acc);
        if (m0 < C) {
            warp_mma_acc<8>(acc, c8, m0, nh * 64u, [&](unsigned int i, unsigned int j) { return sDI[i * SC + j]; }, [&](unsigned int j, unsigned int p) { return sK[j * SK + p]; });
        }
#pragma unroll
        for (int t = 0; t < 8; ++t) {
#pragma unroll
            for (int h2 = 0; h2 < 2; ++h2) {
                const unsigned int r = m0 + g + h2 * 8u;
                if (r < C) {
                    const float2 v = make_float2(dq[t][2 * h2] + scale * acc[t][2 * h2], dq[t][2 * h2 + 1] + scale * acc[t][2 * h2 + 1]);
                    *reinterpret_cast<float2*>(d_query + (row_base + r) * D + nh * 64u + t * 8 + 2 * t4) = v;
                }
            }
        }
        zero_acc<8>(acc);
        if (m0 < C) {
            warp_mma_acc<8>(acc, c8, m0, nh * 64u, [&](unsigned int j, unsigned int i) { return sDI[i * SC + j]; }, [&](unsigned int i, unsigned int p) { return sV[i * SK + p]; });
        }
#pragma unroll
        for (int t = 0; t < 8; ++t) {
#pragma unroll
            for (int h2 = 0; h2 < 2; ++h2) {
                const unsigned int r = m0 + g + h2 * 8u;
                if (r < C) {
                    dk[t][2 * h2] += scale * acc[t][2 * h2];
                    dk[t][2 * h2 + 1] += scale * acc[t][2 * h2 + 1];
                }
            }
        }
    }
    __syncthreads();

    // ---- d_attn0 = T^T @ d_t_mat @ T^T, the adjoint of the inverse ----
    {
        float acc[4][4];
        zero_acc<4>(acc);
        if (m0 < C && nh * 32u < C) {
            warp_mma_acc<4>(acc, c8, m0, nh * 32u, [&](unsigned int i, unsigned int k) { return sT[k * SC + i]; }, [&](unsigned int k, unsigned int j) { return sDT[k * SC + j]; });
        }
        __syncthreads();
#pragma unroll
        for (int t = 0; t < 4; ++t) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const unsigned int r = m0 + g + ((i & 2) ? 8u : 0u);
                const unsigned int col = nh * 32u + t * 8 + 2 * t4 + (i & 1);
                sDI[r * SC + col] = (r < C && col < C) ? acc[t][i] : 0.0f;
            }
        }
    }
    __syncthreads();
    {
        float acc[4][4];
        zero_acc<4>(acc);
        if (m0 < C && nh * 32u < C) {
            warp_mma_acc<4>(acc, c8, m0, nh * 32u, [&](unsigned int i, unsigned int k) { return sDI[i * SC + k]; }, [&](unsigned int k, unsigned int j) { return sT[j * SC + k]; });
        }
        // The strictly lower part is what attn0 owns: d_raw_attn0 = d_attn0 * decay_mask there, and the
        // cells' d_decay_mask gains d_attn0 * raw_attn0. Everything else is structurally constant.
#pragma unroll
        for (int t = 0; t < 4; ++t) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const unsigned int r = m0 + g + ((i & 2) ? 8u : 0u);
                const unsigned int col = nh * 32u + t * 8 + 2 * t4 + (i & 1);
                float dra = 0.0f;
                if (r < C && col < r) {
                    dra = acc[t][i] * expf(sg[r] - sg[col]);
                    sDM[r * SC + col] += acc[t][i] * raw_attn0[cc_base + (unsigned long long)r * C + col];
                }
                sDT[r * SC + col] = dra;
            }
        }
    }
    __syncthreads();

    // ---- decay_mask[i][j] = exp(g_cs[i] - g_cs[j]) backward: a row sum and a column sum ----
    if (tid < C) {
        float acc = 0.0f;
        for (unsigned int j = 0; j <= tid; ++j) { acc += sDM[tid * SC + j] * expf(sg[tid] - sg[j]); }
        atomicAdd(&sdg[tid], acc);
    } else if (tid >= MAXC && tid < MAXC + C) {
        const unsigned int j = tid - MAXC;
        float acc = 0.0f;
        for (unsigned int i = j; i < C; ++i) { acc += sDM[i * SC + j] * expf(sg[i] - sg[j]); }
        atomicAdd(&sdg[j], -acc);
    }

    // ---- raw_attn0 = -(k_beta @ key^T) backward: d_k_beta -= d_raw_attn0 @ key, d_key -= d_raw_attn0^T @ k_beta ----
    {
        float acc[8][4];
        zero_acc<8>(acc);
        if (m0 < C) {
            warp_mma_acc<8>(acc, c8, m0, nh * 64u, [&](unsigned int i, unsigned int j) { return sDT[i * SC + j]; }, [&](unsigned int j, unsigned int p) { return sK[j * SK + p]; });
        }
#pragma unroll
        for (int t = 0; t < 8; ++t) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const unsigned int r = m0 + g + ((i & 2) ? 8u : 0u);
                if (r < C) { sO[r * SK + nh * 64u + t * 8 + 2 * t4 + (i & 1)] -= acc[t][i]; }
            }
        }
        zero_acc<8>(acc);
        if (m0 < C) {
            warp_mma_acc<8>(acc, c8, m0, nh * 64u, [&](unsigned int j, unsigned int i) { return sDT[i * SC + j]; },
                            [&](unsigned int i, unsigned int p) { return sK[i * SK + p] * sbeta[i]; });
        }
#pragma unroll
        for (int t = 0; t < 8; ++t) {
#pragma unroll
            for (int h2 = 0; h2 < 2; ++h2) {
                const unsigned int r = m0 + g + h2 * 8u;
                if (r < C) {
                    dk[t][2 * h2] -= acc[t][2 * h2];
                    dk[t][2 * h2 + 1] -= acc[t][2 * h2 + 1];
                }
            }
        }
    }
    __syncthreads();

    // ---- k_beta = key * beta backward: d_key += beta * d_k_beta, d_beta += rowdot(d_k_beta, key) ----
    {
        float dots[2] = {0.0f, 0.0f};
#pragma unroll
        for (int t = 0; t < 8; ++t) {
#pragma unroll
            for (int h2 = 0; h2 < 2; ++h2) {
                const unsigned int r = m0 + g + h2 * 8u;
                const unsigned int col = nh * 64u + t * 8 + 2 * t4;
                if (r < C) {
                    const float a0 = sO[r * SK + col], a1 = sO[r * SK + col + 1];
                    *reinterpret_cast<float2*>(d_key + (row_base + r) * D + col) = make_float2(dk[t][2 * h2] + a0 * sbeta[r], dk[t][2 * h2 + 1] + a1 * sbeta[r]);
                    dots[h2] += a0 * sK[r * SK + col] + a1 * sK[r * SK + col + 1];
                }
            }
        }
#pragma unroll
        for (int h2 = 0; h2 < 2; ++h2) {
            const float d = quad_sum(dots[h2]);
            const unsigned int r = m0 + g + h2 * 8u;
            if (t4 == 0 && r < C) { atomicAdd(&sdb[r], d); }
        }
    }
    __syncthreads();

    // ---- d_g_cs from its remaining sources, then the suffix sum that is the cumulative sum's adjoint ----
    if (tid < C) {
        sdg[tid] += sdexp[tid] * sexp[tid] - sdds[tid] * sds[tid];
        d_beta[scal_base + tid * scal_stride] = sdb[tid];
    }
    __syncthreads();
    if (tid == 0) {
        // The last cumulative-decay row also receives the decay-scale total and the state-decay term.
        float total = 0.0f, state_term = 0.0f;
        for (unsigned int j = 0; j < C; ++j) { total += sdds[j] * sds[j]; }
        for (unsigned int i = 0; i < THREADS / 32; ++i) { state_term += sred[i]; }
        sdg[C - 1] += total + state_term * expf(sg[C - 1]);
        float running = 0.0f;
        for (int i = (int)C - 1; i >= 0; --i) {
            running += sdg[i];
            d_raw_g[scal_base + (unsigned long long)i * scal_stride] = running;
        }
    }
}
