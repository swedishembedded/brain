// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements tensor-core training kernels for linear-
// attention language models for its clients. If your team needs expertise in
// turning a long chain of small memory-bound launches over a recurrent layer
// into one fused tensor-core kernel without changing the model's numerics,
// then you can procure our services by sending an email to
// info@swedishembedded.com.
//
// The state-independent half of Gated DeltaNet's chunked forward
// (`model::gdn`, steps 1-9 and the intra-chunk score precompute) in ONE launch,
// one block per (head, chunk). The portable path issues about fourteen
// whole-tensor dispatches for it - two row scales, the cumulative decay, the
// decay mask, a Gram product, the strict-lower mask, the unit-triangular
// solve, three more products and the intra-chunk scores - each streaming its
// operands through HBM again. Here a chunk's key, query and value rows are
// read once, every intermediate that is not an output stays in shared memory,
// and the four matrix products run on tensor cores.
//
//   g_cs        = cumsum(raw_g)                    exp_g_cs = exp(g_cs)
//   decay[i][j] = exp(g_cs[i] - g_cs[j])           (j <= i, else 0)
//   attn0       = strict_lower(-(beta_i * key_i . key_j) * decay)
//   T           = (I - attn0)^-1                   (forward substitution, fp32)
//   u           = T @ (value * beta)
//   w           = T @ (key * beta * exp_g_cs)
//   intra       = scale * (query @ key^T) * decay
//
// The recurrence over chunks (`gdn_chunk_loop_f32`) reads w, u, g_cs, exp_g_cs
// and intra; the training entry also writes what the backward reads (k_beta,
// v_beta, k_beta_decay, the decay mask, raw_attn0, t_mat and raw_intra), so
// the backward is unchanged.
//
// NUMERICS. The products run as three TF32 `mma.sync.m16n8k8` per tile
// (a_hi*b_hi + a_hi*b_lo + a_lo*b_hi, each operand split into a TF32 head and a
// TF32 remainder), which keeps about 20 of fp32's 24 mantissa bits with fp32
// accumulation: the result is NOT bit-identical to the fp32 kernels it
// replaces, but agrees to a few units in 1e-6 relative to the operand scale.
// The unit-triangular solve is fp32 in shared memory, never the Neumann
// series: that one cancels catastrophically for near-parallel keys (see
// `model::gdn`'s note at its step 7). The cumulative decay is the same serial
// fp32 sum as `gdn_chunk_cumsum_step`, so g_cs is bit-identical.
//
// TOKEN-MAJOR form (`TOK`, the `_tok_` entries). The mixer feeds the chunk forward
// through five more dispatches per layer pass: an L2 normalisation of query and
// key, the repeat of their heads from `nkh` to `nvh` and a token-major to
// chunk-major permute of the five operands. The `_tok_` entries fold all of it
// into the block's loads: query and key are read in their narrow token-major
// layout (head `h / group`), normalised row by row in shared memory
// (`x * rsqrt(sum x^2 + eps)`, the `l2norm_scale` formula), and value, beta and
// the raw decay are read strided; the normalised chunk-major query and key and
// the chunk-major value and beta (which the loop kernel and the backward read) are
// written from the same tiles.
//
// Contract (u32 params, then pointers; every buffer is the chunk-major layout
// `model::gdn` documents, `bhc = chunk * (B*H) + b*H + h`):
//   params : [bhc, c_len, dk, dv, unused, scale (f32 bits)]
//   read   : query, key [bhc, c_len, 128]; value [bhc, c_len, 128];
//            raw_g, beta [bhc, c_len]
//   write  : g_cs, exp_g_cs [bhc, c_len]; u, w [bhc, c_len, 128];
//            intra_scores [bhc, c_len, c_len]
//   TRAIN  : k_beta, v_beta, k_beta_decay [bhc, c_len, 128];
//            decay_mask, raw_attn0, t_mat, raw_intra [bhc, c_len, c_len]
//   TOK    : params gain [H, n_chunks, group, eps (f32 bits)] at 6..9;
//            query, key [B*T, (H/group)*128] (pre-normalisation), value [B*T, H*128],
//            raw_g, beta [B*T, H] token-major (T = n_chunks * c_len); the entry also
//            writes query_cm, key_cm [bhc, c_len, 128] and, in the training form, value_cm
//            [bhc, c_len, 128] and beta_cm [bhc, c_len].
// dk = dv = 128 and 1 <= c_len <= 64 (the provider checks; the entry does not).
// Dynamic shared memory: PREP_SMEM_BYTES.
//
// No `__restrict__`: brain's device buffers alias by design.

#define D 128                // dk == dv
#define MAXC 64              // rows of a chunk
#define THREADS 256          // 8 warps: 4 row tiles x 2 column halves
#define SK (D + 4)           // row stride of the key tile: conflict-free as a row-major A / column-major B operand
#define SX (D + 8)           // row stride of the query / value tile: conflict-free as a k-major B operand
#define SC (MAXC + 4)        // row stride of the 64 x 64 matrices

// Dynamic shared memory, in floats: key tile, query / value tile, attn0 and the
// solved T, then the per-row cumulative decay, its exponential and beta.
#define PREP_SMEM_FLOATS (MAXC * SK + MAXC * SX + 2 * MAXC * SC + 3 * MAXC + 3 * 8 * MAXC)

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

// One warp's 16 x (8 * NT) strip of  D(r, n) = sum_{k < K} fa(r, k) * fb(k, n),
// rows m0 .., columns n0 .., K a multiple of 8. The operands are functors so the
// scalings the algorithm folds into them (beta, exp(g)) happen on load, which
// reproduces the products the portable path forms as separate tensors.
template <int NT, class FA, class FB>
__device__ __forceinline__ void warp_gemm_3xtf32(float (&acc)[NT][4], unsigned int K, unsigned int m0, unsigned int n0, FA fa, FB fb) {
    const unsigned int lane = threadIdx.x & 31u, g = lane >> 2, t4 = lane & 3u;
#pragma unroll
    for (int t = 0; t < NT; ++t) {
#pragma unroll
        for (int i = 0; i < 4; ++i) { acc[t][i] = 0.0f; }
    }
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
            // Smallest terms first so they are not absorbed by the leading one.
            brain_mma_tf32(acc[t], al, bh0, bh1);
            brain_mma_tf32(acc[t], ah, bl0, bl1);
            brain_mma_tf32(acc[t], ah, bh0, bh1);
        }
    }
}

// Rows of a [rows, 128] source into a zero-padded [64, stride] shared tile. `src` points at chunk
// row 0; consecutive chunk rows are `row_stride` floats apart (128 for chunk-major buffers).
__device__ __forceinline__ void brain_load_rows(float* dst, unsigned int stride, const float* src, unsigned long long row_stride, unsigned int C) {
#pragma unroll
    for (unsigned int it = 0; it < MAXC * (D / 4) / THREADS; ++it) {  // unrolled so the loads of the iterations are in flight together
        const unsigned int e = threadIdx.x + it * THREADS;
        const unsigned int r = e / (D / 4), c4 = e % (D / 4);
        float4 v = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (r < C) { v = *reinterpret_cast<const float4*>(src + (unsigned long long)r * row_stride + c4 * 4); }
        *reinterpret_cast<float4*>(dst + r * stride + c4 * 4) = v;
    }
}

// L2-normalise the C rows of a shared tile in place (one warp per row, `x * rsqrt(sum x^2 + eps)`) and
// write them to the chunk-major buffer `cm`. Rows are private to their warp, so no barrier is needed
// between the read and the write-back; the caller barriers before anyone else reads the tile.
__device__ __forceinline__ void brain_normalize_rows(float* tile, unsigned int stride, float* cm, unsigned int C, float eps) {
    const unsigned int lane = threadIdx.x & 31u, warp = threadIdx.x >> 5;
    for (unsigned int r = warp; r < C; r += THREADS / 32) {
        float4 v = *reinterpret_cast<const float4*>(tile + r * stride + lane * 4);
        float s = v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) { s += __shfl_xor_sync(0xffffffffu, s, o); }
        const float inv = rsqrtf(s + eps);
        v = make_float4(v.x * inv, v.y * inv, v.z * inv, v.w * inv);
        *reinterpret_cast<float4*>(tile + r * stride + lane * 4) = v;
        *reinterpret_cast<float4*>(cm + r * (unsigned long long)D + lane * 4) = v;
    }
}

template <bool TRAIN, bool TOK>
__device__ __forceinline__ void brain_gdn_chunk_prep_body(const unsigned int* params, const float* query, const float* key, const float* value,
                                                          const float* raw_g, const float* beta, float* g_cs_out, float* exp_g_cs_out, float* u_out,
                                                          float* w_out, float* intra_out, float* k_beta_out, float* v_beta_out, float* k_beta_decay_out,
                                                          float* decay_mask_out, float* raw_attn0_out, float* t_mat_out, float* raw_intra_out, float* query_cm_out,
                                                          float* key_cm_out, float* value_cm_out, float* beta_cm_out) {
    const unsigned int bhc = params[0];
    const unsigned int C = params[1];
    const float scale = __uint_as_float(params[5]);

    const unsigned int blk = blockIdx.y * gridDim.x + blockIdx.x;
    if (blk >= bhc) { return; }  // block-uniform

    extern __shared__ float smem[];
    float* sK = smem;                       // key rows, [64][SK]
    float* sX = sK + MAXC * SK;             // query rows, then value rows, [64][SX]
    float* sA = sX + MAXC * SX;             // attn0, [64][SC]
    float* sT = sA + MAXC * SC;             // T = (I - attn0)^-1, [64][SC]
    float* sg = sT + MAXC * SC;             // cumulative decay
    float* sexp = sg + MAXC;                // exp of it
    float* sbeta = sexp + MAXC;
    float* sRed = sbeta + MAXC;             // partial sums of the T solve's panels, [3][8][64]

    const unsigned int tid = threadIdx.x, lane = tid & 31u, warp = tid >> 5;
    const unsigned int g = lane >> 2, t4 = lane & 3u;
    const unsigned long long row_base = (unsigned long long)blk * C;   // first row of this (chunk, head)
    const unsigned long long cc_base = row_base * C;                    // first element of its c x c matrices

    // Token-major sources: this block's first token row, head, and the row/column geometry of each tensor.
    unsigned long long tok0 = 0, beta_stride = 1, qk_stride = D, v_stride = D;
    unsigned int head = 0, src_head = 0;
    float eps = 0.0f;
    if (TOK) {
        const unsigned int H = params[6], n_chunks = params[7], group = params[8];
        const unsigned int bh = bhc / n_chunks, ci = blk / bh, rem = blk % bh;
        head = rem % H;
        src_head = head / group;
        tok0 = ((unsigned long long)(rem / H) * n_chunks + ci) * C;
        beta_stride = H;
        qk_stride = (unsigned long long)(H / group) * D;
        v_stride = (unsigned long long)H * D;
        eps = __uint_as_float(params[9]);
    }
    const float* beta_src = TOK ? beta + tok0 * beta_stride + head : beta + row_base;
    const float* g_src = TOK ? raw_g + tok0 * beta_stride + head : raw_g + row_base;
    const float* key_src = TOK ? key + tok0 * qk_stride + (unsigned long long)src_head * D : key + row_base * D;
    const float* query_src = TOK ? query + tok0 * qk_stride + (unsigned long long)src_head * D : query + row_base * D;
    const float* value_src = TOK ? value + tok0 * v_stride + (unsigned long long)head * D : value + row_base * D;

    for (unsigned int i = tid; i < MAXC * SC; i += THREADS) {
        sA[i] = 0.0f;
        sT[i] = 0.0f;
    }
    if (tid < MAXC) {
        sbeta[tid] = tid < C ? beta_src[tid * beta_stride] : 0.0f;
        sg[tid] = tid < C ? g_src[tid * beta_stride] : 0.0f;
        if (TOK && TRAIN && tid < C) { beta_cm_out[row_base + tid] = sbeta[tid]; }
    }
    __syncthreads();
    // The serial sum `gdn_chunk_cumsum_step` performs, so g_cs is bit-identical.
    if (tid == 0) {
        for (unsigned int i = 1; i < C; ++i) { sg[i] = sg[i] + sg[i - 1]; }
    }
    __syncthreads();
    if (tid < MAXC) {
        const float e = expf(sg[tid]);
        sexp[tid] = e;
        if (tid < C) {
            g_cs_out[row_base + tid] = sg[tid];
            exp_g_cs_out[row_base + tid] = e;
        }
    }
    __syncthreads();

    brain_load_rows(sK, SK, key_src, qk_stride, C);
    if (TOK) {
        __syncthreads();
        brain_normalize_rows(sK, SK, key_cm_out + row_base * D, C, eps);
    }
    __syncthreads();
    if (TRAIN) {
        // k_beta = key * beta and k_beta_decay = k_beta * exp(g_cs), as the portable row scales form them.
        for (unsigned int e = tid; e < C * (D / 4); e += THREADS) {
            const unsigned int r = e / (D / 4), c4 = e % (D / 4);
            const float4 k4 = *reinterpret_cast<const float4*>(sK + r * SK + c4 * 4);
            const float b = sbeta[r], x = sexp[r];
            const float4 kb = make_float4(k4.x * b, k4.y * b, k4.z * b, k4.w * b);
            *reinterpret_cast<float4*>(k_beta_out + (row_base + r) * D + c4 * 4) = kb;
            *reinterpret_cast<float4*>(k_beta_decay_out + (row_base + r) * D + c4 * 4) = make_float4(kb.x * x, kb.y * x, kb.z * x, kb.w * x);
        }
    }
    __syncthreads();

    const unsigned int mt = warp & 3u;          // this warp's 16-row tile
    const unsigned int nh = warp >> 2;          // and which half of the columns
    const unsigned int m0 = mt * 16u;
    const unsigned int c8 = (C + 7u) & ~7u;     // contraction length over chunk rows

    // ---- attn0 = strict_lower(-(k_beta @ key^T) * decay), the Gram product ----
    if (m0 < C && nh * 32u < C) {
        float acc[4][4];
        warp_gemm_3xtf32<4>(
            acc, D, m0, nh * 32u, [&](unsigned int r, unsigned int k) { return sK[r * SK + k] * sbeta[r]; },
            [&](unsigned int k, unsigned int n) { return sK[n * SK + k]; });
#pragma unroll
        for (int t = 0; t < 4; ++t) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const unsigned int r = m0 + g + ((i & 2) ? 8u : 0u);
                const unsigned int col = nh * 32u + t * 8 + 2 * t4 + (i & 1);
                if (r < C && col < C) {
                    const float raw = -acc[t][i];
                    const float dm = col <= r ? expf(sg[r] - sg[col]) : 0.0f;
                    sA[r * SC + col] = col < r ? raw * dm : 0.0f;
                    if (TRAIN) {
                        raw_attn0_out[cc_base + (unsigned long long)r * C + col] = raw;
                        decay_mask_out[cc_base + (unsigned long long)r * C + col] = dm;
                    }
                }
            }
        }
    }
    brain_load_rows(sX, SX, query_src, qk_stride, C);
    if (TOK) {
        __syncthreads();
        brain_normalize_rows(sX, SX, query_cm_out + row_base * D, C, eps);
    }
    __syncthreads();

    // ---- T = (I - attn0)^-1 (unit lower triangular, fp32; `brain_unit_lower_inverse`) ----
    brain_unit_lower_inverse(sA, sT, sRed, SC, C);
    if (TRAIN) {
        for (unsigned int e = tid; e < C * C; e += THREADS) {
            const unsigned int i = e / C, j = e % C;
            t_mat_out[cc_base + e] = sT[i * SC + j];
        }
    }

    // ---- intra = scale * (query @ key^T) * decay ----
    if (m0 < C && nh * 32u < C) {
        float acc[4][4];
        warp_gemm_3xtf32<4>(
            acc, D, m0, nh * 32u, [&](unsigned int r, unsigned int k) { return sX[r * SX + k]; }, [&](unsigned int k, unsigned int n) { return sK[n * SK + k]; });
#pragma unroll
        for (int t = 0; t < 4; ++t) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const unsigned int r = m0 + g + ((i & 2) ? 8u : 0u);
                const unsigned int col = nh * 32u + t * 8 + 2 * t4 + (i & 1);
                if (r < C && col < C) {
                    const float raw = scale * acc[t][i];
                    const float dm = col <= r ? expf(sg[r] - sg[col]) : 0.0f;
                    intra_out[cc_base + (unsigned long long)r * C + col] = raw * dm;
                    if (TRAIN) { raw_intra_out[cc_base + (unsigned long long)r * C + col] = raw; }
                }
            }
        }
    }
    __syncthreads();  // the query tile is dead: the value rows take its place

    brain_load_rows(sX, SX, value_src, v_stride, C);
    __syncthreads();
    if (TRAIN) {
        for (unsigned int e = tid; e < C * (D / 4); e += THREADS) {
            const unsigned int r = e / (D / 4), c4 = e % (D / 4);
            const float4 v4 = *reinterpret_cast<const float4*>(sX + r * SX + c4 * 4);
            if (TOK) { *reinterpret_cast<float4*>(value_cm_out + (row_base + r) * D + c4 * 4) = v4; }
            if (TRAIN) {
                const float b = sbeta[r];
                *reinterpret_cast<float4*>(v_beta_out + (row_base + r) * D + c4 * 4) = make_float4(v4.x * b, v4.y * b, v4.z * b, v4.w * b);
            }
        }
        __syncthreads();
    }

    // ---- u = T @ (value * beta) and w = T @ (key * beta * exp_g_cs): 64 x 128 outputs, K = chunk rows ----
    if (m0 < C) {
        float acc[8][4];
        const unsigned int n0 = nh * 64u;
#pragma unroll
        for (int pass = 0; pass < 2; ++pass) {
            if (pass == 0) {
                warp_gemm_3xtf32<8>(
                    acc, c8, m0, n0, [&](unsigned int r, unsigned int k) { return sT[r * SC + k]; },
                    [&](unsigned int k, unsigned int n) { return sX[k * SX + n] * sbeta[k]; });
            } else {
                warp_gemm_3xtf32<8>(
                    acc, c8, m0, n0, [&](unsigned int r, unsigned int k) { return sT[r * SC + k]; },
                    [&](unsigned int k, unsigned int n) { return (sK[k * SK + n] * sbeta[k]) * sexp[k]; });
            }
            float* dst = pass == 0 ? u_out : w_out;
#pragma unroll
            for (int t = 0; t < 8; ++t) {
#pragma unroll
                for (int h2 = 0; h2 < 2; ++h2) {
                    const unsigned int r = m0 + g + h2 * 8u;
                    if (r < C) {
                        *reinterpret_cast<float2*>(dst + (row_base + r) * D + n0 + t * 8 + 2 * t4) = make_float2(acc[t][2 * h2], acc[t][2 * h2 + 1]);
                    }
                }
            }
        }
    }
}

extern "C" __global__ void __launch_bounds__(THREADS) brain_gdn_chunk_prep_tc(const unsigned int* params, const float* query, const float* key,
                                                                               const float* value, const float* raw_g, const float* beta, float* g_cs,
                                                                               float* exp_g_cs, float* u, float* w, float* intra_scores) {
    brain_gdn_chunk_prep_body<false, false>(params, query, key, value, raw_g, beta, g_cs, exp_g_cs, u, w, intra_scores, nullptr, nullptr, nullptr, nullptr,
                                            nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
}

// Training form: the same launch plus the tensors the backward reads.
extern "C" __global__ void __launch_bounds__(THREADS) brain_gdn_chunk_prep_train_tc(const unsigned int* params, const float* query, const float* key,
                                                                                     const float* value, const float* raw_g, const float* beta, float* g_cs,
                                                                                     float* exp_g_cs, float* u, float* w, float* intra_scores, float* k_beta,
                                                                                     float* v_beta, float* k_beta_decay, float* decay_mask,
                                                                                     float* raw_attn0, float* t_mat, float* raw_intra) {
    brain_gdn_chunk_prep_body<true, false>(params, query, key, value, raw_g, beta, g_cs, exp_g_cs, u, w, intra_scores, k_beta, v_beta, k_beta_decay,
                                           decay_mask, raw_attn0, t_mat, raw_intra, nullptr, nullptr, nullptr, nullptr);
}

// Token-major inference form: query_cm and key_cm are what the recurrence reads.
extern "C" __global__ void __launch_bounds__(THREADS) brain_gdn_chunk_prep_tok_tc(const unsigned int* params, const float* query, const float* key,
                                                                                    const float* value, const float* raw_g, const float* beta, float* g_cs,
                                                                                    float* exp_g_cs, float* u, float* w, float* intra_scores,
                                                                                    float* query_cm, float* key_cm) {
    brain_gdn_chunk_prep_body<false, true>(params, query, key, value, raw_g, beta, g_cs, exp_g_cs, u, w, intra_scores, nullptr, nullptr, nullptr, nullptr,
                                           nullptr, nullptr, nullptr, query_cm, key_cm, nullptr, nullptr);
}

// Token-major training form: the inputs are the mixer's own token-major tensors and the chunk-major
// query, key, value and beta the loop and the backward read are written alongside.
extern "C" __global__ void __launch_bounds__(THREADS) brain_gdn_chunk_prep_tok_train_tc(
    const unsigned int* params, const float* query, const float* key, const float* value, const float* raw_g, const float* beta, float* g_cs, float* exp_g_cs,
    float* u, float* w, float* intra_scores, float* k_beta, float* v_beta, float* k_beta_decay, float* decay_mask, float* raw_attn0, float* t_mat,
    float* raw_intra, float* query_cm, float* key_cm, float* value_cm, float* beta_cm) {
    brain_gdn_chunk_prep_body<true, true>(params, query, key, value, raw_g, beta, g_cs, exp_g_cs, u, w, intra_scores, k_beta, v_beta, k_beta_decay, decay_mask,
                                          raw_attn0, t_mat, raw_intra, query_cm, key_cm, value_cm, beta_cm);
}
