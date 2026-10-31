// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements tensor-core inference kernels for linear-attention language models for its clients.
// If your team needs expertise in preparing the chunk operands of a recurrent layer at the card's bf16 tensor-core
// rate, then you can procure our services by sending an email to info@swedishembedded.com.
//
// The state-independent half of Gated DeltaNet's chunked forward (see `gdn_chunk_prep_tc.cu` for the algorithm) for the
// serving prefill, in the form the bf16 recurrence `gdn_chunk_loop_bf16` reads: one block per (head, chunk), the
// mixer's token-major tensors read directly (query/key L2 normalisation, head repeat and the chunk-major permutes folded
// into the loads), every product on bf16 `mma.sync.m16n8k16` with fp32 accumulation, and the matrices the recurrence
// multiplies written as bf16, rounded once where they are made:
//
//   g_cs      = cumsum(raw_g)                      exp_g_cs = exp(g_cs)     (fp32, as the TF32 kernel)
//   attn0     = strict_lower(-beta_i (k_i . k_j) exp(g_cs_i - g_cs_j))      T = (I - attn0)^-1
//   u         = (T diag(beta)) @ value                                      (fp32 out)
//   w         = (T diag(beta exp_g_cs)) @ key                               (bf16 out)
//   intra     = scale (q_i . k_j) exp(g_cs_i - g_cs_j), j <= i              (bf16 out)
//   query_cm  = scale * q * exp_g_cs                (bf16, the recurrence's inter-chunk operand)
//   key_cm    = k * exp(g_last - g_cs)              (bf16, the recurrence's state-update operand)
//
// u and w are the solutions X of (I - attn0) X = diag(s) R (R the value or the key rows, s beta or beta exp_g_cs), found by
// blocked forward substitution over the four 16-row blocks of the chunk instead of forming T and multiplying by it:
//
//   X_i = (T_ii diag(s_i)) R_i + sum_{j<i} (T_ii attn0_ij) X_j ,        T_ii = (I - attn0_ii)^-1
//
// The 16x16 diagonal inverses and the off-diagonal products T_ii attn0_ij are fp32 forward substitutions (one thread a
// column; never the Neumann series, which cancels catastrophically for near-parallel keys), stored split into a bf16 high
// and low part; the block rows are then bf16 `mma` products with fp32 accumulation held transposed (X^T, value columns as
// the M dimension) so that the finished accumulators of one block row are, repacked, the A operand of the next one's
// products. The finished X_j are split into high and low parts too, so the products keep about fp32 accuracy and the
// result is rounded to bf16 only where it is stored (w). u keeps fp32 (it is subtracted from a product in the recurrence).
//
// Contract: params [bhc, c_len, dk, dv, unused, scale (f32 bits), H, n_chunks, group, eps (f32 bits)]; buffers as
// `gdn_chunk_prep_tok_tc` (query, key, value, raw_g, beta read; g_cs, exp_g_cs, u, w, intra_scores, query_cm, key_cm
// written). dk = dv = 128 and 1 <= c_len <= 64. Dynamic shared memory: PREP_BF16_SMEM_BYTES.
// No `__restrict__`: brain's device buffers alias by design.

#define D 128
#define MAXC 64
#define THREADS 256
#define SKH (D + 8)       // bf16 row stride of the key / query / value tiles (ldmatrix rows 16 B apart mod 128 B)
#define SC (MAXC + 4)     // fp32 row stride of attn0
#define STP 24            // bf16 row stride of the 16 x 16 solve matrices (conflict-free 32-bit fragment loads)
#define SOLVE_MATS (2 * 6 + 4 * 2 * 2)  // hi and low parts of the six T_ii attn0_ij, then (u, w) x (hi, low) of the four T_ii diag(s)

#define PREP_BF16_SMEM_BYTES (2 * MAXC * SKH * 2 + MAXC * SC * 4 + 4 * MAXC * 4 + SOLVE_MATS * 16 * STP * 2)

// Four consecutive elements of a token-major tensor at element offset `at`, widened to f32: the tensor is bf16 rows when `bf16` (the
// serving segments' form, written by the convolution) and f32 rows otherwise.
__device__ __forceinline__ float4 brain_prep_load4(const float* base, unsigned long long at, bool bf16) {
    if (!bf16) { return *reinterpret_cast<const float4*>(base + at); }
    const uint2 w = *reinterpret_cast<const uint2*>(reinterpret_cast<const unsigned short*>(base) + at);
    return make_float4(__uint_as_float(w.x << 16), __uint_as_float(w.x & 0xffff0000u), __uint_as_float(w.y << 16), __uint_as_float(w.y & 0xffff0000u));
}

// One chunk row per warp: the row of `src` (fp32, stride `stride`) is L2-normalised (`x * rsqrt(sum x^2 + eps)`, the
// formula of `gdn_chunk_prep_tc`), written to the bf16 tile `tile` and, scaled by `row_scale[r] * extra`, to the
// chunk-major bf16 buffer `cm`. Rows past C of the tile are zero.
__device__ __forceinline__ void brain_normalize_rows_bf16(unsigned short* tile, const float* src, unsigned long long src_at, bool src_bf16, unsigned long long stride, unsigned short* cm,
                                                          unsigned int C, float eps, const float* row_scale, float extra) {
    const unsigned int lane = threadIdx.x & 31u, warp = threadIdx.x >> 5;
    for (unsigned int r = warp; r < MAXC; r += THREADS / 32) {
        uint2 t = make_uint2(0u, 0u);
        if (r < C) {
            float4 v = brain_prep_load4(src, src_at + r * stride + lane * 4, src_bf16);
            float s = v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) { s += __shfl_xor_sync(0xffffffffu, s, o); }
            const float inv = rsqrtf(s + eps);
            v = make_float4(v.x * inv, v.y * inv, v.z * inv, v.w * inv);
            t = make_uint2(brain_pack_bf16(v.x, v.y), brain_pack_bf16(v.z, v.w));
            const float m = extra * row_scale[r];
            *reinterpret_cast<uint2*>(cm + r * (unsigned long long)D + lane * 4) = make_uint2(brain_pack_bf16(v.x * m, v.y * m), brain_pack_bf16(v.z * m, v.w * m));
        }
        *reinterpret_cast<uint2*>(tile + r * SKH + lane * 4) = t;
    }
}

// One 16-byte `cp.async` into shared memory; `bytes` = 0 zero-fills (rows past a short chunk).
__device__ __forceinline__ void brain_cp16(void* dst, const void* src, unsigned int bytes) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(brain_smem_addr(dst)), "l"(src), "r"(bytes) : "memory");
}

// Rows of normalised bf16 `src` (stride `stride`, element offset `src_at`) into `tile` (the rows past C zero) by `cp.async`, one commit group.
__device__ __forceinline__ void brain_copy_rows_bf16(unsigned short* tile, const float* src, unsigned long long src_at, unsigned long long stride, unsigned int C) {
    const unsigned short* const s = reinterpret_cast<const unsigned short*>(src);
    const unsigned int c = threadIdx.x & 15u, r0 = threadIdx.x >> 4;  // a thread keeps one 16-byte column; each pass moves 16 rows
#pragma unroll
    for (unsigned int it = 0; it < MAXC / (THREADS / 16); ++it) {
        const unsigned int r = r0 + it * (THREADS / 16);
        brain_cp16(tile + r * SKH + c * 8, s + (r < C ? src_at + r * stride + c * 8 : 0), r < C ? 16u : 0u);
    }
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}

// The chunk-major scaled copy `cm = bf16(tile * (extra * row_scale[r]))` of a tile of normalised rows, 16 bytes a thread.
__device__ __forceinline__ void brain_scale_tile_rows(const unsigned short* tile, unsigned short* cm, unsigned int C, const float* row_scale, float extra) {
    for (unsigned int e = threadIdx.x; e < MAXC * (D / 8); e += THREADS) {
        const unsigned int r = e >> 4, c = e & 15u;
        if (r < C) {
            const uint4 v = *reinterpret_cast<const uint4*>(tile + r * SKH + c * 8);
            const float m = extra * row_scale[r];
            *reinterpret_cast<uint4*>(cm + r * (unsigned long long)D + c * 8) =
                make_uint4(brain_pack_bf16(__uint_as_float(v.x << 16) * m, __uint_as_float(v.x & 0xffff0000u) * m), brain_pack_bf16(__uint_as_float(v.y << 16) * m, __uint_as_float(v.y & 0xffff0000u) * m),
                           brain_pack_bf16(__uint_as_float(v.z << 16) * m, __uint_as_float(v.z & 0xffff0000u) * m), brain_pack_bf16(__uint_as_float(v.w << 16) * m, __uint_as_float(v.w & 0xffff0000u) * m));
        }
    }
}

// The tensors one chunk's prep reads and writes, and the head geometry that addresses them.
struct PrepOperands {
    const float *query, *key, *value, *raw_g, *beta;
    float *g_cs_out, *exp_g_cs_out, *u_out, *w_out, *intra_out, *query_cm_out, *key_cm_out;
    // Non-null: `raw_g` and `beta` hold the raw projections a_proj and b_proj, and the gates are formed in the loads
    // (beta = sigmoid(b_proj), raw_g = -exp(a_log) * softplus(a_proj + dt_bias), per value head).
    const float *a_log, *dt_bias;
    float scale, eps;
    unsigned int H, group;
    // query, key and value are bf16 rows (the convolution's bf16 output) rather than f32 rows.
    bool qkv_bf16;
    // query and key rows arrive L2-normalised (the vector convolution's output): they are copied, not normalised (needs `qkv_bf16`).
    bool qk_normalized;
    // Write the scaled chunk-major query / key copies the `mma.sync` recurrence multiplies (the warpgroup recurrence scales the normalised rows itself).
    bool emit_cm;
};

// A value as its bf16 high part and the bf16 rounding of the remainder: hi + lo carries about 16 mantissa bits.
__device__ __forceinline__ void brain_split_bf16(float x, unsigned short& hi, unsigned short& lo) {
    const unsigned int p = brain_pack_bf16(x, 0.0f);
    hi = static_cast<unsigned short>(p);
    lo = static_cast<unsigned short>(brain_pack_bf16(x - __uint_as_float(p << 16), 0.0f));
}

// The A operand (16 x 16) a finished pair of m16n8 accumulator tiles is: the tile pair (16 value columns x 16 chunk rows) read as
// (M = column, K = row), split into high and low bf16 parts.
__device__ __forceinline__ void brain_acc_to_a_split(const float (&t0)[4], const float (&t1)[4], unsigned int (&hi)[4], unsigned int (&lo)[4]) {
    const float v[8] = {t0[0], t0[1], t0[2], t0[3], t1[0], t1[1], t1[2], t1[3]};
    // Register order of the A fragment: (row g, k 0-7), (row g + 8, k 0-7), (row g, k 8-15), (row g + 8, k 8-15).
#pragma unroll
    for (int r = 0; r < 4; ++r) {
        const unsigned int h = brain_pack_bf16(v[2 * r], v[2 * r + 1]);
        hi[r] = h;
        lo[r] = brain_pack_bf16(v[2 * r] - __uint_as_float(h << 16), v[2 * r + 1] - __uint_as_float(h & 0xffff0000u));
    }
}

// Two floats as fp16 halves (`lo` in the low half), round to nearest even.
__device__ __forceinline__ unsigned int brain_pack_f16(float lo, float hi) {
    unsigned int d;
    asm("cvt.rn.f16x2.f32 %0, %1, %2;\n" : "=r"(d) : "f"(hi), "f"(lo));
    return d;
}

__device__ __forceinline__ float brain_ex2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;\n" : "=f"(y) : "f"(x));
    return y;
}

// One block's work: the `C` rows (token rows `tok0 ..` of the token-major tensors) of value head `head`. The chunk-major
// results land at row `row_base` of the scratch tensors (and `intra` at `cc_base`); the caller decides that layout.
__device__ __forceinline__ void brain_gdn_prep_chunk(const PrepOperands& a, const unsigned int C, const unsigned long long row_base, const unsigned long long cc_base,
                                                     const unsigned long long tok0, const unsigned int head) {
    const float scale = a.scale, eps = a.eps;
    const unsigned int H = a.H, group = a.group;
    float *g_cs_out = a.g_cs_out, *exp_g_cs_out = a.exp_g_cs_out, *u_out = a.u_out, *w_out = a.w_out, *intra_out = a.intra_out;
    float *query_cm_out = a.query_cm_out, *key_cm_out = a.key_cm_out;

    extern __shared__ __align__(16) unsigned char smem_raw[];
    unsigned short* sKb = reinterpret_cast<unsigned short*>(smem_raw);  // normalised keys, bf16 [64][SKH]
    unsigned short* sXb = sKb + MAXC * SKH;                               // normalised queries, then values, bf16 [64][SKH]
    float* sA = reinterpret_cast<float*>(sXb + MAXC * SKH);              // attn0, fp32 [64][SC]
    float* sg = sA + MAXC * SC;       // cumulative decay
    float* sexp = sg + MAXC;          // its exponential
    float* sbeta = sexp + MAXC;
    float* sdec = sbeta + MAXC;       // exp(g_last - g_cs)
    unsigned short* sM = reinterpret_cast<unsigned short*>(sdec + MAXC);  // the solve matrices, [SOLVE_MATS][16][STP]
    // T_ii attn0_ij (hi: matrix p, lo: matrix 6 + p, p = i (i - 1) / 2 + j) and T_ii diag(s_i) ((i, set, part): 12 + 4 i + 2 set + part)
    auto mat = [&](unsigned int idx) { return sM + idx * (16 * STP); };

    const unsigned int tid = threadIdx.x, lane = tid & 31u, warp = tid >> 5;
    const unsigned int g = lane >> 2, t4 = lane & 3u, q = lane >> 3, l8 = lane & 7u;
    const unsigned int src_head = head / group;
    const unsigned long long qk_stride = (unsigned long long)(H / group) * D, v_stride = (unsigned long long)H * D;
    const float* beta_src = a.beta + tok0 * H + head;
    const float* g_src = a.raw_g + tok0 * H + head;
    const unsigned long long key_at = tok0 * qk_stride + (unsigned long long)src_head * D;
    const unsigned long long query_at = key_at;
    const unsigned long long value_at = tok0 * v_stride + (unsigned long long)head * D;

    // Normalised rows are plain copies: start them first, they land while the gates and the cumulative decay are formed.
    if (a.qk_normalized) {
        brain_copy_rows_bf16(sKb, a.key, key_at, qk_stride, C);
        brain_copy_rows_bf16(sXb, a.query, query_at, qk_stride, C);
    }
    // A short chunk leaves attn0 entries unwritten: zero them (a full chunk's epilogue writes every entry the solve reads).
    if (C < MAXC) {
        for (unsigned int i = tid; i < MAXC * SC; i += THREADS) { sA[i] = 0.0f; }
    }
    if (tid < MAXC) {
        float b = 0.0f, gate = 0.0f;
        if (tid < C) {
            b = beta_src[tid * H];
            gate = g_src[tid * H];
            if (a.a_log != nullptr) {
                const float x = gate + a.dt_bias[head];
                b = 1.0f / (1.0f + expf(-b));
                gate = -expf(a.a_log[head]) * (fmaxf(x, 0.0f) + logf(1.0f + expf(-fabsf(x))));
            }
        }
        sbeta[tid] = b;
        sg[tid] = gate;
    }
    __syncthreads();
    // The cumulative decay: a warp scan, two rows a lane (the additions are reassociated against `gdn_chunk_cumsum_step`'s serial sum,
    // a few fp32 ulps; the rows past the chunk hold zero so the scan runs over all 64).
    if (tid < 32u) {
        const float2 pair = *reinterpret_cast<const float2*>(sg + 2 * tid);
        float local = pair.x + pair.y, run = local;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const float up = __shfl_up_sync(0xffffffffu, run, o);
            if (tid >= static_cast<unsigned int>(o)) { run += up; }
        }
        const float before = run - local;
        *reinterpret_cast<float2*>(sg + 2 * tid) = make_float2(before + pair.x, run);
    }
    __syncthreads();
    if (tid < MAXC) {
        const float e = expf(sg[tid]);
        sexp[tid] = e;
        sdec[tid] = expf(sg[C - 1] - sg[tid]);
        if (a.emit_cm) {
            if (tid < C) {
                g_cs_out[row_base + tid] = sg[tid];
                exp_g_cs_out[row_base + tid] = e;
            }
        } else {
            // Folded recurrence: per chunk-head 192 floats, `scale exp(g_t)` and `exp(g_last - g_t)` of its 64 tokens (zero past C) and
            // `exp(g_last)` (the state's decay), staged by the recurrence with plain copies.
            float* const aux = g_cs_out + row_base * 3u;
            aux[tid] = tid < C ? scale * e : 0.0f;
            aux[MAXC + tid] = tid < C ? sdec[tid] : 0.0f;
            if (tid == 0) { *reinterpret_cast<float4*>(aux + 2 * MAXC) = make_float4(expf(sg[C - 1]), 0.0f, 0.0f, 0.0f); }
        }
    }
    __syncthreads();

    if (a.qk_normalized) {
        asm volatile("cp.async.wait_group 0;\n" ::: "memory");
        __syncthreads();
        if (a.emit_cm) {
            brain_scale_tile_rows(sKb, reinterpret_cast<unsigned short*>(key_cm_out) + row_base * D, C, sdec, 1.0f);
            brain_scale_tile_rows(sXb, reinterpret_cast<unsigned short*>(query_cm_out) + row_base * D, C, sexp, scale);
        }
    } else {
        brain_normalize_rows_bf16(sKb, a.key, key_at, a.qkv_bf16, qk_stride, reinterpret_cast<unsigned short*>(key_cm_out) + row_base * D, C, eps, sdec, 1.0f);
        brain_normalize_rows_bf16(sXb, a.query, query_at, a.qkv_bf16, qk_stride, reinterpret_cast<unsigned short*>(query_cm_out) + row_base * D, C, eps, sexp, scale);
        __syncthreads();
    }

    const unsigned int mt = warp & 3u, nh = warp >> 2;
    const unsigned int m0 = mt * 16u;
    // ldmatrix lane rows: A tile stored [m][k]; B tile stored [n][k] (the key rows as K^T).
    const unsigned int a_row = (q & 1u) * 8u + l8, a_col = (q >> 1) * 8u;
    const unsigned int bt_row = (q >> 1) * 8u + l8, bt_col = (q & 1u) * 8u;

    // ---- attn0 = strict_lower(-beta (k . k^T) decay) and intra = scale (q . k^T) decay, sharing the K^T fragments ----
    if (m0 < C && nh * 32u < C) {
        float acc_g[4][4], acc_i[4][4];
#pragma unroll
        for (int t = 0; t < 4; ++t) {
#pragma unroll
            for (int i = 0; i < 4; ++i) { acc_g[t][i] = 0.0f; acc_i[t][i] = 0.0f; }
        }
#pragma unroll
        for (unsigned int k0 = 0; k0 < D; k0 += 16) {
            unsigned int ak[4], aq[4];
            brain_ldmatrix_x4(ak, sKb + (m0 + a_row) * SKH + k0 + a_col);
            brain_ldmatrix_x4(aq, sXb + (m0 + a_row) * SKH + k0 + a_col);
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                unsigned int b[4];
                brain_ldmatrix_x4(b, sKb + (nh * 32u + h * 16u + bt_row) * SKH + k0 + bt_col);
                brain_mma_bf16(acc_g[2 * h], ak, b[0], b[1]);
                brain_mma_bf16(acc_g[2 * h + 1], ak, b[2], b[3]);
                brain_mma_bf16(acc_i[2 * h], aq, b[0], b[1]);
                brain_mma_bf16(acc_i[2 * h + 1], aq, b[2], b[3]);
            }
        }
        // exp(g_r - g_c) = 2^(l_r - l_c) with l = g log2(e), the row and column terms held in registers.
        const float kLog2e = 1.4426950408889634f;
        const unsigned int r0 = m0 + g, r1 = m0 + g + 8u;
        const float lr0 = sg[r0] * kLog2e, lr1 = sg[r1] * kLog2e, br0 = sbeta[r0], br1 = sbeta[r1];
        float lc[8];
#pragma unroll
        for (int t = 0; t < 4; ++t) {
            lc[2 * t] = sg[nh * 32u + t * 8 + 2 * t4] * kLog2e;
            lc[2 * t + 1] = sg[nh * 32u + t * 8 + 2 * t4 + 1] * kLog2e;
        }
        if (C == MAXC) {
            // A full chunk: no bounds, aligned 8-byte and 4-byte stores.
#pragma unroll
            for (int t = 0; t < 4; ++t) {
#pragma unroll
                for (int h2 = 0; h2 < 2; ++h2) {
                    const unsigned int r = h2 ? r1 : r0;
                    const float lr = h2 ? lr1 : lr0, nbr = -(h2 ? br1 : br0);
                    const unsigned int col0 = nh * 32u + t * 8 + 2 * t4;
                    const float d0 = col0 <= r ? brain_ex2(lr - lc[2 * t]) : 0.0f;
                    const float d1 = col0 + 1 <= r ? brain_ex2(lr - lc[2 * t + 1]) : 0.0f;
                    const float g0 = acc_g[t][2 * h2], g1 = acc_g[t][2 * h2 + 1], q0 = acc_i[t][2 * h2], q1 = acc_i[t][2 * h2 + 1];
                    *reinterpret_cast<float2*>(sA + r * SC + col0) = make_float2(col0 < r ? nbr * d0 * g0 : 0.0f, col0 + 1 < r ? nbr * d1 * g1 : 0.0f);
                    *reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned short*>(intra_out) + cc_base + r * MAXC + col0) = brain_pack_bf16(scale * d0 * q0, scale * d1 * q1);
                }
            }
        } else {
#pragma unroll
            for (int t = 0; t < 4; ++t) {
#pragma unroll
                for (int h2 = 0; h2 < 2; ++h2) {
                    const unsigned int r = h2 ? r1 : r0;
                    const float lr = h2 ? lr1 : lr0, br = h2 ? br1 : br0;
                    const unsigned int col0 = nh * 32u + t * 8 + 2 * t4;
                    float at[2], it[2];
#pragma unroll
                    for (int e = 0; e < 2; ++e) {
                        const unsigned int col = col0 + e;
                        const float dm = col <= r ? brain_ex2(lr - lc[2 * t + e]) : 0.0f;
                        at[e] = col < r ? -br * acc_g[t][2 * h2 + e] * dm : 0.0f;
                        it[e] = scale * acc_i[t][2 * h2 + e] * dm;
                    }
                    if (r < C && col0 < C) {
                        *reinterpret_cast<float2*>(sA + r * SC + col0) = make_float2(at[0], at[1]);
                        unsigned short* const ip = reinterpret_cast<unsigned short*>(intra_out) + cc_base + (unsigned long long)r * C + col0;
                        ip[0] = static_cast<unsigned short>(brain_pack_bf16(it[0], 0.0f));
                        if (col0 + 1 < C) { ip[1] = static_cast<unsigned short>(brain_pack_bf16(it[1], 0.0f)); }
                    }
                }
            }
        }
    }
    // The query tile is dead after this barrier: the value rows take its space.
    __syncthreads();
    if (a.qk_normalized) {
        brain_copy_rows_bf16(sXb, a.value, value_at, v_stride, C);  // lands during the solve below
    } else {
        for (unsigned int e = tid; e < MAXC * (D / 4); e += THREADS) {
            const unsigned int r = e / (D / 4), c4 = e % (D / 4);
            uint2 t = make_uint2(0u, 0u);
            if (r < C) {
                const float4 v = brain_prep_load4(a.value, value_at + r * v_stride + c4 * 4, a.qkv_bf16);
                t = make_uint2(brain_pack_bf16(v.x, v.y), brain_pack_bf16(v.z, v.w));
            }
            *reinterpret_cast<uint2*>(sXb + r * SKH + c4 * 4) = t;
        }
    }

    // ---- the diagonal blocks' inverses and their products with the blocks below the diagonal ----
    // Thread t < 160 owns one column of Z_i = (I - attn0_ii)^-1 [attn0_i0 | .. | attn0_i,i-1 | I], block row i: forward substitution in fp32.
    if (tid < 160u) {
        const unsigned int i = tid < 16u ? 0u : tid < 48u ? 1u : tid < 96u ? 2u : 3u;
        const unsigned int first = i == 0u ? 0u : i == 1u ? 16u : i == 2u ? 48u : 96u;
        const unsigned int grp = (tid - first) >> 4, c = tid & 15u;
        if (16u * i < C) {
            float x[16];
            const float* const aii = sA + 16u * i * SC + 16u * i;
#pragma unroll
            for (int r = 0; r < 16; ++r) {
                float v = grp < i ? sA[(16u * i + r) * SC + 16u * grp + c] : (static_cast<unsigned int>(r) == c ? 1.0f : 0.0f);
                // Row r of the diagonal block, four entries a load (the entries from r on are zero or unused).
#pragma unroll
                for (int s4 = 0; s4 < r; s4 += 4) {
                    const float4 a4 = *reinterpret_cast<const float4*>(aii + r * SC + s4);
                    v = fmaf(a4.x, x[s4], v);
                    if (s4 + 1 < r) { v = fmaf(a4.y, x[s4 + 1], v); }
                    if (s4 + 2 < r) { v = fmaf(a4.z, x[s4 + 2], v); }
                    if (s4 + 3 < r) { v = fmaf(a4.w, x[s4 + 3], v); }
                }
                x[r] = v;
            }
            if (grp < i) {
                unsigned short* const hi = mat(i * (i - 1) / 2 + grp);
                unsigned short* const lo = mat(6 + i * (i - 1) / 2 + grp);
#pragma unroll
                for (int r = 0; r < 16; ++r) { brain_split_bf16(x[r], hi[r * STP + c], lo[r * STP + c]); }
            } else {
                // Column c of T_ii scaled by s_c: the B operand (k = c, n = row) of R^T_i T_ii^T diag(s).
                const float bs = sbeta[16u * i + c], ws = bs * sexp[16u * i + c];
                unsigned short* const uh = mat(12 + 4 * i + 0);
                unsigned short* const ul = mat(12 + 4 * i + 1);
                unsigned short* const wh = mat(12 + 4 * i + 2);
                unsigned short* const wl = mat(12 + 4 * i + 3);
#pragma unroll
                for (int r = 0; r < 16; ++r) {
                    brain_split_bf16(x[r] * bs, uh[r * STP + c], ul[r * STP + c]);
                    brain_split_bf16(x[r] * ws, wh[r * STP + c], wl[r * STP + c]);
                }
            }
        }
    }
    asm volatile("cp.async.wait_group 0;\n" ::: "memory");
    __syncthreads();

    // ---- X^T = [u^T | w^T] by block rows: this warp owns the 16 value columns 16 warp .. of both ----
    // Accumulators: tile pair (2 i, 2 i + 1) holds block row i, (M = value column, N = chunk row).
    const unsigned int cw = warp * 16u;
    float xu[8][4], xw[8][4];
    unsigned int hu[3][4], lu[3][4], hw[3][4], lw[3][4];  // finished block rows, as split A operands
#pragma unroll
    for (int i = 0; i < 4; ++i) {
#pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
#pragma unroll
            for (int e = 0; e < 4; ++e) { xu[2 * i + nt][e] = 0.0f; xw[2 * i + nt][e] = 0.0f; }
        }
        if (16u * i >= C) { continue; }  // block-uniform; the rows of this block row past the chunk are zero
        // R^T_i (M = value column, K = chunk row) straight from the value / key tiles by a transposing load.
        unsigned int ru[4], rw[4];
        const unsigned int src_off = (16u * i + (q >> 1) * 8u + l8) * SKH + cw + (q & 1u) * 8u;
        brain_ldmatrix_x4_trans(ru, sXb + src_off);
        brain_ldmatrix_x4_trans(rw, sKb + src_off);
#pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
            const unsigned int off = (8u * nt + g) * STP + 2u * t4;
#pragma unroll
            for (int part = 0; part < 2; ++part) {
                const unsigned short* const bu = mat(12 + 4 * i + part);
                const unsigned short* const bw = mat(12 + 4 * i + 2 + part);
                brain_mma_bf16(xu[2 * i + nt], ru, *reinterpret_cast<const unsigned int*>(bu + off), *reinterpret_cast<const unsigned int*>(bu + off + 8));
                brain_mma_bf16(xw[2 * i + nt], rw, *reinterpret_cast<const unsigned int*>(bw + off), *reinterpret_cast<const unsigned int*>(bw + off + 8));
            }
        }
#pragma unroll
        for (int j = 0; j < i; ++j) {
            const unsigned int p = i * (i - 1) / 2 + j;
#pragma unroll
            for (int nt = 0; nt < 2; ++nt) {
                const unsigned int off = (8u * nt + g) * STP + 2u * t4;
                const unsigned short* const ph = mat(p);
                const unsigned short* const pl = mat(6 + p);
                const unsigned int ph0 = *reinterpret_cast<const unsigned int*>(ph + off), ph1 = *reinterpret_cast<const unsigned int*>(ph + off + 8);
                const unsigned int pl0 = *reinterpret_cast<const unsigned int*>(pl + off), pl1 = *reinterpret_cast<const unsigned int*>(pl + off + 8);
                brain_mma_bf16(xu[2 * i + nt], hu[j], ph0, ph1);
                brain_mma_bf16(xu[2 * i + nt], hu[j], pl0, pl1);
                brain_mma_bf16(xu[2 * i + nt], lu[j], ph0, ph1);
                brain_mma_bf16(xw[2 * i + nt], hw[j], ph0, ph1);
                brain_mma_bf16(xw[2 * i + nt], hw[j], pl0, pl1);
                brain_mma_bf16(xw[2 * i + nt], lw[j], ph0, ph1);
            }
        }
        if (i < 3) {
            brain_acc_to_a_split(xu[2 * i], xu[2 * i + 1], hu[i], lu[i]);
            brain_acc_to_a_split(xw[2 * i], xw[2 * i + 1], hw[i], lw[i]);
        }
        // Block row i leaves: u (fp32) and w (bf16) rows, element (value column cw + g (+ 8), chunk row 16 i + 8 nt + 2 t4 (+ 1)).
#pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const unsigned int row = 16u * i + 8u * nt + 2u * t4 + (e & 1u);
                if (row < C) {
                    const unsigned long long at = (row_base + row) * D + cw + g + ((e & 2u) ? 8u : 0u);
                    if (a.emit_cm) { u_out[at] = xu[2 * i + nt][e]; }
                    reinterpret_cast<unsigned short*>(w_out)[at] = static_cast<unsigned short>(brain_pack_bf16(xw[2 * i + nt][e], 0.0f));
                }
            }
        }
    }
    if (!a.emit_cm) {
        // The folded recurrence reads u as fp16 in the order its threads hold it: this thread's 32 values (tile j, element e: value column
        // 16 warp + g + 8 (e >> 1), chunk row 8 j + 2 t4 + (e & 1)) in four 16-byte vectors, a warp's vectors lane after lane.
        unsigned short* const uf = reinterpret_cast<unsigned short*>(u_out) + (row_base / MAXC) * (MAXC * D) + warp * 1024u + lane * 8u;
#pragma unroll
        for (int v = 0; v < 4; ++v) {
            *reinterpret_cast<uint4*>(uf + v * 256) = make_uint4(brain_pack_f16(xu[2 * v][0], xu[2 * v][1]), brain_pack_f16(xu[2 * v][2], xu[2 * v][3]),
                                                                  brain_pack_f16(xu[2 * v + 1][0], xu[2 * v + 1][1]), brain_pack_f16(xu[2 * v + 1][2], xu[2 * v + 1][3]));
        }
    }
}

extern "C" __global__ void __launch_bounds__(THREADS) brain_gdn_chunk_prep_tok_bf16(const unsigned int* params, const float* query, const float* key,
                                                                                     const float* value, const float* raw_g, const float* beta, float* g_cs_out,
                                                                                     float* exp_g_cs_out, float* u_out, float* w_out, float* intra_out,
                                                                                     float* query_cm_out, float* key_cm_out) {
    const unsigned int bhc = params[0];
    const unsigned int C = params[1];
    const unsigned int H = params[6], n_chunks = params[7];
    const unsigned int blk = blockIdx.y * gridDim.x + blockIdx.x;
    if (blk >= bhc) { return; }  // block-uniform
    const PrepOperands a = {query, key, value, raw_g, beta, g_cs_out, exp_g_cs_out, u_out, w_out, intra_out, query_cm_out, key_cm_out,
                            nullptr, nullptr, __uint_as_float(params[5]), __uint_as_float(params[9]), H, params[8], false, false, true};
    const unsigned long long row_base = (unsigned long long)blk * C;
    const unsigned int bh = bhc / n_chunks, ci = blk / bh, rem = blk % bh;
    const unsigned long long tok0 = ((unsigned long long)(rem / H) * n_chunks + ci) * C;
    brain_gdn_prep_chunk(a, C, row_base, row_base * C, tok0, rem % H);
}

// The same preparation for every chunk of a round's prompt segments in one launch: block (chunk, head) reads its rows
// and length from the chunk table (`gdn_segment_table`: [row0, rows, slot, first chunk] per segment, then [token row, C]
// per chunk) and writes its tensors at a fixed 64-row slot of the scratch, so a segment's last, short chunk needs no
// launch of its own. The gates come in raw (b_proj, a_proj with the layer's a_log and dt_bias) and are formed in the load.
// Contract: params [chunks * H, unused..., scale (f32 bits) at 5, H at 6, nseg at 7, group at 8, eps (f32 bits) at 9, flags at 10
// (absent is 0; bit 0: query, key and value are bf16 rows; bit 1: query and key are also L2-normalised, as the vector convolution leaves them, required by bit 2; bit 2: the folded recurrence's form: no query_cm / key_cm, g_cs holds three vectors of 64 floats a chunk-head instead of one, and u is fp16 in thread order, 16 KiB a chunk-head)]; read
// query, key, value, b_proj, a_proj, a_log, dt_bias, table; write the seven tensors of `brain_gdn_chunk_prep_tok_bf16`.
extern "C" __global__ void __launch_bounds__(THREADS, 3) brain_gdn_chunk_prep_seg_bf16(const unsigned int* params, const float* query, const float* key,
                                                                                     const float* value, const float* b_proj, const float* a_proj,
                                                                                     const float* a_log, const float* dt_bias, const unsigned int* table,
                                                                                     float* g_cs_out, float* exp_g_cs_out, float* u_out, float* w_out,
                                                                                     float* intra_out, float* query_cm_out, float* key_cm_out) {
    const unsigned int blocks = params[0];
    const unsigned int H = params[6], nseg = params[7];
    const unsigned int blk = blockIdx.y * gridDim.x + blockIdx.x;
    if (blk >= blocks) { return; }  // block-uniform
    const PrepOperands a = {query, key, value, a_proj, b_proj, g_cs_out, exp_g_cs_out, u_out, w_out, intra_out, query_cm_out, key_cm_out,
                            a_log, dt_bias, __uint_as_float(params[5]), __uint_as_float(params[9]), H, params[8], (params[10] & 1u) != 0u, (params[10] & 2u) != 0u, (params[10] & 4u) == 0u};
    const unsigned int chunk = blk / H, head = blk % H;
    const unsigned int* entry = table + 4 * nseg + 2 * chunk;
    const unsigned long long row_base = (unsigned long long)blk * MAXC;
    brain_gdn_prep_chunk(a, entry[1], row_base, row_base * MAXC, entry[0], head);
}
