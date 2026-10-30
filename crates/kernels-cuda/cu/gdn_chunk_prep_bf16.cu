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
//   attn0     = strict_lower(-beta_i (k_i . k_j) exp(g_cs_i - g_cs_j))      T = (I - attn0)^-1   (fp32 solve)
//   u         = (T diag(beta)) @ value                                      (fp32 out)
//   w         = (T diag(beta exp_g_cs)) @ key                               (bf16 out)
//   intra     = scale (q_i . k_j) exp(g_cs_i - g_cs_j), j <= i              (bf16 out)
//   query_cm  = scale * q * exp_g_cs                (bf16, the recurrence's inter-chunk operand)
//   key_cm    = k * exp(g_last - g_cs)              (bf16, the recurrence's state-update operand)
//
// The row scalings that the TF32 kernel applies to an operand (beta, exp_g_cs) are folded into the columns of the
// small T operand or applied to the fp32 accumulators, so the big tiles (key, query, value) are converted once. The
// unit-triangular solve stays fp32: it is the one step whose conditioning depends on near-parallel keys. u keeps fp32
// (it is subtracted from a product in the recurrence); w, intra, query_cm and key_cm are bf16 buffers of their fp32
// tensors' size.
//
// Contract: params [bhc, c_len, dk, dv, unused, scale (f32 bits), H, n_chunks, group, eps (f32 bits)]; buffers as
// `gdn_chunk_prep_tok_tc` (query, key, value, raw_g, beta read; g_cs, exp_g_cs, u, w, intra_scores, query_cm, key_cm
// written). dk = dv = 128 and 1 <= c_len <= 64. Dynamic shared memory: PREP_BF16_SMEM_BYTES.
// No `__restrict__`: brain's device buffers alias by design.

#define D 128
#define MAXC 64
#define THREADS 256
#define SKH (D + 8)       // bf16 row stride of the key / query / value tiles (ldmatrix rows 16 B apart mod 128 B)
#define STH (MAXC + 8)    // bf16 row stride of the scaled T operands
#define SC (MAXC + 4)     // fp32 row stride of attn0 and T

#define SA_BYTES (2 * MAXC * STH * 2)  // attn0 (64 x 68 fp32 = 17408 B) is dead once T is solved; two bf16 64 x 72 operands take its place

#define PREP_BF16_SMEM_BYTES (2 * MAXC * SKH * 2 + SA_BYTES + MAXC * SC * 4 + 4 * MAXC * 4)

// One chunk row per warp: the row of `src` (fp32, stride `stride`) is L2-normalised (`x * rsqrt(sum x^2 + eps)`, the
// formula of `gdn_chunk_prep_tc`), written to the bf16 tile `tile` and, scaled by `row_scale[r] * extra`, to the
// chunk-major bf16 buffer `cm`. Rows past C of the tile are zero.
__device__ __forceinline__ void brain_normalize_rows_bf16(unsigned short* tile, const float* src, unsigned long long stride, unsigned short* cm, unsigned int C, float eps,
                                                          const float* row_scale, float extra) {
    const unsigned int lane = threadIdx.x & 31u, warp = threadIdx.x >> 5;
    for (unsigned int r = warp; r < MAXC; r += THREADS / 32) {
        uint2 t = make_uint2(0u, 0u);
        if (r < C) {
            float4 v = *reinterpret_cast<const float4*>(src + r * stride + lane * 4);
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

// The tensors one chunk's prep reads and writes, and the head geometry that addresses them.
struct PrepOperands {
    const float *query, *key, *value, *raw_g, *beta;
    float *g_cs_out, *exp_g_cs_out, *u_out, *w_out, *intra_out, *query_cm_out, *key_cm_out;
    // Non-null: `raw_g` and `beta` hold the raw projections a_proj and b_proj, and the gates are formed in the loads
    // (beta = sigmoid(b_proj), raw_g = -exp(a_log) * softplus(a_proj + dt_bias), per value head).
    const float *a_log, *dt_bias;
    float scale, eps;
    unsigned int H, group;
};

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
    float* sA = reinterpret_cast<float*>(sXb + MAXC * SKH);              // attn0, fp32 [64][SC]; after the solve two bf16 [64][STH] operands
    unsigned short* sTu = reinterpret_cast<unsigned short*>(sA);         // T diag(beta)
    unsigned short* sTw = sTu + MAXC * STH;                               // T diag(beta exp_g_cs)
    float* sT = reinterpret_cast<float*>(smem_raw + 2 * MAXC * SKH * 2 + SA_BYTES);  // T, fp32 [64][SC]
    float* sg = sT + MAXC * SC;       // cumulative decay
    float* sexp = sg + MAXC;          // its exponential
    float* sbeta = sexp + MAXC;
    float* sdec = sbeta + MAXC;       // exp(g_last - g_cs)
    float* sRed = reinterpret_cast<float*>(sXb);  // partial sums of the solve, [3][8][64]: the query tile is dead and the values load after

    const unsigned int tid = threadIdx.x, lane = tid & 31u, warp = tid >> 5;
    const unsigned int g = lane >> 2, t4 = lane & 3u, q = lane >> 3, l8 = lane & 7u;
    const unsigned int src_head = head / group;
    const unsigned long long qk_stride = (unsigned long long)(H / group) * D, v_stride = (unsigned long long)H * D;
    const float* beta_src = a.beta + tok0 * H + head;
    const float* g_src = a.raw_g + tok0 * H + head;
    const float* key_src = a.key + tok0 * qk_stride + (unsigned long long)src_head * D;
    const float* query_src = a.query + tok0 * qk_stride + (unsigned long long)src_head * D;
    const float* value_src = a.value + tok0 * v_stride + (unsigned long long)head * D;

    for (unsigned int i = tid; i < MAXC * SC; i += THREADS) {
        sA[i] = 0.0f;
        sT[i] = 0.0f;
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
    // The serial sum `gdn_chunk_cumsum_step` performs, so g_cs is bit-identical.
    if (tid == 0) {
        for (unsigned int i = 1; i < C; ++i) { sg[i] = sg[i] + sg[i - 1]; }
    }
    __syncthreads();
    if (tid < MAXC) {
        const float e = expf(sg[tid]);
        sexp[tid] = e;
        sdec[tid] = expf(sg[C - 1] - sg[tid]);
        if (tid < C) {
            g_cs_out[row_base + tid] = sg[tid];
            exp_g_cs_out[row_base + tid] = e;
        }
    }
    __syncthreads();

    brain_normalize_rows_bf16(sKb, key_src, qk_stride, reinterpret_cast<unsigned short*>(key_cm_out) + row_base * D, C, eps, sdec, 1.0f);
    brain_normalize_rows_bf16(sXb, query_src, qk_stride, reinterpret_cast<unsigned short*>(query_cm_out) + row_base * D, C, eps, sexp, scale);
    __syncthreads();

    const unsigned int mt = warp & 3u, nh = warp >> 2;
    const unsigned int m0 = mt * 16u;
    const unsigned int c16 = (C + 15u) & ~15u;
    // ldmatrix lane rows: A tile stored [m][k]; B tile stored [n][k] (the key rows as K^T); B tile stored [k][n] (transposed load).
    const unsigned int a_row = (q & 1u) * 8u + l8, a_col = (q >> 1) * 8u;
    const unsigned int bt_row = (q >> 1) * 8u + l8, bt_col = (q & 1u) * 8u;
    const unsigned int b_row = (q & 1u) * 8u + l8, b_col = (q >> 1) * 8u;

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
#pragma unroll
        for (int t = 0; t < 4; ++t) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const unsigned int r = m0 + g + ((i & 2) ? 8u : 0u);
                const unsigned int col = nh * 32u + t * 8 + 2 * t4 + (i & 1);
                if (r < C && col < C) {
                    const float dm = col <= r ? expf(sg[r] - sg[col]) : 0.0f;
                    sA[r * SC + col] = col < r ? -sbeta[r] * acc_g[t][i] * dm : 0.0f;
                    reinterpret_cast<unsigned short*>(intra_out)[cc_base + (unsigned long long)r * C + col] =
                        static_cast<unsigned short>(brain_pack_bf16(scale * acc_i[t][i] * dm, 0.0f));
                }
            }
        }
    }
    // The query tile is dead after this barrier: the solve borrows its space for partial sums, the value rows take it after.
    __syncthreads();

    // ---- T = (I - attn0)^-1 (fp32) ----
    brain_unit_lower_inverse(sA, sT, sRed, SC, C);

    // The solve is done with attn0: its space takes T scaled for the two products, as bf16.
    for (unsigned int e = tid; e < MAXC * MAXC; e += THREADS) {
        const unsigned int r = e / MAXC, k = e % MAXC;
        const float t = sT[r * SC + k] * sbeta[k];
        sTu[r * STH + k] = static_cast<unsigned short>(brain_pack_bf16(t, 0.0f));
        sTw[r * STH + k] = static_cast<unsigned short>(brain_pack_bf16(t * sexp[k], 0.0f));
    }
    for (unsigned int e = tid; e < MAXC * (D / 4); e += THREADS) {
        const unsigned int r = e / (D / 4), c4 = e % (D / 4);
        uint2 t = make_uint2(0u, 0u);
        if (r < C) {
            const float4 v = *reinterpret_cast<const float4*>(value_src + r * v_stride + c4 * 4);
            t = make_uint2(brain_pack_bf16(v.x, v.y), brain_pack_bf16(v.z, v.w));
        }
        *reinterpret_cast<uint2*>(sXb + r * SKH + c4 * 4) = t;
    }

    __syncthreads();

    // ---- u = (T diag(beta)) @ value and w = (T diag(beta exp_g_cs)) @ key: 64 x 128 outputs, K = chunk rows ----
    if (m0 < C) {
        const unsigned int n0 = nh * 64u;
#pragma unroll
        for (int pass = 0; pass < 2; ++pass) {
            const unsigned short* sTa = pass == 0 ? sTu : sTw;
            const unsigned short* sB = pass == 0 ? sXb : sKb;
            float acc[8][4];
#pragma unroll
            for (int t = 0; t < 8; ++t) {
#pragma unroll
                for (int i = 0; i < 4; ++i) { acc[t][i] = 0.0f; }
            }
            for (unsigned int k0 = 0; k0 < c16; k0 += 16) {
                unsigned int a[4];
                brain_ldmatrix_x4(a, sTa + (m0 + a_row) * STH + k0 + a_col);
#pragma unroll
                for (int h = 0; h < 4; ++h) {
                    unsigned int b[4];
                    brain_ldmatrix_x4_trans(b, sB + (k0 + b_row) * SKH + n0 + h * 16 + b_col);
                    brain_mma_bf16(acc[2 * h], a, b[0], b[1]);
                    brain_mma_bf16(acc[2 * h + 1], a, b[2], b[3]);
                }
            }
#pragma unroll
            for (int t = 0; t < 8; ++t) {
#pragma unroll
                for (int h2 = 0; h2 < 2; ++h2) {
                    const unsigned int r = m0 + g + h2 * 8u;
                    if (r < C) {
                        const unsigned long long at = (row_base + r) * D + n0 + t * 8 + 2 * t4;
                        if (pass == 0) {
                            *reinterpret_cast<float2*>(u_out + at) = make_float2(acc[t][2 * h2], acc[t][2 * h2 + 1]);
                        } else {
                            *reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned short*>(w_out) + at) = brain_pack_bf16(acc[t][2 * h2], acc[t][2 * h2 + 1]);
                        }
                    }
                }
            }
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
                            nullptr, nullptr, __uint_as_float(params[5]), __uint_as_float(params[9]), H, params[8]};
    const unsigned long long row_base = (unsigned long long)blk * C;
    const unsigned int bh = bhc / n_chunks, ci = blk / bh, rem = blk % bh;
    const unsigned long long tok0 = ((unsigned long long)(rem / H) * n_chunks + ci) * C;
    brain_gdn_prep_chunk(a, C, row_base, row_base * C, tok0, rem % H);
}

// The same preparation for every chunk of a round's prompt segments in one launch: block (chunk, head) reads its rows
// and length from the chunk table (`gdn_segment_table`: [row0, rows, slot, first chunk] per segment, then [token row, C]
// per chunk) and writes its tensors at a fixed 64-row slot of the scratch, so a segment's last, short chunk needs no
// launch of its own. The gates come in raw (b_proj, a_proj with the layer's a_log and dt_bias) and are formed in the load.
// Contract: params [chunks * H, unused..., scale (f32 bits) at 5, H at 6, nseg at 7, group at 8, eps (f32 bits) at 9]; read
// query, key, value, b_proj, a_proj, a_log, dt_bias, table; write the seven tensors of `brain_gdn_chunk_prep_tok_bf16`.
extern "C" __global__ void __launch_bounds__(THREADS) brain_gdn_chunk_prep_seg_bf16(const unsigned int* params, const float* query, const float* key,
                                                                                     const float* value, const float* b_proj, const float* a_proj,
                                                                                     const float* a_log, const float* dt_bias, const unsigned int* table,
                                                                                     float* g_cs_out, float* exp_g_cs_out, float* u_out, float* w_out,
                                                                                     float* intra_out, float* query_cm_out, float* key_cm_out) {
    const unsigned int blocks = params[0];
    const unsigned int H = params[6], nseg = params[7];
    const unsigned int blk = blockIdx.y * gridDim.x + blockIdx.x;
    if (blk >= blocks) { return; }  // block-uniform
    const PrepOperands a = {query, key, value, a_proj, b_proj, g_cs_out, exp_g_cs_out, u_out, w_out, intra_out, query_cm_out, key_cm_out,
                            a_log, dt_bias, __uint_as_float(params[5]), __uint_as_float(params[9]), H, params[8]};
    const unsigned int chunk = blk / H, head = blk % H;
    const unsigned int* entry = table + 4 * nseg + 2 * chunk;
    const unsigned long long row_base = (unsigned long long)blk * MAXC;
    brain_gdn_prep_chunk(a, entry[1], row_base, row_base * MAXC, entry[0], head);
}
