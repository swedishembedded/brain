// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements document-vision serving for clients who read pages at scale. If your team
// needs expertise in putting ViT attention, with or without a relative-position bias, onto the tensor cores
// of the card it runs on, then you can procure our services by sending an email to info@swedishembedded.com.
//
// Bidirectional flash attention at head_dim 64 on fp16 tensor cores, two entry points over one core:
//
// brain_flash_cross_relpos_f16_hd64 - the native form of `flash_attn_cross_relpos.wgsl` (its uniform, its six
// bindings, its output layout; `gpu_core::native_upgrade` redirects it). SAM/ViTDet attention with a
// decomposed relative-position bias, spans (windows) as the batch axis:
//
//   params : u32 [bsz, n_heads, t_dec, t_enc, head_dim, q_stride, q_off, k_stride, k_off, v_stride, v_off,
//                 d_model, kh_ext, kw_ext]
//   q,k,v  : row (b * t + i) holds head h's channels at off + h * head_dim, each row_stride wide
//   rel_h  : [bsz, heads, t_dec, kh_ext]    rel_w : [bsz, heads, t_dec, kw_ext]
//   out    : [bsz * t_dec, d_model]; head h of row (b * t_dec + i) at h * head_dim
//   out[i] = sum_j softmax_j(q_i . k_j / sqrt(head_dim) + rel_h[i, j / kw] + rel_w[i, j % kw]) v_j
//
// brain_flash_bidir_f16_hd64 - the native form of `flash_attn_bidir_reg2.wgsl` at head_dim 64 (CLIP): the
// same without the bias, over a packed qkv slab.
//
//   params : u32 [bsz, n_heads, T, head_dim, qkv_stride, q_off, k_off, v_off, d_model]
//
// Numerics (the contract the gate `gpu-core/tests/flash_attn_f16_native.rs` states): Q, K, V and the
// probabilities P are rounded to fp16 for the tensor cores; products accumulate in fp32; the scores, the bias,
// the running maximum and the sums are fp32. An fp16 operand has the same 10-bit mantissa as TF32, so the
// error is that of the TF32 products the unfused chain used, with the fp16 range (scores are formed in
// fp32 after the product, so only q, k, v and p, all of modest magnitude, must fit it; conversion saturates).
//
// Tiling: a block is 4 warps of 128 threads; each warp owns 16 query rows of one (span, head) over the whole
// 64-wide head, so Q lives in registers as four A fragments and is read from global memory once. Keys stream
// through shared memory in tiles of 64 as fp16 (K and V, 9 KiB each); the next tile's global loads are issued
// before the current tile's matrix work and committed after it, two barriers a tile. The online softmax is in
// the exp2 domain, scores in the m16n8 accumulator layout, P converted to A fragments without leaving the
// registers. The bias is read, not recomputed: two small table loads per score from tables that are small
// against the score slab the unfused chain materialised.
//
// Query rows past t_dec are computed against a zero query and not written; keys past t_enc are zero rows
// whose scores are masked to zero weight. Global buffers alias by design, so no __restrict__.

#define FC_THREADS 128
#define FC_ROWS 64                    // query rows per block, 16 per warp
#define FC_KEYS 64                    // keys per tile
#define FC_HD 64
#define FC_PAD 8                      // smem row padding in halves (keeps ldmatrix rows on distinct banks)
#define FC_ROWH (FC_HD + FC_PAD)
#define FC_SN (FC_KEYS / 8)           // n8 score tiles per tile
#define FC_KK (FC_KEYS / 16)          // k16 steps of P V per tile
#define FC_NT (FC_HD / 8)             // n8 output tiles per warp
#define FC_CHUNKS (FC_HD / 8)         // 8-value chunks per key row
#define FC_ROWS_PER_PASS (FC_THREADS / FC_CHUNKS)
#define FC_STAGE (FC_KEYS / FC_ROWS_PER_PASS)   // chunks a thread stages per tile and plane
#define FC_LOG2E 1.4426950408889634f
#define FC_SCALE_LOG2 (FC_LOG2E * 0.125f)       // log2(e) / sqrt(64)
#define FC_NEG_BIG (-1.0e30f)

__device__ __forceinline__ unsigned int fc_smem_addr(const void* p) {
    return static_cast<unsigned int>(__cvta_generic_to_shared(p));
}

// {lo, hi} -> packed f16x2 with `lo` in the low half (the lower k index).
__device__ __forceinline__ unsigned int fc_pack_f16x2(float lo, float hi) {
    unsigned int r;
    asm("cvt.rn.satfinite.f16x2.f32 %0, %1, %2;\n" : "=r"(r) : "f"(hi), "f"(lo));
    return r;
}

__device__ __forceinline__ void fc_ldsm_x4(unsigned int addr, unsigned int& r0, unsigned int& r1, unsigned int& r2, unsigned int& r3) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}

__device__ __forceinline__ void fc_ldsm_x4_t(unsigned int addr, unsigned int& r0, unsigned int& r1, unsigned int& r2, unsigned int& r3) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}

// d += a * b, m16n8k16, f16 inputs, f32 accumulate.
__device__ __forceinline__ void fc_mma_f16(float (&d)[4], const unsigned int (&a)[4], unsigned int b0, unsigned int b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// One (span, head)'s rows: each pointer is already at the head's first channel of the span's first row.
struct FcHead {
    const float* q;
    const float* k;
    const float* v;
    const float* rel_h;   // this (span, head)'s [t_dec, kh] table, or null
    const float* rel_w;   // [t_dec, kw]
    float* out;
    unsigned int q_stride, k_stride, v_stride, out_stride;
    unsigned int t_dec, t_enc, kh, kw;
};

// Query tile `qt` of one (span, head).
template <bool RELPOS>
__device__ __forceinline__ void fc_core(const FcHead& a, unsigned int qt) {
    __shared__ __align__(16) unsigned short Ks[FC_KEYS * FC_ROWH];
    __shared__ __align__(16) unsigned short Vs[FC_KEYS * FC_ROWH];

    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid & 31u;
    const unsigned int warp = tid >> 5;
    const unsigned int g = lane >> 2;        // MMA groupID
    const unsigned int t4 = lane & 3u;       // MMA threadID_in_group

    const unsigned int row_a = qt * FC_ROWS + warp * 16u + g;      // fragment rows g and g + 8
    const unsigned int row_b = row_a + 8u;
    const bool live_a = row_a < a.t_dec, live_b = row_b < a.t_dec;

    // ---- Q as four A fragments, straight from global memory (a row's eight halves of a k16 step are two float2s).
    unsigned int qa[FC_HD / 16][4];
#pragma unroll
    for (int s = 0; s < FC_HD / 16; ++s) {
        const unsigned int c0 = s * 16u + t4 * 2u;
        float2 x0 = make_float2(0.f, 0.f), x1 = x0, x2 = x0, x3 = x0;
        if (live_a) {
            x0 = *reinterpret_cast<const float2*>(a.q + (unsigned long long)row_a * a.q_stride + c0);
            x2 = *reinterpret_cast<const float2*>(a.q + (unsigned long long)row_a * a.q_stride + c0 + 8u);
        }
        if (live_b) {
            x1 = *reinterpret_cast<const float2*>(a.q + (unsigned long long)row_b * a.q_stride + c0);
            x3 = *reinterpret_cast<const float2*>(a.q + (unsigned long long)row_b * a.q_stride + c0 + 8u);
        }
        qa[s][0] = fc_pack_f16x2(x0.x, x0.y);
        qa[s][1] = fc_pack_f16x2(x1.x, x1.y);
        qa[s][2] = fc_pack_f16x2(x2.x, x2.y);
        qa[s][3] = fc_pack_f16x2(x3.x, x3.y);
    }

    float o[FC_NT][4];
#pragma unroll
    for (int n = 0; n < FC_NT; ++n) { o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.f; }
    float m_a = FC_NEG_BIG, m_b = FC_NEG_BIG, l_a = 0.f, l_b = 0.f;

    // The bias rows of this thread's two query rows.
    const float* rh_a = nullptr; const float* rh_b = nullptr; const float* rw_a = nullptr; const float* rw_b = nullptr;
    if (RELPOS) {
        rh_a = a.rel_h + (unsigned long long)row_a * a.kh;
        rh_b = a.rel_h + (unsigned long long)row_b * a.kh;
        rw_a = a.rel_w + (unsigned long long)row_a * a.kw;
        rw_b = a.rel_w + (unsigned long long)row_b * a.kw;
    }

    // Staging: chunk (row0 + FC_ROWS_PER_PASS c, cc) of a tile is thread tid's c-th: a warp reads four key rows
    // of 64 floats and writes them as fp16 to shared memory.
    const unsigned int cc = tid % FC_CHUNKS;
    const unsigned int row0 = tid / FC_CHUNKS;
    float4 raw_k[FC_STAGE][2];
    float4 raw_v[FC_STAGE][2];
    auto issue = [&](unsigned int kt) {
#pragma unroll
        for (int c = 0; c < FC_STAGE; ++c) {
            const unsigned int j = kt * FC_KEYS + row0 + FC_ROWS_PER_PASS * c;
            float4 k0 = make_float4(0.f, 0.f, 0.f, 0.f), k1 = k0, v0 = k0, v1 = k0;
            if (j < a.t_enc) {
                const float* ks = a.k + (unsigned long long)j * a.k_stride + cc * 8u;
                const float* vs = a.v + (unsigned long long)j * a.v_stride + cc * 8u;
                k0 = *reinterpret_cast<const float4*>(ks);
                k1 = *reinterpret_cast<const float4*>(ks + 4);
                v0 = *reinterpret_cast<const float4*>(vs);
                v1 = *reinterpret_cast<const float4*>(vs + 4);
            }
            raw_k[c][0] = k0; raw_k[c][1] = k1;
            raw_v[c][0] = v0; raw_v[c][1] = v1;
        }
    };
    auto commit = [&]() {
#pragma unroll
        for (int c = 0; c < FC_STAGE; ++c) {
            const unsigned int kr = row0 + FC_ROWS_PER_PASS * c;
            uint4 kv, vv;
            kv.x = fc_pack_f16x2(raw_k[c][0].x, raw_k[c][0].y);
            kv.y = fc_pack_f16x2(raw_k[c][0].z, raw_k[c][0].w);
            kv.z = fc_pack_f16x2(raw_k[c][1].x, raw_k[c][1].y);
            kv.w = fc_pack_f16x2(raw_k[c][1].z, raw_k[c][1].w);
            vv.x = fc_pack_f16x2(raw_v[c][0].x, raw_v[c][0].y);
            vv.y = fc_pack_f16x2(raw_v[c][0].z, raw_v[c][0].w);
            vv.z = fc_pack_f16x2(raw_v[c][1].x, raw_v[c][1].y);
            vv.w = fc_pack_f16x2(raw_v[c][1].z, raw_v[c][1].w);
            *reinterpret_cast<uint4*>(Ks + kr * FC_ROWH + cc * 8u) = kv;
            *reinterpret_cast<uint4*>(Vs + kr * FC_ROWH + cc * 8u) = vv;
        }
    };

    const unsigned int ntiles_k = (a.t_enc + FC_KEYS - 1u) / FC_KEYS;
    if (ntiles_k > 0u) { issue(0u); }

    // ldmatrix lane geometry (see the MMA fragment layouts in the PTX ISA): K is the non-transposed B operand of
    // S = Q K^T, V the transposed B operand of O = P V.
    const unsigned int k_row_lane = (lane & 7u) + ((lane >> 4) << 3);
    const unsigned int k_col_lane = ((lane >> 3) & 1u) << 3;
    const unsigned int v_row_lane = (lane & 7u) + (((lane >> 3) & 1u) << 3);
    const unsigned int v_col_lane = (lane >> 4) << 3;
    const unsigned int ks_base = fc_smem_addr(Ks);
    const unsigned int vs_base = fc_smem_addr(Vs);

    for (unsigned int kt = 0; kt < ntiles_k; ++kt) {
        __syncthreads();                   // every warp is done with the previous tile's K and V
        commit();
        if (kt + 1u < ntiles_k) { issue(kt + 1u); }   // in flight under this tile's matrix work
        __syncthreads();                   // K and V visible

        // ---- S = Q K^T for 16 rows x FC_KEYS keys.
        float sacc[FC_SN][4];
#pragma unroll
        for (int n = 0; n < FC_SN; ++n) { sacc[n][0] = sacc[n][1] = sacc[n][2] = sacc[n][3] = 0.f; }
#pragma unroll
        for (int s = 0; s < FC_HD / 16; ++s) {
#pragma unroll
            for (int np = 0; np < FC_SN / 2; ++np) {
                unsigned int b0, b1, b2, b3;
                fc_ldsm_x4(ks_base + ((np * 16u + k_row_lane) * FC_ROWH + s * 16u + k_col_lane) * 2u, b0, b1, b2, b3);
                fc_mma_f16(sacc[2 * np], qa[s], b0, b1);
                fc_mma_f16(sacc[2 * np + 1], qa[s], b2, b3);
            }
        }

        // ---- scale, bias, mask (exp2 domain), then the online softmax over this tile's keys.
        float mx_a = FC_NEG_BIG, mx_b = FC_NEG_BIG;
#pragma unroll
        for (int n = 0; n < FC_SN; ++n) {
            const unsigned int j0 = kt * FC_KEYS + n * 8u + t4 * 2u;
            float ba0 = 0.f, ba1 = 0.f, bb0 = 0.f, bb1 = 0.f;
            if (RELPOS && j0 < a.t_enc) {
                const unsigned int jh0 = j0 / a.kw, jw0 = j0 - jh0 * a.kw;
                const unsigned int jh1 = (jw0 + 1u < a.kw) ? jh0 : jh0 + 1u;
                const unsigned int jw1 = (jw0 + 1u < a.kw) ? jw0 + 1u : 0u;
                if (live_a) {
                    ba0 = __ldg(rh_a + jh0) + __ldg(rw_a + jw0);
                    if (j0 + 1u < a.t_enc) { ba1 = __ldg(rh_a + jh1) + __ldg(rw_a + jw1); }
                }
                if (live_b) {
                    bb0 = __ldg(rh_b + jh0) + __ldg(rw_b + jw0);
                    if (j0 + 1u < a.t_enc) { bb1 = __ldg(rh_b + jh1) + __ldg(rw_b + jw1); }
                }
            }
            sacc[n][0] = (j0 < a.t_enc) ? fmaf(sacc[n][0], FC_SCALE_LOG2, ba0 * FC_LOG2E) : FC_NEG_BIG;
            sacc[n][1] = (j0 + 1u < a.t_enc) ? fmaf(sacc[n][1], FC_SCALE_LOG2, ba1 * FC_LOG2E) : FC_NEG_BIG;
            sacc[n][2] = (j0 < a.t_enc) ? fmaf(sacc[n][2], FC_SCALE_LOG2, bb0 * FC_LOG2E) : FC_NEG_BIG;
            sacc[n][3] = (j0 + 1u < a.t_enc) ? fmaf(sacc[n][3], FC_SCALE_LOG2, bb1 * FC_LOG2E) : FC_NEG_BIG;
            mx_a = fmaxf(mx_a, fmaxf(sacc[n][0], sacc[n][1]));
            mx_b = fmaxf(mx_b, fmaxf(sacc[n][2], sacc[n][3]));
        }
        mx_a = fmaxf(mx_a, __shfl_xor_sync(0xffffffffu, mx_a, 1));
        mx_a = fmaxf(mx_a, __shfl_xor_sync(0xffffffffu, mx_a, 2));
        mx_b = fmaxf(mx_b, __shfl_xor_sync(0xffffffffu, mx_b, 1));
        mx_b = fmaxf(mx_b, __shfl_xor_sync(0xffffffffu, mx_b, 2));
        const float mn_a = fmaxf(m_a, mx_a), mn_b = fmaxf(m_b, mx_b);
        const float al_a = exp2f(m_a - mn_a), al_b = exp2f(m_b - mn_b);
        m_a = mn_a;
        m_b = mn_b;
        float sum_a = 0.f, sum_b = 0.f;
#pragma unroll
        for (int n = 0; n < FC_SN; ++n) {
            // A masked score is FC_NEG_BIG; when the whole row is masked mn is FC_NEG_BIG too and the
            // difference is 0, so the weight is forced to 0.
            sacc[n][0] = (sacc[n][0] > FC_NEG_BIG) ? exp2f(sacc[n][0] - mn_a) : 0.f;
            sacc[n][1] = (sacc[n][1] > FC_NEG_BIG) ? exp2f(sacc[n][1] - mn_a) : 0.f;
            sacc[n][2] = (sacc[n][2] > FC_NEG_BIG) ? exp2f(sacc[n][2] - mn_b) : 0.f;
            sacc[n][3] = (sacc[n][3] > FC_NEG_BIG) ? exp2f(sacc[n][3] - mn_b) : 0.f;
            sum_a += sacc[n][0] + sacc[n][1];
            sum_b += sacc[n][2] + sacc[n][3];
        }
        l_a = l_a * al_a + sum_a;
        l_b = l_b * al_b + sum_b;
#pragma unroll
        for (int n = 0; n < FC_NT; ++n) {
            o[n][0] *= al_a;
            o[n][1] *= al_a;
            o[n][2] *= al_b;
            o[n][3] *= al_b;
        }

        // P as fp16 A fragments, straight from the score accumulators.
        unsigned int pf[FC_KK][4];
#pragma unroll
        for (int kk = 0; kk < FC_KK; ++kk) {
            pf[kk][0] = fc_pack_f16x2(sacc[2 * kk][0], sacc[2 * kk][1]);
            pf[kk][1] = fc_pack_f16x2(sacc[2 * kk][2], sacc[2 * kk][3]);
            pf[kk][2] = fc_pack_f16x2(sacc[2 * kk + 1][0], sacc[2 * kk + 1][1]);
            pf[kk][3] = fc_pack_f16x2(sacc[2 * kk + 1][2], sacc[2 * kk + 1][3]);
        }

        // ---- O += P V over all 64 output columns.
#pragma unroll
        for (int kk = 0; kk < FC_KK; ++kk) {
#pragma unroll
            for (int np = 0; np < FC_NT / 2; ++np) {
                unsigned int b0, b1, b2, b3;
                fc_ldsm_x4_t(vs_base + (((kk * 16u + v_row_lane) * FC_ROWH) + np * 16u + v_col_lane) * 2u, b0, b1, b2, b3);
                fc_mma_f16(o[2 * np], pf[kk], b0, b1);
                fc_mma_f16(o[2 * np + 1], pf[kk], b2, b3);
            }
        }
    }

    // ---- normalise and store. l is a per-thread partial of its quad's row sum.
    l_a += __shfl_xor_sync(0xffffffffu, l_a, 1);
    l_a += __shfl_xor_sync(0xffffffffu, l_a, 2);
    l_b += __shfl_xor_sync(0xffffffffu, l_b, 1);
    l_b += __shfl_xor_sync(0xffffffffu, l_b, 2);
    const float inv_a = l_a > 0.f ? 1.0f / l_a : 0.f;
    const float inv_b = l_b > 0.f ? 1.0f / l_b : 0.f;
#pragma unroll
    for (int n = 0; n < FC_NT; ++n) {
        const unsigned int d = n * 8u + t4 * 2u;
        if (live_a) {
            *reinterpret_cast<float2*>(a.out + (unsigned long long)row_a * a.out_stride + d) = make_float2(o[n][0] * inv_a, o[n][1] * inv_a);
        }
        if (live_b) {
            *reinterpret_cast<float2*>(a.out + (unsigned long long)row_b * a.out_stride + d) = make_float2(o[n][2] * inv_b, o[n][3] * inv_b);
        }
    }
}

extern "C" __global__ void __launch_bounds__(FC_THREADS, 3) brain_flash_cross_relpos_f16_hd64(
    const unsigned int* params, const float* q, const float* k, const float* v, const float* rel_h, const float* rel_w, float* out) {
    const unsigned int bsz = params[0], heads = params[1], t_dec = params[2], t_enc = params[3], hd = params[4];
    const unsigned int q_stride = params[5], q_off = params[6], k_stride = params[7], k_off = params[8];
    const unsigned int v_stride = params[9], v_off = params[10], d_model = params[11], kh = params[12], kw = params[13];
    const unsigned int ntiles_q = (t_dec + FC_ROWS - 1u) / FC_ROWS;
    const unsigned int blk = blockIdx.y * gridDim.x + blockIdx.x;
    if (ntiles_q == 0u || hd != FC_HD) { return; }
    const unsigned int qt = blk % ntiles_q;
    const unsigned int r = blk / ntiles_q;
    const unsigned int h = r % heads;
    const unsigned int b = r / heads;
    if (b >= bsz) { return; }   // block-uniform: no barrier is skipped per-thread

    FcHead a;
    a.q = q + (unsigned long long)b * t_dec * q_stride + q_off + h * FC_HD;
    a.k = k + (unsigned long long)b * t_enc * k_stride + k_off + h * FC_HD;
    a.v = v + (unsigned long long)b * t_enc * v_stride + v_off + h * FC_HD;
    a.rel_h = rel_h + ((unsigned long long)b * heads + h) * t_dec * kh;
    a.rel_w = rel_w + ((unsigned long long)b * heads + h) * t_dec * kw;
    a.out = out + (unsigned long long)b * t_dec * d_model + h * FC_HD;
    a.q_stride = q_stride; a.k_stride = k_stride; a.v_stride = v_stride; a.out_stride = d_model;
    a.t_dec = t_dec; a.t_enc = t_enc; a.kh = kh; a.kw = kw;
    fc_core<true>(a, qt);
}

extern "C" __global__ void __launch_bounds__(FC_THREADS, 3) brain_flash_bidir_f16_hd64(
    const unsigned int* params, const float* qkv, float* out) {
    const unsigned int bsz = params[0], heads = params[1], t = params[2], hd = params[3];
    const unsigned int stride = params[4], q_off = params[5], k_off = params[6], v_off = params[7], d_model = params[8];
    const unsigned int ntiles_q = (t + FC_ROWS - 1u) / FC_ROWS;
    const unsigned int blk = blockIdx.y * gridDim.x + blockIdx.x;
    if (ntiles_q == 0u || hd != FC_HD) { return; }
    const unsigned int qt = blk % ntiles_q;
    const unsigned int r = blk / ntiles_q;
    const unsigned int h = r % heads;
    const unsigned int b = r / heads;
    if (b >= bsz) { return; }

    FcHead a;
    const unsigned long long base = (unsigned long long)b * t * stride + h * FC_HD;
    a.q = qkv + base + q_off;
    a.k = qkv + base + k_off;
    a.v = qkv + base + v_off;
    a.rel_h = nullptr;
    a.rel_w = nullptr;
    a.out = out + (unsigned long long)b * t * d_model + h * FC_HD;
    a.q_stride = stride; a.k_stride = stride; a.v_stride = stride; a.out_stride = d_model;
    a.t_dec = t; a.t_enc = t; a.kh = 1u; a.kw = 1u;
    fc_core<false>(a, qt);
}
