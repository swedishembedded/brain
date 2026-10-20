// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements tensor-core training stacks that agree with
// their fp32 reference. If your team needs expertise in moving the attention
// backward of a fine-tuning step onto tensor cores without losing the gradient,
// then you can procure our services by sending an email to
// info@swedishembedded.com.
//
// Causal grouped-query flash attention BACKWARD at head_dim 128 on fp16 tensor
// cores (`mma.sync.m16n8k16`, fp32 accumulate): the CUDA form of
// `flash_attn_causal_gqa_bwd_dq.wgsl` and `flash_attn_causal_gqa_bwd_dkv.wgsl`
// over dense `[B*T, ...]` buffers, FlashAttention-2 shaped. Four entries, run in
// this order by `gpu_core::provider::cuda::flash_causal_gqa_bwd_steps`:
//
//   brain_flash_bwd_absmax params, d_o, partials   256 blocks: each one's largest |d_o|
//   brain_flash_bwd_scale  params, partials, scale one block: scale = 2^k with max|d_o| * scale in [32, 64)
//   brain_flash_bwd_dq     params, q, k, v, o, d_o, lse, dq, dsum, scale
//   brain_flash_bwd_dkv    params, q, k, v, d_o, lse, dsum, dk, dv, scale
//
//   params : [bsz, n_heads, n_kv_heads, T, head_dim, group] u32 (the portable kernels')
//   q, o, d_o, dq : [bsz * T, n_heads * HD]    k, v, dk, dv : [bsz * T, n_kv_heads * HD]
//   lse    : [bsz, n_heads, T]   the forward's natural-log log-sum-exp of the scaled scores
//   dsum   : [bsz, n_heads, T]   D_i = sum_d dO_i[d] * O_i[d] (of the SCALED dO, see below), written by
//                                 `dq` and read by `dkv`
//   scale  : [2] f32             {s, 1/s}
//
// With scaled scores s_ij = q_i . k_j / sqrt(HD), P_ij = exp(s_ij - lse_i), dP_ij = dO_i . v_j and
// dS_ij = P_ij (dP_ij - D_i):  dQ_i = sum_j dS_ij k_j / sqrt(HD),  dK_j = sum_i dS_ij q_i / sqrt(HD),
// dV_j = sum_i P_ij dO_i  (sums over the causal j <= i; dK and dV also over the query heads of the group).
//
// NUMERICS. Q, K, V, dO and the products P and dS are rounded to fp16 for the MMAs (`cvt.rn.satfinite`); every
// accumulator, D, the log-sum-exp arithmetic and the outputs are fp32. A gradient is far below fp16's normal
// range (a mean-reduced loss makes dO ~ 1e-5 and dS smaller still), so dO is multiplied by one power of two
// `s` for the whole call: dP, D and dS all carry the factor, which `dq`/`dkv` divide out of their outputs
// exactly. `s` comes from the largest |dO| of the call (`brain_flash_bwd_absmax`, `brain_flash_bwd_scale`), so the largest gradients sit
// near 2^5 and fp16's 11 bits are spent on them; rows a thousand times smaller keep normal fp16 values.
//
// Structure (the paged prefill kernel's, `flash_prefill_f16_hd256.cu`): a block is 8 warps = 4 row groups x 2
// column halves; both warps of a group compute the same 16 x 32 score tile and each accumulates half of the 128
// output columns. `dq` owns 64 query rows of one (sample, head) and walks key tiles of 32; `dkv` owns 64 key rows
// of one (sample, kv head) and walks the query tiles of 32 (from the first that can see them) of every query
// head in the group, computing the transposed tiles S^T = K Q^T and dP^T = V dO^T so the key rows are the MMA's
// M dimension and dV, dK accumulate in registers. Smem rows are padded by 8 halves so every `ldmatrix` phase
// touches eight distinct 16-byte bank groups.
//
// No `__restrict__`: brain's device buffers alias by design.

#define HD 128
#define BR 64              // rows (queries in dq, keys in dkv) per block
#define BC 32              // keys (dq) or queries (dkv) per tile
#define THREADS 256
#define KSTEPS (HD / 16)
#define OTILES (HD / 16)   // n8 output tiles per warp (it owns half the columns)
#define HALF_HD (HD / 2)
#define CHUNKS_PER_ROW (HD / 8)
#define ROWS_PER_PASS (THREADS / CHUNKS_PER_ROW)
#define STAGE_ITERS (BC / ROWS_PER_PASS)
#define PAD_HALVES 8
#define ROW_HALVES (HD + PAD_HALVES)
#define LOG2E 1.4426950408889634f
#define SM_SCALE 0.08838834764831845f                 // 1 / sqrt(128)
#define SM_SCALE_LOG2 (LOG2E * SM_SCALE)
#define NEG_BIG (-1.0e30f)

__device__ __forceinline__ unsigned int brain_smem_addr(const void* p) {
    return static_cast<unsigned int>(__cvta_generic_to_shared(p));
}

// {lo, hi} -> packed f16x2 with `lo` in the low half (the lower k index).
__device__ __forceinline__ unsigned int brain_pack_f16x2(float lo, float hi) {
    unsigned int r;
    asm("cvt.rn.satfinite.f16x2.f32 %0, %1, %2;\n" : "=r"(r) : "f"(hi), "f"(lo));
    return r;
}

__device__ __forceinline__ void brain_ldsm_x4(unsigned int addr, unsigned int& r0, unsigned int& r1, unsigned int& r2, unsigned int& r3) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}

__device__ __forceinline__ void brain_ldsm_x4_t(unsigned int addr, unsigned int& r0, unsigned int& r1, unsigned int& r2, unsigned int& r3) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}

// d += a * b, m16n8k16, f16 inputs, f32 accumulate.
__device__ __forceinline__ void brain_mma_f16(float (&d)[4], const unsigned int (&a)[4], unsigned int b0, unsigned int b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// A-operand fragments of a 16-row x HD matrix `src[row][col]` (row stride `stride` floats, rows beyond `rows` read
// as zero), each element multiplied by `mul`: fragment s covers k = 16 s .. 16 s + 15, registers {row g, row g+8} x
// {k, k+8} as the PTX ISA lays them out. `row_a` is this thread's row g of the group, `row_b = row_a + 8`.
__device__ __forceinline__ void brain_a_fragments(unsigned int (&f)[KSTEPS][4], const float* src, unsigned long long stride, unsigned int row_a, unsigned int row_b, bool live_a, bool live_b, unsigned int t4, float mul) {
#pragma unroll
    for (int s = 0; s < KSTEPS; ++s) {
        const unsigned int d = s * 16u + t4 * 2u;
        float2 a0 = make_float2(0.f, 0.f), a1 = a0, a2 = a0, a3 = a0;
        if (live_a) {
            a0 = *reinterpret_cast<const float2*>(src + row_a * stride + d);
            a2 = *reinterpret_cast<const float2*>(src + row_a * stride + d + 8u);
        }
        if (live_b) {
            a1 = *reinterpret_cast<const float2*>(src + row_b * stride + d);
            a3 = *reinterpret_cast<const float2*>(src + row_b * stride + d + 8u);
        }
        f[s][0] = brain_pack_f16x2(a0.x * mul, a0.y * mul);
        f[s][1] = brain_pack_f16x2(a1.x * mul, a1.y * mul);
        f[s][2] = brain_pack_f16x2(a2.x * mul, a2.y * mul);
        f[s][3] = brain_pack_f16x2(a3.x * mul, a3.y * mul);
    }
}

// Stage `BC` rows of `src` (row r at `src + (row_base + r) * stride + col0`, rows >= `rows` zero) into the padded fp16
// tile `dst`, each value multiplied by `mul`: thread chunk e = tid + 256 c is row tid / CHUNKS_PER_ROW + ROWS_PER_PASS c,
// column chunk tid % CHUNKS_PER_ROW of 8 values.
__device__ __forceinline__ void brain_stage_tile(unsigned short* dst, const float* src, unsigned long long stride, unsigned long long row_base, unsigned int first_row, unsigned int rows, unsigned int col0, float mul, unsigned int tid) {
    const unsigned int cc = tid % CHUNKS_PER_ROW;
    const unsigned int row0 = tid / CHUNKS_PER_ROW;
#pragma unroll
    for (int c = 0; c < STAGE_ITERS; ++c) {
        const unsigned int r = row0 + ROWS_PER_PASS * c;
        const unsigned int j = first_row + r;
        float4 lo = make_float4(0.f, 0.f, 0.f, 0.f), hi = lo;
        if (j < rows) {
            const float* p = src + (row_base + j) * stride + col0 + cc * 8u;
            lo = *reinterpret_cast<const float4*>(p);
            hi = *reinterpret_cast<const float4*>(p + 4);
        }
        uint4 v;
        v.x = brain_pack_f16x2(lo.x * mul, lo.y * mul);
        v.y = brain_pack_f16x2(lo.z * mul, lo.w * mul);
        v.z = brain_pack_f16x2(hi.x * mul, hi.y * mul);
        v.w = brain_pack_f16x2(hi.z * mul, hi.w * mul);
        *reinterpret_cast<uint4*>(dst + r * ROW_HALVES + cc * 8u) = v;
    }
}

// ---- the scale of the call --------------------------------------------------------------------------------

// Pass 1: ABSMAX_BLOCKS blocks each take a grid-strided share of dO (float4 loads: the element count is a
// multiple of HD) and write their largest |d_o|. A single block over the whole tensor ran at under 2 percent of
// the memory roof (a 30 MB dO took 0.65 ms per layer, 7 percent of a small model's training step).
#define ABSMAX_BLOCKS 256
#define ABSMAX_THREADS 512
extern "C" __global__ void __launch_bounds__(ABSMAX_THREADS) brain_flash_bwd_absmax(const unsigned int* params, const float* d_o, float* partials) {
    const unsigned long long n4 = (unsigned long long)params[0] * params[3] * params[1] * (HD / 4);
    float m = 0.f;
    const float4* src = reinterpret_cast<const float4*>(d_o);
    for (unsigned long long i = (unsigned long long)blockIdx.x * ABSMAX_THREADS + threadIdx.x; i < n4; i += (unsigned long long)ABSMAX_BLOCKS * ABSMAX_THREADS) {
        const float4 v = src[i];
        m = fmaxf(m, fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w))));
    }
    __shared__ float part[ABSMAX_THREADS / 32];
    for (int o = 16; o > 0; o >>= 1) { m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o)); }
    if ((threadIdx.x & 31u) == 0u) { part[threadIdx.x >> 5] = m; }
    __syncthreads();
    if (threadIdx.x == 0u) {
        float mx = 0.f;
        for (int w = 0; w < ABSMAX_THREADS / 32; ++w) { mx = fmaxf(mx, part[w]); }
        partials[blockIdx.x] = mx;
    }
}

// Pass 2: one block reduces the partials and derives the call's power-of-two scale.
extern "C" __global__ void __launch_bounds__(ABSMAX_BLOCKS) brain_flash_bwd_scale(const unsigned int* params, const float* partials, float* scale) {
    (void)params;
    float m = partials[threadIdx.x];
    __shared__ float part[ABSMAX_BLOCKS / 32];
    for (int o = 16; o > 0; o >>= 1) { m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o)); }
    if ((threadIdx.x & 31u) == 0u) { part[threadIdx.x >> 5] = m; }
    __syncthreads();
    if (threadIdx.x == 0u) {
        float mx = 0.f;
        for (int w = 0; w < ABSMAX_BLOCKS / 32; ++w) { mx = fmaxf(mx, part[w]); }
        float s = 1.0f;
        if (mx > 0.f && mx < 3.0e38f) {
            int e;
            frexpf(mx, &e);                       // mx = f 2^e, f in [0.5, 1): the largest scaled value is f 2^6 in [32, 64)
            e = max(-120, min(120, e));
            s = ldexpf(1.0f, 6 - e);
        }
        scale[0] = s;
        scale[1] = 1.0f / s;
    }
}

// ---- dQ ---------------------------------------------------------------------------------------------------

extern "C" __global__ void __launch_bounds__(THREADS, 1) brain_flash_bwd_dq(const unsigned int* params, const float* q, const float* k, const float* v, const float* o, const float* d_o, const float* lse, float* dq, float* dsum, const float* scale) {
    const unsigned int bsz = params[0];
    const unsigned int n_heads = params[1];
    const unsigned int n_kv_heads = params[2];
    const unsigned int T = params[3];
    const unsigned int group = params[5];

    const unsigned int blk = blockIdx.y * gridDim.x + blockIdx.x;
    const unsigned int ntiles_q = (T + BR - 1u) / BR;
    if (ntiles_q == 0u) { return; }
    const unsigned int qt = blk % ntiles_q;
    const unsigned int h = (blk / ntiles_q) % n_heads;
    const unsigned int sample = (blk / ntiles_q) / n_heads;
    if (sample >= bsz) { return; }  // block-uniform: no barrier is skipped per-thread
    const unsigned long long base_row = (unsigned long long)sample * T;
    const unsigned int hkv = h / (group == 0u ? 1u : group);
    const unsigned long long q_row = (unsigned long long)n_heads * HD;
    const unsigned long long kv_row = (unsigned long long)n_kv_heads * HD;
    const unsigned long long row_vec = ((unsigned long long)sample * n_heads + h) * T;   // lse / dsum row base
    const float sc = scale[0], inv_sc = scale[1];

    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid & 31u;
    const unsigned int warp = tid >> 5;
    const unsigned int rg = warp >> 1;
    const unsigned int hh = warp & 1u;
    const unsigned int g = lane >> 2;
    const unsigned int t4 = lane & 3u;

    const unsigned int i0 = qt * BR;
    const unsigned int row_a = i0 + rg * 16u + g;
    const unsigned int row_b = row_a + 8u;
    const bool live_a = row_a < T, live_b = row_b < T;
    const unsigned int len_a = live_a ? row_a + 1u : 0u;
    const unsigned int len_b = live_b ? row_b + 1u : 0u;
    const unsigned int max_t = min(i0 + BR, T);
    const unsigned int ntiles_k = (max_t + BC - 1u) / BC;

    __shared__ __align__(16) unsigned short Ks[BC * ROW_HALVES];
    __shared__ __align__(16) unsigned short Vs[BC * ROW_HALVES];

    unsigned int qf[KSTEPS][4], dof[KSTEPS][4];
    brain_a_fragments(qf, q + (base_row * q_row + h * HD), q_row, row_a, row_b, live_a, live_b, t4, 1.0f);
    brain_a_fragments(dof, d_o + (base_row * q_row + h * HD), q_row, row_a, row_b, live_a, live_b, t4, sc);

    // D_i = dO_i . O_i over the scaled dO, in fp32 from the unrounded values.
    float dp_a = 0.f, dp_b = 0.f;
#pragma unroll
    for (int s = 0; s < KSTEPS; ++s) {
        const unsigned int d = s * 16u + t4 * 2u;
        if (live_a) {
            const float2 x0 = *reinterpret_cast<const float2*>(d_o + (base_row + row_a) * q_row + h * HD + d), y0 = *reinterpret_cast<const float2*>(o + (base_row + row_a) * q_row + h * HD + d);
            const float2 x1 = *reinterpret_cast<const float2*>(d_o + (base_row + row_a) * q_row + h * HD + d + 8u), y1 = *reinterpret_cast<const float2*>(o + (base_row + row_a) * q_row + h * HD + d + 8u);
            dp_a += x0.x * y0.x + x0.y * y0.y + x1.x * y1.x + x1.y * y1.y;
        }
        if (live_b) {
            const float2 x0 = *reinterpret_cast<const float2*>(d_o + (base_row + row_b) * q_row + h * HD + d), y0 = *reinterpret_cast<const float2*>(o + (base_row + row_b) * q_row + h * HD + d);
            const float2 x1 = *reinterpret_cast<const float2*>(d_o + (base_row + row_b) * q_row + h * HD + d + 8u), y1 = *reinterpret_cast<const float2*>(o + (base_row + row_b) * q_row + h * HD + d + 8u);
            dp_b += x0.x * y0.x + x0.y * y0.y + x1.x * y1.x + x1.y * y1.y;
        }
    }
    dp_a += __shfl_xor_sync(0xffffffffu, dp_a, 1);
    dp_a += __shfl_xor_sync(0xffffffffu, dp_a, 2);
    dp_b += __shfl_xor_sync(0xffffffffu, dp_b, 1);
    dp_b += __shfl_xor_sync(0xffffffffu, dp_b, 2);
    const float D_a = dp_a * sc, D_b = dp_b * sc;
    const float lse2_a = live_a ? lse[row_vec + row_a] * LOG2E : 0.f;
    const float lse2_b = live_b ? lse[row_vec + row_b] * LOG2E : 0.f;
    if (hh == 0u && t4 == 0u) {
        if (live_a) { dsum[row_vec + row_a] = D_a; }
        if (live_b) { dsum[row_vec + row_b] = D_b; }
    }

    float acc[OTILES][4];
#pragma unroll
    for (int n = 0; n < OTILES; ++n) { acc[n][0] = acc[n][1] = acc[n][2] = acc[n][3] = 0.f; }

    const unsigned int k_row_lane = (lane & 7u) + ((lane >> 4) << 3);
    const unsigned int k_col_lane = ((lane >> 3) & 1u) << 3;
    const unsigned int v_row_lane = (lane & 7u) + (((lane >> 3) & 1u) << 3);
    const unsigned int v_col_lane = (lane >> 4) << 3;
    const unsigned int ks_base = brain_smem_addr(Ks);
    const unsigned int vs_base = brain_smem_addr(Vs);

    for (unsigned int kt = 0; kt < ntiles_k; ++kt) {
        __syncthreads();                   // every warp is done with the previous tile
        brain_stage_tile(Ks, k, kv_row, base_row, kt * BC, max_t, hkv * HD, 1.0f, tid);
        brain_stage_tile(Vs, v, kv_row, base_row, kt * BC, max_t, hkv * HD, 1.0f, tid);
        __syncthreads();

        // S = Q K^T and dP = dO V^T for 16 rows x 32 keys.
        float sacc[4][4], pacc[4][4];
#pragma unroll
        for (int n = 0; n < 4; ++n) {
            sacc[n][0] = sacc[n][1] = sacc[n][2] = sacc[n][3] = 0.f;
            pacc[n][0] = pacc[n][1] = pacc[n][2] = pacc[n][3] = 0.f;
        }
#pragma unroll
        for (int s = 0; s < KSTEPS; ++s) {
#pragma unroll
            for (int np = 0; np < 2; ++np) {
                unsigned int b0, b1, b2, b3;
                brain_ldsm_x4(ks_base + ((np * 16u + k_row_lane) * ROW_HALVES + s * 16u + k_col_lane) * 2u, b0, b1, b2, b3);
                brain_mma_f16(sacc[2 * np], qf[s], b0, b1);
                brain_mma_f16(sacc[2 * np + 1], qf[s], b2, b3);
                brain_ldsm_x4(vs_base + ((np * 16u + k_row_lane) * ROW_HALVES + s * 16u + k_col_lane) * 2u, b0, b1, b2, b3);
                brain_mma_f16(pacc[2 * np], dof[s], b0, b1);
                brain_mma_f16(pacc[2 * np + 1], dof[s], b2, b3);
            }
        }

        // dS = P (dP - D), with P = exp(s - lse) rebuilt from the forward's log-sum-exp; a masked key has weight 0.
#pragma unroll
        for (int n = 0; n < 4; ++n) {
            const unsigned int j0 = kt * BC + n * 8u + t4 * 2u;
            const bool v0 = j0 < len_a, v1 = j0 + 1u < len_a, v2 = j0 < len_b, v3 = j0 + 1u < len_b;
            sacc[n][0] = v0 ? exp2f(sacc[n][0] * SM_SCALE_LOG2 - lse2_a) * (pacc[n][0] - D_a) : 0.f;
            sacc[n][1] = v1 ? exp2f(sacc[n][1] * SM_SCALE_LOG2 - lse2_a) * (pacc[n][1] - D_a) : 0.f;
            sacc[n][2] = v2 ? exp2f(sacc[n][2] * SM_SCALE_LOG2 - lse2_b) * (pacc[n][2] - D_b) : 0.f;
            sacc[n][3] = v3 ? exp2f(sacc[n][3] * SM_SCALE_LOG2 - lse2_b) * (pacc[n][3] - D_b) : 0.f;
        }
        unsigned int pf[2][4];
#pragma unroll
        for (int kk = 0; kk < 2; ++kk) {
            pf[kk][0] = brain_pack_f16x2(sacc[2 * kk][0], sacc[2 * kk][1]);
            pf[kk][1] = brain_pack_f16x2(sacc[2 * kk][2], sacc[2 * kk][3]);
            pf[kk][2] = brain_pack_f16x2(sacc[2 * kk + 1][0], sacc[2 * kk + 1][1]);
            pf[kk][3] = brain_pack_f16x2(sacc[2 * kk + 1][2], sacc[2 * kk + 1][3]);
        }

        // dQ += dS K for this warp's HALF_HD columns (K is the transposed B operand: keys are the MMA's k).
#pragma unroll
        for (int kk = 0; kk < 2; ++kk) {
#pragma unroll
            for (int np = 0; np < OTILES / 2; ++np) {
                unsigned int b0, b1, b2, b3;
                brain_ldsm_x4_t(ks_base + (((kk * 16u + v_row_lane) * ROW_HALVES) + hh * HALF_HD + np * 16u + v_col_lane) * 2u, b0, b1, b2, b3);
                brain_mma_f16(acc[2 * np], pf[kk], b0, b1);
                brain_mma_f16(acc[2 * np + 1], pf[kk], b2, b3);
            }
        }
    }

    const float out_scale = SM_SCALE * inv_sc;
#pragma unroll
    for (int n = 0; n < OTILES; ++n) {
        const unsigned int d = hh * HALF_HD + n * 8u + t4 * 2u;
        if (live_a) { *reinterpret_cast<float2*>(dq + (base_row + row_a) * q_row + h * HD + d) = make_float2(acc[n][0] * out_scale, acc[n][1] * out_scale); }
        if (live_b) { *reinterpret_cast<float2*>(dq + (base_row + row_b) * q_row + h * HD + d) = make_float2(acc[n][2] * out_scale, acc[n][3] * out_scale); }
    }
}

// ---- dK and dV --------------------------------------------------------------------------------------------

extern "C" __global__ void __launch_bounds__(THREADS, 1) brain_flash_bwd_dkv(const unsigned int* params, const float* q, const float* k, const float* v, const float* d_o, const float* lse, const float* dsum, float* dk, float* dv, const float* scale) {
    const unsigned int bsz = params[0];
    const unsigned int n_heads = params[1];
    const unsigned int n_kv_heads = params[2];
    const unsigned int T = params[3];
    const unsigned int group = params[5];

    const unsigned int blk = blockIdx.y * gridDim.x + blockIdx.x;
    const unsigned int ntiles_k = (T + BR - 1u) / BR;
    if (ntiles_k == 0u) { return; }
    const unsigned int kt = blk % ntiles_k;
    const unsigned int hkv = (blk / ntiles_k) % n_kv_heads;
    const unsigned int sample = (blk / ntiles_k) / n_kv_heads;
    if (sample >= bsz) { return; }
    const unsigned long long base_row = (unsigned long long)sample * T;
    const unsigned long long q_row = (unsigned long long)n_heads * HD;
    const unsigned long long kv_row = (unsigned long long)n_kv_heads * HD;
    const float sc = scale[0], inv_sc = scale[1];

    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid & 31u;
    const unsigned int warp = tid >> 5;
    const unsigned int rg = warp >> 1;
    const unsigned int hh = warp & 1u;
    const unsigned int g = lane >> 2;
    const unsigned int t4 = lane & 3u;

    const unsigned int j0 = kt * BR;                       // first key of the block
    const unsigned int key_a = j0 + rg * 16u + g;          // this thread's key rows
    const unsigned int key_b = key_a + 8u;
    const bool live_a = key_a < T, live_b = key_b < T;
    const unsigned int nq_tiles = (T + BC - 1u) / BC;
    const unsigned int q_first = j0 / BC;                  // the first query tile that can see a key of this block

    __shared__ __align__(16) unsigned short Qs[BC * ROW_HALVES];
    __shared__ __align__(16) unsigned short Os[BC * ROW_HALVES];   // the scaled dO tile
    __shared__ float Ls[BC];                                       // lse * log2(e) of the tile's queries, +inf for a dead one
    __shared__ float Ds[BC];

    unsigned int kf[KSTEPS][4], vf[KSTEPS][4];
    brain_a_fragments(kf, k + (base_row * kv_row + hkv * HD), kv_row, key_a, key_b, live_a, live_b, t4, 1.0f);
    brain_a_fragments(vf, v + (base_row * kv_row + hkv * HD), kv_row, key_a, key_b, live_a, live_b, t4, 1.0f);

    float adk[OTILES][4], adv[OTILES][4];
#pragma unroll
    for (int n = 0; n < OTILES; ++n) {
        adk[n][0] = adk[n][1] = adk[n][2] = adk[n][3] = 0.f;
        adv[n][0] = adv[n][1] = adv[n][2] = adv[n][3] = 0.f;
    }

    const unsigned int k_row_lane = (lane & 7u) + ((lane >> 4) << 3);
    const unsigned int k_col_lane = ((lane >> 3) & 1u) << 3;
    const unsigned int v_row_lane = (lane & 7u) + (((lane >> 3) & 1u) << 3);
    const unsigned int v_col_lane = (lane >> 4) << 3;
    const unsigned int qs_base = brain_smem_addr(Qs);
    const unsigned int os_base = brain_smem_addr(Os);

    for (unsigned int hq = hkv * group; hq < (hkv + 1u) * group; ++hq) {
        const unsigned long long row_vec = ((unsigned long long)sample * n_heads + hq) * T;
        for (unsigned int qi = q_first; qi < nq_tiles; ++qi) {
            const unsigned int i_first = qi * BC;
            __syncthreads();               // every warp is done with the previous tile
            brain_stage_tile(Qs, q, q_row, base_row, i_first, T, hq * HD, 1.0f, tid);
            brain_stage_tile(Os, d_o, q_row, base_row, i_first, T, hq * HD, sc, tid);
            if (tid < BC) {
                const unsigned int i = i_first + tid;
                Ls[tid] = i < T ? lse[row_vec + i] * LOG2E : __int_as_float(0x7f800000);   // exp2(s - inf) = 0: a dead query adds nothing
                Ds[tid] = i < T ? dsum[row_vec + i] : 0.f;
            }
            __syncthreads();

            // S^T = K Q^T and dP^T = V dO^T for 16 keys x 32 queries.
            float sacc[4][4], pacc[4][4];
#pragma unroll
            for (int n = 0; n < 4; ++n) {
                sacc[n][0] = sacc[n][1] = sacc[n][2] = sacc[n][3] = 0.f;
                pacc[n][0] = pacc[n][1] = pacc[n][2] = pacc[n][3] = 0.f;
            }
#pragma unroll
            for (int s = 0; s < KSTEPS; ++s) {
#pragma unroll
                for (int np = 0; np < 2; ++np) {
                    unsigned int b0, b1, b2, b3;
                    brain_ldsm_x4(qs_base + ((np * 16u + k_row_lane) * ROW_HALVES + s * 16u + k_col_lane) * 2u, b0, b1, b2, b3);
                    brain_mma_f16(sacc[2 * np], kf[s], b0, b1);
                    brain_mma_f16(sacc[2 * np + 1], kf[s], b2, b3);
                    brain_ldsm_x4(os_base + ((np * 16u + k_row_lane) * ROW_HALVES + s * 16u + k_col_lane) * 2u, b0, b1, b2, b3);
                    brain_mma_f16(pacc[2 * np], vf[s], b0, b1);
                    brain_mma_f16(pacc[2 * np + 1], vf[s], b2, b3);
                }
            }

            // P^T and dS^T: accumulator (row, col) = (key, query); key j sees query i iff j <= i.
            float pt[4][4];
#pragma unroll
            for (int n = 0; n < 4; ++n) {
                const unsigned int c0 = n * 8u + t4 * 2u;        // query column within the tile
                const unsigned int i0 = i_first + c0, i1 = i0 + 1u;
                const float l0 = Ls[c0], l1 = Ls[c0 + 1u], d0 = Ds[c0], d1 = Ds[c0 + 1u];
                const bool m00 = key_a <= i0 && live_a, m01 = key_a <= i1 && live_a, m10 = key_b <= i0 && live_b, m11 = key_b <= i1 && live_b;
                const float p00 = m00 ? exp2f(sacc[n][0] * SM_SCALE_LOG2 - l0) : 0.f;
                const float p01 = m01 ? exp2f(sacc[n][1] * SM_SCALE_LOG2 - l1) : 0.f;
                const float p10 = m10 ? exp2f(sacc[n][2] * SM_SCALE_LOG2 - l0) : 0.f;
                const float p11 = m11 ? exp2f(sacc[n][3] * SM_SCALE_LOG2 - l1) : 0.f;
                pt[n][0] = p00; pt[n][1] = p01; pt[n][2] = p10; pt[n][3] = p11;
                sacc[n][0] = p00 * (pacc[n][0] - d0);
                sacc[n][1] = p01 * (pacc[n][1] - d1);
                sacc[n][2] = p10 * (pacc[n][2] - d0);
                sacc[n][3] = p11 * (pacc[n][3] - d1);
            }
            unsigned int pf[2][4], sf[2][4];
#pragma unroll
            for (int kk = 0; kk < 2; ++kk) {
                pf[kk][0] = brain_pack_f16x2(pt[2 * kk][0], pt[2 * kk][1]);
                pf[kk][1] = brain_pack_f16x2(pt[2 * kk][2], pt[2 * kk][3]);
                pf[kk][2] = brain_pack_f16x2(pt[2 * kk + 1][0], pt[2 * kk + 1][1]);
                pf[kk][3] = brain_pack_f16x2(pt[2 * kk + 1][2], pt[2 * kk + 1][3]);
                sf[kk][0] = brain_pack_f16x2(sacc[2 * kk][0], sacc[2 * kk][1]);
                sf[kk][1] = brain_pack_f16x2(sacc[2 * kk][2], sacc[2 * kk][3]);
                sf[kk][2] = brain_pack_f16x2(sacc[2 * kk + 1][0], sacc[2 * kk + 1][1]);
                sf[kk][3] = brain_pack_f16x2(sacc[2 * kk + 1][2], sacc[2 * kk + 1][3]);
            }

            // dV += P^T dO and dK += dS^T Q for this warp's HALF_HD columns (the queries are the MMA's k).
#pragma unroll
            for (int kk = 0; kk < 2; ++kk) {
#pragma unroll
                for (int np = 0; np < OTILES / 2; ++np) {
                    unsigned int b0, b1, b2, b3;
                    brain_ldsm_x4_t(os_base + (((kk * 16u + v_row_lane) * ROW_HALVES) + hh * HALF_HD + np * 16u + v_col_lane) * 2u, b0, b1, b2, b3);
                    brain_mma_f16(adv[2 * np], pf[kk], b0, b1);
                    brain_mma_f16(adv[2 * np + 1], pf[kk], b2, b3);
                    brain_ldsm_x4_t(qs_base + (((kk * 16u + v_row_lane) * ROW_HALVES) + hh * HALF_HD + np * 16u + v_col_lane) * 2u, b0, b1, b2, b3);
                    brain_mma_f16(adk[2 * np], sf[kk], b0, b1);
                    brain_mma_f16(adk[2 * np + 1], sf[kk], b2, b3);
                }
            }
        }
    }

    const float k_scale = SM_SCALE * inv_sc;
#pragma unroll
    for (int n = 0; n < OTILES; ++n) {
        const unsigned int d = hh * HALF_HD + n * 8u + t4 * 2u;
        if (live_a) {
            *reinterpret_cast<float2*>(dk + (base_row + key_a) * kv_row + hkv * HD + d) = make_float2(adk[n][0] * k_scale, adk[n][1] * k_scale);
            *reinterpret_cast<float2*>(dv + (base_row + key_a) * kv_row + hkv * HD + d) = make_float2(adv[n][0] * inv_sc, adv[n][1] * inv_sc);
        }
        if (live_b) {
            *reinterpret_cast<float2*>(dk + (base_row + key_b) * kv_row + hkv * HD + d) = make_float2(adk[n][2] * k_scale, adk[n][3] * k_scale);
            *reinterpret_cast<float2*>(dv + (base_row + key_b) * kv_row + hkv * HD + d) = make_float2(adv[n][2] * inv_sc, adv[n][3] * inv_sc);
        }
    }
}
