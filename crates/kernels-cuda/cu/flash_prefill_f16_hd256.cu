// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements long-context inference throughput for its
// clients. If your team needs expertise in putting causal attention on tensor
// cores without giving up the paged KV layout its serving stack is built on,
// then you can procure our services by sending an email to
// info@swedishembedded.com.
//
// Fused causal paged-attention PREFILL at head_dim = 256 on fp16 tensor cores
// (`mma.sync.m16n8k16`, fp32 accumulate): the CUDA form of
// `paged_flash_prefill_hd256.wgsl`, with the identical argument contract,
// launch geometry (256 threads, `n_heads * ceil(bsz / 64)` blocks, query-tile
// fastest) and addressing.
//
//   params       : [bsz, n_heads, n_kv_heads, head_dim, group, block_size, max_bt] u32
//   q            : [bsz, n_heads * 256]               f32
//   pool_k/pool_v: [num_blocks * block_size, n_kv_heads * 256]  f32 (paged pool)
//   block_tables : [bsz, max_bt]  u32  (every row of a 64-row tile reads the
//                  table row of the tile's FIRST row, as the portable kernel)
//   seq_lens     : [bsz]          u32  (live key count per query row, non-
//                  decreasing within a tile: causality is carried by it)
//   ctx          : [bsz, n_heads * 256]               f32
//
// head_dim must be 256 (the caller checks; the entry does not re-validate it).
//
// Numerics. Q, K, V and the softmax weights P are rounded to fp16 for the MMAs
// (`cvt.rn.satfinite`, so an outlier saturates instead of becoming inf); every
// accumulator, the running max, the running sum and the output are fp32. That
// is the standard fp16 flash-attention recipe and it is NOT bit-identical to the
// fp32 portable kernel: the agreement bar is stated by
// `crates/gpu-core/tests/cuda_flash_prefill.rs` as an absolute error relative
// to the output scale. The softmax scale 1/sqrt(256) = 1/16 is an exact power of
// two, so folding it into Q before the fp16 rounding costs no precision.
//
// Structure (FlashAttention-2 shaped, sized for the 48 KiB static shared limit):
//   * a block is 8 warps = 4 row groups x 2 head-dim halves. A row group owns 16
//     query rows; its two warps each compute the FULL score tile for those rows
//     (redundantly - 96 MMAs per key tile per warp instead of 64) and each
//     accumulates half of the 256 output columns. That keeps the output
//     accumulator at 64 registers per thread, where one warp holding all 256
//     columns would need 128 and spill. Q lives in registers as fp16 MMA
//     fragments for the whole block.
//   * keys are processed 32 at a time. K and V tiles are staged fp32 -> fp16
//     through registers (the pool is fp32, so there is no async copy to make);
//     the next tile's global loads are issued before the matrix work that
//     hides them. Smem rows are padded by 8 halves so every `ldmatrix` phase
//     touches eight distinct 16-byte bank groups.
//   * online softmax in the exp2 domain; masked scores contribute exactly 0,
//     and a row with no live key writes 0 (as the portable kernel).
//
// No `__restrict__`: brain's device buffers alias by design.

#define BR 64              // query rows per block
#define BC 32              // keys per tile
#define HD 256             // head_dim
#define PAD_HALVES 8       // smem row padding
#define ROW_HALVES (HD + PAD_HALVES)
#define THREADS 256
#define LOG2E 1.4426950408889634f
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

extern "C" __global__ void __launch_bounds__(THREADS, 1) brain_flash_prefill_f16_hd256(const unsigned int* params,
                                                                                       const float* q,
                                                                                       const float* pool_k,
                                                                                       const float* pool_v,
                                                                                       const unsigned int* block_tables,
                                                                                       const unsigned int* seq_lens,
                                                                                       float* ctx) {
    const unsigned int bsz = params[0];
    const unsigned int n_heads = params[1];
    const unsigned int n_kv_heads = params[2];
    const unsigned int group = params[4];
    const unsigned int block_size = params[5];
    const unsigned int max_bt = params[6];

    const unsigned int blk = blockIdx.y * gridDim.x + blockIdx.x;
    const unsigned int ntiles_q = (bsz + BR - 1u) / BR;
    if (ntiles_q == 0u) { return; }
    const unsigned int qt = blk % ntiles_q;
    const unsigned int h = blk / ntiles_q;
    if (h >= n_heads) { return; }  // block-uniform: no barrier is skipped per-thread

    const unsigned int hkv = h / (group == 0u ? 1u : group);
    const unsigned long long q_row = (unsigned long long)n_heads * HD;
    const unsigned long long kv_row = (unsigned long long)n_kv_heads * HD;

    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid & 31u;
    const unsigned int warp = tid >> 5;
    const unsigned int rg = warp >> 1;       // row group: 16 query rows
    const unsigned int hh = warp & 1u;       // which half of the 256 output columns
    const unsigned int g = lane >> 2;        // MMA groupID
    const unsigned int t4 = lane & 3u;       // MMA threadID_in_group

    const unsigned int i0 = qt * BR;
    const unsigned int row_a = i0 + rg * 16u + g;       // fragment rows g and g + 8
    const unsigned int row_b = row_a + 8u;
    const bool live_a = row_a < bsz, live_b = row_b < bsz;
    const unsigned int len_a = live_a ? seq_lens[row_a] : 0u;
    const unsigned int len_b = live_b ? seq_lens[row_b] : 0u;
    const unsigned int max_t = seq_lens[min(i0 + BR - 1u, bsz - 1u)];
    const unsigned int ntiles_k = (max_t + BC - 1u) / BC;
    const unsigned int bt_row = i0;

    __shared__ __align__(16) unsigned short Ks[BC * ROW_HALVES];
    __shared__ __align__(16) unsigned short Vs[BC * ROW_HALVES];

    // ---- Q as fp16 A-fragments: 16 k16-steps x 4 registers, softmax scale folded in.
    const float scale = 0.0625f;  // 1/sqrt(256), exact in binary
    unsigned int qf[16][4];
#pragma unroll
    for (int s = 0; s < 16; ++s) {
        const unsigned int d = s * 16u + t4 * 2u;
        float2 a0 = make_float2(0.f, 0.f), a1 = a0, a2 = a0, a3 = a0;
        if (live_a) {
            a0 = *reinterpret_cast<const float2*>(q + row_a * q_row + h * HD + d);
            a2 = *reinterpret_cast<const float2*>(q + row_a * q_row + h * HD + d + 8u);
        }
        if (live_b) {
            a1 = *reinterpret_cast<const float2*>(q + row_b * q_row + h * HD + d);
            a3 = *reinterpret_cast<const float2*>(q + row_b * q_row + h * HD + d + 8u);
        }
        qf[s][0] = brain_pack_f16x2(a0.x * scale, a0.y * scale);
        qf[s][1] = brain_pack_f16x2(a1.x * scale, a1.y * scale);
        qf[s][2] = brain_pack_f16x2(a2.x * scale, a2.y * scale);
        qf[s][3] = brain_pack_f16x2(a3.x * scale, a3.y * scale);
    }

    float o[16][4];
#pragma unroll
    for (int n = 0; n < 16; ++n) { o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.f; }
    float m_a = NEG_BIG, m_b = NEG_BIG, l_a = 0.f, l_b = 0.f;

    // Staging: 4 chunks of 8 consecutive head-dim values per thread. Chunk e
    // = tid + 256 c is key row (tid >> 5) + 8 c, column chunk tid & 31, so a
    // warp reads one key row (1 KiB, coalesced) and writes 512 contiguous
    // smem bytes (conflict-free).
    const unsigned int cc = tid & 31u;
    float4 raw[4][2];
    auto issue = [&](const float* pool, unsigned int kt) {
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            const unsigned int j = kt * BC + (tid >> 5) + 8u * c;
            float4 lo = make_float4(0.f, 0.f, 0.f, 0.f), hi = lo;
            if (j < max_t) {
                const unsigned int physical = block_tables[(unsigned long long)bt_row * max_bt + j / block_size];
                const float* src = pool + ((unsigned long long)physical * block_size + (j % block_size)) * kv_row + hkv * HD + cc * 8u;
                lo = *reinterpret_cast<const float4*>(src);
                hi = *reinterpret_cast<const float4*>(src + 4);
            }
            raw[c][0] = lo;
            raw[c][1] = hi;
        }
    };
    auto commit = [&](unsigned short* dst) {
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            const unsigned int kr = (tid >> 5) + 8u * c;
            uint4 v;
            v.x = brain_pack_f16x2(raw[c][0].x, raw[c][0].y);
            v.y = brain_pack_f16x2(raw[c][0].z, raw[c][0].w);
            v.z = brain_pack_f16x2(raw[c][1].x, raw[c][1].y);
            v.w = brain_pack_f16x2(raw[c][1].z, raw[c][1].w);
            *reinterpret_cast<uint4*>(dst + kr * ROW_HALVES + cc * 8u) = v;
        }
    };

    if (ntiles_k > 0u) { issue(pool_k, 0u); }

    // ldmatrix lane geometry (see the MMA fragment layouts in the PTX ISA).
    //  K (B operand of S, non-transposed): matrices are (keys 0-7, k 0-7),
    //  (keys 0-7, k 8-15), (keys 8-15, k 0-7), (keys 8-15, k 8-15).
    //  V (B operand of O, transposed): matrices are (keys 0-7, d 0-7),
    //  (keys 8-15, d 0-7), (keys 0-7, d 8-15), (keys 8-15, d 8-15).
    const unsigned int k_row_lane = (lane & 7u) + ((lane >> 4) << 3);
    const unsigned int k_col_lane = ((lane >> 3) & 1u) << 3;
    const unsigned int v_row_lane = (lane & 7u) + (((lane >> 3) & 1u) << 3);
    const unsigned int v_col_lane = (lane >> 4) << 3;
    const unsigned int ks_base = brain_smem_addr(Ks);
    const unsigned int vs_base = brain_smem_addr(Vs);

    for (unsigned int kt = 0; kt < ntiles_k; ++kt) {
        __syncthreads();                   // every warp is done with the previous tile's K and V
        commit(Ks);
        issue(pool_v, kt);                 // in flight while the scores are computed
        __syncthreads();                   // K visible

        // ---- S = Q K^T for 16 rows x 32 keys.
        float sacc[4][4];
#pragma unroll
        for (int n = 0; n < 4; ++n) { sacc[n][0] = sacc[n][1] = sacc[n][2] = sacc[n][3] = 0.f; }
#pragma unroll
        for (int s = 0; s < 16; ++s) {
#pragma unroll
            for (int np = 0; np < 2; ++np) {
                unsigned int b0, b1, b2, b3;
                brain_ldsm_x4(ks_base + ((np * 16u + k_row_lane) * ROW_HALVES + s * 16u + k_col_lane) * 2u, b0, b1, b2, b3);
                brain_mma_f16(sacc[2 * np], qf[s], b0, b1);
                brain_mma_f16(sacc[2 * np + 1], qf[s], b2, b3);
            }
        }

        // ---- online softmax (exp2 domain) over this tile's 32 keys.
        float mx_a = NEG_BIG, mx_b = NEG_BIG;
#pragma unroll
        for (int n = 0; n < 4; ++n) {
            const unsigned int j0 = kt * BC + n * 8u + t4 * 2u;
            sacc[n][0] = (j0 < len_a) ? sacc[n][0] * LOG2E : NEG_BIG;
            sacc[n][1] = (j0 + 1u < len_a) ? sacc[n][1] * LOG2E : NEG_BIG;
            sacc[n][2] = (j0 < len_b) ? sacc[n][2] * LOG2E : NEG_BIG;
            sacc[n][3] = (j0 + 1u < len_b) ? sacc[n][3] * LOG2E : NEG_BIG;
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
        for (int n = 0; n < 4; ++n) {
            // A masked score is NEG_BIG; when the whole row is masked mn is
            // NEG_BIG too and the difference is 0, so the weight is forced to 0.
            sacc[n][0] = (sacc[n][0] > NEG_BIG) ? exp2f(sacc[n][0] - mn_a) : 0.f;
            sacc[n][1] = (sacc[n][1] > NEG_BIG) ? exp2f(sacc[n][1] - mn_a) : 0.f;
            sacc[n][2] = (sacc[n][2] > NEG_BIG) ? exp2f(sacc[n][2] - mn_b) : 0.f;
            sacc[n][3] = (sacc[n][3] > NEG_BIG) ? exp2f(sacc[n][3] - mn_b) : 0.f;
            sum_a += sacc[n][0] + sacc[n][1];
            sum_b += sacc[n][2] + sacc[n][3];
        }
        l_a = l_a * al_a + sum_a;
        l_b = l_b * al_b + sum_b;
#pragma unroll
        for (int n = 0; n < 16; ++n) {
            o[n][0] *= al_a;
            o[n][1] *= al_a;
            o[n][2] *= al_b;
            o[n][3] *= al_b;
        }

        // P as fp16 A-fragments, straight from the score accumulators.
        unsigned int pf[2][4];
#pragma unroll
        for (int kk = 0; kk < 2; ++kk) {
            pf[kk][0] = brain_pack_f16x2(sacc[2 * kk][0], sacc[2 * kk][1]);
            pf[kk][1] = brain_pack_f16x2(sacc[2 * kk][2], sacc[2 * kk][3]);
            pf[kk][2] = brain_pack_f16x2(sacc[2 * kk + 1][0], sacc[2 * kk + 1][1]);
            pf[kk][3] = brain_pack_f16x2(sacc[2 * kk + 1][2], sacc[2 * kk + 1][3]);
        }

        commit(Vs);                        // V loads have had the whole score phase
        if (kt + 1u < ntiles_k) { issue(pool_k, kt + 1u); }   // next K in flight under the PV MMAs
        __syncthreads();                   // V visible

        // ---- O += P V for this warp's 128 output columns.
#pragma unroll
        for (int kk = 0; kk < 2; ++kk) {
#pragma unroll
            for (int np = 0; np < 8; ++np) {
                unsigned int b0, b1, b2, b3;
                brain_ldsm_x4_t(vs_base + (((kk * 16u + v_row_lane) * ROW_HALVES) + hh * 128u + np * 16u + v_col_lane) * 2u, b0, b1, b2, b3);
                brain_mma_f16(o[2 * np], pf[kk], b0, b1);
                brain_mma_f16(o[2 * np + 1], pf[kk], b2, b3);
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
    for (int n = 0; n < 16; ++n) {
        const unsigned int d = hh * 128u + n * 8u + t4 * 2u;
        if (live_a) {
            *reinterpret_cast<float2*>(ctx + row_a * q_row + h * HD + d) = make_float2(o[n][0] * inv_a, o[n][1] * inv_a);
        }
        if (live_b) {
            *reinterpret_cast<float2*>(ctx + row_b * q_row + h * HD + d) = make_float2(o[n][2] * inv_b, o[n][3] * inv_b);
        }
    }
}
