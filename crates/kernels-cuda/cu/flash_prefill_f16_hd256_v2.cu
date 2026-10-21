// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements long-context inference throughput for its
// clients. If your team needs expertise in putting causal attention on tensor
// cores without giving up the paged KV layout its serving stack is built on,
// then you can procure our services by sending an email to
// info@swedishembedded.com.
//
// Fused causal paged-attention PREFILL at head_dim 256, version 2 of flash_prefill_f16_hd256.cu: the same
// argument contract, launch geometry (`n_heads * ceil(bsz / 64)` blocks, query tile fastest), numerics
// (Q, K, V, P rounded to fp16, fp32 accumulators and softmax) and addressing, for the bf16 (BRAIN_KV_TIER 1)
// and int8 (BRAIN_KV_TIER 2) and fp8 E4M3 (BRAIN_KV_TIER 3) caches. It is compiled after flash_prefill_f16_hd256.cu with
// BRAIN_FLASH_HELPERS_ONLY defined, which supplies the helpers, `KvT` and the tier macros used here.
//
// What differs from version 1, and why:
//   * a block is 4 warps of 128 threads and every warp owns 16 query rows over the WHOLE head: no pair of
//     warps recomputes the same scores (version 1 spends 96 MMAs per warp per 32-key tile for 64 useful;
//     here a warp issues 128 per 32 keys for its own 16 rows, 1024 per block per 64 keys against version 1's 1536);
//   * the output accumulator is 16 x 256 fp32 = 128 registers per thread, so Q cannot also live in registers:
//     it stays in shared memory as fp16 and its A fragments are read with ldmatrix each tile;
//   * the block is sized for TWO resident blocks per SM (128 threads, up to 255 registers, 99 KiB of dynamic
//     shared memory), so one block's global-load and barrier latency is covered by the other's matrix work,
//     where version 1 runs one block of eight warps per SM;
//   * a compact cache is staged as raw bytes and widened only when it is written to shared memory (version 1
//     widens at load time and holds fp32), which keeps the in-flight tile at 16-32 registers.
//
// The pipeline per 64-key tile (version 1 uses 32: half the barriers and Q fragment reads per key) is version 1's: write K, start V's global loads, scores, softmax, write V,
// start the next K's loads, then P V.

#ifndef BRAIN_FLASH_HELPERS_ONLY
#error "flash_prefill_f16_hd256_v2.cu is compiled after flash_prefill_f16_hd256.cu with BRAIN_FLASH_HELPERS_ONLY"
#endif
#if BRAIN_HD != 256 || BRAIN_DENSE || BRAIN_KV_TIER == 0
#error "version 2 serves head_dim 256 over a bf16 or int8 paged cache"
#endif

#define V2_THREADS 128
#define V2_ROWS 64                              // query rows per block, 16 per warp
#ifndef V2_KEYS
#define V2_KEYS 64                              // keys per tile; gpu-core's FLASH_PREFILL_V2_DYNAMIC_SHARED_BYTES must follow it
#endif
#define V2_SN (V2_KEYS / 8)                     // n8 score tiles per tile
#define V2_KK (V2_KEYS / 16)                    // k16 steps of P V per tile
#define V2_CHUNKS (HD / 8)                      // 8-value chunks per key row
#define V2_ROWS_PER_PASS (V2_THREADS / V2_CHUNKS)
#define V2_STAGE (V2_KEYS / V2_ROWS_PER_PASS)   // chunks a thread stages per tile and plane
#define V2_NTILES (HD / 8)                      // n8 output tiles per warp

extern "C" __global__ void __launch_bounds__(V2_THREADS, 2) brain_flash_prefill_f16_hd256_v2(FLASH_ARGS) {
    extern __shared__ __align__(16) unsigned short smem[];
    unsigned short* Qs = smem;
    unsigned short* Ks = Qs + V2_ROWS * ROW_HALVES;
    unsigned short* Vs = Ks + V2_KEYS * ROW_HALVES;

    const unsigned int bsz = params[0];
    const unsigned int n_heads = params[1];
    const unsigned int n_kv_heads = params[2];
    const unsigned int group = params[4];
    const unsigned int block_size = params[5];
    const unsigned int max_bt = params[6];
    const unsigned int nrows = bsz;

    const unsigned int blk = blockIdx.y * gridDim.x + blockIdx.x;
    const unsigned int ntiles_q = (nrows + V2_ROWS - 1u) / V2_ROWS;
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
    const unsigned int g = lane >> 2;        // MMA groupID
    const unsigned int t4 = lane & 3u;       // MMA threadID_in_group

    const unsigned int i0 = qt * V2_ROWS;
    const unsigned int row_a = i0 + warp * 16u + g;      // fragment rows g and g + 8
    const unsigned int row_b = row_a + 8u;
    const bool live_a = row_a < nrows, live_b = row_b < nrows;
    const unsigned int len_a = live_a ? seq_lens[row_a] : 0u;
    const unsigned int len_b = live_b ? seq_lens[row_b] : 0u;
    const unsigned int max_t = seq_lens[min(i0 + V2_ROWS - 1u, bsz - 1u)];
    const unsigned int bt_row = i0;
    const unsigned int ntiles_k = (max_t + V2_KEYS - 1u) / V2_KEYS;

    // ---- Q to shared memory as fp16: 64 rows x 32 chunks of 8 values, 16 chunks per thread.
#pragma unroll
    for (int c = 0; c < (V2_ROWS * V2_CHUNKS) / V2_THREADS; ++c) {
        const unsigned int e = tid + V2_THREADS * c;
        const unsigned int r = e / V2_CHUNKS, cc = e % V2_CHUNKS;
        uint4 v = make_uint4(0u, 0u, 0u, 0u);
        if (i0 + r < nrows) {
            const float* src = q + (unsigned long long)(i0 + r) * q_row + h * HD + cc * 8u;
            const float4 lo = *reinterpret_cast<const float4*>(src);
            const float4 hi = *reinterpret_cast<const float4*>(src + 4);
            v.x = brain_pack_f16x2(lo.x, lo.y);
            v.y = brain_pack_f16x2(lo.z, lo.w);
            v.z = brain_pack_f16x2(hi.x, hi.y);
            v.w = brain_pack_f16x2(hi.z, hi.w);
        }
        *reinterpret_cast<uint4*>(Qs + r * ROW_HALVES + cc * 8u) = v;
    }

    float o[V2_NTILES][4];
#pragma unroll
    for (int n = 0; n < V2_NTILES; ++n) { o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.f; }
    float m_a = NEG_BIG, m_b = NEG_BIG, l_a = 0.f, l_b = 0.f;

    // Staging: chunk (row0 + V2_ROWS_PER_PASS c, cc) of a tile is thread tid's c-th: a warp reads one key row
    // (256 values) and writes 512 contiguous shared bytes. The raw bytes stay in registers until commit.
    const unsigned int cc = tid % V2_CHUNKS;
    const unsigned int row0 = tid / V2_CHUNKS;
#if BRAIN_KV_TIER == 1
    uint4 raw[V2_STAGE];
#else
    uint2 raw[V2_STAGE];
    float rsc[V2_STAGE];
#endif
    auto issue = [&](const KvT* pool, const float* scales, unsigned int kt) {
        (void)scales;
#pragma unroll
        for (int c = 0; c < V2_STAGE; ++c) {
            const unsigned int j = kt * V2_KEYS + row0 + V2_ROWS_PER_PASS * c;
#if BRAIN_KV_TIER == 1
            uint4 w = make_uint4(0u, 0u, 0u, 0u);
#else
            uint2 w = make_uint2(0u, 0u);
            float sc = 0.f;
#endif
            if (j < max_t) {
                const unsigned int physical = block_tables[(unsigned long long)bt_row * max_bt + j / block_size];
                const unsigned long long row = (unsigned long long)physical * block_size + (j % block_size);
                const KvT* src = pool + row * kv_row + hkv * HD + cc * 8u;
#if BRAIN_KV_TIER == 1
                w = *reinterpret_cast<const uint4*>(src);
#else
                w = *reinterpret_cast<const uint2*>(src);
                sc = scales[row * n_kv_heads + hkv];
#endif
            }
            raw[c] = w;
#if BRAIN_KV_TIER != 1
            rsc[c] = sc;
#endif
        }
    };
    auto commit = [&](unsigned short* dst) {
#pragma unroll
        for (int c = 0; c < V2_STAGE; ++c) {
            const unsigned int kr = row0 + V2_ROWS_PER_PASS * c;
            uint4 v;
#if BRAIN_KV_TIER == 1
            // Eight bf16: a bf16 is the top half of an f32.
            const uint4 w = raw[c];
            v.x = brain_pack_f16x2(__uint_as_float(w.x << 16), __uint_as_float(w.x & 0xffff0000u));
            v.y = brain_pack_f16x2(__uint_as_float(w.y << 16), __uint_as_float(w.y & 0xffff0000u));
            v.z = brain_pack_f16x2(__uint_as_float(w.z << 16), __uint_as_float(w.z & 0xffff0000u));
            v.w = brain_pack_f16x2(__uint_as_float(w.w << 16), __uint_as_float(w.w & 0xffff0000u));
#else
            const uint2 w = raw[c];
            const float sc = rsc[c];
            auto b = [&](unsigned int word, int h) {
                const float2 f = brain_kv8_pair(word, h);
                return brain_pack_f16x2(f.x * sc, f.y * sc);
            };
            v.x = b(w.x, 0);
            v.y = b(w.x, 1);
            v.z = b(w.y, 0);
            v.w = b(w.y, 1);
#endif
            *reinterpret_cast<uint4*>(dst + kr * ROW_HALVES + cc * 8u) = v;
        }
    };

    if (ntiles_k > 0u) { issue(pool_k, SCALES_K, 0u); }

    // ldmatrix lane geometry (see the MMA fragment layouts in the PTX ISA).
    //  Q (A operand, rows of the warp's 16): matrices (rows 0-7, k 0-7), (rows 8-15, k 0-7),
    //  (rows 0-7, k 8-15), (rows 8-15, k 8-15) - the register order of an m16n8k16 A fragment.
    //  K (B operand of S, non-transposed) and V (B operand of O, transposed): as version 1.
    const unsigned int q_row_lane = (lane & 7u) + (((lane >> 3) & 1u) << 3);
    const unsigned int q_col_lane = (lane >> 4) << 3;
    const unsigned int k_row_lane = (lane & 7u) + ((lane >> 4) << 3);
    const unsigned int k_col_lane = ((lane >> 3) & 1u) << 3;
    const unsigned int v_row_lane = (lane & 7u) + (((lane >> 3) & 1u) << 3);
    const unsigned int v_col_lane = (lane >> 4) << 3;
    const unsigned int qs_base = brain_smem_addr(Qs) + ((warp * 16u + q_row_lane) * ROW_HALVES + q_col_lane) * 2u;
    const unsigned int ks_base = brain_smem_addr(Ks);
    const unsigned int vs_base = brain_smem_addr(Vs);
    __syncthreads();                       // Q visible

    for (unsigned int kt = 0; kt < ntiles_k; ++kt) {
        __syncthreads();                   // every warp is done with the previous tile's K and V
        commit(Ks);
        issue(pool_v, SCALES_V, kt);       // in flight while the scores are computed
        __syncthreads();                   // K visible

        // ---- S = Q K^T for 16 rows x V2_KEYS keys.
        float sacc[V2_SN][4];
#pragma unroll
        for (int n = 0; n < V2_SN; ++n) { sacc[n][0] = sacc[n][1] = sacc[n][2] = sacc[n][3] = 0.f; }
#pragma unroll
        for (int s = 0; s < HD / 16; ++s) {
            unsigned int qa[4];
            brain_ldsm_x4(qs_base + s * 32u, qa[0], qa[1], qa[2], qa[3]);
#pragma unroll
            for (int np = 0; np < V2_SN / 2; ++np) {
                unsigned int b0, b1, b2, b3;
                brain_ldsm_x4(ks_base + ((np * 16u + k_row_lane) * ROW_HALVES + s * 16u + k_col_lane) * 2u, b0, b1, b2, b3);
                brain_mma_f16(sacc[2 * np], qa, b0, b1);
                brain_mma_f16(sacc[2 * np + 1], qa, b2, b3);
            }
        }

        // ---- online softmax (exp2 domain) over this tile's keys.
        float mx_a = NEG_BIG, mx_b = NEG_BIG;
#pragma unroll
        for (int n = 0; n < V2_SN; ++n) {
            const unsigned int j0 = kt * V2_KEYS + n * 8u + t4 * 2u;
            sacc[n][0] = (j0 < len_a) ? sacc[n][0] * SM_SCALE_LOG2 : NEG_BIG;
            sacc[n][1] = (j0 + 1u < len_a) ? sacc[n][1] * SM_SCALE_LOG2 : NEG_BIG;
            sacc[n][2] = (j0 < len_b) ? sacc[n][2] * SM_SCALE_LOG2 : NEG_BIG;
            sacc[n][3] = (j0 + 1u < len_b) ? sacc[n][3] * SM_SCALE_LOG2 : NEG_BIG;
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
        for (int n = 0; n < V2_SN; ++n) {
            // A masked score is NEG_BIG; when the whole row is masked mn is NEG_BIG too and the
            // difference is 0, so the weight is forced to 0.
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
        for (int n = 0; n < V2_NTILES; ++n) {
            o[n][0] *= al_a;
            o[n][1] *= al_a;
            o[n][2] *= al_b;
            o[n][3] *= al_b;
        }

        // P as fp16 A-fragments, straight from the score accumulators.
        unsigned int pf[V2_KK][4];
#pragma unroll
        for (int kk = 0; kk < V2_KK; ++kk) {
            pf[kk][0] = brain_pack_f16x2(sacc[2 * kk][0], sacc[2 * kk][1]);
            pf[kk][1] = brain_pack_f16x2(sacc[2 * kk][2], sacc[2 * kk][3]);
            pf[kk][2] = brain_pack_f16x2(sacc[2 * kk + 1][0], sacc[2 * kk + 1][1]);
            pf[kk][3] = brain_pack_f16x2(sacc[2 * kk + 1][2], sacc[2 * kk + 1][3]);
        }

        commit(Vs);                        // V loads have had the whole score phase
        if (kt + 1u < ntiles_k) { issue(pool_k, SCALES_K, kt + 1u); }   // next K in flight under the PV MMAs
        __syncthreads();                   // V visible

        // ---- O += P V over all 256 output columns.
#pragma unroll
        for (int kk = 0; kk < V2_KK; ++kk) {
#pragma unroll
            for (int np = 0; np < V2_NTILES / 2; ++np) {
                unsigned int b0, b1, b2, b3;
                brain_ldsm_x4_t(vs_base + (((kk * 16u + v_row_lane) * ROW_HALVES) + np * 16u + v_col_lane) * 2u, b0, b1, b2, b3);
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
    for (int n = 0; n < V2_NTILES; ++n) {
        const unsigned int d = n * 8u + t4 * 2u;
        if (live_a) {
            *reinterpret_cast<float2*>(ctx + (unsigned long long)row_a * q_row + h * HD + d) = make_float2(o[n][0] * inv_a, o[n][1] * inv_a);
        }
        if (live_b) {
            *reinterpret_cast<float2*>(ctx + (unsigned long long)row_b * q_row + h * HD + d) = make_float2(o[n][2] * inv_b, o[n][3] * inv_b);
        }
    }
}
