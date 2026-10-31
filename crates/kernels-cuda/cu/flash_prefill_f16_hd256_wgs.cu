// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements long-context inference throughput for its
// clients. If your team needs expertise in keeping Hopper tensor cores fed from a
// compact paged KV cache then you can procure our services by sending an email to
// info@swedishembedded.com.
//
// Fused causal paged-attention PREFILL at head_dim 256 over an int8 or FP8 E4M3 cache, version 4 of flash_prefill_f16_hd256.cu:
// the warpgroup kernel of flash_prefill_f16_hd256_wg.cu with the KV widening taken out of its inner loop.
//
// Version 3 widens each key tile's 8-bit bytes to fp16 inside every block, so a tile is widened once per 128-row query block
// and head (about 512 times per kv head at 64K context) by a producer warpgroup that shares shared-memory bandwidth and issue
// slots with the two consumers (knowledge 383 measured the widening alone at 30 % of the kernel). Here a separate kernel,
// `brain_flash_prefill_wgs_widen`, widens the round's keys ONCE into a staging buffer that already holds the exact bytes the
// tensor cores read (128-byte swizzled 64-key fp16 tiles with the row scales beside them), and the attention kernel's
// producer is one thread issuing bulk copies (`cp.async.bulk`) of whole tiles. The numerics are version 3's: the widening is
// exact, the row scales are applied to the fp32 scores (K, times 1/sqrt(d) log2 e) and to the fp16 softmax weights (V),
// accumulators are fp32.
//
//   widen params : [bsz, n_heads, n_kv_heads, head_dim, group, block_size, max_bt, splits_max, multiprocessors, tiles_cap] u32
//   widen        : pool_k, pool_v (paged 8-bit planes), block_tables (the FIRST row's table is read: one sequence), seq_lens,
//                  k_scales, v_scales -> stage
//   stage        : per (kv head, 64-key tile) a record of WGS_STRIDE bytes: K tile (4 swizzle atom columns of 64 rows x 128 bytes),
//                  V tile, K row scales x 1/sqrt(d) log2 e (64 f32), V row scales (64 f32). Keys at or past the longest
//                  sequence length are zero with zero scales (so a masked key can never poison a product with 0 x NaN).
//   attention    : params (as above), q, stage, seq_lens, ctx, part -- one block is 384 threads, a producer warpgroup (one
//                  thread works) and two consumer warpgroups of 64 query rows; key splits and the combine kernel are version 3's.

#ifndef BRAIN_FLASH_HELPERS_ONLY
#error "flash_prefill_f16_hd256_wgs.cu is compiled after flash_prefill_f16_hd256.cu and flash_prefill_f16_hd256_wg.cu"
#endif

#define WGS_KEYS 128                  // keys a tile: S = Q K^T is 16 wgmma m64n128k16 (91 % of the tensor peak, against 66 % at n64)
#define WGS_ATOM_BYTES 16384          // one 128-row x 64-value fp16 swizzle atom column: 128 rows x 128 bytes
#define WGS_TILE_BYTES 65536          // a 128-row x 256-value fp16 tile: four atom columns
#define WGS_STRIDE 132096             // a staged record: K tile, V tile, 128 K scales, 128 V scales
#define WGS_OFF_STAGE_V 65536
#define WGS_OFF_STAGE_KS 131072
#define WGS_OFF_STAGE_VS 131584

// Dynamic shared memory layout (offsets from the 1024-aligned base).
#define WGS_OFF_Q 0                                      // two warpgroups x 32 KiB
#define WGS_OFF_K (WGS_OFF_Q + 2 * WG_TILE_BYTES)        // one stage x 64 KiB
#define WGS_OFF_V (WGS_OFF_K + WGS_TILE_BYTES)           // one stage x 64 KiB
#define WGS_OFF_KS (WGS_OFF_V + WGS_TILE_BYTES)          // K scales, 128 f32
#define WGS_OFF_VS (WGS_OFF_KS + 512)                    // V scales, 128 f32
#define WGS_OFF_BAR (WGS_OFF_VS + 512)                   // four 8-byte mbarriers
#define WGS_SMEM_END (WGS_OFF_BAR + 64)

// S (64 x 128 f32, 64 registers a thread) = or += Q (64 x 16 f16) . K (128 x 16 f16)^T, both operands from shared memory by descriptor.
__device__ __forceinline__ void wgs_mma_s(float (&d)[64], unsigned long long da, unsigned long long db, bool accumulate) {
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %66, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, %64, %65, p, 1, 1, 0, 0;\n"
        "}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63])
        : "l"(da), "l"(db), "r"(static_cast<int>(accumulate)));
}
// The V operand descriptor of a 128-key tile: N-major, the 64-value atom columns 16384 bytes apart, the 8-key groups 1024.
__device__ __forceinline__ unsigned long long wgs_desc_mn(unsigned int addr) {
    return static_cast<unsigned long long>((addr & 0x3FFFFu) >> 4) | (1024ull << 16) | (64ull << 32) | (1ull << 62);
}
__device__ __forceinline__ void wgs_mbar_expect_tx(unsigned int bar, unsigned int bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(bar), "r"(bytes) : "memory");
}
__device__ __forceinline__ void wgs_bulk_copy(unsigned int dst, const void* src, unsigned int bytes, unsigned int bar) {
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n" ::"r"(dst), "l"(src), "r"(bytes), "r"(bar) : "memory");
}
__device__ __forceinline__ void wgs_fence_barrier_init() { asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory"); }

// Widens the keys of one (128-key tile, kv head) into its staged record: 256 threads, a pair per key row.
extern "C" __global__ void __launch_bounds__(256) brain_flash_prefill_wgs_widen(
    const unsigned int* params, const KvT* pool_k, const KvT* pool_v, const unsigned int* block_tables, const unsigned int* seq_lens,
    const float* k_scales, const float* v_scales, unsigned char* stage) {
    const unsigned int bsz = params[0];
    const unsigned int n_kv_heads = params[2];
    const unsigned int block_size = params[5];
    const unsigned int tiles_cap = params[9];
    const unsigned int blk = blockIdx.y * gridDim.x + blockIdx.x;   // tile fastest
    if (bsz == 0u || tiles_cap == 0u) { return; }
    const unsigned int kt = blk % tiles_cap;
    const unsigned int hkv = blk / tiles_cap;
    if (hkv >= n_kv_heads) { return; }
    const unsigned int max_t = seq_lens[bsz - 1u];
    if (kt * WGS_KEYS >= max_t) { return; }   // past the longest key range: nothing reads this record

    const unsigned int row = threadIdx.x >> 1;
    const unsigned int half = threadIdx.x & 1u;
    const unsigned int j = kt * WGS_KEYS + row;
    const bool live = j < max_t;
    const unsigned long long kv_row = static_cast<unsigned long long>(n_kv_heads) * HD;
    unsigned char* const rec = stage + (static_cast<unsigned long long>(hkv) * tiles_cap + kt) * WGS_STRIDE;

    unsigned long long prow = 0;
    if (live) { prow = static_cast<unsigned long long>(block_tables[j / block_size]) * block_size + j % block_size; }
    const KvT* const ksrc = pool_k + prow * kv_row + hkv * HD;
    const KvT* const vsrc = pool_v + prow * kv_row + hkv * HD;
#pragma unroll
    for (int plane = 0; plane < 2; ++plane) {
        const KvT* const src = plane == 0 ? ksrc : vsrc;
        unsigned char* const tile = rec + plane * WGS_OFF_STAGE_V;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const unsigned int c = 2u * i + half;
            uint4 w = live ? *reinterpret_cast<const uint4*>(src + c * 16u) : make_uint4(0u, 0u, 0u, 0u);
            uint4 o0, o1;
            wg_widen4(w.x, o0.x, o0.y);
            wg_widen4(w.y, o0.z, o0.w);
            wg_widen4(w.z, o1.x, o1.y);
            wg_widen4(w.w, o1.z, o1.w);
            // 16 codes -> d-chunks 2c and 2c + 1 of the row (8 values each), in the atom column c / 4.
            unsigned char* const dst = tile + (c >> 2) * WGS_ATOM_BYTES + row * 128u;
            const unsigned int s0 = ((2u * c) & 7u) ^ (row & 7u);
            const unsigned int s1 = ((2u * c + 1u) & 7u) ^ (row & 7u);
            *reinterpret_cast<uint4*>(dst + s0 * 16u) = o0;
            *reinterpret_cast<uint4*>(dst + s1 * 16u) = o1;
        }
    }
    if (half == 0u) {
        const unsigned long long srow = prow * n_kv_heads + hkv;
        reinterpret_cast<float*>(rec + WGS_OFF_STAGE_KS)[row] = live ? k_scales[srow] * SM_SCALE_LOG2 : 0.f;
        reinterpret_cast<float*>(rec + WGS_OFF_STAGE_VS)[row] = live ? v_scales[srow] : 0.f;
    }
}

extern "C" __global__ void __launch_bounds__(WG_THREADS, 1) brain_flash_prefill_f16_hd256_wgs(
    const unsigned int* params, const float* q, const unsigned char* stage, const unsigned int* seq_lens, float* ctx, float* part) {
    extern __shared__ unsigned char brain_wg_smem_raw[];
    const unsigned int bsz = params[0];
    const unsigned int n_heads = params[1];
    const unsigned int group = params[4];
    const unsigned int smax = params[7];
    const unsigned int nsm = params[8];
    const unsigned int tiles_cap = params[9];
    const unsigned int nrows = bsz;

    const unsigned int ntq = (nrows + WG_ROWS - 1u) / WG_ROWS;
    if (ntq == 0u) { return; }
    const unsigned int base_blocks = ntq * n_heads;
    const unsigned int blk = blockIdx.y * gridDim.x + blockIdx.x;
    const unsigned int qt = ntq - 1u - blk % ntq;   // the longest key ranges first
    const unsigned int h = (blk / ntq) % n_heads;
    const unsigned int split = blk / base_blocks;
    if (split >= smax) { return; }   // block-uniform: no barrier is skipped per-thread

    const unsigned int i0 = qt * WG_ROWS;
    const unsigned int max_t = seq_lens[min(i0 + WG_ROWS - 1u, nrows - 1u)];
    // The number of splits is decided in the units of version 3 (64-key tiles) so the combine kernel, which knows nothing of this
    // kernel's tile, agrees; the tiles of a split are then 128 keys. A split can come out empty: it writes an empty partial.
    const unsigned int n_split = wg_eff_splits((max_t + WG_KEYS - 1u) / WG_KEYS, base_blocks, smax, nsm);
    if (split >= n_split) { return; }
    const unsigned int tiles = (max_t + WGS_KEYS - 1u) / WGS_KEYS;
    const unsigned int per = (tiles + n_split - 1u) / n_split;
    const unsigned int tile_begin = min(split * per, tiles);
    const unsigned int tile_end = min(tile_begin + per, tiles);

    const unsigned int hkv = h / (group == 0u ? 1u : group);
    const unsigned long long q_row = static_cast<unsigned long long>(n_heads) * HD;

    const unsigned int raw_addr = brain_smem_addr(brain_wg_smem_raw);
    const unsigned int base = (raw_addr + 1023u) & ~1023u;
    unsigned char* const gen = brain_wg_smem_raw + (base - raw_addr);
    const unsigned int bar_full_k = base + WGS_OFF_BAR;
    const unsigned int bar_empty_k = base + WGS_OFF_BAR + 8u;
    const unsigned int bar_full_v = base + WGS_OFF_BAR + 16u;
    const unsigned int bar_empty_v = base + WGS_OFF_BAR + 24u;

    const unsigned int tid = threadIdx.x;
    if (tid == 0u) {
        wg_mbar_init(bar_full_k, 1u);   // the producer's expect_tx arrival; the bytes land through the copy
        wg_mbar_init(bar_empty_k, 8u);
        wg_mbar_init(bar_full_v, 1u);
        wg_mbar_init(bar_empty_v, 8u);
        wgs_fence_barrier_init();
    }
    __syncthreads();

    if (tid < 128u) {
        // ================= producer warpgroup: one thread issues the bulk copies =================
        wg_regs_dec();
        if (tid != 0u || tile_begin == tile_end) { return; }
        const unsigned char* const rec0 = stage + (static_cast<unsigned long long>(hkv) * tiles_cap) * WGS_STRIDE;
        for (unsigned int kt = tile_begin; kt < tile_end; ++kt) {
            const unsigned int j = kt - tile_begin;
            const unsigned char* const rec = rec0 + static_cast<unsigned long long>(kt) * WGS_STRIDE;
            wg_mbar_wait(bar_empty_k, (j & 1u) ^ 1u);
            wgs_mbar_expect_tx(bar_full_k, WGS_TILE_BYTES + 512u);
            wgs_bulk_copy(base + WGS_OFF_K, rec, WGS_TILE_BYTES, bar_full_k);
            wgs_bulk_copy(base + WGS_OFF_KS, rec + WGS_OFF_STAGE_KS, 512u, bar_full_k);
            wg_mbar_wait(bar_empty_v, (j & 1u) ^ 1u);
            wgs_mbar_expect_tx(bar_full_v, WGS_TILE_BYTES + 512u);
            wgs_bulk_copy(base + WGS_OFF_V, rec + WGS_OFF_STAGE_V, WGS_TILE_BYTES, bar_full_v);
            wgs_bulk_copy(base + WGS_OFF_VS, rec + WGS_OFF_STAGE_VS, 512u, bar_full_v);
        }
        return;
    }

    // ================= consumer warpgroups =================
    wg_regs_inc();
    const unsigned int cw = __shfl_sync(0xffffffffu, tid >> 7, 0) - 1u;   // which consumer warpgroup (warp-uniform to the compiler): rows 64 cw .. 64 cw + 63
    const unsigned int ct = tid & 127u;
    const unsigned int warp = ct >> 5;
    const unsigned int lane = tid & 31u;
    const unsigned int g = lane >> 2;
    const unsigned int t4 = lane & 3u;

    const unsigned int wg_row0 = i0 + cw * 64u;
    const unsigned int row_a = wg_row0 + warp * 16u + g;
    const unsigned int row_b = row_a + 8u;
    const bool live_a = row_a < nrows, live_b = row_b < nrows;
    const unsigned int len_a = live_a ? seq_lens[row_a] : 0u;
    const unsigned int len_b = live_b ? seq_lens[row_b] : 0u;
    const unsigned int wg_min_len = wg_row0 < nrows ? seq_lens[wg_row0] : 0u;

    // ---- Q (this warpgroup's 64 rows) to shared memory as fp16, swizzled: 64 rows x 32 chunks of 8 values.
    const unsigned int q_tile = base + WGS_OFF_Q + cw * WG_TILE_BYTES;
#pragma unroll 4
    for (int c = 0; c < 16; ++c) {
        const unsigned int e = ct + 128u * c;
        const unsigned int r = e >> 5, cc = e & 31u;
        uint4 v = make_uint4(0u, 0u, 0u, 0u);
        if (wg_row0 + r < nrows) {
            const float* src = q + static_cast<unsigned long long>(wg_row0 + r) * q_row + h * HD + cc * 8u;
            const float4 lo = *reinterpret_cast<const float4*>(src);
            const float4 hi = *reinterpret_cast<const float4*>(src + 4);
            v.x = brain_pack_f16x2(lo.x, lo.y);
            v.y = brain_pack_f16x2(lo.z, lo.w);
            v.z = brain_pack_f16x2(hi.x, hi.y);
            v.w = brain_pack_f16x2(hi.z, hi.w);
        }
        const unsigned int dst = q_tile + (cc >> 3) * WG_SUB_BYTES + r * 128u + (((cc & 7u) ^ (r & 7u)) << 4);
        asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};\n" ::"r"(dst), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w));
    }
    wg_fence_proxy();
    wg_named_barrier(2u + cw);

    float o[128];
#pragma unroll
    for (int i = 0; i < 128; ++i) { o[i] = 0.f; }
    float m_a = NEG_BIG, m_b = NEG_BIG, l_a = 0.f, l_b = 0.f;
    const float* const ks2 = reinterpret_cast<const float*>(gen + WGS_OFF_KS);
    const float* const vs1 = reinterpret_cast<const float*>(gen + WGS_OFF_VS);

    // Every warp runs every tile, a tile past a warpgroup's last row included (the mask weighs all of it 0): a branch around the
    // wgmma that the compiler cannot prove warp-uniform makes it wait for each one before issuing the next.
    for (unsigned int kt = tile_begin; kt < tile_end; ++kt) {
        const unsigned int j = kt - tile_begin;
        const unsigned int parity = j & 1u;
        wg_mbar_wait(bar_full_k, parity);

        float s[64];
        {
            const unsigned int k_tile = base + WGS_OFF_K;
            wg_fence();
#pragma unroll
            for (int k = 0; k < 16; ++k) {
                wgs_mma_s(s, wg_desc_k(q_tile + (k >> 2) * WG_SUB_BYTES + (k & 3) * 32u), wg_desc_k(k_tile + (k >> 2) * WGS_ATOM_BYTES + (k & 3) * 32u), k > 0);
            }
            wg_commit();
        }
        {
            wg_wait0();

            // ---- row scale of K (already times 1/sqrt(d) log2 e) and, on the few tiles that cross a row's own position, the causal mask.
            if ((kt + 1u) * WGS_KEYS <= wg_min_len) {
#pragma unroll
                for (int n = 0; n < 16; ++n) {
                    const float2 kx = *reinterpret_cast<const float2*>(ks2 + n * 8u + t4 * 2u);
                    s[4 * n + 0] *= kx.x;
                    s[4 * n + 1] *= kx.y;
                    s[4 * n + 2] *= kx.x;
                    s[4 * n + 3] *= kx.y;
                }
            } else {
#pragma unroll
                for (int n = 0; n < 16; ++n) {
                    const float2 kx = *reinterpret_cast<const float2*>(ks2 + n * 8u + t4 * 2u);
                    const unsigned int j0 = kt * WGS_KEYS + n * 8u + t4 * 2u;
                    s[4 * n + 0] = j0 < len_a ? s[4 * n + 0] * kx.x : NEG_BIG;
                    s[4 * n + 1] = j0 + 1u < len_a ? s[4 * n + 1] * kx.y : NEG_BIG;
                    s[4 * n + 2] = j0 < len_b ? s[4 * n + 2] * kx.x : NEG_BIG;
                    s[4 * n + 3] = j0 + 1u < len_b ? s[4 * n + 3] * kx.y : NEG_BIG;
                }
            }
        }
        __syncwarp();
        if (lane == 0u) { wg_mbar_arrive(bar_empty_k); }

        unsigned int pf[8][4];
        {
            float mx_a = NEG_BIG, mx_b = NEG_BIG;
#pragma unroll
            for (int n = 0; n < 16; ++n) {
                mx_a = fmaxf(mx_a, fmaxf(s[4 * n + 0], s[4 * n + 1]));
                mx_b = fmaxf(mx_b, fmaxf(s[4 * n + 2], s[4 * n + 3]));
            }
            mx_a = fmaxf(mx_a, __shfl_xor_sync(0xffffffffu, mx_a, 1));
            mx_a = fmaxf(mx_a, __shfl_xor_sync(0xffffffffu, mx_a, 2));
            mx_b = fmaxf(mx_b, __shfl_xor_sync(0xffffffffu, mx_b, 1));
            mx_b = fmaxf(mx_b, __shfl_xor_sync(0xffffffffu, mx_b, 2));
            const float mn_a = fmaxf(m_a, mx_a), mn_b = fmaxf(m_b, mx_b);
            const float al_a = wg_ex2(m_a - mn_a), al_b = wg_ex2(m_b - mn_b);
            m_a = mn_a;
            m_b = mn_b;
            // A row with no live key yet has mn = NEG_BIG; its masked scores must weigh 0, not exp2(0).
            const float ref_a = mn_a > NEG_BIG ? mn_a : 0.f, ref_b = mn_b > NEG_BIG ? mn_b : 0.f;
            float sum_a = 0.f, sum_b = 0.f;
#pragma unroll
            for (int n = 0; n < 16; ++n) {
                s[4 * n + 0] = wg_ex2(s[4 * n + 0] - ref_a);
                s[4 * n + 1] = wg_ex2(s[4 * n + 1] - ref_a);
                s[4 * n + 2] = wg_ex2(s[4 * n + 2] - ref_b);
                s[4 * n + 3] = wg_ex2(s[4 * n + 3] - ref_b);
                sum_a += s[4 * n + 0] + s[4 * n + 1];
                sum_b += s[4 * n + 2] + s[4 * n + 3];
            }
            l_a = l_a * al_a + sum_a;
            l_b = l_b * al_b + sum_b;
            // O only needs rescaling when some row's running maximum grew: nearly never past the first tiles. The first tile's
            // O is still zero, and the vote keeps the warp together.
            if (j > 0u && !__all_sync(0xffffffffu, al_a == 1.0f && al_b == 1.0f)) {
#pragma unroll
                for (int n = 0; n < 32; ++n) {
                    o[4 * n + 0] *= al_a;
                    o[4 * n + 1] *= al_a;
                    o[4 * n + 2] *= al_b;
                    o[4 * n + 3] *= al_b;
                }
            }
        }

        wg_mbar_wait(bar_full_v, parity);   // the V row scales and the V tile of this key tile
        {
            // P as fp16 A fragments, straight from the score accumulators, times the V row scale.
#pragma unroll
            for (int kk = 0; kk < 8; ++kk) {
                const float2 v0 = *reinterpret_cast<const float2*>(vs1 + (2 * kk) * 8u + t4 * 2u);
                const float2 v1 = *reinterpret_cast<const float2*>(vs1 + (2 * kk + 1) * 8u + t4 * 2u);
                pf[kk][0] = brain_pack_f16x2(s[8 * kk + 0] * v0.x, s[8 * kk + 1] * v0.y);
                pf[kk][1] = brain_pack_f16x2(s[8 * kk + 2] * v0.x, s[8 * kk + 3] * v0.y);
                pf[kk][2] = brain_pack_f16x2(s[8 * kk + 4] * v1.x, s[8 * kk + 5] * v1.y);
                pf[kk][3] = brain_pack_f16x2(s[8 * kk + 6] * v1.x, s[8 * kk + 7] * v1.y);
            }
        }
        {
            const unsigned int v_tile = base + WGS_OFF_V;
            wg_fence();
#pragma unroll
            for (int kk = 0; kk < 8; ++kk) { brain_wg_mma_pv(o, pf[kk], wgs_desc_mn(v_tile + kk * 2048u)); }
            wg_commit();
        }
        wg_wait0();
        __syncwarp();
        if (lane == 0u) { wg_mbar_arrive(bar_empty_v); }
    }

    // ---- epilogue. l is a per-thread partial of its quad's row sum.
    l_a += __shfl_xor_sync(0xffffffffu, l_a, 1);
    l_a += __shfl_xor_sync(0xffffffffu, l_a, 2);
    l_b += __shfl_xor_sync(0xffffffffu, l_b, 1);
    l_b += __shfl_xor_sync(0xffffffffu, l_b, 2);
    if (n_split == 1u) {
        const float inv_a = l_a > 0.f ? 1.0f / l_a : 0.f;
        const float inv_b = l_b > 0.f ? 1.0f / l_b : 0.f;
#pragma unroll
        for (int n = 0; n < 32; ++n) {
            const unsigned int d = n * 8u + t4 * 2u;
            if (live_a) { *reinterpret_cast<float2*>(ctx + static_cast<unsigned long long>(row_a) * q_row + h * HD + d) = make_float2(o[4 * n] * inv_a, o[4 * n + 1] * inv_a); }
            if (live_b) { *reinterpret_cast<float2*>(ctx + static_cast<unsigned long long>(row_b) * q_row + h * HD + d) = make_float2(o[4 * n + 2] * inv_b, o[4 * n + 3] * inv_b); }
        }
    } else {
        const unsigned long long split_rows = static_cast<unsigned long long>(split) * bsz;
        float* const part_ml = part + static_cast<unsigned long long>(smax) * bsz * q_row;
#pragma unroll
        for (int n = 0; n < 32; ++n) {
            const unsigned int d = n * 8u + t4 * 2u;
            if (live_a) { *reinterpret_cast<float2*>(part + ((split_rows + row_a) * n_heads + h) * HD + d) = make_float2(o[4 * n], o[4 * n + 1]); }
            if (live_b) { *reinterpret_cast<float2*>(part + ((split_rows + row_b) * n_heads + h) * HD + d) = make_float2(o[4 * n + 2], o[4 * n + 3]); }
        }
        if (t4 == 0u) {
            if (live_a) { *reinterpret_cast<float2*>(part_ml + ((split_rows + row_a) * n_heads + h) * 2u) = make_float2(m_a, l_a); }
            if (live_b) { *reinterpret_cast<float2*>(part_ml + ((split_rows + row_b) * n_heads + h) * 2u) = make_float2(m_b, l_b); }
        }
    }
}
