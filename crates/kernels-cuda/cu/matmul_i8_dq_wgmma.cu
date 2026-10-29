// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for prompt-processing throughput
// on quantised language models for its clients. If your team needs expertise
// in feeding Hopper warpgroup tensor cores from int8 weights without a
// per-group scale fold, then you can procure our services by sending an
// email to info@swedishembedded.com.
//
// Prefill-sized matmul of INT8 weights (one f32 scale per 32 codes) against BF16 activations on Hopper WARPGROUP tensor
// cores (`wgmma.mma_async` m64n128k16, bf16 x bf16 -> f32), the weight codes dequantised as they pass through shared
// memory:
//
//   params : u32 [m, kg, n]   kg = K/4 packed words per weight row, K a multiple of 64
//   x_bf   : [M, K] bf16      activations rounded to bf16 (two per u32; 16-byte aligned)
//   w_q    : [N, kg] u32      weights, 4 signed int8 codes per word (16-byte aligned)
//   sw     : [N, K/32] f32    per-32-element-group weight scale
//   out    : [M, N]  f32      out[m, n] = sum_k x[m,k] * bf16(w_q[n,k] * sw[n, k/32]), accumulated in f32 over ALL of K
//
// Why (knowledge 372): the integer warpgroup kernel is bound by the group-scale fold (the tensor cores idle three
// quarters of the time). Here the scale is applied to the weight code once, as the tile is prepared, and the
// accumulators run straight across K: the fold is gone and the tensor cores can be fed at their own rate.
//
// Data movement and the overlap of dequantisation with the tensor cores
// ---------------------------------------------------------------------
// A ring of BRAIN_DQW_STAGES `cp.async` stages holds, per 64-deep k-tile, the bf16 activation tile (128 rows, one
// 128-byte row per k-tile, 16-byte chunks XOR-swizzled by row mod 8 for the 128-byte swizzle the descriptors name), the
// raw int8 weight tile (128 rows x 64 bytes) and its 256 scales. The weights are dequantised by all 256 threads into one
// of two bf16 tiles in the same swizzled layout (a thread takes 16 consecutive codes of a row at a time, so the work needs
// no k permutation), and the two warpgroups then run `wgmma` m64n128k16 with the activations as A (64 rows each) and the
// dequantised weights as B, four per k-tile, asynchronously. In each iteration the wgmma of tile kt is issued FIRST and
// tile kt + 1's weights are dequantised while the tensor cores run it, into a third bf16 tile (see BRAIN_DQW_DTILES); the
// wait for the wgmma of tile kt - 1 sits between them so the ring slot the next prefetch overwrites is free. The accumulators are the 64 f32 registers of a thread's m64n128 fragment: no fold, no seed.
//
// Dequantisation of a byte pair as bf16, exactly: (u & 0x7F) | 0x4300 is the bf16 128 + (b & 0x7F), (u & 0x80) | 0x4300 is
// 128 or 256 (bit 7 is the exponent's low bit), their difference is the signed code, and one bf16 multiply by the group
// scale rounded to bf16 gives the weight (the same arithmetic as `matmul_i8_dq_stream.cu`).
//
// Row tails: weight rows past n and rows of x past m are zero-filled and dropped at the store.
//
// FP8 weights (BRAIN_DQW_FP8): compiled with it defined, the weights are E4M3 bytes with one f32 scale per 128 x 128 block
// (`sw` is [ceil(N/128), K/128], K a multiple of 128) and the tile's 128 weight rows lie in one band, so a k-tile's scale is
// ONE number (two k-tiles share each). The dequantisation is `matmul_i8_dq_stream.cu`'s: the E4M3 byte becomes the bf16
// bit pattern decode(b) * 2^-120, one bf16x2 multiply by scale * 2^120 gives decode(b) * scale rounded once to bf16. The
// accumulators still run straight across K: there is no fold at all, which is the point of the format against the int8
// tier's scale per 32 codes (an fp8 wgmma on E4M3 activations would halve the tensor-core time again; that needs the
// activations quantised per 128 and is a separate candidate).
//
// No `__restrict__`: brain's device buffers alias by design.

#define BRAIN_DQW_BM 128
#define BRAIN_DQW_BN 128
#define BRAIN_DQW_BK 64
#define BRAIN_DQW_THREADS 256
#define BRAIN_DQW_STAGES 5
#define BRAIN_DQW_PREFETCH 3

// Bytes per ring stage: the activation tile (128 rows x 128 bytes), the raw weight tile (128 rows x 64 bytes), 256 scales.
#define BRAIN_DQW_X_BYTES (BRAIN_DQW_BM * 128)
#define BRAIN_DQW_R_BYTES (BRAIN_DQW_BN * BRAIN_DQW_BK)
// (FP8 uses the first 4 bytes of this area for the one scale of the tile: a stage stays a whole number of KiB so that every
// activation tile keeps the 1024-byte alignment of the 128-byte swizzle.)
#define BRAIN_DQW_S_BYTES (BRAIN_DQW_BN * 2 * 4)
#define BRAIN_DQW_STAGE_BYTES (BRAIN_DQW_X_BYTES + BRAIN_DQW_R_BYTES + BRAIN_DQW_S_BYTES)
// The three dequantised weight tiles (128 rows x 128 bytes each), after the ring. Three, not two: the tile written for
// k-tile kt + 1 was last read by the wgmma of k-tile kt - 2, which BOTH warpgroups are known to have finished at the barrier
// that opens the iteration (each waits for it behind its own commit); with two tiles it would be the wgmma of kt - 1,
// which the other warpgroup may still be running.
#define BRAIN_DQW_D_BYTES (BRAIN_DQW_BN * 128)
#define BRAIN_DQW_DTILES 3

#ifndef BRAIN_PDL_ENTRY
#define BRAIN_PDL_ENTRY() do {} while (0)
#endif

// The kernel's entry point; the FP8 build is registered under its own name.
#ifndef BRAIN_DQW_ENTRY
#define BRAIN_DQW_ENTRY brain_matmul_i8_dq_wgmma
#endif

__device__ __forceinline__ unsigned int brain_dqw_smem(const void* p) {
    return static_cast<unsigned int>(__cvta_generic_to_shared(p));
}

// Global -> shared copy, zero-filled when `bytes` is 0 (a row past the edge of the matrix). The pointer is always valid.
__device__ __forceinline__ void brain_dqw_cp16(unsigned int dst, const void* src, unsigned int bytes) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(bytes) : "memory");
}

__device__ __forceinline__ void brain_dqw_cp4(unsigned int dst, const void* src, unsigned int bytes) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n" ::"r"(dst), "l"(src), "r"(bytes) : "memory");
}

__device__ __forceinline__ void brain_dqw_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }

template <int N>
__device__ __forceinline__ void brain_dqw_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N) : "memory"); }

// Shared-memory matrix descriptor: 128-byte swizzle, K-major, 8-row groups 1024 B apart.
__device__ __forceinline__ unsigned long long brain_dqw_desc(unsigned int smem_addr) {
    return static_cast<unsigned long long>((smem_addr & 0x3FFFFu) >> 4) | (1ull << 16) | (64ull << 32) | (1ull << 62);
}

__device__ __forceinline__ void brain_dqw_fence() { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void brain_dqw_wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
template <int N>
__device__ __forceinline__ void brain_dqw_wg_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(N) : "memory"); }
// Writes made through the generic proxy (the dequantised tile) are visible to the async proxy (wgmma) after this.
__device__ __forceinline__ void brain_dqw_fence_proxy() { asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory"); }

// D (64 x 128 f32, 64 registers per thread) += A (64 x 16 bf16) . B (128 x 16 bf16)^T.
__device__ __forceinline__ void brain_dqw_wgmma(float (&d)[64], unsigned long long da, unsigned long long db) {
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, 1, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
        "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, "
        "%32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, "
        "%48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, "
        "%64, %65, p, 1, 1, 0, 0;\n"
        "}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
          "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
          "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),
          "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]),
          "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]),
          "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]),
          "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]),
          "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63])
        : "l"(da), "l"(db));
}

struct BrainDqwConsts {
    unsigned int low7, sign, magic;
};

// The masks and the bf16 128.0, held in registers so `(u & mask) | magic` is one `lop3` (two immediates would be two instructions).
__device__ __forceinline__ BrainDqwConsts brain_dqw_consts() {
    BrainDqwConsts c;
    asm volatile("mov.b32 %0, 0x007F007F;\n" : "=r"(c.low7));
    asm volatile("mov.b32 %0, 0x00800080;\n" : "=r"(c.sign));
    asm volatile("mov.b32 %0, 0x43004300;\n" : "=r"(c.magic));
    return c;
}

__device__ __forceinline__ unsigned int brain_dqw_and_or(unsigned int a, unsigned int mask, unsigned int magic) {
    unsigned int d;
    asm("lop3.b32 %0, %1, %2, %3, 0xEA;\n" : "=r"(d) : "r"(a), "r"(mask), "r"(magic));
    return d;
}

__device__ __forceinline__ unsigned int brain_dqw_fma2(unsigned int a, unsigned int b, unsigned int c) {
    unsigned int d;
    asm("fma.rn.bf16x2 %0, %1, %2, %3;\n" : "=r"(d) : "r"(a), "r"(b), "r"(c));
    return d;
}

// Two int8 codes in the low bytes of the 16-bit lanes of `u` -> bf16x2 of code * scale (scale2: the scale in both lanes).
__device__ __forceinline__ unsigned int brain_dqw_dequant2(unsigned int u, unsigned int scale2, const BrainDqwConsts& k) {
    const unsigned int h1 = brain_dqw_and_or(u, k.low7, k.magic);
    const unsigned int h2 = brain_dqw_and_or(u, k.sign, k.magic);
    const unsigned int q = brain_dqw_fma2(h2, 0xBF80BF80u, h1);  // h1 - h2
    return brain_dqw_fma2(q, scale2, 0u);
}

// Four consecutive int8 codes (one word) -> two bf16x2 registers, codes (0,1) and (2,3), in k order.
__device__ __forceinline__ void brain_dqw_dequant4(unsigned int word, unsigned int scale2, const BrainDqwConsts& k, unsigned int& lo, unsigned int& hi) {
    unsigned int u01, u23;
    asm("prmt.b32 %0, %1, 0, 0x4140;\n" : "=r"(u01) : "r"(word));
    asm("prmt.b32 %0, %1, 0, 0x4342;\n" : "=r"(u23) : "r"(word));
    lo = brain_dqw_dequant2(u01, scale2, k);
    hi = brain_dqw_dequant2(u23, scale2, k);
}

// Two E4M3 bytes in the HIGH bytes of the 16-bit lanes of `x` -> bf16x2 of decode(byte) * scale (scale2: scale * 2^120 in both lanes).
__device__ __forceinline__ unsigned int brain_dqw_fp8_pair(unsigned int x, unsigned int scale2) {
    const unsigned int pattern = ((x >> 4) & 0x07F007F0u) | (x & 0x80008000u);
    return brain_dqw_fma2(pattern, scale2, 0u);
}

// Four consecutive E4M3 bytes (one word) -> two bf16x2 registers, bytes (0,1) and (2,3), in k order.
__device__ __forceinline__ void brain_dqw_fp8_dequant4(unsigned int word, unsigned int scale2, unsigned int& lo, unsigned int& hi) {
    unsigned int x01, x23;
    asm("prmt.b32 %0, %1, 0, 0x1404;\n" : "=r"(x01) : "r"(word));
    asm("prmt.b32 %0, %1, 0, 0x3424;\n" : "=r"(x23) : "r"(word));
    lo = brain_dqw_fp8_pair(x01, scale2);
    hi = brain_dqw_fp8_pair(x23, scale2);
}

__device__ __forceinline__ unsigned int brain_dqw_splat_bf16(float s) {
    unsigned int d;
    asm("cvt.rn.bf16x2.f32 %0, %1, %1;\n" : "=r"(d) : "f"(s));
    return d;
}

extern "C" __global__ void __launch_bounds__(BRAIN_DQW_THREADS, 1)
BRAIN_DQW_ENTRY(const unsigned int* params, const unsigned int* xb, const unsigned int* wq, const float* sw, float* out) {
    BRAIN_PDL_ENTRY();
    extern __shared__ unsigned char brain_dqw_smem_raw[];
    const unsigned int M = params[0];
    const unsigned int K = params[1] * 4u;
    const unsigned int N = params[2];
    const unsigned int ng = K / 32u;
    const unsigned int kblocks = K / 128u;  // FP8: scale blocks along K
    const unsigned int ktiles = K / BRAIN_DQW_BK;

    // Flat 1-D grid, row tile fastest: the blocks that run together share a weight tile.
    const unsigned int blk = blockIdx.y * gridDim.x + blockIdx.x;
    const unsigned int tiles_m = (M + BRAIN_DQW_BM - 1u) / BRAIN_DQW_BM;
    if (tiles_m == 0u) { return; }
    const unsigned int tile_n = blk / tiles_m;
    const unsigned int tile_m = blk - tile_n * tiles_m;
    const unsigned int m0 = tile_m * BRAIN_DQW_BM;
    const unsigned int n0 = tile_n * BRAIN_DQW_BN;
    if (m0 >= M || n0 >= N) { return; }  // block-uniform

    const unsigned int tid = threadIdx.x;
    const unsigned int wg = tid / 128;  // warpgroup: rows of x 64 * wg .. 64 * wg + 63
    const unsigned int lane = tid & 31u;
    const unsigned int warp = (tid >> 5) & 3u;

    // 1024-byte aligned base: the 128-byte swizzle pattern repeats every 1024 bytes.
    const unsigned int raw_addr = brain_dqw_smem(brain_dqw_smem_raw);
    const unsigned int base = (raw_addr + 1023u) & ~1023u;
    unsigned char* const gen = brain_dqw_smem_raw + (base - raw_addr);
    const unsigned int deq_base = base + BRAIN_DQW_STAGES * BRAIN_DQW_STAGE_BYTES;

    const unsigned char* const x8 = reinterpret_cast<const unsigned char*>(xb);
    const unsigned char* const w8 = reinterpret_cast<const unsigned char*>(wq);

    // Copy plan of one thread, computed once. Activation tile: 1024 chunks of 16 bytes, chunk i = tid + 256 it (row
    // tid / 8 + 32 it, chunk tid % 8), swizzled. Raw weight tile: 512 chunks, chunk i = tid + 256 it (row tid / 4 + 64 it,
    // chunk tid % 4), unswizzled (64-byte rows; the dequantising reads touch two whole rows a quarter-warp). Scales: one
    // float a thread (row tid / 2, group tid % 2).
    const unsigned int xrow = tid >> 3, xc = tid & 7u;
    const unsigned int x_dst = xrow * 128u + ((xc ^ (xrow & 7u)) << 4);  // + 4096 it: 32 rows keep row mod 8
    const unsigned char* x_src[4];
    unsigned int x_len[4];
#pragma unroll
    for (int it = 0; it < 4; ++it) {
        const unsigned int r = m0 + xrow + 32u * it;
        x_len[it] = r < M ? 16u : 0u;
        x_src[it] = x8 + (unsigned long long)(r < M ? r : 0u) * K * 2ull + xc * 16u;
    }
    const unsigned int rrow = tid >> 2, rc = tid & 3u;
    const unsigned char* w_src[2];
    unsigned int w_len[2];
#pragma unroll
    for (int it = 0; it < 2; ++it) {
        const unsigned int r = n0 + rrow + 64u * it;
        w_len[it] = r < N ? 16u : 0u;
        w_src[it] = w8 + (unsigned long long)(r < N ? r : 0u) * K + rc * 16u;
    }
#ifdef BRAIN_DQW_FP8
    const float* const s_src = sw + (unsigned long long)(n0 >> 7) * kblocks;
#else
    const unsigned int s_row = n0 + (tid >> 1);
    const bool s_ok = s_row < N;
    const float* const s_src = sw + (unsigned long long)(s_ok ? s_row : 0u) * ng + (tid & 1u);
#endif

    auto load_stage = [&](unsigned int slot, unsigned int kt) {
        const unsigned int st = base + slot * BRAIN_DQW_STAGE_BYTES;
#pragma unroll
        for (int it = 0; it < 4; ++it) { brain_dqw_cp16(st + x_dst + 4096u * it, x_src[it] + kt * 128u, x_len[it]); }
#pragma unroll
        for (int it = 0; it < 2; ++it) { brain_dqw_cp16(st + BRAIN_DQW_X_BYTES + (rrow + 64u * it) * 64u + rc * 16u, w_src[it] + kt * 64u, w_len[it]); }
#ifdef BRAIN_DQW_FP8
        if (tid == 0u) { brain_dqw_cp4(st + BRAIN_DQW_X_BYTES + BRAIN_DQW_R_BYTES, s_src + (kt >> 1), 4u); }
#else
        brain_dqw_cp4(st + BRAIN_DQW_X_BYTES + BRAIN_DQW_R_BYTES + tid * 4u, s_src + kt * 2u, s_ok ? 4u : 0u);
#endif
    };

    // Dequantising a k-tile's raw weights (ring slot `slot`) into a bf16 tile (0..2): `load_raw` reads this thread's two
    // 16-code chunks (rows tid / 4 and tid / 4 + 64, chunk tid % 4: codes 16 c .. 16 c + 15 of the 64, scale group c / 2)
    // and their scales; `store_dequantised` converts and writes them to the swizzled bf16 tile. (Issuing the loads before the
    // wgmma of the iteration instead was measured 11 % slower: the registers held across the issue cost more than the
    // latency they hide.)
    const BrainDqwConsts consts = brain_dqw_consts();
    uint4 raw_codes[2];
    float raw_scales[2];
    auto load_raw = [&](unsigned int slot) {
        const unsigned char* raw = gen + slot * BRAIN_DQW_STAGE_BYTES + BRAIN_DQW_X_BYTES;
        const float* scales = reinterpret_cast<const float*>(gen + slot * BRAIN_DQW_STAGE_BYTES + BRAIN_DQW_X_BYTES + BRAIN_DQW_R_BYTES);
#pragma unroll
        for (int it = 0; it < 2; ++it) {
            const unsigned int row = rrow + 64u * it;
            raw_codes[it] = *reinterpret_cast<const uint4*>(raw + row * 64u + rc * 16u);
#ifdef BRAIN_DQW_FP8
            raw_scales[it] = scales[0];
#else
            raw_scales[it] = scales[row * 2u + (rc >> 1)];
#endif
        }
    };
    auto store_dequantised = [&](unsigned int dst) {
#pragma unroll
        for (int it = 0; it < 2; ++it) {
            const unsigned int row = rrow + 64u * it;
            const uint4 v = raw_codes[it];
            uint4 lo, hi;  // codes 0-7 and 8-15 as bf16: logical 16-byte chunks 2 rc and 2 rc + 1 of the row
#ifdef BRAIN_DQW_FP8
            const unsigned int s2 = brain_dqw_splat_bf16(raw_scales[it] * 1.329227995784916e36f);
            brain_dqw_fp8_dequant4(v.x, s2, lo.x, lo.y);
            brain_dqw_fp8_dequant4(v.y, s2, lo.z, lo.w);
            brain_dqw_fp8_dequant4(v.z, s2, hi.x, hi.y);
            brain_dqw_fp8_dequant4(v.w, s2, hi.z, hi.w);
#else
            const unsigned int s2 = brain_dqw_splat_bf16(raw_scales[it]);
            brain_dqw_dequant4(v.x, s2, consts, lo.x, lo.y);
            brain_dqw_dequant4(v.y, s2, consts, lo.z, lo.w);
            brain_dqw_dequant4(v.z, s2, consts, hi.x, hi.y);
            brain_dqw_dequant4(v.w, s2, consts, hi.z, hi.w);
#endif
            unsigned char* tile = gen + BRAIN_DQW_STAGES * BRAIN_DQW_STAGE_BYTES + dst * BRAIN_DQW_D_BYTES + row * 128u;
            *reinterpret_cast<uint4*>(tile + (((2u * rc) ^ (row & 7u)) << 4)) = lo;
            *reinterpret_cast<uint4*>(tile + (((2u * rc + 1u) ^ (row & 7u)) << 4)) = hi;
        }
    };

    float acc[64];
#pragma unroll
    for (int i = 0; i < 64; ++i) { acc[i] = 0.0f; }

#pragma unroll
    for (unsigned int s = 0; s < BRAIN_DQW_PREFETCH; ++s) {
        if (s < ktiles) { load_stage(s, s); }
        brain_dqw_commit();
    }
    // Tile 0's raw weights, then publish them to the async proxy before the first wgmma.
    brain_dqw_wait<BRAIN_DQW_PREFETCH - 1>();
    __syncthreads();
    load_raw(0);
    store_dequantised(0);

    for (unsigned int kt = 0; kt < ktiles; ++kt) {
        // Tile kt + 1's copies landed (this thread's) and the dequantised tile kt is complete in every thread: after the
        // barrier the wgmma of tile kt may read it and the raw slot of tile kt + 1 is complete.
        brain_dqw_wait<BRAIN_DQW_PREFETCH - 2>();
        brain_dqw_fence_proxy();  // this thread's cp.async data and last iteration's dequantised stores, for the async proxy
        __syncthreads();
        if (kt + BRAIN_DQW_PREFETCH < ktiles) { load_stage((kt + BRAIN_DQW_PREFETCH) % BRAIN_DQW_STAGES, kt + BRAIN_DQW_PREFETCH); }
        brain_dqw_commit();

        const unsigned int slot = kt % BRAIN_DQW_STAGES;
        const unsigned int a_base = base + slot * BRAIN_DQW_STAGE_BYTES + wg * (64u * 128u);
        const unsigned int b_base = deq_base + (kt % BRAIN_DQW_DTILES) * BRAIN_DQW_D_BYTES;
        brain_dqw_fence();
#pragma unroll
        for (int q = 0; q < 4; ++q) { brain_dqw_wgmma(acc, brain_dqw_desc(a_base + q * 32u), brain_dqw_desc(b_base + q * 32u)); }
        brain_dqw_wg_commit();
        // The wgmma of tile kt - 1 read the bf16 tile this thread now overwrites and the ring slot the prefetch above
        // refilled; wait for it (tile kt's keeps running) before dequantising tile kt + 1 into the other tile.
        brain_dqw_wg_wait<1>();
        if (kt + 1 < ktiles) {
            load_raw((kt + 1) % BRAIN_DQW_STAGES);
            store_dequantised((kt + 1) % BRAIN_DQW_DTILES);
        }
    }
    brain_dqw_wg_wait<0>();

    // out[m, n] = acc: thread layout of the m64n128 accumulator (rows r and r + 8 of the warp's 16, columns 8 j + 2 t, +1).
    const unsigned int frow = wg * 64u + warp * 16u + (lane >> 2);
    const unsigned int fcol = (lane & 3u) * 2u;
    const unsigned int r0 = m0 + frow, r1 = r0 + 8u;
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        const unsigned int c = n0 + j * 8u + fcol;
        if (r0 < M) {
            float* o = out + (unsigned long long)r0 * N + c;
            if (c + 1u < N && (N & 1u) == 0u) { *reinterpret_cast<float2*>(o) = make_float2(acc[4 * j], acc[4 * j + 1]); }
            else { if (c < N) { o[0] = acc[4 * j]; } if (c + 1u < N) { o[1] = acc[4 * j + 1]; } }
        }
        if (r1 < M) {
            float* o = out + (unsigned long long)r1 * N + c;
            if (c + 1u < N && (N & 1u) == 0u) { *reinterpret_cast<float2*>(o) = make_float2(acc[4 * j + 2], acc[4 * j + 3]); }
            else { if (c < N) { o[0] = acc[4 * j + 2]; } if (c + 1u < N) { o[1] = acc[4 * j + 3]; } }
        }
    }
}
