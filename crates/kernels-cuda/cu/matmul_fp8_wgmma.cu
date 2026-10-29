// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for prompt-processing throughput on FP8 language-model releases for its
// clients. If your team needs expertise in feeding Hopper FP8 tensor cores from block-scaled checkpoints, folding the
// scales once per 128 of K, then you can procure our services by sending an email to info@swedishembedded.com.
//
// Prefill-sized matmul of FP8 E4M3 weights (one f32 scale per 128 x 128 block, as the release stores them) against
// E4M3 activations (one f32 scale per row per 128 of K, `quant_e4m3_128`) on Hopper WARPGROUP tensor cores
// (`wgmma.mma_async` m64n128k32, e4m3 x e4m3 -> f32):
//
//   params : u32 [m, kg, n]   kg = K/4 packed words per weight row, K a multiple of 128
//   xq     : [M, K] bytes     activations, E4M3 (4-byte aligned, rows 16-byte aligned)
//   w_q    : [N, kg] u32      weights, E4M3, four to a word (16-byte aligned)
//   sx     : [M, K/128] f32   activation scale per row and 128-block
//   sw     : [ceil(N/128), K/128] f32   weight scale per 128 x 128 block
//   out    : [M, N] f32       out[m, n] = sum over the K/128 blocks b of  sx[m, b] * sw[n/128, b] * (xq[m, b] . w_q[n, b])
//
// The format's point (knowledge 372): the integer tier folds a scale after every 32 of K; here a stage is 128 of K, which
// is exactly one activation scale per row and one weight scale for the whole tile (its 128 weight rows lie in one band of
// the scale grid), so the per-block product of the tensor core (4 wgmma of k32, accumulated in f32 with no scale) is scaled
// and added into the running total ONCE per 128 of K: 64 fused multiply-adds a thread per stage against 4 wgmma.
//
// Data movement: a ring of BRAIN_FW_STAGES `cp.async` stages holds the activation tile (128 rows x 128 bytes), the weight
// tile (128 rows x 128 bytes), both in the 128-byte swizzle the wgmma descriptors name (16-byte chunks XOR-swizzled by row
// mod 8), and the stage's scales. The tensor cores read the bytes as they arrive: no dequantisation pass.
//
// Row tails: weight rows past n and rows of x past m are zero-filled and dropped at the store.
//
// No `__restrict__`: brain's device buffers alias by design.

#define BRAIN_FW_BM 128
#define BRAIN_FW_BN 128
#define BRAIN_FW_THREADS 256
#define BRAIN_FW_STAGES 4
#define BRAIN_FW_PREFETCH 3

// Bytes per ring stage: the activation tile and the weight tile (128 rows x 128 bytes each), then 1 KiB for the scales
// (128 activation scales and one weight scale): a whole number of KiB, so every swizzled tile stays 1024-byte aligned.
#define BRAIN_FW_X_BYTES (BRAIN_FW_BM * 128)
#define BRAIN_FW_W_BYTES (BRAIN_FW_BN * 128)
#define BRAIN_FW_S_BYTES 1024
#define BRAIN_FW_STAGE_BYTES (BRAIN_FW_X_BYTES + BRAIN_FW_W_BYTES + BRAIN_FW_S_BYTES)

#ifndef BRAIN_PDL_ENTRY
#define BRAIN_PDL_ENTRY() do {} while (0)
#endif

__device__ __forceinline__ unsigned int brain_fw_smem(const void* p) {
    return static_cast<unsigned int>(__cvta_generic_to_shared(p));
}

// Global -> shared copy, zero-filled when `bytes` is 0 (a row past the edge of the matrix). The pointer is always valid.
__device__ __forceinline__ void brain_fw_cp16(unsigned int dst, const void* src, unsigned int bytes) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(bytes) : "memory");
}

__device__ __forceinline__ void brain_fw_cp4(unsigned int dst, const void* src, unsigned int bytes) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n" ::"r"(dst), "l"(src), "r"(bytes) : "memory");
}

__device__ __forceinline__ void brain_fw_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }

template <int N>
__device__ __forceinline__ void brain_fw_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N) : "memory"); }

// Shared-memory matrix descriptor: 128-byte swizzle, K-major, 8-row groups 1024 B apart.
__device__ __forceinline__ unsigned long long brain_fw_desc(unsigned int smem_addr) {
    return static_cast<unsigned long long>((smem_addr & 0x3FFFFu) >> 4) | (1ull << 16) | (64ull << 32) | (1ull << 62);
}

__device__ __forceinline__ void brain_fw_fence() { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void brain_fw_wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
template <int N>
__device__ __forceinline__ void brain_fw_wg_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(N) : "memory"); }
// Writes made by cp.async (the generic proxy) are visible to the async proxy (wgmma) after this.
__device__ __forceinline__ void brain_fw_fence_proxy() { asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory"); }

// D (64 x 128 f32, 64 registers per thread) = A (64 x 32 e4m3) . B (128 x 32 e4m3)^T, plus D when `accumulate` is set.
__device__ __forceinline__ void brain_fw_wgmma(float (&d)[64], unsigned long long da, unsigned long long db, int accumulate) {
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %66, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
        "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, "
        "%32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, "
        "%48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, "
        "%64, %65, p, 1, 1;\n"
        "}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
          "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
          "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),
          "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]),
          "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]),
          "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]),
          "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]),
          "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63])
        : "l"(da), "l"(db), "r"(accumulate));
}

extern "C" __global__ void __launch_bounds__(BRAIN_FW_THREADS, 1)
brain_matmul_fp8_wgmma(const unsigned int* params, const unsigned int* xq, const unsigned int* wq, const float* sx, const float* sw, float* out) {
    BRAIN_PDL_ENTRY();
    extern __shared__ unsigned char brain_fw_smem_raw[];
    const unsigned int M = params[0];
    const unsigned int K = params[1] * 4u;
    const unsigned int N = params[2];
    const unsigned int kblocks = K / 128u;

    // Flat 1-D grid, row tile fastest: the blocks that run together share a weight tile.
    const unsigned int blk = blockIdx.y * gridDim.x + blockIdx.x;
    const unsigned int tiles_m = (M + BRAIN_FW_BM - 1u) / BRAIN_FW_BM;
    if (tiles_m == 0u) { return; }
    const unsigned int tile_n = blk / tiles_m;
    const unsigned int tile_m = blk - tile_n * tiles_m;
    const unsigned int m0 = tile_m * BRAIN_FW_BM;
    const unsigned int n0 = tile_n * BRAIN_FW_BN;
    if (m0 >= M || n0 >= N) { return; }  // block-uniform

    const unsigned int tid = threadIdx.x;
    const unsigned int wg = tid / 128;  // warpgroup: rows of x 64 * wg .. 64 * wg + 63
    const unsigned int lane = tid & 31u;
    const unsigned int warp = (tid >> 5) & 3u;

    // 1024-byte aligned base: the 128-byte swizzle pattern repeats every 1024 bytes.
    const unsigned int raw_addr = brain_fw_smem(brain_fw_smem_raw);
    const unsigned int base = (raw_addr + 1023u) & ~1023u;
    unsigned char* const gen = brain_fw_smem_raw + (base - raw_addr);

    const unsigned char* const x8 = reinterpret_cast<const unsigned char*>(xq);
    const unsigned char* const w8 = reinterpret_cast<const unsigned char*>(wq);

    // Copy plan of one thread: each tile is 1024 chunks of 16 bytes, chunk i = tid + 256 it (row tid / 8 + 32 it, chunk
    // tid % 8), swizzled by the row's low three bits.
    const unsigned int crow = tid >> 3, cc = tid & 7u;
    const unsigned int c_dst = crow * 128u + ((cc ^ (crow & 7u)) << 4);  // + 4096 it: 32 rows keep row mod 8
    const unsigned char* x_src[4];
    const unsigned char* w_src[4];
    unsigned int x_len[4], w_len[4];
#pragma unroll
    for (int it = 0; it < 4; ++it) {
        const unsigned int xr = m0 + crow + 32u * it;
        x_len[it] = xr < M ? 16u : 0u;
        x_src[it] = x8 + (unsigned long long)(xr < M ? xr : 0u) * K + cc * 16u;
        const unsigned int wr = n0 + crow + 32u * it;
        w_len[it] = wr < N ? 16u : 0u;
        w_src[it] = w8 + (unsigned long long)(wr < N ? wr : 0u) * K + cc * 16u;
    }
    // Scales: thread t < 128 copies the activation scale of row m0 + t; thread 0 also the weight tile's scale.
    const unsigned int sxr = m0 + tid;
    const bool sx_ok = tid < 128u && sxr < M;
    const float* const sx_src = sx + (unsigned long long)(sx_ok ? sxr : 0u) * kblocks;
    const float* const sw_src = sw + (unsigned long long)(n0 >> 7) * kblocks;

    auto load_stage = [&](unsigned int slot, unsigned int kt) {
        const unsigned int st = base + slot * BRAIN_FW_STAGE_BYTES;
#pragma unroll
        for (int it = 0; it < 4; ++it) {
            brain_fw_cp16(st + c_dst + 4096u * it, x_src[it] + kt * 128u, x_len[it]);
            brain_fw_cp16(st + BRAIN_FW_X_BYTES + c_dst + 4096u * it, w_src[it] + kt * 128u, w_len[it]);
        }
        if (tid < 128u) { brain_fw_cp4(st + BRAIN_FW_X_BYTES + BRAIN_FW_W_BYTES + tid * 4u, sx_src + kt, sx_ok ? 4u : 0u); }
        if (tid == 0u) { brain_fw_cp4(st + BRAIN_FW_X_BYTES + BRAIN_FW_W_BYTES + 512u, sw_src + kt, 4u); }
    };

    float acc[64], d[64];
#pragma unroll
    for (int i = 0; i < 64; ++i) { acc[i] = 0.0f; d[i] = 0.0f; }

#pragma unroll
    for (unsigned int s = 0; s < BRAIN_FW_PREFETCH; ++s) {
        if (s < kblocks) { load_stage(s, s); }
        brain_fw_commit();
    }

    for (unsigned int kt = 0; kt < kblocks; ++kt) {
        // This thread's copies of stage kt landed; after the barrier every thread's did, and the slot refilled below was
        // read by the wgmma of stage kt - 1, which every warpgroup waited for at the end of its iteration.
        brain_fw_wait<BRAIN_FW_PREFETCH - 1>();
        brain_fw_fence_proxy();
        __syncthreads();
        if (kt + BRAIN_FW_PREFETCH < kblocks) { load_stage((kt + BRAIN_FW_PREFETCH) % BRAIN_FW_STAGES, kt + BRAIN_FW_PREFETCH); }
        brain_fw_commit();

        const unsigned int slot = kt % BRAIN_FW_STAGES;
        const unsigned int st = base + slot * BRAIN_FW_STAGE_BYTES;
        const unsigned int a_base = st + wg * (64u * 128u);
        const unsigned int b_base = st + BRAIN_FW_X_BYTES;
        brain_fw_fence();
#pragma unroll
        for (int q = 0; q < 4; ++q) { brain_fw_wgmma(d, brain_fw_desc(a_base + q * 32u), brain_fw_desc(b_base + q * 32u), q > 0 ? 1 : 0); }
        brain_fw_wg_commit();
        brain_fw_wg_wait<0>();

        // The fold, once per 128 of K: the stage's tensor-core sum times the row's activation scale and the tile's weight
        // scale, added into the running total (rows r and r + 8 of this thread's fragment).
        const float* const sxs = reinterpret_cast<const float*>(gen + slot * BRAIN_FW_STAGE_BYTES + BRAIN_FW_X_BYTES + BRAIN_FW_W_BYTES);
        const float wscale = sxs[128];
        const unsigned int frow = wg * 64u + warp * 16u + (lane >> 2);
        const float s0 = sxs[frow] * wscale, s1 = sxs[frow + 8u] * wscale;
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            acc[4 * j] = fmaf(d[4 * j], s0, acc[4 * j]);
            acc[4 * j + 1] = fmaf(d[4 * j + 1], s0, acc[4 * j + 1]);
            acc[4 * j + 2] = fmaf(d[4 * j + 2], s1, acc[4 * j + 2]);
            acc[4 * j + 3] = fmaf(d[4 * j + 3], s1, acc[4 * j + 3]);
        }
    }

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
