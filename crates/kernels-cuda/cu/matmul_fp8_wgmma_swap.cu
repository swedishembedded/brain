// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for batched-decode throughput on FP8 language-model releases for its
// clients. If your team needs expertise in streaming block-scaled FP8 weights through Hopper tensor cores at the memory
// roof, then you can procure our services by sending an email to info@swedishembedded.com.
//
// Small-batch FP8 GEMM (17 to 64 rows of x) with the WEIGHTS as the tensor core's A operand: `wgmma.mma_async`
// m64nNk32 e4m3 x e4m3 -> f32, a warpgroup owning 64 weight rows (M of the MMA) against the NX rows of x (N of the MMA):
//
//   params : u32 [m, kg, n]   as `matmul_fp8_wgmma.cu`; m <= NX
//   out[m, n] = sum over the K/128 blocks b of  sx[m, b] * sw[n/128, b] * (xq[m, b] . w_q[n, b])
//
// Why it exists (knowledge 386, 387): `matmul_fp8_wgmma` puts the rows of x in the 128-row A tile, so at 33 to 64 rows half
// of its MMA work is padding and its 128 x 128 tiles are one block per 128 weight rows. Here the rows of x are the narrow
// side, the tile is BN weight rows x all NX rows, the ring holds BN weight rows and NX rows of x per 128 of K, and a
// warpgroup's accumulator is 64 x NX f32 (NX / 2 registers a thread). The weight copies of the first stages are issued
// before `griddepcontrol.wait` (they do not depend on the previous kernel).
//
// Scale fold: the stage's tensor-core sum is multiplied by the tile's one weight scale and the column's activation scale
// and added into the running total once per 128 of K, as in `matmul_fp8_wgmma`. The accumulator element (weight row r, x row c)
// is stored transposed: out[c * n + r].
//
// No `__restrict__`: brain's device buffers alias by design.

#ifndef BRAIN_FS_BN
#define BRAIN_FS_BN 128  // weight rows a block: 64 or 128 (a warpgroup each 64)
#endif
#ifndef BRAIN_FS_NX
#define BRAIN_FS_NX 64  // rows of x a block holds: 32 or 64
#endif
#ifndef BRAIN_FS_STAGES
#define BRAIN_FS_STAGES 4
#endif
#ifndef BRAIN_FS_MIN_BLOCKS
#define BRAIN_FS_MIN_BLOCKS 1
#endif
#ifndef BRAIN_FS_SPLIT
#define BRAIN_FS_SPLIT 1  // blocks of a cluster sharing a tile of weight rows, each taking a share of K
#endif
#ifndef BRAIN_FS_ENTRY
#define BRAIN_FS_ENTRY brain_matmul_fp8_wgmma_swap
#endif

#define BRAIN_FS_THREADS (BRAIN_FS_BN * 2)
#define BRAIN_FS_W_BYTES (BRAIN_FS_BN * 128)
#define BRAIN_FS_X_BYTES (BRAIN_FS_NX * 128)
#define BRAIN_FS_TILE_BYTES (BRAIN_FS_W_BYTES + BRAIN_FS_X_BYTES)
#define BRAIN_FS_S_BYTES 512  // per stage: NX activation scales, then the tile's weight scale at +256

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
#define BRAIN_FS_PDL_WAIT() asm volatile("griddepcontrol.wait;\n" ::: "memory")
#define BRAIN_FS_PDL_LAUNCH() asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory")
#else
#define BRAIN_FS_PDL_WAIT() do {} while (0)
#define BRAIN_FS_PDL_LAUNCH() do {} while (0)
#endif

__device__ __forceinline__ unsigned int brain_fs_smem(const void* p) { return static_cast<unsigned int>(__cvta_generic_to_shared(p)); }
__device__ __forceinline__ void brain_fs_cp16(unsigned int dst, const void* src, unsigned int bytes) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(bytes) : "memory");
}
__device__ __forceinline__ void brain_fs_cp4(unsigned int dst, const void* src, unsigned int bytes) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n" ::"r"(dst), "l"(src), "r"(bytes) : "memory");
}
__device__ __forceinline__ void brain_fs_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
template <int N>
__device__ __forceinline__ void brain_fs_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N) : "memory"); }
__device__ __forceinline__ unsigned long long brain_fs_desc(unsigned int smem_addr) {
    return static_cast<unsigned long long>((smem_addr & 0x3FFFFu) >> 4) | (1ull << 16) | (64ull << 32) | (1ull << 62);
}
__device__ __forceinline__ void brain_fs_fence() { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void brain_fs_wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
template <int N>
__device__ __forceinline__ void brain_fs_wg_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(N) : "memory"); }
__device__ __forceinline__ void brain_fs_fence_proxy() { asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory"); }

// D (64 x 32, 16 registers) = A (64 x 32 e4m3) . B (32 x 32 e4m3)^T [+ D].
__device__ __forceinline__ void brain_fs_wgmma_n32(float (&d)[16], unsigned long long da, unsigned long long db, int accumulate) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %18, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n32k32.f32.e4m3.e4m3 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, %16, %17, p, 1, 1;\n}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]),
          "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15])
        : "l"(da), "l"(db), "r"(accumulate));
}

// D (64 x 64, 32 registers) = A (64 x 32 e4m3) . B (64 x 32 e4m3)^T [+ D].
__device__ __forceinline__ void brain_fs_wgmma_n64(float (&d)[32], unsigned long long da, unsigned long long db, int accumulate) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %34, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
        "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, p, 1, 1;\n}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]),
          "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]),
          "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]),
          "+f"(d[31])
        : "l"(da), "l"(db), "r"(accumulate));
}

__device__ __forceinline__ void brain_fs_cluster_sync() {
    asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;\n" ::: "memory");
}
__device__ __forceinline__ float brain_fs_remote_ld(unsigned int local_addr, unsigned int rank) {
    unsigned int remote;
    float v;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;\n" : "=r"(remote) : "r"(local_addr), "r"(rank));
    asm volatile("ld.shared::cluster.f32 %0, [%1];\n" : "=f"(v) : "r"(remote) : "memory");
    return v;
}

template <int NX>
__device__ __forceinline__ void brain_fs_mma(float (&d)[NX / 2], unsigned long long da, unsigned long long db, int accumulate);
template <>
__device__ __forceinline__ void brain_fs_mma<32>(float (&d)[16], unsigned long long da, unsigned long long db, int accumulate) { brain_fs_wgmma_n32(d, da, db, accumulate); }
template <>
__device__ __forceinline__ void brain_fs_mma<64>(float (&d)[32], unsigned long long da, unsigned long long db, int accumulate) { brain_fs_wgmma_n64(d, da, db, accumulate); }

extern "C" __global__ void __launch_bounds__(BRAIN_FS_THREADS, BRAIN_FS_MIN_BLOCKS)
BRAIN_FS_ENTRY(const unsigned int* params, const unsigned int* xq, const unsigned int* wq, const float* sx, const float* sw, float* out) {
    constexpr int NX = BRAIN_FS_NX;
    constexpr int BN = BRAIN_FS_BN;
    constexpr int T = BRAIN_FS_THREADS;
    constexpr int STAGES = BRAIN_FS_STAGES;
    constexpr int PREFETCH = STAGES - 1;
    constexpr int W_IT = BN * 8 / T;   // 16-byte weight chunks a thread copies a stage
    constexpr int X_IT = (NX * 8 + T - 1) / T;
    constexpr int ROW_STEP = T / 8;    // rows between a thread's chunks (a multiple of 8: the swizzle phase is constant)
    extern __shared__ unsigned char brain_fs_smem_raw[];
    BRAIN_FS_PDL_LAUNCH();
    const unsigned int M = params[0];
    const unsigned int K = params[1] * 4u;
    const unsigned int N = params[2];
    const unsigned int kblocks = K / 128u;
    constexpr int SPLIT = BRAIN_FS_SPLIT;
    // A cluster's blocks are consecutive and share a tile of weight rows; block `rank` takes a share of the K blocks.
    const unsigned int blk = blockIdx.y * gridDim.x + blockIdx.x;
    const unsigned int rank = blk % SPLIT;
    const unsigned int n0 = (blk / SPLIT) * BN;
    const unsigned int kb0 = rank * kblocks / SPLIT;
    const unsigned int kn = (rank + 1) * kblocks / SPLIT - kb0;
    if (n0 >= N) { BRAIN_FS_PDL_WAIT(); return; }

    const unsigned int tid = threadIdx.x;
    const unsigned int wg = tid / 128;
    const unsigned int lane = tid & 31u;
    const unsigned int warp = (tid >> 5) & 3u;

    const unsigned int raw_addr = brain_fs_smem(brain_fs_smem_raw);
    const unsigned int base = (raw_addr + 1023u) & ~1023u;
    unsigned char* const gen = brain_fs_smem_raw + (base - raw_addr);
    const unsigned int scale_base = base + STAGES * BRAIN_FS_TILE_BYTES;

    const unsigned char* const x8 = reinterpret_cast<const unsigned char*>(xq);
    const unsigned char* const w8 = reinterpret_cast<const unsigned char*>(wq);

    const unsigned int crow = tid >> 3, cc = tid & 7u;
    const unsigned int c_dst = crow * 128u + ((cc ^ (crow & 7u)) << 4);
    const unsigned char* w_src[W_IT];
    unsigned int w_len[W_IT];
#pragma unroll
    for (int it = 0; it < W_IT; ++it) {
        const unsigned int wr = n0 + crow + ROW_STEP * it;
        w_len[it] = wr < N ? 16u : 0u;
        w_src[it] = w8 + (unsigned long long)(wr < N ? wr : 0u) * K + cc * 16u;
    }
    const unsigned char* x_src[X_IT];
    unsigned int x_len[X_IT];
#pragma unroll
    for (int it = 0; it < X_IT; ++it) {
        const unsigned int xr = crow + ROW_STEP * it;
        const bool live = xr < (unsigned int)NX && xr < M;
        x_len[it] = live ? 16u : 0u;
        x_src[it] = x8 + (unsigned long long)(live ? xr : 0u) * K + cc * 16u;
    }
    const bool sx_ok = tid < (unsigned int)NX && tid < M;
    const float* const sx_src = sx + (unsigned long long)(sx_ok ? tid : 0u) * kblocks;
    const float* const sw_src = sw + (unsigned long long)(n0 >> 7) * kblocks;

    auto load_w = [&](unsigned int slot, unsigned int kt) {
        const unsigned int st = base + slot * BRAIN_FS_TILE_BYTES;
#pragma unroll
        for (int it = 0; it < W_IT; ++it) { brain_fs_cp16(st + c_dst + ROW_STEP * 128u * it, w_src[it] + kt * 128u, w_len[it]); }
        if (tid == 0u) { brain_fs_cp4(scale_base + slot * BRAIN_FS_S_BYTES + 256u, sw_src + kt, 4u); }
    };
    auto load_x = [&](unsigned int slot, unsigned int kt) {
        const unsigned int st = base + slot * BRAIN_FS_TILE_BYTES + BRAIN_FS_W_BYTES;
#pragma unroll
        for (int it = 0; it < X_IT; ++it) {
            if (crow + ROW_STEP * it < (unsigned int)NX) { brain_fs_cp16(st + c_dst + ROW_STEP * 128u * it, x_src[it] + kt * 128u, x_len[it]); }
        }
        if (tid < (unsigned int)NX) { brain_fs_cp4(scale_base + slot * BRAIN_FS_S_BYTES + tid * 4u, sx_src + kt, sx_ok ? 4u : 0u); }
    };

    float acc[NX / 2], d[NX / 2];
#pragma unroll
    for (int i = 0; i < NX / 2; ++i) { acc[i] = 0.0f; d[i] = 0.0f; }

    // The weights (and their scales) do not depend on the previous kernel; the rows of x and their scales do.
#pragma unroll
    for (unsigned int s = 0; s < PREFETCH; ++s) {
        if (s < kn) { load_w(s, kb0 + s); }
    }
    BRAIN_FS_PDL_WAIT();
#pragma unroll
    for (unsigned int s = 0; s < PREFETCH; ++s) {
        if (s < kn) { load_x(s, kb0 + s); }
        brain_fs_commit();
    }

    for (unsigned int kt = 0; kt < kn; ++kt) {
        brain_fs_wait<PREFETCH - 1>();
        brain_fs_fence_proxy();
        __syncthreads();
        if (kt + PREFETCH < kn) {
            const unsigned int slot = (kt + PREFETCH) % STAGES;
            load_w(slot, kb0 + kt + PREFETCH);
            load_x(slot, kb0 + kt + PREFETCH);
        }
        brain_fs_commit();

        const unsigned int slot = kt % STAGES;
        const unsigned int st = base + slot * BRAIN_FS_TILE_BYTES;
        const unsigned int a_base = st + wg * (64u * 128u);
        const unsigned int b_base = st + BRAIN_FS_W_BYTES;
        brain_fs_fence();
#pragma unroll
        for (int q = 0; q < 4; ++q) { brain_fs_mma<NX>(d, brain_fs_desc(a_base + q * 32u), brain_fs_desc(b_base + q * 32u), q > 0 ? 1 : 0); }
        brain_fs_wg_commit();
        brain_fs_wg_wait<0>();

        const float* const sxs = reinterpret_cast<const float*>(gen + STAGES * BRAIN_FS_TILE_BYTES + slot * BRAIN_FS_S_BYTES);
        const float wscale = sxs[64];
#pragma unroll
        for (int j = 0; j < NX / 8; ++j) {
            const float c0 = sxs[8 * j + 2 * (lane & 3u)] * wscale, c1 = sxs[8 * j + 2 * (lane & 3u) + 1] * wscale;
            acc[4 * j] = fmaf(d[4 * j], c0, acc[4 * j]);
            acc[4 * j + 1] = fmaf(d[4 * j + 1], c1, acc[4 * j + 1]);
            acc[4 * j + 2] = fmaf(d[4 * j + 2], c0, acc[4 * j + 2]);
            acc[4 * j + 3] = fmaf(d[4 * j + 3], c1, acc[4 * j + 3]);
        }
    }

    // acc[4 j + 2 h + c]: weight row (warp's 16) lane / 4 + 8 h, x row 8 j + 2 (lane % 4) + c.
    if (SPLIT == 1) {
        const unsigned int wrow0 = n0 + wg * 64u + warp * 16u + (lane >> 2);
#pragma unroll
        for (int j = 0; j < NX / 8; ++j) {
#pragma unroll
            for (int c = 0; c < 2; ++c) {
                const unsigned int xr = 8u * j + 2u * (lane & 3u) + c;
                if (xr < M) {
                    float* o = out + (unsigned long long)xr * N;
                    if (wrow0 < N) { o[wrow0] = acc[4 * j + c]; }
                    if (wrow0 + 8u < N) { o[wrow0 + 8u] = acc[4 * j + 2 + c]; }
                }
            }
        }
        return;
    }

    // Split K: park the partial sums ([x row][weight row of the tile], over the drained ring), then each block adds the partials of
    // every block of the cluster, in rank order, for its share of the outputs.
    constexpr int RSTRIDE = BN + 4;
    __syncthreads();
    float* const red = reinterpret_cast<float*>(gen);
    const unsigned int lrow = wg * 64u + warp * 16u + (lane >> 2);
#pragma unroll
    for (int j = 0; j < NX / 8; ++j) {
#pragma unroll
        for (int c = 0; c < 2; ++c) {
            const unsigned int xr = 8u * j + 2u * (lane & 3u) + c;
            red[xr * RSTRIDE + lrow] = acc[4 * j + c];
            red[xr * RSTRIDE + lrow + 8u] = acc[4 * j + 2 + c];
        }
    }
    brain_fs_cluster_sync();
    const unsigned int red_addr = base;
    for (unsigned int idx = rank * T + tid; idx < M * BN; idx += SPLIT * T) {
        const unsigned int col = idx / BN, row = idx % BN;
        float total = 0.0f;
#pragma unroll
        for (int r = 0; r < SPLIT; ++r) { total += brain_fs_remote_ld(red_addr + (col * RSTRIDE + row) * 4u, r); }
        if (n0 + row < N) { out[(unsigned long long)col * N + n0 + row] = total; }
    }
    brain_fs_cluster_sync();  // no block leaves while a peer may still read its partials
}
