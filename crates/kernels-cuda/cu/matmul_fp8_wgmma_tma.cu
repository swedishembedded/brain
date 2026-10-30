// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for prompt-processing throughput on FP8 language-model releases for its
// clients. If your team needs expertise in feeding Hopper FP8 tensor cores through the tensor memory accelerator from
// block-scaled checkpoints, then you can procure our services by sending an email to info@swedishembedded.com.
//
// `matmul_fp8_wgmma.cu` with the tile copies done by the tensor memory accelerator (same parameters, same arithmetic, the same
// result bit for bit; knowledge 387). A persistent grid of at most BRAIN_FWT_MAX_CTAS blocks walks the tiles; a producer warpgroup
// (one thread issues `cp.async.bulk.tensor`, the swizzle and the zero fill of rows past the edge are the descriptor's) publishes
// each stage on an mbarrier, two consumer warpgroups of 64 rows each wait on it, issue their 4 wgmma, drain them, free the slot on a
// second mbarrier and fold the stage's sum. The producer runs ahead across tile boundaries, so a tile's epilogue overlaps the next
// tile's first loads.
//
// The kernel is launched with plain device pointers, so each block builds the two tensor maps (activations, weights) on the device:
// the driver's encoding of everything the shape does not change (BRAIN_FWT_TEMPLATE, held to `cuTensorMapEncodeTiled` by
// `backend-cuda/tests/tensormap_template.rs`) with the global address, dimensions and row stride replaced.

#ifndef BRAIN_FWT_STAGES
#define BRAIN_FWT_STAGES 5
#endif
#ifndef BRAIN_FWT_ENTRY
#define BRAIN_FWT_ENTRY brain_matmul_fp8_wgmma_tma
#endif
#define BRAIN_FWT_THREADS 384
#define BRAIN_FWT_MAX_CTAS 132  // grid size at most, and the SM slots of the maps

// The driver's 128-byte map for a 2-D byte tensor of 256 x 256 at address 0x10000, a 128 x 128 box, the 128-byte swizzle, 256-byte L2 promotion
// and zero fill.
#define BRAIN_FWT_TEMPLATE 0x0000000000010000ull, 0x0000001000066010ull, 0x0000000000000000ull, 0x0000000000000000ull, 0x000000ff000000ffull, 0x0000000000000000ull, 0x7f00000000000000ull, 0x000000000000007full, 0x0000000000004000ull, 0x0000000000000400ull, 0x0000000000000000ull, 0x0000000000000000ull, 0x0000000000000000ull, 0x0000000000000000ull, 0x0000000000000000ull, 0x0000000000000000ull
__device__ __constant__ unsigned long long brain_fwt_template[16] = { BRAIN_FWT_TEMPLATE };
// Each SM's two maps, in global memory (the TMA reads a map from global, constant or parameter memory).
__device__ __align__(128) unsigned long long brain_fwt_maps[BRAIN_FWT_MAX_CTAS][2][16];

#define BRAIN_FWT_BM 128
#define BRAIN_FWT_BN 128

// Bytes per ring stage: the activation tile and the weight tile (128 rows x 128 bytes each), then 1 KiB for the scales
// (128 activation scales and one weight scale): a whole number of KiB, so every swizzled tile stays 1024-byte aligned.
#define BRAIN_FWT_X_BYTES (BRAIN_FWT_BM * 128)
#define BRAIN_FWT_W_BYTES (BRAIN_FWT_BN * 128)
#define BRAIN_FWT_S_BYTES 1024
#define BRAIN_FWT_STAGE_BYTES (BRAIN_FWT_X_BYTES + BRAIN_FWT_W_BYTES + BRAIN_FWT_S_BYTES)

#ifndef BRAIN_PDL_ENTRY
#define BRAIN_PDL_ENTRY() do {} while (0)
#endif

__device__ __forceinline__ unsigned int brain_fwt_smem(const void* p) {
    return static_cast<unsigned int>(__cvta_generic_to_shared(p));
}

// Global -> shared copy, zero-filled when `bytes` is 0 (a row past the edge of the matrix). The pointer is always valid.

__device__ __forceinline__ void brain_fwt_cp4(unsigned int dst, const void* src, unsigned int bytes) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n" ::"r"(dst), "l"(src), "r"(bytes) : "memory");
}

__device__ __forceinline__ void brain_fwt_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }

template <int N>
__device__ __forceinline__ void brain_fwt_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N) : "memory"); }

// Shared-memory matrix descriptor: 128-byte swizzle, K-major, 8-row groups 1024 B apart.
__device__ __forceinline__ unsigned long long brain_fwt_desc(unsigned int smem_addr) {
    return static_cast<unsigned long long>((smem_addr & 0x3FFFFu) >> 4) | (1ull << 16) | (64ull << 32) | (1ull << 62);
}

__device__ __forceinline__ void brain_fwt_fence() { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void brain_fwt_wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
template <int N>
__device__ __forceinline__ void brain_fwt_wg_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(N) : "memory"); }
// Writes made by cp.async (the generic proxy) are visible to the async proxy (wgmma) after this.
__device__ __forceinline__ void brain_fwt_fence_proxy() { asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory"); }

// D (64 x 128 f32, 64 registers per thread) = A (64 x 32 e4m3) . B (128 x 32 e4m3)^T, plus D when `accumulate` is set.
__device__ __forceinline__ void brain_fwt_wgmma(float (&d)[64], unsigned long long da, unsigned long long db, int accumulate) {
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


__device__ __forceinline__ void brain_fwt_mbar_init(unsigned int bar, unsigned int count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(bar), "r"(count) : "memory");
}
__device__ __forceinline__ void brain_fwt_mbar_arrive(unsigned int bar) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" ::"r"(bar) : "memory");
}
__device__ __forceinline__ void brain_fwt_mbar_arrive_expect_tx(unsigned int bar, unsigned int bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(bar), "r"(bytes) : "memory");
}
// The calling thread's earlier cp.async copies arrive on `bar` when they complete (the arrival is counted in the barrier's init count).
__device__ __forceinline__ void brain_fwt_mbar_arrive_cp_async(unsigned int bar) {
    asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];\n" ::"r"(bar) : "memory");
}
__device__ __forceinline__ void brain_fwt_mbar_wait(unsigned int bar, unsigned int parity) {
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "FWT_WAIT:\n"
        "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
        "@!p bra FWT_WAIT;\n"
        "}\n" ::"r"(bar), "r"(parity) : "memory");
}

// 2-D tile load of a `box` (128 bytes of K, 128 rows) at (k byte, row), completing on `bar`.
__device__ __forceinline__ void brain_fwt_tma_load(unsigned int dst, const void* map, int k, int row, unsigned int bar) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];\n"
        ::"r"(dst), "l"(map), "r"(k), "r"(row), "r"(bar) : "memory");
}

// Copy the template to shared memory, replace the address, dimensions and row stride, and publish the map to global memory (one warp).
__device__ __forceinline__ void brain_fwt_build_map(unsigned long long* smem_map, unsigned long long* gmem_map, const void* base, unsigned int cols, unsigned int rows, unsigned int lane) {
    if (lane < 16u) { smem_map[lane] = brain_fwt_template[lane]; }
    __syncwarp();
    if (lane == 0u) {
        const unsigned int s = static_cast<unsigned int>(__cvta_generic_to_shared(smem_map));
        asm volatile("tensormap.replace.tile.global_address.shared::cta.b1024.b64 [%0], %1;\n" ::"r"(s), "l"(base) : "memory");
        asm volatile("tensormap.replace.tile.global_dim.shared::cta.b1024.b32 [%0], 0, %1;\n" ::"r"(s), "r"(cols) : "memory");
        asm volatile("tensormap.replace.tile.global_dim.shared::cta.b1024.b32 [%0], 1, %1;\n" ::"r"(s), "r"(rows) : "memory");
        asm volatile("tensormap.replace.tile.global_stride.shared::cta.b1024.b64 [%0], 0, %1;\n" ::"r"(s), "l"(static_cast<unsigned long long>(cols)) : "memory");
    }
    __syncwarp();
    const unsigned int s = static_cast<unsigned int>(__cvta_generic_to_shared(smem_map));
    asm volatile("tensormap.cp_fenceproxy.global.shared::cta.tensormap::generic.release.gpu.sync.aligned [%0], [%1], 128;\n" ::"l"(gmem_map), "r"(s) : "memory");
    asm volatile("fence.proxy.tensormap::generic.acquire.gpu [%0], 128;\n" ::"l"(gmem_map) : "memory");
}

// Tile order: the blocks that run together cover BRAIN_FWT_GROUP_M row tiles of a few weight tiles, so the activation rows and the weight
// rows they read stay in the L2 whatever M is (a plain column-major walk of a 64-row-tile problem keeps every activation tile live).
#ifndef BRAIN_FWT_GROUP_M
#define BRAIN_FWT_GROUP_M 16
#endif
__device__ __forceinline__ void brain_fwt_tile(unsigned int tile, unsigned int tiles_m, unsigned int tiles_n, unsigned int& m0, unsigned int& n0) {
    const unsigned int per_group = BRAIN_FWT_GROUP_M * tiles_n;
    const unsigned int group = tile / per_group;
    const unsigned int first_m = group * BRAIN_FWT_GROUP_M;
    const unsigned int rows = min(tiles_m - first_m, (unsigned int)BRAIN_FWT_GROUP_M);
    const unsigned int within = tile - group * per_group;
    m0 = (first_m + within % rows) * BRAIN_FWT_BM;
    n0 = (within / rows) * BRAIN_FWT_BN;
}

extern "C" __global__ void __launch_bounds__(BRAIN_FWT_THREADS, 1)
BRAIN_FWT_ENTRY(const unsigned int* params, const unsigned int* xq, const unsigned int* wq, const float* sx, const float* sw, float* out) {
    BRAIN_PDL_ENTRY();
    constexpr int STAGES = BRAIN_FWT_STAGES;
    extern __shared__ unsigned char brain_fwt_smem_raw[];
    __shared__ __align__(128) unsigned long long brain_fwt_smem_maps[2][16];
    const unsigned int M = params[0];
    const unsigned int K = params[1] * 4u;
    const unsigned int N = params[2];
    const unsigned int kblocks = K / 128u;
    const unsigned int tiles_m = (M + BRAIN_FWT_BM - 1u) / BRAIN_FWT_BM;
    const unsigned int tiles_n = (N + BRAIN_FWT_BN - 1u) / BRAIN_FWT_BN;
    const unsigned int tiles = tiles_m * tiles_n;

    const unsigned int tid = threadIdx.x;
    const unsigned int wg = tid / 128;  // 0: producer; 1, 2: consumers of rows 64 * (wg - 1) ..
    const unsigned int lane = tid & 31u;
    const unsigned int warp = (tid >> 5) & 3u;

    const unsigned int raw_addr = brain_fwt_smem(brain_fwt_smem_raw);
    const unsigned int base = (raw_addr + 1023u) & ~1023u;
    unsigned char* const gen = brain_fwt_smem_raw + (base - raw_addr);
    const unsigned int full_bar = base + STAGES * BRAIN_FWT_STAGE_BYTES;  // STAGES mbarriers, then STAGES more
    const unsigned int empty_bar = full_bar + 8u * STAGES;

    if (tid == 0u) {
        for (int s = 0; s < STAGES; ++s) {
            brain_fwt_mbar_init(full_bar + 8u * s, 33u);    // the TMA thread's expect_tx arrival and the 32 lanes' cp.async arrivals
            brain_fwt_mbar_init(empty_bar + 8u * s, 256u);  // the consumers' threads
        }
        asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    }
    __syncthreads();

    if (wg == 0u) {
        // The producer needs few registers: its share goes to the consumers.
        asm volatile("setmaxnreg.dec.sync.aligned.u32 40;\n" ::: "memory");
        if (warp != 0u) { return; }
        // One slot per SM: a block owns its SM's shared memory (one resident block), so no two live blocks, of this launch or of another stream's, share a slot.
        unsigned int smid;
        asm("mov.u32 %0, %%smid;\n" : "=r"(smid));
        unsigned long long* const gmaps = &brain_fwt_maps[smid % BRAIN_FWT_MAX_CTAS][0][0];
        brain_fwt_build_map(&brain_fwt_smem_maps[0][0], gmaps, xq, K, M, lane);
        brain_fwt_build_map(&brain_fwt_smem_maps[1][0], gmaps + 16, wq, K, N, lane);
        unsigned int it = 0;
        for (unsigned int tile = blockIdx.x; tile < tiles; tile += gridDim.x) {
            unsigned int m0, n0;
            brain_fwt_tile(tile, tiles_m, tiles_n, m0, n0);
            const float* sx_src[4];
            bool sx_ok[4];
#pragma unroll
            for (int h = 0; h < 4; ++h) {
                const unsigned int r = m0 + lane + 32u * h;
                sx_ok[h] = r < M;
                sx_src[h] = sx + (unsigned long long)(sx_ok[h] ? r : 0u) * kblocks;
            }
            const float* const sw_src = sw + (unsigned long long)(n0 >> 7) * kblocks;
            for (unsigned int kt = 0; kt < kblocks; ++kt, ++it) {
                const unsigned int slot = it % STAGES;
                if (it >= (unsigned int)STAGES) { brain_fwt_mbar_wait(empty_bar + 8u * slot, ((it / STAGES) - 1u) & 1u); }
                const unsigned int st = base + slot * BRAIN_FWT_STAGE_BYTES;
                if (lane == 0u) {
                    brain_fwt_mbar_arrive_expect_tx(full_bar + 8u * slot, BRAIN_FWT_X_BYTES + BRAIN_FWT_W_BYTES);
                    brain_fwt_tma_load(st, gmaps, kt * 128, m0, full_bar + 8u * slot);
                    brain_fwt_tma_load(st + BRAIN_FWT_X_BYTES, gmaps + 16, kt * 128, n0, full_bar + 8u * slot);
                }
#pragma unroll
                for (int h = 0; h < 4; ++h) { brain_fwt_cp4(st + BRAIN_FWT_X_BYTES + BRAIN_FWT_W_BYTES + (lane + 32u * h) * 4u, sx_src[h] + kt, sx_ok[h] ? 4u : 0u); }
                if (lane == 0u) { brain_fwt_cp4(st + BRAIN_FWT_X_BYTES + BRAIN_FWT_W_BYTES + 512u, sw_src + kt, 4u); }
                brain_fwt_mbar_arrive_cp_async(full_bar + 8u * slot);
            }
        }
        brain_fwt_wait<0>();
        return;
    }

    asm volatile("setmaxnreg.inc.sync.aligned.u32 232;\n" ::: "memory");
    const unsigned int cwg = wg - 1u;  // rows of x 64 * cwg .. 64 * cwg + 63
    const unsigned int frow = cwg * 64u + warp * 16u + (lane >> 2);
    const unsigned int fcol = (lane & 3u) * 2u;
    unsigned int it = 0;
    for (unsigned int tile = blockIdx.x; tile < tiles; tile += gridDim.x) {
        unsigned int m0, n0;
        brain_fwt_tile(tile, tiles_m, tiles_n, m0, n0);
        float acc[64], d[64];
#pragma unroll
        for (int i = 0; i < 64; ++i) { acc[i] = 0.0f; d[i] = 0.0f; }
        for (unsigned int kt = 0; kt < kblocks; ++kt, ++it) {
            const unsigned int slot = it % STAGES;
            brain_fwt_mbar_wait(full_bar + 8u * slot, (it / STAGES) & 1u);
            const unsigned int st = base + slot * BRAIN_FWT_STAGE_BYTES;
            const unsigned int a_base = st + cwg * (64u * 128u);
            const unsigned int b_base = st + BRAIN_FWT_X_BYTES;
            brain_fwt_fence();
#pragma unroll
            for (int q = 0; q < 4; ++q) { brain_fwt_wgmma(d, brain_fwt_desc(a_base + q * 32u), brain_fwt_desc(b_base + q * 32u), q > 0 ? 1 : 0); }
            brain_fwt_wg_commit();
            brain_fwt_wg_wait<0>();

            const float* const sxs = reinterpret_cast<const float*>(gen + slot * BRAIN_FWT_STAGE_BYTES + BRAIN_FWT_X_BYTES + BRAIN_FWT_W_BYTES);
            const float wscale = sxs[128];
            const float s0 = sxs[frow] * wscale, s1 = sxs[frow + 8u] * wscale;
            brain_fwt_mbar_arrive(empty_bar + 8u * slot);  // the slot's tiles and scales are read
#pragma unroll
            for (int j = 0; j < 16; ++j) {
                acc[4 * j] = fmaf(d[4 * j], s0, acc[4 * j]);
                acc[4 * j + 1] = fmaf(d[4 * j + 1], s0, acc[4 * j + 1]);
                acc[4 * j + 2] = fmaf(d[4 * j + 2], s1, acc[4 * j + 2]);
                acc[4 * j + 3] = fmaf(d[4 * j + 3], s1, acc[4 * j + 3]);
            }
        }

        // out[m, n] = acc: thread layout of the m64n128 accumulator (rows r and r + 8 of the warp's 16, columns 8 j + 2 t, +1).
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
}
