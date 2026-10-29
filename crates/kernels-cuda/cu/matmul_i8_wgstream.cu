// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for memory-bandwidth-bound LLM
// decode on GPUs for its clients. If your team needs expertise in streaming
// quantised weights through Hopper warpgroup tensor cores at the speed of the
// memory system then you can procure our services by sending an email to
// info@swedishembedded.com.
//
// Weight-streaming INT8 matmul for 33 to 128 rows of x on Hopper warpgroup tensor cores, the same buffers and uniform
// as `matmul_i8_dyn` and `matmul_i8_stream`:
//
//   params : u32 [m, kg, n]   kg = K/4 packed words per row, K a multiple of 128
//   x_q    : [M, kg] u32      activations, 4 signed int8 per word (16-byte aligned)
//   w_q    : [N, kg] u32      weights, 4 signed int8 per word (16-byte aligned)
//   sx     : [M, K/32] f32    group-wise activation scale
//   sw     : [N, kg/8] f32    per-32-element-group weight scale
//   out    : [M, N]  f32      out[m, n] = sum_k x_q[m,k] * w_q[n,k] * (sw[n, k/32] * sx[m, k/32])
//
// Why a warpgroup kernel for the large decode batch
// -------------------------------------------------
// A decode step of 33-128 sequences still reads every weight once, so the figure of merit is the fraction of the
// memory system's read bandwidth the weights move at. `matmul_i8_stream` (mma.sync) stops at 1.0-1.3 TB/s from 48
// rows: every warp loads its operands with `ldmatrix`, issues the MMA and folds the result, and the three compete for
// one issue slot. Here the weights are the WGMMA's A operand (64 weight rows a warpgroup, read from shared memory by
// descriptor) and the batch of x is its B operand (N = BRAIN_WGS_N rows, one candidate per width), so the tensor
// cores run asynchronously and the issue slots belong to the group-scale fold alone, two fused multiply-adds per
// output (knowledge 372): fma(d, xa, -bias * xa) is the group's integer sum times the activation scale, one more
// folds the weight scale into the running f32 total. The accumulators are seeded with 1.5 * 2^23 so the integer sum
// comes back as an exact float, and the activation scales are rounded to 21 bits (once per stage, by the block) so
// that `bias * xa` is exact.
//
// Data movement: a ring of BRAIN_WGS_STAGES stages of 128 bytes of K per row (cp.async, 128-byte swizzle the
// descriptors name), each holding a 128-row weight tile, the x tile and the raw group scales of both. The prepared
// activation scales of stage kt + 1 are written while stage kt computes, so the ring keeps one stage fewer in flight
// than it holds. K is split over a thread-block cluster when there are too few weight rows to occupy every SM: each
// block streams its stretch of K and the partial sums are added in rank order through distributed shared memory, so
// the result does not depend on scheduling.
//
// Row tails: weight rows past n read the last row and are dropped; rows of x past m read the last row and are
// dropped. Requires K % 128 == 0 and `sm_90a`.
//
// No `__restrict__`: brain's device buffers alias by design.

#ifndef BRAIN_WGS_N
#define BRAIN_WGS_N 64
#endif
#ifndef BRAIN_WGS_STAGES
#define BRAIN_WGS_STAGES 4
#endif
#ifndef BRAIN_WGS_MIN_BLOCKS
#define BRAIN_WGS_MIN_BLOCKS 2
#endif
// Warpgroups of a block; each owns 64 weight rows.
#ifndef BRAIN_WGS_WGS
#define BRAIN_WGS_WGS 2
#endif

#define BRAIN_WGS_THREADS (128 * BRAIN_WGS_WGS)
#define BRAIN_WGS_BM (64 * BRAIN_WGS_WGS)   // weight rows of a block
// Rows a pass of the block's threads copies (8 sixteen-byte chunks of a row each).
#define BRAIN_WGS_RP (BRAIN_WGS_THREADS / 8)
#define BRAIN_WGS_KB 128          // bytes of K a stage holds per row
#define BRAIN_WGS_W_BYTES (BRAIN_WGS_BM * BRAIN_WGS_KB)
#define BRAIN_WGS_X_BYTES (BRAIN_WGS_N * BRAIN_WGS_KB)
#define BRAIN_WGS_SW_BYTES (BRAIN_WGS_BM * 16)
#define BRAIN_WGS_SX_BYTES (BRAIN_WGS_N * 16)
#define BRAIN_WGS_STAGE_BYTES (BRAIN_WGS_W_BYTES + BRAIN_WGS_X_BYTES + BRAIN_WGS_SW_BYTES + BRAIN_WGS_SX_BYTES)
// Prepared activation scales: per buffer and scale group of the stage one float4 {xa, xb, -bias * xa, -bias * xb} for
// each pair of rows of x. Two buffers: the stage being computed and the next.
#define BRAIN_WGS_XP_PAIRS (BRAIN_WGS_N / 2)
#define BRAIN_WGS_XP_BYTES (2 * 4 * BRAIN_WGS_XP_PAIRS * 16)
#define BRAIN_WGS_ACC (BRAIN_WGS_N / 2)       // accumulator registers per thread of a m64nN tile
#define BRAIN_WGS_RED_STRIDE (BRAIN_WGS_BM + 4)

#define BRAIN_WGS_BIAS_BITS 0x4B400000u       // 1.5 * 2^23
#define BRAIN_WGS_BIAS 12582912.0f

#ifndef BRAIN_PDL_ENTRY
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
#define BRAIN_PDL_ENTRY() do { asm volatile("griddepcontrol.wait;\n" ::: "memory"); asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory"); } while (0)
#else
#define BRAIN_PDL_ENTRY() do {} while (0)
#endif
#endif

__device__ __forceinline__ unsigned int brain_wgs_smem(const void* p) { return static_cast<unsigned int>(__cvta_generic_to_shared(p)); }

__device__ __forceinline__ void brain_wgs_cp16(unsigned int dst, const void* src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(dst), "l"(src) : "memory");
}
__device__ __forceinline__ void brain_wgs_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
template <int N>
__device__ __forceinline__ void brain_wgs_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N) : "memory"); }

// Shared-memory matrix descriptor: 128-byte swizzle, K-major, 8-row groups 1024 B apart.
__device__ __forceinline__ unsigned long long brain_wgs_desc(unsigned int smem_addr) {
    return static_cast<unsigned long long>((smem_addr & 0x3FFFFu) >> 4) | (1ull << 16) | (64ull << 32) | (1ull << 62);
}
__device__ __forceinline__ void brain_wgs_fence() { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void brain_wgs_wgmma_commit() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
template <int N>
__device__ __forceinline__ void brain_wgs_wgmma_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(N) : "memory"); }
__device__ __forceinline__ void brain_wgs_fence_proxy_async() { asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory"); }

// D (64 x N s32, N / 2 registers per thread) += A (64 x 32 s8) . B (N x 32 s8)^T, one function per width.
#if BRAIN_WGS_N == 48
__device__ __forceinline__ void brain_wgmma_s8(unsigned int (&d)[24], unsigned long long da, unsigned long long db) {
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, 1, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n48k32.s32.s8.s8 "
        "{"
        "%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, "
        "%19, %20, %21, %22, %23"
        "}, "
        "%24, %25, p;\n"
        "}\n"
        :
          "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3]), "+r"(d[4]), "+r"(d[5]), "+r"(d[6]), "+r"(d[7]),
          "+r"(d[8]), "+r"(d[9]), "+r"(d[10]), "+r"(d[11]), "+r"(d[12]), "+r"(d[13]), "+r"(d[14]),
          "+r"(d[15]), "+r"(d[16]), "+r"(d[17]), "+r"(d[18]), "+r"(d[19]), "+r"(d[20]), "+r"(d[21]),
          "+r"(d[22]), "+r"(d[23])
        : "l"(da), "l"(db));
}
#endif

#if BRAIN_WGS_N == 64
__device__ __forceinline__ void brain_wgmma_s8(unsigned int (&d)[32], unsigned long long da, unsigned long long db) {
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, 1, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n64k32.s32.s8.s8 "
        "{"
        "%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, "
        "%19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31"
        "}, "
        "%32, %33, p;\n"
        "}\n"
        :
          "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3]), "+r"(d[4]), "+r"(d[5]), "+r"(d[6]), "+r"(d[7]),
          "+r"(d[8]), "+r"(d[9]), "+r"(d[10]), "+r"(d[11]), "+r"(d[12]), "+r"(d[13]), "+r"(d[14]),
          "+r"(d[15]), "+r"(d[16]), "+r"(d[17]), "+r"(d[18]), "+r"(d[19]), "+r"(d[20]), "+r"(d[21]),
          "+r"(d[22]), "+r"(d[23]), "+r"(d[24]), "+r"(d[25]), "+r"(d[26]), "+r"(d[27]), "+r"(d[28]),
          "+r"(d[29]), "+r"(d[30]), "+r"(d[31])
        : "l"(da), "l"(db));
}
#endif

#if BRAIN_WGS_N == 96
__device__ __forceinline__ void brain_wgmma_s8(unsigned int (&d)[48], unsigned long long da, unsigned long long db) {
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, 1, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n96k32.s32.s8.s8 "
        "{"
        "%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, "
        "%19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, "
        "%36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47"
        "}, "
        "%48, %49, p;\n"
        "}\n"
        :
          "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3]), "+r"(d[4]), "+r"(d[5]), "+r"(d[6]), "+r"(d[7]),
          "+r"(d[8]), "+r"(d[9]), "+r"(d[10]), "+r"(d[11]), "+r"(d[12]), "+r"(d[13]), "+r"(d[14]),
          "+r"(d[15]), "+r"(d[16]), "+r"(d[17]), "+r"(d[18]), "+r"(d[19]), "+r"(d[20]), "+r"(d[21]),
          "+r"(d[22]), "+r"(d[23]), "+r"(d[24]), "+r"(d[25]), "+r"(d[26]), "+r"(d[27]), "+r"(d[28]),
          "+r"(d[29]), "+r"(d[30]), "+r"(d[31]), "+r"(d[32]), "+r"(d[33]), "+r"(d[34]), "+r"(d[35]),
          "+r"(d[36]), "+r"(d[37]), "+r"(d[38]), "+r"(d[39]), "+r"(d[40]), "+r"(d[41]), "+r"(d[42]),
          "+r"(d[43]), "+r"(d[44]), "+r"(d[45]), "+r"(d[46]), "+r"(d[47])
        : "l"(da), "l"(db));
}
#endif

#if BRAIN_WGS_N == 128
__device__ __forceinline__ void brain_wgmma_s8(unsigned int (&d)[64], unsigned long long da, unsigned long long db) {
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, 1, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n128k32.s32.s8.s8 "
        "{"
        "%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, "
        "%19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, "
        "%36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, "
        "%53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63"
        "}, "
        "%64, %65, p;\n"
        "}\n"
        :
          "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3]), "+r"(d[4]), "+r"(d[5]), "+r"(d[6]), "+r"(d[7]),
          "+r"(d[8]), "+r"(d[9]), "+r"(d[10]), "+r"(d[11]), "+r"(d[12]), "+r"(d[13]), "+r"(d[14]),
          "+r"(d[15]), "+r"(d[16]), "+r"(d[17]), "+r"(d[18]), "+r"(d[19]), "+r"(d[20]), "+r"(d[21]),
          "+r"(d[22]), "+r"(d[23]), "+r"(d[24]), "+r"(d[25]), "+r"(d[26]), "+r"(d[27]), "+r"(d[28]),
          "+r"(d[29]), "+r"(d[30]), "+r"(d[31]), "+r"(d[32]), "+r"(d[33]), "+r"(d[34]), "+r"(d[35]),
          "+r"(d[36]), "+r"(d[37]), "+r"(d[38]), "+r"(d[39]), "+r"(d[40]), "+r"(d[41]), "+r"(d[42]),
          "+r"(d[43]), "+r"(d[44]), "+r"(d[45]), "+r"(d[46]), "+r"(d[47]), "+r"(d[48]), "+r"(d[49]),
          "+r"(d[50]), "+r"(d[51]), "+r"(d[52]), "+r"(d[53]), "+r"(d[54]), "+r"(d[55]), "+r"(d[56]),
          "+r"(d[57]), "+r"(d[58]), "+r"(d[59]), "+r"(d[60]), "+r"(d[61]), "+r"(d[62]), "+r"(d[63])
        : "l"(da), "l"(db));
}
#endif

__device__ __forceinline__ unsigned int brain_wgs_cluster_rank() {
    unsigned int r;
    asm volatile("mov.u32 %0, %%cluster_ctarank;\n" : "=r"(r));
    return r;
}
__device__ __forceinline__ unsigned int brain_wgs_cluster_size() {
    unsigned int r;
    asm volatile("mov.u32 %0, %%cluster_nctarank;\n" : "=r"(r));
    return r;
}
__device__ __forceinline__ void brain_wgs_cluster_sync() {
    asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;\n" ::: "memory");
}
__device__ __forceinline__ float brain_wgs_remote_ld(unsigned int local_addr, unsigned int rank) {
    unsigned int remote;
    float v;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;\n" : "=r"(remote) : "r"(local_addr), "r"(rank));
    asm volatile("ld.shared::cluster.f32 %0, [%1];\n" : "=f"(v) : "r"(remote) : "memory");
    return v;
}

// The activation scale rounded to 21 significant bits, so that `bias * scale` is exact (the bias has two significant
// bits). The relative change is at most 2^-21 = 4.8e-7, under the f32 rounding noise of the running sum.
__device__ __forceinline__ float brain_wgs_round21(float x) { return __uint_as_float((__float_as_uint(x) + 3u) & ~7u); }

// Prepare the activation scales of one stage from the raw [row][group] copy (`sxs`) into `xp` ([group][pair]).
__device__ __forceinline__ void brain_wgs_prepare(const float* sxs, float4* xp) {
    for (unsigned int c = threadIdx.x; c < 4u * BRAIN_WGS_XP_PAIRS; c += BRAIN_WGS_THREADS) {
        const unsigned int q = c / BRAIN_WGS_XP_PAIRS, pair = c % BRAIN_WGS_XP_PAIRS;
        const float xa = brain_wgs_round21(sxs[(2u * pair) * 4u + q]);
        const float xb = brain_wgs_round21(sxs[(2u * pair + 1u) * 4u + q]);
        xp[c] = make_float4(xa, xb, -BRAIN_WGS_BIAS * xa, -BRAIN_WGS_BIAS * xb);
    }
}

extern "C" __global__ void __launch_bounds__(BRAIN_WGS_THREADS, BRAIN_WGS_MIN_BLOCKS)
brain_matmul_i8_wgstream(const unsigned int* params, const unsigned int* xq, const unsigned int* wq, const float* sx, const float* sw, float* out) {
    BRAIN_PDL_ENTRY();
    extern __shared__ __align__(1024) unsigned char smem_raw[];
    const unsigned int m = params[0];
    const unsigned int K = params[1] * 4u;
    const unsigned int n = params[2];
    const unsigned int ng = K / 32u;
    const unsigned int ktiles = K / BRAIN_WGS_KB;
    const unsigned int csize = brain_wgs_cluster_size();
    const unsigned int rank = brain_wgs_cluster_rank();
    // A cluster's blocks are consecutive and share a tile of weight rows.
    const unsigned int blk = blockIdx.y * gridDim.x + blockIdx.x;
    const unsigned int n0 = (blk / csize) * BRAIN_WGS_BM;
    if (n0 >= n || m == 0u) { return; }  // cluster-uniform

    const unsigned int tid = threadIdx.x;
    const unsigned int wg = tid >> 7;
    const unsigned int lane = tid & 31u;
    const unsigned int warp = (tid >> 5) & 3u;
    const unsigned int t4 = lane & 3u;

    // 1024-byte aligned base: the 128-byte swizzle pattern repeats every 1024 bytes.
    const unsigned int raw_addr = brain_wgs_smem(smem_raw);
    const unsigned int base = (raw_addr + 1023u) & ~1023u;
    unsigned char* const gen = smem_raw + (base - raw_addr);
    float4* const xp_base = reinterpret_cast<float4*>(gen + BRAIN_WGS_STAGES * BRAIN_WGS_STAGE_BYTES);

    // This block streams the contiguous stretch [first, first + count) of the K tiles.
    const unsigned int per = (ktiles + csize - 1u) / csize;
    const unsigned int first = min(rank * per, ktiles);
    const unsigned int count = min(per, ktiles - first);

    const unsigned char* const xb = reinterpret_cast<const unsigned char*>(xq);
    const unsigned char* const wb = reinterpret_cast<const unsigned char*>(wq);
    const unsigned long long kbytes = K;

    // Source rows of the 16-byte chunks this thread copies per tile (chunk = tid + THREADS it: row tid / 8 + RP it, chunk
    // tid % 8), computed once.
    const unsigned int crow = tid >> 3, cc = tid & 7u;
    const unsigned int c_dst = crow * 128u + ((cc ^ (crow & 7u)) << 4);   // + 128 RP it: RP rows keep row mod 8
    constexpr int W_PASSES = BRAIN_WGS_BM / BRAIN_WGS_RP;
    constexpr int X_PASSES = (BRAIN_WGS_N + BRAIN_WGS_RP - 1) / BRAIN_WGS_RP;
    const unsigned char* w_src[W_PASSES];
    const unsigned char* x_src[X_PASSES];
#pragma unroll
    for (int it = 0; it < W_PASSES; ++it) { w_src[it] = wb + (unsigned long long)min(n0 + crow + BRAIN_WGS_RP * it, n - 1u) * kbytes + cc * 16u; }
#pragma unroll
    for (int it = 0; it < X_PASSES; ++it) { x_src[it] = xb + (unsigned long long)min(crow + BRAIN_WGS_RP * it, m - 1u) * kbytes + cc * 16u; }

    // Tile kt (a global tile index) into ring slot `slot`: the two int8 tiles and the tile's four group scales per row.
    auto load_stage = [&](unsigned int slot, unsigned int kt) {
        const unsigned int st = base + slot * BRAIN_WGS_STAGE_BYTES;
        const unsigned int kbyte = kt * BRAIN_WGS_KB;
#pragma unroll
        for (int it = 0; it < W_PASSES; ++it) { brain_wgs_cp16(st + c_dst + 128u * BRAIN_WGS_RP * it, w_src[it] + kbyte); }
#pragma unroll
        for (int it = 0; it < X_PASSES; ++it) {
            if (crow + BRAIN_WGS_RP * it < BRAIN_WGS_N) { brain_wgs_cp16(st + BRAIN_WGS_W_BYTES + c_dst + 128u * BRAIN_WGS_RP * it, x_src[it] + kbyte); }
        }
        // The four group scales (16 bytes) of each weight row, then of each row of x.
        for (unsigned int item = tid; item < BRAIN_WGS_BM + BRAIN_WGS_N; item += BRAIN_WGS_THREADS) {
            const bool is_x = item >= BRAIN_WGS_BM;
            const unsigned int row = is_x ? item - BRAIN_WGS_BM : item;
            const float* src = is_x ? sx + (unsigned long long)min(row, m - 1u) * ng : sw + (unsigned long long)min(n0 + row, n - 1u) * ng;
            brain_wgs_cp16(st + BRAIN_WGS_W_BYTES + BRAIN_WGS_X_BYTES + (is_x ? BRAIN_WGS_SW_BYTES : 0u) + row * 16u, src + kt * 4u);
        }
    };

    float f[BRAIN_WGS_ACC];
#pragma unroll
    for (int i = 0; i < BRAIN_WGS_ACC; ++i) { f[i] = 0.0f; }
    unsigned int acc[BRAIN_WGS_ACC];
#pragma unroll
    for (int i = 0; i < BRAIN_WGS_ACC; ++i) { acc[i] = BRAIN_WGS_BIAS_BITS; }

    // The prefetch distance is one stage short of the ring: the scales of the next stage are prepared while a stage
    // computes, so stages kt and kt + 1 have landed when tile kt starts. Tile 0 is prepared here.
#pragma unroll
    for (int s = 0; s < BRAIN_WGS_STAGES - 1; ++s) {
        if (s < count) { load_stage(s, first + s); }
        brain_wgs_commit();
    }
    brain_wgs_wait<BRAIN_WGS_STAGES - 2>();
    brain_wgs_fence_proxy_async();
    __syncthreads();
    if (count > 0) { brain_wgs_prepare(reinterpret_cast<const float*>(gen + BRAIN_WGS_W_BYTES + BRAIN_WGS_X_BYTES + BRAIN_WGS_SW_BYTES), xp_base); }

    // The fragment rows this thread owns: weight rows r and r + 8 of the warp's 16-row slice of the warpgroup's 64.
    const unsigned int wrow = wg * 64u + warp * 16u + (lane >> 2);

    for (unsigned int kt = 0; kt < count; ++kt) {
        brain_wgs_wait<BRAIN_WGS_STAGES - 3>();
        brain_wgs_fence_proxy_async();
        __syncthreads();  // tiles kt and kt + 1 are visible to every warp and so are the prepared scales of kt; every warp is done with tile kt - 1, whose slot the next load takes
        const unsigned int nxt = kt + BRAIN_WGS_STAGES - 1;
        if (nxt < count) { load_stage(nxt % BRAIN_WGS_STAGES, first + nxt); }
        brain_wgs_commit();
        if (kt + 1 < count) {
            const unsigned int ns = (kt + 1) % BRAIN_WGS_STAGES;
            brain_wgs_prepare(reinterpret_cast<const float*>(gen + ns * BRAIN_WGS_STAGE_BYTES + BRAIN_WGS_W_BYTES + BRAIN_WGS_X_BYTES + BRAIN_WGS_SW_BYTES),
                              xp_base + ((kt + 1) & 1u) * (4 * BRAIN_WGS_XP_PAIRS));
        }

        const unsigned int slot = kt % BRAIN_WGS_STAGES;
        const unsigned int a_base = base + slot * BRAIN_WGS_STAGE_BYTES + wg * (64u * BRAIN_WGS_KB);
        const unsigned int b_base = base + slot * BRAIN_WGS_STAGE_BYTES + BRAIN_WGS_W_BYTES;
        const float* sws = reinterpret_cast<const float*>(gen + slot * BRAIN_WGS_STAGE_BYTES + BRAIN_WGS_W_BYTES + BRAIN_WGS_X_BYTES);
        const float4 swl = *reinterpret_cast<const float4*>(sws + wrow * 4u);
        const float4 swh = *reinterpret_cast<const float4*>(sws + (wrow + 8u) * 4u);
        const float4* xp = xp_base + (kt & 1u) * (4 * BRAIN_WGS_XP_PAIRS);
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            brain_wgs_fence();
            brain_wgmma_s8(acc, brain_wgs_desc(a_base + q * 32u), brain_wgs_desc(b_base + q * 32u));
            brain_wgs_wgmma_commit();
            const float sl = q == 0 ? swl.x : q == 1 ? swl.y : q == 2 ? swl.z : swl.w;
            const float sh = q == 0 ? swh.x : q == 1 ? swh.y : q == 2 ? swh.z : swh.w;
            brain_wgs_wgmma_wait<0>();
#pragma unroll
            for (int j = 0; j < BRAIN_WGS_N / 8; ++j) {
                const float4 xv = xp[q * BRAIN_WGS_XP_PAIRS + 4 * j + t4];
                f[4 * j + 0] = fmaf(fmaf(__uint_as_float(acc[4 * j + 0]), xv.x, xv.z), sl, f[4 * j + 0]);
                f[4 * j + 1] = fmaf(fmaf(__uint_as_float(acc[4 * j + 1]), xv.y, xv.w), sl, f[4 * j + 1]);
                f[4 * j + 2] = fmaf(fmaf(__uint_as_float(acc[4 * j + 2]), xv.x, xv.z), sh, f[4 * j + 2]);
                f[4 * j + 3] = fmaf(fmaf(__uint_as_float(acc[4 * j + 3]), xv.y, xv.w), sh, f[4 * j + 3]);
            }
#pragma unroll
            for (int i = 0; i < BRAIN_WGS_ACC; ++i) { acc[i] = BRAIN_WGS_BIAS_BITS; }
        }
    }
    brain_wgs_wait<0>();

    if (csize == 1u) {
#pragma unroll
        for (int j = 0; j < BRAIN_WGS_N / 8; ++j) {
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                const unsigned int row = n0 + wrow + 8u * (q >> 1);
                const unsigned int col = 8u * j + 2u * t4 + (q & 1u);
                if (row < n && col < m) { out[(unsigned long long)col * n + row] = f[4 * j + q]; }
            }
        }
        return;
    }

    // Split K: park this block's partial sums, then each block adds the partials of every block of the cluster, in rank
    // order, for its share of the outputs.
    __syncthreads();  // every warp is done with the stages the buffer aliases
    float* red = reinterpret_cast<float*>(gen);
#pragma unroll
    for (int j = 0; j < BRAIN_WGS_N / 8; ++j) {
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            const unsigned int row = wrow + 8u * (q >> 1);
            const unsigned int col = 8u * j + 2u * t4 + (q & 1u);
            red[col * BRAIN_WGS_RED_STRIDE + row] = f[4 * j + q];
        }
    }
    brain_wgs_cluster_sync();
    const unsigned int red_addr = brain_wgs_smem(red);
    for (unsigned int idx = rank * BRAIN_WGS_THREADS + tid; idx < m * BRAIN_WGS_BM; idx += csize * BRAIN_WGS_THREADS) {
        const unsigned int col = idx / BRAIN_WGS_BM, row = idx % BRAIN_WGS_BM;
        const unsigned int a = red_addr + (col * BRAIN_WGS_RED_STRIDE + row) * 4u;
        float total = 0.0f;
        for (unsigned int r = 0; r < csize; ++r) { total += brain_wgs_remote_ld(a, r); }
        if (n0 + row < n) { out[(unsigned long long)col * n + n0 + row] = total; }
    }
    brain_wgs_cluster_sync();  // no block leaves while a peer may still read its partials
}
