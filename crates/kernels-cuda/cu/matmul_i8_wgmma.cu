// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for prompt-processing throughput
// on quantised language models for its clients. If your team needs expertise
// in putting int8 weights onto Hopper warpgroup tensor cores without giving up
// their group-wise scales, then you can procure our services by sending an
// email to info@swedishembedded.com.
//
// out = x_q . W_q^T with GROUP-WISE activation and weight scales on Hopper
// WARPGROUP tensor cores (`wgmma.mma_async` m64n128k32, s8 x s8 -> s32). The
// same argument contract and arithmetic as `matmul_i8_mma.cu`:
//
//   params : [M, KG, N] u32   KG = K/4 (packed int8 words along K)
//   x_q    : [M, K/4]   u32   4 int8 activations per word, row-major
//   w_q    : [N, K/4]   u32   4 int8 weights per word, row-major
//   sx     : [M, K/32]  f32   per-(row, 32-element group) activation scale
//   sw     : [N, K/32]  f32   per-(row, 32-element group) weight scale
//   out    : [M, N]     f32   out[m,n] = sum_g (sw[n,g] * sx[m,g]) * (sum_{k in g} xq[m,k]*wq[n,k])
//
// Requires K % 128 == 0 (one 128-byte swizzled row per k-tile) and `sm_90a`
// (registered with `ArchFeatures::Required`; a device below it is never
// offered the kernel).
//
// Why a warpgroup kernel, and what bounds it
// ------------------------------------------
// `mma.sync` makes every warp load its own operands with `ldmatrix`, issue
// the MMA and fold the result: three kinds of instruction compete for one
// issue slot, and the kernel stops near 215 TOPS of the part's 1979. A
// `wgmma` reads both operands from shared memory by descriptor and runs
// asynchronously, so the issue slots belong to the group-scale fold alone.
// The fold is irreducible for GGUF Q8_0 (one weight scale per 32 elements,
// one activation scale per 32 elements, both varying along K): every output
// element needs its own int32 sum of every k32 group scaled and added to an
// f32 total. At four FP32 instructions per element per group
// (seed, bias removal, scale product, fused add) the 128 FP32 lanes of an SM
// bound the kernel at 32 MACs x 128 / 4 = 1024 MAC/clk/SM, a quarter of the
// tensor-core rate; the tensor cores idle for the rest, so this kernel's
// compute roof is the fold's, not the MMA's.
//
// The fold: each group's int32 sum is produced by one wgmma into an
// accumulator SEEDED with 0x4B400000 (the float 1.5 * 2^23), so the sum i
// comes back as the bit pattern of the float 1.5 * 2^23 + i - exact for
// |i| < 2^22 and |i| <= 32 * 127 * 128 here - and one FADD removes the bias
// before one FMUL forms the scale product and one FFMA adds it to the running
// f32 total. The sum inside a group is exact integer arithmetic; across groups
// it is f32 in ascending group order, as in `matmul_i8_mma.cu` and agrees with
// it to f32 rounding of the running sum (the scale multiply-add is fused).
// A warpgroup waits for each group's wgmma before folding it; the tensor cores
// are kept busy by the OTHER warpgroup of the block (the fold is four times the
// MMA's time, so one warpgroup's fold covers the other's MMA), which needs one
// accumulator set of 64 registers instead of two and leaves room for the f32
// totals without spilling.
//
// Data movement: a 4-stage cp.async ring, prefetch distance 3 (the stage a
// load overwrites was read by tile kt - 1, whose groups every thread has
// retired before the barrier that precedes the load). Each
// stage holds the 128 x 128 activation tile and the 128 x 128 weight tile,
// both K-major with the 128-byte swizzle the wgmma descriptors name (16-byte
// chunk index XOR row mod 8), and the tile's group scales transposed to
// [group][row] so a thread's two adjacent columns are one 8-byte read.
//
// No `__restrict__`: brain's device buffers alias by design.

#define BM 128
#define BN 128
#define BK 128
#define THREADS 256
#define STAGES 4
#define PREFETCH 3

#define A_BYTES (BM * BK)
#define B_BYTES (BN * BK)
#define SS_FLOATS (4 * BN)                       // [weight row][group]
#define XS_FLOATS (4 * BM)                       // [activation row][group]
#define STAGE_BYTES (A_BYTES + B_BYTES + (SS_FLOATS + XS_FLOATS) * 4)   // 36864

#define FLOAT_BIAS_BITS 0x4B400000u              // 1.5 * 2^23
#define FLOAT_BIAS 12582912.0f

__device__ __forceinline__ unsigned int brain_smem_addr(const void* p) {
    return static_cast<unsigned int>(__cvta_generic_to_shared(p));
}

// Global -> shared copy, zero-filled when `bytes` is 0 (a row past the edge of the matrix). The pointer is
// always a valid address.
__device__ __forceinline__ void brain_cp16(unsigned int dst, const void* src, unsigned int bytes) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(bytes));
}
__device__ __forceinline__ void brain_cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void brain_cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

// Shared-memory matrix descriptor: 128-byte swizzle, K-major, 8-row groups 1024 B apart.
__device__ __forceinline__ unsigned long long brain_desc(unsigned int smem_addr) {
    return static_cast<unsigned long long>((smem_addr & 0x3FFFFu) >> 4) | (1ull << 16) | (64ull << 32) | (1ull << 62);
}

__device__ __forceinline__ void brain_wgmma_fence() { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void brain_wgmma_commit() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
template <int N>
__device__ __forceinline__ void brain_wgmma_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(N) : "memory"); }
__device__ __forceinline__ void brain_fence_proxy_async() { asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory"); }

// D (64 x 128 s32, 64 registers per thread) += A (64 x 32 s8) . B (128 x 32 s8)^T.
__device__ __forceinline__ void brain_wgmma_s8(unsigned int (&d)[64], unsigned long long da, unsigned long long db) {
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, 1, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n128k32.s32.s8.s8 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
        "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, "
        "%32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, "
        "%48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, "
        "%64, %65, p;\n"
        "}\n"
        : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3]), "+r"(d[4]), "+r"(d[5]), "+r"(d[6]), "+r"(d[7]),
          "+r"(d[8]), "+r"(d[9]), "+r"(d[10]), "+r"(d[11]), "+r"(d[12]), "+r"(d[13]), "+r"(d[14]), "+r"(d[15]),
          "+r"(d[16]), "+r"(d[17]), "+r"(d[18]), "+r"(d[19]), "+r"(d[20]), "+r"(d[21]), "+r"(d[22]), "+r"(d[23]),
          "+r"(d[24]), "+r"(d[25]), "+r"(d[26]), "+r"(d[27]), "+r"(d[28]), "+r"(d[29]), "+r"(d[30]), "+r"(d[31]),
          "+r"(d[32]), "+r"(d[33]), "+r"(d[34]), "+r"(d[35]), "+r"(d[36]), "+r"(d[37]), "+r"(d[38]), "+r"(d[39]),
          "+r"(d[40]), "+r"(d[41]), "+r"(d[42]), "+r"(d[43]), "+r"(d[44]), "+r"(d[45]), "+r"(d[46]), "+r"(d[47]),
          "+r"(d[48]), "+r"(d[49]), "+r"(d[50]), "+r"(d[51]), "+r"(d[52]), "+r"(d[53]), "+r"(d[54]), "+r"(d[55]),
          "+r"(d[56]), "+r"(d[57]), "+r"(d[58]), "+r"(d[59]), "+r"(d[60]), "+r"(d[61]), "+r"(d[62]), "+r"(d[63])
        : "l"(da), "l"(db));
}

// Seed every accumulator with the bias, so the next group's sum arrives as 1.5 * 2^23 + i.
__device__ __forceinline__ void brain_seed(unsigned int (&d)[64]) {
#pragma unroll
    for (int i = 0; i < 64; ++i) { d[i] = FLOAT_BIAS_BITS; }
}

// The activation scale rounded to 21 significant bits, so that `bias * scale` is exact (the bias has two
// significant bits): the fold below then needs no separate bias subtraction and no scale product. The relative
// change is at most 2^-21 = 4.8e-7, under the f32 rounding noise of the running sum of K/32 terms.
__device__ __forceinline__ float brain_round21(float x) {
    return __uint_as_float((__float_as_uint(x) + 3u) & ~7u);
}

// f += (d - bias) * sx * sw for one group, with sx already rounded (`xa`, `xb` for the thread's two rows) and
// `nxa = -bias * xa` (exact). fma(d, xa, nxa) is the integer sum i times xa, rounded once, because d = bias + i
// exactly and bias * xa is exact; one more fused multiply-add folds it into the total. `sc` holds this thread's 32
// weight-row scales of the group (columns 8j + 2t and 8j + 2t + 1 of the m64n128 accumulator fragment, j = 0..15).
__device__ __forceinline__ void brain_fold(const unsigned int (&d)[64], float (&f)[64], const float (&sc)[32], float xa, float nxa, float xb, float nxb) {
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        f[4 * j + 0] = fmaf(fmaf(__uint_as_float(d[4 * j + 0]), xa, nxa), sc[2 * j], f[4 * j + 0]);
        f[4 * j + 1] = fmaf(fmaf(__uint_as_float(d[4 * j + 1]), xa, nxa), sc[2 * j + 1], f[4 * j + 1]);
        f[4 * j + 2] = fmaf(fmaf(__uint_as_float(d[4 * j + 2]), xb, nxb), sc[2 * j], f[4 * j + 2]);
        f[4 * j + 3] = fmaf(fmaf(__uint_as_float(d[4 * j + 3]), xb, nxb), sc[2 * j + 1], f[4 * j + 3]);
    }
}

extern "C" __global__ void __launch_bounds__(THREADS, 1) brain_matmul_i8_wgmma(const unsigned int* params,
                                                                               const unsigned int* xq,
                                                                               const unsigned int* wq,
                                                                               const float* sx,
                                                                               const float* sw,
                                                                               float* out) {
    extern __shared__ unsigned char smem_raw[];
    const unsigned int M = params[0];
    const unsigned int K = params[1] * 4u;
    const unsigned int N = params[2];
    const unsigned int ng = K / 32u;
    const unsigned int ktiles = K / BK;

    // Flat 1-D grid, row tile fastest: the blocks that run together share a weight tile.
    const unsigned int blk = blockIdx.y * gridDim.x + blockIdx.x;
    const unsigned int tiles_m = (M + BM - 1u) / BM;
    if (tiles_m == 0u) { return; }
    const unsigned int tile_n = blk / tiles_m;
    const unsigned int tile_m = blk - tile_n * tiles_m;
    const unsigned int m0 = tile_m * BM;
    const unsigned int n0 = tile_n * BN;
    if (m0 >= M || n0 >= N) { return; }  // block-uniform

    const unsigned int tid = threadIdx.x;
    const unsigned int wg = tid / 128;        // consumer warpgroup: rows 64 * wg .. 64 * wg + 63
    const unsigned int lane = tid & 31u;
    const unsigned int warp = (tid >> 5) & 3u;

    // 1024-byte aligned base: the 128-byte swizzle pattern repeats every 1024 bytes.
    const unsigned int raw_addr = brain_smem_addr(smem_raw);
    const unsigned int base = (raw_addr + 1023u) & ~1023u;
    unsigned char* const gen = smem_raw + (base - raw_addr);

    const unsigned char* const xb = reinterpret_cast<const unsigned char*>(xq);
    const unsigned char* const wb = reinterpret_cast<const unsigned char*>(wq);

    // Per-thread source rows of the 4 + 4 sixteen-byte chunks it copies per tile (chunk i = tid + 256 it:
    // row i / 8 = tid / 8 + 32 it, chunk c = tid % 8), computed once. Threads 0..127 also copy the four group
    // scales (16 bytes) of weight row tid, threads 128..255 those of activation row tid - 128.
    const unsigned int crow = tid >> 3, cc = tid & 7u;
    const unsigned int c_dst = crow * 128u + ((cc ^ (crow & 7u)) << 4);   // + 4096 it: 32 rows keep row mod 8
    const unsigned char* a_src[4];
    const unsigned char* b_src[4];
    unsigned int a_len[4], b_len[4];
#pragma unroll
    for (int it = 0; it < 4; ++it) {
        const unsigned int ar = m0 + crow + 32u * it, br = n0 + crow + 32u * it;
        a_len[it] = ar < M ? 16u : 0u;
        b_len[it] = br < N ? 16u : 0u;
        a_src[it] = xb + (unsigned long long)(ar < M ? ar : 0u) * K + cc * 16u;
        b_src[it] = wb + (unsigned long long)(br < N ? br : 0u) * K + cc * 16u;
    }
    const bool scale_x = tid >= 128u;
    const unsigned int srow = (scale_x ? m0 : n0) + (tid & 127u);
    const bool s_ok = srow < (scale_x ? M : N);
    const float* const s_src = (scale_x ? sx : sw) + (unsigned long long)(s_ok ? srow : 0u) * ng;
    const unsigned int s_dst = A_BYTES + B_BYTES + (scale_x ? SS_FLOATS * 4u : 0u) + (tid & 127u) * 16u;

    // Stage `kt` into ring slot `slot`: the two int8 tiles and the tile's four group scales per row.
    auto load_stage = [&](unsigned int slot, unsigned int kt) {
        const unsigned int st = base + slot * STAGE_BYTES;
        const unsigned int kbyte = kt * BK;
#pragma unroll
        for (int it = 0; it < 4; ++it) {
            brain_cp16(st + c_dst + 4096u * it, a_src[it] + kbyte, a_len[it]);
            brain_cp16(st + A_BYTES + c_dst + 4096u * it, b_src[it] + kbyte, b_len[it]);
        }
        brain_cp16(st + s_dst, s_src + kt * 4u, s_ok ? 16u : 0u);
    };

    float f[64];
#pragma unroll
    for (int i = 0; i < 64; ++i) { f[i] = 0.0f; }
    unsigned int acc[64];
    brain_seed(acc);

#pragma unroll
    for (unsigned int s = 0; s < PREFETCH; ++s) {
        if (s < ktiles) { load_stage(s, s); }
        brain_cp_commit();
    }

    // The fragment rows this thread owns: r and r + 8 of the warp's 16-row slice of the warpgroup's 64.
    const unsigned int frow = wg * 64u + warp * 16u + (lane >> 2);
    const unsigned int fcol = (lane & 3u) * 2u;

    for (unsigned int kt = 0; kt < ktiles; ++kt) {
        // Tile kt landed (this thread's copies), made visible to the async proxy, and every thread is done with
        // tile kt - 1: the slot the next load overwrites.
        brain_cp_wait<PREFETCH - 1>();
        brain_fence_proxy_async();
        __syncthreads();
        if (kt + PREFETCH < ktiles) { load_stage((kt + PREFETCH) % STAGES, kt + PREFETCH); }
        brain_cp_commit();

        const unsigned int slot = kt % STAGES;
        const unsigned int a_base = base + slot * STAGE_BYTES + wg * (64u * BK);
        const unsigned int b_base = base + slot * STAGE_BYTES + A_BYTES;
        // This thread's weight-row scales ([row][group]: columns 8j + fcol and +1) and activation-row scales.
        const float* ss = reinterpret_cast<const float*>(gen + slot * STAGE_BYTES + A_BYTES + B_BYTES) + fcol * 4u;
        const float* xs = reinterpret_cast<const float*>(gen + slot * STAGE_BYTES + A_BYTES + B_BYTES) + SS_FLOATS + frow * 4u;
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            brain_wgmma_fence();
            brain_wgmma_s8(acc, brain_desc(a_base + q * 32u), brain_desc(b_base + q * 32u));
            brain_wgmma_commit();
            // The scales are read while the tensor cores run the group, not after they finish.
            float sc[32];
#pragma unroll
            for (int j = 0; j < 16; ++j) {
                sc[2 * j] = ss[j * 32 + q];
                sc[2 * j + 1] = ss[j * 32 + 4 + q];
            }
            const float xa = brain_round21(xs[q]), xb = brain_round21(xs[32 + q]);
            brain_wgmma_wait<0>();
            brain_fold(acc, f, sc, xa, -FLOAT_BIAS * xa, xb, -FLOAT_BIAS * xb);
            brain_seed(acc);
        }
    }

    // out[m, n] = f: thread layout of the m64n128 accumulator.
    const unsigned int r0 = m0 + frow, r1 = r0 + 8u;
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        const unsigned int c = n0 + j * 8u + fcol;
        if (r0 < M) {
            float* o = out + (unsigned long long)r0 * N + c;
            if (c + 1u < N && (N & 1u) == 0u) { *reinterpret_cast<float2*>(o) = make_float2(f[4 * j], f[4 * j + 1]); }
            else { if (c < N) { o[0] = f[4 * j]; } if (c + 1u < N) { o[1] = f[4 * j + 1]; } }
        }
        if (r1 < M) {
            float* o = out + (unsigned long long)r1 * N + c;
            if (c + 1u < N && (N & 1u) == 0u) { *reinterpret_cast<float2*>(o) = make_float2(f[4 * j + 2], f[4 * j + 3]); }
            else { if (c < N) { o[0] = f[4 * j + 2]; } if (c + 1u < N) { o[1] = f[4 * j + 3]; } }
        }
    }
}
