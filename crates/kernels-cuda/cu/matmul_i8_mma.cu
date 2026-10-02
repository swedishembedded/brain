// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for prompt-processing throughput
// on quantised language models for its clients. If your team needs expertise
// in putting int8 weights onto tensor cores without giving up their
// group-wise scales, then you can procure our services by sending an email to
// info@swedishembedded.com.
//
// out = diag(sx) . (x_q . W_q^T) with GROUP-WISE weight scales, on int8
// TENSOR CORES (`mma.sync.m16n8k32` s8 x s8 -> s32). The CUDA form of the
// portable `matmul_i8_dyn.wgsl`, with the identical argument contract:
//
//   params : [M, KG, N] u32   KG = K/4 (packed int8 words along K)
//   x_q    : [M, K/4]   u32   4 int8 activations per word, row-major
//   w_q    : [N, K/4]   u32   4 int8 weights per word, row-major
//   sx     : [M]        f32   per-token activation scale
//   sw     : [N, K/32]  f32   per-(row, 32-element group) weight scale
//   out    : [M, N]     f32   out[m,n] = sx[m] * sum_g sw[n,g] * (sum_{k in g} xq[m,k]*wq[n,k])
//
// Requires K % 64 == 0 (two scale groups per staged k-tile); the provider
// declines any other K and the portable kernel serves it.
//
// Why tensor cores, and why the scale fold decides the design
// ----------------------------------------------------------
// The portable kernel is DP4A: four int8 MACs per FMA-pipe instruction. On a
// part with int8 tensor cores that leaves two orders of magnitude on the
// floor. The obstacle to a stock int8 GEMM library is the weight scale:
// GGUF Q8_0 carries one scale per 32 elements of K, and a library GEMM can
// scale only per output row/column. Folding the scale into the int32 sum is
// legal only at a group boundary - which is exactly one `k32` MMA. So every
// MMA here starts from a fresh accumulator, and its result is folded into
// an f32 running total through that group's scale before the next MMA.
//
// The fold is the bottleneck, not the MMA, so it is built from the two
// cheapest instructions available:
//   * the MMA accumulator is SEEDED with 0x4B400000 (the float 1.5 * 2^23),
//     so the integer result i comes back as the bit pattern of the float
//     (1.5 * 2^23 + i) - exact for |i| < 2^22, and |i| <= 32 * 127 * 128 here.
//     That replaces the int->float conversion, which runs at a fraction of
//     the FMA rate, with a free reinterpret;
//   * one FADD removes the bias (exact) and one FFMA applies the scale.
// The sum inside a group is exact integer arithmetic; across groups it is f32
// in ascending group order, as in the portable kernel. The one difference
// from it is that the scale multiply-add is a single fused rounding rather
// than two (the backend compiles with --fmad=false, so the portable kernel's
// `acc += f32(c) * s` rounds twice; `fmaf` below is explicit and still
// fused). Agreement is therefore to f32 rounding of the running sum, not bit
// identity.
//
// Data movement: 4-stage cp.async pipeline, one __syncthreads per 64-deep
// k-tile. Tiles are stored 64 bytes per row with the 16-byte chunk index
// XOR-swizzled by (row >> 1) & 3, which makes every ldmatrix phase (8 rows,
// same logical chunk) touch eight distinct 16-byte bank groups. The per-
// group weight scales ride the same pipeline (two floats per weight row per
// stage).
//
// No `__restrict__`: brain's device buffers alias by design (sliced steps bind
// ranges of one allocation).

#ifndef BRAIN_I8_BM
#define BRAIN_I8_BM 64    // output rows per block
#endif
#ifndef BRAIN_I8_BN
#define BRAIN_I8_BN 64    // output columns per block
#endif
#ifndef BRAIN_I8_WARPS_M
#define BRAIN_I8_WARPS_M 2
#endif
#ifndef BRAIN_I8_WARPS_N
#define BRAIN_I8_WARPS_N 2
#endif
#ifndef BRAIN_I8_STAGES
#define BRAIN_I8_STAGES 4
#endif

#define BK 64                                    // int8 along K per staged tile
#define WARPS (BRAIN_I8_WARPS_M * BRAIN_I8_WARPS_N)
#define THREADS (WARPS * 32)
#define WM (BRAIN_I8_BM / BRAIN_I8_WARPS_M)      // warp tile rows
#define WN (BRAIN_I8_BN / BRAIN_I8_WARPS_N)      // warp tile cols
#define MT (WM / 16)                             // m16 tiles per warp
#define NT (WN / 8)                              // n8 tiles per warp (even: B loads come in pairs)
#define A_STAGE_BYTES (BRAIN_I8_BM * BK)
#define B_STAGE_BYTES (BRAIN_I8_BN * BK)
#define S_STAGE_FLOATS (BRAIN_I8_BN * 2)

#define FLOAT_BIAS_BITS 0x4B400000u              // 1.5 * 2^23
#define FLOAT_BIAS 12582912.0f

__device__ __forceinline__ unsigned int brain_smem_addr(const void* p) {
    return static_cast<unsigned int>(__cvta_generic_to_shared(p));
}

// 16-byte global -> shared copy, zero-filled when `bytes` is 0 (a row past
// the edge of the matrix). The pointer is always a valid address.
__device__ __forceinline__ void brain_cp16(unsigned int dst, const void* src, unsigned int bytes) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(bytes));
}

__device__ __forceinline__ void brain_cp8(unsigned int dst, const void* src, unsigned int bytes) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8, %2;\n" ::"r"(dst), "l"(src), "r"(bytes));
}

__device__ __forceinline__ void brain_cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }

template <int N>
__device__ __forceinline__ void brain_cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

__device__ __forceinline__ void brain_ldsm_x4(unsigned int addr, unsigned int& r0, unsigned int& r1, unsigned int& r2, unsigned int& r3) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                 : "r"(addr));
}

// d = a * b + bias, per element, in int32. `bias` seeds every accumulator.
__device__ __forceinline__ void brain_mma_s8(unsigned int (&d)[4], const unsigned int (&a)[4], unsigned int b0, unsigned int b1) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %10, %10, %10};\n"
                 : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(FLOAT_BIAS_BITS));
}

// Byte offset of 16-byte chunk `c` (0..3) of tile row `r` inside a 64-byte-row
// tile: the swizzle described in the header.
__device__ __forceinline__ unsigned int brain_tile_off(unsigned int r, unsigned int c) {
    return r * BK + ((c ^ ((r >> 1) & 3u)) << 4);
}

extern "C" __global__ void __launch_bounds__(THREADS) brain_matmul_i8_mma(const unsigned int* params,
                                                                          const unsigned int* xq,
                                                                          const unsigned int* wq,
                                                                          const float* sx,
                                                                          const float* sw,
                                                                          float* out) {
    const unsigned int M = params[0];
    const unsigned int K = params[1] * 4u;
    const unsigned int N = params[2];
    const unsigned int ng = K / 32u;       // weight-scale groups per row
    const unsigned int ktiles = K / BK;

    // Flat 1-D block count, as every kernel in this tree reads it. The row
    // tile is the FAST index, so the blocks that run together share one
    // weight tile and the weights stream from HBM once, not once per row tile.
    const unsigned int blk = blockIdx.y * gridDim.x + blockIdx.x;
    const unsigned int tiles_m = (M + BRAIN_I8_BM - 1u) / BRAIN_I8_BM;
    if (tiles_m == 0u) { return; }
    const unsigned int tile_n = blk / tiles_m;
    const unsigned int tile_m = blk - tile_n * tiles_m;
    const unsigned int row0 = tile_m * BRAIN_I8_BM;
    const unsigned int col0 = tile_n * BRAIN_I8_BN;
    if (row0 >= M || col0 >= N) { return; }  // block-uniform

    __shared__ __align__(128) unsigned char As[BRAIN_I8_STAGES][A_STAGE_BYTES];
    __shared__ __align__(128) unsigned char Bs[BRAIN_I8_STAGES][B_STAGE_BYTES];
    __shared__ __align__(16) float Ss[BRAIN_I8_STAGES][S_STAGE_FLOATS];

    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid & 31u;
    const unsigned int warp = tid >> 5;
    const unsigned int wm = warp / BRAIN_I8_WARPS_N;
    const unsigned int wn = warp - wm * BRAIN_I8_WARPS_N;

    const unsigned char* xb = reinterpret_cast<const unsigned char*>(xq);
    const unsigned char* wb = reinterpret_cast<const unsigned char*>(wq);

    // Everything a stage load needs that does not change with the k-tile is
    // computed once here, so the pipelined loop issues a copy per chunk and
    // little else: an issue slot spent on address arithmetic is one the fold
    // below needs (see the header).
    constexpr unsigned int A_IT = (BRAIN_I8_BM * 4u + THREADS - 1u) / THREADS;
    constexpr unsigned int B_IT = (BRAIN_I8_BN * 4u + THREADS - 1u) / THREADS;
    const unsigned char* a_src[A_IT];
    const unsigned char* b_src[B_IT];
    unsigned int a_dst[A_IT], a_len[A_IT], b_dst[B_IT], b_len[B_IT];
#pragma unroll
    for (unsigned int it = 0; it < A_IT; ++it) {
        const unsigned int i = tid + it * THREADS;
        const unsigned int r = i >> 2, c = i & 3u;
        const bool ok = i < BRAIN_I8_BM * 4u && row0 + r < M;
        a_src[it] = xb + (unsigned long long)(ok ? row0 + r : 0u) * K + c * 16u;
        a_dst[it] = brain_tile_off(r, c);
        a_len[it] = ok ? 16u : 0u;
    }
#pragma unroll
    for (unsigned int it = 0; it < B_IT; ++it) {
        const unsigned int i = tid + it * THREADS;
        const unsigned int r = i >> 2, c = i & 3u;
        const bool ok = i < BRAIN_I8_BN * 4u && col0 + r < N;
        b_src[it] = wb + (unsigned long long)(ok ? col0 + r : 0u) * K + c * 16u;
        b_dst[it] = brain_tile_off(r, c);
        b_len[it] = ok ? 16u : 0u;
    }
    // One 8-byte (two group scales) copy per weight row, by the first BN threads.
    const bool s_ok = tid < BRAIN_I8_BN && col0 + tid < N;
    const float* s_src = sw + (unsigned long long)(s_ok ? col0 + tid : 0u) * ng;
    const unsigned int s_dst = brain_smem_addr(&Ss[0][0]) + tid * 8u;
    const unsigned int a_smem = brain_smem_addr(&As[0][0]);
    const unsigned int b_smem = brain_smem_addr(&Bs[0][0]);

    // Stage `kt` into pipeline slot `s`.
    auto load_stage = [&](unsigned int s, unsigned int kt) {
        const unsigned int kbyte = kt * BK;
#pragma unroll
        for (unsigned int it = 0; it < A_IT; ++it) {
            brain_cp16(a_smem + s * A_STAGE_BYTES + a_dst[it], a_src[it] + kbyte, a_len[it]);
        }
#pragma unroll
        for (unsigned int it = 0; it < B_IT; ++it) {
            brain_cp16(b_smem + s * B_STAGE_BYTES + b_dst[it], b_src[it] + kbyte, b_len[it]);
        }
        if (tid < BRAIN_I8_BN) {
            // ng is even (K % 64 == 0) and kt*2 is even: 8-byte aligned.
            brain_cp8(s_dst + s * (S_STAGE_FLOATS * 4u), s_src + kt * 2u, s_ok ? 8u : 0u);
        }
    };

    float f[MT][NT][4];
#pragma unroll
    for (int i = 0; i < MT; ++i) {
#pragma unroll
        for (int j = 0; j < NT; ++j) {
#pragma unroll
            for (int e = 0; e < 4; ++e) { f[i][j][e] = 0.0f; }
        }
    }

#pragma unroll
    for (unsigned int s = 0; s < BRAIN_I8_STAGES - 1; ++s) {
        if (s < ktiles) { load_stage(s, s); }
        brain_cp_commit();
    }

    // ldmatrix lane geometry. Matrix j = lane / 8 supplies the row address of
    // lane%8 within it. For A (16 rows x 32 bytes): j = 0..3 are
    // (rows 0-7, k 0-15), (rows 8-15, k 0-15), (rows 0-7, k 16-31),
    // (rows 8-15, k 16-31). For B (two n8 tiles x 32 bytes): j = 0..3 are
    // (n0, k 0-15), (n0, k 16-31), (n1, k 0-15), (n1, k 16-31).
    //
    // Offsets are per (k32 step) only: moving down 16 rows keeps
    // `(row >> 1) & 3`, so the swizzle is the same and a further m16 / n16 tile
    // is a constant 16 * 64 bytes - an immediate, not an instruction.
    const unsigned int lj = lane >> 3, lr = lane & 7u;
    const unsigned int a_row = wm * WM + lr + ((lj & 1u) << 3);
    const unsigned int b_row = wn * WN + lr + ((lj >> 1) << 3);
    unsigned int a_off[2], b_off[2];
#pragma unroll
    for (unsigned int ks = 0; ks < 2; ++ks) {
        a_off[ks] = brain_tile_off(a_row, ks * 2u + (lj >> 1));
        b_off[ks] = brain_tile_off(b_row, ks * 2u + (lj & 1u));
    }
    const unsigned int sc_off = (wn * WN + (lane & 3u) * 2u) * 2u;  // floats, this thread's first column

    unsigned int s_rd = 0, s_wr = BRAIN_I8_STAGES - 1;
    for (unsigned int kt = 0; kt < ktiles; ++kt) {
        brain_cp_wait<BRAIN_I8_STAGES - 2>();
        __syncthreads();
        if (kt + BRAIN_I8_STAGES - 1 < ktiles) { load_stage(s_wr, kt + BRAIN_I8_STAGES - 1); }
        brain_cp_commit();
        s_wr = (s_wr + 1 == BRAIN_I8_STAGES) ? 0u : s_wr + 1;

        const unsigned int a_base = a_smem + s_rd * A_STAGE_BYTES;
        const unsigned int b_base = b_smem + s_rd * B_STAGE_BYTES;
        const float* ss = &Ss[s_rd][sc_off];
        s_rd = (s_rd + 1 == BRAIN_I8_STAGES) ? 0u : s_rd + 1;

        // Both k32 scales of both columns of every n8 tile this thread folds
        // into: one 16-byte read per tile (columns 2t, 2t+1, groups 0 and 1).
        float4 sc[NT];
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) { sc[nt] = *reinterpret_cast<const float4*>(ss + nt * 16); }

#pragma unroll
        for (unsigned int ks = 0; ks < 2; ++ks) {
            unsigned int b[NT][2];
#pragma unroll
            for (int np = 0; np < NT / 2; ++np) {
                brain_ldsm_x4(b_base + b_off[ks] + np * (16u * BK), b[2 * np][0], b[2 * np][1], b[2 * np + 1][0], b[2 * np + 1][1]);
            }
#pragma unroll
            for (int mt = 0; mt < MT; ++mt) {
                unsigned int a[4];
                brain_ldsm_x4(a_base + a_off[ks] + mt * (16u * BK), a[0], a[1], a[2], a[3]);
#pragma unroll
                for (int nt = 0; nt < NT; ++nt) {
                    unsigned int d[4];
                    brain_mma_s8(d, a, b[nt][0], b[nt][1]);
                    const float s0 = ks == 0 ? sc[nt].x : sc[nt].y;   // column 2t
                    const float s1 = ks == 0 ? sc[nt].z : sc[nt].w;   // column 2t + 1
                    f[mt][nt][0] = fmaf(__uint_as_float(d[0]) - FLOAT_BIAS, s0, f[mt][nt][0]);
                    f[mt][nt][1] = fmaf(__uint_as_float(d[1]) - FLOAT_BIAS, s1, f[mt][nt][1]);
                    f[mt][nt][2] = fmaf(__uint_as_float(d[2]) - FLOAT_BIAS, s0, f[mt][nt][2]);
                    f[mt][nt][3] = fmaf(__uint_as_float(d[3]) - FLOAT_BIAS, s1, f[mt][nt][3]);
                }
            }
        }
    }
    brain_cp_wait<0>();

    const unsigned int g = lane >> 2;    // MMA groupID
    const unsigned int t4 = lane & 3u;   // MMA threadID_in_group

#pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            const unsigned int r = row0 + wm * WM + mt * 16u + g + half * 8u;
            if (r >= M) { continue; }
            const float s = sx[r];
#pragma unroll
            for (int nt = 0; nt < NT; ++nt) {
                const unsigned int c = col0 + wn * WN + nt * 8u + t4 * 2u;
                float* o = out + (unsigned long long)r * N + c;
                if (c + 1u < N && (N & 1u) == 0u) {
                    *reinterpret_cast<float2*>(o) = make_float2(f[mt][nt][half * 2] * s, f[mt][nt][half * 2 + 1] * s);
                } else {
                    if (c < N) { o[0] = f[mt][nt][half * 2] * s; }
                    if (c + 1u < N) { o[1] = f[mt][nt][half * 2 + 1] * s; }
                }
            }
        }
    }
}
