// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for prompt-processing throughput on
// quantised language models for its clients. If your team needs expertise in
// putting int8 weights onto Hopper warpgroup tensor cores without giving up
// their group-wise scales, then you can procure our services by sending an
// email to info@swedishembedded.com.
//
// out = A . (W_q * S)^T on Hopper WARPGROUP tensor cores (`wgmma.mma_async`,
// bf16 x bf16 -> f32), with the weight kept as int8 with GROUP-WISE scales in
// memory and decoded to bf16 in shared memory, one k-tile ahead of the MMA:
//
//   params : [M, K, N] u32
//   a      : [M, K]      bf16 (two per u32)   the activations, row-major
//   w_q    : [N, K/4]    u32  4 int8 weights per word, row-major
//   sw     : [N, K/32]   f32  per-(row, 32-element group) weight scale
//   out    : [M, N]      f32  out[m,n] = sum_k a[m,k] * bf16(w_q[n,k] * sw[n,k/32])
//
// Requires K % 128 == 0 and N % 128 == 0 (the provider declines other shapes and
// the mma.sync kernel serves them), `sm_90a` (compile with ArchFeatures::Required).
//
// Why this and not the int8 `mma.sync` kernel. A group-wise weight scale can only
// be folded into an int32 accumulator at a group boundary, which is one k32
// instruction, so the int8 kernel pays an int->f32 fold per accumulator per k32
// and is bound by it (about 300 TOPS of a 1979 peak). Decoding the weight BEFORE
// the multiply moves the scale out of the accumulator path: the tensor cores see
// plain bf16 operands and accumulate across the whole K, and the decode (about
// five instructions per two weights) is done once per 128 output rows, in the
// shadow of the previous tile's asynchronous MMAs. The activations are bf16
// instead of dynamically quantised int8, which is the more precise operand.
//
// Structure (one block = 2 consumer warpgroups, 128 x 128 output tile, 128-deep
// k-tiles, 1 block per SM):
//   * `A` and the raw int8 weight tile and their scales ride a 3-deep cp.async
//     ring; the decoded weight tile `Bd` is double-buffered.
//   * per k-tile: issue the 8 wgmma of tile kt (async), then decode tile kt+1 into
//     the other `Bd` while they run, prefetch tile kt+2, wait for the wgmma, fence
//     the generic-proxy shared writes for the async proxy, barrier.
//   * every operand tile is K-major with the 128-byte swizzle (16-byte chunk index
//     XOR row mod 8), the layout the warpgroup MMA descriptors name: a [rows][64
//     bf16] sub-tile has 128-byte rows, 8-row groups 1024 bytes apart.
//
// No `__restrict__`: brain's device buffers alias by design.

#define BM 128
#define BN 128
#define BK 128
#define THREADS 256
#define STAGES 3

#define SUB_BYTES (128 * 128)                 // one [128 rows][64 bf16] sub-tile
#define A_STAGE_BYTES (2 * SUB_BYTES)         // 128 k = two sub-tiles
#define B_STAGE_BYTES (2 * SUB_BYTES)
#define RAW_STAGE_BYTES (128 * 128)           // [128 rows][128 int8]
#define SCL_STAGE_BYTES (128 * 4 * 4)         // [128 rows][4 groups] f32

#define OFF_A 0
#define OFF_RAW (OFF_A + STAGES * A_STAGE_BYTES)
#define OFF_SCL (OFF_RAW + STAGES * RAW_STAGE_BYTES)
#define OFF_B (OFF_SCL + STAGES * SCL_STAGE_BYTES)
#define SMEM_BYTES (OFF_B + 2 * B_STAGE_BYTES + 1024)

__device__ __forceinline__ unsigned int brain_smem_addr(const void* p) {
    return static_cast<unsigned int>(__cvta_generic_to_shared(p));
}

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

// D (64 x 128 f32, 64 registers per thread) += A (64 x 16 bf16) . B (128 x 16 bf16)^T.
__device__ __forceinline__ void brain_wgmma_m64n128k16(float (&d)[64], unsigned long long da, unsigned long long db, int accumulate) {
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %66, 0;\n"
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
        : "l"(da), "l"(db), "r"(accumulate));
}

// The bf16 pair (lo, hi) of two f32 values, `lo` in the low half.
__device__ __forceinline__ unsigned int brain_pack_bf16x2(float lo, float hi) {
    unsigned int r;
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;\n" : "=r"(r) : "f"(hi), "f"(lo));
    return r;
}

// The float 1.5 * 2^23 + (byte ^ 0x80) of byte `I` of `w_flipped` (the word with every byte's top bit
// flipped): the integer v + 128 of the signed weight v sits in the low mantissa bits, and subtracting
// 1.5 * 2^23 + 128 leaves v exactly, with no int->float conversion.
template <int I>
__device__ __forceinline__ float brain_byte_as_float(unsigned int w_flipped) {
    unsigned int bits;
    asm("prmt.b32 %0, %1, %2, %3;\n" : "=r"(bits) : "r"(w_flipped), "r"(0x4B400000u), "n"(0x7640 + I));
    return __uint_as_float(bits);
}

// Decode 16 int8 weights (four words) of one row into eight bf16 pairs, scaled.
__device__ __forceinline__ void brain_decode16(const uint4& raw, float scale, unsigned int (&out)[8]) {
    const float bias = 12582912.0f + 128.0f;
    const unsigned int words[4] = {raw.x ^ 0x80808080u, raw.y ^ 0x80808080u, raw.z ^ 0x80808080u, raw.w ^ 0x80808080u};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const unsigned int w = words[i];
        const float f0 = (brain_byte_as_float<0>(w) - bias) * scale;
        const float f1 = (brain_byte_as_float<1>(w) - bias) * scale;
        const float f2 = (brain_byte_as_float<2>(w) - bias) * scale;
        const float f3 = (brain_byte_as_float<3>(w) - bias) * scale;
        out[2 * i] = brain_pack_bf16x2(f0, f1);
        out[2 * i + 1] = brain_pack_bf16x2(f2, f3);
    }
}

extern "C" __global__ void __launch_bounds__(THREADS, 1) brain_matmul_i8w_bf16a_wgmma(const unsigned int* params,
                                                                                      const unsigned int* a,
                                                                                      const unsigned int* wq,
                                                                                      const float* sw,
                                                                                      float* out) {
    extern __shared__ unsigned char smem_raw[];
    const unsigned int M = params[0];
    const unsigned int K = params[1];
    const unsigned int N = params[2];
    const unsigned int ktiles = K / BK;
    const unsigned int tid = threadIdx.x;
    const unsigned int wg = tid / 128;           // consumer warpgroup: rows 64*wg .. 64*wg+63
    // A flat 1-D grid of ceil(M / 128) * (N / 128) blocks, M tiles fastest: the blocks that share a
    // weight tile run together and read it from L2.
    const unsigned int m_tiles = (M + BM - 1) / BM;
    const unsigned int m0 = (blockIdx.x % m_tiles) * BM;
    const unsigned int n0 = (blockIdx.x / m_tiles) * BN;

    // 1024-byte aligned base: the 128-byte swizzle pattern repeats every 1024 bytes.
    const unsigned int base = (brain_smem_addr(smem_raw) + 1023u) & ~1023u;
    unsigned char* const gen = smem_raw + (base - brain_smem_addr(smem_raw));

    const unsigned char* const a_bytes = reinterpret_cast<const unsigned char*>(a);
    const unsigned char* const w_bytes = reinterpret_cast<const unsigned char*>(wq);

    // cp.async tile `kt` (A rows, raw int8 weights, scales) into ring slot `kt % STAGES`.
    auto issue_tile = [&](unsigned int kt) {
        const unsigned int slot = kt % STAGES;
        const unsigned int a_dst = base + OFF_A + slot * A_STAGE_BYTES;
        for (unsigned int i = 0; i < 8; ++i) {            // 2048 16-byte chunks of A
            const unsigned int q = tid + THREADS * i;
            const unsigned int row = q >> 4, c = q & 15;
            const unsigned int sub = c >> 3, cc = c & 7;
            const unsigned int grow = m0 + row;
            const unsigned char* src = a_bytes + (size_t)(grow < M ? grow : 0) * K * 2 + (size_t)kt * (BK * 2) + c * 16;
            brain_cp16(a_dst + sub * SUB_BYTES + row * 128 + ((cc ^ (row & 7)) << 4), src, grow < M ? 16u : 0u);
        }
        const unsigned int raw_dst = base + OFF_RAW + slot * RAW_STAGE_BYTES;
        for (unsigned int i = 0; i < 4; ++i) {            // 1024 16-byte chunks of the raw weight tile
            const unsigned int q = tid + THREADS * i;
            const unsigned int row = q >> 3, c = q & 7;
            const unsigned char* src = w_bytes + (size_t)(n0 + row) * K + (size_t)kt * BK + c * 16;
            brain_cp16(raw_dst + row * 128 + ((c ^ (row & 7)) << 4), src, 16u);
        }
        if (tid < 128) {                                  // four group scales of a row
            const float* src = sw + (size_t)(n0 + tid) * (K / 32) + (size_t)kt * 4;
            brain_cp16(base + OFF_SCL + slot * SCL_STAGE_BYTES + tid * 16, src, 16u);
        }
    };

    // Decode the raw tile of ring slot `slot` into decoded buffer `buf`: thread t takes row t / 2 and
    // the 64 k of sub-tile t % 2.
    auto decode_tile = [&](unsigned int slot, unsigned int buf) {
        const unsigned int row = tid >> 1, half = tid & 1;
        const unsigned char* raw = gen + OFF_RAW + slot * RAW_STAGE_BYTES;
        const float* scl = reinterpret_cast<const float*>(gen + OFF_SCL + slot * SCL_STAGE_BYTES) + row * 4;
        unsigned char* dst = gen + OFF_B + buf * B_STAGE_BYTES + half * SUB_BYTES + row * 128;
#pragma unroll
        for (int j = 0; j < 4; ++j) {                      // chunk c = 4 * half + j of the 8 raw chunks
            const unsigned int c = 4 * half + j;
            const uint4 r = *reinterpret_cast<const uint4*>(raw + row * 128 + ((c ^ (row & 7)) << 4));
            unsigned int px[8];
            brain_decode16(r, scl[c >> 1], px);
            // 16 bf16 = two 16-byte chunks, 2 * j and 2 * j + 1 of this sub-tile's row.
            *reinterpret_cast<uint4*>(dst + (((2 * j) ^ (row & 7)) << 4)) = make_uint4(px[0], px[1], px[2], px[3]);
            *reinterpret_cast<uint4*>(dst + (((2 * j + 1) ^ (row & 7)) << 4)) = make_uint4(px[4], px[5], px[6], px[7]);
        }
    };

    float acc[64];
#pragma unroll
    for (int i = 0; i < 64; ++i) acc[i] = 0.0f;

    // Prologue: tiles 0 and 1 in flight, tile 0 decoded and visible.
    issue_tile(0);
    brain_cp_commit();
    if (ktiles > 1) issue_tile(1);
    brain_cp_commit();
    brain_cp_wait<1>();
    brain_fence_proxy_async();
    __syncthreads();
    decode_tile(0, 0);
    brain_fence_proxy_async();
    __syncthreads();

    for (unsigned int kt = 0; kt < ktiles; ++kt) {
        // 0. Prefetch tile kt + 2 into the slot tile kt - 1 used (its MMAs retired at the end of the last
        //    iteration). Always commit, so the number of pending groups does not depend on the tail.
        if (kt + 2 < ktiles) issue_tile(kt + 2);
        brain_cp_commit();

        // 1. The 8 k16 MMAs of tile kt, asynchronous.
        const unsigned int a_slot = base + OFF_A + (kt % STAGES) * A_STAGE_BYTES + wg * (64 * 128);
        const unsigned int b_buf = base + OFF_B + (kt & 1) * B_STAGE_BYTES;
        brain_wgmma_fence();
#pragma unroll
        for (int ks = 0; ks < 8; ++ks) {
            const unsigned int sub = ks >> 2, kin = (ks & 3) * 32;   // bytes within the 128-byte row
            brain_wgmma_m64n128k16(acc, brain_desc(a_slot + sub * SUB_BYTES + kin), brain_desc(b_buf + sub * SUB_BYTES + kin), 1);
        }
        brain_wgmma_commit();

        // 2. While they run: decode tile kt + 1. Its copies were issued an iteration ago; only tile
        //    kt + 2's group (just committed) may still be pending.
        if (kt + 1 < ktiles) {
            brain_cp_wait<1>();
            __syncthreads();                   // every thread's copies are in
            decode_tile((kt + 1) % STAGES, (kt + 1) & 1);
        }

        // 3. Retire tile kt's MMAs, publish the decoded tile and the fresh A rows to the async proxy.
        brain_wgmma_wait<0>();
        brain_fence_proxy_async();
        __syncthreads();
    }

    // out[m, n]: thread layout of the m64n128 accumulator.
    const unsigned int warp = (tid % 128) / 32, lane = tid % 32;
    const unsigned int row0 = m0 + wg * 64 + warp * 16 + lane / 4;
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        const unsigned int col = n0 + j * 8 + (lane % 4) * 2;
        if (row0 < M) *reinterpret_cast<float2*>(out + (size_t)row0 * N + col) = make_float2(acc[j * 4 + 0], acc[j * 4 + 1]);
        if (row0 + 8 < M) *reinterpret_cast<float2*>(out + (size_t)(row0 + 8) * N + col) = make_float2(acc[j * 4 + 2], acc[j * 4 + 3]);
    }
}
