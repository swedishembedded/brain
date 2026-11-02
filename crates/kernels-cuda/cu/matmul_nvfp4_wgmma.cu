// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for prompt-processing throughput on 4-bit language-model releases for its
// clients. If your team needs expertise in feeding Hopper FP8 tensor cores from NVFP4 checkpoints without a converted copy
// of the weights, then you can procure our services by sending an email to info@swedishembedded.com.
//
// NVFP4 weights against E4M3 activations on the Hopper warpgroup tensor cores, the weight tile decoded once into shared memory:
//
//   out[m, n] = global * sx[m] * sum_k e4m3(xq[m, k]) * w(n, k)
//   w(n, k)   = RN_e4m3( e2m1(code[n, k]) * e4m3(scale[n, k / 16]) / 8 ) * 8
//
// The E2M1 code times its E4M3 block scale has up to five significant bits and E4M3 holds four, so the product is rounded
// once, to nearest even, to E4M3 (relative error at most 2^-4, zero for the codes 0, 0.5, 1, 2, 4 and for half of the
// scales' mantissas); the factor 1/8 keeps the largest product (6 * 448) inside E4M3's range, and the epilogue multiplies it back
// together with the tensor's f32 scale and the activation row's scale. The block scale never reaches the tensor core as a
// scale: one MMA sums 32 weights of two block scales, so it has to be in the operand.
//
// The activations carry ONE scale per row (`quant_e4m3_row`), not one per 128 of K as the FP8 GEMMs take: a scale that varies along K
// costs a rescale of the whole accumulator at every 128 of K (64 to 128 multiplies a thread a stage, measured as a quarter of the kernel's
// time), a scale per row costs one multiply in the epilogue.
//
// Parameters: u32 [m, k, n, global_bits], k a multiple of 256, n even. Bindings: params, xq (E4M3 activations, four to a
// word, [m, k/4]), w (E2M1 codes, eight to a word, [n, k/8]), s (E4M3 block scales, four to a word, [n, k/64]),
// sx (f32 activation scale per row, [m]), out (f32 [m, n]).
//
// Structure (512 threads, one block an SM, a persistent grid over 256 x 128 output tiles):
//   warps 0..7   consumers: two warpgroups, each `wgmma m64n128k32` over two blocks of 64 rows (rows 64 w and 128 + 64 w of the tile) with the
//                accumulators of both in registers, the result written straight from them; lane 0 of the first warp is also the producer;
//   warps 8..15  decoders: the 4-bit codes of a stage (128 weight rows x 128 of K = 8 KiB) become the E4M3 weight operand
//                (16 KiB, the 128-byte swizzle the tensor core reads) through a table of 256 entries, one per scale byte, that
//                holds the E4M3 images of the eight code magnitudes: four `prmt` per eight codes plus a sign merge.
// The producer issues, per stage, the tensor-memory-accelerator loads of the activation tile (256 rows), the code tile and the scale tile
// into a ring of BRAIN_NW_STAGES stages (tensor maps built on the device as the FP8 GEMMs do) and refills a slot once every consumer has released
// it. A stage's slot holds the activation tile, the decoded weight tile and the raw code and scale tiles.

#ifndef BRAIN_NW_STAGES
#define BRAIN_NW_STAGES 3
#endif
#ifndef BRAIN_NW_ENTRY
#define BRAIN_NW_ENTRY brain_matmul_nvfp4_wgmma
#endif
#define BRAIN_NW_THREADS 512
#define BRAIN_NW_MAX_CTAS 132

#define BRAIN_NW_BM 256
#define BRAIN_NW_BN 128

// Bytes of a stage's parts: activation tile and decoded weight tile (128 rows of 128 bytes), code tile (128 rows of 64 bytes), scale tile
// (128 rows of 16 bytes, the first eight of a row are this stage's). All a multiple of 1 KiB so every swizzled tile stays 1024-byte aligned.
#define BRAIN_NW_A_BYTES (BRAIN_NW_BM * 128)
#define BRAIN_NW_B_BYTES (BRAIN_NW_BN * 128)
#define BRAIN_NW_C_BYTES (BRAIN_NW_BN * 64)
#define BRAIN_NW_S_BYTES (BRAIN_NW_BN * 16)
#define BRAIN_NW_STAGE_BYTES (BRAIN_NW_A_BYTES + BRAIN_NW_B_BYTES + BRAIN_NW_C_BYTES + BRAIN_NW_S_BYTES)

#ifndef BRAIN_PDL_ENTRY
#define BRAIN_PDL_ENTRY() do {} while (0)
#endif

// The driver's 128-byte maps for a 2-D byte tensor of 256 x 256 at address 0x10000 without swizzle: a 64-byte box of 128 rows (the codes) and a
// 16-byte box of 128 rows (the scales). (The activation map is `brain_fwt_template`.)
#define BRAIN_NW_CODES_TEMPLATE 0x0000000000010000ull, 0x0000001000060010ull, 0x0000000000000000ull, 0x0000000000000000ull, 0x000000ff000000ffull, 0x0000000000000000ull, 0x3f00000000000000ull, 0x000000000000007full, 0x0000000000002000ull, 0x0000000000000010ull, 0x0000000000000000ull, 0x0000000000000000ull, 0x0000000000000000ull, 0x0000000000000000ull, 0x0000000000000000ull, 0x0000000000000000ull
#define BRAIN_NW_SCALES_TEMPLATE 0x0000000000010000ull, 0x0000001000060010ull, 0x0000000000000000ull, 0x0000000000000000ull, 0x000000ff000000ffull, 0x0000000000000000ull, 0x0f00000000000000ull, 0x000000000000007full, 0x0000000000000800ull, 0x0000000000000010ull, 0x0000000000000000ull, 0x0000000000000000ull, 0x0000000000000000ull, 0x0000000000000000ull, 0x0000000000000000ull, 0x0000000000000000ull
__device__ __constant__ unsigned long long brain_nw_codes_template[16] = { BRAIN_NW_CODES_TEMPLATE };
__device__ __constant__ unsigned long long brain_nw_scales_template[16] = { BRAIN_NW_SCALES_TEMPLATE };
// Each SM's three maps (activations, codes, scales), in global memory (the TMA reads a map from global, constant or parameter memory).
__device__ __align__(128) unsigned long long brain_nw_maps[BRAIN_NW_MAX_CTAS][3][16];

// D (64 x 128 f32, 64 registers per thread) = A (64 x 32 e4m3) . B (128 x 32 e4m3)^T, plus D when `accumulate` is set.
__device__ __forceinline__ void brain_nw_wgmma(float (&d)[64], unsigned long long da, unsigned long long db, int accumulate) {
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %66, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3 "
        "{"
        "%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
        "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, "
        "%32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, "
        "%48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63"
        "}, "
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

// Tile order: the blocks that run together cover BRAIN_NW_GROUP_M row tiles of a few weight tiles, so the activation rows and the weight rows
// they read stay in the L2 whatever M is.
#ifndef BRAIN_NW_GROUP_M
#define BRAIN_NW_GROUP_M 16
#endif
__device__ __forceinline__ void brain_nw_tile(unsigned int tile, unsigned int tiles_m, unsigned int tiles_n, unsigned int& m0, unsigned int& n0) {
    const unsigned int per_group = BRAIN_NW_GROUP_M * tiles_n;
    const unsigned int group = tile / per_group;
    const unsigned int first_m = group * BRAIN_NW_GROUP_M;
    const unsigned int rows = min(tiles_m - first_m, (unsigned int)BRAIN_NW_GROUP_M);
    const unsigned int within = tile - group * per_group;
    m0 = (first_m + within % rows) * BRAIN_NW_BM;
    n0 = (within / rows) * BRAIN_NW_BN;
}

__device__ __forceinline__ void brain_nw_st_v4(unsigned int addr, unsigned int a, unsigned int b, unsigned int c, unsigned int d) {
    asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};\n" ::"r"(addr), "r"(a), "r"(b), "r"(c), "r"(d) : "memory");
}

// The loads of iteration `p` of a block (the `p % kblocks`-th 128 of K of its `p / kblocks`-th tile): the activation tile as two boxes of 128
// rows, the code tile and the scale tile, onto the barrier of the slot. One lane calls it.
__device__ __forceinline__ void brain_nw_issue(const unsigned long long* gmaps, unsigned int p, unsigned int kblocks, unsigned int tiles_m, unsigned int tiles_n, unsigned int base, unsigned int full_bar) {
    unsigned int m0, n0;
    brain_nw_tile(blockIdx.x + (p / kblocks) * gridDim.x, tiles_m, tiles_n, m0, n0);
    const unsigned int kt = p % kblocks;
    const unsigned int slot = p % BRAIN_NW_STAGES;
    const unsigned int st = base + slot * BRAIN_NW_STAGE_BYTES;
    const unsigned int bar = full_bar + 8u * slot;
    brain_fwt_mbar_arrive_expect_tx(bar, BRAIN_NW_A_BYTES + BRAIN_NW_C_BYTES + BRAIN_NW_S_BYTES);
    brain_fwt_tma_load(st, gmaps, kt * 128, m0, bar);
    brain_fwt_tma_load(st + BRAIN_NW_A_BYTES / 2u, gmaps, kt * 128, m0 + 128u, bar);
    brain_fwt_tma_load(st + BRAIN_NW_A_BYTES + BRAIN_NW_B_BYTES, gmaps + 16, kt * 64, n0, bar);
    // The scale tile is a box of 16 bytes (two stages' worth) that must start on a 16-byte boundary.
    brain_fwt_tma_load(st + BRAIN_NW_A_BYTES + BRAIN_NW_B_BYTES + BRAIN_NW_C_BYTES, gmaps + 32, (kt >> 1) * 16, n0, bar);
}

extern "C" __global__ void __launch_bounds__(BRAIN_NW_THREADS, 1)
BRAIN_NW_ENTRY(const unsigned int* params, const unsigned int* xq, const unsigned int* wq, const unsigned int* sq, const float* sx, float* out) {
    BRAIN_PDL_ENTRY();
    constexpr int STAGES = BRAIN_NW_STAGES;
    extern __shared__ unsigned char brain_nw_smem_raw[];
    __shared__ __align__(128) unsigned long long brain_nw_smem_maps[3][16];
    __shared__ __align__(16) uint2 brain_nw_lut[256];
    const unsigned int M = params[0];
    const unsigned int K = params[1];
    const unsigned int N = params[2];
    const float wscale = __uint_as_float(params[3]) * 8.0f;
    const unsigned int kblocks = K / 128u;
    const unsigned int tiles_m = (M + BRAIN_NW_BM - 1u) / BRAIN_NW_BM;
    const unsigned int tiles_n = (N + BRAIN_NW_BN - 1u) / BRAIN_NW_BN;
    const unsigned int tiles = tiles_m * tiles_n;

    const unsigned int tid = threadIdx.x;
    const unsigned int wg = __shfl_sync(0xffffffffu, tid / 128u, 0);  // 0, 1: consumers; 2, 3: decoders; 4: the producer warp
    const unsigned int lane = tid & 31u;
    const unsigned int warp = (tid >> 5) & 3u;

    const unsigned int raw_addr = brain_fwt_smem(brain_nw_smem_raw);
    const unsigned int base = (raw_addr + 1023u) & ~1023u;
    const unsigned int full_bar = base + STAGES * BRAIN_NW_STAGE_BYTES;  // STAGES mbarriers each: TMA landed, tile decoded, slot released
    const unsigned int dec_bar = full_bar + 8u * STAGES;
    const unsigned int empty_bar = dec_bar + 8u * STAGES;

    // The table of every scale byte (256 entries of 8 bytes).
    if (tid < 256u) { brain_nw_lut[tid] = brain_nw_scale_table(tid); }
    if (tid == 0u) {
        for (int s = 0; s < STAGES; ++s) {
            brain_fwt_mbar_init(full_bar + 8u * s, 1u);     // the producer's expect_tx arrival
            brain_fwt_mbar_init(dec_bar + 8u * s, 256u);    // the decoders' threads
            brain_fwt_mbar_init(empty_bar + 8u * s, 256u);  // the consumers' threads
        }
        asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    }
    __syncthreads();

    if (wg >= 2u) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 88;\n" ::: "memory");
        // Decoder: the stage's codes (512 chunks of 32 weights, four to a row) into the swizzled E4M3 tile.
        const unsigned int d = tid - 256u;
        unsigned int it = 0;
        for (unsigned int tile = blockIdx.x; tile < tiles; tile += gridDim.x) {
            for (unsigned int kt = 0; kt < kblocks; ++kt, ++it) {
                const unsigned int slot = it % STAGES;
                brain_fwt_mbar_wait(full_bar + 8u * slot, (it / STAGES) & 1u);
                const unsigned int st = base + slot * BRAIN_NW_STAGE_BYTES;
                const unsigned int b_tile = st + BRAIN_NW_A_BYTES;
                const unsigned int c_tile = b_tile + BRAIN_NW_B_BYTES;
                const unsigned int s_tile = c_tile + BRAIN_NW_C_BYTES;
#pragma unroll
                for (int pass = 0; pass < 2; ++pass) {
                    const unsigned int c = d + 256u * pass;  // chunk: weight row c / 4, weights 32 * (c % 4) .. +31 of the stage
                    const unsigned int row = c >> 2, q = c & 3u;
                    unsigned int w0, w1, w2, w3;
                    asm volatile("ld.shared.v4.b32 {%0, %1, %2, %3}, [%4];\n" : "=r"(w0), "=r"(w1), "=r"(w2), "=r"(w3) : "r"(c_tile + 16u * c));
                    unsigned int sc;
                    asm volatile("ld.shared.u16 %0, [%1];\n" : "=r"(sc) : "r"(s_tile + 16u * row + 8u * (kt & 1u) + 2u * q));
                    const uint2 t0 = brain_nw_lut[sc & 0xFFu];
                    const uint2 t1 = brain_nw_lut[sc >> 8];
                    unsigned int o[8];
                    brain_nw_decode8(w0, t0, o[0], o[1]);
                    brain_nw_decode8(w1, t0, o[2], o[3]);
                    brain_nw_decode8(w2, t1, o[4], o[5]);
                    brain_nw_decode8(w3, t1, o[6], o[7]);
                    // 16-byte units 2 q and 2 q + 1 of the row's 128 bytes, the 128-byte swizzle (unit ^ row mod 8).
                    const unsigned int rowb = b_tile + row * 128u;
                    brain_nw_st_v4(rowb + (((2u * q) ^ (row & 7u)) << 4), o[0], o[1], o[2], o[3]);
                    brain_nw_st_v4(rowb + (((2u * q + 1u) ^ (row & 7u)) << 4), o[4], o[5], o[6], o[7]);
                }
                brain_fwt_fence_proxy();
                brain_fwt_mbar_arrive(dec_bar + 8u * slot);
            }
        }
        return;
    }

    // Consumer warpgroup `wg`: rows 64 * wg .. +63 of each half (128 rows) of the 256-row tile; the half `h` is `t[h]`.
    asm volatile("setmaxnreg.inc.sync.aligned.u32 168;\n" ::: "memory");
    // The first warp of the first consumer warpgroup is also the producer: it builds the tensor maps and starts the ring (a lane of it refills
    // a slot once every consumer has released it, below), so no warp is spent on loads and the registers go to the accumulators.
    const unsigned int my_tiles = blockIdx.x < tiles ? (tiles - blockIdx.x + gridDim.x - 1u) / gridDim.x : 0u;
    const unsigned int total = my_tiles * kblocks;
    const bool producer = __shfl_sync(0xffffffffu, tid < 32u ? 1 : 0, 0) != 0;  // a shuffle, so the compiler keeps the role warp-uniform
    unsigned int smid;
    asm("mov.u32 %0, %%smid;\n" : "=r"(smid));
    const unsigned long long* const gmaps = &brain_nw_maps[smid % BRAIN_NW_MAX_CTAS][0][0];
    if (producer) {
        unsigned long long* const wmaps = &brain_nw_maps[smid % BRAIN_NW_MAX_CTAS][0][0];
        brain_fwt_build_map(brain_fwt_template, &brain_nw_smem_maps[0][0], wmaps, xq, K, M, lane);
        brain_fwt_build_map(brain_nw_codes_template, &brain_nw_smem_maps[1][0], wmaps + 16, wq, K / 2u, N, lane);
        brain_fwt_build_map(brain_nw_scales_template, &brain_nw_smem_maps[2][0], wmaps + 32, sq, K / 16u, N, lane);
        if (lane == 0u) {
            for (unsigned int p = 0; p < min((unsigned int)STAGES, total); ++p) { brain_nw_issue(gmaps, p, kblocks, tiles_m, tiles_n, base, full_bar); }
        }
        __syncwarp();
    }
    const unsigned int cwg = wg;
    const unsigned int g = lane >> 2, tq = lane & 3u;
    const unsigned int frow = cwg * 64u + warp * 16u + g;  // the thread's row in a half (and frow + 8)
    const unsigned long long a_desc = brain_fwt_desc(base + cwg * (64u * 128u));
    const unsigned long long b_desc = brain_fwt_desc(base + BRAIN_NW_A_BYTES);
    unsigned int it = 0;
    for (unsigned int tile = blockIdx.x; tile < tiles; tile += gridDim.x) {
        unsigned int m0, n0;
        brain_nw_tile(tile, tiles_m, tiles_n, m0, n0);
        // The activation scale of each of the thread's four rows (half, row g or g + 8); a row past M has none and is never stored.
        const bool block_live1 = m0 + 128u + cwg * 64u < M;
        float sc[2][2];
#pragma unroll
        for (int h = 0; h < 2; ++h) {
#pragma unroll
            for (int e = 0; e < 2; ++e) {
                const unsigned int row = m0 + 128u * h + frow + 8u * e;
                sc[h][e] = row < M ? sx[row] * wscale : 0.0f;
            }
        }
        float t0[64], t1[64];
#pragma unroll
        for (int i = 0; i < 64; ++i) { t0[i] = 0.0f; t1[i] = 0.0f; }
        // The groups of the last two stages are in flight: the slot of a stage is released once the next stage's products are issued and the
        // stage's own have completed (`wait_group 1`).
        auto release = [&](unsigned int i) {
            const unsigned int rs = i % STAGES;
            brain_fwt_mbar_arrive(empty_bar + 8u * rs);  // the slot's tiles are read
            if (producer) {
                // Refill the slot for iteration i + STAGES once the other consumer warpgroup has released it as well.
                if (lane == 0u && i + STAGES < total) {
                    brain_fwt_mbar_wait(empty_bar + 8u * rs, (i / STAGES) & 1u);
                    brain_nw_issue(gmaps, i + STAGES, kblocks, tiles_m, tiles_n, base, full_bar);
                }
                __syncwarp();
            }
        };
        if (block_live1) {
            for (unsigned int kt = 0; kt < kblocks; ++kt, ++it) {
                const unsigned int slot = it % STAGES;
                brain_fwt_mbar_wait(full_bar + 8u * slot, (it / STAGES) & 1u);
                brain_fwt_mbar_wait(dec_bar + 8u * slot, (it / STAGES) & 1u);
                const unsigned long long slot_units = static_cast<unsigned long long>(slot) * (BRAIN_NW_STAGE_BYTES >> 4);
                const unsigned long long da0 = a_desc + slot_units, da1 = da0 + ((128u * 128u) >> 4), db = b_desc + slot_units;
                brain_fwt_fence();
#pragma unroll
                for (int q = 0; q < 4; ++q) {
                    brain_nw_wgmma(t0, da0 + 2ull * q, db + 2ull * q, 1);
                    brain_nw_wgmma(t1, da1 + 2ull * q, db + 2ull * q, 1);
                }
                brain_fwt_wg_commit();
                brain_fwt_wg_wait<1>();
                if (kt > 0u) { release(it - 1u); }
            }
            brain_fwt_wg_wait<0>();
            release(it - 1u);
        } else {
            // Only the first half has rows.
            for (unsigned int kt = 0; kt < kblocks; ++kt, ++it) {
                const unsigned int slot = it % STAGES;
                brain_fwt_mbar_wait(full_bar + 8u * slot, (it / STAGES) & 1u);
                brain_fwt_mbar_wait(dec_bar + 8u * slot, (it / STAGES) & 1u);
                const unsigned long long slot_units = static_cast<unsigned long long>(slot) * (BRAIN_NW_STAGE_BYTES >> 4);
                const unsigned long long da0 = a_desc + slot_units, db = b_desc + slot_units;
                brain_fwt_fence();
#pragma unroll
                for (int q = 0; q < 4; ++q) { brain_nw_wgmma(t0, da0 + 2ull * q, db + 2ull * q, 1); }
                brain_fwt_wg_commit();
                brain_fwt_wg_wait<1>();
                if (kt > 0u) { release(it - 1u); }
            }
            brain_fwt_wg_wait<0>();
            release(it - 1u);
        }
        // out[m, n] = t * sx[row] * global * 8: thread layout of the m64n128 accumulator: rows g and g + 8 of the warp's 16, columns 8 j + 2 tq, +1.
#pragma unroll
        for (int h = 0; h < 2; ++h) {
#pragma unroll
            for (int j = 0; j < 16; ++j) {
                const unsigned int col = n0 + 8u * j + 2u * tq;
                if (col < N) {
#pragma unroll
                    for (int e = 0; e < 2; ++e) {
                        if (m0 + 128u * h + frow + 8u * e < M) {
                            const float* tt = h == 0 ? t0 : t1;
                            const float f = sc[h][e];
                            *reinterpret_cast<float2*>(out + static_cast<unsigned long long>(m0 + 128u * h + frow + 8u * e) * N + col) = make_float2(tt[4 * j + 2 * e] * f, tt[4 * j + 2 * e + 1] * f);
                        }
                    }
                }
            }
        }
    }
}
