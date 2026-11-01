// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for prompt-processing throughput on FP8 language-model releases for its
// clients. If your team needs expertise in feeding Hopper FP8 tensor cores through the tensor memory accelerator from
// block-scaled checkpoints, then you can procure our services by sending an email to info@swedishembedded.com.
//
// `matmul_fp8_wgmma_tma.cu` with a 128 x 256 tile: each consumer warpgroup issues `wgmma m64n256k32`, so the weight tile is read from
// shared memory once for 256 columns and the operand bytes a FLOP drop by a sixth. The 128-wide block scales are not folded into a
// second accumulator set (128 + 128 registers do not fit): the accumulator carries the running sum divided by the scale of the last
// 128 of K, and a block starts with one multiply by the ratio of the old and new scale, after which the tensor core adds the block's
// products straight in. Same parameters as `matmul_fp8_wgmma_tma`; the sums agree to f32 rounding, not bit for bit.
//
// The last, part-filled round of the persistent grid is split along N (`brain_fwt_jobs`): when at most half the grid is needed for the leftover
// tiles, each is run as two 128-column halves (`wgmma m64n128k32`) by two blocks, so the round takes about half a tile's time instead of a whole one.

#ifndef BRAIN_FWT_STAGES
#define BRAIN_FWT_STAGES 4
#endif
#ifndef BRAIN_FWT_ENTRY
#define BRAIN_FWT_ENTRY brain_matmul_fp8_wgmma_tma_n256
#endif
// BRAIN_FWT_OUT_BF16: the output rows are bf16 (`out` is [M, N] bf16, N a multiple of 8), accumulated and scaled in f32 and rounded once
// on the way out: the activation stream between two matrix products then carries 2 bytes an element. The store path is the f32 one: the
// warp's 16 rows go through shared memory and the tensor memory accelerator stores 64-byte boxes (32 columns), the same byte-typed
// output map with twice as many columns to a box. Direct 4-byte stores from the accumulator layout wrote half-sectors and ran up to 29 %
// slower than the f32 kernel (knowledge 394).
#define BRAIN_FWT_THREADS 384
#define BRAIN_FWT_MAX_CTAS 132  // grid size at most, and the SM slots of the maps

// The same for the output: a 64-byte box of 16 rows (16 floats a row), the 64-byte swizzle.
#define BRAIN_FWT_OUT_TEMPLATE 0x0000000000010000ull, 0x0000001000064010ull, 0x0000000000000000ull, 0x0000000000000000ull, 0x000000ff000000ffull, 0x0000000000000000ull, 0x3f00000000000000ull, 0x000000000000000full, 0x0000000000000400ull, 0x0000000000000200ull, 0x0000000000000000ull, 0x0000000000000000ull, 0x0000000000000000ull, 0x0000000000000000ull, 0x0000000000000000ull, 0x0000000000000000ull
__device__ __constant__ unsigned long long brain_fwt_out_template[16] = { BRAIN_FWT_OUT_TEMPLATE };
// Each SM's three maps (activations, weights, output), in global memory (the TMA reads a map from global, constant or parameter memory).
__device__ __align__(128) unsigned long long brain_fwt_maps[BRAIN_FWT_MAX_CTAS][3][16];

#define BRAIN_FWT_BM 128
#define BRAIN_FWT_BN 256

// Bytes per ring stage: the activation tile and the weight tile (128 rows x 128 bytes each), a whole number of KiB so every swizzled tile stays
// 1024-byte aligned. The block scales are not staged: the consumers read them from global memory a block ahead.
#define BRAIN_FWT_X_BYTES (BRAIN_FWT_BM * 128)
#define BRAIN_FWT_W_BYTES (BRAIN_FWT_BN * 128)
#define BRAIN_FWT_STAGE_BYTES (BRAIN_FWT_X_BYTES + BRAIN_FWT_W_BYTES)

#ifndef BRAIN_PDL_ENTRY
#define BRAIN_PDL_ENTRY() do {} while (0)
#endif

// Two floats as one word of bf16 (`lo` in the low half): `cvt.rn.bf16x2.f32`, round to nearest even, the rule `cast_f32_bf16` has.
__device__ __forceinline__ unsigned int brain_fwt_pack_bf16(float lo, float hi) {
    unsigned int r;
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;\n" : "=r"(r) : "f"(hi), "f"(lo));
    return r;
}

// D (64 x 256 f32, 128 registers per thread) = A (64 x 32 e4m3) . B (256 x 32 e4m3)^T, plus D when `accumulate` is set.
__device__ __forceinline__ void brain_fwt_wgmma(float (&d)[128], unsigned long long da, unsigned long long db, int accumulate) {
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %130, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n256k32.f32.e4m3.e4m3 "
        "{"
        "%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
        "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, "
        "%32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, "
        "%48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, "
        "%64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, "
        "%80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, "
        "%96, %97, %98, %99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111, "
        "%112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, %123, %124, %125, %126, %127}, "
        "%128, %129, p, 1, 1;\n"
        "}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
          "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
          "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),
          "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]),
          "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]),
          "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]),
          "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]),
          "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63]),
          "+f"(d[64]), "+f"(d[65]), "+f"(d[66]), "+f"(d[67]), "+f"(d[68]), "+f"(d[69]), "+f"(d[70]), "+f"(d[71]),
          "+f"(d[72]), "+f"(d[73]), "+f"(d[74]), "+f"(d[75]), "+f"(d[76]), "+f"(d[77]), "+f"(d[78]), "+f"(d[79]),
          "+f"(d[80]), "+f"(d[81]), "+f"(d[82]), "+f"(d[83]), "+f"(d[84]), "+f"(d[85]), "+f"(d[86]), "+f"(d[87]),
          "+f"(d[88]), "+f"(d[89]), "+f"(d[90]), "+f"(d[91]), "+f"(d[92]), "+f"(d[93]), "+f"(d[94]), "+f"(d[95]),
          "+f"(d[96]), "+f"(d[97]), "+f"(d[98]), "+f"(d[99]), "+f"(d[100]), "+f"(d[101]), "+f"(d[102]), "+f"(d[103]),
          "+f"(d[104]), "+f"(d[105]), "+f"(d[106]), "+f"(d[107]), "+f"(d[108]), "+f"(d[109]), "+f"(d[110]), "+f"(d[111]),
          "+f"(d[112]), "+f"(d[113]), "+f"(d[114]), "+f"(d[115]), "+f"(d[116]), "+f"(d[117]), "+f"(d[118]), "+f"(d[119]),
          "+f"(d[120]), "+f"(d[121]), "+f"(d[122]), "+f"(d[123]), "+f"(d[124]), "+f"(d[125]), "+f"(d[126]), "+f"(d[127])
        : "l"(da), "l"(db), "r"(accumulate));
}

// The same for 64 x 128 (64 registers): the half tile a split tile's block runs (see `brain_fwt_jobs`).
__device__ __forceinline__ void brain_fwt_wgmma(float (&d)[64], unsigned long long da, unsigned long long db, int accumulate) {
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

// a / s with the hardware reciprocal (`rcp.approx.ftz`, 1 ulp, no range scaling): the scales are products of two positive normal floats far from
// the edges of the range, so the library division's denormal handling (about ten instructions a quotient) never applies.
__device__ __forceinline__ float brain_fwt_ratio(float a, float s) {
    float r;
    asm("rcp.approx.ftz.f32 %0, %1;\n" : "=f"(r) : "f"(s));
    return a * r;
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

// A CTA's jobs: the whole 128 x 256 tiles `blockIdx.x + j * gridDim.x` below `whole_tiles`, then at most one half tile. When the tiles do not fill
// the last round of the persistent grid and `rem` of them are left over with `2 * rem` at most the grid, each of those is split into its two 128-column
// halves, run by two blocks (both consumer warpgroups on `wgmma m64n128k32`) instead of `rem` blocks running whole tiles while the rest of the grid
// idles: the last round takes about half a tile's time. `part` is 2 for a whole tile and 0 or 1 for the half.
struct brain_fwt_jobs {
    unsigned int whole_tiles, whole_per_block, count;
    __device__ __forceinline__ brain_fwt_jobs(unsigned int tiles, unsigned int block) {
        const unsigned int grid = gridDim.x;
        const unsigned int rem = tiles % grid;
        const bool split = rem != 0u && 2u * rem <= grid;
        whole_tiles = split ? tiles - rem : tiles;
        whole_per_block = whole_tiles > block ? (whole_tiles - block + grid - 1u) / grid : 0u;
        count = whole_per_block + ((split && block < 2u * rem) ? 1u : 0u);
    }
    __device__ __forceinline__ void at(unsigned int j, unsigned int block, unsigned int& tile, unsigned int& part) const {
        if (j < whole_per_block) { tile = block + j * gridDim.x; part = 2u; }
        else { tile = whole_tiles + block / 2u; part = block & 1u; }
    }
};

// The first output column of a job: the tile's, plus 128 for the second half of a split tile.
__device__ __forceinline__ unsigned int brain_fwt_job_column(unsigned int n0, unsigned int part) { return part == 1u ? n0 + 128u : n0; }

// The state of a consumer warpgroup thread across the jobs of a block: its rows, the ring position, the next job's first scales and the
// output map. A struct with a member template rather than a lambda because the job runs with 32 groups of 8 columns (a whole tile) or 16 (a half).
struct brain_fwt_consumer {
    const float* sx; const float* sw;
    unsigned int M, kblocks, nblocks;
    unsigned int cwg, warp, lane, frow;
    unsigned int full_bar, empty_bar, ebuf_addr, wrow_in_tile;
    unsigned long long a_desc, b_desc;
    const unsigned long long* gmaps_out;
    unsigned int it;
    float a0, a1, b0, b1;  // scales of the first block of a tile for this thread's two rows and the tile's two weight column blocks
    bool first_requested, out_map_acquired;

    __device__ __forceinline__ bool row_live(unsigned int tile_m0, unsigned int offset) const { return tile_m0 + frow + offset < M; }
    __device__ __forceinline__ void first_scales(unsigned int tile_m0, unsigned int tile_n0) {
        a0 = row_live(tile_m0, 0u) ? sx[(unsigned long long)(tile_m0 + frow) * kblocks] : 0.0f;
        a1 = row_live(tile_m0, 8u) ? sx[(unsigned long long)(tile_m0 + frow + 8u) * kblocks] : 0.0f;
        b0 = sw[(unsigned long long)min(tile_n0 >> 7, nblocks - 1u) * kblocks];
        b1 = sw[(unsigned long long)min((tile_n0 >> 7) + 1u, nblocks - 1u) * kblocks];
    }

    // One job: a tile of NG groups of 8 columns from column n0 (NG 32: a whole tile, 16: the half of a split tile).
    template <int NG>
    __device__ __forceinline__ void run(unsigned int m0, unsigned int n0, bool has_next, unsigned int next_m0, unsigned int next_n0) {
        constexpr int STAGES = BRAIN_FWT_STAGES;
        const unsigned int g = lane >> 2, tq = lane & 3u;
        // t holds the tile's running sum divided by the scale `se[row][half]` of the last 128 of K, so the tensor core adds the next block's
        // products straight into it after one multiply by the ratio of the two scales (no second accumulator set).
        float t[4 * NG];
#pragma unroll
        for (int i = 0; i < 4 * NG; ++i) { t[i] = 0.0f; }
        float se[2][2] = {{1.0f, 1.0f}, {1.0f, 1.0f}};
        // The scales are read from global memory a block ahead of their use (rows past M have scale 0, as the zero-filled copies had), so the
        // loads' latency hides behind the block before; the first block's were requested before the previous tile's epilogue.
        const float* const sxp0 = sx + (unsigned long long)(row_live(m0, 0u) ? m0 + frow : 0u) * kblocks;
        const float* const sxp1 = sx + (unsigned long long)(row_live(m0, 8u) ? m0 + frow + 8u : 0u) * kblocks;
        const float live0 = row_live(m0, 0u) ? 1.0f : 0.0f, live1 = row_live(m0, 8u) ? 1.0f : 0.0f;
        const float* const swp0 = sw + (unsigned long long)min(n0 >> 7, nblocks - 1u) * kblocks;
        const float* const swp1 = sw + (unsigned long long)min((n0 >> 7) + 1u, nblocks - 1u) * kblocks;
        if (!first_requested) { first_scales(m0, n0); }
        for (unsigned int kt = 0; kt < kblocks; ++kt, ++it) {
            const unsigned int slot = it % STAGES;
            brain_fwt_mbar_wait(full_bar + 8u * slot, (it / STAGES) & 1u);
            // The descriptors of the stage's first 32 of K (16-byte units); the next 32 are 2 units on.
            const unsigned long long slot_units = static_cast<unsigned long long>(slot) * (BRAIN_FWT_STAGE_BYTES >> 4);
            const unsigned long long da = a_desc + slot_units, db = b_desc + slot_units;

            const unsigned int nk = min(kt + 1u, kblocks - 1u);
            const float na0 = sxp0[nk] * live0, na1 = sxp1[nk] * live1, nb0 = swp0[nk], nb1 = swp1[nk];
            const float xs[2] = {a0, a1}, ws[2] = {b0, b1};
            a0 = na0; a1 = na1; b0 = nb0; b1 = nb1;
            float r[2][2];
#pragma unroll
            for (int row = 0; row < 2; ++row) {
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    float s = xs[row] * ws[h];
                    if (!(s > 0.0f)) { s = se[row][h]; }  // a block of zeros: the sum keeps its scale
                    r[row][h] = brain_fwt_ratio(se[row][h], s);
                    se[row][h] = s;
                }
            }
#pragma unroll
            for (int j = 0; j < NG; ++j) {
                const int h = j >> 4;
                t[4 * j] *= r[0][h];
                t[4 * j + 1] *= r[0][h];
                t[4 * j + 2] *= r[1][h];
                t[4 * j + 3] *= r[1][h];
            }
            brain_fwt_fence();
#pragma unroll
            for (int q = 0; q < 4; ++q) { brain_fwt_wgmma(t, da + 2ull * q, db + 2ull * q, 1); }
            brain_fwt_wg_commit();
            brain_fwt_wg_wait<0>();
            brain_fwt_mbar_arrive(empty_bar + 8u * slot);  // the slot's tiles and scales are read
        }
        // The next tile's first scales are requested now, so the epilogue hides their latency.
        first_requested = has_next;
        if (has_next) { first_scales(next_m0, next_n0); }

        // out[m, n] = t * scale: the warp's 16 rows go to shared memory 16 columns at a time (thread layout of the m64n256 accumulator: rows r and
        // r + 8 of the 16, columns 8 j + 2 t, +1) and the tensor memory accelerator stores each 16 x 16 box, so the store costs the load-store
        // unit one shared-memory write a row pair instead of a global write per sector. Two boxes a warp are in flight.
        if (!out_map_acquired) {
            // The producer published the map before its first load, which this warpgroup has waited for.
            asm volatile("fence.proxy.tensormap::generic.acquire.gpu [%0], 128;\n" ::"l"(gmaps_out) : "memory");
            out_map_acquired = true;
        }
        const unsigned int wrow0 = m0 + wrow_in_tile;  // first output row of this warp
#ifdef BRAIN_FWT_OUT_BF16
        // bf16: a 64-byte box row is 32 columns, so a pass takes four column groups of 8 (one 16-byte chunk each) and there are 8 passes.
#pragma unroll
        for (int pass = 0; pass < NG / 4; ++pass) {
            const unsigned int buf = ebuf_addr + (pass & 1) * 1024u;
            if (pass >= 2) {
                if (lane == 0u) { brain_fwt_bulk_wait_read<1>(); }
                __syncwarp();
            }
#pragma unroll
            for (int jj = 0; jj < 4; ++jj) {
                const int j = 4 * pass + jj;
                const int h = j >> 4;
#pragma unroll
                for (int half = 0; half < 2; ++half) {
                    const unsigned int row = g + 8u * half;
                    const unsigned int word = brain_fwt_pack_bf16(t[4 * j + 2 * half] * se[half][h], t[4 * j + 2 * half + 1] * se[half][h]);
                    const unsigned int off = row * 64u + ((static_cast<unsigned int>(jj) ^ ((row >> 1) & 3u)) << 4) + tq * 4u;
                    asm volatile("st.shared.b32 [%0], %1;\n" ::"r"(buf + off), "r"(word) : "memory");
                }
            }
            brain_fwt_fence_proxy();
            __syncwarp();
            if (lane == 0u) {
                brain_fwt_tma_store(gmaps_out, buf, (n0 + 32u * pass) * 2u, wrow0);
                brain_fwt_bulk_commit();
            }
        }
#else
#pragma unroll
        for (int pass = 0; pass < NG / 2; ++pass) {
            const unsigned int buf = ebuf_addr + (pass & 1) * 1024u;
            if (pass >= 2) {
                if (lane == 0u) { brain_fwt_bulk_wait_read<1>(); }
                __syncwarp();
            }
#pragma unroll
            for (int jj = 0; jj < 2; ++jj) {
                const int j = 2 * pass + jj;
                const int h = j >> 4;
                const unsigned int chunk = jj * 2u + (tq >> 1);  // 16-byte chunk of the 64-byte row
#pragma unroll
                for (int half = 0; half < 2; ++half) {
                    const unsigned int row = g + 8u * half;
                    const float v0 = t[4 * j + 2 * half] * se[half][h], v1 = t[4 * j + 2 * half + 1] * se[half][h];
                    const unsigned int off = row * 64u + ((chunk ^ ((row >> 1) & 3u)) << 4) + (tq & 1u) * 8u;
                    asm volatile("st.shared.v2.f32 [%0], {%1, %2};\n" ::"r"(buf + off), "f"(v0), "f"(v1) : "memory");
                }
            }
            brain_fwt_fence_proxy();
            __syncwarp();
            if (lane == 0u) {
                brain_fwt_tma_store(gmaps_out, buf, (n0 + 16u * pass) * 4u, wrow0);
                brain_fwt_bulk_commit();
            }
        }
#endif
    }
};

extern "C" __global__ void __launch_bounds__(BRAIN_FWT_THREADS, 1)
BRAIN_FWT_ENTRY(const unsigned int* params, const unsigned int* xq, const unsigned int* wq, const float* sx, const float* sw, float* out) {
    BRAIN_PDL_ENTRY();
    constexpr int STAGES = BRAIN_FWT_STAGES;
    extern __shared__ unsigned char brain_fwt_smem_raw[];
    __shared__ __align__(128) unsigned long long brain_fwt_smem_maps[3][16];
    const unsigned int M = params[0];
    const unsigned int K = params[1] * 4u;
    const unsigned int N = params[2];
    const unsigned int kblocks = K / 128u;
    const unsigned int nblocks = (N + 127u) / 128u;  // weight scale blocks along N
    const unsigned int tiles_m = (M + BRAIN_FWT_BM - 1u) / BRAIN_FWT_BM;
    const unsigned int tiles_n = (N + BRAIN_FWT_BN - 1u) / BRAIN_FWT_BN;
    const unsigned int tiles = tiles_m * tiles_n;

    const unsigned int tid = threadIdx.x;
    const unsigned int wg = __shfl_sync(0xffffffffu, tid / 128u, 0);  // 0: producer; a shuffle, so the compiler keeps the role in uniform registers; 1, 2: consumers of rows 64 * (wg - 1) ..
    const unsigned int lane = tid & 31u;
    const unsigned int warp = (tid >> 5) & 3u;

    const unsigned int raw_addr = brain_fwt_smem(brain_fwt_smem_raw);
    const unsigned int base = (raw_addr + 1023u) & ~1023u;
    const unsigned int full_bar = base + STAGES * BRAIN_FWT_STAGE_BYTES;  // STAGES mbarriers, then STAGES more
    const unsigned int empty_bar = full_bar + 8u * STAGES;
    const unsigned int epi_base = (empty_bar + 8u * STAGES + 1023u) & ~1023u;  // 16 boxes of 1 KiB: two a consumer warp

    if (tid == 0u) {
        for (int s = 0; s < STAGES; ++s) {
            brain_fwt_mbar_init(full_bar + 8u * s, 1u);     // the TMA thread's expect_tx arrival
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
        brain_fwt_build_map(brain_fwt_template, &brain_fwt_smem_maps[0][0], gmaps, xq, K, M, lane);
        brain_fwt_build_map(brain_fwt_template, &brain_fwt_smem_maps[1][0], gmaps + 16, wq, K, N, lane);
#ifdef BRAIN_FWT_OUT_BF16
        brain_fwt_build_map(brain_fwt_out_template, &brain_fwt_smem_maps[2][0], gmaps + 32, out, N * 2u, M, lane);
#else
        brain_fwt_build_map(brain_fwt_out_template, &brain_fwt_smem_maps[2][0], gmaps + 32, out, N * 4u, M, lane);
#endif
        unsigned int it = 0;
        const brain_fwt_jobs jobs(tiles, blockIdx.x);
        for (unsigned int j = 0; j < jobs.count; ++j) {
            unsigned int tile, part, m0, n0;
            jobs.at(j, blockIdx.x, tile, part);
            brain_fwt_tile(tile, tiles_m, tiles_n, m0, n0);
            n0 = brain_fwt_job_column(n0, part);
            const unsigned int w_bytes = part == 2u ? BRAIN_FWT_W_BYTES : BRAIN_FWT_W_BYTES / 2u;
            for (unsigned int kt = 0; kt < kblocks; ++kt, ++it) {
                const unsigned int slot = it % STAGES;
                if (it >= (unsigned int)STAGES) { brain_fwt_mbar_wait(empty_bar + 8u * slot, ((it / STAGES) - 1u) & 1u); }
                const unsigned int st = base + slot * BRAIN_FWT_STAGE_BYTES;
                if (lane == 0u) {
                    brain_fwt_mbar_arrive_expect_tx(full_bar + 8u * slot, BRAIN_FWT_X_BYTES + w_bytes);
                    brain_fwt_tma_load(st, gmaps, kt * 128, m0, full_bar + 8u * slot);
                    brain_fwt_tma_load(st + BRAIN_FWT_X_BYTES, gmaps + 16, kt * 128, n0, full_bar + 8u * slot);
                    if (part == 2u) { brain_fwt_tma_load(st + BRAIN_FWT_X_BYTES + 128u * 128u, gmaps + 16, kt * 128, n0 + 128u, full_bar + 8u * slot); }
                }
            }
        }
        brain_fwt_wait<0>();
        return;
    }

    asm volatile("setmaxnreg.inc.sync.aligned.u32 232;\n" ::: "memory");
    const unsigned int cwg = wg - 1u;  // rows of x 64 * cwg .. 64 * cwg + 63
    unsigned int smid;
    asm("mov.u32 %0, %%smid;\n" : "=r"(smid));
    brain_fwt_consumer c;
    c.sx = sx; c.sw = sw; c.M = M; c.kblocks = kblocks; c.nblocks = nblocks;
    c.cwg = cwg; c.warp = warp; c.lane = lane;
    c.frow = cwg * 64u + warp * 16u + (lane >> 2);
    c.full_bar = full_bar; c.empty_bar = empty_bar;
    c.ebuf_addr = epi_base + (cwg * 4u + warp) * 2048u;
    c.wrow_in_tile = cwg * 64u + warp * 16u;  // first output row of this warp within the tile
    c.a_desc = brain_fwt_desc(base + cwg * (64u * 128u)); c.b_desc = brain_fwt_desc(base + BRAIN_FWT_X_BYTES);
    c.gmaps_out = &brain_fwt_maps[smid % BRAIN_FWT_MAX_CTAS][2][0];
    c.it = 0u; c.a0 = c.a1 = c.b0 = c.b1 = 0.0f; c.first_requested = false; c.out_map_acquired = false;

    const brain_fwt_jobs jobs(tiles, blockIdx.x);
    for (unsigned int j = 0; j < jobs.count; ++j) {
        unsigned int tile, part, m0, n0, next_tile, next_part, next_m0 = 0, next_n0 = 0;
        jobs.at(j, blockIdx.x, tile, part);
        brain_fwt_tile(tile, tiles_m, tiles_n, m0, n0);
        n0 = brain_fwt_job_column(n0, part);
        const bool has_next = j + 1u < jobs.count;
        if (has_next) {
            jobs.at(j + 1u, blockIdx.x, next_tile, next_part);
            brain_fwt_tile(next_tile, tiles_m, tiles_n, next_m0, next_n0);
            next_n0 = brain_fwt_job_column(next_n0, next_part);
        }
        if (part == 2u) { c.run<32>(m0, n0, has_next, next_m0, next_n0); }
        else { c.run<16>(m0, n0, has_next, next_m0, next_n0); }
    }
    if (lane == 0u) { brain_fwt_bulk_wait<0>(); }
}
