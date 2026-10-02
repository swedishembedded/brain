// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for prompt-processing throughput on
// sparse mixture-of-experts language models for its clients. If your team needs
// expertise in running a quantised expert bank on int8 tensor cores without
// changing a single output bit then you can procure our services by sending an
// email to info@swedishembedded.com.
//
// The sparse-MoE int8 GEMM over a fused expert bank on int8 TENSOR CORES
// (`mma.sync.m16n8k32`): the CUDA form of `moe_i8_grouped.wgsl`, with its
// identical argument contract and its identical RESULT TO THE LAST BIT.
//
//   params : u32 [tiles, kg, n, xdiv, ne]
//   xq     : [rows, kg] u32          4 int8 per word
//   sx     : [rows] f32              per-row activation scale
//   tab    : [2 * (ne + 1)] u32      expert offsets, then each expert's first tile
//   perm   : [slots] u32             slots in expert-major order
//   wq     : [ne * n, kg] u32        the bank
//   sw     : [ne * n, kg/8] f32      one scale per 32 weights
//   out    : [slots, n] f32          slot-major
//
//   out[s, col] = sx[s / xdiv] * sum_g f32(dot_g) * sw[row_w, g],  row_w = expert(s) * n + col
//
// A "tile" is up to 8 slots of one expert (the routing tables of
// `moe_route_*.wgsl` are built for that size). A warp computes 16 weight rows
// against the tile's 8 slots with one MMA per 32-element weight group: the
// weights are the A operand (16 x 32) and the activations the B operand
// (32 x 8), so the 16 x 8 result is [weight row, slot] and one MMA is exactly
// one scale group.
//
// How it stays bit-identical to the WGSL kernel
// ---------------------------------------------
// A group's integer dot product is exact however it is summed, so the MMA gives
// the WGSL kernel's `dot32` bit for bit. What is not free is the f32 arithmetic
// around it, which the WGSL kernel fixes: every output has 16 lane accumulators,
// lane l adding `f32(dot_g) * sw[g]` for the groups g = l, l+16, ... in ascending
// order, and the 16 are folded in ascending lane order from 0.0. Here each
// thread owns four outputs of a warp's tile and keeps their 16 lane accumulators
// (64 registers); the group number picks the accumulator at compile time because
// the loop over groups is unrolled in blocks of 16. The fold is then a plain
// in-register sum. Every operation is an explicit round-to-nearest intrinsic, so
// no contraction can change a bit.
//
// The conversion of the integer result to f32 is free: the MMA accumulator is
// seeded with 0x4B400000 (the float 1.5 * 2^23), so the int32 i comes back as
// the bit pattern of 1.5 * 2^23 + i, and subtracting the bias is exact (|i| <=
// 32 * 128 * 128 < 2^22).
//
// Operand order. The k index within a group is a bijection shared by both
// operands, so a thread loads 8 CONTIGUOUS bytes of its row (a 32-byte group is
// then one sector per row, four threads wide) and feeds them to a0/a2 and b0/b1
// instead of the layout the instruction's figure shows; the integer sum is
// order-free.
//
// No `__restrict__`: brain's device buffers alias by design.

#define BRAIN_MG_WARPS 4
#define BRAIN_MG_THREADS (BRAIN_MG_WARPS * 32)
#define BRAIN_MG_COLS (BRAIN_MG_WARPS * 16)   // weight rows per block
#define BRAIN_MG_MR 8                         // slots per tile
#define BRAIN_MG_LANES 16                     // the WGSL kernel's lane accumulators
#define BRAIN_MG_UNROLL 4                     // groups whose loads are issued together

#define FLOAT_BIAS_BITS 0x4B400000u           // 1.5 * 2^23
#define FLOAT_BIAS 12582912.0f

// d = a * b + bias, per element, in int32.
__device__ __forceinline__ void brain_mma_s8(unsigned int (&d)[4], const unsigned int (&a)[4], unsigned int b0, unsigned int b1) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %10, %10, %10};\n"
                 : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(FLOAT_BIAS_BITS));
}

// Eight bytes (two words) from `p`: one load when the pointer is 8-byte
// aligned, two otherwise.
template <bool VEC>
__device__ __forceinline__ uint2 brain_ld8(const unsigned int* p) {
    if (VEC) { return __ldg(reinterpret_cast<const uint2*>(p)); }
    return make_uint2(__ldg(p), __ldg(p + 1));
}

template <bool VEC>
__device__ __forceinline__ void brain_moe_grouped(const unsigned int* params, const unsigned int* xq, const float* sx,
                                                  const unsigned int* tab, const unsigned int* perm,
                                                  const unsigned int* wq, const float* sw, float* out) {
    const unsigned int kg = params[1];
    const unsigned int n = params[2];
    const unsigned int xdiv = params[3];
    const unsigned int ne = params[4];

    // Flat 1-D block count the host may wrap into a second grid dimension. The
    // tile is the slow index: blocks that run together share a tile's
    // activations, and consecutive tiles of one expert find its weights in L2.
    const unsigned int blk = blockIdx.y * gridDim.x + blockIdx.x;
    const unsigned int col_blocks = (n + BRAIN_MG_COLS - 1u) / BRAIN_MG_COLS;
    if (col_blocks == 0u) { return; }
    const unsigned int tile = blk / col_blocks;
    const unsigned int cb = blk - tile * col_blocks;
    if (tile >= tab[2u * ne + 1u]) { return; }  // block-uniform: past the router's last tile

    // The expert whose tile range holds `tile`: the last e with tile_start[e] <= tile.
    unsigned int lo = 0u, hi = ne;
    while (hi - lo > 1u) {
        const unsigned int mid = (lo + hi) >> 1;
        if (tab[ne + 1u + mid] <= tile) { lo = mid; } else { hi = mid; }
    }
    const unsigned int e = lo;
    const unsigned int first = tab[e] + (tile - tab[ne + 1u + e]) * BRAIN_MG_MR;
    const unsigned int cnt = min(BRAIN_MG_MR, tab[e + 1u] - first);

    const unsigned int warp = threadIdx.x >> 5;
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int g = lane >> 2;      // MMA groupID: a weight row pair, and an activation slot
    const unsigned int tq = lane & 3u;     // MMA threadID_in_group
    const unsigned int colbase = cb * BRAIN_MG_COLS + warp * 16u;
    if (colbase >= n) { return; }          // warp-uniform

    const unsigned int ng = kg >> 3;
    const unsigned int col0 = colbase + g, col1 = colbase + g + 8u;
    const bool ok0 = col0 < n, ok1 = col1 < n;
    // A row past the matrix reads the last real row and its result is dropped.
    const unsigned long long row0 = static_cast<unsigned long long>(e) * n + (ok0 ? col0 : n - 1u);
    const unsigned long long row1 = static_cast<unsigned long long>(e) * n + (ok1 ? col1 : n - 1u);
    const unsigned int* w0 = wq + row0 * kg + tq * 2u;
    const unsigned int* w1 = wq + row1 * kg + tq * 2u;
    const float* s0p = sw + row0 * ng;
    const float* s1p = sw + row1 * ng;
    // This thread's activation row: slot `g` of the tile (an empty one reads slot 0's).
    const unsigned int xrow = perm[first + (g < cnt ? g : 0u)] / xdiv;
    const unsigned int* xp = xq + static_cast<unsigned long long>(xrow) * kg + tq * 2u;

    float acc[BRAIN_MG_LANES][4];
#pragma unroll
    for (int l = 0; l < BRAIN_MG_LANES; ++l) {
#pragma unroll
        for (int i = 0; i < 4; ++i) { acc[l][i] = 0.0f; }
    }

    for (unsigned int sb = 0; sb < ng; sb += BRAIN_MG_LANES) {
#pragma unroll
        for (int blk4 = 0; blk4 < BRAIN_MG_LANES / BRAIN_MG_UNROLL; ++blk4) {
            uint2 wa[BRAIN_MG_UNROLL], wb[BRAIN_MG_UNROLL], xb[BRAIN_MG_UNROLL];
            float sa[BRAIN_MG_UNROLL], sbv[BRAIN_MG_UNROLL];
            // Every load of the stage before any is consumed.
#pragma unroll
            for (int u = 0; u < BRAIN_MG_UNROLL; ++u) {
                const unsigned int gi = sb + blk4 * BRAIN_MG_UNROLL + u;
                if (gi < ng) {
                    const unsigned int off = gi * 8u;
                    wa[u] = brain_ld8<VEC>(w0 + off);
                    wb[u] = brain_ld8<VEC>(w1 + off);
                    xb[u] = brain_ld8<VEC>(xp + off);
                    sa[u] = __ldg(s0p + gi);
                    sbv[u] = __ldg(s1p + gi);
                }
            }
#pragma unroll
            for (int u = 0; u < BRAIN_MG_UNROLL; ++u) {
                const unsigned int gi = sb + blk4 * BRAIN_MG_UNROLL + u;
                if (gi < ng) {
                    const unsigned int a[4] = {wa[u].x, wb[u].x, wa[u].y, wb[u].y};
                    unsigned int d[4];
                    brain_mma_s8(d, a, xb[u].x, xb[u].y);
                    const int l = blk4 * BRAIN_MG_UNROLL + u;   // group gi's lane accumulator
                    acc[l][0] = __fadd_rn(acc[l][0], __fmul_rn(__fsub_rn(__uint_as_float(d[0]), FLOAT_BIAS), sa[u]));
                    acc[l][1] = __fadd_rn(acc[l][1], __fmul_rn(__fsub_rn(__uint_as_float(d[1]), FLOAT_BIAS), sa[u]));
                    acc[l][2] = __fadd_rn(acc[l][2], __fmul_rn(__fsub_rn(__uint_as_float(d[2]), FLOAT_BIAS), sbv[u]));
                    acc[l][3] = __fadd_rn(acc[l][3], __fmul_rn(__fsub_rn(__uint_as_float(d[3]), FLOAT_BIAS), sbv[u]));
                }
            }
        }
    }

    // The fold, ascending lane order from zero, as the WGSL kernel does it.
    float total[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        float t = 0.0f;
#pragma unroll
        for (int l = 0; l < BRAIN_MG_LANES; ++l) { t = __fadd_rn(t, acc[l][i]); }
        total[i] = t;
    }

    // d[0], d[1]: weight row `g`, slots 2*tq and 2*tq+1; d[2], d[3]: row `g + 8`.
#pragma unroll
    for (int j = 0; j < 2; ++j) {
        const unsigned int s = 2u * tq + j;
        if (s < cnt) {
            const unsigned int slot = perm[first + s];
            const float scale = sx[slot / xdiv];
            float* o = out + static_cast<unsigned long long>(slot) * n;
            if (ok0) { o[col0] = __fmul_rn(total[j], scale); }
            if (ok1) { o[col1] = __fmul_rn(total[2 + j], scale); }
        }
    }
}

extern "C" __global__ void __launch_bounds__(BRAIN_MG_THREADS)
brain_moe_i8_grouped_mma(const unsigned int* params, const unsigned int* xq, const float* sx,
                         const unsigned int* tab, const unsigned int* perm,
                         const unsigned int* wq, const float* sw, float* out) {
    const bool aligned = ((reinterpret_cast<unsigned long long>(xq) | reinterpret_cast<unsigned long long>(wq)) & 7ull) == 0ull;
    if (aligned) {
        brain_moe_grouped<true>(params, xq, sx, tab, perm, wq, sw, out);
    } else {
        brain_moe_grouped<false>(params, xq, sx, tab, perm, wq, sw, out);
    }
}
