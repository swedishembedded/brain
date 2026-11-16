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
//   sx     : [rows, kg/8] f32        one activation scale per 32 elements
//   tab    : [2 * (ne + 1)] u32      expert offsets, then each expert's first tile
//   perm   : [slots] u32             slots in expert-major order
//   wq     : [ne * n, kg] u32        the bank
//   sw     : [ne * n, kg/8] f32      one scale per 32 weights
//   out    : [slots, n] f32          slot-major
//
//   out[s, col] = sum_g f32(dot_g) * (sw[row_w, g] * sx[s / xdiv, g]),  row_w = expert(s) * n + col
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
// lane l adding `f32(dot_g) * (sw[g] * sx[g])` for the groups g = l, l+16, ... in ascending
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
    // The activation rows of the two slots whose outputs this thread holds
    // (B's columns 2*tq and 2*tq + 1): their group scales.
    const unsigned int sa_slot = 2u * tq, sb_slot = 2u * tq + 1u;
    const float* xsa = sx + static_cast<unsigned long long>(perm[first + (sa_slot < cnt ? sa_slot : 0u)] / xdiv) * ng;
    const float* xsb = sx + static_cast<unsigned long long>(perm[first + (sb_slot < cnt ? sb_slot : 0u)] / xdiv) * ng;

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
            float sa[BRAIN_MG_UNROLL], sbv[BRAIN_MG_UNROLL], xa[BRAIN_MG_UNROLL], xbs[BRAIN_MG_UNROLL];
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
                    xa[u] = __ldg(xsa + gi);
                    xbs[u] = __ldg(xsb + gi);
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
                    acc[l][0] = __fadd_rn(acc[l][0], __fmul_rn(__fsub_rn(__uint_as_float(d[0]), FLOAT_BIAS), __fmul_rn(sa[u], xa[u])));
                    acc[l][1] = __fadd_rn(acc[l][1], __fmul_rn(__fsub_rn(__uint_as_float(d[1]), FLOAT_BIAS), __fmul_rn(sa[u], xbs[u])));
                    acc[l][2] = __fadd_rn(acc[l][2], __fmul_rn(__fsub_rn(__uint_as_float(d[2]), FLOAT_BIAS), __fmul_rn(sbv[u], xa[u])));
                    acc[l][3] = __fadd_rn(acc[l][3], __fmul_rn(__fsub_rn(__uint_as_float(d[3]), FLOAT_BIAS), __fmul_rn(sbv[u], xbs[u])));
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
            float* o = out + static_cast<unsigned long long>(slot) * n;
            if (ok0) { o[col0] = total[j]; }
            if (ok1) { o[col1] = total[2 + j]; }
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

// ---------------------------------------------------------------------------
// Runs of tiles on staged weight rows (`brain_moe_i8_grouped_mma_runs*`)
// ---------------------------------------------------------------------------
//
// The kernel above reads a weight row once per 8-slot tile, so a prefill round,
// where an expert holds tens of tiles, streams every expert's rows once per tile:
// each weight byte meets 8 slots, and the GEMM is bound by those reads at a few
// percent of the tensor cores' rate. Here a block takes 32 weight rows and a RUN
// of up to BRAIN_MGR_RUN consecutive tiles: it copies the rows (all of K) and their
// scales into shared memory once per expert the run touches, and every tile of the
// run reads them there. Warp w computes rows 16 * (w & 1) .. + 16 of the tiles of
// parity w >> 1.
//
// Every output is computed by exactly the arithmetic above - the same MMA per
// group, the same 16 lane accumulators taken in ascending group order, the same
// fold - so the result is the WGSL kernel's to the last bit. Only where the
// weight words come from changes.
//
// Shared layout: the 32 rows at a stride of `kg` rounded so the stride is 8 mod 32
// words (a half-warp's 8-byte reads of four rows then cover the 32 banks once),
// then the scales at a stride of `ng + 1`. The registration's dynamic shared
// memory is sized for a largest `kg` (`kernels_cuda::moe_grouped_runs_shared_bytes`); the selection serves no wider
// `kg` than the registration was sized for.

#define BRAIN_MGR_ROWS 32
#define BRAIN_MGR_RUN 8
#define BRAIN_MGR_UNROLL 2                    // groups whose loads are issued together: registers, not latency, bound this form
// Each warp stages its tile's activations a chunk of 16 groups (512 elements) at a time, in two buffers: the copy of
// chunk c + 1 is in flight while chunk c is multiplied. A chunk is 8 slot rows of 128 words at a stride of 136 (8 mod
// 32, as the weight rows), then their 8 x 16 group scales.
#define BRAIN_MGR_CHUNK_WORDS 128
#define BRAIN_MGR_XSTRIDE 136
#define BRAIN_MGR_XBUF (8 * BRAIN_MGR_XSTRIDE + 8 * 16)

__device__ __forceinline__ unsigned int brain_mgr_stride(unsigned int kg) {
    return kg + ((8u - (kg & 31u)) & 31u);
}

__device__ __forceinline__ void brain_cp_async16(void* smem, const void* gmem) {
    const unsigned int s = static_cast<unsigned int>(__cvta_generic_to_shared(smem));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(gmem));
}

__device__ __forceinline__ void brain_cp_async4(void* smem, const void* gmem) {
    const unsigned int s = static_cast<unsigned int>(__cvta_generic_to_shared(smem));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"(s), "l"(gmem));
}

__device__ __forceinline__ void brain_cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }

template <int N>
__device__ __forceinline__ void brain_cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

// Copy chunk `c` of the tile's 8 activation rows (`xrows`, one per lane group of 4) and their scales into `buf`.
// `X16`: every row starts on 16 bytes (the buffer does and `kg` is a multiple of 4), so a row's 128 words go as 32
// copies of 16 bytes; otherwise word by word.
template <bool X16>
__device__ __forceinline__ void brain_mgr_stage_x(unsigned int* buf, const unsigned int* xq, const float* sx, const unsigned int (&xrows)[8],
                                                  unsigned int c, unsigned int kg, unsigned int ng, unsigned int lane) {
    const unsigned int w0 = c * BRAIN_MGR_CHUNK_WORDS;
    const unsigned int words = min(BRAIN_MGR_CHUNK_WORDS, kg - w0);
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        const unsigned int* src = xq + static_cast<unsigned long long>(xrows[r]) * kg + w0;
        unsigned int* dst = buf + r * BRAIN_MGR_XSTRIDE;
        if (X16) {
            if (lane * 4u < words) { brain_cp_async16(dst + lane * 4u, src + lane * 4u); }
        } else {
            for (unsigned int i = lane; i < words; i += 32u) { brain_cp_async4(dst + i, src + i); }
        }
    }
    const unsigned int g0 = c * 16u;
    const unsigned int groups = min(16u, ng - g0);
    float* sdst = reinterpret_cast<float*>(buf + 8 * BRAIN_MGR_XSTRIDE);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const unsigned int i = lane + 32u * j;   // 128 = 8 rows x 16 groups
        const unsigned int r = i >> 4, gi = i & 15u;
        if (gi < groups) { brain_cp_async4(sdst + i, sx + static_cast<unsigned long long>(xrows[r]) * ng + g0 + gi); }
    }
}

template <bool W16, bool X16>
__device__ __forceinline__ void brain_moe_grouped_runs(const unsigned int* params, const unsigned int* xq, const float* sx,
                                                       const unsigned int* tab, const unsigned int* perm,
                                                       const unsigned int* wq, const float* sw, float* out) {
    extern __shared__ uint4 brain_mgr_smem[];
    unsigned int* ws = reinterpret_cast<unsigned int*>(brain_mgr_smem);

    const unsigned int kg = params[1];
    const unsigned int n = params[2];
    const unsigned int xdiv = params[3];
    const unsigned int ne = params[4];
    const unsigned int ng = kg >> 3;
    const unsigned int stride = brain_mgr_stride(kg);
    const unsigned int sstride = ng + 1u;
    float* ss = reinterpret_cast<float*>(ws + BRAIN_MGR_ROWS * stride);

    const unsigned int blk = blockIdx.y * gridDim.x + blockIdx.x;
    const unsigned int col_blocks = (n + BRAIN_MGR_ROWS - 1u) / BRAIN_MGR_ROWS;
    if (col_blocks == 0u) { return; }
    const unsigned int run = blk / col_blocks;
    const unsigned int cb = blk - run * col_blocks;
    const unsigned int total = tab[2u * ne + 1u];
    const unsigned int tile0 = run * BRAIN_MGR_RUN;
    if (tile0 >= total) { return; }  // block-uniform
    const unsigned int tile_end = min(tile0 + BRAIN_MGR_RUN, total);

    const unsigned int tid = threadIdx.x;
    const unsigned int warp = tid >> 5;
    const unsigned int lane = tid & 31u;
    const unsigned int g = lane >> 2;
    const unsigned int tq = lane & 3u;
    const unsigned int rb = warp & 1u;      // which 16 of the block's 32 rows
    const unsigned int parity = warp >> 1;  // which tiles of the run
    const unsigned int rowbase = cb * BRAIN_MGR_ROWS;
    const unsigned int colbase = rowbase + rb * 16u;
    const bool active = colbase < n;        // warp-uniform; an idle warp still stages and synchronises
    const unsigned int col0 = colbase + g, col1 = colbase + g + 8u;
    const bool ok0 = col0 < n, ok1 = col1 < n;
    const unsigned int* w0 = ws + (rb * 16u + g) * stride + tq * 2u;
    const unsigned int* w1 = ws + (rb * 16u + g + 8u) * stride + tq * 2u;
    const float* s0p = ss + (rb * 16u + g) * sstride;
    const float* s1p = ss + (rb * 16u + g + 8u) * sstride;
    // This warp's two activation buffers, after the staged rows and their scales (16-byte aligned: the stride and
    // the scales' rows are whole words, rounded up below).
    const unsigned int a_words = (BRAIN_MGR_ROWS * (stride + sstride) + 3u) & ~3u;
    unsigned int* xbuf = ws + a_words + warp * 2u * BRAIN_MGR_XBUF;
    const unsigned int chunks = (kg + BRAIN_MGR_CHUNK_WORDS - 1u) / BRAIN_MGR_CHUNK_WORDS;

    unsigned int t = tile0;
    while (t < tile_end) {
        // The expert whose tile range holds `t`, and where its tiles end within this run.
        unsigned int lo = 0u, hi = ne;
        while (hi - lo > 1u) {
            const unsigned int mid = (lo + hi) >> 1;
            if (tab[ne + 1u + mid] <= t) { lo = mid; } else { hi = mid; }
        }
        const unsigned int e = lo;
        const unsigned int seg_end = min(tile_end, tab[ne + 2u + e]);

        // Stage the expert's 32 rows and their scales (a row past the matrix stages the last real row; its
        // results are dropped).
        // Every copy is issued before any is waited for (a load-then-store loop waits a memory latency per word).
        __syncthreads();  // the previous expert's readers are done
        const unsigned int vecs = kg >> 2;  // 16-byte vectors of a row
        for (unsigned int i = tid; i < BRAIN_MGR_ROWS * vecs; i += blockDim.x) {
            const unsigned int r = i / vecs, c = (i - r * vecs) * 4u;
            const unsigned long long row = static_cast<unsigned long long>(e) * n + min(rowbase + r, n - 1u);
            if (W16) {
                brain_cp_async16(ws + r * stride + c, wq + row * kg + c);
            } else {
#pragma unroll
                for (int j = 0; j < 4; ++j) { brain_cp_async4(ws + r * stride + c + j, wq + row * kg + c + j); }
            }
        }
        for (unsigned int i = tid; i < BRAIN_MGR_ROWS * ng; i += blockDim.x) {
            const unsigned int r = i / ng, c = i - r * ng;
            const unsigned long long row = static_cast<unsigned long long>(e) * n + min(rowbase + r, n - 1u);
            brain_cp_async4(ss + r * sstride + c, sw + row * ng + c);
        }
        brain_cp_async_commit();
        brain_cp_async_wait<0>();
        __syncthreads();

        for (unsigned int tile = t + parity; active && tile < seg_end; tile += 2u) {
            const unsigned int first = tab[e] + (tile - tab[ne + 1u + e]) * BRAIN_MG_MR;
            const unsigned int cnt = min(BRAIN_MG_MR, tab[e + 1u] - first);
            // The tile's 8 activation rows (an empty slot reads slot 0's).
            unsigned int xrows[8];
#pragma unroll
            for (int r = 0; r < 8; ++r) { xrows[r] = perm[first + (static_cast<unsigned int>(r) < cnt ? static_cast<unsigned int>(r) : 0u)] / xdiv; }

            float acc[BRAIN_MG_LANES][4];
#pragma unroll
            for (int l = 0; l < BRAIN_MG_LANES; ++l) {
#pragma unroll
                for (int i = 0; i < 4; ++i) { acc[l][i] = 0.0f; }
            }
            __syncwarp();  // the previous tile's readers of both buffers are done
            brain_mgr_stage_x<X16>(xbuf, xq, sx, xrows, 0u, kg, ng, lane);
            brain_cp_async_commit();
            for (unsigned int c = 0; c < chunks; ++c) {
                if (c + 1u < chunks) {
                    brain_mgr_stage_x<X16>(xbuf + ((c + 1u) & 1u) * BRAIN_MGR_XBUF, xq, sx, xrows, c + 1u, kg, ng, lane);
                    brain_cp_async_commit();
                    brain_cp_async_wait<1>();
                } else {
                    brain_cp_async_wait<0>();
                }
                __syncwarp();
                const unsigned int* xb_s = xbuf + (c & 1u) * BRAIN_MGR_XBUF;
                const unsigned int* xp = xb_s + g * BRAIN_MGR_XSTRIDE + tq * 2u;
                const float* xsa = reinterpret_cast<const float*>(xb_s + 8 * BRAIN_MGR_XSTRIDE) + (2u * tq) * 16u;
                const float* xsb = xsa + 16;
                const unsigned int sb = c * BRAIN_MG_LANES;
#pragma unroll
                for (int blk4 = 0; blk4 < BRAIN_MG_LANES / BRAIN_MGR_UNROLL; ++blk4) {
                    uint2 wa[BRAIN_MGR_UNROLL], wb[BRAIN_MGR_UNROLL], xb[BRAIN_MGR_UNROLL];
                    float sa[BRAIN_MGR_UNROLL], sbv[BRAIN_MGR_UNROLL], xa[BRAIN_MGR_UNROLL], xbs[BRAIN_MGR_UNROLL];
#pragma unroll
                    for (int u = 0; u < BRAIN_MGR_UNROLL; ++u) {
                        const unsigned int gl = blk4 * BRAIN_MGR_UNROLL + u;   // the group within the chunk
                        const unsigned int gi = sb + gl;
                        if (gi < ng) {
                            wa[u] = *reinterpret_cast<const uint2*>(w0 + gi * 8u);
                            wb[u] = *reinterpret_cast<const uint2*>(w1 + gi * 8u);
                            xb[u] = *reinterpret_cast<const uint2*>(xp + gl * 8u);
                            sa[u] = s0p[gi];
                            sbv[u] = s1p[gi];
                            xa[u] = xsa[gl];
                            xbs[u] = xsb[gl];
                        }
                    }
#pragma unroll
                    for (int u = 0; u < BRAIN_MGR_UNROLL; ++u) {
                        const unsigned int gi = sb + blk4 * BRAIN_MGR_UNROLL + u;
                        if (gi < ng) {
                            const unsigned int a[4] = {wa[u].x, wb[u].x, wa[u].y, wb[u].y};
                            unsigned int d[4];
                            brain_mma_s8(d, a, xb[u].x, xb[u].y);
                            const int l = blk4 * BRAIN_MGR_UNROLL + u;
                            acc[l][0] = __fadd_rn(acc[l][0], __fmul_rn(__fsub_rn(__uint_as_float(d[0]), FLOAT_BIAS), __fmul_rn(sa[u], xa[u])));
                            acc[l][1] = __fadd_rn(acc[l][1], __fmul_rn(__fsub_rn(__uint_as_float(d[1]), FLOAT_BIAS), __fmul_rn(sa[u], xbs[u])));
                            acc[l][2] = __fadd_rn(acc[l][2], __fmul_rn(__fsub_rn(__uint_as_float(d[2]), FLOAT_BIAS), __fmul_rn(sbv[u], xa[u])));
                            acc[l][3] = __fadd_rn(acc[l][3], __fmul_rn(__fsub_rn(__uint_as_float(d[3]), FLOAT_BIAS), __fmul_rn(sbv[u], xbs[u])));
                        }
                    }
                }
                __syncwarp();  // this buffer is refilled two chunks on
            }
            float total4[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                float s = 0.0f;
#pragma unroll
                for (int l = 0; l < BRAIN_MG_LANES; ++l) { s = __fadd_rn(s, acc[l][i]); }
                total4[i] = s;
            }
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                const unsigned int s = 2u * tq + j;
                if (s < cnt) {
                    float* o = out + static_cast<unsigned long long>(perm[first + s]) * n;
                    if (ok0) { o[col0] = total4[j]; }
                    if (ok1) { o[col1] = total4[2 + j]; }
                }
            }
        }
        t = seg_end;
    }
}

__device__ __forceinline__ void brain_moe_grouped_runs_any(const unsigned int* params, const unsigned int* xq, const float* sx,
                                                           const unsigned int* tab, const unsigned int* perm,
                                                           const unsigned int* wq, const float* sw, float* out) {
    // A buffer's rows all start on 16 bytes when the buffer does and a row is a whole number of 4-word vectors (`kg`
    // is a multiple of 8 for every dispatch served).
    const bool w16 = (reinterpret_cast<unsigned long long>(wq) & 15ull) == 0ull;
    const bool x16 = (reinterpret_cast<unsigned long long>(xq) & 15ull) == 0ull;
    if (w16) {
        if (x16) { brain_moe_grouped_runs<true, true>(params, xq, sx, tab, perm, wq, sw, out); }
        else { brain_moe_grouped_runs<true, false>(params, xq, sx, tab, perm, wq, sw, out); }
    } else {
        if (x16) { brain_moe_grouped_runs<false, true>(params, xq, sx, tab, perm, wq, sw, out); }
        else { brain_moe_grouped_runs<false, false>(params, xq, sx, tab, perm, wq, sw, out); }
    }
}

// One entry per registration: a native kernel's launch configuration (here its dynamic shared memory) is fixed by
// the first registration of its name (knowledge 412).
extern "C" __global__ void __launch_bounds__(BRAIN_MG_THREADS, 4)
brain_moe_i8_grouped_mma_runs(const unsigned int* params, const unsigned int* xq, const float* sx,
                              const unsigned int* tab, const unsigned int* perm,
                              const unsigned int* wq, const float* sw, float* out) {
    brain_moe_grouped_runs_any(params, xq, sx, tab, perm, wq, sw, out);
}

extern "C" __global__ void __launch_bounds__(BRAIN_MG_THREADS, 4)
brain_moe_i8_grouped_mma_runs_k512(const unsigned int* params, const unsigned int* xq, const float* sx,
                                   const unsigned int* tab, const unsigned int* perm,
                                   const unsigned int* wq, const float* sw, float* out) {
    brain_moe_grouped_runs_any(params, xq, sx, tab, perm, wq, sw, out);
}

// ---------------------------------------------------------------------------
// Four tiles a warp, four accumulators an output (`brain_moe_i8_grouped_mma_wide`)
// ---------------------------------------------------------------------------
//
// The two kernels above keep the WGSL tier's 16 lane accumulators per output, which costs 64 registers a thread for
// its four outputs and leaves 3-4 blocks an SM: they are latency-bound at 12-18 % occupancy. Here a warp computes its
// 16 weight rows against FOUR 8-slot tiles of one expert at once - each weight fragment and its scales loaded once per
// group serve four MMAs - and an output keeps four accumulators (groups g = 0, 1, 2, 3 mod 4, ascending), folded in
// ascending order. Same registers for the accumulators, four times the outputs, four independent MMA chains a group.
//
// NOT bit-identical to the WGSL tier: the f32 sum of the per-group terms is reassociated (chains of ng/4 terms and a
// fold of 4, against chains of ng/16 and a fold of 16; the same depth). Every group's integer dot product and its
// scale product are exact or rounded exactly as there. Its admission is the tuner's (`gpu_core::native_tune`, the
// `moe_i8_grouped` slot): against the f64 product of the activations as produced and the weights as stored, within
// `ALPHA` times what rounding the activations to their int8 codes alone costs.
//
// A block is 4 warps x 16 rows = 64 weight rows by a run of 4 consecutive tiles of the routing table; tiles of the
// run that belong to another expert are a second segment of the same loop.

#define BRAIN_MGW_TILES 4
#define BRAIN_MGW_ACC 4
#define BRAIN_MGW_UNROLL 2

template <bool VEC>
__device__ __forceinline__ void brain_moe_grouped_wide(const unsigned int* params, const unsigned int* xq, const float* sx,
                                                       const unsigned int* tab, const unsigned int* perm,
                                                       const unsigned int* wq, const float* sw, float* out) {
    const unsigned int kg = params[1];
    const unsigned int n = params[2];
    const unsigned int xdiv = params[3];
    const unsigned int ne = params[4];
    const unsigned int ng = kg >> 3;

    const unsigned int blk = blockIdx.y * gridDim.x + blockIdx.x;
    const unsigned int col_blocks = (n + BRAIN_MG_COLS - 1u) / BRAIN_MG_COLS;
    if (col_blocks == 0u) { return; }
    const unsigned int run = blk / col_blocks;
    const unsigned int cb = blk - run * col_blocks;
    const unsigned int total = tab[2u * ne + 1u];
    const unsigned int tile0 = run * BRAIN_MGW_TILES;
    if (tile0 >= total) { return; }
    const unsigned int tile_end = min(tile0 + BRAIN_MGW_TILES, total);

    const unsigned int warp = threadIdx.x >> 5;
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int g = lane >> 2;
    const unsigned int tq = lane & 3u;
    const unsigned int colbase = cb * BRAIN_MG_COLS + warp * 16u;
    if (colbase >= n) { return; }  // warp-uniform; no barrier below
    const unsigned int col0 = colbase + g, col1 = colbase + g + 8u;
    const bool ok0 = col0 < n, ok1 = col1 < n;

    unsigned int t = tile0;
    while (t < tile_end) {
        unsigned int lo = 0u, hi = ne;
        while (hi - lo > 1u) {
            const unsigned int mid = (lo + hi) >> 1;
            if (tab[ne + 1u + mid] <= t) { lo = mid; } else { hi = mid; }
        }
        const unsigned int e = lo;
        const unsigned int seg_end = min(tile_end, tab[ne + 2u + e]);
        const unsigned int nt = seg_end - t;

        const unsigned long long row0 = static_cast<unsigned long long>(e) * n + (ok0 ? col0 : n - 1u);
        const unsigned long long row1 = static_cast<unsigned long long>(e) * n + (ok1 ? col1 : n - 1u);
        const unsigned int* w0 = wq + row0 * kg + tq * 2u;
        const unsigned int* w1 = wq + row1 * kg + tq * 2u;
        const float* s0p = sw + row0 * ng;
        const float* s1p = sw + row1 * ng;

        // Per tile of the segment: this thread's activation row (slot g) and the scale rows of its two output slots.
        unsigned int xrow[BRAIN_MGW_TILES], sarow[BRAIN_MGW_TILES], sbrow[BRAIN_MGW_TILES], first[BRAIN_MGW_TILES], cnt[BRAIN_MGW_TILES];
#pragma unroll
        for (int j = 0; j < BRAIN_MGW_TILES; ++j) {
            const unsigned int tile = t + min(static_cast<unsigned int>(j), nt - 1u);  // a tile past the segment repeats its last
            first[j] = tab[e] + (tile - tab[ne + 1u + e]) * BRAIN_MG_MR;
            cnt[j] = (static_cast<unsigned int>(j) < nt) ? min(BRAIN_MG_MR, tab[e + 1u] - first[j]) : 0u;
            const unsigned int c = max(cnt[j], 1u);
            xrow[j] = perm[first[j] + (g < c ? g : 0u)] / xdiv;
            sarow[j] = perm[first[j] + (2u * tq < c ? 2u * tq : 0u)] / xdiv;
            sbrow[j] = perm[first[j] + (2u * tq + 1u < c ? 2u * tq + 1u : 0u)] / xdiv;
        }

        float acc[BRAIN_MGW_TILES][BRAIN_MGW_ACC][4];
#pragma unroll
        for (int j = 0; j < BRAIN_MGW_TILES; ++j) {
#pragma unroll
            for (int l = 0; l < BRAIN_MGW_ACC; ++l) {
#pragma unroll
                for (int i = 0; i < 4; ++i) { acc[j][l][i] = 0.0f; }
            }
        }

        for (unsigned int sb = 0; sb < ng; sb += BRAIN_MGW_ACC) {
#pragma unroll
            for (int h = 0; h < BRAIN_MGW_ACC / BRAIN_MGW_UNROLL; ++h) {
                // Every load of the step - the weight fragments and scales, then each tile's activation fragment and
                // scales - before any product consumes one.
                uint2 wa[BRAIN_MGW_UNROLL], wb[BRAIN_MGW_UNROLL], xb[BRAIN_MGW_TILES][BRAIN_MGW_UNROLL];
                float sa[BRAIN_MGW_UNROLL], sbv[BRAIN_MGW_UNROLL], xa[BRAIN_MGW_TILES][BRAIN_MGW_UNROLL], xbs[BRAIN_MGW_TILES][BRAIN_MGW_UNROLL];
#pragma unroll
                for (int u = 0; u < BRAIN_MGW_UNROLL; ++u) {
                    const unsigned int gi = sb + h * BRAIN_MGW_UNROLL + u;
                    if (gi < ng) {
                        wa[u] = brain_ld8<VEC>(w0 + gi * 8u);
                        wb[u] = brain_ld8<VEC>(w1 + gi * 8u);
                        sa[u] = __ldg(s0p + gi);
                        sbv[u] = __ldg(s1p + gi);
#pragma unroll
                        for (int j = 0; j < BRAIN_MGW_TILES; ++j) {
                            if (static_cast<unsigned int>(j) < nt) {  // warp-uniform
                                xb[j][u] = brain_ld8<VEC>(xq + static_cast<unsigned long long>(xrow[j]) * kg + tq * 2u + gi * 8u);
                                xa[j][u] = __ldg(sx + static_cast<unsigned long long>(sarow[j]) * ng + gi);
                                xbs[j][u] = __ldg(sx + static_cast<unsigned long long>(sbrow[j]) * ng + gi);
                            }
                        }
                    }
                }
#pragma unroll
                for (int u = 0; u < BRAIN_MGW_UNROLL; ++u) {
                    const unsigned int gi = sb + h * BRAIN_MGW_UNROLL + u;
                    if (gi < ng) {
                        const unsigned int a[4] = {wa[u].x, wb[u].x, wa[u].y, wb[u].y};
                        const int l = h * BRAIN_MGW_UNROLL + u;  // group gi's accumulator: gi mod 4
#pragma unroll
                        for (int j = 0; j < BRAIN_MGW_TILES; ++j) {
                            if (static_cast<unsigned int>(j) < nt) {
                                unsigned int d[4];
                                brain_mma_s8(d, a, xb[j][u].x, xb[j][u].y);
                                acc[j][l][0] = __fadd_rn(acc[j][l][0], __fmul_rn(__fsub_rn(__uint_as_float(d[0]), FLOAT_BIAS), __fmul_rn(sa[u], xa[j][u])));
                                acc[j][l][1] = __fadd_rn(acc[j][l][1], __fmul_rn(__fsub_rn(__uint_as_float(d[1]), FLOAT_BIAS), __fmul_rn(sa[u], xbs[j][u])));
                                acc[j][l][2] = __fadd_rn(acc[j][l][2], __fmul_rn(__fsub_rn(__uint_as_float(d[2]), FLOAT_BIAS), __fmul_rn(sbv[u], xa[j][u])));
                                acc[j][l][3] = __fadd_rn(acc[j][l][3], __fmul_rn(__fsub_rn(__uint_as_float(d[3]), FLOAT_BIAS), __fmul_rn(sbv[u], xbs[j][u])));
                            }
                        }
                    }
                }
            }
        }

#pragma unroll
        for (int j = 0; j < BRAIN_MGW_TILES; ++j) {
            float total4[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                float s = 0.0f;
#pragma unroll
                for (int l = 0; l < BRAIN_MGW_ACC; ++l) { s = __fadd_rn(s, acc[j][l][i]); }
                total4[i] = s;
            }
#pragma unroll
            for (int q = 0; q < 2; ++q) {
                const unsigned int s = 2u * tq + q;
                if (s < cnt[j]) {
                    float* o = out + static_cast<unsigned long long>(perm[first[j] + s]) * n;
                    if (ok0) { o[col0] = total4[q]; }
                    if (ok1) { o[col1] = total4[2 + q]; }
                }
            }
        }
        t = seg_end;
    }
}

extern "C" __global__ void __launch_bounds__(BRAIN_MG_THREADS, 3)
brain_moe_i8_grouped_mma_wide(const unsigned int* params, const unsigned int* xq, const float* sx,
                              const unsigned int* tab, const unsigned int* perm,
                              const unsigned int* wq, const float* sw, float* out) {
    const bool aligned = ((reinterpret_cast<unsigned long long>(xq) | reinterpret_cast<unsigned long long>(wq)) & 7ull) == 0ull;
    if (aligned) {
        brain_moe_grouped_wide<true>(params, xq, sx, tab, perm, wq, sw, out);
    } else {
        brain_moe_grouped_wide<false>(params, xq, sx, tab, perm, wq, sw, out);
    }
}
