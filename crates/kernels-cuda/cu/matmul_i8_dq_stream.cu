// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for memory-bandwidth-bound LLM
// decode on GPUs for its clients. If your team needs expertise in streaming
// quantised weights through shared memory and feeding them to tensor cores
// without a per-group scale fold, then you can procure our services by
// sending an email to info@swedishembedded.com.
//
// Weight-streaming matmul of INT8 weights (one f32 scale per 32 codes) against BF16 activations for up to 128 rows of
// x (the registry serves 33 to 128: a decode step of a large batch), with the weight codes dequantised IN REGISTERS:
//
//   params : u32 [m, kg, n]   kg = K/4 packed words per weight row, K a multiple of 32
//   x_bf   : [M, K] bf16      activations rounded to bf16 (two per u32; 16-byte aligned)
//   w_q    : [N, kg] u32      weights, 4 signed int8 codes per word (16-byte aligned)
//   sw     : [N, K/32] f32    per-32-element-group weight scale
//   out    : [M, N]  f32      out[m, n] = sum_k x[m,k] * bf16(w_q[n,k] * sw[n, k/32]), accumulated in f32 over ALL of K
//
// Why this kernel exists (knowledge 372, 373)
// -------------------------------------------
// The integer-MMA kernels (`matmul_i8_stream`) keep the activations as int8 with one scale per 32 elements, so every
// MMA of one k32 group must be followed by a scale fold into an f32 total: about 12 FP instructions per MMA per
// thread, which bounds the weight stream at 1.1-1.3 TB/s of the ~3.7 TB/s the memory delivers. Here the scale is
// applied to the WEIGHT once, as the code leaves shared memory: code x scale -> bf16 in registers, the A operand of a
// bf16 `mma.sync.m16n8k16`, accumulated in f32 straight across K. There is no fold, no activation quantisation and no
// activation scale operand. The dequantisation costs instructions per weight element (about 2.5), independent of the
// number of rows of x that share the fetched weight.
//
// Structure (the data movement of `matmul_i8_stream.cu`, which this file adapts):
//  * weights, their scales and the block's slice of x travel global -> shared memory by `cp.async` through a ring of
//    BRAIN_DQS_STAGES stages of 128 bytes of K per weight row (one cache line), 16 bytes a lane;
//  * the weights are the A operand (16 weight rows a tile), the rows of x the N side (8 a tile);
//  * K is split over a cluster of blocks (distributed shared memory) when there are too few weight rows to keep every
//    SM streaming; the partial sums are added in rank order, so the result does not depend on scheduling.
//
// The k permutation that makes the dequantisation cheap
// -----------------------------------------------------
// `ldmatrix` on the int8 tile hands thread (g = lane / 4, t = lane % 4) bytes 4t..4t+3 of a 16-byte row chunk. The MMA
// does not care which true k a position of its 16-deep reduction stands for as long as A and B agree, so A position
// (2t, 2t+1) takes true k (4t, 4t+1) and position (2t+8, 2t+9) takes (4t+2, 4t+3). B then needs, per thread, four
// consecutive bf16 of a row of x: one 8-byte shared load, whose two halves are the two B registers as they are.
//
// Dequantisation of a byte pair (b0, b1) as a bf16x2 pair, exactly, in four instructions plus the scale:
//   u  = prmt(word)                   // [b0, 0, b1, 0] as two 16-bit lanes
//   h1 = (u & 0x007F) | 0x4300        // bf16 128 + (b & 0x7F)       (0x4300 is 128.0, the low 7 bits are its mantissa)
//   h2 = (u & 0x0080) | 0x4300        // bf16 128 (sign clear) or 256 (sign set: bit 7 is the exponent's low bit)
//   q  = h1 - h2                      // = b as a signed integer, exactly (a small integer in bf16)
//   w  = q * bf16(scale)              // one rounding to bf16
// The weight is therefore the bf16 rounding of code x (the group scale rounded to bf16): a relative error of at most
// about 2^-8 per weight against the stored code x scale, the resolution of the bf16 operand it feeds. The scale is
// shared by 32 weights, so the error is a per-group gain, not noise; `gpu_core::native_tune` holds the candidate to
// eight times the error of rounding its inputs to bf16 and the model-level quality gate checks the tokens.
//
// Row tails: weight rows past n read the last row and are dropped; rows of x past m read the last row and are
// dropped; a final stage with fewer than four groups copies only the groups that exist.
//
// No `__restrict__`: brain's device buffers alias by design.

#ifndef BRAIN_DQS_WARPS
#define BRAIN_DQS_WARPS 4
#endif
// m16 tiles of weight rows one warp covers.
#ifndef BRAIN_DQS_ATILES
#define BRAIN_DQS_ATILES 1
#endif
#ifndef BRAIN_DQS_STAGES
#define BRAIN_DQS_STAGES 4
#endif
#ifndef BRAIN_DQS_MIN_BLOCKS
#define BRAIN_DQS_MIN_BLOCKS 2
#endif
// Rows of x one launch serves (a multiple of 8): sizes the x stages in shared memory.
#ifndef BRAIN_DQS_MAX_ROWS
#define BRAIN_DQS_MAX_ROWS 32
#endif

#define BRAIN_DQS_THREADS (32 * BRAIN_DQS_WARPS)
// Weight rows of one block.
#define BRAIN_DQS_BN (16 * BRAIN_DQS_ATILES * BRAIN_DQS_WARPS)
// Bytes of int8 K per stage and weight row (one cache line), the scale groups in it, and the bf16 bytes of x per row.
#define BRAIN_DQS_KB 128
#define BRAIN_DQS_GPS 4
#define BRAIN_DQS_XB 256
// Stage layout (bytes): weights, x, weight scales.
#define BRAIN_DQS_W_BYTES (BRAIN_DQS_BN * BRAIN_DQS_KB)
#define BRAIN_DQS_X_BYTES (BRAIN_DQS_MAX_ROWS * BRAIN_DQS_XB)
#define BRAIN_DQS_SW_BYTES (BRAIN_DQS_BN * BRAIN_DQS_GPS * 4)
#define BRAIN_DQS_STAGE_BYTES (BRAIN_DQS_W_BYTES + BRAIN_DQS_X_BYTES + BRAIN_DQS_SW_BYTES)
// Rows covered by one pass of the block's threads over a 16-byte-chunk copy of the weight tile (8 chunks a row), the x tile
// (16 chunks a row) and the scale tile (4 groups a row).
#define BRAIN_DQS_WROWS (BRAIN_DQS_THREADS / 8)
#define BRAIN_DQS_XROWS (BRAIN_DQS_THREADS / 16)
#define BRAIN_DQS_SROWS (BRAIN_DQS_THREADS / 4)
// Partial-sum buffer for the cluster reduction ([row of x][weight row], padded), aliased on the stages.
#define BRAIN_DQS_RED_STRIDE (BRAIN_DQS_BN + 4)

// Programmatic dependent launch (compute capability 9.0), see `matmul_i8_gemv.cu`.
#ifndef BRAIN_PDL_ENTRY
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
#define BRAIN_PDL_ENTRY() do { asm volatile("griddepcontrol.wait;\n" ::: "memory"); asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory"); } while (0)
#else
#define BRAIN_PDL_ENTRY() do {} while (0)
#endif
#endif

__device__ __forceinline__ unsigned int brain_dqs_smem(const void* p) {
    return static_cast<unsigned int>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void brain_dqs_cp16(unsigned int dst, const void* src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(dst), "l"(src) : "memory");
}

__device__ __forceinline__ void brain_dqs_cp4(unsigned int dst, const void* src) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"(dst), "l"(src) : "memory");
}

__device__ __forceinline__ void brain_dqs_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }

template <int N>
__device__ __forceinline__ void brain_dqs_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N) : "memory"); }

__device__ __forceinline__ void brain_dqs_ldsm4(unsigned int (&r)[4], unsigned int addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}

__device__ __forceinline__ void brain_dqs_lds64(unsigned int& lo, unsigned int& hi, unsigned int addr) {
    asm volatile("ld.shared.v2.b32 {%0, %1}, [%2];\n" : "=r"(lo), "=r"(hi) : "r"(addr));
}

// d (f32 x4) += A (bf16 16x16, weights) * B (bf16 16x8, rows of x).
__device__ __forceinline__ void brain_dqs_mma(float (&d)[4], const unsigned int (&a)[4], unsigned int b0, unsigned int b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ unsigned int brain_dqs_fma2(unsigned int a, unsigned int b, unsigned int c) {
    unsigned int d;
    asm("fma.rn.bf16x2 %0, %1, %2, %3;\n" : "=r"(d) : "r"(a), "r"(b), "r"(c));
    return d;
}

// Two int8 codes in the low bytes of the 16-bit lanes of `u` -> bf16x2 of code * scale (scale2: the scale in both lanes).
// The masks and the bf16 128.0 of the dequantisation, held in registers: `lop3` takes three register operands, so
// `(u & mask) | magic` is ONE instruction only when neither constant is an immediate (two immediates make the compiler
// emit the AND and the OR separately). `brain_dqs_consts` makes them opaque to constant propagation.
struct BrainDqConsts {
    unsigned int low7, sign, magic;
};

__device__ __forceinline__ BrainDqConsts brain_dqs_consts() {
    BrainDqConsts c;
    asm volatile("mov.b32 %0, 0x007F007F;\n" : "=r"(c.low7));
    asm volatile("mov.b32 %0, 0x00800080;\n" : "=r"(c.sign));
    asm volatile("mov.b32 %0, 0x43004300;\n" : "=r"(c.magic));
    return c;
}

// (a & mask) | magic in one instruction.
__device__ __forceinline__ unsigned int brain_dqs_and_or(unsigned int a, unsigned int mask, unsigned int magic) {
    unsigned int d;
    asm("lop3.b32 %0, %1, %2, %3, 0xEA;\n" : "=r"(d) : "r"(a), "r"(mask), "r"(magic));
    return d;
}

// Two int8 codes in the low bytes of the 16-bit lanes of `u` -> bf16x2 of code * scale (scale2: the scale in both lanes).
__device__ __forceinline__ unsigned int brain_dqs_dequant2(unsigned int u, unsigned int scale2, const BrainDqConsts& k) {
    const unsigned int h1 = brain_dqs_and_or(u, k.low7, k.magic);
    const unsigned int h2 = brain_dqs_and_or(u, k.sign, k.magic);
    const unsigned int q = brain_dqs_fma2(h2, 0xBF80BF80u, h1);  // h1 - h2
    return brain_dqs_fma2(q, scale2, 0u);
}

// Four int8 codes (one word) -> two bf16x2 registers, codes (0,1) and (2,3).
__device__ __forceinline__ void brain_dqs_dequant4(unsigned int word, unsigned int scale2, const BrainDqConsts& k, unsigned int& lo, unsigned int& hi) {
    unsigned int u01, u23;
    asm("prmt.b32 %0, %1, 0, 0x4140;\n" : "=r"(u01) : "r"(word));
    asm("prmt.b32 %0, %1, 0, 0x4342;\n" : "=r"(u23) : "r"(word));
    lo = brain_dqs_dequant2(u01, scale2, k);
    hi = brain_dqs_dequant2(u23, scale2, k);
}

__device__ __forceinline__ unsigned int brain_dqs_splat_bf16(float s) {
    unsigned int d;
    asm("cvt.rn.bf16x2.f32 %0, %1, %1;\n" : "=r"(d) : "f"(s));
    return d;
}

__device__ __forceinline__ unsigned int brain_dqs_cluster_rank() {
    unsigned int r;
    asm volatile("mov.u32 %0, %%cluster_ctarank;\n" : "=r"(r));
    return r;
}

__device__ __forceinline__ unsigned int brain_dqs_cluster_size() {
    unsigned int r;
    asm volatile("mov.u32 %0, %%cluster_nctarank;\n" : "=r"(r));
    return r;
}

__device__ __forceinline__ void brain_dqs_cluster_sync() {
    asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;\n" ::: "memory");
}

__device__ __forceinline__ float brain_dqs_remote_ld(unsigned int local_addr, unsigned int rank) {
    unsigned int remote;
    float v;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;\n" : "=r"(remote) : "r"(local_addr), "r"(rank));
    asm volatile("ld.shared::cluster.f32 %0, [%1];\n" : "=f"(v) : "r"(remote) : "memory");
    return v;
}

// The copies one thread issues for a stage, as offsets computed once: stage `s` covers K elements [128 s, 128 s + 128)
// of every weight row of the block and every row of x, and a thread's chunks differ between stages only by the K
// offset, so the row arithmetic (clamped to the last row of each matrix, whose results are dropped at the store) is
// done at the start, not at every stage. The x tile is stored 256 bytes a row, its 16-byte chunk index XOR-swizzled
// by (row & 3) * 2, so the 8-byte loads of a quarter-warp (four rows, two chunks each) touch eight distinct 16-byte
// bank groups.
template <int T>
struct BrainDqsCopies {
    static constexpr int W = BRAIN_DQS_BN * 8 / BRAIN_DQS_THREADS;                 // weight chunks a thread copies per stage
    static constexpr int X = (T * 128 + BRAIN_DQS_THREADS - 1) / BRAIN_DQS_THREADS;  // x chunks (T * 8 rows x 16 chunks)
    static constexpr int S = BRAIN_DQS_BN * BRAIN_DQS_GPS / BRAIN_DQS_THREADS;      // scale words
    unsigned int wsrc[W];  // weight chunk i: byte offset of (row, 16 * ch) in wq
    unsigned int wdst;     // smem offset of chunk 0; chunk i is BRAIN_DQS_WROWS rows further on
    unsigned int xsrc[X];  // x chunk i: byte offset of (row, 2 * 8 * ch) in xb
    unsigned int xdst;     // smem offset of chunk 0 within the x tile; chunk i is BRAIN_DQS_XROWS rows further on
    unsigned int ssrc[S];  // scale word i: element offset of (row, 0) in sw
    unsigned int wch, xch, sgi;
};

template <int T>
__device__ __forceinline__ void brain_dqs_prepare(BrainDqsCopies<T>& c, unsigned int m, unsigned int groups, unsigned int n, unsigned int n0) {
    const unsigned int tid = threadIdx.x;
    const unsigned int k = groups * 32u;
    c.wch = tid & 7u;
    const unsigned int wrow = tid >> 3;
#pragma unroll
    for (int i = 0; i < BrainDqsCopies<T>::W; ++i) {
        c.wsrc[i] = min(n0 + wrow + BRAIN_DQS_WROWS * i, n - 1u) * k + c.wch * 16u;
    }
    c.wdst = wrow * BRAIN_DQS_KB + ((c.wch ^ (wrow & 7u)) << 4);
    c.xch = tid & 15u;
    const unsigned int xrow = tid >> 4;
#pragma unroll
    for (int i = 0; i < BrainDqsCopies<T>::X; ++i) {
        c.xsrc[i] = min(xrow + BRAIN_DQS_XROWS * i, m - 1u) * k * 2u + c.xch * 16u;
    }
    c.xdst = xrow * BRAIN_DQS_XB + ((c.xch ^ ((xrow & 3u) << 1)) << 4);
    c.sgi = tid & 3u;
#pragma unroll
    for (int i = 0; i < BrainDqsCopies<T>::S; ++i) {
        c.ssrc[i] = min(n0 + (tid >> 2) + BRAIN_DQS_SROWS * i, n - 1u) * groups + c.sgi;
    }
}

template <int T>
__device__ __forceinline__ void brain_dqs_issue(const BrainDqsCopies<T>& c, unsigned int stage_smem, unsigned int s, const unsigned char* wq, const unsigned char* xb,
                                                const float* sw, unsigned int groups) {
    const unsigned int k = groups * 32u;
    const unsigned int koff = s * BRAIN_DQS_KB;
    const unsigned int g0 = s * BRAIN_DQS_GPS;
    const bool full = koff + BRAIN_DQS_KB <= k;  // every chunk of the stage exists
    if (full || koff + c.wch * 16u < k) {
#pragma unroll
        for (int i = 0; i < BrainDqsCopies<T>::W; ++i) {
            brain_dqs_cp16(stage_smem + c.wdst + BRAIN_DQS_WROWS * BRAIN_DQS_KB * i, wq + (c.wsrc[i] + koff));
        }
    }
    const unsigned int xs = stage_smem + BRAIN_DQS_W_BYTES;
    if (full || koff + c.xch * 8u < k) {
#pragma unroll
        for (int i = 0; i < BrainDqsCopies<T>::X; ++i) {
            // The last pass of a tile that is not a whole number of passes leaves the upper threads idle.
            if (BrainDqsCopies<T>::X * BRAIN_DQS_THREADS == T * 128 || threadIdx.x + BRAIN_DQS_THREADS * i < T * 128) {
                brain_dqs_cp16(xs + c.xdst + BRAIN_DQS_XROWS * BRAIN_DQS_XB * i, xb + (c.xsrc[i] + koff * 2u));
            }
        }
    }
    const unsigned int sws = xs + BRAIN_DQS_X_BYTES;
    if (full || g0 + c.sgi < groups) {
#pragma unroll
        for (int i = 0; i < BrainDqsCopies<T>::S; ++i) {
            brain_dqs_cp4(sws + (threadIdx.x + BRAIN_DQS_THREADS * i) * 4u, sw + (c.ssrc[i] + g0));
        }
    }
}

// One scale group (32 k) of one 16-row weight tile against T tiles of eight rows of x, accumulated into f.
// `xa` is the stage's x tile address plus this thread's 8-byte load offset for the group's first 16-deep step (`xoff`
// of the second step is a constant further on).
template <int T>
__device__ __forceinline__ void brain_dqs_group(float (&f)[T][4], const BrainDqConsts& k, unsigned int wmat_addr, unsigned int xa0, unsigned int xa1, unsigned int sl, unsigned int sh) {
    unsigned int af[4];
    brain_dqs_ldsm4(af, wmat_addr);
    // af[0], af[1]: rows g and g + 8, bytes 4t..4t+3 of k 0..15; af[2], af[3]: the same of k 16..31.
    unsigned int a1[4], a2[4];
    brain_dqs_dequant4(af[0], sl, k, a1[0], a1[2]);
    brain_dqs_dequant4(af[1], sh, k, a1[1], a1[3]);
    brain_dqs_dequant4(af[2], sl, k, a2[0], a2[2]);
    brain_dqs_dequant4(af[3], sh, k, a2[1], a2[3]);
#pragma unroll
    for (int t = 0; t < T; ++t) {
        unsigned int b0, b1, c0, c1;
        brain_dqs_lds64(b0, b1, xa0 + t * 8u * BRAIN_DQS_XB);
        brain_dqs_lds64(c0, c1, xa1 + t * 8u * BRAIN_DQS_XB);
        brain_dqs_mma(f[t], a1, b0, b1);
        brain_dqs_mma(f[t], a2, c0, c1);
    }
}

template <int T>
__device__ __forceinline__ void brain_dqs_block(const unsigned char* wq, const unsigned char* xb, const float* sw, float* out, unsigned int m,
                                                unsigned int groups, unsigned int n, unsigned int n0, unsigned int rank, unsigned int csize,
                                                unsigned char* smem) {
    constexpr int AT = BRAIN_DQS_ATILES;
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int warp = threadIdx.x >> 5;
    const unsigned int g = lane >> 2, t4 = lane & 3u;
    const unsigned int smem0 = brain_dqs_smem(smem);

    // Stages of 128 bytes of K; this block streams the contiguous stretch [first, first + count) of them.
    const unsigned int stages = (groups + BRAIN_DQS_GPS - 1u) / BRAIN_DQS_GPS;
    const unsigned int per = (stages + csize - 1u) / csize;
    const unsigned int first = min(rank * per, stages);
    const unsigned int count = min(per, stages - first);

    float f[AT][T][4];
#pragma unroll
    for (int a = 0; a < AT; ++a)
#pragma unroll
        for (int t = 0; t < T; ++t) { f[a][t][0] = f[a][t][1] = f[a][t][2] = f[a][t][3] = 0.0f; }

    const BrainDqConsts consts = brain_dqs_consts();
    BrainDqsCopies<T> copies;
    brain_dqs_prepare<T>(copies, m, groups, n, n0);
#pragma unroll
    for (int p = 0; p < BRAIN_DQS_STAGES - 1; ++p) {
        if (p < count) { brain_dqs_issue<T>(copies, smem0 + p * BRAIN_DQS_STAGE_BYTES, first + p, wq, xb, sw, groups); }
        brain_dqs_commit();
    }

    // Shared-memory offsets of this thread's fragments, constant over the loop. A matrix: ldmatrix lane l addresses
    // row (l & 7) + 8 * (l >> 3 & 1) of its 16-row tile, 16-byte chunk (2 j + (l >> 4)) of group j, swizzled by the
    // row's low three bits. B: the 8-byte load of 16-deep step kb (0..7) at row g of tile 0: chunk 2 kb + (t4 >> 1),
    // swizzled by (g & 3) * 2, then half (t4 & 1) of the chunk.
    const unsigned int lrow = lane & 7u;
    const unsigned int mi = lane >> 3;
    unsigned int aoff[AT];
#pragma unroll
    for (int a = 0; a < AT; ++a) { aoff[a] = ((warp * AT + a) * 16u + lrow + ((mi & 1u) << 3)) * BRAIN_DQS_KB; }
    unsigned int achunk[BRAIN_DQS_GPS];
#pragma unroll
    for (int j = 0; j < BRAIN_DQS_GPS; ++j) { achunk[j] = ((2u * j + (mi >> 1)) ^ lrow) << 4; }
    unsigned int xoff[8];
#pragma unroll
    for (int kb = 0; kb < 8; ++kb) { xoff[kb] = g * BRAIN_DQS_XB + ((((2u * kb) ^ ((g & 3u) << 1)) + (t4 >> 1)) << 4) + (t4 & 1u) * 8u; }
    for (unsigned int i = 0; i < count; ++i) {
        brain_dqs_wait<BRAIN_DQS_STAGES - 2>();
        __syncthreads();  // the stage is visible to every warp, and the slot refilled below was read by all of them
        const unsigned int nxt = i + BRAIN_DQS_STAGES - 1;
        if (nxt < count) { brain_dqs_issue<T>(copies, smem0 + (nxt % BRAIN_DQS_STAGES) * BRAIN_DQS_STAGE_BYTES, first + nxt, wq, xb, sw, groups); }
        brain_dqs_commit();

        const unsigned int st = smem0 + (i % BRAIN_DQS_STAGES) * BRAIN_DQS_STAGE_BYTES;
        const unsigned int xs = st + BRAIN_DQS_W_BYTES;
        const float* sws = reinterpret_cast<const float*>(smem + (st - smem0) + BRAIN_DQS_W_BYTES + BRAIN_DQS_X_BYTES);
        const unsigned int gin = min((unsigned int)BRAIN_DQS_GPS, groups - (first + i) * BRAIN_DQS_GPS);  // groups in this stage
#pragma unroll
        for (int a = 0; a < AT; ++a) {
            // The four scales of the stage for this thread's rows g and g + 8 of the tile, as bf16 pairs.
            const unsigned int arow0 = (warp * AT + a) * 16u;
            const float4 lo = *reinterpret_cast<const float4*>(sws + (arow0 + g) * BRAIN_DQS_GPS);
            const float4 hi = *reinterpret_cast<const float4*>(sws + (arow0 + g + 8u) * BRAIN_DQS_GPS);
            const unsigned int sl[4] = {brain_dqs_splat_bf16(lo.x), brain_dqs_splat_bf16(lo.y), brain_dqs_splat_bf16(lo.z), brain_dqs_splat_bf16(lo.w)};
            const unsigned int sh[4] = {brain_dqs_splat_bf16(hi.x), brain_dqs_splat_bf16(hi.y), brain_dqs_splat_bf16(hi.z), brain_dqs_splat_bf16(hi.w)};
            if (gin == BRAIN_DQS_GPS) {
#pragma unroll
                for (int j = 0; j < BRAIN_DQS_GPS; ++j) {
                    brain_dqs_group<T>(f[a], consts, st + aoff[a] + achunk[j], xs + xoff[2 * j], xs + xoff[2 * j + 1], sl[j], sh[j]);
                }
            } else {
                for (int j = 0; j < (int)gin; ++j) {
                    brain_dqs_group<T>(f[a], consts, st + aoff[a] + achunk[j], xs + xoff[2 * j], xs + xoff[2 * j + 1], sl[j], sh[j]);
                }
            }
        }
    }
    brain_dqs_wait<0>();

    if (csize == 1u) {
#pragma unroll
        for (int a = 0; a < AT; ++a)
#pragma unroll
            for (int t = 0; t < T; ++t)
#pragma unroll
                for (int q = 0; q < 4; ++q) {
                    const unsigned int row = n0 + (warp * AT + a) * 16u + g + 8u * (q >> 1);
                    const unsigned int col = 8u * t + 2u * t4 + (q & 1u);
                    if (row < n && col < m) { out[(unsigned long long)col * n + row] = f[a][t][q]; }
                }
        return;
    }

    // Split K: park this block's partial sums, then each block adds the partials of every block of the cluster, in rank
    // order, for its share of the outputs.
    __syncthreads();  // every warp is done with the stages the buffer aliases
    float* red = reinterpret_cast<float*>(smem);
#pragma unroll
    for (int a = 0; a < AT; ++a)
#pragma unroll
        for (int t = 0; t < T; ++t)
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                const unsigned int row = (warp * AT + a) * 16u + g + 8u * (q >> 1);
                const unsigned int col = 8u * t + 2u * t4 + (q & 1u);
                red[col * BRAIN_DQS_RED_STRIDE + row] = f[a][t][q];
            }
    brain_dqs_cluster_sync();
    for (unsigned int idx = rank * BRAIN_DQS_THREADS + threadIdx.x; idx < m * BRAIN_DQS_BN; idx += csize * BRAIN_DQS_THREADS) {
        const unsigned int col = idx / BRAIN_DQS_BN, row = idx % BRAIN_DQS_BN;
        const unsigned int a = smem0 + (col * BRAIN_DQS_RED_STRIDE + row) * 4u;
        float total = 0.0f;
        for (unsigned int r = 0; r < csize; ++r) { total += brain_dqs_remote_ld(a, r); }
        if (n0 + row < n) { out[(unsigned long long)col * n + n0 + row] = total; }
    }
    brain_dqs_cluster_sync();  // no block leaves while a peer may still read its partials
}

extern "C" __global__ void __launch_bounds__(BRAIN_DQS_THREADS, BRAIN_DQS_MIN_BLOCKS)
brain_matmul_i8_dq_stream(const unsigned int* params, const unsigned int* xb, const unsigned int* wq, const float* sw, float* out) {
    BRAIN_PDL_ENTRY();
    extern __shared__ __align__(1024) unsigned char brain_dqs_smem_base[];
    const unsigned int m = params[0];
    const unsigned int groups = params[1] >> 3;
    const unsigned int n = params[2];
    const unsigned int csize = brain_dqs_cluster_size();
    const unsigned int rank = brain_dqs_cluster_rank();
    // A cluster's blocks are consecutive and share a tile of weight rows.
    const unsigned int blk = blockIdx.y * gridDim.x + blockIdx.x;
    const unsigned int n0 = (blk / csize) * BRAIN_DQS_BN;
    if (n0 >= n || m == 0u) { return; }  // cluster-uniform
    const unsigned char* w8 = reinterpret_cast<const unsigned char*>(wq);
    const unsigned char* x8 = reinterpret_cast<const unsigned char*>(xb);
    if (m <= 8u)       { brain_dqs_block<1>(w8, x8, sw, out, m, groups, n, n0, rank, csize, brain_dqs_smem_base); }
    else if (m <= 16u) { brain_dqs_block<2>(w8, x8, sw, out, m, groups, n, n0, rank, csize, brain_dqs_smem_base); }
    else if (m <= 24u) { brain_dqs_block<3>(w8, x8, sw, out, m, groups, n, n0, rank, csize, brain_dqs_smem_base); }
#if BRAIN_DQS_MAX_ROWS > 32
    else if (m <= 32u) { brain_dqs_block<4>(w8, x8, sw, out, m, groups, n, n0, rank, csize, brain_dqs_smem_base); }
    else if (m <= 48u) { brain_dqs_block<6>(w8, x8, sw, out, m, groups, n, n0, rank, csize, brain_dqs_smem_base); }
    else if (m <= 64u) { brain_dqs_block<8>(w8, x8, sw, out, m, groups, n, n0, rank, csize, brain_dqs_smem_base); }
#if BRAIN_DQS_MAX_ROWS > 64
    else if (m <= 96u) { brain_dqs_block<12>(w8, x8, sw, out, m, groups, n, n0, rank, csize, brain_dqs_smem_base); }
    else               { brain_dqs_block<16>(w8, x8, sw, out, m, groups, n, n0, rank, csize, brain_dqs_smem_base); }
#endif
#else
    else               { brain_dqs_block<4>(w8, x8, sw, out, m, groups, n, n0, rank, csize, brain_dqs_smem_base); }
#endif
}
