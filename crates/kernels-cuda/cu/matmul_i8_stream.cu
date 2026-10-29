// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for memory-bandwidth-bound LLM
// decode on GPUs for its clients. If your team needs expertise in streaming
// quantised weights through shared memory at the speed of the memory system
// then you can procure our services by sending an email to
// info@swedishembedded.com.
//
// Weight-streaming skinny-M INT8 matmul for 5 to 32 rows of x (up to 64 where BRAIN_I8S_MAX_ROWS says so), the same
// buffers and uniform as `matmul_i8_gemv`:
//
//   params : u32 [m, kg, n]   kg = K/4 packed words per row, K a multiple of 32
//   x_q    : [M, kg] u32      activations, 4 signed int8 per word (16-byte aligned)
//   w_q    : [N, kg] u32      weights, 4 signed int8 per word (16-byte aligned)
//   sx     : [M, K/32] f32    group-wise activation scale
//   sw     : [N, kg/8] f32    per-32-element-group weight scale
//   out    : [M, N]  f32      out[m, n] = sum_k x_q[m,k] * w_q[n,k] * (sw[n, k/32] * sx[m, k/32])
//
// Why a second kernel for the same slot
// -------------------------------------
// A decode step streams every weight byte once, so the figure of merit is the
// fraction of the memory system's read bandwidth the weights move at. The
// GEMV blocks of `matmul_i8_gemv` keep a few register-staged 8-byte loads per
// thread in flight: at 12-32 rows that is 0.75-1.2 TB/s of a 3.7 TB/s part
// (knowledge 368), and replacing every MMA by an XOR left it at 2.1 TB/s, so
// the limit is the load structure. This kernel changes the structure:
//  * the weights, their scales and the block's slice of x travel global ->
//    shared memory by `cp.async` (no register staging), through a ring of
//    BRAIN_I8S_STAGES stages of 128 bytes of K per row, so a block keeps
//    (STAGES - 1) x 8 KiB of weights in flight and two blocks share an SM;
//  * a warp reads 4 full 128-byte lines per load instruction (16 bytes a lane);
//  * the tiles are swizzled so `ldmatrix` hands the MMA its A (16 weight
//    rows x 32 k) and B (8 rows of x x 32 k) fragments without bank conflicts;
//  * K is split over a cluster of blocks (thread-block cluster, distributed
//    shared memory) when there are too few weight rows to keep every SM
//    streaming: each block streams its own stretch of K and the partial sums
//    are added in rank order through the cluster, so the result does not
//    depend on scheduling and no workspace is needed. The cluster size is read
//    from the launch, so one source serves every split.
//
// Arithmetic (as `matmul_i8_gemv`'s tensor-core block): one `mma.m16n8k32`
// per 32-element group, the group's integer sum returned as a float bit
// pattern (the accumulator is seeded with 1.5 * 2^23), then one FADD removes
// the bias and one FFMA applies sw * sx to the running f32 total. The result
// agrees with the reference to f32 rounding of the running sum.
//
// Row tails: weight rows past n read the last row and are dropped; rows of x
// past m read the last row and are dropped; a final stage with fewer than four
// groups copies only the groups that exist.
//
// No `__restrict__`: brain's device buffers alias by design.

#ifndef BRAIN_I8S_WARPS
#define BRAIN_I8S_WARPS 4
#endif
// m16 tiles of weight rows one warp covers.
#ifndef BRAIN_I8S_ATILES
#define BRAIN_I8S_ATILES 1
#endif
#ifndef BRAIN_I8S_STAGES
#define BRAIN_I8S_STAGES 5
#endif
#ifndef BRAIN_I8S_MIN_BLOCKS
#define BRAIN_I8S_MIN_BLOCKS 2
#endif

#define BRAIN_I8S_THREADS (32 * BRAIN_I8S_WARPS)
// Weight rows of one block.
#define BRAIN_I8S_BN (16 * BRAIN_I8S_ATILES * BRAIN_I8S_WARPS)
// Bytes of K per stage and row (one cache line), and the scale groups in it.
#define BRAIN_I8S_KB 128
#define BRAIN_I8S_GPS 4
// Rows of x one launch serves (a multiple of 8): sizes the x stages in shared memory.
#ifndef BRAIN_I8S_MAX_ROWS
#define BRAIN_I8S_MAX_ROWS 32
#endif
// Stage layout (bytes): weights, x, weight scales, activation scales.
#define BRAIN_I8S_W_BYTES (BRAIN_I8S_BN * BRAIN_I8S_KB)
#define BRAIN_I8S_X_BYTES (BRAIN_I8S_MAX_ROWS * BRAIN_I8S_KB)
#define BRAIN_I8S_SW_BYTES (BRAIN_I8S_BN * BRAIN_I8S_GPS * 4)
#define BRAIN_I8S_SX_BYTES (BRAIN_I8S_MAX_ROWS * BRAIN_I8S_GPS * 4)
#define BRAIN_I8S_STAGE_BYTES (BRAIN_I8S_W_BYTES + BRAIN_I8S_X_BYTES + BRAIN_I8S_SW_BYTES + BRAIN_I8S_SX_BYTES)
// Partial-sum buffer for the cluster reduction ([row of x][weight row], padded), aliased on the stages.
#define BRAIN_I8S_RED_STRIDE (BRAIN_I8S_BN + 4)

// Programmatic dependent launch (compute capability 9.0), see `matmul_i8_gemv.cu`.
#ifndef BRAIN_PDL_ENTRY
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
#define BRAIN_PDL_ENTRY() do { asm volatile("griddepcontrol.wait;\n" ::: "memory"); asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory"); } while (0)
#else
#define BRAIN_PDL_ENTRY() do {} while (0)
#endif
#endif

__device__ __forceinline__ unsigned int brain_i8s_smem(const void* p) {
    return static_cast<unsigned int>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void brain_i8s_cp16(unsigned int dst, const void* src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(dst), "l"(src) : "memory");
}

__device__ __forceinline__ void brain_i8s_cp4(unsigned int dst, const void* src) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"(dst), "l"(src) : "memory");
}

__device__ __forceinline__ void brain_i8s_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }

template <int N>
__device__ __forceinline__ void brain_i8s_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N) : "memory"); }

__device__ __forceinline__ void brain_i8s_ldsm4(unsigned int (&r)[4], unsigned int addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}

__device__ __forceinline__ void brain_i8s_mma(unsigned int (&d)[4], const unsigned int (&a)[4], unsigned int b0, unsigned int b1) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %10, %10, %10};\n"
                 : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(0x4B400000u));
}

__device__ __forceinline__ unsigned int brain_i8s_cluster_rank() {
    unsigned int r;
    asm volatile("mov.u32 %0, %%cluster_ctarank;\n" : "=r"(r));
    return r;
}

__device__ __forceinline__ unsigned int brain_i8s_cluster_size() {
    unsigned int r;
    asm volatile("mov.u32 %0, %%cluster_nctarank;\n" : "=r"(r));
    return r;
}

__device__ __forceinline__ void brain_i8s_cluster_sync() {
    asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;\n" ::: "memory");
}

__device__ __forceinline__ float brain_i8s_remote_ld(unsigned int local_addr, unsigned int rank) {
    unsigned int remote;
    float v;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;\n" : "=r"(remote) : "r"(local_addr), "r"(rank));
    asm volatile("ld.shared::cluster.f32 %0, [%1];\n" : "=f"(v) : "r"(remote) : "memory");
    return v;
}

// One stage's copies: stage `s` covers K bytes [128 s, 128 s + 128) of every weight row of the block and every row of
// x. `T` tiles of eight rows of x are live. Rows past the matrix read its last row (dropped later).
template <int T>
__device__ __forceinline__ void brain_i8s_issue(unsigned int stage_smem, unsigned int s, const unsigned char* wq, const unsigned char* xq,
                                                const float* sx, const float* sw, unsigned int m, unsigned int groups, unsigned int n,
                                                unsigned int n0) {
    const unsigned int tid = threadIdx.x;
    const unsigned long long kbytes = (unsigned long long)groups * 32ull;
    const unsigned int koff = s * BRAIN_I8S_KB;
    const unsigned int g0 = s * BRAIN_I8S_GPS;
#pragma unroll
    for (int c0 = 0; c0 < BRAIN_I8S_BN * 8; c0 += BRAIN_I8S_THREADS) {
        const unsigned int c = c0 + tid;
        const unsigned int row = c >> 3, ch = c & 7u;
        if (koff + ch * 16u < kbytes) {
            const unsigned int wrow = min(n0 + row, n - 1u);
            brain_i8s_cp16(stage_smem + row * BRAIN_I8S_KB + ((ch ^ (row & 7u)) << 4), wq + (unsigned long long)wrow * kbytes + koff + ch * 16u);
        }
    }
    const unsigned int xs = stage_smem + BRAIN_I8S_W_BYTES;
#pragma unroll
    for (int c0 = 0; c0 < T * 8 * 8; c0 += BRAIN_I8S_THREADS) {
        const unsigned int c = c0 + tid;
        if (c < T * 64u) {
            const unsigned int row = c >> 3, ch = c & 7u;
            if (koff + ch * 16u < kbytes) {
                brain_i8s_cp16(xs + row * BRAIN_I8S_KB + ((ch ^ (row & 7u)) << 4), xq + (unsigned long long)min(row, m - 1u) * kbytes + koff + ch * 16u);
            }
        }
    }
    const unsigned int sws = xs + BRAIN_I8S_X_BYTES;
#pragma unroll
    for (int c0 = 0; c0 < BRAIN_I8S_BN * BRAIN_I8S_GPS; c0 += BRAIN_I8S_THREADS) {
        const unsigned int c = c0 + tid;
        const unsigned int row = c >> 2, gi = c & 3u;
        if (g0 + gi < groups) {
            brain_i8s_cp4(sws + c * 4u, sw + (unsigned long long)min(n0 + row, n - 1u) * groups + g0 + gi);
        }
    }
    const unsigned int sxs = sws + BRAIN_I8S_SW_BYTES;
#pragma unroll
    for (int c0 = 0; c0 < T * 8 * BRAIN_I8S_GPS; c0 += BRAIN_I8S_THREADS) {
        const unsigned int c = c0 + tid;
        if (c < T * 32u) {
            const unsigned int row = c >> 2, gi = c & 3u;
            if (g0 + gi < groups) {
                brain_i8s_cp4(sxs + c * 4u, sx + (unsigned long long)min(row, m - 1u) * groups + g0 + gi);
            }
        }
    }
}

template <int T>
__device__ __forceinline__ void brain_i8s_block(const unsigned char* wq, const unsigned char* xq, const float* sx, const float* sw, float* out,
                                                unsigned int m, unsigned int groups, unsigned int n, unsigned int n0, unsigned int rank,
                                                unsigned int csize, unsigned char* smem) {
    constexpr int AT = BRAIN_I8S_ATILES;
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int warp = threadIdx.x >> 5;
    const unsigned int g = lane >> 2, t4 = lane & 3u;
    const unsigned int smem0 = brain_i8s_smem(smem);

    // Stages of 128 bytes of K; this block streams the contiguous stretch [first, first + count) of them.
    const unsigned int stages = (groups + BRAIN_I8S_GPS - 1u) / BRAIN_I8S_GPS;
    const unsigned int per = (stages + csize - 1u) / csize;
    const unsigned int first = min(rank * per, stages);
    const unsigned int count = min(per, stages - first);

    float f[AT][T][4];
#pragma unroll
    for (int a = 0; a < AT; ++a)
#pragma unroll
        for (int t = 0; t < T; ++t) { f[a][t][0] = f[a][t][1] = f[a][t][2] = f[a][t][3] = 0.0f; }

#pragma unroll
    for (int p = 0; p < BRAIN_I8S_STAGES - 1; ++p) {
        if (p < count) { brain_i8s_issue<T>(smem0 + p * BRAIN_I8S_STAGE_BYTES, first + p, wq, xq, sx, sw, m, groups, n, n0); }
        brain_i8s_commit();
    }

    const unsigned int lrow = lane & 7u;       // row within an ldmatrix 8x8
    const unsigned int mi = lane >> 3;         // which of the four matrices this lane addresses
    for (unsigned int i = 0; i < count; ++i) {
        brain_i8s_wait<BRAIN_I8S_STAGES - 2>();
        __syncthreads();  // the stage is visible to every warp, and the slot refilled below was read by all of them
        const unsigned int nxt = i + BRAIN_I8S_STAGES - 1;
        if (nxt < count) { brain_i8s_issue<T>(smem0 + (nxt % BRAIN_I8S_STAGES) * BRAIN_I8S_STAGE_BYTES, first + nxt, wq, xq, sx, sw, m, groups, n, n0); }
        brain_i8s_commit();

        const unsigned int st = smem0 + (i % BRAIN_I8S_STAGES) * BRAIN_I8S_STAGE_BYTES;
        const unsigned int xs = st + BRAIN_I8S_W_BYTES;
        const float* sws = reinterpret_cast<const float*>(smem + (st - smem0) + BRAIN_I8S_W_BYTES + BRAIN_I8S_X_BYTES);
        const float* sxs = sws + BRAIN_I8S_BN * BRAIN_I8S_GPS;
        const unsigned int gin = min((unsigned int)BRAIN_I8S_GPS, groups - (first + i) * BRAIN_I8S_GPS);  // groups in this stage
#pragma unroll
        for (int pr = 0; pr < BRAIN_I8S_GPS / 2; ++pr) {
            if (2 * pr >= (int)gin) { break; }
            // B fragments of this pair of groups, per tile of eight rows of x: matrices (group, k lo), (group, k hi),
            // (group + 1, k lo), (group + 1, k hi).
            unsigned int b[T][4];
#pragma unroll
            for (int t = 0; t < T; ++t) {
                const unsigned int row = 8u * t + lrow;
                const unsigned int chunk = (2u * pr + (mi >> 1)) * 2u + (mi & 1u);
                brain_i8s_ldsm4(b[t], xs + row * BRAIN_I8S_KB + ((chunk ^ (row & 7u)) << 4));
            }
#pragma unroll
            for (int jj = 0; jj < 2; ++jj) {
                const int j = 2 * pr + jj;
                if (j >= (int)gin) { break; }
#pragma unroll
                for (int a = 0; a < AT; ++a) {
                    // A fragment: matrices (rows 0-7, k lo), (rows 8-15, k lo), (rows 0-7, k hi), (rows 8-15, k hi).
                    const unsigned int arow0 = (warp * AT + a) * 16u;
                    const unsigned int row = arow0 + lrow + ((mi & 1u) << 3);
                    const unsigned int chunk = j * 2u + (mi >> 1);
                    unsigned int af[4];
                    brain_i8s_ldsm4(af, st + row * BRAIN_I8S_KB + ((chunk ^ (row & 7u)) << 4));
                    const float sl = sws[(arow0 + g) * BRAIN_I8S_GPS + j];
                    const float sh = sws[(arow0 + g + 8u) * BRAIN_I8S_GPS + j];
#pragma unroll
                    for (int t = 0; t < T; ++t) {
                        unsigned int d[4];
                        brain_i8s_mma(d, af, b[t][2 * jj], b[t][2 * jj + 1]);
                        const float xa = sxs[(8u * t + 2u * t4) * BRAIN_I8S_GPS + j];
                        const float xb = sxs[(8u * t + 2u * t4 + 1u) * BRAIN_I8S_GPS + j];
                        f[a][t][0] = fmaf(__uint_as_float(d[0]) - 12582912.0f, sl * xa, f[a][t][0]);
                        f[a][t][1] = fmaf(__uint_as_float(d[1]) - 12582912.0f, sl * xb, f[a][t][1]);
                        f[a][t][2] = fmaf(__uint_as_float(d[2]) - 12582912.0f, sh * xa, f[a][t][2]);
                        f[a][t][3] = fmaf(__uint_as_float(d[3]) - 12582912.0f, sh * xb, f[a][t][3]);
                    }
                }
            }
        }
    }
    brain_i8s_wait<0>();

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
                red[col * BRAIN_I8S_RED_STRIDE + row] = f[a][t][q];
            }
    brain_i8s_cluster_sync();
    for (unsigned int idx = rank * BRAIN_I8S_THREADS + threadIdx.x; idx < m * BRAIN_I8S_BN; idx += csize * BRAIN_I8S_THREADS) {
        const unsigned int col = idx / BRAIN_I8S_BN, row = idx % BRAIN_I8S_BN;
        const unsigned int a = smem0 + (col * BRAIN_I8S_RED_STRIDE + row) * 4u;
        float total = 0.0f;
        for (unsigned int r = 0; r < csize; ++r) { total += brain_i8s_remote_ld(a, r); }
        if (n0 + row < n) { out[(unsigned long long)col * n + n0 + row] = total; }
    }
    brain_i8s_cluster_sync();  // no block leaves while a peer may still read its partials
}

extern "C" __global__ void __launch_bounds__(BRAIN_I8S_THREADS, BRAIN_I8S_MIN_BLOCKS)
brain_matmul_i8_stream(const unsigned int* params, const unsigned int* xq, const unsigned int* wq,
                       const float* sx, const float* sw, float* out) {
    BRAIN_PDL_ENTRY();
    extern __shared__ __align__(1024) unsigned char brain_i8s_smem_base[];
    const unsigned int m = params[0];
    const unsigned int groups = params[1] >> 3;
    const unsigned int n = params[2];
    const unsigned int csize = brain_i8s_cluster_size();
    const unsigned int rank = brain_i8s_cluster_rank();
    // A cluster's blocks are consecutive and share a tile of weight rows.
    const unsigned int blk = blockIdx.y * gridDim.x + blockIdx.x;
    const unsigned int n0 = (blk / csize) * BRAIN_I8S_BN;
    if (n0 >= n || m == 0u) { return; }  // cluster-uniform
    const unsigned char* w8 = reinterpret_cast<const unsigned char*>(wq);
    const unsigned char* x8 = reinterpret_cast<const unsigned char*>(xq);
    if (m <= 8u)       { brain_i8s_block<1>(w8, x8, sx, sw, out, m, groups, n, n0, rank, csize, brain_i8s_smem_base); }
    else if (m <= 16u) { brain_i8s_block<2>(w8, x8, sx, sw, out, m, groups, n, n0, rank, csize, brain_i8s_smem_base); }
    else if (m <= 24u) { brain_i8s_block<3>(w8, x8, sx, sw, out, m, groups, n, n0, rank, csize, brain_i8s_smem_base); }
#if BRAIN_I8S_MAX_ROWS > 32
    else if (m <= 32u) { brain_i8s_block<4>(w8, x8, sx, sw, out, m, groups, n, n0, rank, csize, brain_i8s_smem_base); }
    else if (m <= 48u) { brain_i8s_block<6>(w8, x8, sx, sw, out, m, groups, n, n0, rank, csize, brain_i8s_smem_base); }
    else if (m <= 64u) { brain_i8s_block<8>(w8, x8, sx, sw, out, m, groups, n, n0, rank, csize, brain_i8s_smem_base); }
#else
    else               { brain_i8s_block<4>(w8, x8, sx, sw, out, m, groups, n, n0, rank, csize, brain_i8s_smem_base); }
#endif
}
