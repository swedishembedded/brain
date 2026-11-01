// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for latency-bound LLM decode on GPUs for its clients. If your team needs expertise in
// skinny dense projections that fill the card without a second reduction launch then you can procure our services by sending an
// email to info@swedishembedded.com.
//
// The dense bf16 projections of a decode step that are too thin to fill the card (a Gated DeltaNet layer's two gate projections:
// 48 weight rows each over a 5120-wide row) against 2 to 16 rows of f32 activations, up to four matrices in ONE launch and with
// no activation cast and no reduction launch:
//
//   y_i[r, c] = sum_k bf16(x[r, k]) * w_i[c, k]        r < m <= 16, c < n_i, each n_i a multiple of 16
//
//   params : u32 [m, k, n0, n1, n2, n3]   k a multiple of 32, at most 8192; n_i == 0 leaves set i unused
//   bufs   : x f32 [m, k], then for each set w u32 [n_i, k/2] (two bf16 to a word, the even element in the low half) and y f32 [m, n_i]
//
// Why: the library's skinny bf16 product runs a split-K kernel and a reduction kernel per matrix (5.3 + 1.7 us at 16 rows, twice a
// layer) after a cast launch of the activations, 15.8 us a layer for 1 MB of weights. Here a cluster of 8 blocks takes a 16-row
// tile of weight rows and splits K between them: the tile's 16 x 640 weights and the activations' 640 columns go through
// tensor-core MMAs (m16n8k16, the activations rounded to bf16 exactly as `cast_f32_bf16` does, f32 accumulation), the 8 warps of a
// block reduce through shared memory and the 8 blocks through the cluster's shared memory, in ascending order, so the result is
// deterministic and independent of the launch. It agrees with the f64 product of the bf16 activations and the bf16 weights to f32
// accumulation (the gate holds it to 1e-5 relative).
//
// K permutation: the MMA sums over its 16 k slots in any order, so any assignment of actual k to slots is the same product as long
// as the activations and the weights use the same one. A lane (gid = lane / 4, tig = lane % 4) takes the 8 consecutive k
// 32 g + 8 tig .. + 7 of a group g: ONE 16-byte load of its weight row (the B operands of two consecutive MMA steps: words 0, 1 of the
// load for step 0, words 2, 3 for step 1) and two float4 loads of each of its two activation rows (gid, gid + 8).
//
// The block is small on purpose (256 threads, 9 KiB of shared memory): it is launched while the FP8 product before it still runs
// and that product keeps 210 KiB of an SM's shared memory (knowledge 420). The weights do not depend on the kernel before this one,
// so their loads are issued before the wait for it.
//
// No `__restrict__` anywhere: brain's device buffers alias by design.

#define BRAIN_BS_THREADS 256
#define BRAIN_BS_TILE 16            // weight rows a cluster
#define BRAIN_BS_SPLIT 8            // blocks (K slices) a cluster
#define BRAIN_BS_GROUP 32           // k elements one lane-quad-row load covers: 4 tig x 8
#define BRAIN_BS_MAX_GROUPS_WARP 4  // groups one warp walks: k <= 8 warps x 8 blocks x 4 x 32 = 8192

__device__ __forceinline__ unsigned int brain_bs_cluster_rank() {
    unsigned int r;
    asm volatile("mov.u32 %0, %%cluster_ctarank;\n" : "=r"(r));
    return r;
}
__device__ __forceinline__ void brain_bs_cluster_sync() {
    asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;\n" ::: "memory");
}
// The release makes the peer loads before it complete first: a peer that passed its wait could otherwise exit, and free its shared
// memory, while a load of it was in flight.
__device__ __forceinline__ void brain_bs_cluster_arrive() { asm volatile("barrier.cluster.arrive.release.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void brain_bs_cluster_wait() { asm volatile("barrier.cluster.wait.acquire.aligned;\n" ::: "memory"); }
__device__ __forceinline__ float brain_bs_peer_float(const float* local, unsigned int rank) {
    const unsigned int addr = static_cast<unsigned int>(__cvta_generic_to_shared(local));
    unsigned int remote;
    float v;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;\n" : "=r"(remote) : "r"(addr), "r"(rank));
    asm volatile("ld.shared::cluster.f32 %0, [%1];\n" : "=f"(v) : "r"(remote) : "memory");
    return v;
}

__device__ __forceinline__ uint4 brain_bs_ld_stream(const uint4* p) {
    uint4 v;
    asm("ld.global.nc.L1::no_allocate.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
    return v;
}

// Two floats as the bf16 pair `cast_f32_bf16` makes of them (round to nearest even), `lo` in the low half.
__device__ __forceinline__ unsigned int brain_bs_bf16x2(float lo, float hi) {
    unsigned int r;
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;\n" : "=r"(r) : "f"(hi), "f"(lo));
    return r;
}

__device__ __forceinline__ void brain_bs_mma(float* d, unsigned int a0, unsigned int a1, unsigned int a2, unsigned int a3, unsigned int b0, unsigned int b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

extern "C" __global__ void __launch_bounds__(BRAIN_BS_THREADS)
brain_matmul_bf16_gemv_splitk(const unsigned int* params, const float* x, const unsigned int* w0, float* y0, const unsigned int* w1, float* y1,
                              const unsigned int* w2, float* y2, const unsigned int* w3, float* y3) {
    const unsigned int m = params[0], k = params[1];
    // The grid is `tiles * SPLIT` blocks, a tile's cluster consecutive. Tile t of the sets in order is set s, local tile t - (tiles before s).
    unsigned int tile = (blockIdx.y * gridDim.x + blockIdx.x) / BRAIN_BS_SPLIT;
    const unsigned int rank = brain_bs_cluster_rank();
    const unsigned int* w = w0;
    float* y = y0;
    unsigned int n = params[2];
    {
        // Set 0's tiles first, then set 1's, ...: block-uniform selects, no indexing of a pointer array.
        bool found = tile < params[2] / BRAIN_BS_TILE;
        if (!found) { tile -= params[2] / BRAIN_BS_TILE; }
        if (!found && tile < params[3] / BRAIN_BS_TILE) { w = w1; y = y1; n = params[3]; found = true; }
        else if (!found) { tile -= params[3] / BRAIN_BS_TILE; }
        if (!found && tile < params[4] / BRAIN_BS_TILE) { w = w2; y = y2; n = params[4]; found = true; }
        else if (!found) { tile -= params[4] / BRAIN_BS_TILE; }
        if (!found) { w = w3; y = y3; n = params[5]; }
    }
    const unsigned int tid = threadIdx.x, lane = tid & 31u, warp = tid >> 5;
    const unsigned int gid = lane >> 2, tig = lane & 3u;

    __shared__ float red[8][BRAIN_BS_TILE][BRAIN_BS_TILE];  // each warp's partial 16 x 16 tile
    __shared__ float blockp[BRAIN_BS_TILE * BRAIN_BS_TILE];  // the block's partial tile

    // This block's K slice, in groups of 32: rank q takes groups [q G / SPLIT, (q + 1) G / SPLIT); the warps take its groups by
    // stride 8.
    const unsigned int groups = k / BRAIN_BS_GROUP;
    const unsigned int g_begin = rank * groups / BRAIN_BS_SPLIT, g_end = (rank + 1) * groups / BRAIN_BS_SPLIT;

    // The weights first: they do not depend on the kernel before this one. Lane (gid, tig) of group g, n-tile j: weight row
    // tile * 16 + 8 j + gid, k elements 32 g + 8 tig .. + 7.
    uint4 bw[BRAIN_BS_MAX_GROUPS_WARP][2];
#pragma unroll
    for (int i = 0; i < BRAIN_BS_MAX_GROUPS_WARP; ++i) {
        const unsigned int g = g_begin + warp + 8u * i;
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            const unsigned int row = tile * BRAIN_BS_TILE + 8u * j + gid;
            bw[i][j] = g < g_end ? brain_bs_ld_stream(reinterpret_cast<const uint4*>(w) + static_cast<unsigned long long>(row) * (k >> 3) + 4u * g + tig)
                                 : make_uint4(0u, 0u, 0u, 0u);
        }
    }
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.wait;\n" ::: "memory");
    asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");
#endif

    // The activations: rows gid and gid + 8 (zero past m), the same 8 consecutive k as the weights.
    float4 xa[BRAIN_BS_MAX_GROUPS_WARP][2][2];
#pragma unroll
    for (int i = 0; i < BRAIN_BS_MAX_GROUPS_WARP; ++i) {
        const unsigned int g = g_begin + warp + 8u * i;
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const unsigned int r = gid + 8u * h;
            const bool live = g < g_end && r < m;
            const float4* p = reinterpret_cast<const float4*>(x + static_cast<unsigned long long>(r) * k + BRAIN_BS_GROUP * g + 8u * tig);
            xa[i][h][0] = live ? p[0] : make_float4(0.f, 0.f, 0.f, 0.f);
            xa[i][h][1] = live ? p[1] : make_float4(0.f, 0.f, 0.f, 0.f);
        }
    }

    float acc[2][4];
#pragma unroll
    for (int j = 0; j < 2; ++j) {
#pragma unroll
        for (int e = 0; e < 4; ++e) { acc[j][e] = 0.0f; }
    }
#pragma unroll
    for (int i = 0; i < BRAIN_BS_MAX_GROUPS_WARP; ++i) {
#pragma unroll
        for (int step = 0; step < 2; ++step) {
            // Step 0 takes the lane's weights 0..3 (words x, y), step 1 weights 4..7 (words z, w); a0/a1 the low slots of the rows
            // gid / gid + 8, a2/a3 the high slots.
            const unsigned int a0 = brain_bs_bf16x2(step ? xa[i][0][1].x : xa[i][0][0].x, step ? xa[i][0][1].y : xa[i][0][0].y);
            const unsigned int a2 = brain_bs_bf16x2(step ? xa[i][0][1].z : xa[i][0][0].z, step ? xa[i][0][1].w : xa[i][0][0].w);
            const unsigned int a1 = brain_bs_bf16x2(step ? xa[i][1][1].x : xa[i][1][0].x, step ? xa[i][1][1].y : xa[i][1][0].y);
            const unsigned int a3 = brain_bs_bf16x2(step ? xa[i][1][1].z : xa[i][1][0].z, step ? xa[i][1][1].w : xa[i][1][0].w);
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                const unsigned int b0 = step ? bw[i][j].z : bw[i][j].x;
                const unsigned int b1 = step ? bw[i][j].w : bw[i][j].y;
                brain_bs_mma(acc[j], a0, a1, a2, a3, b0, b1);
            }
        }
    }

    // The warp's partial tile: lane (gid, tig) holds rows gid and gid + 8, columns 8 j + 2 tig and + 1.
#pragma unroll
    for (int j = 0; j < 2; ++j) {
        const unsigned int col = 8u * j + 2u * tig;
        red[warp][gid][col] = acc[j][0];
        red[warp][gid][col + 1] = acc[j][1];
        red[warp][gid + 8][col] = acc[j][2];
        red[warp][gid + 8][col + 1] = acc[j][3];
    }
    __syncthreads();
    // The block's partial: thread t sums the warps' (row t / 16, column t % 16) in ascending order.
    {
        float s = 0.0f;
#pragma unroll
        for (int wv = 0; wv < 8; ++wv) { s = __fadd_rn(s, red[wv][tid / BRAIN_BS_TILE][tid % BRAIN_BS_TILE]); }
        blockp[tid] = s;
    }
    brain_bs_cluster_sync();
    // Block q finishes outputs 32 q .. 32 q + 31 of the tile: the 8 blocks' partials in ascending order.
    float total = 0.0f;
    const unsigned int o = rank * 32u + tid;
    if (tid < 32u) {
#pragma unroll
        for (int p = 0; p < BRAIN_BS_SPLIT; ++p) { total = __fadd_rn(total, brain_bs_peer_float(&blockp[o], p)); }
    }
    brain_bs_cluster_arrive();
    if (tid < 32u && o / BRAIN_BS_TILE < m) {
        y[static_cast<unsigned long long>(o / BRAIN_BS_TILE) * n + tile * BRAIN_BS_TILE + o % BRAIN_BS_TILE] = total;
    }
    brain_bs_cluster_wait();  // no block exits while a peer may still be reading its partials
}
