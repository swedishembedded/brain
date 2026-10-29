// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements tensor-core training kernels for linear-
// attention language models for its clients. If your team needs expertise in
// running a sequential recurrence over chunks at tensor-core rate instead of
// on the CUDA cores, then you can procure our services by sending an email to
// info@swedishembedded.com.
//
// Gated DeltaNet's whole across-chunk recurrence (`model::gdn` step 10) in ONE
// launch on tensor cores: the same contract as `gdn_chunk_loop_f32`, with the
// four matrix products of a chunk
//
//   v_new = u_c - w_c @ state              out_c = (scale*query_c*exp_g_cs) @ state + intra_c @ v_new
//   state = state * exp(g_last) + (key_c * exp(g_last - g_cs))^T @ v_new
//
// run as three-pass TF32 `mma.sync.m16n8k8` (about 20 mantissa bits, fp32
// accumulation and fp32 state). Eight warps a block: four 16-row tiles of the chunk, each split over two halves of the block's columns. A head is split over four blocks of 32 value
// columns (192 blocks for the model's 48 heads, all resident at once); each block keeps its 128 x 32 slice of the state in shared memory for
// all chunks and streams the chunk's w, query, intra and key tiles through
// two shared buffers. NOT bit-identical to the fp32 kernel; the model gates it
// behind the same opt-in as the tensor-core prefix.
//
// Contract (u32 params, then pointers; chunk-major layout, `bhc = chunk*bh + head`):
//   params : [bh, c_len, dk, dv, n_chunks, scale (f32 bits), heads_tok]
//   key, query, w : [n_chunks, bh, c_len, 128]   u, out : [n_chunks, bh, c_len, 128]
//     (with heads_tok != 0 - the value-head count, bh = batch * heads_tok - `out` is written in the mixer's
//      token-major layout [tokens, heads_tok, 128] instead, which folds the chunk-major to token-major permute
//      of its consumer into the stores)
//   g_cs, exp_g_cs : [n_chunks, bh, c_len]       intra : [n_chunks, bh, c_len, c_len]
//   initial_state, final_state : [bh, 128, 128]
//   TRAIN: state_hist [n_chunks + 1, bh, 128, 128] (the state ENTERING each chunk, then
//          the final one), decay_hist [n_chunks, bh, c_len] (exp(g_last - g_cs))
// dk = dv = 128 and c_len <= 64 (the provider checks; the entry does not).
// Dynamic shared memory: LOOP_SMEM_FLOATS floats.
//
// No `__restrict__`: brain's device buffers alias by design.

#define D 128
#define MAXC 64
#define CS 32                 // value columns per block
#define MMA_NT (CS / 8)           // 8-column MMA tiles per block
#define SPLITS (D / CS)       // blocks per head
#define THREADS 256           // 8 warps: four 16-row tiles of the chunk, each split over two halves of the block's columns
#define WARPS_M 4             // warps down the chunk's rows
#define MMA_NTW (MMA_NT / 2)  // 8-column MMA tiles one warp owns
#define SA (D + 4)            // row stride of the staged A tiles (conflict-free fragment loads)
#define SS (CS + 8)           // row stride of the state slice and of v_new (conflict-free as a k-major B operand)

#define LOOP_SMEM_FLOATS (2 * MAXC * SA + D * SS + MAXC * SS + MAXC)

// x = hi + lo with both parts TF32-representable. Truncation (a mask) instead of `cvt.rna.tf32`: the convert runs on
// a slow pipe and the splits are most of the instructions around the MMAs. lo = x - hi is exact in fp32 and is then
// truncated to TF32 itself, so the pair carries about 20 mantissa bits.
__device__ __forceinline__ void brain_split_tf32(float x, unsigned int& hi, unsigned int& lo) {
    hi = __float_as_uint(x) & 0xFFFFE000u;
    lo = __float_as_uint(x - __uint_as_float(hi)) & 0xFFFFE000u;
}

__device__ __forceinline__ void brain_mma_tf32(float (&d)[4], const unsigned int (&a)[4], unsigned int b0, unsigned int b1) {
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// acc(16 x 8*NT) += sum_{k < K} fa(r, k) * fb(k, n) for this warp's rows m0 .. and columns n0 .., as
// three TF32 products per tile (head*head + head*remainder + remainder*head). The caller zeroes acc.
template <int NT, class FA, class FB>
__device__ __forceinline__ void warp_mma_acc(float (&acc)[NT][4], unsigned int K, unsigned int m0, unsigned int n0, FA fa, FB fb) {
    const unsigned int lane = threadIdx.x & 31u, g = lane >> 2, t4 = lane & 3u;
    for (unsigned int k0 = 0; k0 < K; k0 += 8) {
        unsigned int ah[4], al[4];
        brain_split_tf32(fa(m0 + g, k0 + t4), ah[0], al[0]);
        brain_split_tf32(fa(m0 + g + 8, k0 + t4), ah[1], al[1]);
        brain_split_tf32(fa(m0 + g, k0 + t4 + 4), ah[2], al[2]);
        brain_split_tf32(fa(m0 + g + 8, k0 + t4 + 4), ah[3], al[3]);
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            const unsigned int n = n0 + t * 8 + g;
            unsigned int bh0, bl0, bh1, bl1;
            brain_split_tf32(fb(k0 + t4, n), bh0, bl0);
            brain_split_tf32(fb(k0 + t4 + 4, n), bh1, bl1);
            brain_mma_tf32(acc[t], al, bh0, bh1);
            brain_mma_tf32(acc[t], ah, bl0, bl1);
            brain_mma_tf32(acc[t], ah, bh0, bh1);
        }
    }
}

template <bool TRAIN>
__device__ __forceinline__ void brain_gdn_chunk_loop_tc_body(const unsigned int* params, const float* key, const float* query, const float* w, const float* u,
                                                             const float* g_cs, const float* exp_g_cs, const float* intra, const float* initial_state,
                                                             float* out, float* final_state, float* state_hist, float* decay_hist) {
    const unsigned int bh = params[0];
    const unsigned int C = params[1];
    const unsigned int n_chunks = params[4];
    const float scale = __uint_as_float(params[5]);
    const unsigned int heads_tok = params[6];

    const unsigned int blk = blockIdx.y * gridDim.x + blockIdx.x;
    const unsigned int bhi = blk / SPLITS;
    const unsigned int j0 = (blk % SPLITS) * CS;
    if (bhi >= bh) { return; }  // block-uniform

    extern __shared__ float smem[];
    float* sA = smem;                   // staged w, then intra, [64][SA]
    float* sB = sA + MAXC * SA;         // staged scaled query, then decayed key, [64][SA]
    float* sS = sB + MAXC * SA;         // this block's state slice, [128][SS]
    float* sV = sS + D * SS;            // v_new, [64][SS]
    float* sdec = sV + MAXC * SS;       // exp(g_last - g_cs) per row

    const unsigned int tid = threadIdx.x, lane = tid & 31u, warp = tid >> 5;
    const unsigned int g = lane >> 2, t4 = lane & 3u;
    const unsigned int wr = warp % WARPS_M, wc = warp / WARPS_M;
    const unsigned int m0 = wr * 16u;                    // this warp's chunk rows
    const unsigned int n0 = wc * (MMA_NTW * 8u);         // and its first column of the block's slice
    const unsigned int c8 = (C + 7u) & ~7u;              // contraction length over chunk rows

    for (unsigned int i = tid; i < D * CS; i += THREADS) {
        const unsigned int k = i / CS, j = i % CS;
        sS[k * SS + j] = initial_state[((unsigned long long)bhi * D + k) * D + j0 + j];
    }
    __syncthreads();

    for (unsigned int ci = 0; ci < n_chunks; ++ci) {
        const unsigned long long row_base = ((unsigned long long)ci * bh + bhi) * C;
        const unsigned long long off_dk = row_base * D;
        const unsigned long long off_g = row_base;
        const unsigned long long off_cc = row_base * C;
        const float g_last = g_cs[off_g + C - 1];
        // Where this (head, chunk)'s output rows go: chunk-major, or the mixer's token-major rows.
        float* out_chunk = out + off_dk;
        unsigned long long out_stride = D;
        if (heads_tok != 0) {
            const unsigned long long tok0 = ((unsigned long long)(bhi / heads_tok) * n_chunks + ci) * C;
            out_chunk = out + (tok0 * heads_tok + bhi % heads_tok) * D;
            out_stride = (unsigned long long)heads_tok * D;
        }

        if (TRAIN) {
            for (unsigned int i = tid; i < D * CS; i += THREADS) {
                const unsigned int k = i / CS, j = i % CS;
                state_hist[(((unsigned long long)ci * bh + bhi) * D + k) * D + j0 + j] = sS[k * SS + j];
            }
        }
        if (tid < MAXC) {
            const float e = tid < C ? expf(g_last - g_cs[off_g + tid]) : 0.0f;
            sdec[tid] = e;
            if (TRAIN && j0 == 0 && tid < C) { decay_hist[off_g + tid] = e; }
        }
        // Stage w into sA and scale*query*exp_g_cs into sB.
#pragma unroll
        for (unsigned int it = 0; it < MAXC * (D / 4) / THREADS; ++it) {
            const unsigned int e = tid + it * THREADS;
            const unsigned int r = e / (D / 4), c4 = e % (D / 4);
            float4 wv = make_float4(0.0f, 0.0f, 0.0f, 0.0f), qv = wv;
            if (r < C) {
                wv = *reinterpret_cast<const float4*>(w + off_dk + (unsigned long long)r * D + c4 * 4);
                qv = *reinterpret_cast<const float4*>(query + off_dk + (unsigned long long)r * D + c4 * 4);
                const float ex = exp_g_cs[off_g + r];
                qv = make_float4((scale * qv.x) * ex, (scale * qv.y) * ex, (scale * qv.z) * ex, (scale * qv.w) * ex);
            }
            *reinterpret_cast<float4*>(sA + r * SA + c4 * 4) = wv;
            *reinterpret_cast<float4*>(sB + r * SA + c4 * 4) = qv;
        }
        __syncthreads();

        // v_new = u - w @ state, and the inter-chunk output (scaled query) @ state, both from the OLD state.
        float acc[MMA_NTW][4], acc2[MMA_NTW][4];
#pragma unroll
        for (int t = 0; t < MMA_NTW; ++t) {
#pragma unroll
            for (int i = 0; i < 4; ++i) { acc[t][i] = 0.0f; acc2[t][i] = 0.0f; }
        }
        warp_mma_acc<MMA_NTW>(acc, D, m0, n0, [&](unsigned int r, unsigned int k) { return sA[r * SA + k]; }, [&](unsigned int k, unsigned int n) { return sS[k * SS + n]; });
        warp_mma_acc<MMA_NTW>(acc2, D, m0, n0, [&](unsigned int r, unsigned int k) { return sB[r * SA + k]; }, [&](unsigned int k, unsigned int n) { return sS[k * SS + n]; });
#pragma unroll
        for (int t = 0; t < MMA_NTW; ++t) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const unsigned int r = m0 + g + ((i & 2) ? 8u : 0u);
                const unsigned int col = n0 + t * 8 + 2 * t4 + (i & 1);
                sV[r * SS + col] = r < C ? u[off_dk + (unsigned long long)r * D + j0 + col] - acc[t][i] : 0.0f;
            }
        }
        __syncthreads();

        // Stage intra into sA and the decayed key into sB.
#pragma unroll
        for (unsigned int it = 0; it < MAXC * MAXC / THREADS; ++it) {
            const unsigned int e = tid + it * THREADS;
            const unsigned int r = e / MAXC, k = e % MAXC;
            sA[r * SA + k] = (r < C && k < C) ? intra[off_cc + (unsigned long long)r * C + k] : 0.0f;
        }
#pragma unroll
        for (unsigned int it = 0; it < MAXC * (D / 4) / THREADS; ++it) {
            const unsigned int e = tid + it * THREADS;
            const unsigned int r = e / (D / 4), c4 = e % (D / 4);
            float4 kv = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (r < C) {
                kv = *reinterpret_cast<const float4*>(key + off_dk + (unsigned long long)r * D + c4 * 4);
                const float s = sdec[r];
                kv = make_float4(kv.x * s, kv.y * s, kv.z * s, kv.w * s);
            }
            *reinterpret_cast<float4*>(sB + r * SA + c4 * 4) = kv;
        }
        __syncthreads();

        // out += intra @ v_new.
        warp_mma_acc<MMA_NTW>(acc2, c8, m0, n0, [&](unsigned int r, unsigned int k) { return sA[r * SA + k]; }, [&](unsigned int k, unsigned int n) { return sV[k * SS + n]; });
#pragma unroll
        for (int t = 0; t < MMA_NTW; ++t) {
#pragma unroll
            for (int h2 = 0; h2 < 2; ++h2) {
                const unsigned int r = m0 + g + h2 * 8u;
                if (r < C) {
                    *reinterpret_cast<float2*>(out_chunk + (unsigned long long)r * out_stride + j0 + n0 + t * 8 + 2 * t4) = make_float2(acc2[t][2 * h2], acc2[t][2 * h2 + 1]);
                }
            }
        }

        // state = state * exp(g_last) + decayed_key^T @ v_new: 128 state rows, two 16-row tiles per warp.
        const float decay = expf(g_last);
#pragma unroll
        for (int mi = 0; mi < 2; ++mi) {
            float up[MMA_NTW][4];
#pragma unroll
            for (int t = 0; t < MMA_NTW; ++t) {
#pragma unroll
                for (int i = 0; i < 4; ++i) { up[t][i] = 0.0f; }
            }
            const unsigned int p0 = (wr + 4u * mi) * 16u;
            warp_mma_acc<MMA_NTW>(up, c8, p0, n0, [&](unsigned int p, unsigned int c) { return sB[c * SA + p]; }, [&](unsigned int c, unsigned int n) { return sV[c * SS + n]; });
#pragma unroll
            for (int t = 0; t < MMA_NTW; ++t) {
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    float* s = &sS[(p0 + g + ((i & 2) ? 8u : 0u)) * SS + n0 + t * 8 + 2 * t4 + (i & 1)];
                    *s = *s * decay + up[t][i];
                }
            }
        }
        __syncthreads();
    }

    for (unsigned int i = tid; i < D * CS; i += THREADS) {
        const unsigned int k = i / CS, j = i % CS;
        final_state[((unsigned long long)bhi * D + k) * D + j0 + j] = sS[k * SS + j];
        if (TRAIN) { state_hist[(((unsigned long long)n_chunks * bh + bhi) * D + k) * D + j0 + j] = sS[k * SS + j]; }
    }
}

extern "C" __global__ void __launch_bounds__(THREADS, 2) brain_gdn_chunk_loop_tc(const unsigned int* params, const float* key, const float* query, const float* w,
                                                                               const float* u, const float* g_cs, const float* exp_g_cs, const float* intra,
                                                                               const float* initial_state, float* out, float* final_state) {
    brain_gdn_chunk_loop_tc_body<false>(params, key, query, w, u, g_cs, exp_g_cs, intra, initial_state, out, final_state, nullptr, nullptr);
}

extern "C" __global__ void __launch_bounds__(THREADS, 2) brain_gdn_chunk_loop_train_tc(const unsigned int* params, const float* key, const float* query,
                                                                                     const float* w, const float* u, const float* g_cs, const float* exp_g_cs,
                                                                                     const float* intra, const float* initial_state, float* out,
                                                                                     float* final_state, float* state_hist, float* decay_hist) {
    brain_gdn_chunk_loop_tc_body<true>(params, key, query, w, u, g_cs, exp_g_cs, intra, initial_state, out, final_state, state_hist, decay_hist);
}
